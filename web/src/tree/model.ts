/**
 * Core data model for the Tree/Node restructure.
 *
 * See web/RESTRUCTURE_PLAN.md for the full rationale. In short, this replaces
 * the entangled "flat Rete nodes + state.groups side-table + facade
 * recreate/remap" model with a **definition-vs-instance** model:
 *
 *   - A {@link Tree} is a reusable *class definition*. The main graph, every
 *     group, AND every leaf block are all Trees.
 *   - A {@link TreeNode} is an *instance* that references a Tree by `name`.
 *   - A {@link Forest} owns the single global node store plus the tree dict.
 *
 * Collapse/expand is a pure derived computation over this model
 * (`buildDynamicMainTree`, Step 3), never a mutation of live editor state -
 * which is what kills the bug class behind the recent group fixes.
 *
 * This file is Step 0: types + the Forest store. No UI wiring; it lives
 * alongside the existing model behind conversion shims until each consumer is
 * ported (see the porting order in the plan).
 *
 * NOTE on naming: the instance type is called `TreeNode` rather than `Node` to
 * avoid shadowing the DOM `Node` global in this browser-targeted project.
 */

import type { Shape } from '../shape.ts'

/** A per-instance unique id (uuid-like). One per {@link TreeNode}. */
export type NodeId = string

/**
 * A tree's unique name. Doubles as the generated Python class name and the
 * key under which the tree is registered in {@link Forest.trees}.
 */
export type TreeName = string

/**
 * A single directed edge between two node ports, within one tree.
 *
 * `id` is collision-free by construction (it embeds both endpoints and both
 * port names), fixing the old `{from}_{output}_{to}_{input}` scheme that could
 * collide when port names contained underscores.
 */
export interface Connection {
  /** Source node id. */
  from: NodeId
  /** Source output port name. */
  fromOutput: string
  /** Target node id. */
  to: NodeId
  /** Target input port name. */
  toInput: string
  /** `${from}::${fromOutput}->${to}::${toInput}` - unique within a tree. */
  id: string
}

/** Build the canonical, collision-free connection id. */
export function connectionId(
  from: NodeId,
  fromOutput: string,
  to: NodeId,
  toInput: string
): string {
  return `${from}::${fromOutput}->${to}::${toInput}`
}

/** Make a {@link Connection} with its id derived from the endpoints. */
export function makeConnection(
  from: NodeId,
  fromOutput: string,
  to: NodeId,
  toInput: string
): Connection {
  return { from, fromOutput, to, toInput, id: connectionId(from, fromOutput, to, toInput) }
}

/**
 * One declared parameter of a tree (a constructor / init-time value). Mirrors
 * the manifest {@link import('../types.ts').CtorParam} but carries optional
 * shape/value for learnable params and constants.
 */
export interface ParamSpec {
  /** Declared value (a constant's value, or a ctor default). */
  value?: unknown
  /** Declared type hint, e.g. `'int'`, `'float'`, `'bool'`, `'str'`. */
  dtype: string
  /** Tensor shape, for learnable params / buffers. */
  shape?: Shape
  /** Whether the parameter must be supplied for valid codegen. */
  required?: boolean
  /** Enumerated choices surfaced as a dropdown in the inspector. */
  choices?: string[]
}

/**
 * A reusable class definition. The main graph, every group, and every leaf
 * block are all Trees.
 *
 * `inputs` / `outputs` are *boundary refs*: each entry is either a bare local
 * name (`"x"`) for a dangling port the tree exposes, or `"<childId>@<port>"`
 * to route a boundary slot to a specific inner child port. The authoritative,
 * derived view of the boundary is produced by
 * `getInputOutputParamsSignature(tree)` (Step 2) - these strings are the stored
 * hints it consults.
 */
export interface Tree {
  /** Unique name; also the generated class name and the Rete-rendered label. */
  name: TreeName
  /** Ids of the nodes this tree contains. Nodes live in {@link Forest.nodes}. */
  list_of_nodes: NodeId[]
  /** Edges between this tree's nodes. */
  list_of_connections: Connection[]
  /** Boundary input refs: `"<localName>"` or `"<childId>@<port>"`. */
  inputs: string[]
  /** Boundary output refs: `"<localName>"` or `"<childId>@<port>"`. */
  outputs: string[]
  /** Declared init params, keyed by param name. */
  params: Record<string, ParamSpec>
}

/**
 * An instance that references a {@link Tree} by name. Its own input/output
 * ports are *derived* from the referenced tree's signature, never stored here.
 */
export interface TreeNode {
  /** Stable per-instance id. */
  id: NodeId
  /** The tree this node instantiates (a block, a group, or a custom class). */
  name: TreeName
  /** Weight-sharing tag -> shared `self.<tag>` class property in codegen. */
  tag?: string
  /** Per-instance ctor-param overrides, keyed by param name. */
  values?: Record<string, unknown>
}

/**
 * The whole model: one global node store and the dictionary of trees.
 *
 * A node belongs to exactly one tree; that owning relationship is recoverable
 * by scanning trees' `list_of_nodes` (and indexed by {@link ownerTreeOf} when a
 * fast lookup is needed).
 */
export interface Forest {
  /** The single global node store. */
  nodes: Record<NodeId, TreeNode>
  /** `"main"` + every group + every block definition, keyed by tree name. */
  trees: Record<TreeName, Tree>
  /** Name of the root tree, usually `"main"`. */
  mainTreeName: TreeName
}

// ---------------------------------------------------------------------------
// Store helpers (thin, pure-ish operations over a Forest).
// ---------------------------------------------------------------------------

/** Create an empty forest with a single (empty) main tree. */
export function emptyForest(mainTreeName: TreeName = 'main'): Forest {
  return {
    nodes: {},
    trees: {
      [mainTreeName]: {
        name: mainTreeName,
        list_of_nodes: [],
        list_of_connections: [],
        inputs: [],
        outputs: [],
        params: {},
      },
    },
    mainTreeName,
  }
}

/** Look up a tree by name (throws if absent - callers expect registered trees). */
export function getTree(forest: Forest, name: TreeName): Tree {
  const tree = forest.trees[name]
  if (!tree) throw new Error(`Tree not found: ${name}`)
  return tree
}

/** Look up a node by id (throws if absent). */
export function getNode(forest: Forest, id: NodeId): TreeNode {
  const node = forest.nodes[id]
  if (!node) throw new Error(`Node not found: ${id}`)
  return node
}

/** Register (or replace) a tree definition. */
export function addTree(forest: Forest, tree: Tree): void {
  forest.trees[tree.name] = tree
}

/**
 * Add a node instance to the store and attach it to a tree's `list_of_nodes`.
 *
 * `node.name` may reference either a registered composite {@link Tree} OR a
 * leaf block resolved by the codegen/render `BlockResolver` (the base case).
 * The model cannot distinguish the two - that is the resolver's job - so no
 * registration invariant is enforced here.
 */
export function addNode(forest: Forest, treeName: TreeName, node: TreeNode): void {
  const tree = getTree(forest, treeName)
  forest.nodes[node.id] = node
  if (!tree.list_of_nodes.includes(node.id)) tree.list_of_nodes.push(node.id)
}

/** The set of trees that directly contain a given node id, by `list_of_nodes`. */
export function ownerTreeOf(forest: Forest, id: NodeId): TreeName | null {
  for (const name of Object.keys(forest.trees)) {
    if (forest.trees[name]!.list_of_nodes.includes(id)) return name
  }
  return null
}

/** Resolve the {@link Tree} a node instantiates, or null if not registered. */
export function referencedTree(forest: Forest, node: TreeNode): Tree | null {
  return forest.trees[node.name] ?? null
}

/**
 * Parse a boundary ref. `"x"` -> `{ local: 'x' }`; `"n7@out"` ->
 * `{ childId: 'n7', childPort: 'out' }`.
 */
export function parseBoundaryRef(
  ref: string
): { local: string } | { childId: NodeId; childPort: string } {
  const at = ref.indexOf('@')
  if (at < 0) return { local: ref }
  return { childId: ref.slice(0, at), childPort: ref.slice(at + 1) }
}

/** Format a routed boundary ref: `(childId, port) -> "childId@port"`. */
export function boundaryRef(childId: NodeId, childPort: string): string {
  return `${childId}@${childPort}`
}
