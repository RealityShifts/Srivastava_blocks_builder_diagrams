/**
 * Step 4: the two-way adapter between the legacy serialized graph (`GraphData`,
 * the payload `getGraphData`/`importGraph` already exchange) and the new
 * {@link Forest} model.
 *
 * We adapt through the SERIALIZED payload rather than the live Rete editor on
 * purpose: the payload is the stable persistence boundary, so the conversion
 * touches no Rete plumbing and the live app keeps running unchanged behind it.
 *
 * Direction in (`graphDataToForest`) is what feeds codegen-from-tree (Step 5)
 * and render-from-dynamic-tree (Step 6). Direction out (`forestToGraphData`)
 * is what persistence (Step 7) will serialize.
 *
 * The legacy model is FLAT: every node (top-level, grouped child, and group
 * facade) lives in one `nodes[]` array, with `groupId` membership and a
 * separate `groups[]` table. We reconstruct the nested Tree structure from
 * that flatness:
 *   - each group -> a Tree named by the group's name (its class name);
 *   - a group facade node -> a TreeNode in its PARENT tree referencing that Tree;
 *   - a grouped child -> a TreeNode inside its group's Tree;
 *   - everything ungrouped -> a TreeNode in `main`.
 */

import { normalize } from '../shape.ts'
import type { ManifestEntry, NodeKind, NodeLike, Connection as FlatConnection } from '../types.ts'
import type { Forest, Tree, TreeNode, NodeId } from './model.ts'
import { emptyForest, makeConnection, boundaryRef } from './model.ts'
import type { BlockResolver, ResolvedBlock } from './signature.ts'
import { getInputOutputParamsSignature } from './signature.ts'

/** The manifest catalogue keyed by block name (as loaded from the JSON). */
export type ManifestCatalogue = Record<string, ManifestEntry>

/** A serialized node spec, as produced by the legacy `getGraphData`. */
export interface GraphNodeSpec {
  id: string
  name: string
  kind: NodeKind
  instanceName?: string
  tag?: string
  values?: Record<string, unknown>
  groupId?: string | null
  /** Group facades only: their group id + boundary port map. */
  portMap?: { inputs?: any[]; outputs?: any[]; params?: any[] }
  /** Non-facade nodes: ctor param names exposed as `__param__*` ports. */
  exposedParams?: string[]
  position?: { x: number; y: number }
}

/** A serialized connection. */
export interface GraphConnSpec {
  source: string
  sourceOutput: string
  target: string
  targetInput: string
}

/** A serialized group descriptor. */
export interface GraphGroupSpec {
  id: string
  name: string
  description?: string
  tag?: string
  collapsed?: boolean
  facadeNodeId?: string | null
  /** The enclosing group's id when nested, else null/undefined (top-level). */
  memberOf?: string | null
}

/** Facade reconstruction metadata captured during import, keyed by facade node id. */
export interface FacadeMeta {
  inPorts: any[]
  outPorts: any[]
  params: any[]
  groupId: string
}

/** The legacy serialized graph payload. */
export interface GraphData {
  version?: number
  framework?: string
  nodes: GraphNodeSpec[]
  connections: GraphConnSpec[]
  groups?: GraphGroupSpec[]
}

// ---------------------------------------------------------------------------
// Manifest-backed BlockResolver
// ---------------------------------------------------------------------------

/**
 * Build the {@link BlockResolver} the signature/expand modules need, backed by
 * the manifest catalogue and a map of per-node exposed-param sets recovered
 * from the serialized specs.
 *
 * A node resolves to a leaf block when its `name` is in the catalogue; group
 * trees (whose names are NOT block names) return null so the signature recurses.
 */
export function makeManifestResolver(
  catalogue: ManifestCatalogue,
  exposedByNode: Map<NodeId, Set<string>> = new Map()
): BlockResolver {
  return (node: TreeNode): ResolvedBlock | null => {
    const entry = catalogue[node.name]
    if (!entry) return null
    return {
      kind: entry.kind,
      inputs: entry.inputs.map((p) => ({
        name: p.name,
        shape: normalize(p.shape),
        dtype: p.dtype,
        optional: p.optional,
        variadic: p.variadic,
      })),
      outputs: entry.outputs.map((p) => ({
        name: p.name,
        shape: normalize(p.shape),
        dtype: p.dtype,
      })),
      ctor: entry.ctor.map((c) => ({
        name: c.name,
        type: c.type,
        default: c.default,
        required: c.required,
        choices: c.choices,
      })),
      exposedParams: exposedByNode.get(node.id),
    }
  }
}

// ---------------------------------------------------------------------------
// GraphData -> Forest
// ---------------------------------------------------------------------------

/** Ensure an (empty) Tree exists under `name`, returning it. */
function ensureTree(forest: Forest, name: string): Tree {
  let t = forest.trees[name]
  if (!t) {
    t = { name, list_of_nodes: [], list_of_connections: [], inputs: [], outputs: [], params: {} }
    forest.trees[name] = t
  }
  return t
}

/**
 * Convert a legacy serialized graph into a {@link Forest}, plus the per-node
 * exposed-param map needed to build the resolver.
 *
 * Group facades become composite Trees; the facade node becomes a TreeNode in
 * its parent referencing that Tree. Each group's boundary refs are derived from
 * the facade's `portMap` (childId@childPort), matching how `buildDynamicMainTree`
 * and `getInputOutputParamsSignature` read them.
 */
export function graphDataToForest(data: GraphData): {
  forest: Forest
  exposedByNode: Map<NodeId, Set<string>>
  facadeMeta: Map<NodeId, FacadeMeta>
} {
  const forest = emptyForest('main')
  const exposedByNode = new Map<NodeId, Set<string>>()
  const facadeMeta = new Map<NodeId, FacadeMeta>()

  // 1. Register a Tree for every group (named by its class name).
  const groupById = new Map<string, GraphGroupSpec>()
  const treeNameForGroup = new Map<string, string>()
  for (const g of data.groups ?? []) {
    groupById.set(g.id, g)
    const name = g.name || `Group_${g.id}`
    treeNameForGroup.set(g.id, name)
    ensureTree(forest, name)
  }

  // 2. Map facadeNodeId -> the group it represents, so we can route a facade
  //    node to its Tree and place its boundary refs.
  const facadeNodeToGroup = new Map<string, string>()
  for (const g of data.groups ?? []) {
    if (g.facadeNodeId) facadeNodeToGroup.set(g.facadeNodeId, g.id)
  }

  // 3. Place each node into its owning tree.
  //    - facade node N for group G -> TreeNode in N's PARENT, referencing the
  //      Tree named for G;
  //    - grouped child (groupId set, not a facade) -> TreeNode in its group Tree;
  //    - else -> TreeNode in main.
  for (const spec of data.nodes) {
    const node: TreeNode = {
      id: spec.id,
      // A facade's referenced tree is the GROUP's class name; everything else
      // references its block name directly (or its instanceName when renamed -
      // rename-to-tree is handled by the live editor via renameNode, so the
      // serialized `name` is authoritative here).
      name: spec.name,
      tag: spec.tag || undefined,
      values: spec.values ? { ...spec.values } : {},
    }

    if (spec.exposedParams && spec.exposedParams.length) {
      exposedByNode.set(spec.id, new Set(spec.exposedParams))
    }

    const asFacadeGroup = facadeNodeToGroup.get(spec.id)
    if (asFacadeGroup) {
      // Facade node: references the group's Tree, and lives in its PARENT tree.
      // The parent is the group's `memberOf` (the enclosing group when nested),
      // NOT the facade's own `groupId` - a top-level facade carries its own gid
      // there, so trusting it would place the group inside itself.
      node.name = treeNameForGroup.get(asFacadeGroup)!
      const memberOf = groupById.get(asFacadeGroup)?.memberOf
      const parentName = memberOf ? treeNameForGroup.get(memberOf) ?? 'main' : 'main'
      const parentTree = ensureTree(forest, parentName)
      forest.nodes[node.id] = node
      parentTree.list_of_nodes.push(node.id)

      // Capture the group Tree's boundary, preserving the FACADE port names
      // (in0/out0/...) the boundary edges use - those names must survive so the
      // reconstructed facade ports line up with the parent's connections.
      const groupTree = ensureTree(forest, treeNameForGroup.get(asFacadeGroup)!)
      const pmIn = spec.portMap?.inputs ?? []
      const pmOut = spec.portMap?.outputs ?? []
      groupTree.inputs = pmIn
        .filter((m: any) => m?.childNodeId && m?.childPort)
        .map((m: any) => boundaryRef(m.childNodeId, m.childPort))
      groupTree.outputs = pmOut
        .filter((m: any) => m?.childNodeId && m?.childPort)
        .map((m: any) => boundaryRef(m.childNodeId, m.childPort))
      // Stash the facade's declared port specs + portMap so forestToGenerateInput
      // can rebuild the facade verbatim (names, shapes, optionality, routing).
      facadeMeta.set(spec.id, {
        inPorts: spec.portMap?.inputs ?? [],
        outPorts: spec.portMap?.outputs ?? [],
        params: spec.portMap?.params ?? [],
        groupId: asFacadeGroup,
      })
      continue
    }

    // A non-facade node's owner is the group it's a member of, or main.
    const ownerTreeName = spec.groupId ? treeNameForGroup.get(spec.groupId) ?? 'main' : 'main'
    const ownerTree = ensureTree(forest, ownerTreeName)
    forest.nodes[node.id] = node
    ownerTree.list_of_nodes.push(node.id)
  }

  // 4. Distribute connections into the tree that owns BOTH endpoints. A
  //    connection touching a facade boundary stays in the parent tree; internal
  //    group edges land in the group Tree.
  const treeOf = new Map<NodeId, string>()
  for (const name of Object.keys(forest.trees)) {
    for (const id of forest.trees[name]!.list_of_nodes) treeOf.set(id, name)
  }
  for (const c of data.connections) {
    // Skip __param__ wires: those are the exposed-param plumbing, captured via
    // exposedByNode instead of as graph edges.
    if (String(c.targetInput).startsWith('__param__')) continue
    const st = treeOf.get(c.source)
    const tt = treeOf.get(c.target)
    if (!st || !tt) continue
    const owner = st === tt ? st : 'main'
    const conn = makeConnection(c.source, c.sourceOutput, c.target, c.targetInput)
    forest.trees[owner]?.list_of_connections.push(conn)
  }

  return { forest, exposedByNode, facadeMeta }
}

// ---------------------------------------------------------------------------
// Forest -> legacy codegen input (NodeLike[] + Connection[])
// ---------------------------------------------------------------------------

/**
 * Reconstruct the flat `NodeLike[]` / `Connection[]` that the EXISTING
 * `generate()` consumes, from a forest's main tree.
 *
 * Scope (Step 5): the FLAT case - a main tree whose children are all leaf
 * blocks (no group facades). For this case the output is identical to the
 * legacy path by construction (same nodes, same edges, same manifest entries),
 * which is exactly the parity the plan's Step 5 gate requires.
 *
 * Grouped graphs are emitted via a dedicated tree-based emitter (next task),
 * NOT by re-synthesizing facade entries here - re-synthesizing them would
 * re-introduce the very `makeGroupEntry`/`portMap` coupling the restructure
 * retires. `hasGroups` reports whether this forest needs that path.
 */
export function forestToFlatCodegen(
  forest: Forest,
  catalogue: ManifestCatalogue
): { nodes: NodeLike[]; connections: FlatConnection[]; hasGroups: boolean } {
  const main = forest.trees[forest.mainTreeName]
  if (!main) throw new Error(`Main tree not found: ${forest.mainTreeName}`)

  const isGroupTree = (name: string) => !catalogue[name] && Boolean(forest.trees[name])
  let hasGroups = false
  const nodes: NodeLike[] = []

  for (const id of main.list_of_nodes) {
    const node = forest.nodes[id]
    if (!node) continue
    if (isGroupTree(node.name)) {
      hasGroups = true
      continue
    }
    const entry = catalogue[node.name]
    if (!entry) continue
    nodes.push({
      id: node.id,
      entry,
      name: '', // legacy instanceName; rename is modeled by tree name, not here
      tag: node.tag ?? '',
      groupId: null,
      values: { ...(node.values ?? {}) },
    })
  }

  const idSet = new Set(nodes.map((n) => n.id))
  const connections: FlatConnection[] = main.list_of_connections
    .filter((c) => idSet.has(c.from) && idSet.has(c.to))
    .map((c) => ({
      id: c.id,
      source: c.from,
      sourceOutput: c.fromOutput,
      target: c.to,
      targetInput: c.toInput,
    }))

  return { nodes, connections, hasGroups }
}

// ---------------------------------------------------------------------------
// Forest -> full codegen input (flat NodeLike[] + Connection[], groups included)
// ---------------------------------------------------------------------------

/**
 * Reconstruct the COMPLETE flat `NodeLike[]` / `Connection[]` that the existing
 * `generate()` consumes - including group facades with synthetic group entries
 * and portMaps - from a forest. This makes the FOREST the single source of
 * truth for codegen while reusing the proven emission core (no discrepancy).
 *
 * Per tree:
 *   - a leaf-block child -> a flat NodeLike carrying its owning group's id;
 *   - a group tree -> one facade NodeLike in its PARENT, with a synthetic
 *     `kind:'group'` entry whose portMap is derived from the group tree's
 *     boundary refs (`childId@port`) and whose boundary port names come from
 *     `getInputOutputParamsSignature` (so facade-edge port names line up).
 *
 * Boundary edges (parent edge wired to a facade port) and internal group edges
 * are emitted verbatim from the forest connections; `partitionByGroup` /
 * `buildSubgraphView` in codegen then reconstruct each group class exactly as
 * for a live editor graph.
 */
export function forestToGenerateInput(
  forest: Forest,
  catalogue: ManifestCatalogue,
  exposedByNode: Map<NodeId, Set<string>> = new Map(),
  facadeMeta: Map<NodeId, FacadeMeta> = new Map()
): { nodes: NodeLike[]; connections: FlatConnection[] } {
  const isGroupTree = (name: string) => !catalogue[name] && Boolean(forest.trees[name])
  const nodes: NodeLike[] = []
  const connections: FlatConnection[] = []

  // The original group id (gid) for each group tree, recovered from the facade
  // metadata. `entry.groupId` and child membership use this gid, exactly as the
  // live editor does - so two same-name groups still collapse to one class and
  // facade boundary edges (wired on in0/out0...) match verbatim.
  const gidForTree = new Map<string, string>()
  for (const [facadeId, fm] of facadeMeta) {
    const facadeNode = forest.nodes[facadeId]
    if (facadeNode) gidForTree.set(facadeNode.name, fm.groupId)
  }

  // Node id -> owning gid (for child `groupId` membership). A child of a group
  // tree is a member of that tree's gid; main's direct children belong to none.
  const memberGidOf = new Map<NodeId, string | null>()
  for (const name of Object.keys(forest.trees)) {
    const gid = isGroupTree(name) ? gidForTree.get(name) ?? name : null
    for (const id of forest.trees[name]!.list_of_nodes) memberGidOf.set(id, gid)
  }

  for (const name of Object.keys(forest.trees)) {
    const tree = forest.trees[name]!
    for (const id of tree.list_of_nodes) {
      const node = forest.nodes[id]
      if (!node) continue

      if (isGroupTree(node.name)) {
        // Facade node: rebuild VERBATIM from the captured facade metadata so
        // port names/shapes/routing and the gid match the live editor exactly.
        const fm = facadeMeta.get(node.id)
        const gid = fm?.groupId ?? node.name
        const pmIn = fm?.inPorts ?? []
        const pmOut = fm?.outPorts ?? []
        const entry: ManifestEntry = {
          name: node.name, // group name -> drives groupClassName -> class name
          module: '__group__',
          kind: 'group',
          ctor: [],
          inputs: pmIn.map((m: any) => ({
            name: m.facadePort,
            shape: m.shape ?? ['...'],
            dtype: m.dtype ?? 'any',
            optional: Boolean(m.optional),
          })),
          outputs: pmOut.map((m: any) => ({
            name: m.facadePort,
            shape: m.shape ?? ['...'],
            dtype: m.dtype ?? 'any',
          })),
          groupId: gid,
          portMap: { inputs: pmIn, outputs: pmOut, params: fm?.params ?? [] },
        } as ManifestEntry
        nodes.push({
          id: node.id,
          entry,
          name: '',
          tag: node.tag ?? '',
          groupId: memberGidOf.get(node.id) ?? null,
          values: { ...(node.values ?? {}) },
        })
        continue
      }

      // Leaf block.
      const entry = catalogue[node.name]
      if (!entry) continue
      nodes.push({
        id: node.id,
        entry,
        name: '',
        tag: node.tag ?? '',
        groupId: memberGidOf.get(node.id) ?? null,
        values: { ...(node.values ?? {}) },
      })
    }

    for (const c of tree.list_of_connections) {
      connections.push({
        id: c.id,
        source: c.from,
        sourceOutput: c.fromOutput,
        target: c.to,
        targetInput: c.toInput,
      })
    }
  }

  return { nodes, connections }
}

// ---------------------------------------------------------------------------
// Forest -> GraphData (flatten back)
// ---------------------------------------------------------------------------

/**
 * Flatten a {@link Forest} back into the legacy serialized payload. Group Trees
 * become `groups[]` entries with a facade node; every leaf TreeNode becomes a
 * flat node carrying its owning `groupId`.
 *
 * This is the inverse used by persistence (Step 7). It is intentionally lossy
 * about layout (positions/offsets) - those are re-derived by the editor - but
 * round-trips structure, names, tags, values, and group boundaries.
 */
export function forestToGraphData(
  forest: Forest,
  catalogue: ManifestCatalogue,
  exposedByNode: Map<NodeId, Set<string>> = new Map()
): GraphData {
  const out: GraphData = { version: 1, nodes: [], connections: [], groups: [] }
  const isGroupTree = (name: string) => !catalogue[name] && forest.trees[name]

  // Group ids: reuse the tree name as a stable group id surrogate.
  for (const name of Object.keys(forest.trees)) {
    if (name === forest.mainTreeName) continue
    if (!isGroupTree(name)) continue
    out.groups!.push({
      id: name,
      name,
      collapsed: true,
      facadeNodeId: `facade::${name}`,
    })
  }

  const emitNode = (node: TreeNode, groupId: string | null) => {
    const entry = catalogue[node.name]
    const spec: GraphNodeSpec = {
      id: node.id,
      name: node.name,
      kind: (entry?.kind ?? (isGroupTree(node.name) ? 'group' : 'module')) as NodeKind,
      tag: node.tag ?? '',
      values: { ...(node.values ?? {}) },
      groupId: groupId ?? undefined,
    }
    const exposed = exposedByNode.get(node.id)
    if (exposed && exposed.size) spec.exposedParams = [...exposed]
    out.nodes.push(spec)
  }

  for (const name of Object.keys(forest.trees)) {
    const tree = forest.trees[name]!
    const groupId = name === forest.mainTreeName ? null : isGroupTree(name) ? name : null
    for (const id of tree.list_of_nodes) {
      const node = forest.nodes[id]
      if (node) emitNode(node, groupId)
    }
    for (const c of tree.list_of_connections) {
      out.connections.push({
        source: c.from,
        sourceOutput: c.fromOutput,
        target: c.to,
        targetInput: c.toInput,
      })
    }
  }

  return out
}
