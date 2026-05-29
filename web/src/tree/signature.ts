/**
 * Step 2 - the keystone: `getInputOutputParamsSignature(tree)`.
 *
 * THE single source of truth for three consumers that must never disagree
 * (their disagreement is the bug class this restructure removes):
 *   1. the rendering boundary - what ports a collapsed facade shows;
 *   2. the codegen `forward(...)` signature of `class <tree.name>`;
 *   3. the codegen `__init__(...)` signature of `class <tree.name>`.
 *
 * Computed PER-TREE, IN ISOLATION: it reads only `tree.list_of_nodes`,
 * `tree.list_of_connections`, and (for child nodes that reference other trees)
 * each child's OWN recursively-computed signature. It never reaches into a
 * `dynamicMainTree` or sibling trees. That isolation is what makes a Tree a
 * genuine reusable class.
 *
 * See web/RESTRUCTURE_PLAN.md "getInputOutputParamsSignature" for the exact
 * derivation rules this implements.
 */

import type { Shape } from '../shape.ts'
import type { NodeKind } from '../types.ts'
import type { Forest, Tree, TreeNode, NodeId } from './model.ts'
import { referencedTree } from './model.ts'

/** A boundary tensor port (a forward input or a forward output). */
export interface SigPort {
  /** Boundary port name (the local name, NOT `childId@port`). */
  name: string
  shape: Shape
  dtype: string
  optional?: boolean
  /** Provenance: which inner child port this boundary slot routes to. */
  childId: NodeId
  childPort: string
}

/** A boundary init param (a `__init__` argument). */
export interface SigParam {
  /** `__init__` arg name (sanitized + deduped). */
  name: string
  dtype: string
  /** Present => optional kwarg with this default; absent => required. */
  default?: unknown
  /** For learnable params / buffers. */
  shape?: Shape
  /** Where the value is injected at init time. */
  childId: NodeId
  /** The `__param__` port it feeds, if exposed-param style. */
  childPort?: string
}

/** The full per-tree signature. */
export interface Signature {
  /** forward() tensor args, in (node order, port order). */
  forwardInputs: SigPort[]
  /** forward() return tuple, in (node order, port order). */
  forwardOutputs: SigPort[]
  /** __init__ args. */
  initParams: SigParam[]
}

/**
 * The base-case description of a leaf block, supplied by the adapter (backed by
 * the manifest). The signature function uses this when a child node references
 * a block that is NOT a composite tree - i.e. the recursion's leaves.
 *
 * Returning `null` means "not a known leaf block" - the function then falls
 * back to recursing into the child's referenced tree.
 */
export interface ResolvedBlock {
  kind: NodeKind
  inputs: Array<{ name: string; shape: Shape; dtype?: string; optional?: boolean; variadic?: boolean }>
  outputs: Array<{ name: string; shape: Shape; dtype?: string }>
  ctor: Array<{ name: string; type: string; default?: unknown; required?: boolean; choices?: string[] }>
  /** Ctor param names currently exposed as `__param__` ports on this instance. */
  exposedParams?: Set<string>
}

/**
 * Resolves a child node to its leaf-block description, or null when the child
 * is a composite tree (recurse instead). The adapter implements this over the
 * manifest; tests provide a small stub.
 */
export type BlockResolver = (node: TreeNode, forest: Forest) => ResolvedBlock | null

/** Internal: a unified per-child view the derivation iterates over. */
interface ChildView {
  id: NodeId
  kind: NodeKind
  inputs: Array<{ name: string; shape: Shape; dtype: string; optional: boolean; variadic: boolean }>
  outputs: Array<{ name: string; shape: Shape; dtype: string }>
  /** init params this child contributes, before tree-level dedup. */
  params: Array<{ name: string; dtype: string; default?: unknown; shape?: Shape; childPort?: string }>
}

const ANY = 'any'

/** A name-deduping allocator: `x`, `x2`, `x3`, ... (matches codegen). */
function makeAllocator(used: Set<string>): (base: string) => string {
  return (base: string) => {
    let candidate = base || 'arg'
    let i = 2
    while (used.has(candidate)) candidate = `${base}${i++}`
    used.add(candidate)
    return candidate
  }
}

/**
 * Build the unified per-child view for one node.
 *
 * - If the resolver returns a leaf block, use its ports/ctor directly.
 * - Otherwise the child references a composite tree: recurse, and project that
 *   tree's signature into ports (its forwardInputs/Outputs become this child's
 *   input/output ports; its initParams become this child's params). This is the
 *   recursion the plan calls for - composition at boundaries, not peeking.
 */
function childView(
  node: TreeNode,
  forest: Forest,
  resolve: BlockResolver,
  seen: Set<string>
): ChildView {
  const block = resolve(node, forest)
  if (block) {
    const exposed = block.exposedParams ?? new Set<string>()
    const params: ChildView['params'] = []
    if (block.kind === 'const' || block.kind === 'learnable') {
      // A const/learnable child surfaces a single init param. The param name is
      // refined at the tree level; here we seed it from the child's tag/value.
      const valueParam = block.ctor.find((c) => c.name === 'value') ?? block.ctor[0]
      params.push({
        name: String(node.tag ?? node.name ?? 'param'),
        dtype: valueParam?.type ?? (block.kind === 'learnable' ? 'Tensor' : 'float'),
        default: node.values?.[valueParam?.name ?? 'value'] ?? valueParam?.default,
        shape: block.kind === 'learnable' ? block.outputs[0]?.shape : undefined,
        childPort: valueParam?.name,
      })
    } else {
      for (const c of block.ctor) {
        if (!exposed.has(c.name)) continue
        params.push({
          name: c.name,
          dtype: c.type,
          default: node.values?.[c.name] ?? c.default,
          childPort: c.name,
        })
      }
    }
    return {
      id: node.id,
      kind: block.kind,
      inputs: block.inputs.map((p) => ({
        name: p.name,
        shape: p.shape,
        dtype: p.dtype ?? ANY,
        optional: Boolean(p.optional),
        variadic: Boolean(p.variadic),
      })),
      outputs: block.outputs.map((p) => ({ name: p.name, shape: p.shape, dtype: p.dtype ?? ANY })),
      params,
    }
  }

  // Composite tree child: project its own signature into ports.
  const childTree = referencedTree(forest, node)
  if (!childTree) {
    // Unknown reference: treat as an opaque zero-port node.
    return { id: node.id, kind: 'module', inputs: [], outputs: [], params: [] }
  }
  const sig = signatureOf(childTree, forest, resolve, seen)
  return {
    id: node.id,
    kind: 'group',
    inputs: sig.forwardInputs.map((p) => ({
      name: p.name,
      shape: p.shape,
      dtype: p.dtype,
      optional: Boolean(p.optional),
      variadic: false,
    })),
    outputs: sig.forwardOutputs.map((p) => ({ name: p.name, shape: p.shape, dtype: p.dtype })),
    params: sig.initParams.map((p) => ({
      name: p.name,
      dtype: p.dtype,
      default: p.default,
      shape: p.shape,
      childPort: p.childPort,
    })),
  }
}

/**
 * Core derivation, with a `seen` guard against cyclic tree references (a tree
 * that, directly or indirectly, contains itself - which would be a user error,
 * but we must not infinite-loop on it).
 */
function signatureOf(
  tree: Tree,
  forest: Forest,
  resolve: BlockResolver,
  seen: Set<string>
): Signature {
  if (seen.has(tree.name)) {
    // Cycle: stop recursing. Return an empty signature for this level.
    return { forwardInputs: [], forwardOutputs: [], initParams: [] }
  }
  const nextSeen = new Set(seen)
  nextSeen.add(tree.name)

  const children: ChildView[] = tree.list_of_nodes
    .map((id) => forest.nodes[id])
    .filter((n): n is TreeNode => Boolean(n))
    .map((n) => childView(n, forest, resolve, nextSeen))

  // Membership tests over INTERNAL connections only (both ends in this tree).
  const childIds = new Set(children.map((c) => c.id))
  const internal = tree.list_of_connections.filter((c) => childIds.has(c.from) && childIds.has(c.to))
  const hasIncoming = new Set(internal.map((c) => `${c.to}/${c.toInput}`))
  const hasOutgoing = new Set(internal.map((c) => `${c.from}/${c.fromOutput}`))

  const inUsed = new Set<string>()
  const outUsed = new Set<string>()
  const allocIn = makeAllocator(inUsed)
  const allocOut = makeAllocator(outUsed)

  const forwardInputs: SigPort[] = []
  const forwardOutputs: SigPort[] = []
  const initParams: SigParam[] = []
  const paramUsed = new Set<string>()
  const allocParam = makeAllocator(paramUsed)

  for (const child of children) {
    // Explicit `input` / `output` kinds are graph endpoints, not ports of THIS
    // class - they're handled by codegen's input/output machinery, matching
    // findEntryInputs which skips kind input/output.
    if (child.kind === 'input' || child.kind === 'output') continue

    // forwardInputs: required, non-variadic input ports with no internal wire.
    for (const port of child.inputs) {
      if (port.optional || port.variadic) continue
      if (hasIncoming.has(`${child.id}/${port.name}`)) continue
      forwardInputs.push({
        name: allocIn(port.name),
        shape: port.shape,
        dtype: port.dtype,
        optional: false,
        childId: child.id,
        childPort: port.name,
      })
    }

    // forwardOutputs: output ports with no internal outgoing wire (graph sink).
    for (const port of child.outputs) {
      if (hasOutgoing.has(`${child.id}/${port.name}`)) continue
      forwardOutputs.push({
        name: allocOut(port.name),
        shape: port.shape,
        dtype: port.dtype,
        childId: child.id,
        childPort: port.name,
      })
    }

    // initParams: const/learnable children + exposed (unwired) param ports.
    for (const p of child.params) {
      // An exposed param fed by an internal wire is satisfied internally - skip.
      if (p.childPort && hasIncoming.has(`${child.id}/__param__${p.childPort}`)) continue
      initParams.push({
        name: allocParam(p.name),
        dtype: p.dtype,
        default: p.default,
        shape: p.shape,
        childId: child.id,
        childPort: p.childPort,
      })
    }
  }

  return { forwardInputs, forwardOutputs, initParams }
}

/**
 * Public entry point. Computes the boundary + init/forward signature of one
 * tree, in isolation, recursing into composite child trees as needed.
 *
 * @param tree     the tree to describe
 * @param forest   the owning forest (for node + child-tree lookup)
 * @param resolve  maps a child node to its leaf-block description, or null to
 *                 recurse into the child's referenced tree
 */
export function getInputOutputParamsSignature(
  tree: Tree,
  forest: Forest,
  resolve: BlockResolver
): Signature {
  return signatureOf(tree, forest, resolve, new Set())
}
