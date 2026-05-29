/**
 * Step 1: register a manifest block as a one-node {@link Tree}.
 *
 * Every default block from the block library becomes a Tree with exactly one
 * child node whose `name` is the block/operator name. Crucially:
 *
 *   - The TREE's name is what Rete renders and what codegen uses as the class
 *     name. The inner node's name does NOT contribute to the rendered name.
 *     This is the "rename a ConvBlock" mechanism: a one-node tree named
 *     `MyConv` whose child instantiates the `ConvBlock` block renders + emits
 *     as `MyConv`.
 *   - Because a one-node tree has no internal connections, EVERY input port is
 *     dangling and EVERY output port is dangling. So
 *     `getInputOutputParamsSignature(blockTree)` reproduces the block's own
 *     ports exactly - which is what lets rendering and codegen treat leaf
 *     blocks and groups through the same code path.
 *
 * The inner child node also lives in the global node store, but a leaf block
 * tree is recognised at render time by the rule in the plan: "a tree with
 * exactly one child whose name is a known block -> render as that block."
 */

import { normalize } from '../shape.ts'
import type { ManifestEntry } from '../types.ts'
import type { Forest, Tree, TreeNode, ParamSpec } from './model.ts'
import { addTree, addNode } from './model.ts'
import { freshNodeId } from './ids.ts'

/** Build the `params` map for a block tree from its manifest ctor list. */
function paramsFromCtor(entry: ManifestEntry): Record<string, ParamSpec> {
  const params: Record<string, ParamSpec> = {}
  for (const p of entry.ctor) {
    params[p.name] = {
      value: p.default,
      dtype: p.type,
      required: Boolean(p.required),
      choices: p.choices,
    }
  }
  return params
}

/**
 * The set of leaf-block tree names registered into a forest. Used by the
 * render rule to decide "is this tree just a renamed block?" without re-reading
 * the manifest.
 */
export type BlockTreeIndex = Set<string>

/**
 * Register one manifest entry as a one-node tree, returning the created Tree.
 *
 * `treeName` defaults to the block name; pass a custom name to model a rename
 * (e.g. `MyConv` wrapping a `ConvBlock` child). The inner child's `name` is
 * always the underlying block name so render/codegen can resolve the block.
 *
 * Idempotent on tree name: if a tree with `treeName` already exists it is
 * returned untouched (so the manifest can be registered once at startup and
 * re-registration is a no-op).
 */
export function registerBlockTree(
  forest: Forest,
  entry: ManifestEntry,
  treeName: string = entry.name
): Tree {
  const existing = forest.trees[treeName]
  if (existing) return existing

  const childId = freshNodeId('blk')
  const child: TreeNode = {
    id: childId,
    name: entry.name, // the underlying block/operator name
    values: Object.fromEntries(entry.ctor.map((p) => [p.name, p.default])),
  }

  const tree: Tree = {
    name: treeName,
    list_of_nodes: [],
    list_of_connections: [],
    // One-node tree: every port is dangling, routed to the single child.
    inputs: entry.inputs.map((p) => `${childId}@${p.name}`),
    outputs: entry.outputs.map((p) => `${childId}@${p.name}`),
    params: paramsFromCtor(entry),
  }

  addTree(forest, tree)
  addNode(forest, treeName, child)
  return tree
}

/**
 * Register a whole manifest (catalogue of blocks) into a forest, returning the
 * index of leaf-block tree names. Safe to call on an existing forest; existing
 * trees are left in place.
 */
export function registerManifest(
  forest: Forest,
  manifest: ManifestEntry[]
): BlockTreeIndex {
  const index: BlockTreeIndex = new Set()
  for (const entry of manifest) {
    registerBlockTree(forest, entry)
    index.add(entry.name)
  }
  return index
}

/**
 * Normalize a manifest entry's port shapes in place-free fashion: returns a
 * shallow copy with `inputs`/`outputs` shapes run through `normalize` so the
 * tree model sees `Shape` tokens (numbers vs axis vars), matching what the rest
 * of the pipeline expects. Used by the adapter (Step 4) when ingesting raw
 * manifest JSON.
 */
export function normalizeManifestEntry(entry: ManifestEntry): ManifestEntry {
  return {
    ...entry,
    inputs: entry.inputs.map((p) => ({ ...p, shape: normalize(p.shape) })),
    outputs: entry.outputs.map((p) => ({ ...p, shape: normalize(p.shape) })),
  }
}
