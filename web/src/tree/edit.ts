/**
 * Mutating forest edits that aren't pure derivations: renaming a node (which,
 * because a node's `name` IS a tree name, may create a new tree) and pruning
 * trees no longer referenced by any node.
 */

import type { Forest, Tree, TreeName, NodeId, TreeNode } from './model.ts'
import { getNode, getTree } from './model.ts'
import { freshNodeId } from './ids.ts'

/**
 * Deep-clone a tree under a new name, giving every child node a fresh,
 * instance-namespaced id so the clone is independent of the original. Boundary
 * refs and connections are rewired onto the new child ids.
 *
 * The cloned child NODES are added to `forest.nodes`; the new tree is added to
 * `forest.trees`. Returns the new tree.
 */
function cloneTreeAs(forest: Forest, source: Tree, newName: TreeName): Tree {
  // Map each source child id -> a fresh id, and clone the node object.
  const idMap = new Map<NodeId, NodeId>()
  for (const oldId of source.list_of_nodes) {
    const node = forest.nodes[oldId]
    if (!node) continue
    const newId = freshNodeId('blk')
    idMap.set(oldId, newId)
    forest.nodes[newId] = { ...node, id: newId, values: { ...(node.values ?? {}) } }
  }
  const remap = (id: NodeId): NodeId => idMap.get(id) ?? id
  const remapRef = (ref: string): string => {
    const at = ref.indexOf('@')
    if (at < 0) return ref
    const childId = ref.slice(0, at)
    const rest = ref.slice(at)
    return `${remap(childId)}${rest}`
  }

  const clone: Tree = {
    name: newName,
    list_of_nodes: source.list_of_nodes.map(remap),
    list_of_connections: source.list_of_connections.map((c) => ({
      from: remap(c.from),
      fromOutput: c.fromOutput,
      to: remap(c.to),
      toInput: c.toInput,
      id: `${remap(c.from)}::${c.fromOutput}->${remap(c.to)}::${c.toInput}`,
    })),
    inputs: source.inputs.map(remapRef),
    outputs: source.outputs.map(remapRef),
    params: { ...source.params },
  }
  forest.trees[newName] = clone
  return clone
}

/**
 * Rename a node, i.e. point it at the tree named `newName`.
 *
 * Because a node's `name` IS a tree name, this has definition-level semantics:
 *   - If `newName` already names a tree, the node simply references it (it now
 *     SHARES that definition with every other node of that name).
 *   - If `newName` is free, a NEW tree is created by cloning the node's current
 *     tree under `newName`, then the node is repointed to it. This is the
 *     "rename one instance without affecting its siblings" mechanism: a single
 *     ConvBlock renamed to "MyConv" gets its own one-node tree wrapping the
 *     same underlying block.
 *
 * Pass `pruneAfter` (default true) to drop any tree the rename orphaned.
 * Returns the (possibly newly created) tree the node now references.
 */
export function renameNode(
  forest: Forest,
  nodeId: NodeId,
  newName: TreeName,
  pruneAfter = true
): Tree {
  const node = getNode(forest, nodeId)
  const oldName = node.name
  if (oldName === newName) return getTree(forest, newName)

  let target = forest.trees[newName]
  if (!target) {
    // Create the new tree by cloning whatever the node currently references.
    // If the node referenced a leaf block (no registered tree), the clone is a
    // fresh one-node tree wrapping that block.
    const source = forest.trees[oldName]
    if (source) {
      target = cloneTreeAs(forest, source, newName)
    } else {
      // Leaf-block node with no wrapper tree: wrap it now under newName.
      const childId = freshNodeId('blk')
      forest.nodes[childId] = { id: childId, name: oldName, values: { ...(node.values ?? {}) } }
      target = {
        name: newName,
        list_of_nodes: [childId],
        list_of_connections: [],
        inputs: [],
        outputs: [],
        params: {},
      }
      forest.trees[newName] = target
    }
  }

  node.name = newName
  if (pruneAfter) pruneUnusedTrees(forest)
  return target
}

/**
 * Remove every tree that no longer appears as some node's `name`, EXCEPT the
 * main tree (the root is kept even with zero references) and the leaf trees
 * still referenced indirectly through another tree's children.
 *
 * A tree is "used" if any node anywhere in the forest references it by name -
 * including child nodes inside other trees. We therefore scan ALL nodes in the
 * store, not just the main tree's roots.
 *
 * Removing a tree also deletes the child nodes it solely owned (nodes that
 * appear only in the pruned tree's `list_of_nodes`). Returns the names removed.
 */
export function pruneUnusedTrees(forest: Forest): TreeName[] {
  // A tree is reachable if it's the main tree, or referenced by a LIVE node.
  // "Live" must be computed transitively: a node inside a pruned tree doesn't
  // keep that node's referenced tree alive. We iterate to a fixpoint.
  const keep = new Set<TreeName>([forest.mainTreeName])
  let changed = true
  while (changed) {
    changed = false
    // Every node that lives inside a kept tree contributes its referenced name.
    for (const name of keep) {
      const tree = forest.trees[name]
      if (!tree) continue
      for (const childId of tree.list_of_nodes) {
        const child = forest.nodes[childId]
        if (!child) continue
        if (forest.trees[child.name] && !keep.has(child.name)) {
          keep.add(child.name)
          changed = true
        }
      }
    }
  }

  const removed: TreeName[] = []
  const survivingNodeOwners = new Map<NodeId, number>()
  for (const name of keep) {
    for (const id of forest.trees[name]?.list_of_nodes ?? []) {
      survivingNodeOwners.set(id, (survivingNodeOwners.get(id) ?? 0) + 1)
    }
  }

  for (const name of Object.keys(forest.trees)) {
    if (keep.has(name)) continue
    const tree = forest.trees[name]!
    // Delete child nodes owned solely by this (pruned) tree.
    for (const id of tree.list_of_nodes) {
      if (!survivingNodeOwners.has(id)) delete forest.nodes[id]
    }
    delete forest.trees[name]
    removed.push(name)
  }
  return removed
}
