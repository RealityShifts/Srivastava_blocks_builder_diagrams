/**
 * Step 3: `buildDynamicMainTree(forest, expanded)` - the derived "current
 * connection graph".
 *
 * This is the PURE computation that replaces the old mutate-the-editor
 * collapse/expand. Source trees are never modified; we build a fresh throwaway
 * Tree (`__dynamic__`) whose nodes/connections reflect every currently-expanded
 * group spliced inline.
 *
 * Splicing one expanded node N (an instance of composite tree T):
 *   1. Substitute N with T's child nodes, given fresh instance-namespaced ids
 *      (`${N.id}/${childId}`) so two expanded instances of T don't collide.
 *   2. Rewrite every connection that referenced N: a `to`/`from` pointing at N
 *      is replaced by the freshened child id resolved from T's matching boundary
 *      ref (`childId@port`), and the port name becomes the inner child port.
 *   3. Append T's internal connections (with freshened endpoint ids).
 *   4. Every connection's id is recomputed via `makeConnection` (collision-free).
 *
 * Expansion recurses: if a spliced child is itself an expanded composite, it is
 * spliced too. `expanded` is the set of NODE ids the user has opened.
 *
 * Nodes left collapsed stay as single nodes referencing their tree; the render
 * adapter (plan Step 6) decides block-vs-group rendering from the tree shape.
 */

import type { Forest, Tree, TreeNode, Connection, NodeId } from './model.ts'
import { makeConnection, referencedTree, parseBoundaryRef } from './model.ts'

/** The derived tree's reserved name. Never registered into `forest.trees`. */
export const DYNAMIC_TREE_NAME = '__dynamic__'

/** A node in the dynamic tree carries its origin chain for debugging/render. */
export interface DynamicNode extends TreeNode {
  /** The original (pre-splice) node id this instance derives from. */
  originId: NodeId
  /** Ids of the expanded ancestor instances this node was spliced under. */
  splicePath: NodeId[]
}

/** Namespaced id for a child spliced under an expanded instance. */
function spliceId(instanceId: NodeId, childId: NodeId): NodeId {
  return `${instanceId}/${childId}`
}

/**
 * Build the dynamic main tree AND the placed node objects in one pass.
 *
 * @param forest    the source forest (never mutated)
 * @param expanded  set of node ids the user has expanded (opened groups)
 */
export function buildDynamicMainTreeWithNodes(
  forest: Forest,
  expanded: Set<NodeId> = new Set()
): { tree: Tree; nodes: DynamicNode[] } {
  const main = forest.trees[forest.mainTreeName]
  if (!main) throw new Error(`Main tree not found: ${forest.mainTreeName}`)

  const outNodes: DynamicNode[] = []
  const outConns: Connection[] = []

  /**
   * Resolve a connection endpoint to its dynamic (id, port). If the endpoint
   * node was spliced (an expanded composite), redirect into the matching inner
   * child port via the child tree's boundary refs. Returns null when the
   * endpoint routes to a dangling boundary (no child binding) - that edge is
   * dropped.
   */
  function rewriteEndpoint(
    localId: NodeId,
    port: string,
    side: 'from' | 'to',
    idOf: (localId: NodeId) => NodeId
  ): { id: NodeId; port: string } | null {
    const dynId = idOf(localId)
    const node = forest.nodes[localId]
    const childTree = node ? referencedTree(forest, node) : null
    const wasSpliced = Boolean(childTree) && expanded.has(localId)
    if (!wasSpliced || !childTree) return { id: dynId, port }

    // The node was replaced by its children: redirect to the bound child port.
    // `from` endpoints map through the child tree's OUTPUT boundary refs;
    // `to` endpoints through its INPUT refs. Refs are `childId@childPort`, and
    // the facade exposes each under its childPort name (matching signature.ts).
    const refs = side === 'from' ? childTree.outputs : childTree.inputs
    for (const ref of refs) {
      const parsed = parseBoundaryRef(ref)
      if ('childPort' in parsed && parsed.childPort === port) {
        return { id: spliceId(dynId, parsed.childId), port: parsed.childPort }
      }
    }
    return null
  }

  /**
   * Place a tree's nodes into the output, expanding any node that is both
   * currently expanded AND a composite tree. `idOf` maps a tree-local node id
   * to its dynamic id (identity at the top level, namespaced when spliced).
   */
  function placeTree(tree: Tree, idOf: (localId: NodeId) => NodeId, splicePath: NodeId[]): void {
    for (const localId of tree.list_of_nodes) {
      const node = forest.nodes[localId]
      if (!node) continue
      const dynId = idOf(localId)
      const childTree = referencedTree(forest, node)
      const isExpandable = Boolean(childTree) && expanded.has(localId)
      if (isExpandable && childTree) {
        placeTree(childTree, (cid) => spliceId(dynId, cid), [...splicePath, dynId])
      } else {
        outNodes.push({ ...node, id: dynId, originId: localId, splicePath })
      }
    }
    for (const conn of tree.list_of_connections) {
      const from = rewriteEndpoint(conn.from, conn.fromOutput, 'from', idOf)
      const to = rewriteEndpoint(conn.to, conn.toInput, 'to', idOf)
      if (!from || !to) continue
      outConns.push(makeConnection(from.id, from.port, to.id, to.port))
    }
  }

  placeTree(main, (id) => id, [])

  return {
    tree: {
      name: DYNAMIC_TREE_NAME,
      list_of_nodes: outNodes.map((n) => n.id),
      list_of_connections: outConns,
      inputs: main.inputs,
      outputs: main.outputs,
      params: main.params,
    },
    nodes: outNodes,
  }
}

/** Build just the dynamic main tree (drops the placed-node objects). */
export function buildDynamicMainTree(forest: Forest, expanded: Set<NodeId> = new Set()): Tree {
  return buildDynamicMainTreeWithNodes(forest, expanded).tree
}
