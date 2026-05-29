/**
 * Tests for Step 3 (buildDynamicMainTree) plus renameNode / pruneUnusedTrees.
 * Pure logic - run via:  node test-tree-expand.ts
 */

import { emptyForest, addTree, addNode, makeConnection, getTree } from './src/tree/model.ts'
import type { Forest, TreeNode, Tree } from './src/tree/model.ts'
import { buildDynamicMainTree } from './src/tree/expand.ts'
import { renameNode, pruneUnusedTrees } from './src/tree/edit.ts'

let pass = 0
let fail = 0
function check(name: string, cond: boolean, info?: unknown) {
  if (cond) {
    pass++
    console.log(`  ok   ${name}`)
  } else {
    fail++
    console.log(`  FAIL ${name}`, info ?? '')
  }
}

/** Edge set as `from@out->to@in` strings, for order-independent comparison. */
function edgeSet(tree: Tree): Set<string> {
  return new Set(tree.list_of_connections.map((c) => `${c.from}@${c.fromOutput}->${c.to}@${c.toInput}`))
}

/**
 * A forest with:
 *   tree "Encoder": c1(Conv) -> c2(Conv); inputs [c1@x], outputs [c2@y]
 *   main: in0(Input) -> enc(Encoder) -> out0(Output)
 */
function makeForest(): Forest {
  const f = emptyForest()
  // Encoder composite.
  addTree(f, {
    name: 'Encoder',
    list_of_nodes: [],
    list_of_connections: [],
    inputs: ['c1@x'],
    outputs: ['c2@y'],
    params: {},
  })
  const c1: TreeNode = { id: 'c1', name: 'Conv', values: {} }
  const c2: TreeNode = { id: 'c2', name: 'Conv', values: {} }
  addNode(f, 'Encoder', c1)
  addNode(f, 'Encoder', c2)
  getTree(f, 'Encoder').list_of_connections = [makeConnection('c1', 'y', 'c2', 'x')]

  // Main graph.
  const inp: TreeNode = { id: 'in0', name: 'Input', values: {} }
  const enc: TreeNode = { id: 'enc', name: 'Encoder', values: {} }
  const outp: TreeNode = { id: 'out0', name: 'Output', values: {} }
  for (const n of [inp, enc, outp]) addNode(f, 'main', n)
  getTree(f, 'main').list_of_connections = [
    makeConnection('in0', 'y', 'enc', 'x'),
    makeConnection('enc', 'y', 'out0', 'x'),
  ]
  return f
}

// --------------------------------------------------------------------------
// Test 1: collapsed (nothing expanded) -> main graph verbatim.
// --------------------------------------------------------------------------
{
  const f = makeForest()
  const dyn = buildDynamicMainTree(f, new Set())
  check('collapsed: 3 nodes', dyn.list_of_nodes.length === 3, dyn.list_of_nodes)
  check('collapsed: enc still present as one node', dyn.list_of_nodes.includes('enc'))
  const e = edgeSet(dyn)
  check('collapsed: in0->enc and enc->out0', e.has('in0@y->enc@x') && e.has('enc@y->out0@x'), [...e])
}

// --------------------------------------------------------------------------
// Test 2: expand enc -> splice Encoder, rewire boundary edges.
// --------------------------------------------------------------------------
{
  const f = makeForest()
  const dyn = buildDynamicMainTree(f, new Set(['enc']))
  // enc replaced by enc/c1 and enc/c2; in0 and out0 remain.
  check('expanded: enc removed', !dyn.list_of_nodes.includes('enc'))
  check('expanded: enc/c1 and enc/c2 spliced in', dyn.list_of_nodes.includes('enc/c1') && dyn.list_of_nodes.includes('enc/c2'), dyn.list_of_nodes)
  check('expanded: 4 nodes (in0, enc/c1, enc/c2, out0)', dyn.list_of_nodes.length === 4, dyn.list_of_nodes)
  const e = edgeSet(dyn)
  check('expanded: boundary in0->enc rewired to in0->enc/c1', e.has('in0@y->enc/c1@x'), [...e])
  check('expanded: boundary enc->out0 rewired to enc/c2->out0', e.has('enc/c2@y->out0@x'), [...e])
  check('expanded: internal c1->c2 spliced as enc/c1->enc/c2', e.has('enc/c1@y->enc/c2@x'), [...e])
  // Every connection id is collision-free (contains :: and ->).
  check('expanded: ids well-formed', dyn.list_of_connections.every((c) => c.id.includes('::') && c.id.includes('->')))
}

// --------------------------------------------------------------------------
// Test 3: two expanded instances of the same tree don't collide.
// --------------------------------------------------------------------------
{
  const f = makeForest()
  const enc2: TreeNode = { id: 'enc2', name: 'Encoder', values: {} }
  addNode(f, 'main', enc2)
  const dyn = buildDynamicMainTree(f, new Set(['enc', 'enc2']))
  check('two instances: enc/c1 and enc2/c1 both present', dyn.list_of_nodes.includes('enc/c1') && dyn.list_of_nodes.includes('enc2/c1'), dyn.list_of_nodes)
  check('two instances: no duplicate ids', new Set(dyn.list_of_nodes).size === dyn.list_of_nodes.length)
}

// --------------------------------------------------------------------------
// Test 4: rename to a FREE name clones the tree; siblings unaffected.
// --------------------------------------------------------------------------
{
  const f = makeForest()
  // Add a second Encoder instance so we can prove the sibling is untouched.
  const enc2: TreeNode = { id: 'enc2', name: 'Encoder', values: {} }
  addNode(f, 'main', enc2)
  const before = getTree(f, 'Encoder')
  renameNode(f, 'enc', 'Decoder')
  check('rename-free: node enc now references Decoder', f.nodes['enc']!.name === 'Decoder')
  check('rename-free: Decoder tree created', Boolean(f.trees['Decoder']))
  check('rename-free: sibling enc2 still Encoder', f.nodes['enc2']!.name === 'Encoder')
  check('rename-free: Encoder tree still exists (enc2 uses it)', Boolean(f.trees['Encoder']))
  check('rename-free: Decoder has its own fresh child ids', getTree(f, 'Decoder').list_of_nodes.every((id) => !before.list_of_nodes.includes(id)), getTree(f, 'Decoder').list_of_nodes)
}

// --------------------------------------------------------------------------
// Test 5: rename to an EXISTING name shares the definition (no clone).
// --------------------------------------------------------------------------
{
  const f = makeForest()
  addTree(f, { name: 'Other', list_of_nodes: [], list_of_connections: [], inputs: [], outputs: [], params: {} })
  const blk: TreeNode = { id: 'ob', name: 'Conv', values: {} }
  addNode(f, 'Other', blk)
  const otherChildren = [...getTree(f, 'Other').list_of_nodes]
  renameNode(f, 'enc', 'Other')
  check('rename-existing: enc references Other', f.nodes['enc']!.name === 'Other')
  check('rename-existing: Other not cloned (same children)', JSON.stringify(getTree(f, 'Other').list_of_nodes) === JSON.stringify(otherChildren))
}

// --------------------------------------------------------------------------
// Test 6: rename orphans the old tree -> prune removes it + its child nodes.
// --------------------------------------------------------------------------
{
  const f = makeForest()
  const c1Id = getTree(f, 'Encoder').list_of_nodes[0]!
  // enc is the ONLY Encoder instance; renaming it orphans Encoder.
  renameNode(f, 'enc', 'Decoder') // prunes by default
  check('prune: orphaned Encoder removed', !f.trees['Encoder'])
  check('prune: Encoder child nodes deleted', !f.nodes[c1Id], c1Id)
  check('prune: main kept', Boolean(f.trees['main']))
  check('prune: Decoder kept (referenced by enc)', Boolean(f.trees['Decoder']))
}

// --------------------------------------------------------------------------
// Test 7: prune is transitive - a tree only reachable via a pruned tree dies.
// --------------------------------------------------------------------------
{
  const f = emptyForest()
  // Unused outer tree referencing an inner tree, neither used by main.
  addTree(f, { name: 'Outer', list_of_nodes: [], list_of_connections: [], inputs: [], outputs: [], params: {} })
  addTree(f, { name: 'Inner', list_of_nodes: [], list_of_connections: [], inputs: [], outputs: [], params: {} })
  addNode(f, 'Outer', { id: 'oi', name: 'Inner', values: {} })
  addNode(f, 'Inner', { id: 'ic', name: 'Conv', values: {} })
  const removed = pruneUnusedTrees(f)
  check('prune-transitive: both Outer and Inner removed', removed.includes('Outer') && removed.includes('Inner'), removed)
  check('prune-transitive: forest has only main', Object.keys(f.trees).length === 1 && Boolean(f.trees['main']), Object.keys(f.trees))
}

console.log(`\n${pass} passed, ${fail} failed`)
if (fail > 0) process.exit(1)
