// Pure-node test for subgraphUpTo(): truncating the graph to a node's ancestor
// closure so the shape runner can execute "up to the selected node only".
import { subgraphUpTo } from './src/runtime.ts'

let pass = 0, fail = 0
const check = (name: string, cond: boolean, info?: any) => {
  if (cond) { pass++; console.log('  ok ', name) }
  else { fail++; console.log('  FAIL', name, info ?? '') }
}

const node = (id: string, kind = 'module', groupId: string | null = null, gid?: string) => ({
  id, groupId,
  entry: { kind, name: id, groupId: gid, ctor: [], inputs: [], outputs: [], module: 'm' },
})
const edge = (source: string, target: string) => ({ id: `${source}->${target}`, source, sourceOutput: 'out', target, targetInput: 'x' })
const editorOf = (nodes: any[], connections: any[]): any => ({
  getNodes: () => nodes,
  getConnections: () => connections,
})

// --- Linear chain: Input -> a -> b -> c -> Output. Stop at b.
{
  const nodes = [node('inp', 'input'), node('a'), node('b'), node('c'), node('out', 'output')]
  const conns = [edge('inp', 'a'), edge('a', 'b'), edge('b', 'c'), edge('c', 'out')]
  const view = subgraphUpTo(editorOf(nodes, conns), 'b')
  const ids = new Set(view.nodes.map((n) => n.id))
  console.log('Linear chain, stop at b')
  check('keeps input', ids.has('inp'))
  check('keeps a (ancestor)', ids.has('a'))
  check('keeps b (target)', ids.has('b'))
  check('drops c (downstream)', !ids.has('c'))
  check('drops output (downstream)', !ids.has('out'))
  check('drops edge b->c', !view.connections.some((c) => c.target === 'c'))
}

// --- Diamond: inp -> a -> {b, c} -> d. Stop at b: c and d dropped, a+inp kept.
{
  const nodes = ['inp', 'a', 'b', 'c', 'd'].map((id) => node(id, id === 'inp' ? 'input' : 'module'))
  const conns = [edge('inp', 'a'), edge('a', 'b'), edge('a', 'c'), edge('b', 'd'), edge('c', 'd')]
  const view = subgraphUpTo(editorOf(nodes, conns), 'b')
  const ids = new Set(view.nodes.map((n) => n.id))
  console.log('Diamond, stop at b')
  check('keeps inp + a + b', ids.has('inp') && ids.has('a') && ids.has('b'))
  check('drops sibling c', !ids.has('c'))
  check('drops join d', !ids.has('d'))
}

// --- Multiple inputs: a second, unrelated Input is still kept (forward args).
{
  const nodes = [node('inp1', 'input'), node('inp2', 'input'), node('a'), node('b')]
  const conns = [edge('inp1', 'a'), edge('a', 'b')]
  const view = subgraphUpTo(editorOf(nodes, conns), 'a')
  const ids = new Set(view.nodes.map((n) => n.id))
  console.log('Two inputs, stop at a')
  check('keeps wired input inp1', ids.has('inp1'))
  check('keeps unrelated input inp2', ids.has('inp2'))
  check('drops downstream b', !ids.has('b'))
}

// --- Group: stop at a collapsed facade -> facade kept, ALL its members kept,
//     downstream dropped.
{
  const facade = node('fac', 'group', null, 'g1')
  const child1 = node('m1', 'module', 'g1')
  const child2 = node('m2', 'module', 'g1')
  const nodes = [node('inp', 'input'), facade, child1, child2, node('down')]
  const conns = [edge('inp', 'fac'), edge('fac', 'down')]
  const view = subgraphUpTo(editorOf(nodes, conns), 'fac')
  const ids = new Set(view.nodes.map((n) => n.id))
  console.log('Collapsed facade, stop at facade')
  check('keeps facade', ids.has('fac'))
  check('keeps group member m1 (needed by codegen)', ids.has('m1'))
  check('keeps group member m2', ids.has('m2'))
  check('drops downstream node', !ids.has('down'))
}

// --- Group member selected -> its facade is also kept so the group compiles.
{
  const facade = node('fac', 'group', null, 'g1')
  const child1 = node('m1', 'module', 'g1')
  const nodes = [node('inp', 'input'), facade, child1]
  const conns = [edge('inp', 'fac')]
  const view = subgraphUpTo(editorOf(nodes, conns), 'm1')
  const ids = new Set(view.nodes.map((n) => n.id))
  console.log('Grouped child selected')
  check('keeps the child', ids.has('m1'))
  check('keeps its facade', ids.has('fac'))
}

// --- Unknown id -> returns the whole graph unchanged.
{
  const nodes = [node('inp', 'input'), node('a')]
  const conns = [edge('inp', 'a')]
  const view = subgraphUpTo(editorOf(nodes, conns), 'nope')
  console.log('Unknown stop id')
  check('returns full node set', view.nodes.length === 2)
  check('returns full connection set', view.connections.length === 1)
}

console.log(`\n${pass} pass, ${fail} fail`)
if (fail > 0) process.exit(1)
