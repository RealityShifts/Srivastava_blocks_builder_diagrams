/**
 * Unit tests for the Step 0-2 tree model: model store, manifest shim, and the
 * keystone `getInputOutputParamsSignature`.
 *
 * Pure logic test - no puppeteer, no DOM. Run via:  node test-tree-signature.ts
 *
 * The headline check is PARITY: the new signature's `forwardInputs` must match
 * the CURRENT codegen's `planGraph(...).entryInputs` (unwired required input
 * ports -> forward args) on the same flat graph. This proves the new function
 * subsumes current behavior before anything is rewired (plan Step 2 gate).
 */

import { planGraph } from './src/codegen.ts'
import type { NodeLike, Connection as FlatConnection, ManifestEntry } from './src/types.ts'

import { emptyForest, addTree, addNode, makeConnection, getTree } from './src/tree/model.ts'
import type { Forest, TreeNode } from './src/tree/model.ts'
import { registerBlockTree } from './src/tree/manifest.ts'
import { getInputOutputParamsSignature } from './src/tree/signature.ts'
import type { BlockResolver, ResolvedBlock } from './src/tree/signature.ts'

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

// --------------------------------------------------------------------------
// A tiny block catalogue, expressed both as manifest entries (for the forest)
// and reused to build the flat NodeLike fixtures (for codegen parity).
// --------------------------------------------------------------------------

const M: Record<string, ManifestEntry> = {
  Conv: {
    name: 'Conv',
    module: 'm',
    kind: 'module',
    ctor: [
      { name: 'out_ch', type: 'int', default: 64, required: false },
      { name: 'k', type: 'int', default: 3, required: false },
    ],
    inputs: [{ name: 'x', shape: ['B', 'C', 'H', 'W'] }],
    outputs: [{ name: 'y', shape: ['B', 'C_out', 'H', 'W'] }],
  },
  Add: {
    name: 'Add',
    module: 'm',
    kind: 'module',
    ctor: [],
    inputs: [
      { name: 'a', shape: ['B', 'C'] },
      { name: 'b', shape: ['B', 'C'] },
    ],
    outputs: [{ name: 'out', shape: ['B', 'C'] }],
  },
  Drop: {
    name: 'Drop',
    module: 'm',
    kind: 'module',
    ctor: [{ name: 'p', type: 'float', default: 0.1 }],
    inputs: [{ name: 'x', shape: ['B', 'C'], optional: true }],
    outputs: [{ name: 'y', shape: ['B', 'C'] }],
  },
}

/** Resolver backed by the manifest catalogue: every child is a leaf block. */
function makeResolver(exposed: Record<string, Set<string>> = {}): BlockResolver {
  return (node: TreeNode): ResolvedBlock | null => {
    const m = M[node.name]
    if (!m) return null
    return {
      kind: m.kind,
      inputs: m.inputs,
      outputs: m.outputs,
      ctor: m.ctor.map((c) => ({ ...c })),
      exposedParams: exposed[node.id],
    }
  }
}

/** Build a flat NodeLike from a manifest entry + id (for codegen parity). */
function flatNode(id: string, entry: ManifestEntry): NodeLike {
  return { id, entry, values: {} }
}

// --------------------------------------------------------------------------
// Test 1: leaf block tree signature == the block's own ports.
// --------------------------------------------------------------------------
{
  const forest = emptyForest()
  registerBlockTree(forest, M.Conv)
  const sig = getInputOutputParamsSignature(getTree(forest, 'Conv'), forest, makeResolver())
  check('leaf: one forward input named x', sig.forwardInputs.length === 1 && sig.forwardInputs[0]!.name === 'x', sig.forwardInputs)
  check('leaf: one forward output named y', sig.forwardOutputs.length === 1 && sig.forwardOutputs[0]!.name === 'y', sig.forwardOutputs)
  check('leaf: no init params (none exposed)', sig.initParams.length === 0, sig.initParams)
}

// --------------------------------------------------------------------------
// Test 2: rename mechanism - tree named MyConv wrapping a Conv child.
// --------------------------------------------------------------------------
{
  const forest = emptyForest()
  const tree = registerBlockTree(forest, M.Conv, 'MyConv')
  check('rename: tree name is MyConv', tree.name === 'MyConv')
  check('rename: child references Conv', forest.nodes[tree.list_of_nodes[0]!]!.name === 'Conv')
  const sig = getInputOutputParamsSignature(tree, forest, makeResolver())
  check('rename: same ports as Conv', sig.forwardInputs.length === 1 && sig.forwardOutputs.length === 1)
}

// --------------------------------------------------------------------------
// Test 3: PARITY with codegen.planGraph().entryInputs on a flat graph.
//   Graph:  c1(Conv) -> c2(Conv)        [c2.x is wired; c1.x dangles]
//           plus a standalone Add whose b port dangles, a wired.
// --------------------------------------------------------------------------
{
  // --- forest version ---
  const forest = emptyForest()
  for (const e of Object.values(M)) registerBlockTree(forest, e)
  // Build a "main"-like composite tree whose children are block instances.
  const c1: TreeNode = { id: 'c1', name: 'Conv', values: {} }
  const c2: TreeNode = { id: 'c2', name: 'Conv', values: {} }
  const ad: TreeNode = { id: 'ad', name: 'Add', values: {} }
  const src: TreeNode = { id: 'src', name: 'Conv', values: {} }
  addTree(forest, { name: 'main', list_of_nodes: [], list_of_connections: [], inputs: [], outputs: [], params: {} })
  forest.mainTreeName = 'main'
  for (const n of [c1, c2, ad, src]) addNode(forest, 'main', n)
  const main = getTree(forest, 'main')
  main.list_of_connections = [
    makeConnection('c1', 'y', 'c2', 'x'), // c2.x wired
    makeConnection('src', 'y', 'ad', 'a'), // ad.a wired
  ]
  const sig = getInputOutputParamsSignature(main, forest, makeResolver())

  // --- flat codegen version ---
  const nodes: NodeLike[] = [flatNode('c1', M.Conv), flatNode('c2', M.Conv), flatNode('ad', M.Add), flatNode('src', M.Conv)]
  const conns: FlatConnection[] = [
    { id: 'e1', source: 'c1', sourceOutput: 'y', target: 'c2', targetInput: 'x' },
    { id: 'e2', source: 'src', sourceOutput: 'y', target: 'ad', targetInput: 'a' },
  ]
  const plan = planGraph(nodes, conns)!
  const entry = plan.entryInputs

  // Compare as sets of `${nodeId}/${portName}` - both must surface exactly:
  //   c1/x (Conv input dangling), ad/b (Add second input dangling).
  // (src/x and c2 is wired-on-x; src.x ALSO dangles -> appears in both.)
  const fromSig = new Set(sig.forwardInputs.map((p) => `${p.childId}/${p.childPort}`))
  const fromCg = new Set(entry.map((e: any) => `${e.nodeId}/${e.portName}`))
  const same = fromSig.size === fromCg.size && [...fromSig].every((k) => fromCg.has(k))
  check('PARITY: forwardInputs == codegen entryInputs', same, { fromSig: [...fromSig], fromCg: [...fromCg] })

  // The dangling output of c2 (y) and ad (out) and src? src.y is wired to ad.a,
  // so src.y is NOT a sink. Sinks: c2.y, ad.out.  c1.y wired to c2.x -> not sink.
  const sinks = new Set(sig.forwardOutputs.map((p) => `${p.childId}/${p.childPort}`))
  check('outputs: sinks are c2/y and ad/out', sinks.size === 2 && sinks.has('c2/y') && sinks.has('ad/out'), [...sinks])
}

// --------------------------------------------------------------------------
// Test 4: optional input port does NOT become a forward input.
// --------------------------------------------------------------------------
{
  const forest = emptyForest()
  registerBlockTree(forest, M.Drop)
  const d: TreeNode = { id: 'd', name: 'Drop', values: {} }
  addTree(forest, { name: 'main', list_of_nodes: [], list_of_connections: [], inputs: [], outputs: [], params: {} })
  forest.mainTreeName = 'main'
  addNode(forest, 'main', d)
  const sig = getInputOutputParamsSignature(getTree(forest, 'main'), forest, makeResolver())
  check('optional: Drop.x (optional) is not a forward input', sig.forwardInputs.length === 0, sig.forwardInputs)
}

// --------------------------------------------------------------------------
// Test 5: exposed ctor param becomes an init param.
// --------------------------------------------------------------------------
{
  const forest = emptyForest()
  registerBlockTree(forest, M.Conv)
  const c: TreeNode = { id: 'cc', name: 'Conv', values: { out_ch: 128 } }
  addTree(forest, { name: 'main', list_of_nodes: [], list_of_connections: [], inputs: [], outputs: [], params: {} })
  forest.mainTreeName = 'main'
  addNode(forest, 'main', c)
  const resolver = makeResolver({ cc: new Set(['out_ch']) })
  const sig = getInputOutputParamsSignature(getTree(forest, 'main'), forest, resolver)
  const p = sig.initParams.find((x) => x.name === 'out_ch')
  check('exposed: out_ch surfaces as init param', Boolean(p), sig.initParams)
  check('exposed: carries instance value 128', p?.default === 128, p)
}

// --------------------------------------------------------------------------
// Test 6: composite-in-composite recursion - a tree containing a group child.
// --------------------------------------------------------------------------
{
  const forest = emptyForest()
  for (const e of Object.values(M)) registerBlockTree(forest, e)
  // Inner tree "Encoder": single Conv child, both ports dangling.
  const inner: TreeNode = { id: 'in_c', name: 'Conv', values: {} }
  addTree(forest, { name: 'Encoder', list_of_nodes: [], list_of_connections: [], inputs: [], outputs: [], params: {} })
  addNode(forest, 'Encoder', inner)
  // Main contains one Encoder instance, unwired.
  const enc: TreeNode = { id: 'enc1', name: 'Encoder', values: {} }
  addTree(forest, { name: 'main', list_of_nodes: [], list_of_connections: [], inputs: [], outputs: [], params: {} })
  forest.mainTreeName = 'main'
  addNode(forest, 'main', enc)
  const sig = getInputOutputParamsSignature(getTree(forest, 'main'), forest, makeResolver())
  check('recursion: Encoder instance exposes 1 forward input', sig.forwardInputs.length === 1, sig.forwardInputs)
  check('recursion: Encoder instance exposes 1 forward output', sig.forwardOutputs.length === 1, sig.forwardOutputs)
  check('recursion: input provenance points at the Encoder child id', sig.forwardInputs[0]?.childId === 'enc1', sig.forwardInputs[0])
}

console.log(`\n${pass} passed, ${fail} failed`)
if (fail > 0) process.exit(1)
