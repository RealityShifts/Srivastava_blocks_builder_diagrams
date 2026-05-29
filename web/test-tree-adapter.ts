/**
 * Tests for Step 4 (adapter) and Step 5 (codegen-from-tree, flat case).
 * Pure logic - run via:  node test-tree-adapter.ts
 *
 * Headline check: `generate()` on the forest-derived flat codegen input is
 * BYTE-IDENTICAL to `generate()` on the original flat nodes. That is the Step 5
 * parity gate.
 */

import { generate } from './src/codegen.ts'
import type { NodeLike, Connection as FlatConn, ManifestEntry } from './src/types.ts'
import {
  graphDataToForest,
  forestToFlatCodegen,
  forestToGenerateInput,
  forestToGraphData,
  makeManifestResolver,
} from './src/tree/adapter.ts'
import type { GraphData, ManifestCatalogue } from './src/tree/adapter.ts'
import { getInputOutputParamsSignature } from './src/tree/signature.ts'
import { getTree } from './src/tree/model.ts'

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

// A small pytorch-flavored manifest catalogue.
const CAT: ManifestCatalogue = {
  Input: { name: 'Input', module: '__builtin__', kind: 'input', ctor: [{ name: 'name', type: 'str', default: 'x' }], inputs: [], outputs: [{ name: 'y', shape: ['B', 'C', 'H', 'W'] }] },
  Output: { name: 'Output', module: '__builtin__', kind: 'output', ctor: [{ name: 'name', type: 'str', default: 'out' }], inputs: [{ name: 'x', shape: ['...'] }], outputs: [] },
  Conv: { name: 'Conv', module: 'torch.nn', kind: 'module', ctor: [{ name: 'out_ch', type: 'int', default: 64 }], inputs: [{ name: 'x', shape: ['B', 'C', 'H', 'W'] }], outputs: [{ name: 'y', shape: ['B', 'C_out', 'H', 'W'] }] },
}

/** Flat fixture: Input -> Conv -> Conv -> Output. */
function flatFixture(): { nodes: NodeLike[]; conns: FlatConn[]; data: GraphData } {
  const mk = (id: string, name: string): NodeLike => ({ id, entry: CAT[name]!, name: '', tag: '', groupId: null, values: Object.fromEntries(CAT[name]!.ctor.map((c) => [c.name, c.default])) })
  const nodes = [mk('i', 'Input'), mk('c1', 'Conv'), mk('c2', 'Conv'), mk('o', 'Output')]
  const conns: FlatConn[] = [
    { id: 'e1', source: 'i', sourceOutput: 'y', target: 'c1', targetInput: 'x' },
    { id: 'e2', source: 'c1', sourceOutput: 'y', target: 'c2', targetInput: 'x' },
    { id: 'e3', source: 'c2', sourceOutput: 'y', target: 'o', targetInput: 'x' },
  ]
  const data: GraphData = {
    version: 1,
    framework: 'pytorch',
    nodes: nodes.map((n) => ({ id: n.id, name: n.entry.name, kind: n.entry.kind, tag: '', values: { ...n.values }, groupId: null })),
    connections: conns.map((c) => ({ source: c.source, sourceOutput: c.sourceOutput, target: c.target, targetInput: c.targetInput })),
    groups: [],
  }
  return { nodes, conns, data }
}

// --------------------------------------------------------------------------
// Test 1: graphDataToForest places a flat graph into `main`.
// --------------------------------------------------------------------------
{
  const { data } = flatFixture()
  const { forest } = graphDataToForest(data)
  const main = getTree(forest, 'main')
  check('adapter: 4 nodes in main', main.list_of_nodes.length === 4, main.list_of_nodes)
  check('adapter: 3 connections in main', main.list_of_connections.length === 3, main.list_of_connections.length)
  check('adapter: ids preserved', main.list_of_nodes.includes('c1') && main.list_of_nodes.includes('o'))
}

// --------------------------------------------------------------------------
// Test 2: signature through the manifest resolver (end-to-end).
//   c1.x is wired from Input; Conv has no other inputs; so no dangling tensor
//   forward inputs among module nodes. Output sink is the Output node (skipped).
// --------------------------------------------------------------------------
{
  const { data } = flatFixture()
  const { forest, exposedByNode } = graphDataToForest(data)
  const resolver = makeManifestResolver(CAT, exposedByNode)
  const sig = getInputOutputParamsSignature(getTree(forest, 'main'), forest, resolver)
  // No module input ports dangle (c1.x wired; c2.x wired). Input/Output kinds skipped.
  check('signature: no dangling module forward inputs', sig.forwardInputs.length === 0, sig.forwardInputs)
  // c2.y is wired to Output, so it is NOT a module-sink; nothing dangles.
  check('signature: no dangling outputs (c2.y wired to Output)', sig.forwardOutputs.length === 0, sig.forwardOutputs)
}

// --------------------------------------------------------------------------
// Test 3: PARITY - generate() on forest-derived input == legacy generate().
// --------------------------------------------------------------------------
{
  const { nodes, conns, data } = flatFixture()
  const legacy = generate(nodes, conns, 'pytorch')

  const { forest } = graphDataToForest(data)
  const { nodes: fNodes, connections: fConns, hasGroups } = forestToFlatCodegen(forest, CAT)
  check('codegen: flat fixture has no groups', hasGroups === false)
  const viaForest = generate(fNodes, fConns, 'pytorch')

  check('PARITY: forest codegen == legacy codegen', viaForest === legacy, { legacy, viaForest })
}

// --------------------------------------------------------------------------
// Test 4: GraphData round-trips through the forest (structure preserved).
// --------------------------------------------------------------------------
{
  const { data } = flatFixture()
  const { forest, exposedByNode } = graphDataToForest(data)
  const back = forestToGraphData(forest, CAT, exposedByNode)
  check('round-trip: 4 nodes', back.nodes.length === 4, back.nodes.length)
  check('round-trip: 3 connections', back.connections.length === 3, back.connections.length)
  const names = back.nodes.map((n) => n.name).sort()
  check('round-trip: names preserved', JSON.stringify(names) === JSON.stringify(['Conv', 'Conv', 'Input', 'Output']), names)
}

// --------------------------------------------------------------------------
// Test 5: a grouped GraphData reconstructs the group as a Tree.
// --------------------------------------------------------------------------
{
  const data: GraphData = {
    version: 1,
    framework: 'pytorch',
    nodes: [
      { id: 'i', name: 'Input', kind: 'input', values: {} },
      { id: 'gc1', name: 'Conv', kind: 'module', values: {}, groupId: 'g1' },
      { id: 'gc2', name: 'Conv', kind: 'module', values: {}, groupId: 'g1' },
      { id: 'o', name: 'Output', kind: 'output', values: {} },
      { id: 'fac', name: 'Encoder', kind: 'group', portMap: { inputs: [{ childNodeId: 'gc1', childPort: 'x' }], outputs: [{ childNodeId: 'gc2', childPort: 'y' }] } },
    ],
    connections: [
      { source: 'i', sourceOutput: 'y', target: 'fac', targetInput: 'x' },
      { source: 'gc1', sourceOutput: 'y', target: 'gc2', targetInput: 'x' },
      { source: 'fac', sourceOutput: 'y', target: 'o', targetInput: 'x' },
    ],
    groups: [{ id: 'g1', name: 'Encoder', collapsed: true, facadeNodeId: 'fac' }],
  }
  const { forest } = graphDataToForest(data)
  check('group: Encoder tree created', Boolean(forest.trees['Encoder']))
  const enc = getTree(forest, 'Encoder')
  check('group: Encoder has 2 children', enc.list_of_nodes.length === 2, enc.list_of_nodes)
  check('group: boundary inputs gc1@x', JSON.stringify(enc.inputs) === JSON.stringify(['gc1@x']), enc.inputs)
  check('group: boundary outputs gc2@y', JSON.stringify(enc.outputs) === JSON.stringify(['gc2@y']), enc.outputs)
  check('group: internal edge gc1->gc2 in Encoder', enc.list_of_connections.some((c) => c.from === 'gc1' && c.to === 'gc2'))
  // The facade node lives in main, referencing the Encoder tree.
  const main = getTree(forest, 'main')
  check('group: facade node in main references Encoder', main.list_of_nodes.includes('fac') && forest.nodes['fac']!.name === 'Encoder')
}

console.log(`\n${pass} passed, ${fail} failed`)
if (fail > 0) process.exit(1)
