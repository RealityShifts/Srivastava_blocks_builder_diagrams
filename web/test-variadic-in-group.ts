// Regression: a variadic Elementwise/Concat/Stack port fed THROUGH a collapsed
// group facade boundary must count its fan-in. Previously the validator only
// counted edges landing directly on the child port, so an Elementwise inside a
// group whose `xs` is wired via the facade's `in3` reported "needs >=2, got 0".
//   node test-variadic-in-group.ts
import { validate } from './src/validator.ts'

let pass = 0, fail = 0
const check = (name: string, cond: boolean, info?: unknown) => {
  if (cond) { pass++; console.log(`  ok   ${name}`) }
  else { fail++; console.log(`  FAIL ${name}`, info ?? '') }
}

// Minimal node objects satisfying the validator's structural needs.
const port = (name: string, variadic = false) => ({ name, shape: ['...'], dtype: 'any', variadic, optional: false })
const mkNode = (id: string, kind: string, inputs: any[], outputs: any[], extra: any = {}) => ({
  id,
  entry: { name: extra.name ?? id, kind, ctor: [], inputs, outputs, ...(extra.portMap ? { kind: 'group', portMap: extra.portMap } : {}) },
  name: '', tag: extra.tag ?? '', groupId: extra.groupId ?? null, values: extra.values ?? {},
  inputs: Object.fromEntries(inputs.map((p: any) => [p.name, {}])),
  outputs: Object.fromEntries(outputs.map((p: any) => [p.name, {}])),
  freshenedShape: () => ['...'],
  applyParamBindings: () => {},
})

// MHA (3 inputs + mask), Elementwise (variadic xs), and a group facade whose
// in3 routes to the Elementwise xs, fed by TWO external edges.
const mha = mkNode('mha', 'module', [port('query'), port('key'), port('value'), { ...port('mask'), optional: true }], [port('out')])
const elt = mkNode('elt', 'eltwise', [port('xs', true)], [port('out')], { tag: 'ky3au' })
const facade = mkNode('fac', 'group', [port('in0'), port('in1'), port('in2'), port('in3'), { ...port('in4'), optional: true }], [port('out0')], {
  name: 'Group1',
  portMap: {
    inputs: [
      { facadePort: 'in0', childNodeId: 'mha', childPort: 'query' },
      { facadePort: 'in1', childNodeId: 'mha', childPort: 'key' },
      { facadePort: 'in2', childNodeId: 'mha', childPort: 'value' },
      { facadePort: 'in3', childNodeId: 'elt', childPort: 'xs' },
      { facadePort: 'in4', childNodeId: 'mha', childPort: 'mask' },
    ],
    outputs: [{ facadePort: 'out0', childNodeId: 'elt', childPort: 'out' }],
    params: [],
  },
})
const input = mkNode('inp', 'input', [], [port('out')], { values: { name: 'x' } })
elt.groupId = 'g1'; mha.groupId = 'g1'

const connections = [
  { id: 'c0', source: 'inp', sourceOutput: 'out', target: 'fac', targetInput: 'in0' },
  { id: 'c1', source: 'inp', sourceOutput: 'out', target: 'fac', targetInput: 'in1' },
  { id: 'c2', source: 'inp', sourceOutput: 'out', target: 'fac', targetInput: 'in2' },
  // Two edges into facade in3 -> routes to elt/xs (the variadic fan-in):
  { id: 'c3', source: 'mha', sourceOutput: 'out', target: 'fac', targetInput: 'in3' },
  { id: 'c4', source: 'inp', sourceOutput: 'out', target: 'fac', targetInput: 'in3' },
]
const nodes = [input, mha, elt, facade]
const editor: any = {
  getNodes: () => nodes,
  getConnections: () => connections,
  getNode: (id: string) => nodes.find((n) => n.id === id) ?? null,
}

const res = validate(editor)
const variadicWarn = res.warnings.filter((w: any) => w.kind === 'variadic-min')
check('no variadic-min warning when xs fed via facade in3 (2 edges)', variadicWarn.length === 0, variadicWarn.map((w: any) => w.message))

// Negative control: only ONE edge into in3 -> should still warn.
const oneEdge = connections.filter((c) => !(c.id === 'c4'))
const editor1: any = { getNodes: () => nodes, getConnections: () => oneEdge, getNode: (id: string) => nodes.find((n) => n.id === id) ?? null }
const res1 = validate(editor1)
const warn1 = res1.warnings.filter((w: any) => w.kind === 'variadic-min')
check('still warns with only 1 edge into the variadic port', warn1.length === 1, warn1.map((w: any) => w.message))

console.log(`\n${pass} passed, ${fail} failed`)
if (fail > 0) process.exit(1)
