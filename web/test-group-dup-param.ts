// Regression test: a group facade that exposes the SAME ctor param from more
// than one child must not emit a repeated keyword argument when instantiated.
// Repro of "keyword argument repeated: style_dim" seen when collapsing a group
// whose children share an exposed param wired to a single Constant.
import { generate } from './src/codegen.ts'

const tensorPort = (name: string, optional = false) => ({
  name, shape: ['...'], dtype: 'any', optional, variadic: false,
})

// Two AdaIN-like children, each exposing `style_dim`, both fed by one Constant.
const adainEntry = {
  kind: 'module', name: 'AdaIN', module: 'm',
  ctor: [
    { name: 'num_features', type: 'int', default: 0 },
    { name: 'style_dim', type: 'int', default: 0 },
  ],
  inputs: [tensorPort('x'), { name: 'style', shape: ['...'], dtype: 'any', optional: false, variadic: false }],
  outputs: [tensorPort('out')],
  bindings: {},
}

const mkNode = (id: string, entry: any, values: any, groupId: string | null) => ({
  id, entry, label: entry.name, tag: '', groupId, values,
  inputs: Object.fromEntries((entry.inputs || []).map((p: any) => {
    // param ports use a synthetic spec; tensor ports carry the port itself
    return [p.name, { portSpec: p }]
  })),
  outputs: Object.fromEntries((entry.outputs || []).map((p: any) => [p.name, { portSpec: p }])),
  freshenedShape() { return null },
  applyParamBindings() {},
})

// Attach __param__style_dim input ports (kind: 'param') so const wiring resolves.
const withParamPort = (node: any, paramName: string) => {
  node.inputs[`__param__${paramName}`] = { portSpec: { kind: 'param', paramName, name: `__param__${paramName}` } }
  return node
}

const ad1 = withParamPort(mkNode('ad1', adainEntry, { num_features: 64, style_dim: 256 }, 'g_dec'), 'style_dim')
const ad2 = withParamPort(mkNode('ad2', adainEntry, { num_features: 32, style_dim: 256 }, 'g_dec'), 'style_dim')

const constEntry = { kind: 'const', name: 'Constant', module: 'm', ctor: [], inputs: [], outputs: [tensorPort('out')], bindings: {} }
const styleConst = mkNode('sc', constEntry, { value_type: 'int', value: '256' }, null)

// Group facade exposing style_dim TWICE (one portMap entry per child) — the
// exact shape the editor persists for a shared exposed param.
const facadeEntry = {
  kind: 'group', name: 'Decoder', groupId: 'g_dec', module: 'm', ctor: [],
  inputs: [tensorPort('in0')], outputs: [tensorPort('out0')],
  bindings: {},
  portMap: {
    inputs: [{ facadePort: 'in0', childNodeId: 'ad1', childPort: 'x', shape: ['...'] }],
    outputs: [{ facadePort: 'out0', childNodeId: 'ad2', childPort: 'out', shape: ['...'] }],
    params: [
      { facadePort: '__param__style_dim', childNodeId: 'ad1', childPort: '__param__style_dim', paramName: 'style_dim', paramType: 'int' },
      { facadePort: '__param__style_dim', childNodeId: 'ad2', childPort: '__param__style_dim', paramName: 'style_dim', paramType: 'int' },
    ],
  },
}
// Expose the param port on the facade itself (the editor adds these on
// collapse) so the top-level const can wire straight into the facade.
const facade = withParamPort(mkNode('fac', facadeEntry, {}, null), 'style_dim')

const input = mkNode('inp', { kind: 'input', name: 'Input', module: '__builtin__', ctor: [], inputs: [], outputs: [tensorPort('out')], bindings: {} }, { name: 'x', shape: '1 3 8 8', dtype: 'float' }, null)

const nodes = [input, facade, ad1, ad2, styleConst]
const connections = [
  { source: 'inp', sourceOutput: 'out', target: 'fac', targetInput: 'in0' },
  { source: 'ad1', sourceOutput: 'out', target: 'ad2', targetInput: 'x' },
  // single const feeding the facade's exposed style_dim param. The editor
  // persists one such edge per underlying child, so the const lands on the
  // facade twice — mirror that here.
  { source: 'sc', sourceOutput: 'out', target: 'fac', targetInput: '__param__style_dim' },
  { source: 'sc', sourceOutput: 'out', target: 'fac', targetInput: '__param__style_dim' },
]

const code = generate(nodes as any, connections as any, 'pytorch')

// Assert no line repeats a keyword argument.
let failed = false
for (const line of code.split('\n')) {
  const m = /=\s*\w+/g
  const kwargs = [...line.matchAll(/(\b\w+)=/g)].map((x) => x[1])
  const dup = kwargs.find((k, i) => kwargs.indexOf(k) !== i)
  if (dup && line.includes('Decoder(')) {
    console.error('FAIL: repeated kwarg on line:', line.trim())
    failed = true
  }
}

const decoderInstLine = code.split('\n').find((l) => l.includes('= Decoder('))
console.log('Decoder instantiation:', decoderInstLine?.trim())

if (failed) {
  console.error('\n--- generated code ---\n' + code)
  process.exit(1)
}
console.log('PASS: no repeated keyword argument in group instantiation')
