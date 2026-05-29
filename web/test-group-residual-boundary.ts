// Regression: a collapsed group with a SELF-REFERENTIAL boundary edge - a
// child output wired into its own group's facade input, which the portMap then
// routes into a sibling child's VARIADIC port (an Elementwise `xs`). This is the
// residual-skip pattern produced by the self-attention groups: the first
// attention output feeds the group's `in_i`, and `in_i` folds into an Add along
// with the external skip.
//
// Before the fix, partitionByGroup dropped the child->own-facade edge (it isn't
// a same-group edge), and buildSubgraphView only minted a synthetic Input for
// that port - so the Add saw a single operand and codegen emitted `Add = in1`,
// silently losing the attention output. The fix rewrites the self-referential
// boundary edge back into the direct internal child->child edge the portMap
// implies, so both operands fold: `Add = (in0 + mha)`.
import { generate } from './src/codegen.ts'
import { ELEMENTWISE_ENTRY } from './src/nodes.ts'

let pass = 0, fail = 0
const check = (name: string, cond: boolean, info?: any) => {
  if (cond) { pass++; console.log('  ok ', name) }
  else { fail++; console.log('  FAIL', name, info ?? '') }
}

const tp = (n: string, variadic = false) => ({ name: n, shape: ['...'], dtype: 'any', optional: false, variadic })
const mk = (id: string, entry: any, values: any = {}, groupId: any = null) => ({
  id, entry, label: entry.name, tag: '', groupId, values,
  inputs: Object.fromEntries((entry.inputs || []).map((p: any) => [p.name, { portSpec: p }])),
  outputs: Object.fromEntries((entry.outputs || []).map((p: any) => [p.name, { portSpec: p }])),
  freshenedShape() { return null }, applyParamBindings() {},
})

const inEntry = { kind: 'input', name: 'Input', module: '__builtin__', ctor: [], inputs: [], outputs: [tp('out')], bindings: {} }
const outEntry = { kind: 'output', name: 'Output', module: '__builtin__', ctor: [], inputs: [tp('x')], outputs: [], bindings: {} }
// A minimal "attention-like" module with one input and one output.
const mhaEntry = { kind: 'module', name: 'MultiHeadAttention', module: 'attn', ctor: [], inputs: [tp('x')], outputs: [tp('out')], bindings: {} }

const GID = 'gRes'
// Children of the collapsed group: an MHA and an Add (variadic xs).
const mha = mk('mha', mhaEntry, {}, GID)
const add = mk('add', ELEMENTWISE_ENTRY, { op: 'add' }, GID)

// Facade has ONE boundary input `in0` mapped to add/xs. It is fed by BOTH an
// external source (the skip the parent passes) AND, self-referentially, the
// group's own mha output - exactly the self-attention residual layout. The
// synthetic input carries the external skip; the reconstructed internal edge
// carries the mha output, so the Add folds the two.
const facade = {
  id: 'facade',
  entry: {
    kind: 'group',
    groupId: GID,
    name: 'SelfAttn',
    module: '__group__',
    ctor: [],
    inputs: [tp('in0')],
    outputs: [tp('out0')],
    bindings: {},
    portMap: {
      inputs: [
        { facadePort: 'in0', childNodeId: 'add', childPort: 'xs', shape: ['...'] },
      ],
      outputs: [{ facadePort: 'out0', childNodeId: 'add', childPort: 'out', shape: ['...'] }],
      params: [],
    },
  },
  label: 'SelfAttn', tag: '', groupId: null, values: {},
  inputs: { in0: { portSpec: tp('in0') } },
  outputs: { out0: { portSpec: tp('out0') } },
  freshenedShape() { return null }, applyParamBindings() {},
}

// Top-level: a skip Input feeds the facade in0; another Input feeds the mha.
const skip = mk('skip', inEntry, { name: 'skip', shape: 'B 144 128', dtype: 'float' })
const src = mk('src', inEntry, { name: 'src', shape: 'B 144 128', dtype: 'float' })
const out = mk('out', outEntry, { name: 'y' })

const conns = [
  { id: 'c1', source: 'src', sourceOutput: 'out', target: 'mha', targetInput: 'x' },
  // self-referential boundary edge: mha (child of GID) -> facade in0
  { id: 'c2', source: 'mha', sourceOutput: 'out', target: 'facade', targetInput: 'in0' },
  // external skip into the same boundary port: skip -> facade in0
  { id: 'c3', source: 'skip', sourceOutput: 'out', target: 'facade', targetInput: 'in0' },
  // group output -> model output
  { id: 'c4', source: 'facade', sourceOutput: 'out0', target: 'out', targetInput: 'x' },
]

const code = generate([skip, src, mha, add, facade, out] as any, conns as any, 'pytorch')

console.log('Self-referential boundary residual fold')
// The subclass Add must fold BOTH operands: the synthetic boundary input `in0`
// (external skip) and the reconstructed internal mha output (a local var).
const foldsBoth = /=\s*\(in0\s*\+\s*\w+\)/.test(code) || /=\s*\(\w+\s*\+\s*in0\)/.test(code)
check('Add folds external skip AND attention output (no dropped operand)', foldsBoth, code)
check('does NOT emit single-operand Add = in0', !/elementwise_\w+\s*=\s*in0\s*$/m.test(code), code)

console.log(`\n${pass} pass, ${fail} fail`)
if (fail > 0) process.exitCode = 1
