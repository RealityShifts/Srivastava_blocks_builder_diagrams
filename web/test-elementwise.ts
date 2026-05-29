// Codegen tests for the Elementwise (add / multiply) utility node.
import { generate } from './src/codegen.ts'
import { ELEMENTWISE_ENTRY } from './src/nodes.ts'

let pass = 0, fail = 0
const check = (name: string, cond: boolean, info?: any) => {
  if (cond) { pass++; console.log('  ok ', name) }
  else { fail++; console.log('  FAIL', name, info ?? '') }
}

const tp = (n: string, variadic = false) => ({ name: n, shape: ['...'], dtype: 'any', optional: false, variadic })
const mk = (id: string, entry: any, values: any = {}) => ({
  id, entry, label: entry.name, tag: '', groupId: null, values,
  inputs: Object.fromEntries((entry.inputs || []).map((p: any) => [p.name, { portSpec: p }])),
  outputs: Object.fromEntries((entry.outputs || []).map((p: any) => [p.name, { portSpec: p }])),
  freshenedShape() { return null }, applyParamBindings() {},
})
const inEntry = { kind: 'input', name: 'Input', module: '__builtin__', ctor: [], inputs: [], outputs: [tp('out')], bindings: {} }
const outEntry = { kind: 'output', name: 'Output', module: '__builtin__', ctor: [], inputs: [tp('x')], outputs: [], bindings: {} }

const build = (op: string, framework: string) => {
  const x = mk('x', inEntry, { name: 'x', shape: '1 8', dtype: 'float' })
  const z = mk('z', inEntry, { name: 'z', shape: '1 8', dtype: 'float' })
  const e = mk('e', ELEMENTWISE_ENTRY, { op })
  const out = mk('out', outEntry, { name: 'y' })
  const conns = [
    { id: '1', source: 'x', sourceOutput: 'out', target: 'e', targetInput: 'xs' },
    { id: '2', source: 'z', sourceOutput: 'out', target: 'e', targetInput: 'xs' },
    { id: '3', source: 'e', sourceOutput: 'out', target: 'out', targetInput: 'x' },
  ]
  return generate([x, z, e, out] as any, conns as any, framework)
}

console.log('Elementwise entry')
check('is a built-in utility', ELEMENTWISE_ENTRY.module === '__utility__' && ELEMENTWISE_ENTRY.kind === 'eltwise')
check('has variadic xs input', ELEMENTWISE_ENTRY.inputs[0].variadic === true)
check('op choices are add/multiply', JSON.stringify(ELEMENTWISE_ENTRY.ctor[0].choices) === JSON.stringify(['add', 'multiply']))

console.log('Add (pytorch)')
{
  const code = build('add', 'pytorch')
  check('folds operands with +', /elementwise_\w+ = \(x \+ z\)/.test(code), code)
}
console.log('Multiply (pytorch)')
{
  const code = build('multiply', 'pytorch')
  check('folds operands with *', /elementwise_\w+ = \(x \* z\)/.test(code), code)
}
console.log('Add (flax) is identical infix')
{
  const code = build('add', 'flax')
  check('flax also uses + infix', /elementwise_\w+ = \(x \+ z\)/.test(code), code)
}

console.log('Three-way fold')
{
  const a = mk('a', inEntry, { name: 'a', shape: '1 8', dtype: 'float' })
  const b = mk('b', inEntry, { name: 'b', shape: '1 8', dtype: 'float' })
  const c = mk('c', inEntry, { name: 'c', shape: '1 8', dtype: 'float' })
  const e = mk('e', ELEMENTWISE_ENTRY, { op: 'add' })
  const out = mk('out', outEntry, { name: 'y' })
  const conns = [
    { id: '1', source: 'a', sourceOutput: 'out', target: 'e', targetInput: 'xs' },
    { id: '2', source: 'b', sourceOutput: 'out', target: 'e', targetInput: 'xs' },
    { id: '3', source: 'c', sourceOutput: 'out', target: 'e', targetInput: 'xs' },
    { id: '4', source: 'e', sourceOutput: 'out', target: 'out', targetInput: 'x' },
  ]
  const code = generate([a, b, c, e, out] as any, conns as any, 'pytorch')
  check('folds three operands', /elementwise_\w+ = \(a \+ b \+ c\)/.test(code), code)
}

console.log(`\n${pass} pass, ${fail} fail`)
if (fail > 0) process.exit(1)
