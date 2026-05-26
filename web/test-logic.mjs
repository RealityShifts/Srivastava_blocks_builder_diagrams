// Quick smoke test of the pure-JS modules (no rete / DOM needed).
import { normalize, freshen, prettyShape } from './src/shape.js'
import { unifyShape, UnifyError } from './src/unify.js'
import { generate } from './src/codegen.js'

let pass = 0
let fail = 0
const check = (name, cond, info) => {
  if (cond) {
    pass++
    console.log(`  ok  ${name}`)
  } else {
    fail++
    console.log(`  FAIL ${name}`, info ?? '')
  }
}

// --- shape ---
console.log('shape.js')
const s1 = normalize(['B', '128', 'H', 'W'])
check('normalize coerces literals', s1[1] === 128 && s1[0] === 'B')
const s2 = freshen(['B', 'C_in', 'H', 'W'], 'n1')
check('freshen adds node tag', s2[0] === 'B#n1' && s2[1] === 'C_in#n1')

// --- unify ---
console.log('unify.js')
{
  const sub = new Map()
  unifyShape(['B#a', 'C#a', 'H#a', 'W#a'], ['B#b', 128, 'H#b', 'W#b'], sub)
  // resolve C#a -> 128 via aliases
  // The simplest invariant: walking C#a eventually yields 128
  let cur = 'C#a'
  for (let i = 0; i < 10 && sub.has(cur); i++) cur = sub.get(cur)
  check('unify binds variable to literal across alias', cur === 128, [...sub])
}
{
  const sub = new Map()
  let thrown = false
  try {
    unifyShape(['B', 128], ['B', 64], sub)
  } catch (e) {
    thrown = e instanceof UnifyError
  }
  check('unify rejects literal mismatch', thrown)
}
{
  const sub = new Map()
  unifyShape(['B', '...', 'D'], ['B', 'X', 'Y', 'D'], sub)
  check('unify with rest absorbs middle axes', true)
}
{
  const sub = new Map()
  let thrown = false
  try {
    unifyShape(['B', 'C', 'H', 'W'], ['B', 'D'], sub)
  } catch (e) {
    thrown = true
  }
  check('unify rejects rank mismatch (no rest)', thrown)
}

// --- pretty ---
console.log('pretty')
{
  const sub = new Map()
  unifyShape(['B#a', 'C_in#a', 'H#a', 'W#a'], ['B#b', 3, 'H#b', 'W#b'], sub)
  const pretty = prettyShape(['B#a', 'C_in#a', 'H#a', 'W#a'], sub)
  check('prettyShape resolves a literal binding', pretty.includes('3'), pretty)
}

// --- codegen ---
console.log('codegen.js')
{
  const makeNode = (id, entry, values = {}) => ({
    id,
    entry,
    values: { ...Object.fromEntries(entry.ctor.map((p) => [p.name, p.default])), ...values },
  })
  const conv = {
    name: 'ConvBlock',
    module: 'pytorch_blocks.core_blocks',
    framework: 'pytorch',
    kind: 'module',
    ctor: [
      { name: 'in_ch', type: 'int', default: null, required: true },
      { name: 'out_ch', type: 'int', default: null, required: true },
    ],
    inputs: [{ name: 'x', shape: ['B', 'C_in', 'H', 'W'], dtype: 'float', optional: false, variadic: false }],
    outputs: [{ name: 'out', shape: ['B', 'C_out', 'H_out', 'W_out'], dtype: 'float', optional: false, variadic: false }],
    bindings: { C_in: 'in_ch', C_out: 'out_ch' },
  }
  const n1 = makeNode('n1', conv, { in_ch: 3, out_ch: 16 })
  const n2 = makeNode('n2', conv, { in_ch: 16, out_ch: 32 })
  const conns = [{ source: 'n1', sourceOutput: 'out', target: 'n2', targetInput: 'x' }]
  const code = generate([n1, n2], conns, 'pytorch')
  console.log('--- generated code ---')
  console.log(code)
  console.log('--- end ---')
  check('codegen produces forward()', code.includes('def forward'))
  check('codegen wires n1 -> n2', code.includes('self.conv_block_1(x=conv_block_0)'))
  check('codegen imports ConvBlock', code.includes('from pytorch_blocks.core_blocks import ConvBlock'))
}

console.log(`\n${pass} pass, ${fail} fail`)
process.exit(fail === 0 ? 0 : 1)
