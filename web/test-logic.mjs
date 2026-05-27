// Quick smoke test of the pure-JS modules (no rete / DOM needed).
import { normalize, freshen, prettyShape } from './src/shape.js'
import { unifyShape, UnifyError } from './src/unify.js'
import { generate } from './src/codegen.js'
import {
  parseEinopsPattern,
  parseLengthsString,
  REARRANGE_ENTRY,
  RESHAPE_ENTRY,
  CONCAT_ENTRY,
  STACK_ENTRY,
  RearrangeNode,
  ReshapeNode,
  paletteGroup,
} from './src/nodes.js'

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

// --- einops parser ---
console.log('einops parser')
{
  const { lhs, rhs } = parseEinopsPattern('b c h w -> b (h w) c')
  check('lhs split into 4 atoms', lhs.length === 4 && lhs[0] === 'b', lhs)
  check('rhs preserves "(h w)" group', rhs.length === 3 && rhs[1] === '(h w)', rhs)
}
{
  const { lhs, rhs } = parseEinopsPattern('  b  (h s1) (w s2)   c  ->  b   (h w)   (c s1 s2) ')
  check('lhs handles loose whitespace', lhs.join('|') === 'b|(h s1)|(w s2)|c', lhs)
  check('rhs handles loose whitespace', rhs.join('|') === 'b|(h w)|(c s1 s2)', rhs)
}
{
  let thrown = false
  try { parseEinopsPattern('b c h w') } catch { thrown = true }
  check('parser rejects pattern without "->"', thrown)
}
{
  const lens = parseLengthsString('h=8, w=16; s1=2')
  check(
    'lengths parser handles commas and semicolons',
    lens.get('h') === 8 && lens.get('w') === 16 && lens.get('s1') === 2,
    [...lens]
  )
}

// --- Rearrange shape inference ---
console.log('RearrangeNode')
{
  // Construct manually since rete's NodeEditor isn't available here.
  const node = Object.assign(Object.create(RearrangeNode.prototype), {
    id: 'r1',
    entry: REARRANGE_ENTRY,
    values: { pattern: 'b c h w -> b (h w) c', lengths: '' },
  })
  const inS = node.freshenedShape('x', 'in')
  const outS = node.freshenedShape('out', 'out')
  check('rearrange input has rank 4', inS.length === 4, inS)
  check('rearrange output has rank 3', outS.length === 3, outS)
  check(
    'b/c axes share names between in/out (binding survives unification)',
    inS[0].split('#')[0] === outS[0].split('#')[0] &&
      inS[1].split('#')[0] === outS[2].split('#')[0],
    { inS, outS }
  )
}
{
  // With kwargs the (h w) group can fold to a numeric literal.
  const node = Object.assign(Object.create(RearrangeNode.prototype), {
    id: 'r2',
    entry: REARRANGE_ENTRY,
    values: { pattern: 'b c h w -> b (h w) c', lengths: 'h=8, w=16' },
  })
  const outS = node.freshenedShape('out', 'out')
  check('rearrange folds (h w) with lengths -> 128 literal', outS[1] === 128, outS)
}

// --- Reshape shape inference ---
console.log('ReshapeNode')
{
  const node = Object.assign(Object.create(ReshapeNode.prototype), {
    id: 'rs1',
    entry: RESHAPE_ENTRY,
    values: { shape: '-1 3 224 224' },
  })
  const outS = node.freshenedShape('out', 'out')
  check(
    'reshape output is [-1, 3, 224, 224]',
    outS.length === 4 && outS[0] === -1 && outS[2] === 224,
    outS
  )
}

// --- Rearrange / Reshape codegen ---
console.log('codegen (rearrange + reshape)')
{
  const inputEntry = {
    name: 'Input',
    module: '__builtin__',
    framework: 'any',
    kind: 'input',
    ctor: [
      { name: 'name', type: 'str', default: 'x' },
      { name: 'shape', type: 'str', default: 'B C H W' },
      { name: 'dtype', type: 'str', default: 'float' },
    ],
    inputs: [],
    outputs: [{ name: 'out', shape: ['B', 'C', 'H', 'W'], dtype: 'float' }],
    bindings: {},
  }
  const make = (id, entry, values = {}) => ({
    id,
    entry,
    values: {
      ...Object.fromEntries(entry.ctor.map((p) => [p.name, p.default])),
      ...values,
    },
  })
  const n_in = make('in', inputEntry, { name: 'x' })
  const n_re = make('re', REARRANGE_ENTRY, {
    pattern: 'b c h w -> b (h w) c',
    lengths: 'h=8, w=16',
  })
  const n_rs = make('rs', RESHAPE_ENTRY, { shape: '-1 128' })
  const conns = [
    { source: 'in', sourceOutput: 'out', target: 're', targetInput: 'x' },
    { source: 're', sourceOutput: 'out', target: 'rs', targetInput: 'x' },
  ]
  const code = generate([n_in, n_re, n_rs], conns, 'pytorch')
  check('einops import emitted', code.includes('from einops import rearrange'), code)
  check(
    'rearrange call uses pattern + length kwargs',
    code.includes('rearrange(x, "b c h w -> b (h w) c", h=8, w=16)'),
    code
  )
  check(
    'reshape call uses tensor.reshape(-1, 128)',
    /\w+\.reshape\(-1, 128\)/.test(code),
    code
  )
  check('reshape feeds from previous rearrange output', code.includes('.reshape'), code)
  check('builtin nodes do NOT appear in import block', !code.includes('__builtin__'), code)

  const codeFlax = generate([n_in, n_re, n_rs], conns, 'flax')
  check(
    'flax reshape uses jnp.reshape(x, (-1, 128))',
    codeFlax.includes('jnp.reshape(') && codeFlax.includes(', (-1, 128))'),
    codeFlax
  )
}

// --- tag-based weight sharing in codegen ---
console.log('codegen tag weight-sharing')
{
  const conv = {
    name: 'ConvBlock',
    module: 'pytorch_blocks.core_blocks',
    framework: 'pytorch',
    kind: 'module',
    ctor: [
      { name: 'in_ch', type: 'int', default: null, required: true },
      { name: 'out_ch', type: 'int', default: null, required: true },
    ],
    inputs: [{ name: 'x', shape: ['B', 'C_in', 'H', 'W'], dtype: 'float' }],
    outputs: [{ name: 'out', shape: ['B', 'C_out', 'H', 'W'], dtype: 'float' }],
    bindings: { C_in: 'in_ch', C_out: 'out_ch' },
  }
  const make = (id, entry, values = {}, tag = '') => ({
    id,
    entry,
    tag,
    values: { ...Object.fromEntries(entry.ctor.map((p) => [p.name, p.default])), ...values },
  })
  // Two ConvBlocks sharing tag "shared" with the SAME ctor values -> one __init__ slot.
  const a = make('a', conv, { in_ch: 3, out_ch: 16 }, 'shared')
  const b = make('b', conv, { in_ch: 3, out_ch: 16 }, 'shared')
  const code = generate([a, b], [], 'pytorch')

  const initInstances = (code.match(/self\.shared = ConvBlock\(/g) || []).length
  check('shared tag emits ONE self.shared = ConvBlock(...) slot', initInstances === 1, initInstances)

  // Both nodes call self.shared in forward, with distinct local var names so
  // the first call's output isn't clobbered.
  const callSites = (code.match(/self\.shared\(/g) || []).length
  check('forward has two call sites to self.shared', callSites === 2, callSites)
  check(
    'distinct local vars per call site (no var name collision)',
    /\bshared\s*=\s*self\.shared\(/.test(code) && /\bshared2\s*=\s*self\.shared\(/.test(code),
    code
  )

  // Three-way share also collapses to one slot.
  const c = make('c', conv, { in_ch: 3, out_ch: 16 }, 'shared')
  const code3 = generate([a, b, c], [], 'pytorch')
  const slots3 = (code3.match(/self\.shared = ConvBlock\(/g) || []).length
  check('3-way shared tag still emits ONE slot', slots3 === 1, slots3)
  const sites3 = (code3.match(/self\.shared\(/g) || []).length
  check('3-way shared tag produces three call sites', sites3 === 3, sites3)

  // Empty tag = no sharing. Same block type, no tag => two separate slots.
  const x = make('x', conv, { in_ch: 3, out_ch: 16 })
  const y = make('y', conv, { in_ch: 3, out_ch: 16 })
  const codeUntagged = generate([x, y], [], 'pytorch')
  const untaggedSlots = (codeUntagged.match(/= ConvBlock\(/g) || []).length
  check('empty tag = each node gets its own slot', untaggedSlots === 2, untaggedSlots)

  // Different ctor values produce one slot with the FIRST node's ctor; the
  // validator surfaces a hard error separately.
  const aWide = make('aw', conv, { in_ch: 3, out_ch: 16 }, 'wide')
  const bWide = make('bw', conv, { in_ch: 3, out_ch: 32 }, 'wide')
  const codeDisagree = generate([aWide, bWide], [], 'pytorch')
  const wideSlots = (codeDisagree.match(/self\.wide = ConvBlock\(/g) || []).length
  check('ctor-disagreement still collapses to one slot (validator rejects)', wideSlots === 1, wideSlots)
}

// --- validator: tag conflict ---
console.log('validator tag conflicts')
{
  // Synthesize the smallest editor-shaped object the validator needs.
  async function loadValidator() {
    return (await import('./src/validator.js')).validate
  }
  const validate = await loadValidator()
  const conv = {
    name: 'ConvBlock',
    module: 'pytorch_blocks.core_blocks',
    framework: 'pytorch',
    kind: 'module',
    ctor: [
      { name: 'in_ch', type: 'int', default: null, required: true },
      { name: 'out_ch', type: 'int', default: null, required: true },
    ],
    inputs: [{ name: 'x', shape: ['B', 'C_in', 'H', 'W'], dtype: 'float' }],
    outputs: [{ name: 'out', shape: ['B', 'C_out', 'H', 'W'], dtype: 'float' }],
    bindings: { C_in: 'in_ch', C_out: 'out_ch' },
  }
  const linear = {
    name: 'Linear',
    module: 'pytorch_blocks.core_blocks',
    framework: 'pytorch',
    kind: 'module',
    ctor: [{ name: 'out_dim', type: 'int', default: null, required: true }],
    inputs: [{ name: 'x', shape: ['B', 'D'], dtype: 'float' }],
    outputs: [{ name: 'out', shape: ['B', 'D_out'], dtype: 'float' }],
    bindings: {},
  }
  const mkEd = (nodes) => ({
    getNodes: () => nodes,
    getConnections: () => [],
    getNode: (id) => nodes.find((n) => n.id === id),
  })
  const mkN = (id, entry, values, tag = '') => ({
    id,
    entry,
    tag,
    label: entry.name,
    values: { ...Object.fromEntries(entry.ctor.map((p) => [p.name, p.default])), ...values },
    inputs: {},
    outputs: {},
    freshenedShape: () => null,
    applyParamBindings: () => {},
  })

  // Same tag, same block, same ctor => OK.
  let res = validate(
    mkEd([
      mkN('a', conv, { in_ch: 3, out_ch: 16 }, 'down1'),
      mkN('b', conv, { in_ch: 3, out_ch: 16 }, 'down1'),
    ])
  )
  check(
    'tag-conflict: identical twins produce no error',
    res.errors.filter((e) => e.kind === 'tag-conflict').length === 0,
    res.errors
  )

  // Same tag, different block type => error.
  res = validate(
    mkEd([
      mkN('a', conv, { in_ch: 3, out_ch: 16 }, 'shared'),
      mkN('b', linear, { out_dim: 16 }, 'shared'),
    ])
  )
  check(
    'tag-conflict: different block types -> error',
    res.errors.some((e) => e.kind === 'tag-conflict' && /different block types/.test(e.message)),
    res.errors
  )

  // Same tag, same block, different ctor => error.
  res = validate(
    mkEd([
      mkN('a', conv, { in_ch: 3, out_ch: 16 }, 'down1'),
      mkN('b', conv, { in_ch: 3, out_ch: 32 }, 'down1'),
    ])
  )
  check(
    'tag-conflict: ctor disagreement -> error',
    res.errors.some((e) => e.kind === 'tag-conflict' && /out_ch/.test(e.message)),
    res.errors
  )

  // Empty tag never triggers a conflict, even with same block + same ctor.
  res = validate(
    mkEd([
      mkN('a', conv, { in_ch: 3, out_ch: 16 }, ''),
      mkN('b', conv, { in_ch: 3, out_ch: 16 }, ''),
    ])
  )
  check(
    'tag-conflict: empty tag does NOT trigger conflict',
    res.errors.filter((e) => e.kind === 'tag-conflict').length === 0,
    res.errors
  )
}

// --- Concat / Stack utility nodes ---
console.log('codegen (concat + stack)')
{
  const inputEntry = {
    name: 'Input',
    module: '__builtin__',
    framework: 'any',
    kind: 'input',
    ctor: [
      { name: 'name', type: 'str', default: 'x' },
      { name: 'shape', type: 'str', default: 'B C H W' },
      { name: 'dtype', type: 'str', default: 'float' },
    ],
    inputs: [],
    outputs: [{ name: 'out', shape: ['B', 'C', 'H', 'W'], dtype: 'float' }],
    bindings: {},
  }
  const make = (id, entry, values = {}) => ({
    id,
    entry,
    tag: '',
    values: {
      ...Object.fromEntries(entry.ctor.map((p) => [p.name, p.default])),
      ...values,
    },
  })
  const in1 = make('in1', inputEntry, { name: 'x1' })
  const in2 = make('in2', inputEntry, { name: 'x2' })
  const concat = make('cat', CONCAT_ENTRY, { dim: 1 })
  const stack = make('stk', STACK_ENTRY, { dim: 0 })
  const conns = [
    { source: 'in1', sourceOutput: 'out', target: 'cat', targetInput: 'xs' },
    { source: 'in2', sourceOutput: 'out', target: 'cat', targetInput: 'xs' },
    { source: 'in1', sourceOutput: 'out', target: 'stk', targetInput: 'xs' },
    { source: 'in2', sourceOutput: 'out', target: 'stk', targetInput: 'xs' },
  ]
  const codePt = generate([in1, in2, concat, stack], conns, 'pytorch')
  check('pytorch concat uses torch.cat([...], dim=)', codePt.includes('torch.cat([x1, x2], dim=1)'), codePt)
  check('pytorch stack uses torch.stack([...], dim=)', codePt.includes('torch.stack([x1, x2], dim=0)'), codePt)
  check('utility nodes skip block imports', !codePt.includes('__utility__'), codePt)

  const codeFlax = generate([in1, in2, concat], conns.slice(0, 2), 'flax')
  check(
    'flax concat uses jnp.concatenate',
    codeFlax.includes('jnp.concatenate([x1, x2], axis=1)'),
    codeFlax
  )
}

console.log('paletteGroup')
{
  check('Concat maps to utility palette', paletteGroup(CONCAT_ENTRY) === 'utility')
  check('Input stays in built-in', paletteGroup(inputEntryForPalette()) === 'built-in')
}

console.log('codegen (constant)')
{
  const cEntry = {
    name: 'Constant',
    module: '__builtin__',
    framework: 'any',
    kind: 'const',
    ctor: [
      { name: 'value_type', type: 'str', default: 'int' },
      { name: 'value', type: 'str', default: '1' },
    ],
    inputs: [],
    outputs: [{ name: 'out', shape: [], dtype: 'any' }],
    bindings: {},
  }
  const make = (id, entry, values = {}) => ({
    id,
    entry,
    tag: '',
    values: { ...Object.fromEntries(entry.ctor.map((p) => [p.name, p.default])), ...values },
  })
  const cInt = make('c1', cEntry, { value_type: 'int', value: '7' })
  const cBool = make('c2', cEntry, { value_type: 'bool', value: 'true' })
  const code = generate([cInt, cBool], [], 'pytorch')
  check('constant int emits numeric literal', code.includes('= 7'), code)
  check('constant bool emits Python bool literal', code.includes('= True'), code)
}

function inputEntryForPalette() {
  return {
    name: 'Input',
    module: '__builtin__',
    framework: 'any',
    kind: 'input',
    ctor: [],
    inputs: [],
    outputs: [],
    bindings: {},
  }
}

console.log(`\n${pass} pass, ${fail} fail`)
process.exit(fail === 0 ? 0 : 1)
