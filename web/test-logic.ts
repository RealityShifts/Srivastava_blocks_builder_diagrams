// Quick smoke test of the pure-JS modules (no rete / DOM needed).
import { normalize, freshen, prettyShape } from './src/shape.ts'
import { unifyShape, UnifyError } from './src/unify.ts'
import { generate, inputForwardArgName, outputReturnArgName } from './src/codegen.ts'
import {
  parseEinopsPattern,
  parseLengthsString,
  REARRANGE_ENTRY,
  RESHAPE_ENTRY,
  CONCAT_ENTRY,
  STACK_ENTRY,
  UNBIND_ENTRY,
  POOL_ENTRY,
  UPSAMPLE_ENTRY,
  OUTPUT_ENTRY,
  CONST_ENTRY,
  LEARNABLE_TENSOR_ENTRY,
  makeGroupEntry,
  makeNode,
  RearrangeNode,
  ReshapeNode,
  paletteGroup,
} from './src/nodes.ts'
import {
  boundarySignatureFromBoundary,
  boundarySignatureFromEntry,
  boundarySignaturesMatch,
  applySignatureToBoundary,
} from './src/groupBoundary.ts'
import { copyNodeValues, nodesInSameNameFamily } from './src/tagSync.ts'
import {
  makeTagAtlas,
  registerNodeMember,
  registerGroupMember,
  unregisterMember,
  recordValueChange,
  recordAllValues,
  recordExposedParamChange,
  recordGroupMeta,
  adoptValuesFromAtlas,
  adoptExposedParamsFromAtlas,
} from './src/tagAtlas.ts'

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

// --- jaxtyping annotations ---
console.log('codegen jaxtyping annotations')
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
  const mk = (id, entry, values = {}, tag = '') => ({
    id,
    entry,
    tag,
    values: { ...Object.fromEntries(entry.ctor.map((p) => [p.name, p.default])), ...values },
  })

  const inp = mk('inp', inputEntry, { name: 'x', shape: 'B 3 224 224', dtype: 'float' })
  const c1 = mk('c1', conv, { in_ch: 3, out_ch: 16 })
  const code = generate(
    [inp, c1],
    [{ source: 'inp', sourceOutput: 'out', target: 'c1', targetInput: 'x' }],
    'pytorch'
  )
  check('jaxtyping import emitted', /^from jaxtyping import [A-Z][A-Za-z, ]+$/m.test(code), code)
  check(
    'torch Tensor symbol imported for jaxtyping params',
    /^from torch import Tensor$/m.test(code),
    code
  )
  check(
    'forward arg uses Float[Tensor, "B 3 224 224"]',
    /def forward\(self, x: Float\[Tensor, "B 3 224 224"\]\)/.test(code),
    code
  )
  check(
    'forward return type uses the source port shape (C_out, not C_in)',
    /-> Float\[Tensor, "B C_out H W"\]:/.test(code),
    code
  )

  // Empty Input shape and unknown dtype -> Shaped with `...`.
  const inp2 = mk('inp2', inputEntry, { name: 'x', shape: '', dtype: '' })
  const c2 = mk('c2', conv, { in_ch: 3, out_ch: 16 })
  const code2 = generate(
    [inp2, c2],
    [{ source: 'inp2', sourceOutput: 'out', target: 'c2', targetInput: 'x' }],
    'pytorch'
  )
  check(
    'empty shape -> Shaped[Tensor, "..."]',
    /def forward\(self, x: Shaped\[Tensor, "\.\.\."\]\)/.test(code2),
    code2
  )

  // Input tag becomes forward arg when name is still the default `x`.
  check(
    'inputForwardArgName prefers custom name over tag',
    inputForwardArgName({ values: { name: 'rgb' }, tag: 'stem' }) === 'rgb'
  )
  check(
    'inputForwardArgName uses tag when name is default x',
    inputForwardArgName({ values: { name: 'x' }, tag: 'stem' }) === 'stem'
  )
  check(
    'inputForwardArgName uses tag when name is empty',
    inputForwardArgName({ values: { name: '' }, tag: 'images' }) === 'images'
  )
  const mkTagged = (id, entry, values = {}, tag = '') => ({
    id,
    entry,
    tag,
    values: {
      ...Object.fromEntries(entry.ctor.map((p) => [p.name, p.default])),
      ...values,
    },
  })
  const inpTag = mkTagged('inpT', inputEntry, { name: 'x', shape: 'B 3 224 224', dtype: 'float' }, 'images')
  const cTag = mk('cTag', conv, { in_ch: 3, out_ch: 16 })
  const codeTag = generate(
    [inpTag, cTag],
    [{ source: 'inpT', sourceOutput: 'out', target: 'cTag', targetInput: 'x' }],
    'pytorch'
  )
  check(
    'codegen forward arg uses Input tag when name is x',
    /def forward\(self, images: Float\[Tensor, "B 3 224 224"\]\)/.test(codeTag),
    codeTag
  )
  check('codegen wires tagged Input arg into block', /self\.conv_block_1\(x=images\)/.test(codeTag), codeTag)

  // Multi-output return -> tuple[...]
  const fork = {
    name: 'ForkBlock',
    module: 'pytorch_blocks.core_blocks',
    framework: 'pytorch',
    kind: 'module',
    ctor: [],
    inputs: [{ name: 'x', shape: ['B', 'C', 'H', 'W'], dtype: 'float' }],
    outputs: [
      { name: 'a', shape: ['B', 'C', 'H', 'W'], dtype: 'float' },
      { name: 'b', shape: ['B', 'C', 'H', 'W'], dtype: 'float' },
    ],
    bindings: {},
  }
  const inp3 = mk('inp3', inputEntry, { name: 'x', shape: 'B C H W', dtype: 'float' })
  const f = mk('f', fork, {})
  const code3 = generate(
    [inp3, f],
    [{ source: 'inp3', sourceOutput: 'out', target: 'f', targetInput: 'x' }],
    'pytorch'
  )
  check(
    'multi-output return uses tuple[]',
    /-> tuple\[Float\[Tensor, "B C H W"\], Float\[Tensor, "B C H W"\]\]:/.test(code3),
    code3
  )

  // Flax framework uses Array, not Tensor.
  const codeFlax = generate(
    [inp, c1],
    [{ source: 'inp', sourceOutput: 'out', target: 'c1', targetInput: 'x' }],
    'flax'
  )
  check(
    'flax uses Array for jaxtyping params',
    /^from jax import Array$/m.test(codeFlax) && /Float\[Array,/.test(codeFlax),
    codeFlax
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
  // The editable `name` drives the generated attribute (self.<name>); `tag`
  // groups instances into one shared slot. Weight-shared nodes share a name.
  const make = (id, entry, values = {}, tag = '', name = '') => ({
    id,
    entry,
    tag,
    name,
    values: { ...Object.fromEntries(entry.ctor.map((p) => [p.name, p.default])), ...values },
  })
  // Two ConvBlocks named "shared" sharing tag "shared" with the SAME ctor values
  // -> one __init__ slot named self.shared.
  const a = make('a', conv, { in_ch: 3, out_ch: 16 }, 'shared', 'shared')
  const b = make('b', conv, { in_ch: 3, out_ch: 16 }, 'shared', 'shared')
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
  const c = make('c', conv, { in_ch: 3, out_ch: 16 }, 'shared', 'shared')
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
  const aWide = make('aw', conv, { in_ch: 3, out_ch: 16 }, 'wide', 'wide')
  const bWide = make('bw', conv, { in_ch: 3, out_ch: 32 }, 'wide', 'wide')
  const codeDisagree = generate([aWide, bWide], [], 'pytorch')
  const wideSlots = (codeDisagree.match(/self\.wide = ConvBlock\(/g) || []).length
  check('ctor-disagreement still collapses to one slot (validator rejects)', wideSlots === 1, wideSlots)

  // Group facades sharing a tag -> one instance in GeneratedModel (same as modules).
  const residual = {
    name: 'ResidualBlock',
    module: 'pytorch_blocks.core_blocks',
    framework: 'pytorch',
    kind: 'module',
    ctor: [
      { name: 'in_ch', type: 'int', default: null, required: true },
      { name: 'out_ch', type: 'int', default: null, required: true },
    ],
    inputs: [{ name: 'x', shape: ['B', 'C', 'H', 'W'], dtype: 'float' }],
    outputs: [{ name: 'out', shape: ['B', 'C', 'H', 'W'], dtype: 'float' }],
    bindings: {},
  }
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
  const mkFacade = (id, gid, childId, tag) => ({
    id,
    entry: makeGroupEntry(gid, 'Group1', {
      inputs: [{ childNodeId: childId, childPort: 'x', shape: ['B', 3, 224, 224] }],
      outputs: [{ childNodeId: childId, childPort: 'out', shape: ['B', 4, 224, 224] }],
    }),
    tag,
    values: {},
    groupId: null,
  })
  const c1 = make('c1', residual, { in_ch: 3, out_ch: 4 })
  c1.groupId = 'g1'
  const c2 = make('c2', residual, { in_ch: 3, out_ch: 4 })
  c2.groupId = 'g2'
  const f1 = mkFacade('f1', 'g1', 'c1', 'encoder')
  const f2 = mkFacade('f2', 'g2', 'c2', 'encoder')
  const inp = make('inp', inputEntry, { name: 'x' })
  const codeSharedGroups = generate(
    [inp, f1, f2, c1, c2],
    [
      { source: 'inp', sourceOutput: 'out', target: 'f1', targetInput: 'in0' },
      { source: 'inp', sourceOutput: 'out', target: 'f2', targetInput: 'in0' },
    ],
    'pytorch'
  )
  check(
    'shared tag on group facades emits ONE self.encoder = Group1(...) slot',
    (codeSharedGroups.match(/self\.encoder = Group1\(\)/g) || []).length === 1,
    codeSharedGroups
  )
  check(
    'shared tag on groups: two forward call sites to self.encoder',
    (codeSharedGroups.match(/self\.encoder\(/g) || []).length === 2,
    codeSharedGroups
  )
  check(
    'shared tag on groups: only one Group1 class definition',
    (codeSharedGroups.match(/^class Group1\(nn\.Module\):/gm) || []).length === 1,
    codeSharedGroups
  )

  // Synced tag template can add facade ports without child bindings yet.
  const fUnmapped = mkFacade('fu', 'gu', 'cu', '')
  fUnmapped.entry = makeGroupEntry('gu', 'Group1', {
    inputs: [
      { childNodeId: 'cu', childPort: 'x', shape: ['B', 3, 224, 224] },
      { shape: ['B', 3, 224, 224] },
    ],
    outputs: [{ childNodeId: 'cu', childPort: 'out', shape: ['B', 4, 224, 224] }],
  })
  const cu = make('cu', residual, { in_ch: 3, out_ch: 4 })
  cu.groupId = 'gu'
  let threw = false
  try {
    generate([inp, fUnmapped, cu], [{ source: 'inp', sourceOutput: 'out', target: 'fu', targetInput: 'in0' }], 'pytorch')
  } catch {
    threw = true
  }
  check('group with unmapped synced port does not crash codegen', !threw, threw)

  // Different tags (or empty) -> separate instances, current behaviour preserved.
  const f3 = mkFacade('f3', 'g3', 'c3', '')
  const c3 = make('c3', residual, { in_ch: 3, out_ch: 4 })
  c3.groupId = 'g3'
  const f4 = mkFacade('f4', 'g4', 'c4', 'other')
  const c4 = make('c4', residual, { in_ch: 3, out_ch: 4 })
  c4.groupId = 'g4'
  const codeUntaggedGroups = generate([inp, f3, f4, c3, c4], [], 'pytorch')
  const separateInits = (codeUntaggedGroups.match(/= Group1\(\)/g) || []).length
  check('untagged / distinct-tag groups get separate instances', separateInits === 2, separateInits)
}

// --- tag sync ---
console.log('tag sync')
{
  const conv = {
    name: 'ConvBlock',
    kind: 'module',
    ctor: [
      { name: 'in_ch', type: 'int' },
      { name: 'out_ch', type: 'int' },
    ],
  }
  const a = { id: 'a', entry: conv, name: 'stem', tag: '', values: { in_ch: 3, out_ch: 16 } }
  const b = { id: 'b', entry: conv, name: 'stem', tag: '', values: { in_ch: 3, out_ch: 32 } }
  check('nodesInSameNameFamily matches same block + name', nodesInSameNameFamily(a, b))
  const bTag = { id: 'b2', entry: conv, name: '', tag: 'stem', values: {} }
  check('nodesInSameNameFamily ignores tag (blank names do not sync)', !nodesInSameNameFamily(a, bTag))
  check(
    'copyNodeValues mirrors ctor fields onto peer',
    copyNodeValues(a, b) && b.values.out_ch === 16,
    b.values
  )
  const c = { id: 'c', entry: { ...conv, name: 'Linear' }, name: 'stem', tag: '', values: {} }
  check('copyNodeValues skips different block types', !copyNodeValues(a, c))
}

// --- tag atlas: single source of truth for tagged nodes & groups ---
console.log('tag atlas')
{
  const conv = {
    name: 'ConvBlock',
    kind: 'module',
    ctor: [
      { name: 'in_ch', type: 'int' },
      { name: 'out_ch', type: 'int' },
    ],
  }
  // Regular nodes are keyed by their editable name (param-sync identity).
  const mkNode = (id, name, values = {}, inputs = {}) => ({
    id,
    entry: conv,
    name,
    tag: '',
    values: { ...values },
    inputs,
  })

  const atlas = makeTagAtlas()
  const a = mkNode('a', 'down1', { in_ch: 3, out_ch: 16 })
  const b = mkNode('b', 'down1', { in_ch: 99, out_ch: 99 })
  const entry1 = registerNodeMember(atlas, a)
  check('first member becomes canonical seed', entry1?.values.out_ch === 16, entry1)
  const entry2 = registerNodeMember(atlas, b)
  check('second member joins the same entry', entry1 === entry2 && entry2.members.size === 2)

  // Second node should adopt canonical values from atlas (peer joined).
  adoptValuesFromAtlas(atlas, b)
  check('joined member adopts canonical values', b.values.out_ch === 16, b.values)

  // Source of truth: mutate a, propagate.
  a.values.out_ch = 64
  const peersToUpdate = recordValueChange(atlas, a, 'out_ch')
  check('recordValueChange reports peer ids', peersToUpdate.length === 1 && peersToUpdate[0] === 'b')
  check('atlas canonical value updated', atlas.get('down1').values.out_ch === 64)

  // Exposed params: structural change propagation.
  const inputsWith = { __param__kernel_size: { portSpec: { kind: 'param', paramName: 'kernel_size' } } }
  const c = mkNode('c', 'block', { in_ch: 3, out_ch: 16 }, inputsWith)
  registerNodeMember(atlas, c)
  const peers2 = recordExposedParamChange(atlas, c, 'kernel_size', true)
  check('exposed-param change tracked even with no peers', peers2.length === 0)
  check('atlas tracks exposed params', atlas.get('block').exposedParams.has('kernel_size'))

  const d = mkNode('d', 'block', { in_ch: 3, out_ch: 16 })
  registerNodeMember(atlas, d)
  const diff = adoptExposedParamsFromAtlas(atlas, d)
  check('new peer is told to expose missing param ports', diff.toExpose.includes('kernel_size'))

  // Unnamed nodes never touch the atlas.
  const unnamed = mkNode('u', '', { in_ch: 0, out_ch: 0 })
  check('unnamed registration is a no-op', registerNodeMember(atlas, unnamed) === null)

  // Unregister: last member removes entry.
  unregisterMember(atlas, 'down1', 'a')
  unregisterMember(atlas, 'down1', 'b')
  check('atlas drops empty entries', !atlas.has('down1'))

  // Group meta propagates through atlas.
  const facade = { entry: { kind: 'group', name: 'Group1' }, tag: 'enc' }
  const g1 = { id: 'g1', name: 'Group1', description: '', facadeTag: 'enc' }
  const g2 = { id: 'g2', name: 'Group1', description: '', facadeTag: 'enc' }
  registerGroupMember(atlas, g1, facade)
  registerGroupMember(atlas, g2, facade)
  const gPeers = recordGroupMeta(atlas, g1, { name: 'Encoder', description: 'note' })
  check('group meta peers reported', gPeers.length === 1 && gPeers[0] === 'g2')
  check('atlas stores canonical group name', atlas.get('enc').name === 'Encoder')
  check('atlas stores canonical group description', atlas.get('enc').description === 'note')
}

// --- group boundary signatures ---
console.log('group boundary signatures')
{
  const shapeKey = (shape) => JSON.stringify(shape)
  const sig1 = boundarySignatureFromBoundary({
    inputs: [{ shape: ['B', 'C', 'H', 'W'], dtype: 'float' }],
    outputs: [{ shape: ['B', 'C2', 'H', 'W'], dtype: 'float' }],
    params: [{ paramName: 'kernel_size', paramType: 'int' }],
  })
  const sig2 = boundarySignatureFromBoundary({
    inputs: [{ shape: ['B', 'C', 'H', 'W'], dtype: 'float' }],
    outputs: [{ shape: ['B', 'C2', 'H', 'W'], dtype: 'float' }],
    params: [{ paramName: 'kernel_size', paramType: 'int' }],
  })
  check('matching signatures compare equal', boundarySignaturesMatch(sig1, sig2))

  // Regression: freshened shape tokens carry a `#<nodeId>` suffix that must
  // be stripped before comparison, otherwise two facades over different
  // children always look like different interfaces.
  const sigFreshA = boundarySignatureFromBoundary({
    inputs: [{ shape: ['B#nodeA', 'C#nodeA', 'H#nodeA', 'W#nodeA'], dtype: 'float' }],
    outputs: [{ shape: ['B#nodeA', 'C2#nodeA', 'H#nodeA', 'W#nodeA'], dtype: 'float' }],
  })
  const sigFreshB = boundarySignatureFromBoundary({
    inputs: [{ shape: ['B#nodeB', 'C#nodeB', 'H#nodeB', 'W#nodeB'], dtype: 'float' }],
    outputs: [{ shape: ['B#nodeB', 'C2#nodeB', 'H#nodeB', 'W#nodeB'], dtype: 'float' }],
  })
  check(
    'freshen-suffixed shapes still compare equal across instances',
    boundarySignaturesMatch(sigFreshA, sigFreshB)
  )

  const sig3 = boundarySignatureFromBoundary({
    inputs: [
      { shape: ['B', 'C', 'H', 'W'], dtype: 'float' },
      { shape: ['B', 'C', 'H', 'W'], dtype: 'float' },
    ],
    outputs: [{ shape: ['B', 'C2', 'H', 'W'], dtype: 'float' }],
  })
  check('extra input port breaks match', !boundarySignaturesMatch(sig1, sig3))

  const entry = makeGroupEntry('g1', 'Group1', {
    inputs: [{ shape: ['B', 3, 'H', 'W'], childNodeId: 'c1', childPort: 'x' }],
    outputs: [{ shape: ['B', 16, 'H', 'W'], childNodeId: 'c2', childPort: 'out' }],
    params: [],
  })
  const fromEntry = boundarySignatureFromEntry(entry)
  check('signature from entry preserves input shape', shapeKey(fromEntry.inputs[0].shape) === shapeKey(['B', 3, 'H', 'W']))

  const merged = applySignatureToBoundary(
    {
      inputs: [{ shape: ['B', 3, 'H', 'W'], childNodeId: 'c9', childPort: 'x' }],
      outputs: [],
      params: [],
    },
    sig1,
    { inputs: [{ childNodeId: 'old', childPort: 'x' }], outputs: [], params: [] }
  )
  check('applySignature pads outputs from template', merged.outputs.length === 1)
  check('applySignature keeps local child binding', merged.inputs[0].childNodeId === 'c9')
  check('applySignature adds param port from template', merged.params.length === 1)
}

// --- validator: tag conflict ---
console.log('validator tag conflicts')
{
  // Synthesize the smallest editor-shaped object the validator needs.
  async function loadValidator() {
    return (await import('./src/validator.ts')).validate
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

  const groupEntry1 = makeGroupEntry('g1', 'Enc', {
    inputs: [{ shape: ['B', 'C', 'H', 'W'], childNodeId: 'c1', childPort: 'x' }],
    outputs: [{ shape: ['B', 'C2', 'H', 'W'], childNodeId: 'c2', childPort: 'out' }],
  })
  const groupEntry2 = makeGroupEntry('g2', 'Enc', {
    inputs: [
      { shape: ['B', 'C', 'H', 'W'], childNodeId: 'c3', childPort: 'x' },
      { shape: ['B', 'C', 'H', 'W'], childNodeId: 'c4', childPort: 'x' },
    ],
    outputs: [{ shape: ['B', 'C2', 'H', 'W'], childNodeId: 'c5', childPort: 'out' }],
  })
  res = validate(
    mkEd([
      mkN('f1', groupEntry1, {}, 'shared-enc'),
      mkN('f2', groupEntry2, {}, 'shared-enc'),
    ])
  )
  check(
    'tag-conflict: group facades with mismatched boundary -> error',
    res.errors.some((e) => e.kind === 'tag-conflict' && /boundary interface differs/.test(e.message)),
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

console.log('codegen (pool + upsample)')
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
  const inp = make('in1', inputEntry, { name: 'x' })
  const pool = make('pool', POOL_ENTRY, { mode: 'max', kernel_size: 2, stride: 0, padding: 0 })
  const up = make('up', UPSAMPLE_ENTRY, { scale_factor: 2, align_corners: false })
  const conns = [
    { source: 'in1', sourceOutput: 'out', target: 'pool', targetInput: 'x' },
    { source: 'pool', sourceOutput: 'out', target: 'up', targetInput: 'x' },
  ]
  const codePt = generate([inp, pool, up], conns, 'pytorch')
  check(
    'pytorch pool uses max_pool2d',
    codePt.includes('torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)'),
    codePt
  )
  check(
    'pytorch upsample uses bilinear interpolate',
    codePt.includes("torch.nn.functional.interpolate(pool2d_1, scale_factor=2, mode='bilinear', align_corners=False)"),
    codePt
  )
  check('pool/upsample skip block imports', !codePt.includes('__utility__'), codePt)

  const poolAvg = make('pool2', POOL_ENTRY, { mode: 'avg', kernel_size: 3, stride: 2, padding: 1 })
  const codeFlax = generate([inp, poolAvg], conns.slice(0, 1).map((c) => ({ ...c, target: 'pool2' })), 'flax')
  check('flax pool imports linen pooling', codeFlax.includes('from flax.linen.pooling import avg_pool, max_pool'), codeFlax)
  check('flax avg pool transposes NCHW<->NHWC', codeFlax.includes('avg_pool(jnp.transpose('), codeFlax)

  const codeFlaxUp = generate(
    [inp, up],
    [{ source: 'in1', sourceOutput: 'out', target: 'up', targetInput: 'x' }],
    'flax'
  )
  check('flax upsample imports jax.image', codeFlaxUp.includes('import jax.image'), codeFlaxUp)
  check('flax upsample uses jax.image.resize linear', codeFlaxUp.includes("method='linear'"), codeFlaxUp)
}

console.log('paletteGroup')
{
  check('Concat maps to utility palette', paletteGroup(CONCAT_ENTRY) === 'utility')
  check('Pool2d maps to utility palette', paletteGroup(POOL_ENTRY) === 'utility')
  check('Upsample maps to utility palette', paletteGroup(UPSAMPLE_ENTRY) === 'utility')
  check('Input stays in built-in', paletteGroup(inputEntryForPalette()) === 'built-in')
  check('Constant stays in built-in', paletteGroup(CONST_ENTRY) === 'built-in')
  check('LearnableTensor stays in built-in', paletteGroup(LEARNABLE_TENSOR_ENTRY) === 'built-in')
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
  check(
    'unwired constants become __init__ params with UI defaults',
    /def __init__\(self, constant_0: int = 7, constant_1: bool = True\)/.test(code),
    code
  )
  check('constants are not stored on self', !/self\.constant_\d+\s*=/.test(code), code)
  check('constants are not emitted in forward', !/def forward[\s\S]*=\s*7/.test(code), code)

  // Wired constant -> __init__ kwarg with default + module ctor references self.<param>.
  const conv = {
    name: 'ConvBlock',
    module: 'pytorch_blocks.core_blocks',
    framework: 'pytorch',
    kind: 'module',
    ctor: [
      { name: 'in_ch', type: 'int', default: null, required: true },
      { name: 'out_ch', type: 'int', default: null, required: true },
      { name: 'kernel_size', type: 'int', default: 3, required: false },
    ],
    inputs: [
      { name: 'x', shape: ['B', 'C_in', 'H', 'W'], dtype: 'float' },
      { name: '__param__kernel_size', shape: [], dtype: 'any', kind: 'param', paramName: 'kernel_size' },
    ],
    outputs: [{ name: 'out', shape: ['B', 'C_out', 'H', 'W'], dtype: 'float' }],
    bindings: { C_in: 'in_ch', C_out: 'out_ch' },
  }
  const cKs = make('cks', cEntry, { value_type: 'int', value: '5' })
  const convNode = make('conv', conv, { in_ch: 3, out_ch: 16 })
  convNode.inputs = {
    x: { portSpec: conv.inputs[0] },
    __param__kernel_size: {
      portSpec: {
        kind: 'param',
        paramName: 'kernel_size',
        name: '__param__kernel_size',
        type: 'int',
      },
    },
  }
  const wiredCode = generate(
    [cKs, convNode],
    [{ source: 'cks', sourceOutput: 'out', target: 'conv', targetInput: '__param__kernel_size' }],
    'pytorch'
  )
  check(
    'wired constant becomes __init__ kwarg with default',
    /def __init__\(self, kernel_size: int = 5\)/.test(wiredCode),
    wiredCode
  )
  check(
    'wired constant is not stored on self',
    !/self\.kernel_size\s*=/.test(wiredCode),
    wiredCode
  )
  check(
    'module ctor references __init__ param from constant',
    /ConvBlock\([^)]*kernel_size=kernel_size/.test(wiredCode),
    wiredCode
  )
}

console.log('codegen (learnable tensor)')
{
  const make = (id, entry, values = {}) => ({
    id,
    entry,
    tag: '',
    values: { ...Object.fromEntries(entry.ctor.map((p) => [p.name, p.default])), ...values },
  })
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
  const add = {
    name: 'Concat',
    module: '__utility__',
    framework: 'any',
    kind: 'concat',
    ctor: [{ name: 'dim', type: 'int', default: 1, required: false }],
    inputs: [{ name: 'xs', shape: ['...'], dtype: 'float', optional: false, variadic: true }],
    outputs: [{ name: 'out', shape: ['...'], dtype: 'float', optional: false, variadic: false }],
    bindings: {},
  }
  const inp = make('inp', inputEntry, { name: 'x', shape: 'B 3 32 32' })
  const token = make('tok', LEARNABLE_TENSOR_ENTRY, {
    name: 'cls_token',
    shape: '1 1 768',
    init: 'zeros',
  })
  const cat = make('cat', add, { dim: 1 })
  const conns = [
    { source: 'inp', sourceOutput: 'out', target: 'cat', targetInput: 'xs' },
    { source: 'tok', sourceOutput: 'out', target: 'cat', targetInput: 'xs' },
  ]
  const code = generate([inp, token, cat], conns, 'pytorch')
  check(
    'learnable tensor becomes nn.Parameter in __init__',
    /self\.cls_token = nn\.Parameter\(torch\.zeros\(1, 1, 768\)\)/.test(code),
    code
  )
  check(
    'learnable tensor wired as self.<name> in forward',
    /torch\.cat\(\[x, self\.cls_token\], dim=1\)/.test(code),
    code
  )
  check('learnable tensor skips __builtin__ import', !code.includes('__builtin__'), code)

  const flaxToken = make('lt', LEARNABLE_TENSOR_ENTRY, { init: 'randn' })
  const flaxCode = generate([flaxToken], [], 'flax')
  check(
    'flax learnable tensor uses nnx.Param with jax random init',
    /self\.param = nnx\.Param\(jax\.random\.normal\(rngs\.params\(\), \(1, 768\)\)\)/.test(flaxCode),
    flaxCode
  )
}

console.log('codegen (explicit Output node)')
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
  const make = (id, entry, values = {}) => ({
    id,
    entry,
    tag: '',
    values: { ...Object.fromEntries(entry.ctor.map((p) => [p.name, p.default])), ...values },
  })
  const inp = make('inp', inputEntry, { name: 'x' })
  const c1 = make('c1', conv, { in_ch: 3, out_ch: 16 })
  const c2 = make('c2', conv, { in_ch: 16, out_ch: 32 })
  const outNode = make('o1', OUTPUT_ENTRY, { name: 'y' })
  const conns = [
    { source: 'inp', sourceOutput: 'out', target: 'c1', targetInput: 'x' },
    { source: 'c1', sourceOutput: 'out', target: 'c2', targetInput: 'x' },
    { source: 'c1', sourceOutput: 'out', target: 'o1', targetInput: 'x' },
  ]
  const code = generate([inp, c1, c2, outNode], conns, 'pytorch')
  check(
    'with Output node, assigns default return alias y from wired source',
    /y = conv_block_1/.test(code) && /return y/.test(code),
    code
  )
  check(
    'with Output node, leaf output is ignored',
    !/return conv_block_2/.test(code),
    code
  )

  check(
    'outputReturnArgName prefers custom name over tag',
    outputReturnArgName({ values: { name: 'logits' }, tag: 'head' }) === 'logits'
  )
  check(
    'outputReturnArgName uses tag when name is default y',
    outputReturnArgName({ values: { name: 'y' }, tag: 'features' }) === 'features'
  )
  const outTagged = make('o2', OUTPUT_ENTRY, { name: 'y' })
  outTagged.tag = 'features'
  const codeOutTag = generate(
    [inp, c1, outTagged],
    [
      { source: 'inp', sourceOutput: 'out', target: 'c1', targetInput: 'x' },
      { source: 'c1', sourceOutput: 'out', target: 'o2', targetInput: 'x' },
    ],
    'pytorch'
  )
  check(
    'codegen return uses Output tag when name is y',
    /features = conv_block_1/.test(codeOutTag) && /return features/.test(codeOutTag),
    codeOutTag
  )
}

console.log('codegen (group facade param port from external constant)')
{
  const inputEntry = {
    name: 'Input', module: '__builtin__', framework: 'any', kind: 'input',
    ctor: [
      { name: 'name', type: 'str', default: 'x' },
      { name: 'shape', type: 'str', default: 'B C H W' },
      { name: 'dtype', type: 'str', default: 'float' },
    ],
    inputs: [],
    outputs: [{ name: 'out', shape: ['B', 'C', 'H', 'W'], dtype: 'float' }],
    bindings: {},
  }
  const conv = {
    name: 'ConvBlock', module: 'pytorch_blocks.core_blocks', framework: 'pytorch', kind: 'module',
    ctor: [
      { name: 'in_ch', type: 'int', default: null, required: true },
      { name: 'out_ch', type: 'int', default: null, required: true },
      { name: 'kernel_size', type: 'int', default: 3, required: false },
    ],
    inputs: [{ name: 'x', shape: ['B', 'C_in', 'H', 'W'], dtype: 'float' }],
    outputs: [{ name: 'out', shape: ['B', 'C_out', 'H', 'W'], dtype: 'float' }],
    bindings: { C_in: 'in_ch', C_out: 'out_ch' },
  }
  const constEntry = {
    name: 'Constant', module: '__builtin__', framework: 'any', kind: 'const',
    ctor: [
      { name: 'value_type', type: 'str', default: 'int' },
      { name: 'value', type: 'str', default: '1' },
    ],
    inputs: [],
    outputs: [{ name: 'out', shape: [], dtype: 'any' }],
    bindings: {},
  }
  const mk = (id, entry, values = {}, extra = {}) => ({
    id, entry, tag: '', ...extra,
    values: { ...Object.fromEntries(entry.ctor.map((p) => [p.name, p.default])), ...values },
  })
  const inp = mk('inp', inputEntry, { name: 'x', shape: '1 3 200 200', dtype: 'float' })
  const c1 = mk('c1', conv, { in_ch: 3, out_ch: 8, kernel_size: 5 }, { groupId: 'g1' })
  const k = mk('k', constEntry, { value: '5', value_type: 'int' })
  const facadeEntry = {
    name: 'Group1', module: '__group__', framework: 'any', kind: 'group', groupId: 'g1',
    ctor: [],
    inputs: [{ name: 'in0', shape: ['B', 'C_in', 'H', 'W'], dtype: 'any' }],
    outputs: [{ name: 'out0', shape: ['B', 'C_out', 'H', 'W'], dtype: 'any' }],
    portMap: {
      inputs: [{ facadePort: 'in0', childNodeId: 'c1', childPort: 'x', shape: ['B', 'C_in', 'H', 'W'] }],
      outputs: [{ facadePort: 'out0', childNodeId: 'c1', childPort: 'out', shape: ['B', 'C_out', 'H', 'W'] }],
      params: [{
        facadePort: '__param__kernel_size',
        childNodeId: 'c1',
        childPort: '__param__kernel_size',
        paramName: 'kernel_size',
        paramType: 'int',
      }],
    },
    bindings: {},
  }
  const fac = mk('fac', facadeEntry, {})
  fac.inputs = {
    in0: { portSpec: facadeEntry.inputs[0] },
    __param__kernel_size: {
      portSpec: { kind: 'param', paramName: 'kernel_size', paramType: 'int', required: true },
    },
  }
  const conns = [
    { source: 'inp', sourceOutput: 'out', target: 'fac', targetInput: 'in0' },
    { source: 'k', sourceOutput: 'out', target: 'fac', targetInput: '__param__kernel_size' },
  ]
  const code = generate([inp, fac, c1, k], conns, 'pytorch')
  check('subclass __init__ has kernel_size', /class Group1[\s\S]+?def __init__\(self, kernel_size: int = 5\)/.test(code), code)
  check('main __init__ has kernel_size from const', /class GeneratedModel[\s\S]+?def __init__\(self, kernel_size: int = 5\)/.test(code), code)
  check('main passes kernel_size to group', /Group1\(kernel_size=kernel_size\)/.test(code), code)
  check('subclass passes kernel_size to ConvBlock', /ConvBlock\([^)]*kernel_size=kernel_size/.test(code), code)
}

console.log('codegen (expanded group: children with stale groupId)')
{
  // Regression: when a group is expanded, the facade is gone but children
  // keep their groupId. Codegen must treat those orphans as top-level
  // instead of silently dropping them (which would produce an empty model
  // and break runtime shape check with `x=None  # TODO: dangling required input`).
  const inputEntry = {
    name: 'Input', module: '__builtin__', framework: 'any', kind: 'input',
    ctor: [
      { name: 'name', type: 'str', default: 'x' },
      { name: 'shape', type: 'str', default: 'B C H W' },
      { name: 'dtype', type: 'str', default: 'float' },
    ],
    inputs: [],
    outputs: [{ name: 'out', shape: ['B', 'C', 'H', 'W'], dtype: 'float' }],
    bindings: {},
  }
  const conv = {
    name: 'ConvBlock', module: 'pytorch_blocks.core_blocks', framework: 'pytorch', kind: 'module',
    ctor: [
      { name: 'in_ch', type: 'int', default: null, required: true },
      { name: 'out_ch', type: 'int', default: null, required: true },
    ],
    inputs: [{ name: 'x', shape: ['B', 'C_in', 'H', 'W'], dtype: 'float' }],
    outputs: [{ name: 'out', shape: ['B', 'C_out', 'H', 'W'], dtype: 'float' }],
    bindings: { C_in: 'in_ch', C_out: 'out_ch' },
  }
  const mk = (id, entry, values = {}, extra = {}) => ({
    id, entry, tag: '', ...extra,
    values: { ...Object.fromEntries(entry.ctor.map((p) => [p.name, p.default])), ...values },
  })
  const inp = mk('inp', inputEntry, { name: 'x', shape: 'B 3 224 224', dtype: 'float' })
  // Two children that USED to live in a group "g1" but are now expanded
  // (no facade present). They MUST still appear in the generated forward().
  const c1 = mk('c1', conv, { in_ch: 3, out_ch: 16 }, { groupId: 'g1' })
  const c2 = mk('c2', conv, { in_ch: 16, out_ch: 32 }, { groupId: 'g1' })
  const conns = [
    { source: 'inp', sourceOutput: 'out', target: 'c1', targetInput: 'x' },
    { source: 'c1', sourceOutput: 'out', target: 'c2', targetInput: 'x' },
  ]
  const code = generate([inp, c1, c2], conns, 'pytorch')
  check(
    'expanded group: both children land in __init__',
    (code.match(/ConvBlock\(in_ch=/g) || []).length === 2,
    code
  )
  check(
    'expanded group: forward wires c1 from input x',
    /self\.\w*conv_block\w*\(x=x\)/.test(code),
    code
  )
  check(
    'expanded group: forward chains c2 from c1',
    /self\.\w*conv_block\w*\(x=conv_block_\d+\)/.test(code),
    code
  )
  check(
    'expanded group: NO dangling required input',
    !/dangling required input/.test(code),
    code
  )
}

console.log('codegen (test case + trace)')
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
  const mk = (id, entry, values = {}) => ({
    id,
    entry,
    tag: '',
    values: { ...Object.fromEntries(entry.ctor.map((p) => [p.name, p.default])), ...values },
  })
  const inp = mk('inp', inputEntry, { name: 'x', shape: 'B 3 224 224', dtype: 'float' })
  const c1 = mk('c1', conv, { in_ch: 3, out_ch: 16 })
  const conns = [{ source: 'inp', sourceOutput: 'out', target: 'c1', targetInput: 'x' }]

  // With test case, no trace: emits test_GeneratedModel + __main__ guard.
  const codeWithTest = generate([inp, c1], conns, 'pytorch', {
    testCase: [{ arg: 'x', shape: [2, 3, 224, 224], dtype: 'float' }],
  })
  check(
    'test mode emits def test_GeneratedModel()',
    /^def test_GeneratedModel\(\) -> None:/m.test(codeWithTest),
    codeWithTest
  )
  check(
    'test mode emits __main__ guard',
    /if __name__ == "__main__":\n\s+test_GeneratedModel\(\)/.test(codeWithTest),
    codeWithTest
  )
  check(
    'test mode builds dummy tensor with concrete shape',
    /x = torch\.randn\(\(2, 3, 224, 224\)\)/.test(codeWithTest),
    codeWithTest
  )
  check(
    'test mode wraps call in torch.no_grad()',
    /with torch\.no_grad\(\):\n\s+out = model\(x=x\)/.test(codeWithTest),
    codeWithTest
  )
  check(
    'plain test mode prints output shape (not asserting dict)',
    /print\(f"out: \{tuple\(out\.shape\)\}"\)/.test(codeWithTest) &&
      !/assert isinstance\(out, dict\)/.test(codeWithTest),
    codeWithTest
  )

  // Trace mode + test: forward() returns dict, test asserts dict + prints entries.
  const codeTrace = generate([inp, c1], conns, 'pytorch', {
    trace: true,
    testCase: [{ arg: 'x', shape: [2, 3, 224, 224], dtype: 'float' }],
  })
  check(
    'trace mode: forward() returns _runtime_shapes',
    /return _runtime_shapes/.test(codeTrace),
    codeTrace
  )
  check(
    'trace mode: test asserts dict return',
    /assert isinstance\(out, dict\)/.test(codeTrace),
    codeTrace
  )
  check(
    'trace mode: test iterates the runtime shapes dict',
    /for k, v in out\.items\(\):/.test(codeTrace),
    codeTrace
  )

  // No testCase option -> no test/main block at all (backward compatible).
  const codePlain = generate([inp, c1], conns, 'pytorch')
  check(
    'plain mode does NOT emit __main__ block',
    !/if __name__ == "__main__":/.test(codePlain),
    codePlain
  )
  check(
    'plain mode does NOT emit test_GeneratedModel',
    !/def test_GeneratedModel/.test(codePlain),
    codePlain
  )

  // Int dtype -> torch.randint, not torch.randn.
  const codeIntInput = generate([inp, c1], conns, 'pytorch', {
    testCase: [{ arg: 'x', shape: [2, 16], dtype: 'long' }],
  })
  check(
    'int dtype uses torch.randint',
    /x = torch\.randint\(0, 100, \(2, 16\)\)/.test(codeIntInput),
    codeIntInput
  )

  // Flax: nnx.Rngs init, jnp.zeros tensors, no torch.no_grad block.
  const codeFlax = generate([inp, c1], conns, 'flax', {
    testCase: [{ arg: 'x', shape: [2, 3, 224, 224], dtype: 'float' }],
  })
  check(
    'flax test instantiates with nnx.Rngs(0)',
    /model = GeneratedModel\(rngs=nnx\.Rngs\(0\)\)/.test(codeFlax),
    codeFlax
  )
  check(
    'flax test uses jnp.zeros with float32',
    /x = jnp\.zeros\(\(2, 3, 224, 224\), dtype=jnp\.float32\)/.test(codeFlax),
    codeFlax
  )
  check(
    'flax test calls model directly (no torch.no_grad)',
    !/torch\.no_grad/.test(codeFlax) && /out = model\(x=x\)/.test(codeFlax),
    codeFlax
  )
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

// --- Unbind: dynamic output ports ---
console.log('unbind dynamic output ports')
{
  const u: any = makeNode(UNBIND_ENTRY)
  check('unbind starts with default 2 outputs', u.entry.outputs.length === 2, u.entry.outputs.length)
  check('unbind does NOT mutate the shared catalogue entry', UNBIND_ENTRY.outputs.length === 2)

  u.values.count = 4
  const grow = u.rebuildOutputs()
  check('grow to 4 adds out2,out3', u.entry.outputs.length === 4 && grow.added.length === 2 && grow.removed.length === 0, grow)
  check('grown ports are named out0..out3', u.entry.outputs.map((p: any) => p.name).join(',') === 'out0,out1,out2,out3')

  u.values.count = 2
  const shrink = u.rebuildOutputs()
  check('shrink to 2 removes out2,out3', u.entry.outputs.length === 2 && shrink.removed.join(',') === 'out3,out2', shrink)

  const noop = u.rebuildOutputs()
  check('rebuildOutputs is idempotent', noop.added.length === 0 && noop.removed.length === 0)

  u.values.count = 0
  u.rebuildOutputs()
  check('count<1 clamps to 1 output', u.entry.outputs.length === 1, u.entry.outputs.length)
}

// --- Unbind: codegen (both frameworks + count=1) ---
console.log('unbind codegen')
{
  const inputEntry = {
    name: 'Input', module: '__builtin__', framework: 'any', kind: 'input',
    ctor: [{ name: 'name', type: 'str', default: 'x' }],
    inputs: [], outputs: [{ name: 'out', shape: ['B', 'C', 'H', 'W'], dtype: 'float' }], bindings: {},
  }
  const sink = {
    name: 'ConvBlock', module: 'pytorch_blocks.core_blocks', framework: 'pytorch', kind: 'module',
    ctor: [], inputs: [{ name: 'x', shape: ['...'], dtype: 'float' }],
    outputs: [{ name: 'out', shape: ['...'], dtype: 'float' }], bindings: {},
  }
  const mk = (id: any, entry: any, values: any = {}) => ({
    id, entry, tag: '', name: '',
    values: { ...Object.fromEntries(entry.ctor.map((p: any) => [p.name, p.default])), ...values },
    groupId: null,
  })
  const unbindNode = (id: any, dim: number, count: number) => ({
    id, tag: '', name: '', groupId: null, values: { dim, count },
    entry: { ...UNBIND_ENTRY, outputs: Array.from({ length: count }, (_, i) => ({ name: `out${i}`, shape: ['...'], dtype: 'float', optional: false, variadic: false })) },
  })
  const build = (dim: number, count: number, fw: any) => {
    const inp = mk('inp', inputEntry, { name: 'x' })
    const u = unbindNode('u', dim, count)
    const sinks = Array.from({ length: count }, (_, i) => mk('s' + i, sink))
    const conns = [
      { source: 'inp', sourceOutput: 'out', target: 'u', targetInput: 'x' },
      ...sinks.map((s, i) => ({ source: 'u', sourceOutput: `out${i}`, target: s.id, targetInput: 'x' })),
    ]
    return generate([inp, u, ...sinks], conns, fw)
  }

  const pt = build(1, 3, 'pytorch')
  check('unbind pytorch: tuple-unpacks torch.unbind into N vars',
    /unbind_1_out0, unbind_1_out1, unbind_1_out2 = torch\.unbind\(x, dim=1\)/.test(pt), pt)

  const fx = build(1, 3, 'flax')
  check('unbind flax: tuple(jnp.moveaxis(...)) unpack',
    /unbind_1_out0, unbind_1_out1, unbind_1_out2 = tuple\(jnp\.moveaxis\(x, 1, 0\)\)/.test(fx), fx)

  const one = build(0, 1, 'pytorch')
  check('unbind count=1: indexes [0] so a tensor (not a 1-tuple) is assigned',
    /= torch\.unbind\(x, dim=0\)\[0\]/.test(one) && !/= torch\.unbind\(x, dim=0\)\n/.test(one), one)
}

console.log(`\n${pass} pass, ${fail} fail`)
process.exit(fail === 0 ? 0 : 1)
