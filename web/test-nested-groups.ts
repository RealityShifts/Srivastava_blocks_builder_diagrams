// Pure-node codegen tests for NESTED GROUPS (arbitrary depth). Each group
// compiles to its own Python class; an inner group nested in an outer group is
// a member module of the outer class. A nested facade carries BOTH
// `entry.groupId` (its identity -> its class) AND `node.groupId` (the outer
// group it belongs to). Mirrors the hand-built style of test-group-dup-param.ts.
import { generate } from './src/codegen.ts'

const tensorPort = (name: string, optional = false) => ({
  name, shape: ['...'], dtype: 'any', optional, variadic: false,
})

const mkNode = (id: string, entry: any, values: any, groupId: string | null) => ({
  id, entry, label: entry.name, tag: '', groupId, values,
  inputs: Object.fromEntries((entry.inputs || []).map((p: any) => [p.name, { portSpec: p }])),
  outputs: Object.fromEntries((entry.outputs || []).map((p: any) => [p.name, { portSpec: p }])),
  freshenedShape() { return null },
  applyParamBindings() {},
})

// A real leaf module (maps to a Python class from the manifest).
const leafEntry = (name: string) => ({
  kind: 'module', name, module: 'm',
  ctor: [{ name: 'out_ch', type: 'int', default: 8 }],
  inputs: [tensorPort('x')], outputs: [tensorPort('out')], bindings: {},
})

// A group facade: entry.groupId = its own identity; in0/out0 boundary ports.
const facadeEntry = (gid: string, name: string) => ({
  kind: 'group', name, groupId: gid, module: 'm', ctor: [],
  inputs: [tensorPort('in0')], outputs: [tensorPort('out0')], bindings: {},
  portMap: { inputs: [], outputs: [], params: [] }, // filled per-case
})

const inputNode = () =>
  mkNode('inp', { kind: 'input', name: 'Input', module: '__builtin__', ctor: [], inputs: [], outputs: [tensorPort('out')], bindings: {} },
    { name: 'x', shape: '1 3 8 8', dtype: 'float' }, null)

let failures = 0
const classIdx = (code: string, cls: string) => code.indexOf(`class ${cls}(`)
const countClass = (code: string, cls: string) =>
  (code.match(new RegExp(`class ${cls}\\(`, 'g')) || []).length
const hasRepeatedKwarg = (code: string) =>
  code.split('\n').some((line) => {
    const kwargs = [...line.matchAll(/(\b\w+)=/g)].map((x) => x[1])
    return kwargs.some((k, i) => kwargs.indexOf(k) !== i)
  })

function check(name: string, cond: boolean, code?: string) {
  if (cond) { console.log('  ok ', name); return }
  failures++
  console.error('  FAIL', name)
  if (code) console.error('--- code ---\n' + code + '\n------------')
}

// ---------------------------------------------------------------------------
// Case A - 2-level nesting: Outer { Inner { Conv } }
// ---------------------------------------------------------------------------
{
  const conv = mkNode('conv', leafEntry('Conv'), { out_ch: 16 }, 'g_in')
  const inner = mkNode('fac_in', facadeEntry('g_in', 'Inner'), {}, 'g_out') // nested!
  ;(inner.entry as any).portMap = {
    inputs: [{ facadePort: 'in0', childNodeId: 'conv', childPort: 'x', shape: ['...'] }],
    outputs: [{ facadePort: 'out0', childNodeId: 'conv', childPort: 'out', shape: ['...'] }],
    params: [],
  }
  const outer = mkNode('fac_out', facadeEntry('g_out', 'Outer'), {}, null) // top-level
  ;(outer.entry as any).portMap = {
    inputs: [{ facadePort: 'in0', childNodeId: 'fac_in', childPort: 'in0', shape: ['...'] }],
    outputs: [{ facadePort: 'out0', childNodeId: 'fac_in', childPort: 'out0', shape: ['...'] }],
    params: [],
  }
  const inp = inputNode()
  const nodes = [inp, outer, inner, conv]
  const connections = [
    { source: 'inp', sourceOutput: 'out', target: 'fac_out', targetInput: 'in0' },
    // inner edge of g_in: inner facade boundary feeds conv (handled by subgraph view)
    // inner edge of g_out: outer boundary feeds inner facade (handled by subgraph view)
  ]
  const code = generate(nodes as any, connections as any, 'pytorch')
  console.log('Case A - 2-level nesting')
  check('Inner defined before Outer', classIdx(code, 'Inner') >= 0 && classIdx(code, 'Inner') < classIdx(code, 'Outer'), code)
  check('Outer __init__ instantiates Inner', /self\.\w+ = Inner\(/.test(code), code)
  check('Inner __init__ instantiates Conv', /self\.\w+ = Conv\(/.test(code), code)
  check('GeneratedModel instantiates Outer', /self\.\w+ = Outer\(/.test(code), code)
  check('no repeated kwargs', !hasRepeatedKwarg(code), code)
  check('Outer forward calls inner member', /class Outer[\s\S]*self\.\w+\(/.test(code), code)
}

// ---------------------------------------------------------------------------
// Case B - 3-level nesting: A { B { C { Conv } } }
// ---------------------------------------------------------------------------
{
  const conv = mkNode('conv', leafEntry('Conv'), { out_ch: 16 }, 'g_c')
  const cf = mkNode('fac_c', facadeEntry('g_c', 'C'), {}, 'g_b')
  ;(cf.entry as any).portMap = {
    inputs: [{ facadePort: 'in0', childNodeId: 'conv', childPort: 'x', shape: ['...'] }],
    outputs: [{ facadePort: 'out0', childNodeId: 'conv', childPort: 'out', shape: ['...'] }], params: [],
  }
  const bf = mkNode('fac_b', facadeEntry('g_b', 'B'), {}, 'g_a')
  ;(bf.entry as any).portMap = {
    inputs: [{ facadePort: 'in0', childNodeId: 'fac_c', childPort: 'in0', shape: ['...'] }],
    outputs: [{ facadePort: 'out0', childNodeId: 'fac_c', childPort: 'out0', shape: ['...'] }], params: [],
  }
  const af = mkNode('fac_a', facadeEntry('g_a', 'A'), {}, null)
  ;(af.entry as any).portMap = {
    inputs: [{ facadePort: 'in0', childNodeId: 'fac_b', childPort: 'in0', shape: ['...'] }],
    outputs: [{ facadePort: 'out0', childNodeId: 'fac_b', childPort: 'out0', shape: ['...'] }], params: [],
  }
  const inp = inputNode()
  const nodes = [inp, af, bf, cf, conv]
  const connections = [{ source: 'inp', sourceOutput: 'out', target: 'fac_a', targetInput: 'in0' }]
  const code = generate(nodes as any, connections as any, 'pytorch')
  console.log('Case B - 3-level nesting')
  check('class order C < B < A', classIdx(code, 'C') < classIdx(code, 'B') && classIdx(code, 'B') < classIdx(code, 'A'), code)
  check('A instantiates B', /self\.\w+ = B\(/.test(code), code)
  check('B instantiates C', /self\.\w+ = C\(/.test(code), code)
  check('C instantiates Conv', /self\.\w+ = Conv\(/.test(code), code)
  check('GeneratedModel instantiates A', /self\.\w+ = A\(/.test(code), code)
  check('no repeated kwargs', !hasRepeatedKwarg(code), code)
}

// ---------------------------------------------------------------------------
// Case C - group containing ONLY another group (no sibling leaves)
// ---------------------------------------------------------------------------
{
  const conv = mkNode('conv', leafEntry('Conv'), { out_ch: 16 }, 'g_in')
  const inner = mkNode('fac_in', facadeEntry('g_in', 'Inner'), {}, 'g_out')
  ;(inner.entry as any).portMap = {
    inputs: [{ facadePort: 'in0', childNodeId: 'conv', childPort: 'x', shape: ['...'] }],
    outputs: [{ facadePort: 'out0', childNodeId: 'conv', childPort: 'out', shape: ['...'] }], params: [],
  }
  const outer = mkNode('fac_out', facadeEntry('g_out', 'OnlyWrap'), {}, null)
  ;(outer.entry as any).portMap = {
    inputs: [{ facadePort: 'in0', childNodeId: 'fac_in', childPort: 'in0', shape: ['...'] }],
    outputs: [{ facadePort: 'out0', childNodeId: 'fac_in', childPort: 'out0', shape: ['...'] }], params: [],
  }
  const inp = inputNode()
  const code = generate([inp, outer, inner, conv] as any,
    [{ source: 'inp', sourceOutput: 'out', target: 'fac_out', targetInput: 'in0' }] as any, 'pytorch')
  console.log('Case C - group wrapping only a group')
  check('OnlyWrap instantiates Inner', /self\.\w+ = Inner\(/.test(code), code)
  check('OnlyWrap body has no bare pass', !/class OnlyWrap[\s\S]*?def forward[\s\S]*?\n        pass\n/.test(code), code)
  check('no repeated kwargs', !hasRepeatedKwarg(code), code)
}

// ---------------------------------------------------------------------------
// Case D - same-name dedup with nesting (two Outers, each wraps an Inner)
// ---------------------------------------------------------------------------
{
  const mkStack = (sfx: string) => {
    const conv = mkNode(`conv_${sfx}`, leafEntry('Conv'), { out_ch: 16 }, `g_in_${sfx}`)
    const inner = mkNode(`fac_in_${sfx}`, facadeEntry(`g_in_${sfx}`, 'Inner'), {}, `g_out_${sfx}`)
    ;(inner.entry as any).portMap = {
      inputs: [{ facadePort: 'in0', childNodeId: `conv_${sfx}`, childPort: 'x', shape: ['...'] }],
      outputs: [{ facadePort: 'out0', childNodeId: `conv_${sfx}`, childPort: 'out', shape: ['...'] }], params: [],
    }
    const outer = mkNode(`fac_out_${sfx}`, facadeEntry(`g_out_${sfx}`, 'Outer'), {}, null)
    ;(outer.entry as any).portMap = {
      inputs: [{ facadePort: 'in0', childNodeId: `fac_in_${sfx}`, childPort: 'in0', shape: ['...'] }],
      outputs: [{ facadePort: 'out0', childNodeId: `fac_in_${sfx}`, childPort: 'out0', shape: ['...'] }], params: [],
    }
    return { conv, inner, outer }
  }
  const a = mkStack('a'), b = mkStack('b')
  const inp = inputNode()
  const nodes = [inp, a.outer, a.inner, a.conv, b.outer, b.inner, b.conv]
  const connections = [
    { source: 'inp', sourceOutput: 'out', target: 'fac_out_a', targetInput: 'in0' },
    { source: 'fac_out_a', sourceOutput: 'out0', target: 'fac_out_b', targetInput: 'in0' },
  ]
  const code = generate(nodes as any, connections as any, 'pytorch')
  console.log('Case D - same-name dedup with nesting')
  check('class Inner defined exactly once', countClass(code, 'Inner') === 1, code)
  check('class Outer defined exactly once', countClass(code, 'Outer') === 1, code)
  check('both Outer instances in GeneratedModel', (code.match(/= Outer\(/g) || []).length === 2, code)
  check('Inner before Outer', classIdx(code, 'Inner') < classIdx(code, 'Outer'), code)
  check('no repeated kwargs', !hasRepeatedKwarg(code), code)
}

// ---------------------------------------------------------------------------
// Case E - a ctor param exposed THROUGH a nested group keeps its real default.
// Regression for "weight of size [0, 128, 3, 3]": an inner group `Style` owns a
// ConvBlock whose `out_ch` is exposed and set to 256; that param is re-exposed
// on the outer `DecoderWithStyle`. The outer __init__ default must resolve
// recursively to 256, NOT fall back to 0 (which yields a zero-sized weight).
{
  const convEntry = () => ({ kind: 'module', name: 'ConvBlock', module: 'm',
    ctor: [{ name: 'in_ch', type: 'int', default: 0 }, { name: 'out_ch', type: 'int', default: 0 }],
    inputs: [tensorPort('x')], outputs: [tensorPort('out')], bindings: {} })
  const withParam = (node: any, p: string) => {
    node.inputs[`__param__${p}`] = { portSpec: { kind: 'param', paramName: p, paramType: 'int' } }
    return node
  }
  const constEntry = { kind: 'const', name: 'Constant', module: 'm', ctor: [], inputs: [], outputs: [tensorPort('out')], bindings: {} }

  const c = mkNode('c256', constEntry, { value_type: 'int', value: '256' }, 'g_style')
  const sconv = withParam(mkNode('sconv', convEntry(), { in_ch: 128, out_ch: 256 }, 'g_style'), 'out_ch')
  const styleFac = withParam(mkNode('styleFac', facadeEntry('g_style', 'Style'), {}, 'g_outer'), 'out_ch')
  ;(styleFac.entry as any).portMap = {
    inputs: [{ facadePort: 'in0', childNodeId: 'sconv', childPort: 'x', shape: ['...'] }],
    outputs: [{ facadePort: 'out0', childNodeId: 'sconv', childPort: 'out', shape: ['...'] }],
    params: [{ facadePort: '__param__out_ch', childNodeId: 'sconv', childPort: '__param__out_ch', paramName: 'out_ch', paramType: 'int' }],
  }
  const outerFac = withParam(mkNode('outerFac', facadeEntry('g_outer', 'DecoderWithStyle'), {}, null), 'out_ch')
  ;(outerFac.entry as any).portMap = {
    inputs: [{ facadePort: 'in0', childNodeId: 'styleFac', childPort: 'in0', shape: ['...'] }],
    outputs: [{ facadePort: 'out0', childNodeId: 'styleFac', childPort: 'out0', shape: ['...'] }],
    params: [{ facadePort: '__param__out_ch', childNodeId: 'styleFac', childPort: '__param__out_ch', paramName: 'out_ch', paramType: 'int' }],
  }
  const inp = mkNode('inp', { kind: 'input', name: 'Input', module: '__builtin__', ctor: [], inputs: [], outputs: [tensorPort('out')], bindings: {} },
    { name: 'x', shape: '1 128 8 8', dtype: 'float' }, null)
  const code = generate(
    [inp, outerFac, styleFac, sconv, c] as any,
    [
      { source: 'inp', sourceOutput: 'out', target: 'outerFac', targetInput: 'in0' },
      { source: 'c256', sourceOutput: 'out', target: 'sconv', targetInput: '__param__out_ch' },
    ] as any,
    'pytorch'
  )
  console.log('Case E - param exposed through a nested group keeps its default')
  check('Style defaults out_ch to 256', /class Style[\s\S]*def __init__\(self, out_ch: int = 256\)/.test(code), code)
  check('DecoderWithStyle defaults out_ch to 256 (not 0)', /class DecoderWithStyle[\s\S]*def __init__\(self, out_ch: int = 256\)/.test(code), code)
  check('no out_ch=0 anywhere', !/out_ch\s*=\s*0\b/.test(code), code)
  check('no repeated kwargs', !hasRepeatedKwarg(code), code)
}

// ---------------------------------------------------------------------------
// Case F - outer re-exposes the SAME param from two distinct nested children
// (mirrors `DecoderWithStyle` wrapping two `Style` instances). The recursive
// default must resolve for each, and the outer instantiation must emit the
// kwarg once (dedup), with no out_ch=0 leaking to either inner conv.
{
  const convEntry = () => ({ kind: 'module', name: 'ConvBlock', module: 'm',
    ctor: [{ name: 'in_ch', type: 'int', default: 0 }, { name: 'out_ch', type: 'int', default: 0 }],
    inputs: [tensorPort('x')], outputs: [tensorPort('out')], bindings: {} })
  const withParam = (node: any, p: string) => {
    node.inputs[`__param__${p}`] = { portSpec: { kind: 'param', paramName: p, paramType: 'int' } }
    return node
  }
  const mkStyle = (sfx: string) => {
    const sconv = withParam(mkNode(`sconv_${sfx}`, convEntry(), { in_ch: 128, out_ch: 256 }, `g_style_${sfx}`), 'out_ch')
    const styleFac = withParam(mkNode(`styleFac_${sfx}`, facadeEntry(`g_style_${sfx}`, 'Style'), {}, 'g_outer'), 'out_ch')
    ;(styleFac.entry as any).portMap = {
      inputs: [{ facadePort: 'in0', childNodeId: `sconv_${sfx}`, childPort: 'x', shape: ['...'] }],
      outputs: [{ facadePort: 'out0', childNodeId: `sconv_${sfx}`, childPort: 'out', shape: ['...'] }],
      params: [{ facadePort: '__param__out_ch', childNodeId: `sconv_${sfx}`, childPort: '__param__out_ch', paramName: 'out_ch', paramType: 'int' }],
    }
    return { sconv, styleFac }
  }
  const a = mkStyle('a'), b = mkStyle('b')
  const outerFac = withParam(mkNode('outerFac', facadeEntry('g_outer', 'DecoderWithStyle'), {}, null), 'out_ch')
  ;(outerFac.entry as any).portMap = {
    inputs: [{ facadePort: 'in0', childNodeId: 'styleFac_a', childPort: 'in0', shape: ['...'] }],
    outputs: [{ facadePort: 'out0', childNodeId: 'styleFac_b', childPort: 'out0', shape: ['...'] }],
    params: [
      { facadePort: '__param__out_ch', childNodeId: 'styleFac_a', childPort: '__param__out_ch', paramName: 'out_ch', paramType: 'int' },
      { facadePort: '__param__out_ch', childNodeId: 'styleFac_b', childPort: '__param__out_ch', paramName: 'out_ch', paramType: 'int' },
    ],
  }
  const inp = mkNode('inp', { kind: 'input', name: 'Input', module: '__builtin__', ctor: [], inputs: [], outputs: [tensorPort('out')], bindings: {} },
    { name: 'x', shape: '1 128 8 8', dtype: 'float' }, null)
  const code = generate(
    [inp, outerFac, a.styleFac, a.sconv, b.styleFac, b.sconv] as any,
    [{ source: 'inp', sourceOutput: 'out', target: 'outerFac', targetInput: 'in0' }] as any,
    'pytorch'
  )
  console.log('Case F - same param re-exposed from two nested children')
  check('DecoderWithStyle defaults out_ch to 256', /class DecoderWithStyle[\s\S]*def __init__\(self, out_ch: int = 256\)/.test(code), code)
  check('outer instantiation emits out_ch once', (code.match(/= Style\(out_ch=out_ch\)/g) || []).length === 2, code)
  check('no out_ch=0 anywhere', !/out_ch\s*=\s*0\b/.test(code), code)
  check('no repeated kwargs', !hasRepeatedKwarg(code), code)
}

if (failures > 0) { console.error(`\n${failures} assertion(s) failed`); process.exit(1) }
console.log('\nall nested-group codegen assertions passed')
