// Exposed group params bubble up into the containing class's __init__ and are
// threaded down, instead of being frozen at each group's default.
import { generate } from './src/codegen.ts'

let pass = 0, fail = 0
const check = (name: string, cond: boolean, info?: any) => {
  if (cond) { pass++; console.log('  ok ', name) }
  else { fail++; console.log('  FAIL', name, info ?? '') }
}

const tp = (name: string) => ({ name, shape: ['...'], dtype: 'any', optional: false, variadic: false })
const mk = (id: string, entry: any, values: any = {}, groupId: string | null = null) => {
  const inputs: any = Object.fromEntries((entry.inputs || []).map((p: any) => [p.name, { portSpec: p }]))
  for (const ep of entry.__exposed || []) inputs[`__param__${ep}`] = { portSpec: { kind: 'param', paramName: ep, paramType: 'int' } }
  return {
    id, entry, label: entry.name, tag: '', groupId, values,
    inputs, outputs: Object.fromEntries((entry.outputs || []).map((p: any) => [p.name, { portSpec: p }])),
    freshenedShape() { return null }, applyParamBindings() {},
  }
}
const leaf = (name: string, ctor: any[], exposed: string[] = []) => ({
  kind: 'module', name, module: 'm', ctor, inputs: [tp('x')], outputs: [tp('out')], bindings: {}, __exposed: exposed,
})
const fac = (gid: string, name: string, exposed: string[] = []) => ({
  kind: 'group', name, groupId: gid, module: '__group__', ctor: [],
  inputs: [tp('in0')], outputs: [tp('out0')], bindings: {}, __exposed: exposed,
  portMap: { inputs: [], outputs: [], params: [] },
})
const inputEntry = { kind: 'input', name: 'Input', module: '__builtin__', ctor: [], inputs: [], outputs: [tp('out')], bindings: {} }

// --- Outer "Net" contains a Decoder group that exposes `style_dim` (no local
//     Constant). It should bubble into Net.__init__ and pass down. ---
{
  const inner = mk('dec_inner', leaf('AdaIN', [{ name: 'style_dim', type: 'int', default: 256 }], ['style_dim']),
    { style_dim: 256 }, 'g_dec')
  const decFac = mk('decFac', fac('g_dec', 'Decoder', ['style_dim']), {}, null)
  ;(decFac.entry as any).portMap = {
    inputs: [{ facadePort: 'in0', childNodeId: 'dec_inner', childPort: 'x', shape: ['...'] }],
    outputs: [{ facadePort: 'out0', childNodeId: 'dec_inner', childPort: 'out', shape: ['...'] }],
    params: [{ facadePort: '__param__style_dim', childNodeId: 'dec_inner', childPort: '__param__style_dim', paramName: 'style_dim', paramType: 'int' }],
  }
  const inp = mk('inp', inputEntry, { name: 'x', shape: '1 64 8 8', dtype: 'float' })
  const code = generate([inp, decFac, inner] as any,
    [{ id: 'e', source: 'inp', sourceOutput: 'out', target: 'decFac', targetInput: 'in0' }] as any, 'pytorch')
  console.log('Single exposed param bubbles to top')
  check('GeneratedModel __init__ gains style_dim', /class GeneratedModel[\s\S]*def __init__\(self, style_dim: int = 256\)/.test(code), code)
  check('Decoder instantiated with style_dim=style_dim', /self\.\w+ = Decoder\(style_dim=style_dim\)/.test(code), code)
  check('Decoder class itself defaults style_dim=256', /class Decoder[\s\S]*def __init__\(self, style_dim: int = 256\)/.test(code), code)
}

// --- Two Style groups both exposing out_ch (=256) share ONE outer arg. ---
{
  const mkStyle = (sfx: string) => {
    const conv = mk(`conv_${sfx}`, leaf('ConvBlock', [{ name: 'out_ch', type: 'int', default: 8 }], ['out_ch']),
      { out_ch: 256 }, `g_${sfx}`)
    const f = mk(`fac_${sfx}`, fac(`g_${sfx}`, 'Style', ['out_ch']), {}, null)
    ;(f.entry as any).portMap = {
      inputs: [{ facadePort: 'in0', childNodeId: `conv_${sfx}`, childPort: 'x', shape: ['...'] }],
      outputs: [{ facadePort: 'out0', childNodeId: `conv_${sfx}`, childPort: 'out', shape: ['...'] }],
      params: [{ facadePort: '__param__out_ch', childNodeId: `conv_${sfx}`, childPort: '__param__out_ch', paramName: 'out_ch', paramType: 'int' }],
    }
    return { conv, f }
  }
  const a = mkStyle('a'), b = mkStyle('b')
  const inp = mk('inp', inputEntry, { name: 'x', shape: '1 128 8 8', dtype: 'float' })
  const code = generate([inp, a.f, a.conv, b.f, b.conv] as any,
    [{ id: 'e', source: 'inp', sourceOutput: 'out', target: 'fac_a', targetInput: 'in0' }] as any, 'pytorch')
  console.log('Same-named exposed params share one outer arg')
  check('one out_ch param in GeneratedModel __init__', (code.match(/class GeneratedModel[\s\S]*?def __init__\(self, out_ch: int = 256\)/)||[]).length === 1, code)
  check('both Styles get out_ch=out_ch', (code.match(/= Style\(out_ch=out_ch\)/g) || []).length === 2, code)
  check('no repeated kwargs', !code.split('\n').some((l) => { const k = [...l.matchAll(/(\b\w+)=/g)].map(x => x[1]); return k.some((x, i) => k.indexOf(x) !== i) }), code)
}

// --- A param driven by a LOCAL Constant wire is NOT bubbled (stays internal). ---
{
  const conv = mk('conv', leaf('ConvBlock', [{ name: 'out_ch', type: 'int', default: 8 }], ['out_ch']), { out_ch: 256 }, 'g1')
  const f = mk('fac', fac('g1', 'Style', ['out_ch']), {}, null)
  ;(f.entry as any).portMap = {
    inputs: [{ facadePort: 'in0', childNodeId: 'conv', childPort: 'x', shape: ['...'] }],
    outputs: [{ facadePort: 'out0', childNodeId: 'conv', childPort: 'out', shape: ['...'] }],
    params: [{ facadePort: '__param__out_ch', childNodeId: 'conv', childPort: '__param__out_ch', paramName: 'out_ch', paramType: 'int' }],
  }
  // wire __param__out_ch on the FACADE from a top-level constant
  const c = mk('c', { kind: 'const', name: 'Constant', module: 'm', ctor: [], inputs: [], outputs: [tp('out')], bindings: {} }, { value_type: 'int', value: '512' })
  const inp = mk('inp', inputEntry, { name: 'x', shape: '1 128 8 8', dtype: 'float' })
  const code = generate([inp, f, conv, c] as any, [
    { id: 'e1', source: 'inp', sourceOutput: 'out', target: 'fac', targetInput: 'in0' },
    { id: 'e2', source: 'c', sourceOutput: 'out', target: 'fac', targetInput: '__param__out_ch' },
  ] as any, 'pytorch')
  console.log('Locally-wired constant takes precedence over bubbling')
  check('Style gets out_ch from the constant param', /= Style\(out_ch=\w+\)/.test(code), code)
  check('constant default 512 present', /512/.test(code), code)
}

console.log(`\n${pass} pass, ${fail} fail`)
if (fail > 0) process.exit(1)
