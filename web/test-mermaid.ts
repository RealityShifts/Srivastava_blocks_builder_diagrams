// Pure-node tests for the Mermaid export + Python-with-mermaid codegen.
import { graphToMermaid } from './src/mermaid/graphMermaid.ts'
import { generateWithMermaid } from './src/mermaid/codegenWithMermaid.ts'

let pass = 0, fail = 0
const check = (name: string, cond: boolean, info?: any) => {
  if (cond) { pass++; console.log('  ok ', name) }
  else { fail++; console.log('  FAIL', name, info ?? '') }
}

const tp = (name: string) => ({ name, shape: ['...'], dtype: 'any', optional: false, variadic: false })
const mk = (id: string, entry: any, values: any = {}, groupId: string | null = null, tag = '') => ({
  id, entry, label: entry.name, tag, groupId, values,
  inputs: Object.fromEntries((entry.inputs || []).map((p: any) => [p.name, { portSpec: p }])),
  outputs: Object.fromEntries((entry.outputs || []).map((p: any) => [p.name, { portSpec: p }])),
  freshenedShape() { return null }, applyParamBindings() {},
})
const conv = (name = 'ConvBlock') => ({ kind: 'module', name, module: 'm',
  ctor: [{ name: 'in_ch', type: 'int', default: 4 }, { name: 'out_ch', type: 'int', default: 8 }],
  inputs: [tp('x')], outputs: [tp('out')], bindings: {} })
const inputEntry = { kind: 'input', name: 'Input', module: '__builtin__', ctor: [], inputs: [], outputs: [tp('out')], bindings: {} }

// --- graphToMermaid: basic flowchart + colors + edges ---
{
  const inp = mk('inp', inputEntry, { name: 'x', shape: '1 4 8 8', dtype: 'float' })
  const a = mk('a', conv(), { in_ch: 4, out_ch: 8 }, null, 'down1')
  const nodes = [inp, a]
  const conns = [{ id: 'e1', source: 'inp', sourceOutput: 'out', target: 'a', targetInput: 'x' }]
  const mmd = graphToMermaid(nodes as any, conns as any, {
    colorOf: (n) => (n.id === 'a' ? 'hsl(137, 68%, 58%)' : null),
  })
  console.log('graphToMermaid basics')
  check('starts with flowchart', mmd.startsWith('flowchart TD'), mmd)
  check('renders input as stadium', /n_inp\(\["x"\]\)/.test(mmd), mmd)
  check('renders conv node', /n_a\["ConvBlock · down1"\]/.test(mmd), mmd)
  check('emits edge inp->a', /n_inp --> n_a/.test(mmd), mmd)
  // Mermaid needs hex fills (it can't parse hsl()'s commas), so hsl is converted.
  check('styles colored node with hex fill', /style n_a fill:#[0-9a-f]{6},/.test(mmd), mmd)
  check('does NOT emit raw hsl() in a style', !/style[^\n]*hsl\(/.test(mmd), mmd)
  check('no style for uncolored node', !/style n_inp /.test(mmd), mmd)
}

// --- param wires render dashed ---
{
  const c = mk('c', { kind: 'const', name: 'Constant', module: 'm', ctor: [], inputs: [], outputs: [tp('out')], bindings: {} }, { value_type: 'int', value: '8' })
  const a = mk('a', conv())
  const nodes = [c, a]
  const conns = [{ id: 'e', source: 'c', sourceOutput: 'out', target: 'a', targetInput: '__param__out_ch' }]
  const mmd = graphToMermaid(nodes as any, conns as any)
  console.log('param wires')
  check('param edge is dashed with label', /n_c -\. out_ch \.-> n_a/.test(mmd), mmd)
}

// --- collapsed group: facade shown, hidden children dropped ---
{
  const facade = mk('fac', { kind: 'group', name: 'Encoder', groupId: 'g1', module: '__group__',
    ctor: [], inputs: [tp('in0')], outputs: [tp('out0')], bindings: {}, portMap: { inputs: [], outputs: [], params: [] } })
  const child = mk('m1', conv(), {}, 'g1')
  const inp = mk('inp', inputEntry, { name: 'x' })
  const nodes = [inp, facade, child]
  const conns = [{ id: 'e', source: 'inp', sourceOutput: 'out', target: 'fac', targetInput: 'in0' }]
  const mmd = graphToMermaid(nodes as any, conns as any)
  console.log('collapsed group')
  check('facade rendered as subroutine', /n_fac\[\["Encoder"\]\]/.test(mmd), mmd)
  check('hidden child NOT rendered', !/n_m1/.test(mmd), mmd)
}

// --- generateWithMermaid: embeds per-class + module overview ---
{
  const inp = mk('inp', inputEntry, { name: 'x', shape: '1 4 8 8', dtype: 'float' })
  const a = mk('a', conv(), { in_ch: 4, out_ch: 8 })
  const out = mk('out', { kind: 'output', name: 'Output', module: '__builtin__', ctor: [], inputs: [tp('x')], outputs: [], bindings: {} }, { name: 'y' })
  const nodes = [inp, a, out]
  const conns = [
    { id: 'e1', source: 'inp', sourceOutput: 'out', target: 'a', targetInput: 'x' },
    { id: 'e2', source: 'a', sourceOutput: 'out', target: 'out', targetInput: 'x' },
  ]
  const code = generateWithMermaid(nodes as any, conns as any, 'pytorch', {
    mermaidForBlock: (name) => (name === 'GeneratedModel' ? 'flowchart TD\n  x --> y' : null),
  })
  console.log('generateWithMermaid')
  check('keeps __future__ import first', code.startsWith('from __future__ import annotations'), code.slice(0, 60))
  check('has module overview docstring', /Architecture overview:/.test(code), code.slice(0, 400))
  check('module overview has a mermaid fence', (code.match(/```mermaid/g) || []).length >= 1, code)
  check('class GeneratedModel got a docstring mermaid', /class GeneratedModel[\s\S]*?"""[\s\S]*?```mermaid/.test(code), code)
  check('still valid-looking: defines GeneratedModel', /class GeneratedModel\(nn\.Module\):/.test(code), code)
}

console.log(`\n${pass} pass, ${fail} fail`)
if (fail > 0) process.exit(1)
