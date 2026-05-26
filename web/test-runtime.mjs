// Integration test: traced codegen + HTTP shape runner.
import puppeteer from 'puppeteer'
import { generate } from './src/codegen.js'
import { RUNNER_URL } from './src/runtime.js'

const RUNNER = RUNNER_URL.replace('/run', '')
const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] })
const page = await browser.newPage()

let pass = 0,
  fail = 0
const check = (name, cond, info) => {
  if (cond) {
    pass++
    console.log(`  ok  ${name}`)
  } else {
    fail++
    console.log(`  FAIL ${name}`, info ?? '')
  }
}

try {
  // 1. trace codegen emits _runtime_shapes
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
  const inputEntry = {
    name: 'Input',
    module: '__builtin__',
    kind: 'input',
    ctor: [],
    inputs: [],
    outputs: [{ name: 'out', shape: ['B', 'C', 'H', 'W'], dtype: 'float', optional: false, variadic: false }],
    bindings: {},
  }
  const nIn = { id: 'in1', entry: inputEntry, values: { name: 'x', shape: '2 3 32 32', dtype: 'float' }, freshenedShape: () => [2, 3, 32, 32] }
  const nConv = {
    id: 'c1',
    entry: conv,
    values: { in_ch: 3, out_ch: 16 },
    freshenedShape(port, side) {
      return side === 'in' ? ['2', 'C_in#c1', '32', '32'] : ['2', 'C_out#c1', 'H_out#c1', 'W_out#c1']
    },
  }
  const code = generate([nIn, nConv], [{ source: 'in1', sourceOutput: 'out', target: 'c1', targetInput: 'x' }], 'pytorch', { trace: true })
  check('trace codegen returns _runtime_shapes', code.includes('return _runtime_shapes'))
  check('trace codegen records conv output', code.includes('_runtime_shapes["c1/out"]'))

  // 2. HTTP runner (requires python tools/shape_runner.py)
  const res = await fetch(RUNNER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      framework: 'pytorch',
      code,
      inputs: [{ arg: 'x', shape: [2, 3, 32, 32], dtype: 'float' }],
    }),
  })
  const body = await res.json()
  check('runner HTTP 200', res.ok, body)
  check('conv output spatial dims computed', body.shapes?.['c1/out']?.join(' ') === '2 16 32 32', body.shapes)

  // 3. UI button path via __blocks
  await page.goto(process.env.URL || 'http://127.0.0.1:5173/', { waitUntil: 'networkidle0' })
  await page.waitForFunction(() => !!window.__blocks, { timeout: 3000 }).catch(() => null)
  if (await page.evaluate(() => !!window.__blocks)) {
    await page.evaluate(async () => {
      await window.__blocks.clearGraph()
      await window.__blocks.createNode('Input')
      await window.__blocks.createNode('ConvBlock')
      const [inp, conv] = window.__blocks.editor.getNodes()
      inp.values.shape = '2 3 32 32'
      inp.values.name = 'x'
      conv.values.in_ch = 3
      conv.values.out_ch = 16
      await window.__blocks.addConnection(inp.id, 'out', conv.id, 'x')
      window.__blocks.runValidation()
    })
    await page.evaluate(() => window.__blocks.runRuntimeShapeCheck())
    await new Promise((r) => setTimeout(r, 3000))
    const rt = await page.evaluate(() => ({
      size: window.__blocks.state.runtimeShapes?.size,
      conv: window.__blocks.state.runtimeShapes?.get(
        `${window.__blocks.editor.getNodes()[1].id}/out`
      ),
      err: window.__blocks.state.runtimeError,
    }))
    check('UI runtime check captured conv shape', rt.conv?.join(' ') === '2 16 32 32', rt)
  } else {
    console.log('  skip UI test (dev server not running on 5173)')
  }
} catch (e) {
  console.error('FAIL:', e.message)
  fail++
} finally {
  await browser.close()
}

console.log(`\n${pass} pass, ${fail} fail`)
process.exit(fail === 0 ? 0 : 1)
