// Test the new built-in Input node:
//  1. It appears in the palette
//  2. Its user-typed shape literals propagate through unification
//  3. A mismatch between Input's shape and a downstream block's ctor is rejected
//  4. Codegen emits the Input as a forward() arg with the user-typed name
import puppeteer from 'puppeteer'

const URL = process.env.URL || 'http://127.0.0.1:5174/'
const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] })
const page = await browser.newPage()
page.setDefaultTimeout(10000)

const errs = []
page.on('pageerror', (e) => errs.push(`pageerror: ${e.message}`))
page.on('console', (m) => {
  if (m.type() === 'error' && !m.text().includes('favicon')) {
    errs.push(`console.error: ${m.text()}`)
  }
})

let pass = 0,
  fail = 0
const check = (name, cond, info) => {
  if (cond) {
    pass++
    console.log(`  ok  ${name}`)
  } else {
    fail++
    console.log(`  FAIL ${name}`, JSON.stringify(info))
  }
}

try {
  await page.goto(URL, { waitUntil: 'networkidle0' })
  await page.waitForFunction(() => !!window.__blocks)

  // 1. Palette includes the built-in Input entry.
  const hasInput = await page.$eval('#palette', (root) =>
    [...root.querySelectorAll('.block-item')].some((el) => el.dataset.name === 'Input')
  )
  check('palette has built-in Input entry', hasInput)

  // 2. Shape literals from Input propagate to downstream ConvBlock and validation
  //    binds C_in via unification.
  await page.evaluate(async () => {
    await window.__blocks.clearGraph()
    await window.__blocks.createNode('Input')
    await window.__blocks.createNode('ConvBlock')
    const [inp, conv] = window.__blocks.editor.getNodes()
    inp.values.name = 'image'
    inp.values.shape = 'B 3 224 224'
    inp.values.dtype = 'float'
    conv.values.in_ch = 3
    conv.values.out_ch = 16
    await window.__blocks.addConnection(inp.id, 'out', conv.id, 'x')
    window.__blocks.runValidation()
  })

  const valid = await page.evaluate(() => {
    const r = window.__blocks.state.lastResult
    return { errors: r.errors.length, conns: window.__blocks.editor.getConnections().length }
  })
  check('valid Input(B,3,224,224) -> Conv(in=3): no errors', valid.errors === 0 && valid.conns === 1, valid)

  // 3. Mismatch: Input(B,3,...) -> Conv(in_ch=64) is refused by the validator pipe.
  const mismatch = await page.evaluate(async () => {
    await window.__blocks.clearGraph()
    await window.__blocks.createNode('Input')
    await window.__blocks.createNode('ConvBlock')
    const [inp, conv] = window.__blocks.editor.getNodes()
    inp.values.shape = 'B 3 224 224'
    conv.values.in_ch = 64
    try {
      await window.__blocks.addConnection(inp.id, 'out', conv.id, 'x')
    } catch {}
    return window.__blocks.editor.getConnections().length
  })
  check('mismatched Input(3) -> Conv(64) edge rejected', mismatch === 0, { mismatch })

  // 4. Codegen: Input node becomes a forward() arg with the sanitized name.
  await page.evaluate(async () => {
    await window.__blocks.clearGraph()
    await window.__blocks.createNode('Input')
    await window.__blocks.createNode('ConvBlock')
    const [inp, conv] = window.__blocks.editor.getNodes()
    inp.values.name = 'rgb image'  // has a space - must be sanitized
    inp.values.shape = 'B 3 224 224'
    conv.values.in_ch = 3
    conv.values.out_ch = 16
    await window.__blocks.addConnection(inp.id, 'out', conv.id, 'x')
  })
  const code = await page.evaluate(() => window.__blocks.runCodegen())
  console.log('--- generated python ---')
  console.log(code)

  check('codegen uses sanitized Input name as forward arg', code.includes('def forward(self, rgb_image):'))
  check('codegen wires Input arg into ConvBlock', code.includes('self.conv_block_1(x=rgb_image)'))
  check('codegen does NOT register Input as a self.* attribute', !code.includes('self.input'))
  check('codegen does NOT import a fake Input module', !code.includes('__builtin__'))
  check('codegen returns the conv output, not the raw input', code.includes('return conv_block_1'))

  // 5. Two Input nodes with colliding names get disambiguated.
  await page.evaluate(async () => {
    await window.__blocks.clearGraph()
    await window.__blocks.createNode('Input')
    await window.__blocks.createNode('Input')
    const [a, b] = window.__blocks.editor.getNodes()
    a.values.name = 'x'
    b.values.name = 'x'
  })
  const code2 = await page.evaluate(() => window.__blocks.runCodegen())
  console.log('--- two-Input codegen ---')
  console.log(code2)
  check('duplicate Input names are disambiguated', code2.includes('def forward(self, x, x2):'))
} catch (e) {
  console.error('FAIL:', e.stack)
  fail++
} finally {
  await browser.close()
}

console.log(`\n${pass} pass, ${fail} fail`)
if (errs.length) {
  console.error('--- runtime errors ---')
  errs.forEach((e) => console.error(e))
}
process.exit(fail === 0 && errs.length === 0 ? 0 : 1)
