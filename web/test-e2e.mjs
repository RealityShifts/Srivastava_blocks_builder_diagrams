// End-to-end smoke test: load the editor, build a small graph via the
// window.__blocks test harness, exercise the validator, and check codegen.
import puppeteer from 'puppeteer'

const URL = process.env.URL || 'http://127.0.0.1:5174/'

const browser = await puppeteer.launch({
  headless: 'new',
  args: ['--no-sandbox'],
})
const page = await browser.newPage()
page.setDefaultTimeout(10000)

const consoleErrors = []
page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`))
page.on('console', (m) => {
  if (m.type() === 'error') {
    const text = m.text()
    if (text.includes('favicon')) return
    consoleErrors.push(`console.error: ${text}`)
  }
})

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

try {
  await page.goto(URL, { waitUntil: 'networkidle0' })
  await page.waitForSelector('#palette .block-item')
  await page.waitForFunction(() => !!window.__blocks)

  const paletteCount = await page.$$eval('#palette .block-item', (els) => els.length)
  check('palette has >=100 blocks', paletteCount >= 100, paletteCount)

  // ---- valid graph: ConvBlock(3->16) -> ConvBlock(16->32) ----
  let nodeIds = await page.evaluate(async () => {
    await window.__blocks.clearGraph()
    await window.__blocks.createNode('ConvBlock')
    await window.__blocks.createNode('ConvBlock')
    const nodes = window.__blocks.editor.getNodes()
    nodes[0].values.in_ch = 3
    nodes[0].values.out_ch = 16
    nodes[1].values.in_ch = 16
    nodes[1].values.out_ch = 32
    const [a, b] = nodes
    await window.__blocks.addConnection(a.id, 'out', b.id, 'x')
    window.__blocks.runValidation()
    return [a.id, b.id]
  })
  check('two conv nodes created', nodeIds.length === 2)

  let diag = await page.$$eval('#diag-list li', (els) =>
    els.map((e) => ({ cls: e.className, text: e.textContent }))
  )
  console.log('  diag (valid):', diag)
  check(
    'valid graph reports no errors',
    diag.every((d) => !d.cls.includes('err'))
  )

  // ---- shape conflict: ConvBlock(3->16) -> ConvBlock(64->32) ----
  // The validator pipe should refuse the bad edge entirely.
  const conflictResult = await page.evaluate(async () => {
    await window.__blocks.clearGraph()
    await window.__blocks.createNode('ConvBlock')
    await window.__blocks.createNode('ConvBlock')
    const nodes = window.__blocks.editor.getNodes()
    nodes[0].values.in_ch = 3
    nodes[0].values.out_ch = 16
    nodes[1].values.in_ch = 64  // <-- mismatch
    nodes[1].values.out_ch = 32
    const [a, b] = nodes
    let added = false
    try {
      await window.__blocks.addConnection(a.id, 'out', b.id, 'x')
      added = true
    } catch (e) {
      /* expected */
    }
    window.__blocks.runValidation()
    return {
      added,
      connections: window.__blocks.editor.getConnections().length,
    }
  })
  console.log('  mismatch result:', conflictResult)
  check('mismatched edge is rejected (no connection added)', conflictResult.connections === 0)

  // ---- codegen on the valid graph ----
  await page.evaluate(async () => {
    await window.__blocks.clearGraph()
    await window.__blocks.createNode('ConvBlock')
    await window.__blocks.createNode('ConvBlock')
    const nodes = window.__blocks.editor.getNodes()
    nodes[0].values.in_ch = 3
    nodes[0].values.out_ch = 16
    nodes[1].values.in_ch = 16
    nodes[1].values.out_ch = 32
    const [a, b] = nodes
    await window.__blocks.addConnection(a.id, 'out', b.id, 'x')
  })
  await page.click('#codegen-btn')
  await page.waitForSelector('#codegen-output')
  const code = await page.$eval('#codegen-output', (el) => el.textContent)
  console.log('--- generated python ---')
  console.log(code)
  check('codegen imports ConvBlock', code.includes('from pytorch_blocks.core_blocks import ConvBlock'))
  check('codegen wires conv0 -> conv1', code.includes('self.conv_block_1(x=conv_block_0)'))
  check('codegen sets ctor args', code.includes('in_ch=3, out_ch=16'))
} catch (e) {
  console.error('TEST FAILED:', e.stack)
  fail++
} finally {
  await browser.close()
}

console.log(`\n${pass} pass, ${fail} fail`)
if (consoleErrors.length > 0) {
  console.error('--- runtime console errors ---')
  consoleErrors.forEach((e) => console.error(e))
}
process.exit(fail === 0 && consoleErrors.length === 0 ? 0 : 1)
