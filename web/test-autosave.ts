// Smoke test for localStorage-backed autosave.
//
// 1. Build a small graph (2 ConvBlocks wired).
// 2. Force a synchronous save (debounce-friendly: drive saveToStorage directly).
// 3. Reload the page.
// 4. Assert the graph rehydrated automatically (same node count, same wiring,
//    same ctor values, same Python codegen output).
import puppeteer from 'puppeteer'

const URL = process.env.URL || 'http://127.0.0.1:5174/'
const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] })
const page = await browser.newPage()
page.setDefaultTimeout(15000)

const consoleErrors = []
page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`))
page.on('console', (m) => {
  if (m.type() === 'error') {
    const t = m.text()
    if (t.includes('favicon')) return
    consoleErrors.push(`console.error: ${t}`)
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
    console.log(`  FAIL ${name}`, info ?? '')
  }
}

try {
  // Fresh start — kill any prior autosave so we don't restore on first load.
  await page.goto(URL, { waitUntil: 'networkidle0' })
  await page.waitForFunction(() => !!window.__blocks)
  await page.evaluate(() => localStorage.removeItem(window.__blocks.AUTOSAVE_KEY))

  // Build the graph.
  const built = await page.evaluate(async () => {
    await window.__blocks.clearGraph()
    const a = await window.__blocks.createNode('ConvBlock')
    const b = await window.__blocks.createNode('ConvBlock')
    a.values.in_ch = 3
    a.values.out_ch = 16
    b.values.in_ch = 16
    b.values.out_ch = 32
    await window.__blocks.addConnection(a.id, 'out', b.id, 'x')
    window.__blocks.runValidation()
    return {
      nodes: window.__blocks.editor.getNodes().length,
      conns: window.__blocks.editor.getConnections().length,
    }
  })
  check('built 2-node graph with 1 connection', built.nodes === 2 && built.conns === 1, built)

  // Force a synchronous save (instead of waiting the 800ms debounce).
  const saved = await page.evaluate(() => {
    window.__blocks.saveToStorage()
    const raw = localStorage.getItem(window.__blocks.AUTOSAVE_KEY)
    return JSON.parse(raw)
  })
  check('autosave key written', !!saved && saved.version === 1, saved?.version)
  check('payload has 2 nodes', saved.graph?.nodes?.length === 2, saved.graph?.nodes?.length)
  check('payload has 1 connection', saved.graph?.connections?.length === 1)
  check(
    'payload preserves ctor values',
    saved.graph.nodes[0].values?.in_ch === 3 && saved.graph.nodes[1].values?.in_ch === 16,
    saved.graph.nodes.map((n) => n.values)
  )

  // Capture pre-reload codegen for byte-equal comparison after restore.
  const codeBefore = await page.evaluate(() => window.__blocks.runCodegen())

  // Reload the page — autosave should kick in during bootstrap.
  await page.reload({ waitUntil: 'networkidle0' })
  await page.waitForFunction(() => !!window.__blocks)
  // Give bootstrap time to finish restoreFromAutosave().
  await page.waitForFunction(
    () => window.__blocks.editor.getNodes().length === 2,
    { timeout: 5000 }
  )

  const restored = await page.evaluate(() => ({
    nodes: window.__blocks.editor.getNodes().length,
    conns: window.__blocks.editor.getConnections().length,
    values: window.__blocks.editor.getNodes().map((n) => ({
      name: n.entry.name,
      values: n.values,
    })),
  }))
  check('restored 2 nodes after reload', restored.nodes === 2, restored)
  check('restored 1 connection after reload', restored.conns === 1, restored)
  check(
    'restored ctor values intact',
    restored.values[0].values.in_ch === 3 && restored.values[1].values.in_ch === 16,
    restored.values
  )

  const codeAfter = await page.evaluate(() => window.__blocks.runCodegen())
  check('post-reload codegen matches pre-reload', codeBefore === codeAfter)

  // Edit a ctor value, fire the same path the inspector uses (queueAutosave
  // after mutating values), and wait past the 800ms debounce.
  await page.evaluate(() => {
    const n = window.__blocks.editor.getNodes()[1]
    n.values.out_ch = 64
    window.__blocks.queueAutosave()
  })
  await new Promise((r) => setTimeout(r, 1100))
  const persisted = await page.evaluate(() => {
    const raw = localStorage.getItem(window.__blocks.AUTOSAVE_KEY)
    return JSON.parse(raw).graph.nodes[1].values.out_ch
  })
  check('value edit re-saved via runValidation→autosave path', persisted === 64, persisted)
} catch (e) {
  console.error('TEST FAILED:', e.stack)
  fail++
} finally {
  await browser.close()
}

console.log(`\n${pass} pass, ${fail} fail`)
if (consoleErrors.length) {
  console.error('--- runtime console errors ---')
  consoleErrors.forEach((e) => console.error(e))
}
process.exit(fail === 0 && consoleErrors.length === 0 ? 0 : 1)
