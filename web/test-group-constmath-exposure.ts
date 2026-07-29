// Test that constmath nodes inside groups expose their params to the facade.
// ConstMath has params like 'op' and 'operand' that should be exposed.

import puppeteer from 'puppeteer'

const URL = process.env.URL || 'http://127.0.0.1:5173/'
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
  await page.goto(URL, { waitUntil: 'networkidle0' })
  await page.waitForFunction(() => !!window.__blocks)
  await page.evaluate(() => localStorage.removeItem(window.__blocks.AUTOSAVE_KEY))

  // Build: Constant -> ConstMath -> (group) -> Output
  const ids = await page.evaluate(async () => {
    await window.__blocks.clearGraph()
    const c = await window.__blocks.createNode('Constant')
    c.values.value_type = 'int'
    c.values.value = '64'
    const cm = await window.__blocks.createNode('ConstMath')
    cm.values.op = 'div'
    cm.values.operand = '2'
    const out = await window.__blocks.createNode('Output')
    out.values.name = 'result'
    await window.__blocks.addConnection(c.id, 'out', cm.id, 'x')
    window.__blocks.runValidation()
    return { const: c.id, constmath: cm.id, out: out.id }
  })
  check('created Constant, ConstMath, Output', !!ids.constmath && !!ids.out)

  // Group just the ConstMath node.
  const grouped = await page.evaluate(async (cmId) => {
    await window.__blocks.groupNodes([cmId])
    await new Promise((r) => setTimeout(r, 30))
    const groups = [...window.__blocks.state.groups.values()]
    const facade = groups[0]
      ? window.__blocks.editor
          .getNodes()
          .find((n) => n.entry.kind === 'group' && n.entry.groupId === groups[0].id)
      : null
    return {
      groupCount: groups.length,
      facadeId: facade?.id,
      facadeInputs: facade ? Object.keys(facade.inputs) : [],
      facadeOutputs: facade ? Object.keys(facade.outputs) : [],
    }
  }, ids.constmath)
  check('one group created', grouped.groupCount === 1, grouped)
  check('facade has inputs', grouped.facadeInputs.length > 0, grouped.facadeInputs)
  check('facade has outputs', grouped.facadeOutputs.length > 0, grouped.facadeOutputs)

  // The critical test: the facade should expose __param__ ports for constmath's params.
  // ConstMath has 'op' and 'operand' params that should be exposed.
  const facadeParams = await page.evaluate((facadeId) => {
    const facade = window.__blocks.editor.getNode(facadeId)
    if (!facade) return []
    return Object.keys(facade.inputs).filter((k) => k.startsWith('__param__'))
  }, grouped.facadeId)
  check('facade exposes constmath param ports', facadeParams.length > 0, facadeParams)
  check('facade exposes __param__op', facadeParams.includes('__param__op'), facadeParams)
  check('facade exposes __param__operand', facadeParams.includes('__param__operand'), facadeParams)

  console.log(`\nPassed: ${pass}, Failed: ${fail}`)
  if (consoleErrors.length > 0) {
    console.error('Console errors:', consoleErrors)
  }
} catch (e) {
  console.error('Test error:', e)
  fail++
} finally {
  await browser.close()
  process.exit(fail > 0 ? 1 : 0)
}
