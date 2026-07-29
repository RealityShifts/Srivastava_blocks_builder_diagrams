// Test that constants inside groups expose their params to the facade.
// Scenario: Group contains a Constant node. The group facade should expose
// the constant's __param__ ports so external values can wire into it.

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

  // Build: Input -> Constant -> (group) -> Output
  const ids = await page.evaluate(async () => {
    await window.__blocks.clearGraph()
    const input = await window.__blocks.createNode('Input')
    input.values.shape = 'B C H W'
    input.values.dtype = 'float'
    const c = await window.__blocks.createNode('Constant')
    c.values.value_type = 'int'
    c.values.value = '32'
    const out = await window.__blocks.createNode('Output')
    out.values.name = 'out'
    window.__blocks.runValidation()
    return { input: input.id, constant: c.id, out: out.id }
  })
  check('created Input, Constant, Output', !!ids.constant && !!ids.out)

  // Group just the Constant node.
  const grouped = await page.evaluate(async (constId) => {
    await window.__blocks.groupNodes([constId])
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
  }, ids.constant)
  check('one group created', grouped.groupCount === 1, grouped)
  check('facade has outputs', grouped.facadeOutputs.length > 0, grouped.facadeOutputs)

  // The critical test: the facade should expose __param__ ports for the constant's params.
  // Since Constant has params like 'value' and 'value_type', they should be exposed
  // as __param__value and __param__value_type on the facade so external nodes can wire values in.
  const facadeParams = await page.evaluate((facadeId) => {
    const facade = window.__blocks.editor.getNode(facadeId)
    if (!facade) return []
    return Object.keys(facade.inputs).filter((k) => k.startsWith('__param__'))
  }, grouped.facadeId)
  check('facade exposes const param ports', facadeParams.length > 0, facadeParams)
  check('facade exposes __param__value', facadeParams.includes('__param__value'), facadeParams)

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
