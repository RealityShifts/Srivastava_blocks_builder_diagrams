// Smoke test for the Inspector "Info" tab.
// Verifies block_info.json loads, the tab strip renders, and the fetched
// Mermaid is converted to an inline <svg> for a representative block.
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
  await page.goto(URL, { waitUntil: 'networkidle0' })
  await page.waitForFunction(() => !!window.__blocks)

  // 1. block_info.json must be loaded.
  const blockInfoSize = await page.evaluate(() => window.__blocks.state.blockInfo.size)
  check('blockInfo loaded (>= 100 entries)', blockInfoSize >= 100, blockInfoSize)

  // 2. Create + select a ConvBlock (has a known upstream diagram).
  await page.evaluate(async () => {
    await window.__blocks.clearGraph()
    const n = await window.__blocks.createNode('ConvBlock')
    window.__blocks.state.selectedNodeId = n.id
    window.__blocks.runValidation()
  })

  // 3. Tab strip renders with two buttons.
  await page.waitForSelector('#inspector-body .tabs .tab-btn[data-tab="info"]')
  const tabCount = await page.$$eval('#inspector-body .tabs .tab-btn', (els) => els.length)
  check('inspector has 2 tabs', tabCount === 2, tabCount)

  // 4. Click Info tab.
  await page.click('#inspector-body .tabs .tab-btn[data-tab="info"]')
  await page.waitForFunction(
    () => {
      const p = document.querySelector('#inspector-body .tab-panel[data-tab="info"]')
      return p && !p.hidden
    }
  )

  // 5. Description + shapes lines populate.
  const desc = await page.$eval(
    '#inspector-body .info-panel .info-desc',
    (el) => el.textContent
  )
  check('description text present', desc?.length > 5, desc)

  const shapes = await page.$eval(
    '#inspector-body .info-panel .info-shapes code',
    (el) => el.textContent
  )
  check('shapes line present', /\(B/.test(shapes), shapes)

  // 6. Mermaid renders to inline <svg> (async; wait for it).
  await page.waitForSelector('#inspector-body .info-mermaid svg', { timeout: 12000 })
  const svgInfo = await page.$eval('#inspector-body .info-mermaid svg', (svg) => ({
    nodeCount: svg.querySelectorAll('.node, g.node').length,
    viewBox: svg.getAttribute('viewBox'),
  }))
  check('mermaid rendered to SVG with nodes', svgInfo.nodeCount >= 3, svgInfo)

  // 7. Switch back to Params tab; Info panel hides.
  await page.click('#inspector-body .tabs .tab-btn[data-tab="params"]')
  const infoHidden = await page.$eval(
    '#inspector-body .tab-panel[data-tab="info"]',
    (p) => p.hidden
  )
  check('Info panel hidden after switching back to Params', infoHidden)

  // 8. Missing-block fallback: pick a block that genuinely has no upstream entry.
  await page.evaluate(async () => {
    await window.__blocks.clearGraph()
    const entries = window.__blocks.state.entries
    const info = window.__blocks.state.blockInfo
    const target = entries.find((e) => !info.has(e.name))
    if (!target) throw new Error('every manifest entry has upstream info — test needs new pick')
    const n = await window.__blocks.createNode(target.name)
    window.__blocks.state.selectedNodeId = n.id
    window.__blocks.runValidation()
  })
  await page.click('#inspector-body .tabs .tab-btn[data-tab="info"]')
  const fallbackText = await page.$eval(
    '#inspector-body .tab-panel[data-tab="info"]',
    (p) => p.textContent
  )
  check('fallback shown for un-documented block', /No reference diagram/i.test(fallbackText), fallbackText.slice(0, 80))
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
