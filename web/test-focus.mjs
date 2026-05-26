// Reproduce the focus-loss bug and verify the incremental-render fix:
//   - select an Input node
//   - focus its `shape` input
//   - type several characters in a row (each triggers validation)
//   - confirm focus is preserved and the full string lands in the field
//   - confirm a switch to a different node DOES rebuild the inspector
import puppeteer from 'puppeteer'

const URL = process.env.URL || 'http://127.0.0.1:5175/'
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

  // Spawn an Input node and pick it so the inspector shows its controls.
  await page.evaluate(async () => {
    await window.__blocks.clearGraph()
    await window.__blocks.createNode('Input')
  })
  await page.waitForSelector('#inspector input')

  // Find the shape input. It's the second text input on an Input node
  // (name, shape, dtype-select). We locate by placeholder-less type=text and
  // the value matching the default 'B C H W'.
  const shapeSel = await page.evaluate(() => {
    const inputs = [...document.querySelectorAll('#inspector input[type="text"]')]
    for (let i = 0; i < inputs.length; i++) {
      if (inputs[i].value.includes('B C H W')) {
        inputs[i].id = '__shape_test'
        return '#__shape_test'
      }
    }
    return null
  })
  check('found shape input', !!shapeSel)

  // Focus + clear + type a long string char-by-char. Each char triggers
  // an input event -> queueValidation -> runValidation -> refreshInspector.
  await page.click(shapeSel)
  await page.evaluate((sel) => {
    document.querySelector(sel).value = ''
  }, shapeSel)
  await page.focus(shapeSel)

  const target = 'B 64 32 32 D'
  for (const ch of target) {
    await page.keyboard.type(ch)
    // tiny delay so the 60ms validation debounce can elapse and force a
    // refreshInspector call between keystrokes.
    await new Promise((r) => setTimeout(r, 80))
  }

  // Wait one more debounce tick so the trailing validation runs.
  await new Promise((r) => setTimeout(r, 120))

  const state = await page.evaluate((sel) => {
    const el = document.querySelector(sel)
    return {
      value: el?.value,
      focused: document.activeElement === el,
    }
  }, shapeSel)
  console.log('  shape input state:', state)
  check('full string captured in shape input', state.value === target, state)
  check('shape input still focused after typing', state.focused, state)

  // Switching to a different node should still do a full rebuild.
  await page.evaluate(async () => {
    await window.__blocks.createNode('ConvBlock')
    const ns = window.__blocks.editor.getNodes()
    window.__blocks.state.selectedNodeId = ns[1].id
  })
  // Trigger an inspector refresh by running validation manually.
  await page.evaluate(() => window.__blocks.runValidation())
  await new Promise((r) => setTimeout(r, 100))
  const switched = await page.evaluate(() => {
    const header = document.querySelector('#inspector .header strong')
    return header?.textContent
  })
  check('switching nodes rebuilt the inspector header', switched === 'ConvBlock', switched)
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
