// The inspector's same-node fast path must reconcile each param control's
// displayed value with node.values, which change programmatically while the
// node stays selected: paste/duplicate (values assigned after the node is
// already selected), implicit ctor back-fill (e.g. inferred in_ch), tag-sync.
// The control the user is actively editing is left alone. See renderInspector
// in ui.js.
import puppeteer from 'puppeteer'

const URL = process.env.URL || 'http://127.0.0.1:5173/'
const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] })
const page = await browser.newPage()
page.setDefaultTimeout(20000)

let pass = 0, fail = 0
const check = (name, cond, info) => {
  if (cond) { pass++; console.log(`  ok  ${name}`) }
  else { fail++; console.log(`  FAIL ${name}`, info === undefined ? '' : JSON.stringify(info)) }
}
const ctrlVal = (nodeId, param) =>
  page.$eval(`[id="ctrl-${nodeId}-${param}"]`, (el) => el.value).catch(() => null)

try {
  await page.goto(URL, { waitUntil: 'networkidle0' })
  await page.waitForFunction(() => !!window.__blocks)
  await page.evaluate(() => {
    localStorage.removeItem(window.__blocks.AUTOSAVE_KEY)
    localStorage.removeItem(window.__blocks.CLIPBOARD_KEY)
  })
  await page.reload({ waitUntil: 'networkidle0' })
  await page.waitForFunction(() => !!window.__blocks)

  // --- 1) duplicate reflects the source node's edited values -------------
  const ids = await page.evaluate(async () => {
    const B = window.__blocks
    await B.clearGraph()
    const a = await B.createNode('ConvBlock', { x: 0, y: 0 })
    Object.assign(a.values, { in_ch: 7, out_ch: 99, norm: 'batch' })
    B.state.selectedNodeId = a.id
    B.runValidation()
    B.copySelection()
    await B.pasteClipboard() // pasted node becomes selected
    const pasted = B.editor.getNodes().map((n) => n.id).find((id) => id !== a.id)
    return { a: a.id, pasted }
  })
  check('pasted control: out_ch reflects source', (await ctrlVal(ids.pasted, 'out_ch')) === '99')
  check('pasted control: in_ch reflects source', (await ctrlVal(ids.pasted, 'in_ch')) === '7')
  check('pasted control: norm reflects source', (await ctrlVal(ids.pasted, 'norm')) === 'batch')

  // --- 2) programmatic change refreshes while same node stays selected ---
  const sel = await page.evaluate(() => {
    const B = window.__blocks
    const n = B.editor.getNode(B.state.selectedNodeId)
    n.values.out_ch = 256
    B.runValidation() // same-node path
    return B.state.selectedNodeId
  })
  check('same-node control updates after programmatic change', (await ctrlVal(sel, 'out_ch')) === '256')

  // --- 3) the actively-edited control is not clobbered -------------------
  const editId = await page.evaluate(async () => {
    const B = window.__blocks
    await B.clearGraph()
    const a = await B.createNode('ConvBlock', { x: 0, y: 0 })
    Object.assign(a.values, { out_ch: 32 })
    B.state.selectedNodeId = a.id
    B.runValidation()
    return a.id
  })
  const editSel = `[id="ctrl-${editId}-out_ch"]`
  await page.focus(editSel)
  await page.$eval(editSel, (el) => { el.value = '' })
  await page.type(editSel, '64')
  await page.evaluate(() => window.__blocks.runValidation()) // background refresh while focused
  const edited = await page.$eval(editSel, (el) => ({ value: el.value, focused: el === document.activeElement }))
  check('focused control keeps its in-progress value', edited.value === '64', edited)
  check('focused control keeps focus across refresh', edited.focused === true, edited)

  console.log(`\n${pass} pass, ${fail} fail`)
  await browser.close()
  process.exit(fail ? 1 : 0)
} catch (e) {
  console.error('TEST ERROR', e)
  await browser.close()
  process.exit(2)
}
