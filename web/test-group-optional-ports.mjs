// A group facade must keep its proxied child ports' optionality. An optional
// child input (e.g. an attention mask) that is left dangling becomes an
// optional facade input — so the generated subclass forward() gives it a
// `= None` default and the parent call omits it. Otherwise the call raises
// "forward() missing N required positional arguments". The flag is lost on the
// saved portMap, so the import round-trip is the path that regressed.
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

try {
  await page.goto(URL, { waitUntil: 'networkidle0' })
  await page.waitForFunction(() => !!window.__blocks)
  await page.evaluate(() => {
    localStorage.removeItem(window.__blocks.AUTOSAVE_KEY)
    localStorage.removeItem(window.__blocks.CLIPBOARD_KEY)
  })
  await page.reload({ waitUntil: 'networkidle0' })
  await page.waitForFunction(() => !!window.__blocks)

  const out = await page.evaluate(async () => {
    const B = window.__blocks
    await B.clearGraph()
    const inp = await B.createNode('Input', { x: -400, y: 0 })
    Object.assign(inp.values, { name: 'x', shape: '2 16 128', dtype: 'float' })
    // Two MHAs chained; query/key/value wired, mask left dangling -> the group
    // boundary surfaces the optional masks as optional facade inputs.
    const m1 = await B.createNode('MultiHeadAttention', { x: 0, y: 0 })
    const m2 = await B.createNode('MultiHeadAttention', { x: 300, y: 0 })
    for (const p of ['query', 'key', 'value']) await B.addConnection(inp.id, 'out', m1.id, p)
    for (const p of ['query', 'key', 'value']) await B.addConnection(m1.id, 'out', m2.id, p)
    await B.groupNodes([m1.id, m2.id])

    const optAtCreation = B.editor
      .getNodes()
      .find((n) => n.entry?.kind === 'group')
      .entry.inputs.map((p) => Boolean(p.optional))

    // Round-trip through export/import — this is where `optional` was dropped.
    await B.importGraph(B.getGraphData())
    const facade = B.editor.getNodes().find((n) => n.entry?.kind === 'group')
    const optAfterImport = facade.entry.inputs.map((p) => Boolean(p.optional))

    const code = B.runCodegen()
    const lines = code.split('\n')
    const groupForward = lines.find(
      (l) => l.includes('def forward') && /in0/.test(l)
    )
    const callSite = lines.find((l) => /=\s*self\.\w+\(in0=/.test(l))
    const hasTodo = lines.some((l) => l.includes('TODO'))
    return { optAtCreation, optAfterImport, groupForward, callSite, hasTodo }
  })

  // query is required; key/value/mask are optional on MultiHeadAttention.
  check('boundary marks the dangling optional ports optional at creation', out.optAtCreation.slice(1).every(Boolean) && out.optAtCreation[0] === false, out.optAtCreation)
  check('optional flag survives the import round-trip', JSON.stringify(out.optAfterImport) === JSON.stringify(out.optAtCreation), out.optAfterImport)
  check('subclass forward defaults the optional inputs to None', /in0[^=]*\)/.test(out.groupForward || '') === false && /= None/.test(out.groupForward || ''), out.groupForward)
  const in0Segment = (out.groupForward || '').split(',').find((s) => s.includes('in0')) || ''
  check('first (required) input has no default', Boolean(in0Segment) && !in0Segment.includes('= None'), in0Segment)
  check('parent call omits the optional inputs', Boolean(out.callSite) && !/in3=/.test(out.callSite), out.callSite)
  check('no dangling-required TODO emitted', out.hasTodo === false, out.hasTodo)

  console.log(`\n${pass} pass, ${fail} fail`)
  await browser.close()
  process.exit(fail ? 1 : 0)
} catch (e) {
  console.error('TEST ERROR', e)
  await browser.close()
  process.exit(2)
}
