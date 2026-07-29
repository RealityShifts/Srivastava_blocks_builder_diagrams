// Two children in one group exposing the SAME ctor param name (in_ch) must each
// keep their own facade port. Regression: facadePort was `__param__<paramName>`,
// so both collapsed onto one port and expand reattached only one constant.
import puppeteer from 'puppeteer'

const URL = process.env.URL || 'http://127.0.0.1:5173/'
const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] })
const page = await browser.newPage()
page.setDefaultTimeout(15000)

let pass = 0, fail = 0
const check = (name, cond, info) => {
  if (cond) { pass++; console.log(`  ok  ${name}`) }
  else { fail++; console.log(`  FAIL ${name}`, info ?? '') }
}

try {
  await page.goto(URL, { waitUntil: 'networkidle0' })
  await page.waitForFunction(() => !!window.__blocks)
  await page.evaluate(() => localStorage.removeItem(window.__blocks.AUTOSAVE_KEY))

  // Two ConvBlocks, each exposing in_ch, each fed by its own Constant.
  const built = await page.evaluate(async () => {
    const blocks = window.__blocks
    await blocks.clearGraph()
    const a = await blocks.createNode('ConvBlock')
    const b = await blocks.createNode('ConvBlock')
    a.exposeParam('in_ch')
    b.exposeParam('in_ch')
    const k1 = await blocks.createNode('Constant')
    k1.values.value = '3'
    const k2 = await blocks.createNode('Constant')
    k2.values.value = '16'
    await blocks.addConnection(k1.id, 'out', a.id, '__param__in_ch')
    await blocks.addConnection(k2.id, 'out', b.id, '__param__in_ch')
    return { a: a.id, b: b.id, k1: k1.id, k2: k2.id }
  })

  const wiredBefore = await page.evaluate((a, b) => {
    return window.__blocks.editor.getConnections()
      .filter((c) => c.targetInput === '__param__in_ch' && (c.target === a || c.target === b)).length
  }, built.a, built.b)
  check('both constants wired before grouping', wiredBefore === 2, wiredBefore)

  // Collapse both ConvBlocks into one group.
  const collapsed = await page.evaluate(async (a, b) => {
    const blocks = window.__blocks
    await blocks.groupNodes([a, b])
    await new Promise((r) => setTimeout(r, 40))
    const g = [...blocks.state.groups.values()][0]
    const facade = blocks.editor.getNodes()
      .find((n) => n.entry.kind === 'group' && n.entry.groupId === g.id)
    return {
      groupId: g.id,
      facadeParamPorts: Object.keys(facade.inputs).filter((k) => k.startsWith('__param__')),
      paramMappings: (g.portMap.params ?? []).length,
      edgesIntoFacadeParams: blocks.editor.getConnections()
        .filter((c) => c.target === facade.id && c.targetInput.startsWith('__param__')).length,
    }
  }, built.a, built.b)

  check('portMap has one mapping per child param', collapsed.paramMappings === 2, collapsed.paramMappings)
  check('facade has a DISTINCT port per child param',
    collapsed.facadeParamPorts.length === 2, collapsed.facadeParamPorts)
  check('both constant edges survive collapse',
    collapsed.edgesIntoFacadeParams === 2, collapsed.edgesIntoFacadeParams)

  // Expand and confirm BOTH constants reattach to their original child.
  const expanded = await page.evaluate(async (gid, a, b, k1, k2) => {
    const blocks = window.__blocks
    await blocks.expandGroup(gid)
    await new Promise((r) => setTimeout(r, 40))
    const conns = blocks.editor.getConnections()
      .filter((c) => c.targetInput === '__param__in_ch')
    return {
      count: conns.length,
      aFedByK1: conns.some((c) => c.target === a && c.source === k1),
      bFedByK2: conns.some((c) => c.target === b && c.source === k2),
    }
  }, collapsed.groupId, built.a, built.b, built.k1, built.k2)

  check('both param edges restored after expand', expanded.count === 2, expanded.count)
  check('constant k1 reattached to child a', expanded.aFedByK1, expanded)
  check('constant k2 reattached to child b', expanded.bFedByK2, expanded)

  console.log(`\n${pass} pass, ${fail} fail`)
} catch (e) {
  console.error('Test error:', e)
  fail++
} finally {
  await browser.close()
  process.exit(fail > 0 ? 1 : 0)
}
