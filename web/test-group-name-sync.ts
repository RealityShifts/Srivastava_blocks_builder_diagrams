// Structural node-sync must key on the group NAME, not only the tag. Codegen
// emits one class per group name (same-name groups dedupe to a single class
// body), while the tag selects the weight-shared instance. So two groups that
// share a name but have different tags must still be kept structurally
// identical, otherwise the one generated class is wrong for the other
// instance. See groupsAreStructuralPeers / syncStructuralGroupPeers in main.js.
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

    // Build a chained group of n 128->128 ConvBlocks with the given name+tag.
    const built = []
    async function buildGroup(n, name, tag, x0) {
      const ns = []
      for (let i = 0; i < n; i++) {
        const c = await B.createNode('ConvBlock', { x: x0 + i * 180, y: 0 })
        Object.assign(c.values, { in_ch: 128, out_ch: 128 })
        ns.push(c)
      }
      for (let i = 0; i < n - 1; i++) await B.addConnection(ns[i].id, 'out', ns[i + 1].id, 'x')
      await B.groupNodes(ns.map((c) => c.id))
      const facade = B.editor
        .getNodes()
        .find((f) => f.entry?.kind === 'group' && !built.includes(f.entry.groupId))
      const gid = facade.entry.groupId
      built.push(gid)
      if (tag) B.applyNodeTag(facade, tag)
      const g = B.state.groups.get(gid)
      g.name = name
      facade.entry.name = name
      return gid
    }
    const childCount = (gid) =>
      B.editor.getNodes().filter((n) => n.groupId === gid && n.entry?.kind !== 'group').length

    // 1) same NAME, different TAG -> sync from the fuller one.
    await B.clearGraph()
    const gA = await buildGroup(3, 'SelfAttn', 'facial', 0)
    const gB = await buildGroup(2, 'SelfAttn', 'body', 3000)
    const sameNameBefore = { gA: childCount(gA), gB: childCount(gB) }
    await B.syncStructuralGroupPeers(gB) // canonical should be gA (3 children)
    const sameNameAfter = { gA: childCount(gA), gB: childCount(gB) }

    // 2) different name AND different tag -> must NOT sync.
    await B.clearGraph()
    const gC = await buildGroup(3, 'Encoder', 'enc', 0)
    const gD = await buildGroup(2, 'Decoder', 'dec', 3000)
    await B.syncStructuralGroupPeers(gD)
    const unrelated = { gC: childCount(gC), gD: childCount(gD) }

    // 3) renaming a group to match another's name triggers a sync.
    await B.clearGraph()
    const gE = await buildGroup(3, 'BlockX', 'tagx', 0)
    const gF = await buildGroup(2, 'Temp', 'tagf', 3000)
    const renameBefore = childCount(gF)
    B.state.groups.get(gF).name = 'BlockX'
    const fFacade = B.editor.getNodes().find((n) => n.entry?.groupId === gF)
    if (fFacade) fFacade.entry.name = 'BlockX'
    await B.syncStructuralGroupPeers(gF)
    const renameAfter = childCount(gF)

    return { sameNameBefore, sameNameAfter, unrelated, renameBefore, renameAfter }
  })

  check('same-name groups start with different child counts', out.sameNameBefore.gA === 3 && out.sameNameBefore.gB === 2, out.sameNameBefore)
  check('same-name/different-tag groups sync to the fuller structure', out.sameNameAfter.gA === 3 && out.sameNameAfter.gB === 3, out.sameNameAfter)
  check('different-name groups are NOT synced', out.unrelated.gC === 3 && out.unrelated.gD === 2, out.unrelated)
  check('renaming to a shared name triggers a sync', out.renameBefore === 2 && out.renameAfter === 3, { before: out.renameBefore, after: out.renameAfter })

  console.log(`\n${pass} pass, ${fail} fail`)
  await browser.close()
  process.exit(fail ? 1 : 0)
} catch (e) {
  console.error('TEST ERROR', e)
  await browser.close()
  process.exit(2)
}
