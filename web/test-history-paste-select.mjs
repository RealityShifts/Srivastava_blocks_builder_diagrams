// Covers two editor UX features:
//   1. Paste/duplicate leaves exactly the newly-pasted nodes selected (the
//      originals are deselected; a collapsed group selects its facade, not the
//      hidden children).
//   2. Undo / redo (Ctrl+Z / Ctrl+Shift+Z) via snapshot history — including a
//      group operation, which a core rete-history plugin could not restore
//      because the group lives in state.groups, not the editor graph.
import puppeteer from 'puppeteer'

const URL = process.env.URL || 'http://127.0.0.1:5173/'
const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] })
const page = await browser.newPage()
page.setDefaultTimeout(15000)

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

  // --- 1) paste keeps the pasted nodes selected --------------------------
  const sel = await page.evaluate(async () => {
    const B = window.__blocks

    // single node
    await B.clearGraph()
    const a = await B.createNode('ConvBlock', { x: 0, y: 0 })
    B.state.selectedNodeId = a.id
    B.copySelection()
    await B.pasteClipboard()
    const ids1 = B.editor.getNodes().map((n) => n.id)
    const single = {
      total: ids1.length,
      selected: ids1.filter((id) => B.isNodeSelected(id)).length,
      originalSelected: B.isNodeSelected(a.id),
    }

    // multi node
    await B.clearGraph()
    const x = await B.createNode('ConvBlock', { x: 0, y: 0 })
    const y = await B.createNode('Pool2d', { x: 200, y: 0 })
    await B.addConnection(x.id, 'out', y.id, 'x')
    B.selectNodeIds([x.id, y.id])
    B.copySelection()
    await B.pasteClipboard()
    const ids2 = B.editor.getNodes().map((n) => n.id)
    const multi = {
      total: ids2.length,
      selected: ids2.filter((id) => B.isNodeSelected(id)).length,
      originalsSelected: B.isNodeSelected(x.id) || B.isNodeSelected(y.id),
    }

    // collapsed group: facade selected, hidden children not
    await B.clearGraph()
    const g1 = await B.createNode('ConvBlock', { x: 0, y: 0 })
    const g2 = await B.createNode('ConvBlock', { x: 200, y: 0 })
    Object.assign(g1.values, { in_ch: 8, out_ch: 8 })
    Object.assign(g2.values, { in_ch: 8, out_ch: 8 })
    await B.addConnection(g1.id, 'out', g2.id, 'x')
    await B.groupNodes([g1.id, g2.id])
    const facade = B.editor.getNodes().find((n) => n.entry?.kind === 'group')
    B.selectNodeIds([facade.id])
    B.copySelection()
    await B.pasteClipboard()
    const facades = B.editor.getNodes().filter((n) => n.entry?.kind === 'group')
    const group = {
      facades: facades.length,
      selFacades: facades.filter((f) => B.isNodeSelected(f.id)).length,
      selChildren: B.editor
        .getNodes()
        .filter((n) => n.groupId && n.entry?.kind !== 'group' && B.isNodeSelected(n.id)).length,
    }
    return { single, multi, group }
  })

  check('paste single: 2 nodes total', sel.single.total === 2, sel.single)
  check('paste single: only the copy is selected', sel.single.selected === 1 && !sel.single.originalSelected, sel.single)
  check('paste multi: 4 nodes total', sel.multi.total === 4, sel.multi)
  check('paste multi: both copies selected, originals not', sel.multi.selected === 2 && !sel.multi.originalsSelected, sel.multi)
  check('paste group: 2 facades', sel.group.facades === 2, sel.group)
  check('paste group: only pasted facade selected', sel.group.selFacades === 1, sel.group)
  check('paste group: no hidden children selected', sel.group.selChildren === 0, sel.group)

  // --- 2) undo / redo -----------------------------------------------------
  const hist = await page.evaluate(async () => {
    const B = window.__blocks
    const settle = () => new Promise((r) => setTimeout(r, 420)) // history debounce
    const count = () => B.editor.getNodes().length

    await B.clearGraph()
    B.initHistory() // baseline = empty graph
    await B.createNode('ConvBlock', { x: 0, y: 0 })
    await settle()
    await B.createNode('Pool2d', { x: 200, y: 0 })
    await settle()

    const before = count()
    await B.undo(); const u1 = count()
    await B.undo(); const u2 = count()
    await B.undo(); const u3 = count() // clamped at baseline
    await B.redo(); const r1 = count()
    await B.redo(); const r2 = count()
    await B.redo(); const r3 = count() // clamped at tip

    // group op then undo (snapshot history must restore state.groups)
    await B.clearGraph()
    B.initHistory()
    const a = await B.createNode('ConvBlock', { x: 0, y: 0 })
    const b = await B.createNode('ConvBlock', { x: 200, y: 0 })
    Object.assign(a.values, { in_ch: 8, out_ch: 8 })
    Object.assign(b.values, { in_ch: 8, out_ch: 8 })
    await B.addConnection(a.id, 'out', b.id, 'x')
    await settle()
    await B.groupNodes([a.id, b.id])
    await settle()
    const groupsAfter = B.state.groups.size
    await B.undo()
    const groupsAfterUndo = B.state.groups.size

    return { before, u1, u2, u3, r1, r2, r3, groupsAfter, groupsAfterUndo }
  })

  check('undo/redo: built 2 nodes', hist.before === 2, hist)
  check('undo once -> 1 node', hist.u1 === 1, hist)
  check('undo twice -> 0 nodes', hist.u2 === 0, hist)
  check('undo past baseline is clamped', hist.u3 === 0, hist)
  check('redo once -> 1 node', hist.r1 === 1, hist)
  check('redo twice -> 2 nodes', hist.r2 === 2, hist)
  check('redo past tip is clamped', hist.r3 === 2, hist)
  check('group op created a group', hist.groupsAfter === 1, hist)
  check('undo removes the group (state.groups restored)', hist.groupsAfterUndo === 0, hist)

  console.log(`\n${pass} pass, ${fail} fail`)
  await browser.close()
  process.exit(fail ? 1 : 0)
} catch (e) {
  console.error('TEST ERROR', e)
  await browser.close()
  process.exit(2)
}
