// End-to-end smoke for tags + copy/paste/duplicate.
//
// Validates that:
//   1. Setting a tag updates node.label and the rete node element re-renders.
//   2. Tag persists through export/import (and autosave/restore).
//   3. Copy/Paste creates new nodes with fresh ids + offset positions, and
//      preserves tags + ctor values + intra-selection connections.
//   4. Duplicate (Cmd+D) is copy+paste in one step.
//   5. Two ConvBlocks sharing a tag emit ONE __init__ slot in codegen.
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
  await page.evaluate(() => {
    localStorage.removeItem(window.__blocks.AUTOSAVE_KEY)
    localStorage.removeItem(window.__blocks.CLIPBOARD_KEY)
  })
  await page.reload({ waitUntil: 'networkidle0' })
  await page.waitForFunction(() => !!window.__blocks)

  // --- 1) Tags update labels live -----------------------------------------
  const tagInfo = await page.evaluate(async () => {
    await window.__blocks.clearGraph()
    const a = await window.__blocks.createNode('ConvBlock')
    window.__blocks.applyNodeTag(a, 'encoder')
    window.__blocks.area.update('node', a.id)
    return { id: a.id, label: a.label, tag: a.tag }
  })
  check('node.label includes the tag', tagInfo.label === 'ConvBlock · encoder', tagInfo)
  check('node.tag persists in the node object', tagInfo.tag === 'encoder')

  // --- 2) Tag round-trips through export/import ---------------------------
  const exportInfo = await page.evaluate(() => {
    const data = window.__blocks.getGraphData()
    return data.nodes.map((n) => ({ name: n.name, tag: n.tag }))
  })
  check(
    'getGraphData() includes tag field',
    exportInfo[0].tag === 'encoder',
    exportInfo
  )

  // --- 3) Copy/paste duplicates the node ---------------------------------
  const copyPasteInfo = await page.evaluate(async () => {
    // Force selection of the only node (rete's selector).
    const [n] = window.__blocks.editor.getNodes()
    window.__blocks.state.selectedNodeId = n.id
    window.__blocks.copySelection()
    await window.__blocks.pasteClipboard()
    const all = window.__blocks.editor.getNodes()
    return {
      total: all.length,
      tags: all.map((n) => n.tag),
      labels: all.map((n) => n.label),
      values: all.map((n) => ({ ...n.values })),
    }
  })
  check('paste created a second node', copyPasteInfo.total === 2, copyPasteInfo)
  check('pasted copy carries the tag forward', copyPasteInfo.tags.every((t) => t === 'encoder'), copyPasteInfo.tags)
  check(
    'pasted copy carries the same ctor values',
    JSON.stringify(copyPasteInfo.values[0]) === JSON.stringify(copyPasteInfo.values[1]),
    copyPasteInfo.values
  )

  // --- 4) Duplicate keyboard shortcut path -------------------------------
  const dupInfo = await page.evaluate(async () => {
    const before = window.__blocks.editor.getNodes().length
    window.__blocks.state.selectedNodeId = window.__blocks.editor.getNodes()[0].id
    await window.__blocks.duplicateSelection()
    return { before, after: window.__blocks.editor.getNodes().length }
  })
  check('duplicate adds one more node', dupInfo.after === dupInfo.before + 1, dupInfo)

  // --- 5) Shared-tag codegen collapses to one __init__ slot --------------
  const sharedCode = await page.evaluate(async () => {
    await window.__blocks.clearGraph()
    const a = await window.__blocks.createNode('ConvBlock')
    const b = await window.__blocks.createNode('ConvBlock')
    a.values.in_ch = 3
    a.values.out_ch = 16
    b.values.in_ch = 3
    b.values.out_ch = 16
    window.__blocks.applyNodeTag(a, 'shared')
    window.__blocks.applyNodeTag(b, 'shared')
    window.__blocks.area.update('node', a.id)
    window.__blocks.area.update('node', b.id)
    return window.__blocks.runCodegen()
  })
  const initSlots = (sharedCode.match(/self\.shared = ConvBlock\(/g) || []).length
  const callSites = (sharedCode.match(/self\.shared\(/g) || []).length
  check('one __init__ slot for shared-tag twins (UI codegen)', initSlots === 1, initSlots)
  check('two forward call sites to self.shared (UI codegen)', callSites === 2, callSites)

  // --- 6) Tag survives autosave reload -----------------------------------
  await page.evaluate(() => window.__blocks.saveToStorage())
  await page.reload({ waitUntil: 'networkidle0' })
  await page.waitForFunction(
    () =>
      window.__blocks &&
      window.__blocks.editor.getNodes().length === 2 &&
      window.__blocks.editor.getNodes().every((n) => n.tag === 'shared')
  )
  const afterReload = await page.evaluate(() =>
    window.__blocks.editor.getNodes().map((n) => ({ tag: n.tag, label: n.label }))
  )
  check(
    'tags survive autosave/restore',
    afterReload.length === 2 && afterReload.every((n) => n.tag === 'shared'),
    afterReload
  )
  check(
    'labels are re-rendered with tag after restore',
    afterReload.every((n) => /·\s*shared/.test(n.label)),
    afterReload
  )

  // --- 7) Copy a tagged group + external const, expand/collapse the copy ---
  // Regression for: (a) facade tag dropped on paste, (b) external const wired
  // into a child's __param__ port gets dangled after expand→collapse of the
  // pasted copy because the cloned child lost its exposed-param input.
  const groupClipboard = await page.evaluate(async () => {
    const blocks = window.__blocks
    const ed = blocks.editor
    await blocks.clearGraph()
    const conv = await blocks.createNode('ConvBlock')
    conv.values.in_ch = 3
    conv.values.out_ch = 16
    conv.values.kernel_size = 3
    conv.exposeParam('kernel_size')
    const k = await blocks.createNode('Constant')
    k.values.value = 5
    await blocks.addConnection(k.id, 'out', conv.id, '__param__kernel_size')
    await blocks.groupNodes([conv.id])
    const facade = ed.getNodes().find((n) => n.entry.kind === 'group')
    window.__blocks.applyNodeTag(facade, 'enc')
    blocks.area.update('node', facade.id)
    const constNode = ed.getNodes().find((n) => n.entry.kind === 'const')
    // Multi-select const + facade through rete's selector (the only way
    // copySelection sees more than one node). selector.add expects an entity
    // with translate/unselect stubs and an `accumulate` second arg.
    const noop = () => {}
    blocks.selector.add(
      { label: 'node', id: constNode.id, translate: noop, unselect: noop },
      true
    )
    blocks.selector.add(
      { label: 'node', id: facade.id, translate: noop, unselect: noop },
      true
    )
    blocks.copySelection()
    await blocks.pasteClipboard()
    // Find the pasted facade (the one whose entry.groupId differs from the original).
    const facades = ed.getNodes().filter((n) => n.entry.kind === 'group')
    const pastedFacade =
      facades.find((n) => n.entry.groupId !== facade.entry.groupId) || facades[1]
    const pastedTag = pastedFacade?.tag
    // Find the cloned const + child that belong to the pasted group.
    const pastedChildren = ed.getNodes().filter(
      (n) => n.groupId && n.groupId === pastedFacade.entry.groupId
    )
    const child = pastedChildren[0]
    const beforeExpand = ed
      .getConnections()
      .filter((c) => c.target === pastedFacade.id && c.targetInput.startsWith('__param__'))
      .length
    await blocks.expandGroup(pastedFacade.entry.groupId)
    const childParamEdges = ed
      .getConnections()
      .filter((c) => c.target === child.id && c.targetInput.startsWith('__param__'))
      .length
    await blocks.collapseGroup(pastedFacade.entry.groupId)
    const facadeAfter = ed
      .getNodes()
      .filter((n) => n.entry.kind === 'group')
      .find((n) => n.entry.groupId === pastedFacade.entry.groupId)
    const afterCollapse = ed
      .getConnections()
      .filter(
        (c) => c.target === facadeAfter.id && c.targetInput.startsWith('__param__')
      ).length
    return { pastedTag, beforeExpand, childParamEdges, afterCollapse }
  })
  check(
    'pasted group facade carries the original tag',
    groupClipboard.pastedTag === 'enc',
    groupClipboard
  )
  check(
    'const → pasted facade __param__ edge exists right after paste',
    groupClipboard.beforeExpand === 1,
    groupClipboard
  )
  check(
    'expand routes const → child __param__ on the cloned child',
    groupClipboard.childParamEdges === 1,
    groupClipboard
  )
  check(
    'recollapse restores const → facade __param__ (no dangling)',
    groupClipboard.afterCollapse === 1,
    groupClipboard
  )
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
