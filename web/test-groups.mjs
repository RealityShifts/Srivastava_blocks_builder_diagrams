// End-to-end smoke test for subgraph grouping.
//
// Builds Input -> ConvBlock -> ConvBlock -> Output, groups the two convs,
// then asserts:
//   - facade node appears with auto-derived in0/out0 ports
//   - boundary edges are rerouted to the facade
//   - children are CSS-hidden
//   - codegen emits a subclass + the main class instantiates it
//   - autosave roundtrip preserves group state
//   - expand restores children & edges
//   - re-collapse rebuilds a new facade
//   - ungroup dissolves cleanly
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

  // Build the chain.
  const ids = await page.evaluate(async () => {
    await window.__blocks.clearGraph()
    const input = await window.__blocks.createNode('Input')
    input.values.shape = 'B 3 224 224'
    input.values.dtype = 'float'
    const a = await window.__blocks.createNode('ConvBlock')
    a.values.in_ch = 3
    a.values.out_ch = 16
    const b = await window.__blocks.createNode('ConvBlock')
    b.values.in_ch = 16
    b.values.out_ch = 32
    const out = await window.__blocks.createNode('Output')
    out.values.name = 'features'
    await window.__blocks.addConnection(input.id, 'out', a.id, 'x')
    await window.__blocks.addConnection(a.id, 'out', b.id, 'x')
    await window.__blocks.addConnection(b.id, 'out', out.id, 'x')
    window.__blocks.runValidation()
    return { input: input.id, a: a.id, b: b.id, out: out.id }
  })
  check('built 4-node chain', !!ids.a && !!ids.b)

  // Group the two ConvBlocks programmatically.
  const grouped = await page.evaluate(async (aId, bId) => {
    await window.__blocks.groupNodes([aId, bId])
    await new Promise((r) => setTimeout(r, 30))
    const groups = [...window.__blocks.state.groups.values()]
    const facade = groups[0]
      ? window.__blocks.editor
          .getNodes()
          .find((n) => n.entry.kind === 'group' && n.entry.groupId === groups[0].id)
      : null
    return {
      groupCount: groups.length,
      groupId: groups[0]?.id,
      collapsed: groups[0]?.collapsed,
      facadeId: facade?.id,
      facadeInputs: facade ? Object.keys(facade.inputs) : [],
      facadeOutputs: facade ? Object.keys(facade.outputs) : [],
    }
  }, ids.a, ids.b)
  check('one group created', grouped.groupCount === 1, grouped)
  check('group starts collapsed', grouped.collapsed === true)
  check('facade has 1 input port', grouped.facadeInputs.length === 1, grouped.facadeInputs)
  check('facade has 1 output port', grouped.facadeOutputs.length === 1, grouped.facadeOutputs)

  // Boundary edges rerouted, children hidden.
  const rerouted = await page.evaluate((aId, bId, facadeId) => {
    const ed = window.__blocks.editor
    const area = window.__blocks.area
    const conns = ed.getConnections()
    const aHidden = area.nodeViews.get(aId).element.classList.contains('group-hidden')
    const bHidden = area.nodeViews.get(bId).element.classList.contains('group-hidden')
    const fEl = area.nodeViews.get(facadeId).element
    const facadeStyled = fEl.classList.contains('group-facade')
    return {
      childrenHidden: aHidden && bHidden,
      facadeStyled,
      facadeInConns: conns.filter((c) => c.target === facadeId).length,
      facadeOutConns: conns.filter((c) => c.source === facadeId).length,
      directBoundaryConns: conns.filter(
        (c) =>
          (c.target === aId || c.source === bId) &&
          c.source !== facadeId &&
          c.target !== facadeId
      ).length,
    }
  }, ids.a, ids.b, grouped.facadeId)
  check('children hidden via CSS', rerouted.childrenHidden)
  check('facade has group-facade class', rerouted.facadeStyled)
  check('one input edge into facade', rerouted.facadeInConns === 1)
  check('one output edge out of facade', rerouted.facadeOutConns === 1)
  check('no direct external boundary edges remain', rerouted.directBoundaryConns === 0)

  // Codegen: subclass + main class.
  const code = await page.evaluate(() => window.__blocks.runCodegen())
  const subClassMatch = /^class\s+Group\w+\(nn\.Module\):/m.test(code)
  const subInit = /self\.\w+\s*=\s*ConvBlock\(in_ch=3, out_ch=16\)/.test(code)
  const subInit2 = /self\.\w+\s*=\s*ConvBlock\(in_ch=16, out_ch=32\)/.test(code)
  const mainInstantiates = /self\.\w+\s*=\s*Group\w+\(\)/.test(code)
  const mainHasGeneratedModel = /class\s+GeneratedModel\(nn\.Module\):/.test(code)
  const mainCallsGroup = /self\.\w+\(in0=/.test(code)
  check('codegen emits a Group_* subclass', subClassMatch, code.split('\n').slice(0, 30).join('\n'))
  check('subclass __init__ contains first ConvBlock', subInit)
  check('subclass __init__ contains second ConvBlock', subInit2)
  check('main class instantiates subgroup', mainInstantiates)
  check('GeneratedModel still emitted', mainHasGeneratedModel)
  check('main forward calls facade with in0=…', mainCallsGroup, code)
  check(
    'subclass appears before main class',
    code.indexOf('class Group') < code.indexOf('class GeneratedModel')
  )

  // Runtime-trace codegen: the subclass MUST return real tensors (not the
  // _runtime_shapes dict) or the shape-runner reports empty shapes for
  // everything downstream of a group.
  const traceCode = await page.evaluate(async () => {
    const { generate } = await import('/src/codegen.js')
    return generate(
      window.__blocks.editor.getNodes(),
      window.__blocks.editor.getConnections(),
      'pytorch',
      { trace: true }
    )
  })
  const subBlock = traceCode.match(/class\s+Group\w+[\s\S]+?(?=\n\nclass\s|\Z)/)?.[0] || ''
  const mainBlock = traceCode.match(/class\s+GeneratedModel[\s\S]+/)?.[0] || ''
  check('trace: subclass does NOT return _runtime_shapes', !/return\s+_runtime_shapes/.test(subBlock), subBlock.slice(-200))
  check('trace: subclass returns a real tensor', /return\s+\w+(?:\s*,\s*\w+)*\s*$/m.test(subBlock.trim()))
  check('trace: subclass does not create a _runtime_shapes dict', !/_runtime_shapes\s*=\s*\{\}/.test(subBlock))
  check('trace: main class still builds and returns _runtime_shapes', /_runtime_shapes\s*=\s*\{\}/.test(mainBlock) && /return\s+_runtime_shapes/.test(mainBlock))
  check('trace: facade output is recorded by the main class', /_runtime_shapes\["[^"]+\/out0"\]\s*=\s*\(list\(encoder_1\.shape\)/.test(mainBlock) || /_runtime_shapes\["[^"]+\/out0"\]/.test(mainBlock), mainBlock)

  // Autosave roundtrip: save, reload, verify state survives.
  await page.evaluate(() => window.__blocks.saveToStorage())
  await page.reload({ waitUntil: 'networkidle0' })
  await page.waitForFunction(() => !!window.__blocks)
  await page.waitForFunction(() => window.__blocks.editor.getNodes().length >= 5)
  const afterReload = await page.evaluate(() => {
    const groups = [...window.__blocks.state.groups.values()]
    const facade = window.__blocks.editor
      .getNodes()
      .find((n) => n.entry.kind === 'group')
    return {
      groupCount: groups.length,
      collapsed: groups[0]?.collapsed,
      facadeInputs: facade ? Object.keys(facade.inputs).length : 0,
      facadeOutputs: facade ? Object.keys(facade.outputs).length : 0,
      childCount: window.__blocks.editor.getNodes().filter((n) => n.groupId).length,
      hiddenChildren: window.__blocks.editor
        .getNodes()
        .filter(
          (n) =>
            n.groupId &&
            window.__blocks.area.nodeViews
              .get(n.id)
              ?.element.classList.contains('group-hidden')
        ).length,
    }
  })
  check('group survives reload', afterReload.groupCount === 1, afterReload)
  check('group stays collapsed after reload', afterReload.collapsed === true)
  check(
    'facade port count preserved',
    afterReload.facadeInputs === 1 && afterReload.facadeOutputs === 1
  )
  check('child groupId preserved', afterReload.childCount === 2)
  check('children stay CSS-hidden after reload', afterReload.hiddenChildren === 2)

  const codeAfter = await page.evaluate(() => window.__blocks.runCodegen())
  check('codegen identical after autosave roundtrip', codeAfter === code, {
    before: code.length,
    after: codeAfter.length,
  })

  // Expand the group.
  const expanded = await page.evaluate(async () => {
    const gid = [...window.__blocks.state.groups.keys()][0]
    await window.__blocks.expandGroup(gid)
    await new Promise((r) => setTimeout(r, 30))
    const facade = window.__blocks.editor.getNodes().find((n) => n.entry.kind === 'group')
    return {
      collapsed: window.__blocks.state.groups.get(gid)?.collapsed,
      facadeGone: !facade,
      childrenVisible: window.__blocks.editor
        .getNodes()
        .filter((n) => n.groupId)
        .every(
          (n) =>
            !window.__blocks.area.nodeViews
              .get(n.id)
              .element.classList.contains('group-hidden')
        ),
      directConns: window.__blocks.editor
        .getConnections()
        .filter((c) => c.source && c.target).length,
    }
  })
  check('expand removes facade', expanded.facadeGone)
  check('expand flips collapsed=false', expanded.collapsed === false)
  check('children visible after expand', expanded.childrenVisible)

  // Re-collapse.
  const recollapsed = await page.evaluate(async () => {
    const gid = [...window.__blocks.state.groups.keys()][0]
    await window.__blocks.collapseGroup(gid)
    await new Promise((r) => setTimeout(r, 30))
    const facade = window.__blocks.editor.getNodes().find((n) => n.entry.kind === 'group')
    return {
      collapsed: window.__blocks.state.groups.get(gid)?.collapsed,
      facadeBack: !!facade,
      facadeIn: facade ? Object.keys(facade.inputs).length : 0,
      facadeOut: facade ? Object.keys(facade.outputs).length : 0,
    }
  })
  check('re-collapse recreates facade', recollapsed.facadeBack)
  check('re-collapse flips collapsed=true', recollapsed.collapsed === true)
  check(
    're-collapse preserves port count',
    recollapsed.facadeIn === 1 && recollapsed.facadeOut === 1
  )

  // Duplicate the (collapsed) group: copySelection() falls back to
  // state.selectedNodeId when nothing is in the selector, so we just set
  // that and exercise the same code path Ctrl/Cmd+D uses.
  const duped = await page.evaluate(async () => {
    const ed = window.__blocks.editor
    const facade = ed.getNodes().find((n) => n.entry.kind === 'group')
    window.__blocks.state.selectedNodeId = facade.id
    const ok = window.__blocks.copySelection()
    await window.__blocks.pasteClipboard()
    await new Promise((r) => setTimeout(r, 60))
    const groups = [...window.__blocks.state.groups.values()]
    const facades = ed.getNodes().filter((n) => n.entry.kind === 'group')
    const children = ed.getNodes().filter((n) => n.groupId)
    const distinctGids = new Set(children.map((c) => c.groupId))
    // Each facade's portMap must point at children that actually exist.
    const facadeChildIdsValid = facades.every((f) =>
      (f.entry.portMap.inputs || [])
        .concat(f.entry.portMap.outputs || [])
        .every((m) => !!ed.getNode(m.childNodeId))
    )
    return {
      copied: ok,
      groupCount: groups.length,
      facadeCount: facades.length,
      childCount: children.length,
      distinctGids: distinctGids.size,
      facadeChildIdsValid,
    }
  })
  check('copySelection succeeded for facade-only selection', duped.copied)
  check('duplicate produces a second group', duped.groupCount === 2, duped)
  check('duplicate produces a second facade', duped.facadeCount === 2)
  check('duplicate clones all 2 children', duped.childCount === 4)
  check('duplicated children get a fresh gid', duped.distinctGids === 2)
  check('every facade portMap points at live children', duped.facadeChildIdsValid)

  // Original and copy generate the same subclass body (only class name differs).
  const dupedCode = await page.evaluate(() => window.__blocks.runCodegen())
  const subClasses = (dupedCode.match(/^class\s+(\w+)\(nn\.Module\):/gm) || []).map((s) =>
    s.match(/class\s+(\w+)/)[1]
  )
  check(
    'two subclasses + GeneratedModel in codegen',
    subClasses.length === 3 && subClasses.includes('GeneratedModel'),
    subClasses
  )

  // Tear down for the ungroup assertion: remove the duplicate so the rest of
  // the test continues with a single group as before.
  await page.evaluate(async () => {
    const dup = [...window.__blocks.state.groups.values()][1]
    if (dup) await window.__blocks.ungroup(dup.id)
  })

  // Ungroup.
  const ungrouped = await page.evaluate(async () => {
    const gid = [...window.__blocks.state.groups.keys()][0]
    await window.__blocks.ungroup(gid)
    return {
      groupCount: window.__blocks.state.groups.size,
      facadeGone: !window.__blocks.editor
        .getNodes()
        .find((n) => n.entry.kind === 'group'),
      orphanedChildren: window.__blocks.editor.getNodes().filter((n) => n.groupId).length,
    }
  })
  check('ungroup removes group state', ungrouped.groupCount === 0)
  check('ungroup deletes facade', ungrouped.facadeGone)
  check('ungroup clears groupId on children', ungrouped.orphanedChildren === 0)

  // After ungroup, codegen should once again be a single class (no subgroup).
  const codeFlat = await page.evaluate(() => window.__blocks.runCodegen())
  check(
    'post-ungroup codegen has no Group_* subclass',
    !/^class\s+Group\w+\(/m.test(codeFlat),
    codeFlat
  )

  if (consoleErrors.length > 0) {
    console.log('')
    console.log('Console errors observed:')
    consoleErrors.forEach((e) => console.log('  ', e))
    fail += consoleErrors.length
  }

  console.log('')
  console.log(`${pass} pass, ${fail} fail`)
  if (fail > 0) process.exitCode = 1
} catch (err) {
  console.error('TEST FAILED:', err)
  process.exitCode = 1
} finally {
  await browser.close()
}
