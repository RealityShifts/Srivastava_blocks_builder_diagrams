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

  // Auto-tagging: every previously-untagged child must end up with a non-empty,
  // distinct tag stamped during groupSelected().
  const childTags = await page.evaluate((aId, bId) => {
    const ed = window.__blocks.editor
    return {
      a: ed.getNode(aId)?.tag ?? '',
      b: ed.getNode(bId)?.tag ?? '',
    }
  }, ids.a, ids.b)
  check('child a got a non-empty random tag', /^[a-z0-9]{3,}$/.test(childTags.a), childTags)
  check('child b got a non-empty random tag', /^[a-z0-9]{3,}$/.test(childTags.b), childTags)
  check('the two random tags are distinct', childTags.a !== childTags.b, childTags)

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
  const subClassMatch = /^class\s+\w+\(nn\.Module\):[\s\S]+^class\s+GeneratedModel\(nn\.Module\)/m.test(
    code
  )
  const subInit = /self\.\w+\s*=\s*ConvBlock\(in_ch=3, out_ch=16\)/.test(code)
  const subInit2 = /self\.\w+\s*=\s*ConvBlock\(in_ch=16, out_ch=32\)/.test(code)
  const mainInstantiates = /self\.\w+\s*=\s*Group\w*\(\)/.test(code)
  const mainHasGeneratedModel = /class\s+GeneratedModel\(nn\.Module\):/.test(code)
  const mainCallsGroup = /self\.\w+\(in0=/.test(code)
  check('codegen emits a subclass before the main class', subClassMatch, code.split('\n').slice(0, 30).join('\n'))
  check('subclass __init__ contains first ConvBlock', subInit)
  check('subclass __init__ contains second ConvBlock', subInit2)
  check('main class instantiates subgroup', mainInstantiates)
  check('GeneratedModel still emitted', mainHasGeneratedModel)
  check('main forward calls facade with in0=…', mainCallsGroup, code)
  check(
    'subclass appears before main class',
    /^class\s+\w+\(nn\.Module\)/m.exec(code).index < code.indexOf('class GeneratedModel')
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

  // Visual color: both facades + all 4 children should land on the SAME
  // --tag-color value because the group name is the same on both copies.
  const colors = await page.evaluate(() => {
    const area = window.__blocks.area
    const facades = window.__blocks.editor.getNodes().filter((n) => n.entry.kind === 'group')
    const children = window.__blocks.editor.getNodes().filter((n) => n.groupId)
    const colorOf = (n) => area.nodeViews.get(n.id).element.style.getPropertyValue('--tag-color')
    const tagged = (n) =>
      area.nodeViews.get(n.id).element.classList.contains('tagged-node')
    return {
      facadeColors: facades.map(colorOf),
      childColors: children.map(colorOf),
      allTagged: [...facades, ...children].every(tagged),
    }
  })
  check(
    'duplicated groups share a single --tag-color across all facades/children',
    colors.allTagged &&
      new Set([...colors.facadeColors, ...colors.childColors].filter(Boolean)).size === 1,
    colors
  )

  // Two groups with the same name should share a single emitted class
  // (Encoder duplicated -> still one `class Encoder(nn.Module)`), plus
  // GeneratedModel. The main class instantiates the shared class twice.
  const dupedCode = await page.evaluate(() => window.__blocks.runCodegen())
  const subClasses = (dupedCode.match(/^class\s+(\w+)\(nn\.Module\):/gm) || []).map((s) =>
    s.match(/class\s+(\w+)/)[1]
  )
  check(
    'duplicate groups share a single subclass (dedup by name)',
    subClasses.length === 2 && subClasses.includes('GeneratedModel'),
    subClasses
  )
  const groupClass = subClasses.find((c) => c !== 'GeneratedModel')
  const instantiateCount = (
    dupedCode.match(new RegExp(`self\\.\\w+\\s*=\\s*${groupClass}\\(\\)`, 'g')) || []
  ).length
  check(
    'main class instantiates the shared class twice',
    instantiateCount === 2,
    `instantiateCount=${instantiateCount}, code=${dupedCode}`
  )

  // Same tag on both facades -> weight sharing: one instance, two call sites.
  const sharedTagCode = await page.evaluate(async () => {
    const ed = window.__blocks.editor
    for (const f of ed.getNodes().filter((n) => n.entry.kind === 'group')) {
      window.__blocks.applyNodeTag(f, 'encoder')
      window.__blocks.area.update('node', f.id)
    }
    return window.__blocks.runCodegen()
  })
  check(
    'same tag on group facades -> one __init__ slot',
    (sharedTagCode.match(/self\.encoder = \w+\(\)/g) || []).length === 1,
    sharedTagCode
  )
  check(
    'same tag on group facades -> two forward call sites',
    (sharedTagCode.match(/self\.encoder\(/g) || []).length === 2,
    sharedTagCode
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

  // After ungroup, codegen should be a single GeneratedModel class.
  const codeFlat = await page.evaluate(() => window.__blocks.runCodegen())
  const classesFlat = (codeFlat.match(/^class\s+(\w+)\(nn\.Module\)/gm) || []).map((s) =>
    s.match(/class\s+(\w+)/)[1]
  )
  check(
    'post-ungroup codegen has only GeneratedModel',
    classesFlat.length === 1 && classesFlat[0] === 'GeneratedModel',
    classesFlat
  )

  // --- New scenario: pre-existing tags are preserved; facade-delete cascades.
  const tagAndDelete = await page.evaluate(async () => {
    const ed = window.__blocks.editor
    await window.__blocks.clearGraph()
    const input = await window.__blocks.createNode('Input')
    input.values.shape = 'B 3 224 224'
    input.values.dtype = 'float'
    const a = await window.__blocks.createNode('ConvBlock')
    a.values.in_ch = 3
    a.values.out_ch = 16
    // User-set tag on `a` must NOT be overwritten by the auto-random tag.
    window.__blocks.applyNodeTag(a, 'my-conv')
    const b = await window.__blocks.createNode('ConvBlock')
    b.values.in_ch = 16
    b.values.out_ch = 32
    const out = await window.__blocks.createNode('Output')
    out.values.name = 'features'
    await window.__blocks.addConnection(input.id, 'out', a.id, 'x')
    await window.__blocks.addConnection(a.id, 'out', b.id, 'x')
    await window.__blocks.addConnection(b.id, 'out', out.id, 'x')
    await window.__blocks.groupNodes([a.id, b.id])

    const tags = { a: ed.getNode(a.id)?.tag, b: ed.getNode(b.id)?.tag }
    const facade = ed.getNodes().find((n) => n.entry.kind === 'group')
    const beforeIds = ed.getNodes().map((n) => n.id)

    // Cascade-delete via the same code path the toolbar / Del key drive.
    window.__blocks.state.selectedNodeId = facade.id
    await window.__blocks.deleteSelected()

    const afterNodes = ed.getNodes()
    return {
      tags,
      beforeCount: beforeIds.length,
      afterCount: afterNodes.length,
      facadeStillThere: afterNodes.some((n) => n.entry.kind === 'group'),
      childAStillThere: afterNodes.some((n) => n.id === a.id),
      childBStillThere: afterNodes.some((n) => n.id === b.id),
      inputStillThere: afterNodes.some((n) => n.id === input.id),
      outputStillThere: afterNodes.some((n) => n.id === out.id),
      remainingGroups: window.__blocks.state.groups.size,
    }
  })
  check('pre-existing tag on child is NOT overwritten', tagAndDelete.tags.a === 'my-conv', tagAndDelete.tags)
  check('untagged sibling still gets a random tag', /^[a-z0-9]{3,}$/.test(tagAndDelete.tags.b || ''), tagAndDelete.tags)
  check('delete facade removes the facade itself', !tagAndDelete.facadeStillThere)
  check('delete facade removes child a', !tagAndDelete.childAStillThere)
  check('delete facade removes child b', !tagAndDelete.childBStillThere)
  check('delete facade leaves unrelated Input alive', tagAndDelete.inputStillThere)
  check('delete facade leaves unrelated Output alive', tagAndDelete.outputStillThere)
  check('delete facade drops the group from state.groups', tagAndDelete.remainingGroups === 0)
  check(
    'node count drops by exactly 3 (facade + 2 children)',
    tagAndDelete.beforeCount - tagAndDelete.afterCount === 3,
    tagAndDelete
  )

  // --- New scenario: child positions follow the facade across collapse/expand.
  // Drag the facade, then expand: children must land at facadePos + offset.
  const drag = await page.evaluate(async () => {
    const blocks = window.__blocks
    const ed = blocks.editor
    const area = blocks.area
    await blocks.clearGraph()
    const a = await blocks.createNode('ConvBlock')
    a.values.in_ch = 3
    a.values.out_ch = 16
    const b = await blocks.createNode('ConvBlock')
    b.values.in_ch = 16
    b.values.out_ch = 32
    await blocks.addConnection(a.id, 'out', b.id, 'x')
    // Park children at known coords so we can compute the expected offset.
    await area.translate(a.id, { x: 100, y: 200 })
    await area.translate(b.id, { x: 400, y: 250 })
    const aBefore = { ...area.nodeViews.get(a.id).position }
    const bBefore = { ...area.nodeViews.get(b.id).position }
    await blocks.groupNodes([a.id, b.id])
    await new Promise((r) => setTimeout(r, 20))
    const facade = ed.getNodes().find((n) => n.entry.kind === 'group')
    const facadeAtGroup = { ...area.nodeViews.get(facade.id).position }
    const group = blocks.state.groups.get(facade.entry.groupId)
    const recordedOffsetA = group.childOffsets[a.id]
    const recordedOffsetB = group.childOffsets[b.id]
    // Move the facade by (~+300, ~-150) - Rete snaps to a grid so the actual
    // position is whatever shows up after translate; read it back.
    await area.translate(facade.id, { x: facadeAtGroup.x + 300, y: facadeAtGroup.y - 150 })
    const facadeDragged = { ...area.nodeViews.get(facade.id).position }
    await blocks.expandGroup(facade.entry.groupId)
    await new Promise((r) => setTimeout(r, 20))
    const aAfter = { ...area.nodeViews.get(a.id).position }
    const bAfter = { ...area.nodeViews.get(b.id).position }
    // Now collapse again from the dragged location; verify the facade lands
    // where the user left it (and not at the original creation centroid).
    const gid = facade.entry.groupId
    await blocks.collapseGroup(gid)
    await new Promise((r) => setTimeout(r, 20))
    const facade2 = ed.getNodes().find((n) => n.entry.kind === 'group')
    const facadeAfterRecollapse = { ...area.nodeViews.get(facade2.id).position }
    return {
      aBefore, bBefore,
      facadeAtGroup,
      recordedOffsetA,
      recordedOffsetB,
      facadeDragged,
      aAfter, bAfter,
      facadeAfterRecollapse,
    }
  })
  // Allow a few pixels of slack: Rete snaps translate() targets to a grid,
  // so positions can drift by up to ~8px from the requested values.
  const eq = (n, m, eps = 2) => Math.abs(n - m) <= eps
  check(
    'child offset captured = aBefore - facadeAtGroup',
    eq(drag.recordedOffsetA.dx, drag.aBefore.x - drag.facadeAtGroup.x) &&
      eq(drag.recordedOffsetA.dy, drag.aBefore.y - drag.facadeAtGroup.y),
    drag
  )
  check(
    'child offset captured for b',
    eq(drag.recordedOffsetB.dx, drag.bBefore.x - drag.facadeAtGroup.x) &&
      eq(drag.recordedOffsetB.dy, drag.bBefore.y - drag.facadeAtGroup.y),
    drag
  )
  // After dragging the facade and expanding, children must follow:
  //   newChildPos == draggedFacadePos + originalOffset (post-grid-snap).
  check(
    'expand places child a at dragged facade + offset',
    eq(drag.aAfter.x, drag.facadeDragged.x + drag.recordedOffsetA.dx) &&
      eq(drag.aAfter.y, drag.facadeDragged.y + drag.recordedOffsetA.dy),
    drag
  )
  check(
    'expand places child b at dragged facade + offset',
    eq(drag.bAfter.x, drag.facadeDragged.x + drag.recordedOffsetB.dx) &&
      eq(drag.bAfter.y, drag.facadeDragged.y + drag.recordedOffsetB.dy),
    drag
  )
  // Sanity: the new positions are NOT the original (pre-collapse) ones -
  // i.e. we genuinely moved them, not just no-op'd.
  check(
    'children did NOT snap back to original absolute coords',
    !(eq(drag.aAfter.x, drag.aBefore.x) && eq(drag.aAfter.y, drag.aBefore.y)),
    drag
  )
  // Re-collapse must put the facade where the user left it.
  check(
    're-collapse keeps facade at the dragged position',
    eq(drag.facadeAfterRecollapse.x, drag.facadeDragged.x) &&
      eq(drag.facadeAfterRecollapse.y, drag.facadeDragged.y),
    drag
  )

  // --- Autosave roundtrip preserves the offsets so this still works after reload.
  const reloadDrag = await page.evaluate(async () => {
    const blocks = window.__blocks
    const ed = blocks.editor
    const area = blocks.area
    // Persist current state, then re-import to simulate a reload.
    const snapshot = blocks.getGraphData()
    await blocks.clearGraph()
    await blocks.importGraph(snapshot)
    await new Promise((r) => setTimeout(r, 20))
    const facade = ed.getNodes().find((n) => n.entry.kind === 'group')
    const facadePos = { ...area.nodeViews.get(facade.id).position }
    const move = { x: facadePos.x - 50, y: facadePos.y + 80 }
    await area.translate(facade.id, move)
    await blocks.expandGroup(facade.entry.groupId)
    await new Promise((r) => setTimeout(r, 20))
    const children = ed.getNodes().filter((n) => n.groupId)
    const childPositions = children.map((c) => ({
      id: c.id,
      pos: { ...area.nodeViews.get(c.id).position },
    }))
    return { move, childPositions, facadePosBeforeMove: facadePos }
  })
  check(
    'after autosave roundtrip, all children still positioned relative to the facade',
    reloadDrag.childPositions.length >= 2 &&
      reloadDrag.childPositions.every(
        (c) =>
          Number.isFinite(c.pos.x) &&
          Number.isFinite(c.pos.y) &&
          // The expand point moved by (-50, +80) vs facadePosBeforeMove, so
          // children should also have moved by exactly that delta. We don't
          // know the offsets here, but we can verify NONE of them sit at the
          // un-dragged facade's coords.
          !(
            Math.abs(c.pos.x - reloadDrag.facadePosBeforeMove.x) < 0.5 &&
            Math.abs(c.pos.y - reloadDrag.facadePosBeforeMove.y) < 0.5
          )
      ),
    reloadDrag
  )

  // --- Add ungrouped node to an existing group (auto-expands when collapsed).
  const addToGroup = await page.evaluate(async () => {
    const blocks = window.__blocks
    const ed = blocks.editor
    await blocks.clearGraph()
    const a = await blocks.createNode('ConvBlock')
    a.values.in_ch = 3
    a.values.out_ch = 16
    const b = await blocks.createNode('ConvBlock')
    b.values.in_ch = 16
    b.values.out_ch = 32
    await blocks.addConnection(a.id, 'out', b.id, 'x')
    await blocks.groupNodes([a.id, b.id])
    await new Promise((r) => setTimeout(r, 20))
    const gid = [...blocks.state.groups.keys()][0]
    const c = await blocks.createNode('ConvBlock')
    c.values.in_ch = 32
    c.values.out_ch = 64
    await blocks.addConnection(b.id, 'out', c.id, 'x')
    const wasCollapsed = blocks.state.groups.get(gid)?.collapsed
    await blocks.addNodesToGroup(gid, new Set([c.id]))
    await new Promise((r) => setTimeout(r, 20))
    const group = blocks.state.groups.get(gid)
    return {
      gid,
      wasCollapsed,
      collapsed: group?.collapsed,
      cGroupId: ed.getNode(c.id)?.groupId,
      childCount: ed.getNodes().filter((n) => n.groupId === gid).length,
    }
  })
  check('group started collapsed before add', addToGroup.wasCollapsed === true, addToGroup)
  check('addNodesToGroup leaves group expanded for editing', addToGroup.collapsed === false, addToGroup)
  check('added node receives groupId', addToGroup.cGroupId === addToGroup.gid, addToGroup)
  check('group now has three children', addToGroup.childCount === 3, addToGroup)

  // --- Dangling child ports surface on the facade.
  const dangling = await page.evaluate(async () => {
    const blocks = window.__blocks
    const ed = blocks.editor
    await blocks.clearGraph()
    // MultiHeadAttention has three inputs: query, key, value. Only wire one
    // of them to confirm the other two still surface as facade inputs.
    const inp = await blocks.createNode('Input')
    inp.values.shape = 'B 4 64'
    const mha = await blocks.createNode('MultiHeadAttention')
    mha.values.dim = 64
    await blocks.addConnection(inp.id, 'out', mha.id, 'query')
    await blocks.groupNodes([mha.id])
    await new Promise((r) => setTimeout(r, 20))
    const facade = ed.getNodes().find((n) => n.entry.kind === 'group')
    return {
      facadeInputs: facade ? Object.keys(facade.inputs).filter((k) => !k.startsWith('__param__')) : [],
      facadeOutputs: facade ? Object.keys(facade.outputs) : [],
    }
  })
  check(
    'facade exposes every child input (wired + dangling)',
    dangling.facadeInputs.length === 3,
    dangling.facadeInputs
  )
  check(
    'facade exposes child output even when no external consumer',
    dangling.facadeOutputs.length === 1,
    dangling.facadeOutputs
  )

  // --- Atlas: edit one tagged node, peers update.
  const atlasSync = await page.evaluate(async () => {
    const blocks = window.__blocks
    const ed = blocks.editor
    await blocks.clearGraph()
    const a = await blocks.createNode('ConvBlock')
    a.values.in_ch = 3
    a.values.out_ch = 16
    blocks.applyNodeTag(a, 'down1')
    const b = await blocks.createNode('ConvBlock')
    b.values.in_ch = 9
    b.values.out_ch = 9
    blocks.applyNodeTag(b, 'down1')
    blocks.refreshTagAtlas()
    // Adopt canonical values onto b (simulates the re-tag adopt path).
    const { adoptValuesFromAtlas } = await import('/src/tagAtlas.js')
    adoptValuesFromAtlas(blocks.state.tagAtlas, b)
    return {
      atlas: blocks.tagAtlasSummary,
      bOutCh: ed.getNode(b.id)?.values?.out_ch,
    }
  })
  check('atlas has the tagged family', !!atlasSync.atlas['down1'], atlasSync)
  check('adopt copies canonical out_ch onto peer', atlasSync.bOutCh === 16, atlasSync)

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
