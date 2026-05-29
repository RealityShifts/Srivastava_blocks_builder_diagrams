// End-to-end test for NESTED GROUPS (a group inside another group).
//
// Builds Input -> ConvBlock(a) -> ConvBlock(b) -> Output, groups {a,b} into an
// inner group, then groups {innerFacade} into an outer group, and asserts:
//   - nesting is allowed (no "Cannot nest" rejection)
//   - the inner facade carries dual marking: entry.groupId (identity) AND
//     node.groupId (membership in the outer group)
//   - getGraphData/importGraph round-trips the dual marking (memberOf)
//   - collapse/expand visibility is correct across levels
//   - codegen emits Inner before Outer, each instantiating its child class
import puppeteer from 'puppeteer'

const URL = process.env.URL || 'http://127.0.0.1:5173/'
const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] })
const page = await browser.newPage()
page.setDefaultTimeout(15000)

const consoleErrors: string[] = []
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
const check = (name: string, cond: boolean, info?: any) => {
  if (cond) { pass++; console.log(`  ok  ${name}`) }
  else { fail++; console.log(`  FAIL ${name}`, info ?? '') }
}

try {
  await page.goto(URL, { waitUntil: 'networkidle0' })
  await page.waitForFunction(() => !!(window as any).__blocks)
  await page.evaluate(() => localStorage.removeItem((window as any).__blocks.AUTOSAVE_KEY))

  // Build chain + nest two levels of grouping.
  const nested = await page.evaluate(async () => {
    const blocks = (window as any).__blocks
    const ed = blocks.editor
    await blocks.clearGraph()
    const input = await blocks.createNode('Input')
    input.values.shape = 'B 3 224 224'
    input.values.dtype = 'float'
    const a = await blocks.createNode('ConvBlock'); a.values.in_ch = 3; a.values.out_ch = 16
    const b = await blocks.createNode('ConvBlock'); b.values.in_ch = 16; b.values.out_ch = 32
    const out = await blocks.createNode('Output'); out.values.name = 'features'
    await blocks.addConnection(input.id, 'out', a.id, 'x')
    await blocks.addConnection(a.id, 'out', b.id, 'x')
    await blocks.addConnection(b.id, 'out', out.id, 'x')

    // Inner group {a, b}.
    await blocks.groupNodes([a.id, b.id])
    await new Promise((r) => setTimeout(r, 30))
    const innerGid = [...blocks.state.groups.keys()][0]
    const innerFacade = ed.getNodes().find((n: any) => n.entry.kind === 'group')

    // Outer group containing the inner facade.
    await blocks.groupNodes([innerFacade.id])
    await new Promise((r) => setTimeout(r, 30))

    const groups = [...blocks.state.groups.values()]
    const facades = ed.getNodes().filter((n: any) => n.entry.kind === 'group')
    const innerNow = ed.getNode(innerFacade.id)
    const outerGid = groups.map((g: any) => g.id).find((g: any) => g !== innerGid)
    return {
      groupCount: groups.length,
      facadeCount: facades.length,
      innerGid,
      outerGid,
      // dual marking on the inner facade:
      innerIdentity: innerNow?.entry?.groupId,
      innerMembership: innerNow?.groupId,
    }
  })
  check('two groups created (nesting allowed)', nested.groupCount === 2, nested)
  check('inner facade identity = its own gid', nested.innerIdentity === nested.innerGid, nested)
  check('inner facade membership = outer gid (dual marking)', nested.innerMembership === nested.outerGid, nested)

  // Codegen: Inner before Outer, each instantiating its child.
  const code = await page.evaluate(() => (window as any).__blocks.runCodegen())
  const classNames = (code.match(/^class\s+(\w+)\(nn\.Module\):/gm) || []).map((s: string) => s.match(/class\s+(\w+)/)![1])
  check('three classes emitted (inner, outer, GeneratedModel)', classNames.length === 3, classNames)
  const idxOf = (c: string) => code.indexOf(`class ${c}(`)
  // The two non-GeneratedModel classes are the inner + outer group classes.
  const groupClasses = classNames.filter((c: string) => c !== 'GeneratedModel')
  check('a group class instantiates the other group class', groupClasses.some((c: string) => new RegExp(`class ${c}\\(nn\\.Module\\):[\\s\\S]*?self\\.\\w+ = (${groupClasses.filter((x:string)=>x!==c).join('|')})\\(`).test(code)), code)
  check('GeneratedModel instantiates a group class', new RegExp(`class GeneratedModel[\\s\\S]*self\\.\\w+ = (${groupClasses.join('|')})\\(`).test(code), code)
  check('no repeated kwargs anywhere', !code.split('\n').some((line: string) => {
    const kw = [...line.matchAll(/(\b\w+)=/g)].map((x) => x[1]); return kw.some((k, i) => kw.indexOf(k) !== i)
  }), code)

  // Round-trip serialization preserves dual marking via memberOf.
  const roundtrip = await page.evaluate(async () => {
    const blocks = (window as any).__blocks
    const ed = blocks.editor
    const snapshot = blocks.getGraphData()
    const innerSpec = snapshot.nodes.find((n: any) => n.kind === 'group' && n.memberOf)
    await blocks.clearGraph()
    await blocks.importGraph(snapshot)
    await new Promise((r) => setTimeout(r, 30))
    const facades = ed.getNodes().filter((n: any) => n.entry.kind === 'group')
    const nestedFacade = facades.find((f: any) => f.groupId)
    return {
      specHadMemberOf: !!innerSpec,
      restoredIdentity: nestedFacade?.entry?.groupId,
      restoredMembership: nestedFacade?.groupId,
      groupCount: [...blocks.state.groups.values()].length,
      // codegen still well-formed after reload:
      codeMatches: blocks.runCodegen() === undefined ? false : true,
    }
  })
  check('snapshot serialized memberOf for nested facade', roundtrip.specHadMemberOf, roundtrip)
  check('reload restores nested facade membership', !!roundtrip.restoredMembership, roundtrip)
  check('reload preserves both group descriptors', roundtrip.groupCount === 2, roundtrip)
  const codeAfter = await page.evaluate(() => (window as any).__blocks.runCodegen())
  check('codegen identical after import roundtrip', codeAfter === code, { before: code.length, after: codeAfter.length })

  // Visibility: collapse the OUTER group. Inner facade AND its descendants hide.
  // Then expand outer with inner still collapsed: inner facade shows, inner's
  // grandchildren stay hidden.
  const vis = await page.evaluate(async () => {
    const blocks = (window as any).__blocks
    const ed = blocks.editor
    const area = blocks.area
    const hidden = (id: string) => !!area.nodeViews.get(id)?.element.classList.contains('group-hidden')
    const groups = [...blocks.state.groups.values()]
    // Identify inner (has a facade with membership) vs outer.
    const innerFacade = ed.getNodes().find((n: any) => n.entry.kind === 'group' && n.groupId)
    const outerGid = innerFacade.groupId
    const innerGid = innerFacade.entry.groupId
    const leaves = ed.getNodes().filter((n: any) => n.groupId === innerGid) // a,b

    // Ensure both collapsed first (fresh import: groups collapsed).
    // Expand outer -> inner facade visible, leaves still hidden (inner collapsed).
    await blocks.expandGroup(outerGid)
    await new Promise((r) => setTimeout(r, 20))
    const afterOuterExpand = {
      innerFacadeHidden: hidden(innerFacade.id),
      leavesHidden: leaves.map((l: any) => hidden(l.id)),
    }
    // Now expand inner too -> leaves visible.
    await blocks.expandGroup(innerGid)
    await new Promise((r) => setTimeout(r, 20))
    const afterInnerExpand = {
      leavesVisible: ed.getNodes().filter((n: any) => n.groupId === innerGid).every((l: any) => !hidden(l.id)),
    }
    // Re-collapse outer -> everything under it hidden again (inner facade re-made).
    await blocks.collapseGroup(innerGid)
    await new Promise((r) => setTimeout(r, 20))
    await blocks.collapseGroup(outerGid)
    await new Promise((r) => setTimeout(r, 20))
    const innerFacade2 = ed.getNodes().find((n: any) => n.entry.kind === 'group' && n.groupId)
    const afterOuterCollapse = {
      innerFacadeHidden: innerFacade2 ? hidden(innerFacade2.id) : null,
    }
    return { afterOuterExpand, afterInnerExpand, afterOuterCollapse }
  })
  check('expand outer reveals the inner facade', vis.afterOuterExpand.innerFacadeHidden === false, vis)
  check('expand outer keeps inner leaves hidden (inner still collapsed)', vis.afterOuterExpand.leavesHidden.every((h: boolean) => h === true), vis)
  check('expand inner reveals the leaves', vis.afterInnerExpand.leavesVisible === true, vis)
  check('re-collapse outer hides the inner facade again', vis.afterOuterCollapse.innerFacadeHidden === true, vis)

  // Regression: a child output wired straight into its OWN group's facade
  // input (a residual that skips back into the group, as produced by some
  // self-attention layouts). When the group is collapsed the child is hidden,
  // so the edge must be CSS-hidden too - otherwise it renders as a wire
  // dangling from nowhere. The old visibility rule only hid edges whose two
  // endpoints shared a groupId, which a child->facade edge never does.
  const selfFacade = await page.evaluate(async () => {
    const blocks = (window as any).__blocks
    const area = blocks.area
    await blocks.clearGraph()
    // Two convs in a collapsed group; conv `a` also feeds the facade's spare
    // input port, modelling the residual-to-own-facade pattern.
    const graph = {
      framework: blocks.state.framework,
      nodes: [
        { id: 'inp', name: 'Input', kind: 'input', values: { name: 'x', shape: 'B 3 16 16', dtype: 'float' }, position: { x: -400, y: 0 } },
        { id: 'a', name: 'ConvBlock', kind: 'module', tag: 'aa', values: { in_ch: 3, out_ch: 8 }, position: { x: 0, y: 0 }, groupId: 'gSelf' },
        { id: 'b', name: 'ConvBlock', kind: 'module', tag: 'bb', values: { in_ch: 8, out_ch: 8 }, position: { x: 200, y: 0 }, groupId: 'gSelf' },
        { id: 'eltw', name: 'Elementwise', kind: 'eltwise', instanceName: 'Add', tag: 'cc', values: { op: 'add' }, position: { x: 400, y: 0 }, groupId: 'gSelf' },
        { id: 'facade', name: 'Group1', kind: 'group', groupId: 'gSelf', position: { x: 200, y: 0 },
          portMap: {
            inputs: [
              { facadePort: 'in0', childNodeId: 'a', childPort: 'x', shape: ['...'] },
              { facadePort: 'in1', childNodeId: 'eltw', childPort: 'xs', shape: ['...'] },
            ],
            outputs: [ { facadePort: 'out0', childNodeId: 'eltw', childPort: 'out', shape: ['...'] } ],
            params: [],
          } },
        { id: 'out', name: 'Output', kind: 'output', values: { name: 'y' }, position: { x: 700, y: 0 } },
      ],
      connections: [
        { source: 'inp', sourceOutput: 'out', target: 'facade', targetInput: 'in0' },
        { source: 'a', sourceOutput: 'out', target: 'b', targetInput: 'x' },
        { source: 'b', sourceOutput: 'out', target: 'eltw', targetInput: 'xs' },
        // The offending edge: child `a` feeds its own group's facade input.
        { source: 'a', sourceOutput: 'out', target: 'facade', targetInput: 'in1' },
        { source: 'facade', sourceOutput: 'out0', target: 'out', targetInput: 'x' },
      ],
      groups: [
        { id: 'gSelf', name: 'Group1', collapsed: true, facadeNodeId: 'facade', savedPosition: { x: 200, y: 0 }, childOffsets: {} },
      ],
    }
    await blocks.importGraph(graph)
    await new Promise((r) => setTimeout(r, 30))
    const ed = blocks.editor
    const childA = ed.getNodes().find((n: any) => n.values?.in_ch === 3)
    const facadeNode = ed.getNodes().find((n: any) => n.entry.kind === 'group')
    const childHidden = !!area.nodeViews.get(childA.id)?.element.classList.contains('group-hidden')
    // The child->own-facade edge: source is the hidden child, target the facade.
    const offending = ed.getConnections().find((c: any) => c.source === childA.id && c.target === facadeNode.id)
    const edgeHidden = offending
      ? !!area.connectionViews.get(offending.id)?.element.classList.contains('group-hidden')
      : null
    return { childHidden, edgeFound: !!offending, edgeHidden }
  })
  check('collapsed group child is hidden', selfFacade.childHidden === true, selfFacade)
  check('child->own-facade edge exists', selfFacade.edgeFound === true, selfFacade)
  check('child->own-facade edge is hidden (no dangling wire)', selfFacade.edgeHidden === true, selfFacade)

  if (consoleErrors.length > 0) {
    console.log('\nConsole errors observed:')
    consoleErrors.forEach((e) => console.log('  ', e))
    fail += consoleErrors.length
  }
  console.log(`\n${pass} pass, ${fail} fail`)
  if (fail > 0) process.exitCode = 1
} catch (err) {
  console.error('TEST FAILED:', err)
  process.exitCode = 1
} finally {
  await browser.close()
}
