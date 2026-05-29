// Regression for two bugs when copying a tagged group:
//
//  1. Spurious (cyclic / extra) internal edges in the pasted copy. The
//     peer-group sync maps source children to peer children; when several
//     children share a tag, that map must pair them positionally (in topo
//     order) rather than collapsing onto one peer and mis-pairing the rest.
//     See buildSourceToPeerChildMap in src/main.ts.
//
//  2. A false "boundary interface differs" weight-shared error that appears
//     immediately after paste and only clears on the next structural action.
//     The pasted facade must carry the proxied child port dtypes (not 'any')
//     so its interface matches the original's. See pasteClipboard's facade
//     boundary reconstruction in src/main.ts.
//
// Case: three uniform 128->128 ConvBlocks all tagged "dup", grouped and
// chained out of creation order. Uniform channels mean even a mis-paired edge
// unifies on shape and slips past dryRunEdge (which only checks shape, not
// cycles), exactly as in the real all-128 ConvBlock stack.
import puppeteer from 'puppeteer'

const URL = process.env.URL || 'http://127.0.0.1:5173/'
const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] })
const page = await browser.newPage()
page.setDefaultTimeout(15000)

function hasCycle(nodeIds, edges) {
  const out = new Map(nodeIds.map((id) => [id, []]))
  for (const e of edges) out.get(e.source)?.push(e.target)
  const seen = new Map() // 0=unvisited,1=onstack,2=done
  const dfs = (id) => {
    seen.set(id, 1)
    for (const t of out.get(id) ?? []) {
      const s = seen.get(t) ?? 0
      if (s === 1) return true
      if (s === 0 && dfs(t)) return true
    }
    seen.set(id, 2)
    return false
  }
  for (const id of nodeIds) if ((seen.get(id) ?? 0) === 0 && dfs(id)) return true
  return false
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

  const result = await page.evaluate(async () => {
    const B = window.__blocks
    await B.clearGraph()
    const nodes = []
    for (let i = 0; i < 3; i++) {
      const n = await B.createNode('ConvBlock', { x: i * 200, y: 0 })
      Object.assign(n.values, { in_ch: 128, out_ch: 128 })
      B.applyNodeTag(n, 'dup')
      nodes.push(n)
    }
    // Wire the chain in an order that does NOT match creation order
    // (chain is nodes[2] -> nodes[0] -> nodes[1]). This mirrors the real
    // graph, whose nodes are listed out of topological order, and is what
    // makes the buggy tag-collapse map disagree with the topo fallback.
    await B.addConnection(nodes[2].id, 'out', nodes[0].id, 'x')
    await B.addConnection(nodes[0].id, 'out', nodes[1].id, 'x')
    await B.groupNodes(nodes.map((n) => n.id))

    const facade = B.editor.getNodes().find((n) => n.entry?.kind === 'group')
    B.applyNodeTag(facade, 'Face_to_embedding')
    B.state.selectedNodeId = facade.id
    B.copySelection()
    await B.pasteClipboard()

    // Immediately after paste — before any structural action — the pasted
    // facade's boundary interface (including port dtypes) must already match
    // the original's. Otherwise the weight-shared boundary check fires and
    // only clears once a later action re-derives the boundary.
    B.runValidation()
    await new Promise((r) => setTimeout(r, 100))
    const boundaryErrors = (B.state.lastResult?.errors ?? [])
      .filter((e) => e.kind === 'tag-conflict')
      .map((e) => e.message)

    // The shared-tag peer sync runs on collapse/expand/tag edits in the real
    // app; invoke it directly to exercise the source->peer child mapping.
    await B.syncAllTaggedGroupInstances('Face_to_embedding')

    const byGroup = {}
    for (const n of B.editor.getNodes()) {
      if (!n.groupId || n.entry?.kind === 'group') continue
      ;(byGroup[n.groupId] ??= []).push(n.id)
    }
    const conns = B.editor.getConnections()
    const groups = Object.entries(byGroup).map(([gid, ids]) => {
      const set = new Set(ids)
      const edges = conns
        .filter((c) => set.has(c.source) && set.has(c.target) && c.targetInput === 'x')
        .map((c) => ({ source: c.source, target: c.target }))
      return { gid, ids, edges }
    })
    return { groups, boundaryErrors }
  })

  let pass = 0, fail = 0
  const check = (name, cond, info) => {
    if (cond) { pass++; console.log(`  ok  ${name}`) }
    else { fail++; console.log(`  FAIL ${name}`, JSON.stringify(info)) }
  }

  const { groups, boundaryErrors } = result
  check('two group instances exist', groups.length === 2, groups.map((g) => g.gid))
  for (const g of groups) {
    check(`group ${g.gid}: exactly two internal data edges`, g.edges.length === 2, g.edges)
    check(`group ${g.gid}: no cycle`, !hasCycle(g.ids, g.edges), g.edges)
  }
  check(
    'pasted facade boundary matches original (no tag-conflict)',
    boundaryErrors.length === 0,
    boundaryErrors
  )

  console.log(`\n${pass} pass, ${fail} fail`)
  await browser.close()
  process.exit(fail ? 1 : 0)
} catch (e) {
  console.error('TEST ERROR', e)
  await browser.close()
  process.exit(2)
}
