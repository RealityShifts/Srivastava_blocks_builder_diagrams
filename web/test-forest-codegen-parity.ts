// E2E parity: forest codegen must be BYTE-IDENTICAL to the live editor codegen
// across flat, grouped, nested, and weight-shared graphs. This is the proof
// that making the forest the single source of truth introduces NO discrepancy.
//
// Run with the dev server up:  URL=http://127.0.0.1:5173/ node test-forest-codegen-parity.ts
import puppeteer from 'puppeteer'

const URL = process.env.URL || 'http://127.0.0.1:5173/'
const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] })
const page = await browser.newPage()
page.setDefaultTimeout(15000)

const consoleErrors: string[] = []
page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`))
page.on('console', (m) => {
  if (m.type() === 'error' && !m.text().includes('favicon')) consoleErrors.push(`console.error: ${m.text()}`)
})

let pass = 0, fail = 0
const check = (name: string, cond: boolean, info?: any) => {
  if (cond) { pass++; console.log(`  ok  ${name}`) }
  else { fail++; console.log(`  FAIL ${name}`, typeof info === 'string' ? '\n' + info.slice(0, 1200) : (info ?? '')) }
}

/** First N chars where the two strings differ, for a readable diff. */
function firstDiff(a: string, b: string): string {
  const al = a.split('\n'), bl = b.split('\n')
  for (let i = 0; i < Math.max(al.length, bl.length); i++) {
    if (al[i] !== bl[i]) return `line ${i + 1}:\n  legacy: ${JSON.stringify(al[i])}\n  forest: ${JSON.stringify(bl[i])}`
  }
  return '(no line diff — length only)'
}

try {
  await page.goto(URL, { waitUntil: 'networkidle0' })
  await page.waitForFunction(() => !!(window as any).__blocks)
  await page.evaluate(() => localStorage.removeItem((window as any).__blocks.AUTOSAVE_KEY))

  // Helper run in-page: build a graph via `build`, then return both codegens.
  const parity = async (label: string, build: string) => {
    const res = await page.evaluate(async (buildSrc: string) => {
      const blocks = (window as any).__blocks
      await blocks.clearGraph()
      const fn = new Function('blocks', 'ed', `return (async () => { ${buildSrc} })()`)
      await fn(blocks, blocks.editor)
      await new Promise((r) => setTimeout(r, 40))
      const legacy = blocks.runCodegen()
      const forest = blocks.runCodegenFromForest()
      return { legacy, forest }
    }, build)
    check(`${label}: forest codegen == legacy`, res.legacy === res.forest, firstDiff(res.legacy, res.forest))
  }

  // 1. Flat chain.
  await parity('flat', `
    const i = await blocks.createNode('Input'); i.values.shape='B 3 8 8'; i.values.dtype='float';
    const a = await blocks.createNode('ConvBlock'); a.values.in_ch=3; a.values.out_ch=16;
    const o = await blocks.createNode('Output'); o.values.name='y';
    await blocks.addConnection(i.id,'out',a.id,'x');
    await blocks.addConnection(a.id,'out',o.id,'x');
  `)

  // 2. Single group.
  await parity('single-group', `
    const i = await blocks.createNode('Input'); i.values.shape='B 3 8 8'; i.values.dtype='float';
    const a = await blocks.createNode('ConvBlock'); a.values.in_ch=3; a.values.out_ch=16;
    const b = await blocks.createNode('ConvBlock'); b.values.in_ch=16; b.values.out_ch=32;
    const o = await blocks.createNode('Output'); o.values.name='y';
    await blocks.addConnection(i.id,'out',a.id,'x');
    await blocks.addConnection(a.id,'out',b.id,'x');
    await blocks.addConnection(b.id,'out',o.id,'x');
    await blocks.groupNodes([a.id,b.id]);
    await new Promise(r=>setTimeout(r,30));
  `)

  // 3. Nested groups.
  await parity('nested-groups', `
    const i = await blocks.createNode('Input'); i.values.shape='B 3 8 8'; i.values.dtype='float';
    const a = await blocks.createNode('ConvBlock'); a.values.in_ch=3; a.values.out_ch=16;
    const b = await blocks.createNode('ConvBlock'); b.values.in_ch=16; b.values.out_ch=32;
    const o = await blocks.createNode('Output'); o.values.name='y';
    await blocks.addConnection(i.id,'out',a.id,'x');
    await blocks.addConnection(a.id,'out',b.id,'x');
    await blocks.addConnection(b.id,'out',o.id,'x');
    await blocks.groupNodes([a.id,b.id]);
    await new Promise(r=>setTimeout(r,30));
    const f = blocks.editor.getNodes().find(n=>n.entry.kind==='group');
    await blocks.groupNodes([f.id]);
    await new Promise(r=>setTimeout(r,30));
  `)

  // 4. Two ConvBlocks sharing a tag (weight sharing) inside a group.
  await parity('weight-shared-in-group', `
    const i = await blocks.createNode('Input'); i.values.shape='B 3 8 8'; i.values.dtype='float';
    const a = await blocks.createNode('ConvBlock'); a.values.in_ch=3; a.values.out_ch=3; a.tag='shared';
    const b = await blocks.createNode('ConvBlock'); b.values.in_ch=3; b.values.out_ch=3; b.tag='shared';
    const o = await blocks.createNode('Output'); o.values.name='y';
    await blocks.addConnection(i.id,'out',a.id,'x');
    await blocks.addConnection(a.id,'out',b.id,'x');
    await blocks.addConnection(b.id,'out',o.id,'x');
    await blocks.groupNodes([a.id,b.id]);
    await new Promise(r=>setTimeout(r,30));
  `)

  // 5. Two outputs (multi-return) from a flat graph.
  await parity('multi-output', `
    const i = await blocks.createNode('Input'); i.values.shape='B 3 8 8'; i.values.dtype='float';
    const a = await blocks.createNode('ConvBlock'); a.values.in_ch=3; a.values.out_ch=16;
    const o1 = await blocks.createNode('Output'); o1.values.name='feat';
    const o2 = await blocks.createNode('Output'); o2.values.name='raw';
    await blocks.addConnection(i.id,'out',a.id,'x');
    await blocks.addConnection(a.id,'out',o1.id,'x');
    await blocks.addConnection(i.id,'out',o2.id,'x');
  `)

  if (consoleErrors.length) check('no console errors during runs', false, consoleErrors.join('\n'))
  else check('no console errors during runs', true)
} catch (e: any) {
  check('test harness ran', false, e?.message ?? String(e))
} finally {
  await browser.close()
}

console.log(`\n${pass} passed, ${fail} failed`)
process.exit(fail > 0 ? 1 : 0)
