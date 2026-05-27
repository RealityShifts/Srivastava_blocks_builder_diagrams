/**
 * Bootstrap the Rete v2 editor with the Lit render plugin, wire palette /
 * inspector / diagnostics panels, and gate connection creation on the
 * unification-based validator so invalid edges are rejected at the source.
 */

import { NodeEditor, ClassicPreset } from 'rete'
import { AreaPlugin, AreaExtensions } from 'rete-area-plugin'
import { ConnectionPlugin, Presets as ConnectionPresets } from 'rete-connection-plugin'
import { LitPlugin, Presets as LitPresets } from '@retejs/lit-plugin'

import { makeNode, INPUT_ENTRY } from './nodes.js'
import { validate, dryRunEdge } from './validator.js'
import { generate as generateCode } from './codegen.js'
import { isFullyConcrete, runShapeCheck } from './runtime.js'
import { resolve } from './shape.js'
import {
  renderPalette,
  filterPalette,
  renderInspector,
  renderDiagnostics,
  showCode,
  wireCodeDialog,
  updateRuntimePanel,
} from './ui.js'

// --- state ---
const state = {
  framework: 'pytorch',
  entries: [],
  byName: new Map(),
  blockInfo: new Map(), // BlockName -> { description, shapes, mermaid, source, category }
  selectedNodeId: null,
  lastResult: null,
  runtimeShapes: null,
  batchSize: 2,
  runtimeRunning: false,
  runtimeError: null,
  runtimeErrorNodeId: null,
}

let editor, area, connection, render, selector

async function bootstrap() {
  const container = document.getElementById('editor')

  editor = new NodeEditor()
  area = new AreaPlugin(container)
  connection = new ConnectionPlugin()
  render = new LitPlugin()

  selector = AreaExtensions.selector()
  AreaExtensions.selectableNodes(area, selector, {
    accumulating: AreaExtensions.accumulateOnCtrl(),
  })
  AreaExtensions.simpleNodesOrder(area)
  AreaExtensions.snapGrid(area, { dynamic: true, size: 16 })

  render.addPreset(LitPresets.classic.setup())
  connection.addPreset(ConnectionPresets.classic.setup())

  editor.use(area)
  area.use(connection)
  area.use(render)

  // Validate every connection attempt with a dry-run unification.
  editor.addPipe((context) => {
    if (context.type === 'connectioncreate') {
      const { source, sourceOutput, target, targetInput } = context.data
      if (source === target) {
        flashDiagnostic('cannot connect a node to itself')
        return // cancel
      }
      const srcNode = editor.getNode(source)
      const tgtNode = editor.getNode(target)
      if (!srcNode || !tgtNode) return context
      const check = dryRunEdge(editor, srcNode, sourceOutput, tgtNode, targetInput)
      if (!check.ok) {
        flashDiagnostic(`refused edge: ${check.reason}`)
        return // cancel
      }
    }
    return context
  })

  const structuralSignals = new Set([
    'connectioncreated',
    'connectionremoved',
    'nodecreated',
    'noderemoved',
  ])
  editor.addPipe((context) => {
    if (structuralSignals.has(context.type)) queueValidation()
    return context
  })

  // Track selection for the inspector.
  area.addPipe((context) => {
    if (context.type === 'nodepicked') {
      state.selectedNodeId = context.data.id
      refreshInspector()
    }
    return context
  })

  AreaExtensions.zoomAt(area, editor.getNodes())

  // Drop-from-palette support.
  container.addEventListener('dragover', (e) => e.preventDefault())
  container.addEventListener('drop', async (e) => {
    e.preventDefault()
    const name = e.dataTransfer.getData('application/x-block-name')
    if (!name) return
    const rect = container.getBoundingClientRect()
    const screen = { x: e.clientX - rect.left, y: e.clientY - rect.top }
    const t = area.area.transform // { x, y, k }
    const canvas = { x: (screen.x - t.x) / t.k, y: (screen.y - t.y) / t.k }
    await createNode(name, canvas)
  })

  // Toolbar
  wireCodeDialog()
  document.getElementById('framework-select').addEventListener('change', async (e) => {
    state.framework = e.target.value
    state.runtimeShapes = null
    state.runtimeError = null
    state.runtimeErrorNodeId = null
    await loadManifest()
    await clearGraph()
  })
  document.getElementById('search').addEventListener('input', (e) => {
    filterPalette(document.getElementById('palette'), e.target.value)
  })
  document.getElementById('clear-btn').addEventListener('click', () => clearGraph())
  document.getElementById('import-btn').addEventListener('click', () => {
    document.getElementById('import-file-input').click()
  })
  document.getElementById('import-file-input').addEventListener('change', async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      const text = await file.text()
      const data = JSON.parse(text)
      const stats = await importGraph(data)
      flashDiagnostic(
        `Imported ${stats.nodes} node(s), ${stats.connections} connection(s)${
          stats.dropped > 0 ? `, dropped ${stats.dropped}` : ''
        }`
      )
    } catch (err) {
      flashDiagnostic(`Import failed: ${err.message || String(err)}`)
    } finally {
      e.target.value = ''
    }
  })
  document.getElementById('export-btn').addEventListener('click', exportGraph)
  document.getElementById('codegen-btn').addEventListener('click', runCodegen)
  document.getElementById('focus-input-btn').addEventListener('click', () => focusInputNode())
  document.getElementById('delete-btn').addEventListener('click', () => deleteSelected())
  document.getElementById('run-shapes-btn').addEventListener('click', () => runRuntimeShapeCheck())
  document.getElementById('batch-size').addEventListener('input', (e) => {
    state.batchSize = Math.max(1, Math.trunc(Number(e.target.value) || 2))
    state.runtimeShapes = null
    state.runtimeError = null
    state.runtimeErrorNodeId = null
    queueValidation()
  })
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault()
      document.getElementById('search').focus()
      return
    }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      if (isEditingText(document.activeElement)) return
      e.preventDefault()
      deleteSelected()
    }
  })

  await Promise.all([loadManifest(), loadBlockInfo()])
  queueValidation()
}

/** Per-block reference docs scraped from RealityShifts/Srivastava-book-of-Blocks-diagrams. */
async function loadBlockInfo() {
  try {
    const res = await fetch('/block_info.json')
    if (!res.ok) return
    const obj = await res.json()
    state.blockInfo = new Map(Object.entries(obj))
  } catch {
    state.blockInfo = new Map()
  }
}

async function loadManifest() {
  const url = `/manifests/${state.framework}.json`
  const res = await fetch(url)
  const fetched = await res.json()
  // Prepend the built-in Input node so it's always available regardless of framework.
  state.entries = [INPUT_ENTRY, ...fetched]
  state.byName = new Map(state.entries.map((e) => [e.name, e]))
  renderPalette(document.getElementById('palette'), state.entries, (entry) =>
    createNode(entry.name)
  )
}

async function createNode(name, pos) {
  const entry = state.byName.get(name)
  if (!entry) return null
  const node = makeNode(entry)
  await editor.addNode(node)
  if (pos) await area.translate(node.id, pos)
  state.selectedNodeId = node.id
  refreshInspector()
  return node
}

async function clearGraph() {
  for (const c of [...editor.getConnections()]) await editor.removeConnection(c.id)
  for (const n of [...editor.getNodes()]) await editor.removeNode(n.id)
  state.selectedNodeId = null
  state.runtimeShapes = null
  state.runtimeError = null
  state.runtimeErrorNodeId = null
  refreshInspector()
  queueValidation()
}

async function importGraph(data) {
  if (!data || typeof data !== 'object') {
    throw new Error('Invalid graph JSON: expected object')
  }
  const nodesIn = Array.isArray(data.nodes) ? data.nodes : []
  const connsIn = Array.isArray(data.connections) ? data.connections : []
  const frameworkIn = data.framework

  if (frameworkIn && frameworkIn !== state.framework) {
    if (!['pytorch', 'flax'].includes(frameworkIn)) {
      throw new Error(`Unsupported framework in file: ${frameworkIn}`)
    }
    state.framework = frameworkIn
    document.getElementById('framework-select').value = frameworkIn
    await loadManifest()
  }

  await clearGraph()

  const idMap = new Map()
  let dropped = 0
  for (const spec of nodesIn) {
    const name = spec?.name
    if (typeof name !== 'string') {
      dropped++
      continue
    }
    const pos =
      spec?.position &&
      Number.isFinite(spec.position.x) &&
      Number.isFinite(spec.position.y)
        ? { x: spec.position.x, y: spec.position.y }
        : undefined
    const node = await createNode(name, pos)
    if (!node) {
      dropped++
      continue
    }
    if (spec?.values && typeof spec.values === 'object') {
      Object.assign(node.values, spec.values)
    }
    idMap.set(spec.id, node.id)
  }

  let restoredConnections = 0
  for (const c of connsIn) {
    const source = idMap.get(c?.source)
    const target = idMap.get(c?.target)
    if (!source || !target) {
      dropped++
      continue
    }
    const srcNode = editor.getNode(source)
    const tgtNode = editor.getNode(target)
    if (!srcNode || !tgtNode) {
      dropped++
      continue
    }
    try {
      const conn = new ClassicPreset.Connection(srcNode, c.sourceOutput, tgtNode, c.targetInput)
      await editor.addConnection(conn)
      restoredConnections++
    } catch {
      dropped++
    }
  }

  queueValidation()
  return { nodes: idMap.size, connections: restoredConnections, dropped }
}

/** Delete every selected node (and the picked node as a fallback). */
async function deleteSelected() {
  // Collect ids from the selector; fall back to the last picked node so a
  // single click + Delete still works even without an explicit selection box.
  const ids = new Set()
  for (const n of editor.getNodes()) {
    if (selector?.isSelected({ label: 'node', id: n.id })) ids.add(n.id)
  }
  if (ids.size === 0 && state.selectedNodeId) ids.add(state.selectedNodeId)
  if (ids.size === 0) return

  // Remove incident connections first - rete throws otherwise.
  for (const c of [...editor.getConnections()]) {
    if (ids.has(c.source) || ids.has(c.target)) {
      await editor.removeConnection(c.id)
    }
  }
  for (const id of ids) {
    selector?.remove({ label: 'node', id })
    await editor.removeNode(id)
  }
  if (ids.has(state.selectedNodeId)) state.selectedNodeId = null
  state.runtimeShapes = null
  state.runtimeError = null
  state.runtimeErrorNodeId = null
  refreshInspector()
  queueValidation()
}

function isEditingText(el) {
  if (!el) return false
  const tag = el.tagName
  return (
    tag === 'INPUT' ||
    tag === 'TEXTAREA' ||
    tag === 'SELECT' ||
    el.isContentEditable === true
  )
}

let validateTimer = null
function queueValidation() {
  clearTimeout(validateTimer)
  validateTimer = setTimeout(runValidation, 60)
}
function runValidation() {
  // Validate + back-fill implicit ctor params (e.g. infer in_ch from
  // resolved C_in after wiring from a previous layer).
  let result = validate(editor)
  for (let i = 0; i < 3; i++) {
    const changed = inferImplicitCtorParams(result)
    if (!changed) break
    result = validate(editor)
  }
  state.lastResult = result
  editor.__lastValidationSub = state.lastResult.sub
  const concrete =
    state.framework === 'pytorch'
      ? isFullyConcrete(editor, state.lastResult.sub, state.batchSize)
      : { ok: false }
  state.lastResult.canRunShapes = concrete.ok
  state.lastResult.concreteReason = concrete.reason
  renderDiagnostics(document.getElementById('diag-list'), state.lastResult)
  refreshInspector()
  refreshRuntimePanel()
  applyRuntimeErrorHighlight()
}

function inferImplicitCtorParams(result) {
  let changed = false
  for (const n of editor.getNodes()) {
    if (!n?.entry || n.entry.kind === 'input') continue
    const ctorByName = new Map((n.entry.ctor || []).map((p) => [p.name, p]))
    for (const [axis, paramName] of Object.entries(n.entry.bindings || {})) {
      const cur = n.values?.[paramName]
      if (cur !== null && cur !== undefined && cur !== '') continue
      const param = ctorByName.get(paramName)
      const resolved = resolve(`${axis}#${n.id}`, result.sub)
      if (typeof resolved !== 'number' || !Number.isFinite(resolved)) continue
      if (param?.type === 'float') n.values[paramName] = resolved
      else n.values[paramName] = Math.trunc(resolved)
      changed = true
    }
  }
  return changed
}

function refreshRuntimePanel() {
  updateRuntimePanel({
    framework: state.framework,
    lastResult: state.lastResult,
    batchSize: state.batchSize,
    runtimeShapes: state.runtimeShapes,
    running: state.runtimeRunning,
    lastError: state.runtimeError,
  })
}

async function runRuntimeShapeCheck() {
  if (state.runtimeRunning || state.framework !== 'pytorch') return
  state.runtimeRunning = true
  state.runtimeError = null
  state.runtimeErrorNodeId = null
  refreshRuntimePanel()
  applyRuntimeErrorHighlight()
  try {
    const { shapes } = await runShapeCheck(editor, state.framework, state.batchSize)
    state.runtimeShapes = shapes
    state.runtimeErrorNodeId = null
  } catch (e) {
    state.runtimeShapes = null
    state.runtimeError = e.message || String(e)
    state.runtimeErrorNodeId = e.nodeId || null
  } finally {
    state.runtimeRunning = false
    refreshRuntimePanel()
    applyRuntimeErrorHighlight()
    refreshInspector()
  }
}

function refreshInspector() {
  const node = state.selectedNodeId ? editor.getNode(state.selectedNodeId) : null
  renderInspector(
    document.getElementById('inspector-body'),
    node,
    state.lastResult?.sub ?? new Map(),
    () => {
      state.runtimeShapes = null
      state.runtimeError = null
      state.runtimeErrorNodeId = null
      queueValidation()
    },
    state.runtimeShapes,
    state.blockInfo
  )
}

function flashDiagnostic(text) {
  const ul = document.getElementById('diag-list')
  const li = document.createElement('li')
  li.className = 'err'
  li.textContent = text
  ul.prepend(li)
  setTimeout(() => li.remove(), 3500)
}

async function focusInputNode() {
  const nodes = editor.getNodes()
  const input = nodes.find((n) => n.entry?.kind === 'input')
  if (!input) {
    flashDiagnostic('No Input node found')
    return
  }
  state.selectedNodeId = input.id
  refreshInspector()
  // Zoom and center view around the input node.
  await AreaExtensions.zoomAt(area, [input])
}

function applyRuntimeErrorHighlight() {
  // Remove old markers.
  for (const [, view] of area.nodeViews) {
    view.element?.classList?.remove('runtime-error-node')
  }
  // Mark the failing node (if any) with a red outline.
  if (!state.runtimeErrorNodeId) return
  const v = area.nodeViews.get(state.runtimeErrorNodeId)
  if (v?.element?.classList) {
    v.element.classList.add('runtime-error-node')
  }
}

function exportGraph() {
  const data = getGraphData()
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `graph-${state.framework}.json`
  a.click()
  URL.revokeObjectURL(a.href)
}

function getGraphData() {
  return {
    framework: state.framework,
    nodes: editor.getNodes().map((n) => ({
      id: n.id,
      name: n.entry.name,
      values: n.values,
      position: (() => {
        const view = area?.nodeViews?.get(n.id)
        const p = view?.position
        if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) return undefined
        return { x: p.x, y: p.y }
      })(),
    })),
    connections: editor.getConnections().map((c) => ({
      source: c.source,
      sourceOutput: c.sourceOutput,
      target: c.target,
      targetInput: c.targetInput,
    })),
  }
}

function runCodegen() {
  const code = generateCode(editor.getNodes(), editor.getConnections(), state.framework)
  showCode(code)
}

bootstrap().catch((err) => {
  console.error(err)
  alert(`Failed to bootstrap editor: ${err.message}`)
})

// Test / inspection harness. Exposed on window so headless tests can drive
// the editor without simulating mouse interactions. Not relied on by the UI.
if (typeof window !== 'undefined') {
  window.__blocks = {
    get editor() {
      return editor
    },
    get area() {
      return area
    },
    state,
    createNode,
    clearGraph,
    deleteSelected,
    runValidation,
    runRuntimeShapeCheck,
    getGraphData,
    importGraph,
    runCodegen: () =>
      generateCode(editor.getNodes(), editor.getConnections(), state.framework),
    addConnection: async (source, sourceOutput, target, targetInput) => {
      const srcNode = editor.getNode(source)
      const tgtNode = editor.getNode(target)
      if (!srcNode || !tgtNode) throw new Error('node not found')
      // Use ClassicPreset connection so the area pipe runs (including validation).
      const { ClassicPreset } = await import('rete')
      const conn = new ClassicPreset.Connection(
        srcNode,
        sourceOutput,
        tgtNode,
        targetInput
      )
      return editor.addConnection(conn)
    },
  }
}
