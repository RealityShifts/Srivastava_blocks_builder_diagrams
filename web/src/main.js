/**
 * Bootstrap the Rete v2 editor with the Lit render plugin, wire palette /
 * inspector / diagnostics panels, and gate connection creation on the
 * unification-based validator so invalid edges are rejected at the source.
 */

import { NodeEditor, ClassicPreset } from 'rete'
import { AreaPlugin, AreaExtensions } from 'rete-area-plugin'
import { ConnectionPlugin, Presets as ConnectionPresets } from 'rete-connection-plugin'
import { LitPlugin, Presets as LitPresets } from '@retejs/lit-plugin'

import {
  makeNode,
  INPUT_ENTRY,
  CONST_ENTRY,
  REARRANGE_ENTRY,
  RESHAPE_ENTRY,
  CONCAT_ENTRY,
  STACK_ENTRY,
  applyNodeTag,
  parseShapeString,
} from './nodes.js'
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
  restoring: false, // true while restoreFromAutosave is mutating the editor
  clipboard: null, // in-memory copy of last copy/duplicate (mirrors localStorage)
}

// --- autosave (localStorage, single slot, debounced) ---
const AUTOSAVE_KEY = 'blocks-builder:autosave:v1'
const AUTOSAVE_VERSION = 1
const AUTOSAVE_DEBOUNCE_MS = 800

let autosaveTimer = null
function queueAutosave() {
  if (state.restoring) return
  clearTimeout(autosaveTimer)
  autosaveTimer = setTimeout(saveToStorage, AUTOSAVE_DEBOUNCE_MS)
}

function saveToStorage() {
  try {
    const payload = {
      version: AUTOSAVE_VERSION,
      savedAt: Date.now(),
      batchSize: state.batchSize,
      graph: getGraphData(),
    }
    localStorage.setItem(AUTOSAVE_KEY, JSON.stringify(payload))
  } catch {
    // Storage may be unavailable (private mode, quota, disabled cookies).
    // Better to silently skip than crash the editor.
  }
}

function loadFromStorage() {
  try {
    const raw = localStorage.getItem(AUTOSAVE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (parsed?.version !== AUTOSAVE_VERSION) return null
    return parsed
  } catch {
    return null
  }
}

// --- clipboard (copy / paste / duplicate selected nodes) ---
const CLIPBOARD_KEY = 'blocks-builder:clipboard:v1'
const PASTE_OFFSET_PX = 24

function selectedNodeIds() {
  const ids = new Set()
  for (const n of editor.getNodes()) {
    if (selector?.isSelected({ label: 'node', id: n.id })) ids.add(n.id)
  }
  if (ids.size === 0 && state.selectedNodeId) ids.add(state.selectedNodeId)
  return ids
}

function nodePosition(n) {
  const view = area?.nodeViews?.get(n.id)
  const p = view?.position
  if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) return undefined
  return { x: p.x, y: p.y }
}

function isParamInput(node, inputName) {
  return node?.inputs?.[inputName]?.portSpec?.kind === 'param'
}

function reverseBindings(entry) {
  const out = new Map()
  for (const [axis, param] of Object.entries(entry.bindings || {})) out.set(param, axis)
  return out
}

function guessDimFromTokens(tokens, axisHint) {
  if (!tokens?.length) return null
  const asNum = (v) => {
    const n = Number(v)
    return Number.isFinite(n) ? Math.trunc(n) : null
  }
  const axis = String(axisHint || '').toUpperCase()
  const byIdx = { B: 0, C: 1, H: 2, W: 3, T: 1, D: 1, N: 1 }
  if (axis && byIdx[axis[0]] != null) {
    const n = asNum(tokens[byIdx[axis[0]]])
    if (n != null) return n
  }
  for (const t of tokens) {
    const n = asNum(t)
    if (n != null) return n
  }
  return null
}

function parseConstValue(sourceNode, targetParamType = 'int') {
  if (sourceNode?.entry?.kind !== 'const') return null
  const raw = sourceNode.values?.value
  const declared = sourceNode.values?.value_type || targetParamType
  if (declared === 'bool') {
    if (raw === true || raw === 'true' || raw === '1' || raw === 1) return true
    if (raw === false || raw === 'false' || raw === '0' || raw === 0) return false
    return null
  }
  if (declared === 'str') return String(raw ?? '')
  const n = Number(raw)
  if (!Number.isFinite(n)) return null
  if (declared === 'int') return Math.trunc(n)
  return n
}

/**
 * Pull ctor param values from dedicated parameter edges.
 * - Constant -> 🔴param: direct value
 * - Input    -> 🔴param: best-effort pick from input shape tokens using binding hint
 */
function applyCtorValuesFromParamEdges(result) {
  let changed = false
  for (const c of editor.getConnections()) {
    const target = editor.getNode(c.target)
    const source = editor.getNode(c.source)
    if (!target || !source) continue
    const spec = target.inputs?.[c.targetInput]?.portSpec
    if (spec?.kind !== 'param') continue
    const paramName = spec.paramName
    const paramDef = (target.entry.ctor || []).find((p) => p.name === paramName)
    if (!paramDef) continue

    let next = parseConstValue(source, paramDef.type)
    if (next == null && source.entry.kind === 'input') {
      const axisHint = reverseBindings(target.entry).get(paramName) || ''
      const toks = parseShapeString(source.values?.shape)
      next = guessDimFromTokens(toks, axisHint)
    }
    if (next == null && result?.sub) {
      const axisHint = reverseBindings(target.entry).get(paramName)
      const outShape = source.freshenedShape?.(c.sourceOutput, 'out')
      if (outShape) {
        const resolved = outShape.map((t) => resolve(t, result.sub))
        next = guessDimFromTokens(resolved, axisHint)
      }
    }
    if (next == null) continue
    if (target.values[paramName] !== next) {
      target.values[paramName] = next
      changed = true
    }
  }
  return changed
}

function styleConnectionPath(connectionId, isParam) {
  const view = area?.connectionViews?.get(connectionId)
  const path = view?.element
    ?.querySelector('rete-connection-wrapper')
    ?.shadowRoot?.querySelector('rete-connection')
    ?.shadowRoot?.querySelector('path')
  if (!path) return
  if (isParam) {
    path.style.strokeDasharray = '6 5'
    path.style.opacity = '0.45'
    path.style.stroke = '#ef4444'
  } else {
    path.style.strokeDasharray = ''
    path.style.opacity = ''
    path.style.stroke = ''
  }
}

function applyAllConnectionStyles() {
  for (const c of editor.getConnections()) {
    const target = editor.getNode(c.target)
    styleConnectionPath(c.id, isParamInput(target, c.targetInput))
  }
}

/** Snapshot selection (or focused node) into in-memory + localStorage. */
function copySelection() {
  const ids = selectedNodeIds()
  if (ids.size === 0) return false
  const payload = {
    version: 1,
    framework: state.framework,
    nodes: [],
    connections: [],
  }
  for (const n of editor.getNodes()) {
    if (!ids.has(n.id)) continue
    payload.nodes.push({
      id: n.id,
      name: n.entry.name,
      tag: n.tag ?? '',
      values: { ...n.values },
      position: nodePosition(n),
    })
  }
  for (const c of editor.getConnections()) {
    if (ids.has(c.source) && ids.has(c.target)) {
      payload.connections.push({
        source: c.source,
        sourceOutput: c.sourceOutput,
        target: c.target,
        targetInput: c.targetInput,
      })
    }
  }
  state.clipboard = payload
  try {
    localStorage.setItem(CLIPBOARD_KEY, JSON.stringify(payload))
  } catch {
    // localStorage not available - in-memory clipboard still works for this tab.
  }
  return true
}

function readClipboard() {
  if (state.clipboard) return state.clipboard
  try {
    const raw = localStorage.getItem(CLIPBOARD_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (parsed?.version !== 1) return null
    return parsed
  } catch {
    return null
  }
}

/** Paste with +offset and re-mapped IDs; only restore intra-selection edges. */
async function pasteClipboard() {
  const payload = readClipboard()
  if (!payload || !Array.isArray(payload.nodes) || payload.nodes.length === 0) return
  if (payload.framework && payload.framework !== state.framework) {
    flashDiagnostic(`Clipboard is from ${payload.framework}; switch framework to paste.`)
    return
  }
  // Deselect the existing selection so the newly-pasted copies are what gets
  // selected after the paste (the area-pipe nodepicked fires per createNode).
  if (selector) {
    for (const n of editor.getNodes()) selector.remove({ label: 'node', id: n.id })
  }
  const idMap = new Map()
  for (const spec of payload.nodes) {
    const pos = spec.position
      ? { x: spec.position.x + PASTE_OFFSET_PX, y: spec.position.y + PASTE_OFFSET_PX }
      : undefined
    const node = await createNode(spec.name, pos)
    if (!node) continue
    if (spec.values) Object.assign(node.values, spec.values)
    if (typeof spec.tag === 'string' && spec.tag) {
      applyNodeTag(node, spec.tag)
      area.update('node', node.id)
    }
    idMap.set(spec.id, node.id)
  }
  for (const c of payload.connections ?? []) {
    const source = idMap.get(c.source)
    const target = idMap.get(c.target)
    if (!source || !target) continue
    const srcNode = editor.getNode(source)
    const tgtNode = editor.getNode(target)
    if (!srcNode || !tgtNode) continue
    try {
      const conn = new ClassicPreset.Connection(srcNode, c.sourceOutput, tgtNode, c.targetInput)
      await editor.addConnection(conn)
    } catch {
      // dryRunEdge in the connection pipe may reject if the paste lands in a
      // place that creates a shape conflict; silent skip is fine here.
    }
  }
  refreshInspector()
  queueValidation()
  queueAutosave()
}

async function duplicateSelection() {
  if (copySelection()) await pasteClipboard()
}

async function restoreFromAutosave() {
  const payload = loadFromStorage()
  if (!payload?.graph) return
  if (!Array.isArray(payload.graph.nodes) || payload.graph.nodes.length === 0) return
  state.restoring = true
  try {
    const stats = await importGraph(payload.graph)
    if (Number.isFinite(payload.batchSize)) {
      state.batchSize = Math.max(1, Math.trunc(payload.batchSize))
    }
    const note =
      stats.dropped > 0
        ? `Restored ${stats.nodes} node(s) from autosave (dropped ${stats.dropped})`
        : `Restored ${stats.nodes} node(s) from autosave`
    flashDiagnostic(note)
  } catch (err) {
    flashDiagnostic(`Autosave restore failed: ${err.message || String(err)}`)
  } finally {
    state.restoring = false
  }
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
    if (structuralSignals.has(context.type)) {
      queueValidation()
      queueAutosave()
      if (context.type === 'connectioncreated') {
        const c = context.data
        const target = editor.getNode(c.target)
        const param = isParamInput(target, c.targetInput)
        setTimeout(() => styleConnectionPath(c.id, param), 0)
      }
    }
    return context
  })

  // Track selection for the inspector; also save after a node drag finishes.
  area.addPipe((context) => {
    if (context.type === 'nodepicked') {
      state.selectedNodeId = context.data.id
      refreshInspector()
    } else if (context.type === 'nodetranslated') {
      queueAutosave()
      applyAllConnectionStyles()
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
  document.getElementById('duplicate-btn').addEventListener('click', () => duplicateSelection())
  document.getElementById('delete-btn').addEventListener('click', () => deleteSelected())
  document.getElementById('run-shapes-btn').addEventListener('click', () => runRuntimeShapeCheck())
  document.getElementById('batch-size').addEventListener('input', (e) => {
    state.batchSize = Math.max(1, Math.trunc(Number(e.target.value) || 2))
    state.runtimeShapes = null
    state.runtimeError = null
    state.runtimeErrorNodeId = null
    queueValidation()
    queueAutosave()
  })
  document.addEventListener('keydown', (e) => {
    const mod = e.ctrlKey || e.metaKey
    const key = e.key.toLowerCase()
    if (mod && key === 'k') {
      e.preventDefault()
      document.getElementById('search').focus()
      return
    }
    // Skip clipboard/delete handlers while the user is typing in any control
    // so native copy/paste/text editing in inputs still works.
    if (isEditingText(document.activeElement)) return

    if (mod && key === 'c') {
      if (copySelection()) e.preventDefault()
      return
    }
    if (mod && key === 'v') {
      e.preventDefault()
      pasteClipboard()
      return
    }
    if (mod && key === 'd') {
      e.preventDefault()
      duplicateSelection()
      return
    }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault()
      deleteSelected()
    }
  })

  await Promise.all([loadManifest(), loadBlockInfo()])
  await restoreFromAutosave()
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
  // Keep Constant available for createNode/import, but hide it from the
  // palette so users spawn it explicitly from node params ("+ const").
  state.entries = [
    INPUT_ENTRY,
    CONST_ENTRY,
    REARRANGE_ENTRY,
    RESHAPE_ENTRY,
    CONCAT_ENTRY,
    STACK_ENTRY,
    ...fetched,
  ]
  state.byName = new Map(state.entries.map((e) => [e.name, e]))
  const paletteEntries = state.entries.filter((e) => e.kind !== 'const')
  renderPalette(document.getElementById('palette'), paletteEntries, (entry) =>
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
    for (const p of spec?.exposedParams || []) {
      node.exposeParam?.(p)
    }
    if (typeof spec?.tag === 'string' && spec.tag) {
      applyNodeTag(node, spec.tag)
      area.update('node', node.id)
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
    if (String(c.targetInput || '').startsWith('__param__')) {
      const pName = String(c.targetInput).replace(/^__param__/, '')
      tgtNode.exposeParam?.(pName)
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
    const fromEdges = applyCtorValuesFromParamEdges(result)
    const fromBindings = inferImplicitCtorParams(result)
    const changed = fromEdges || fromBindings
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
  applyAllConnectionStyles()
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
      queueAutosave()
    },
    state.runtimeShapes,
    state.blockInfo,
    (n, newTag) => {
      applyNodeTag(n, newTag)
      area.update('node', n.id)
      state.runtimeShapes = null
      state.runtimeError = null
      state.runtimeErrorNodeId = null
      queueValidation()
      queueAutosave()
    },
    async (targetNode, param) => {
      await addConstantForParam(targetNode, param)
    },
    async (targetNode, param, shouldExpose) => {
      const key = `__param__${param.name}`
      if (shouldExpose) {
        targetNode.exposeParam?.(param.name)
      } else {
        for (const c of [...editor.getConnections()]) {
          if (c.target === targetNode.id && c.targetInput === key) {
            await editor.removeConnection(c.id)
          }
        }
        targetNode.hideParam?.(param.name)
      }
      await area.update('node', targetNode.id)
      queueValidation()
      queueAutosave()
      refreshInspector()
    }
  )
}

async function addConstantForParam(targetNode, param) {
  const host = area?.nodeViews?.get(targetNode.id)?.position ?? { x: 0, y: 0 }
  const pos = { x: host.x - 260, y: host.y + 22 * Math.max(0, targetNode.entry.ctor.indexOf(param)) }
  const c = await createNode('Constant', pos)
  if (!c) return
  targetNode.exposeParam?.(param.name)
  await area.update('node', targetNode.id)

  const t = param?.type || 'int'
  c.values.value_type = t === 'float' || t === 'bool' || t === 'str' ? t : 'int'
  const existing = targetNode.values?.[param.name]
  if (existing !== null && existing !== undefined && existing !== '') {
    c.values.value = String(existing)
  } else if (c.values.value_type === 'bool') {
    c.values.value = 'false'
  } else if (c.values.value_type === 'str') {
    c.values.value = ''
  } else {
    c.values.value = '1'
  }
  const targetInput = `__param__${param.name}`
  try {
    const conn = new ClassicPreset.Connection(c, 'out', targetNode, targetInput)
    await editor.addConnection(conn)
  } catch (e) {
    flashDiagnostic(`Failed to wire constant: ${e.message || String(e)}`)
  }
  queueValidation()
  queueAutosave()
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
      tag: n.tag ?? '',
      values: n.values,
      exposedParams: Object.keys(n.inputs || {})
        .filter((k) => k.startsWith('__param__'))
        .map((k) => k.replace(/^__param__/, '')),
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
    saveToStorage,
    loadFromStorage,
    restoreFromAutosave,
    queueAutosave,
    copySelection,
    pasteClipboard,
    duplicateSelection,
    applyNodeTag,
    AUTOSAVE_KEY,
    CLIPBOARD_KEY,
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
