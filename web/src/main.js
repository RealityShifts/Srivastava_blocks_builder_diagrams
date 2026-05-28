/**
 * Bootstrap the Rete v2 editor with the Lit render plugin, wire palette /
 * inspector / diagnostics panels, and gate connection creation on the
 * unification-based validator so invalid edges are rejected at the source.
 */

import { NodeEditor, ClassicPreset } from 'rete'
import { AreaPlugin, AreaExtensions } from 'rete-area-plugin'
import { ConnectionPlugin, Presets as ConnectionPresets } from 'rete-connection-plugin'
import { setupNodeSelection } from './selection.js'
import { LitPlugin, Presets as LitPresets } from '@retejs/lit-plugin'

import {
  boundarySignatureFromBoundary,
  boundarySignatureFromEntry,
  boundarySignaturesMatch,
  applySignatureToBoundary,
} from './groupBoundary.js'
import { copyNodeValues, nodeTagKey, nodesInSameTagFamily } from './tagSync.js'
import {
  makeTagAtlas,
  registerNodeMember,
  registerGroupMember,
  unregisterMember,
  adoptValuesFromAtlas,
  adoptExposedParamsFromAtlas,
  recordValueChange,
  recordAllValues,
  recordExposedParamChange,
  recordGroupMeta,
  rebuildAtlas,
  atlasSummary,
  getAtlasEntry,
} from './tagAtlas.js'
import {
  makeNode,
  makeGroupEntry,
  INPUT_ENTRY,
  OUTPUT_ENTRY,
  CONST_ENTRY,
  LEARNABLE_TENSOR_ENTRY,
  REARRANGE_ENTRY,
  RESHAPE_ENTRY,
  CONCAT_ENTRY,
  STACK_ENTRY,
  POOL_ENTRY,
  UPSAMPLE_ENTRY,
  applyNodeTag,
  computeNodeLabel,
  colorForTag,
  parseShapeString,
} from './nodes.js'
import { validate, dryRunEdge } from './validator.js'
import { generate as generateCode } from './codegen.js'
import { isFullyConcrete, runShapeCheck, resolveInputSpecs } from './runtime.js'
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
  runtimeNumParams: null,
  batchSize: 2,
  runtimeRunning: false,
  runtimeError: null,
  runtimeErrorNodeId: null,
  restoring: false, // true while restoreFromAutosave is mutating the editor
  clipboard: null, // in-memory copy of last copy/duplicate (mirrors localStorage)
  // groupId -> { id, name, description, facadeTag, collapsed, facadeNodeId, portMap,
  //              savedPosition, childOffsets }
  // See groupSelected/expandGroup/collapseGroup for the lifecycle. The portMap
  // is the source of truth used by validator and codegen to "see through" a
  // collapsed facade back to its underlying child ports. childOffsets is
  // {childId: {dx, dy}} relative to the facade and anchors the children to
  // the facade across collapse/expand cycles so dragging the collapsed
  // facade moves the whole subgraph (no snap-back on expand).
  groups: new Map(),
  // Single source of truth for everything tagged. Tag key (lowercased) ->
  // canonical AtlasEntry { tag, family, values, exposedParams, members, ... }.
  // Members can be node ids (BlockNodes) or group ids (state.groups entries).
  // See tagAtlas.js for the shape and the public API.
  tagAtlas: makeTagAtlas(),
}

let _groupCounter = 0
function freshGroupId() {
  // Monotonic + random suffix so re-importing on top of an existing graph
  // doesn't collide.
  return `g${++_groupCounter}_${Math.random().toString(36).slice(2, 8)}`
}

/** Short alphanumeric token (base36) used to stamp a unique tag onto group
 *  children. 5 chars ~ 60M combinations - collisions are negligible. */
function randomChildTag() {
  return Math.random().toString(36).slice(2, 7)
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

/** Snapshot each child's position relative to the facade's *actual placed*
 *  position into group.childOffsets, and sync group.savedPosition to it.
 *  Called on collapse so the next expand can place children at
 *  `currentFacadePos + offset` instead of their original absolute coords.
 *  We read the facade's real position from area (rather than trusting the
 *  passed-in center) because Rete snaps positions to a grid, so the post-
 *  translate coords can differ from the requested ones by a few pixels. */
function captureChildOffsets(group, children) {
  if (!group) return
  const facade = group.facadeNodeId ? editor.getNode(group.facadeNodeId) : null
  const facadePos = (facade && nodePosition(facade)) ?? group.savedPosition
  if (!facadePos) return
  group.savedPosition = facadePos
  const out = {}
  for (const child of children) {
    const p = nodePosition(child)
    if (!p) continue
    out[child.id] = { dx: p.x - facadePos.x, dy: p.y - facadePos.y }
  }
  group.childOffsets = out
}

/** Translate group children to `facadePos + offset`. No-op for children
 *  without a recorded offset (leaves them where they are). */
async function applyChildOffsets(group, facadePos) {
  const offsets = group?.childOffsets
  if (!offsets || !facadePos) return
  for (const child of getGroupChildren(group.id)) {
    const off = offsets[child.id]
    if (!off || !Number.isFinite(off.dx) || !Number.isFinite(off.dy)) continue
    await area.translate(child.id, { x: facadePos.x + off.dx, y: facadePos.y + off.dy })
  }
}

/** Rebuild a {oldId: offset} map using a fresh idMap (paste / import). */
function remapChildOffsets(raw, idMap) {
  const out = {}
  if (!raw || typeof raw !== 'object') return out
  for (const [oldId, off] of Object.entries(raw)) {
    if (!off || !Number.isFinite(off.dx) || !Number.isFinite(off.dy)) continue
    const newId = idMap?.get(oldId) ?? oldId
    out[newId] = { dx: off.dx, dy: off.dy }
  }
  return out
}

function applyTagStyle(node) {
  const el = area?.nodeViews?.get(node.id)?.element
  if (!el) return
  // Group membership wins over the user-set tag for visual color so a group
  // and all its children read as one unit. Color is keyed off the group
  // *name*, so two groups both called "Encoder" (e.g. after duplicate)
  // pick up the same hue. We deliberately do NOT touch node.tag - tags
  // mean explicit weight sharing and must stay under user control.
  let color = null
  if (node?.entry?.kind === 'group') {
    const g = state.groups.get(node.entry.groupId)
    if (g?.name) color = colorForTag(g.name)
  } else if (node?.groupId) {
    const g = state.groups.get(node.groupId)
    if (g?.name) color = colorForTag(g.name)
  }
  if (!color) color = colorForTag(node.tag)
  if (!color) {
    el.classList.remove('tagged-node')
    el.style.removeProperty('--tag-color')
    return
  }
  el.classList.add('tagged-node')
  el.style.setProperty('--tag-color', color)
}

function applyAllTagStyles() {
  for (const n of editor.getNodes()) applyTagStyle(n)
}

function persistGroupFacadeTag(group, tag) {
  if (!group) return
  group.facadeTag = String(tag ?? '').trim()
}

function applyNodeTagOnNode(node, tag) {
  applyNodeTag(node, tag)
  if (isGroupFacade(node)) {
    persistGroupFacadeTag(state.groups.get(node.entry.groupId), tag)
  }
}

function restoreNodeTag(node, tag) {
  if (typeof tag !== 'string' || !tag) return
  applyNodeTagOnNode(node, tag)
  area.update('node', node.id)
  applyTagStyle(node)
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
  const initial = selectedNodeIds()
  if (initial.size === 0) return false

  // If a facade is in the selection, pull its children + facade into the
  // payload automatically - copying a "single" group should always copy the
  // whole subgraph, never half of it. Plain child-only selections are left
  // alone (they'll paste as standalone nodes without the group association).
  const ids = new Set(initial)
  const fullGroups = new Set()
  for (const id of initial) {
    const n = editor.getNode(id)
    if (n?.entry?.kind === 'group') fullGroups.add(n.entry.groupId)
  }
  for (const gid of fullGroups) {
    const facade = getFacadeNode(gid)
    if (facade) ids.add(facade.id)
    for (const child of getGroupChildren(gid)) ids.add(child.id)
  }

  const payload = {
    version: 1,
    framework: state.framework,
    nodes: [],
    connections: [],
    groups: [],
  }
  for (const n of editor.getNodes()) {
    if (!ids.has(n.id)) continue
    const spec = {
      id: n.id,
      name: n.entry.name,
      kind: n.entry.kind,
      tag: n.tag ?? '',
      values: { ...n.values },
      position: nodePosition(n),
    }
    if (n.entry.kind === 'group') {
      spec.groupId = n.entry.groupId
      spec.portMap = n.entry.portMap
    } else {
      // Persist which ctor params are exposed as __param__* input ports so a
      // paste can reattach external constants. Without this, expanding the
      // pasted group would re-route const -> child.__param__X onto a child
      // that no longer has that port and the edge would be silently dropped.
      spec.exposedParams = Object.keys(n.inputs || {})
        .filter((k) => k.startsWith('__param__'))
        .map((k) => k.replace(/^__param__/, ''))
      if (n.groupId && fullGroups.has(n.groupId)) {
        // Only persist group membership when the *whole* group was copied;
        // otherwise the child is meant to land as a standalone.
        spec.groupId = n.groupId
      }
    }
    payload.nodes.push(spec)
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
  for (const gid of fullGroups) {
    const g = state.groups.get(gid)
    if (!g) continue
    payload.groups.push({
      id: g.id,
      name: g.name,
      description: g.description ?? '',
      tag: g.facadeTag ?? editor.getNode(g.facadeNodeId)?.tag ?? '',
      collapsed: g.collapsed,
      facadeNodeId: g.facadeNodeId,
      savedPosition: g.savedPosition,
      childOffsets: g.childOffsets ?? {},
    })
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

  // Fresh group ids so a paste alongside the original doesn't collide on the
  // gid registry / facade entry.groupId.
  const gidMap = new Map()
  for (const g of payload.groups ?? []) gidMap.set(g.id, freshGroupId())

  // Phase 1: regular (non-facade) nodes. Facades have to wait until their
  // childNodeIds can be remapped, so split them out.
  const idMap = new Map()
  const facadeSpecs = []
  for (const spec of payload.nodes) {
    if (spec.kind === 'group') {
      facadeSpecs.push(spec)
      continue
    }
    const pos = spec.position
      ? { x: spec.position.x + PASTE_OFFSET_PX, y: spec.position.y + PASTE_OFFSET_PX }
      : undefined
    const node = await createNode(spec.name, pos)
    if (!node) continue
    if (spec.values) Object.assign(node.values, spec.values)
    for (const p of spec.exposedParams || []) {
      node.exposeParam?.(p)
    }
    if (typeof spec.tag === 'string' && spec.tag) {
      restoreNodeTag(node, spec.tag)
    }
    if (spec.groupId && gidMap.has(spec.groupId)) {
      node.groupId = gidMap.get(spec.groupId)
    }
    idMap.set(spec.id, node.id)
  }

  // Phase 2: rebuild each facade with a fresh portMap whose childNodeIds
  // point at the freshly-cloned children.
  for (const spec of facadeSpecs) {
    const newGid = gidMap.get(spec.groupId) ?? freshGroupId()
    if (!gidMap.has(spec.groupId)) gidMap.set(spec.groupId, newGid)
    const portMap = spec.portMap || { inputs: [], outputs: [] }
    const boundary = {
      inputs: (portMap.inputs || []).map((m) => ({
        childNodeId: idMap.get(m.childNodeId) ?? m.childNodeId,
        childPort: m.childPort,
        shape: m.shape,
        dtype: 'any',
      })),
      outputs: (portMap.outputs || []).map((m) => ({
        childNodeId: idMap.get(m.childNodeId) ?? m.childNodeId,
        childPort: m.childPort,
        shape: m.shape,
        dtype: 'any',
      })),
      params: (portMap.params || []).map((m) => ({
        childNodeId: idMap.get(m.childNodeId) ?? m.childNodeId,
        childPort: m.childPort,
        paramName: m.paramName,
        paramType: m.paramType ?? 'int',
      })),
    }
    const entry = makeGroupEntry(newGid, spec.name, boundary)
    const facade = makeNode(entry)
    await editor.addNode(facade)
    if (spec.position) {
      await area.translate(facade.id, {
        x: spec.position.x + PASTE_OFFSET_PX,
        y: spec.position.y + PASTE_OFFSET_PX,
      })
    }
    markFacadeElement(facade.id)
    if (typeof spec.tag === 'string' && spec.tag) {
      restoreNodeTag(facade, spec.tag)
    }
    idMap.set(spec.id, facade.id)
  }

  // Phase 3: register the new state.groups entries.
  for (const g of payload.groups ?? []) {
    const newGid = gidMap.get(g.id)
    if (!newGid) continue
    const newFacadeId = g.facadeNodeId ? idMap.get(g.facadeNodeId) : null
    const facade = newFacadeId ? editor.getNode(newFacadeId) : null
    state.groups.set(newGid, {
      id: newGid,
      name: g.name || 'Group',
      description: g.description ?? '',
      facadeTag: g.tag ?? facade?.tag ?? '',
      collapsed: Boolean(g.collapsed),
      facadeNodeId: newFacadeId ?? null,
      portMap: facade?.entry?.portMap ?? { inputs: [], outputs: [] },
      savedPosition: g.savedPosition
        ? {
            x: g.savedPosition.x + PASTE_OFFSET_PX,
            y: g.savedPosition.y + PASTE_OFFSET_PX,
          }
        : { x: 0, y: 0 },
      // Offsets are deltas to the facade - no need to PASTE_OFFSET them.
      childOffsets: remapChildOffsets(g.childOffsets, idMap),
    })
  }

  // Phase 4: intra-selection edges (id-remapped).
  for (const c of payload.connections ?? []) {
    const source = idMap.get(c.source)
    const target = idMap.get(c.target)
    if (!source || !target) continue
    const srcNode = editor.getNode(source)
    const tgtNode = editor.getNode(target)
    if (!srcNode || !tgtNode) continue
    // Defensive: if the saved edge points at a __param__* port (e.g. a const
    // wired into a child's exposed ctor param), make sure the target node
    // actually has that input port. Mirrors importGraph's behaviour.
    if (String(c.targetInput || '').startsWith('__param__')) {
      const pName = String(c.targetInput).replace(/^__param__/, '')
      tgtNode.exposeParam?.(pName)
    }
    try {
      const conn = new ClassicPreset.Connection(srcNode, c.sourceOutput, tgtNode, c.targetInput)
      await editor.addConnection(conn)
    } catch {
      // dryRunEdge in the connection pipe may reject if the paste lands in a
      // place that creates a shape conflict; silent skip is fine here.
    }
  }

  // Phase 5: ensure collapsed groups stay visually collapsed (CSS).
  if ((payload.groups ?? []).length > 0) applyAllGroupStyles()
  refreshTagAtlas()

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

let editor, area, connection, render, selector, nodeSelection

async function bootstrap() {
  const container = document.getElementById('editor')

  editor = new NodeEditor()
  area = new AreaPlugin(container)
  connection = new ConnectionPlugin()
  render = new LitPlugin()

  nodeSelection = setupNodeSelection(area, editor, {
    onSelectionChanged(nodeId) {
      state.selectedNodeId = nodeId
      refreshInspector()
    },
  })
  selector = nodeSelection.selector
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
    state.runtimeNumParams = null
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
  document.getElementById('codegen-btn').addEventListener('click', () => runCodegen())
  document.getElementById('codegen-test-btn').addEventListener('click', () =>
    runCodegen({ withTest: true })
  )
  document.getElementById('codegen-trace-btn').addEventListener('click', () =>
    runCodegen({ trace: true, withTest: true })
  )
  document.getElementById('focus-input-btn').addEventListener('click', () => focusInputNode())
  document.getElementById('duplicate-btn').addEventListener('click', () => duplicateSelection())
  document.getElementById('delete-btn').addEventListener('click', () => deleteSelected())
  document.getElementById('group-btn').addEventListener('click', () => groupSelected())
  document.getElementById('ungroup-btn').addEventListener('click', () => ungroupFocused())
  document.getElementById('add-to-group-btn').addEventListener('click', () => addToGroupFocused())
  document.getElementById('collapse-all-btn').addEventListener('click', () => collapseAllGroups())
  document.getElementById('expand-all-btn').addEventListener('click', () => expandAllGroups())
  document.getElementById('run-shapes-btn').addEventListener('click', () => runRuntimeShapeCheck())
  document.getElementById('batch-size').addEventListener('input', (e) => {
    state.batchSize = Math.max(1, Math.trunc(Number(e.target.value) || 2))
    state.runtimeShapes = null
    state.runtimeNumParams = null
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
    if (mod && key === 'g') {
      e.preventDefault()
      if (e.shiftKey) ungroupFocused()
      else if (e.altKey) void addToGroupFocused()
      else groupSelected()
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
  const builtins = [
    INPUT_ENTRY,
    OUTPUT_ENTRY,
    CONST_ENTRY,
    LEARNABLE_TENSOR_ENTRY,
    REARRANGE_ENTRY,
    RESHAPE_ENTRY,
    CONCAT_ENTRY,
    STACK_ENTRY,
    POOL_ENTRY,
    UPSAMPLE_ENTRY,
  ]
  // Built-in / utility names always win. The manifest is auto-generated and
  // historically picked up stale module-kind aliases (e.g. Pool2d, Upsample)
  // that would emit broken imports if a palette click resolved to them.
  const builtinNames = new Set(builtins.map((e) => e.name))
  state.entries = [...builtins, ...fetched.filter((e) => !builtinNames.has(e.name))]
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
  state.runtimeNumParams = null
  state.runtimeError = null
  state.runtimeErrorNodeId = null
  state.groups.clear()
  state.tagAtlas = makeTagAtlas()
  refreshInspector()
  queueValidation()
}

/** Rebuild the tag atlas from the current editor + groups (post import / paste). */
function refreshTagAtlas() {
  state.tagAtlas = rebuildAtlas({
    editor,
    groups: state.groups,
    getNode: (id) => editor.getNode(id),
  })
}

async function importGraph(data) {
  if (!data || typeof data !== 'object') {
    throw new Error('Invalid graph JSON: expected object')
  }
  const nodesIn = Array.isArray(data.nodes) ? data.nodes : []
  const connsIn = Array.isArray(data.connections) ? data.connections : []
  const groupsIn = Array.isArray(data.groups) ? data.groups : []
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

  // Phase A: split specs into regular nodes vs group facades. Regular nodes
  // are created first so the facade's portMap childNodeIds can be remapped
  // to the freshly-allocated ids.
  const facadeSpecs = []
  const regularSpecs = []
  for (const s of nodesIn) {
    if (s?.kind === 'group') facadeSpecs.push(s)
    else regularSpecs.push(s)
  }

  const idMap = new Map()
  let dropped = 0

  // Phase B: regular nodes.
  for (const spec of regularSpecs) {
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
      restoreNodeTag(node, spec.tag)
    }
    if (typeof spec?.groupId === 'string') {
      node.groupId = spec.groupId
    }
    idMap.set(spec.id, node.id)
  }

  // Phase C: facade nodes, rebuilt from saved portMap with remapped child ids.
  for (const spec of facadeSpecs) {
    const portMap = spec?.portMap || { inputs: [], outputs: [] }
    const remap = (m) => ({
      ...m,
      childNodeId: idMap.get(m.childNodeId) ?? m.childNodeId,
    })
    const boundary = {
      inputs: (portMap.inputs || []).map((m) => ({
        childNodeId: idMap.get(m.childNodeId) ?? m.childNodeId,
        childPort: m.childPort,
        shape: m.shape,
        dtype: 'any',
      })),
      outputs: (portMap.outputs || []).map((m) => ({
        childNodeId: idMap.get(m.childNodeId) ?? m.childNodeId,
        childPort: m.childPort,
        shape: m.shape,
        dtype: 'any',
      })),
      params: (portMap.params || []).map((m) => ({
        childNodeId: idMap.get(m.childNodeId) ?? m.childNodeId,
        childPort: m.childPort,
        paramName: m.paramName,
        paramType: m.paramType ?? 'int',
      })),
    }
    const entry = makeGroupEntry(spec.groupId, spec.name, boundary)
    const facade = makeNode(entry)
    await editor.addNode(facade)
    const pos =
      spec?.position &&
      Number.isFinite(spec.position.x) &&
      Number.isFinite(spec.position.y)
        ? { x: spec.position.x, y: spec.position.y }
        : undefined
    if (pos) await area.translate(facade.id, pos)
    markFacadeElement(facade.id)
    if (typeof spec?.tag === 'string' && spec.tag) {
      restoreNodeTag(facade, spec.tag)
    }
    idMap.set(spec.id, facade.id)
  }

  // Phase D: rebuild state.groups from the saved descriptors.
  state.groups.clear()
  for (const g of groupsIn) {
    if (!g?.id) continue
    const newFacadeId = g.facadeNodeId ? idMap.get(g.facadeNodeId) : null
    const facade = newFacadeId ? editor.getNode(newFacadeId) : null
    state.groups.set(g.id, {
      id: g.id,
      name: g.name || 'Group',
      description: g.description ?? '',
      facadeTag: g.tag ?? facade?.tag ?? '',
      collapsed: Boolean(g.collapsed),
      facadeNodeId: newFacadeId ?? null,
      portMap: facade?.entry?.portMap ?? { inputs: [], outputs: [] },
      savedPosition: g.savedPosition ?? { x: 0, y: 0 },
      childOffsets: remapChildOffsets(g.childOffsets, idMap),
    })
  }

  // Phase E: connections (id-remapped).
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

  // Phase F: apply collapsed-state CSS so hidden children stay hidden.
  applyAllGroupStyles()
  applyAllTagStyles()
  // Rebuild the tag atlas now that every node + group is back in place.
  refreshTagAtlas()

  queueValidation()
  return {
    nodes: idMap.size,
    connections: restoredConnections,
    groups: state.groups.size,
    dropped,
  }
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

  const peerChildRemovals = new Map()
  for (const id of ids) {
    const n = editor.getNode(id)
    if (!n?.groupId || isGroupFacade(n)) continue
    const childKey = nodeTagKey(n.tag)
    if (!childKey) continue
    const g = state.groups.get(n.groupId)
    if (!g || !groupTag(g)) continue
    if (!peerChildRemovals.has(n.groupId)) peerChildRemovals.set(n.groupId, new Set())
    peerChildRemovals.get(n.groupId).add(childKey)
  }

  // Facades: cascade-delete. Removing a group drops everything it contains
  // along with it (matches the "folder" mental model). To keep children
  // alive after dropping the group, use Ungroup instead.
  for (const id of [...ids]) {
    const n = editor.getNode(id)
    if (!n || !isGroupFacade(n)) continue
    const gid = n.entry.groupId
    for (const child of getGroupChildren(gid)) ids.add(child.id)
    state.groups.delete(gid)
  }

  // Remove incident connections first - rete throws otherwise.
  for (const c of [...editor.getConnections()]) {
    if (ids.has(c.source) || ids.has(c.target)) {
      await editor.removeConnection(c.id)
    }
  }
  for (const id of ids) {
    const n = editor.getNode(id)
    if (n) {
      if (isGroupFacade(n)) unregisterMember(state.tagAtlas, n.tag, n.entry.groupId)
      else unregisterMember(state.tagAtlas, n.tag, id)
    }
    selector?.remove({ label: 'node', id })
    await editor.removeNode(id)
  }
  for (const [sourceGid, tagKeys] of peerChildRemovals) {
    await removePeerChildrenByTags(sourceGid, tagKeys)
  }
  for (const sourceGid of peerChildRemovals.keys()) {
    const tag = groupTag(state.groups.get(sourceGid))
    if (tag) await syncAllTaggedGroupInstances(tag)
  }
  if (ids.has(state.selectedNodeId)) state.selectedNodeId = null
  state.runtimeShapes = null
  state.runtimeNumParams = null
  state.runtimeError = null
  state.runtimeErrorNodeId = null
  refreshInspector()
  queueValidation()
}

// ---------------------------------------------------------------------------
// Subgraph grouping
//
// A group is a real BlockNode (kind:'group') whose ports proxy boundary
// child ports. Children of the group always live in editor.getNodes() with
// node.groupId === <gid>; they're CSS-hidden when the group is collapsed.
// Boundary edges are *rerouted in the model* on collapse/expand so the
// facade really is the only thing the user can wire to when collapsed.
//
// state.groups[gid].portMap is the round-trip table used by validator and
// codegen to translate facade endpoints back to the underlying children.
// ---------------------------------------------------------------------------

function getGroupChildren(groupId) {
  return editor.getNodes().filter((n) => n.groupId === groupId)
}

function isGroupFacade(n) {
  return n?.entry?.kind === 'group'
}

/** Find the facade node carrying a given groupId, if any. */
function getFacadeNode(groupId) {
  for (const n of editor.getNodes()) {
    if (isGroupFacade(n) && n.entry.groupId === groupId) return n
  }
  return null
}

/** Non-empty tag on a group (facade node or persisted facadeTag). */
function groupTag(g) {
  if (!g) return ''
  const facade = g.facadeNodeId ? editor.getNode(g.facadeNodeId) : null
  const tag = String(g.facadeTag ?? facade?.tag ?? '').trim()
  if (tag && !g.facadeTag) persistGroupFacadeTag(g, tag)
  return tag
}

/** Invoke fn(peerGid, peerGroup) for every other group sharing the same tag. */
function forEachPeerGroup(sourceGid, tag, fn) {
  const key = String(tag ?? '').trim().toLowerCase()
  if (!key) return
  for (const [gid, g] of state.groups) {
    if (gid === sourceGid) continue
    if (groupTag(g).toLowerCase() !== key) continue
    fn(gid, g)
  }
}

function applyGroupName(g, name) {
  g.name = name
  const facade = g.facadeNodeId ? editor.getNode(g.facadeNodeId) : null
  if (facade) {
    facade.entry.name = name
    facade.label = computeNodeLabel(facade)
    area.update('node', facade.id)
    applyTagStyle(facade)
  }
  for (const child of getGroupChildren(g.id)) applyTagStyle(child)
}

function applyGroupTag(g, tag) {
  g.facadeTag = String(tag ?? '')
  const facade = g.facadeNodeId ? editor.getNode(g.facadeNodeId) : null
  if (facade) restoreNodeTag(facade, tag)
}

/** Resolve a tagged member id to a live BlockNode (skips group ids). */
function atlasNodeMember(id) {
  const n = editor.getNode(id)
  return n && !isGroupFacade(n) ? n : null
}

/**
 * Mutator: a node's ctor values changed via the inspector. Push the new
 * canonical values into the atlas and mirror them to every peer member.
 */
function syncTaggedNodePeers(sourceNode) {
  if (!nodeTagKey(sourceNode?.tag)) return
  const peers = recordAllValues(state.tagAtlas, sourceNode)
  for (const peerId of peers) {
    const peer = atlasNodeMember(peerId)
    if (!peer) continue
    if (!nodesInSameTagFamily(sourceNode, peer)) continue
    if (copyNodeValues(sourceNode, peer)) {
      area.update('node', peer.id)
      applyTagStyle(peer)
    }
  }
}

/**
 * Mutator: a node just got a new tag. Pull canonical values + exposed-param
 * structure from the atlas onto the node so it lines up with its new peers.
 */
async function adoptTaggedPeerValues(node) {
  const entry = getAtlasEntry(state.tagAtlas, node?.tag)
  if (!entry) return false
  if (entry.family && entry.family.split(':')[1] !== node.entry?.name) return false
  let changed = false
  if (adoptValuesFromAtlas(state.tagAtlas, node).length > 0) changed = true
  const diff = adoptExposedParamsFromAtlas(state.tagAtlas, node)
  for (const name of diff.toExpose) {
    node.exposeParam?.(name)
    changed = true
  }
  for (const name of diff.toHide) {
    for (const c of [...editor.getConnections()]) {
      if (c.target === node.id && c.targetInput === `__param__${name}`) {
        await editor.removeConnection(c.id)
      }
    }
    node.hideParam?.(name)
    changed = true
  }
  if (changed) area.update('node', node.id)
  return true
}

async function syncTaggedPeerParamPort(sourceNode, param, shouldExpose) {
  const key = `__param__${param.name}`
  const peers = recordExposedParamChange(state.tagAtlas, sourceNode, param.name, shouldExpose)
  for (const peerId of peers) {
    const peer = atlasNodeMember(peerId)
    if (!peer) continue
    if (!nodesInSameTagFamily(sourceNode, peer)) continue
    if (shouldExpose) {
      peer.exposeParam?.(param.name)
    } else {
      for (const c of [...editor.getConnections()]) {
        if (c.target === peer.id && c.targetInput === key) {
          await editor.removeConnection(c.id)
        }
      }
      peer.hideParam?.(param.name)
    }
    await area.update('node', peer.id)
  }
}

function groupBoundarySignature(g) {
  const facade = g.facadeNodeId ? editor.getNode(g.facadeNodeId) : null
  if (facade?.entry) return boundarySignatureFromEntry(facade.entry)
  const pm = g.portMap
  if (pm?.inputs?.length || pm?.outputs?.length || pm?.params?.length) {
    return boundarySignatureFromBoundary({
      inputs: (pm.inputs ?? []).map((m) => ({ shape: m.shape })),
      outputs: (pm.outputs ?? []).map((m) => ({ shape: m.shape })),
      params: pm.params ?? [],
    })
  }
  return null
}

function findPeerGroupWithTag(excludeGid, tag) {
  const key = String(tag ?? '').trim().toLowerCase()
  if (!key) return null
  for (const [peerGid, peer] of state.groups) {
    if (peerGid === excludeGid) continue
    if (groupTag(peer).toLowerCase() !== key) continue
    return { gid: peerGid, g: peer }
  }
  return null
}

function getTagBoundaryTemplate(excludeGid, tag) {
  const peer = findPeerGroupWithTag(excludeGid, tag)
  return peer ? groupBoundarySignature(peer.g) : null
}

async function replaceCollapsedFacade(group, boundary) {
  const facade = group.facadeNodeId ? editor.getNode(group.facadeNodeId) : null
  const pos = (facade && nodePosition(facade)) ?? group.savedPosition ?? { x: 0, y: 0 }
  const tag = String(facade?.tag ?? group.facadeTag ?? '')

  if (facade) {
    await rerouteBoundaryEdges(group, 'to-children')
    for (const c of [...editor.getConnections()]) {
      if (c.source === facade.id || c.target === facade.id) {
        await editor.removeConnection(c.id)
      }
    }
    await editor.removeNode(facade.id)
  }

  const facadeEntry = makeGroupEntry(group.id, group.name, boundary)
  const newFacade = makeNode(facadeEntry)
  await editor.addNode(newFacade)
  await area.translate(newFacade.id, pos)
  markFacadeElement(newFacade.id)
  if (tag) restoreNodeTag(newFacade, tag)

  group.facadeNodeId = newFacade.id
  group.portMap = facadeEntry.portMap
  group.savedPosition = nodePosition(newFacade) ?? pos
  await rerouteBoundaryEdges(group, 'to-facade')
  applyTagStyle(newFacade)
  for (const child of getGroupChildren(group.id)) applyTagStyle(child)
}

async function alignGroupBoundary(groupId, signature) {
  const group = state.groups.get(groupId)
  if (!group || !signature) return

  if (!group.collapsed) {
    group.pendingBoundarySignature = signature
    return
  }

  const currentSig = groupBoundarySignature(group)
  if (currentSig && boundarySignaturesMatch(currentSig, signature)) return

  const childIds = new Set(getGroupChildren(groupId).map((n) => n.id))
  const merged = applySignatureToBoundary(computeBoundary(childIds), signature, group.portMap)
  await replaceCollapsedFacade(group, merged)
}

async function syncPeerGroupBoundaries(sourceGid, signature) {
  const tag = groupTag(state.groups.get(sourceGid))
  if (!tag || !signature) return
  for (const [peerGid, peer] of state.groups) {
    if (peerGid === sourceGid) continue
    if (groupTag(peer).toLowerCase() !== tag.toLowerCase()) continue
    await alignGroupBoundary(peerGid, signature)
  }
}

async function alignTaggedGroupToPeers(gid, tag) {
  await syncAllTaggedGroupInstances(tag)
  queueValidation()
  queueAutosave()
}

let syncingPeerGroups = false

function taggedGroupHasPeers(groupId) {
  const tag = groupTag(state.groups.get(groupId))
  if (!tag) return false
  const key = tag.toLowerCase()
  for (const [gid, g] of state.groups) {
    if (gid === groupId) continue
    if (groupTag(g).toLowerCase() === key) return true
  }
  return false
}

function groupChildrenByTag(groupId) {
  const map = new Map()
  for (const n of getGroupChildren(groupId)) {
    const key = nodeTagKey(n.tag)
    if (key) map.set(key, n)
  }
  return map
}

/** Topological order of children using only intra-group edges. */
function topoOrderedGroupChildren(groupId) {
  const children = getGroupChildren(groupId)
  if (children.length <= 1) return children
  const ids = new Set(children.map((n) => n.id))
  const conns = editor
    .getConnections()
    .filter((c) => ids.has(c.source) && ids.has(c.target))
  const indeg = new Map(children.map((n) => [n.id, 0]))
  const out = new Map(children.map((n) => [n.id, []]))
  for (const c of conns) {
    indeg.set(c.target, (indeg.get(c.target) ?? 0) + 1)
    out.get(c.source)?.push(c.target)
  }
  const queue = [...indeg.entries()].filter(([, d]) => d === 0).map(([id]) => id)
  const order = []
  while (queue.length) {
    const id = queue.shift()
    order.push(id)
    for (const tgt of out.get(id) ?? []) {
      indeg.set(tgt, indeg.get(tgt) - 1)
      if (indeg.get(tgt) === 0) queue.push(tgt)
    }
  }
  if (order.length !== children.length) return children
  const byId = new Map(children.map((n) => [n.id, n]))
  return order.map((id) => byId.get(id)).filter(Boolean)
}

/**
 * Map each source child to a peer child: first by matching tag, then by block
 * name in topo order (covers separately-created groups whose random child tags
 * differ but topology matches).
 */
function buildSourceToPeerChildMap(sourceGid, peerGid) {
  const map = new Map()
  const usedPeerIds = new Set()
  const peerByTag = groupChildrenByTag(peerGid)

  for (const src of getGroupChildren(sourceGid)) {
    const key = nodeTagKey(src.tag)
    if (!key) continue
    const peer = peerByTag.get(key)
    if (peer && !usedPeerIds.has(peer.id)) {
      map.set(src.id, peer)
      usedPeerIds.add(peer.id)
    }
  }

  const peerLeftByName = new Map()
  for (const peer of topoOrderedGroupChildren(peerGid)) {
    if (usedPeerIds.has(peer.id)) continue
    const name = peer.entry?.name ?? ''
    if (!peerLeftByName.has(name)) peerLeftByName.set(name, [])
    peerLeftByName.get(name).push(peer)
  }

  for (const src of topoOrderedGroupChildren(sourceGid)) {
    if (map.has(src.id)) continue
    const name = src.entry?.name ?? ''
    const bucket = peerLeftByName.get(name)
    const peer = bucket?.shift()
    if (peer) {
      map.set(src.id, peer)
      usedPeerIds.add(peer.id)
    }
  }

  const orphanPeerIds = getGroupChildren(peerGid)
    .filter((p) => !usedPeerIds.has(p.id))
    .map((p) => p.id)
  return { map, orphanPeerIds }
}

function canonicalTaggedGroupId(tag) {
  const key = String(tag ?? '').trim().toLowerCase()
  if (!key) return null
  let best = null
  let bestCount = -1
  for (const [gid, g] of state.groups) {
    if (groupTag(g).toLowerCase() !== key) continue
    const count = getGroupChildren(gid).length
    if (count > bestCount) {
      bestCount = count
      best = gid
    }
  }
  return best
}

function groupFacadeAnchor(group) {
  if (!group) return null
  if (group.collapsed && group.facadeNodeId) {
    const facade = editor.getNode(group.facadeNodeId)
    const p = facade && nodePosition(facade)
    if (p) return p
  }
  return group.savedPosition ?? null
}

function signatureForGroup(groupId) {
  const g = state.groups.get(groupId)
  const fromFacade = g && groupBoundarySignature(g)
  if (fromFacade) return fromFacade
  const childIds = new Set(getGroupChildren(groupId).map((n) => n.id))
  return boundarySignatureFromBoundary(computeBoundary(childIds))
}

function copyExposedParamsFromSource(source, target) {
  for (const portName of Object.keys(source.inputs ?? {})) {
    if (!portName.startsWith('__param__')) continue
    target.exposeParam?.(portName.slice('__param__'.length))
  }
}

async function replicateGroupChildToPeer(sourceChild, peerGid, peerGroup, sourceGroup) {
  const entry = state.byName.get(sourceChild.entry?.name)
  if (!entry || entry.kind === 'input' || entry.kind === 'output') return null

  const node = makeNode(entry)
  await editor.addNode(node)
  if (sourceChild.values) Object.assign(node.values, { ...sourceChild.values })
  copyNodeValues(sourceChild, node)

  const childTag = String(sourceChild.tag ?? '').trim()
  if (childTag) restoreNodeTag(node, childTag)
  else applyNodeTag(node, randomChildTag())

  copyExposedParamsFromSource(sourceChild, node)
  node.groupId = peerGid
  registerNodeMember(state.tagAtlas, node)

  const offset = sourceGroup.childOffsets?.[sourceChild.id]
  const anchor = groupFacadeAnchor(peerGroup) ?? groupFacadeAnchor(sourceGroup)
  if (offset && anchor) {
    await area.translate(node.id, { x: anchor.x + offset.dx, y: anchor.y + offset.dy })
    peerGroup.childOffsets = peerGroup.childOffsets ?? {}
    peerGroup.childOffsets[node.id] = { dx: offset.dx, dy: offset.dy }
  } else {
    const sp = nodePosition(sourceChild)
    if (sp) await area.translate(node.id, { x: sp.x, y: sp.y })
  }

  if (peerGroup.collapsed) setNodeHidden(node.id, true)
  applyTagStyle(node)
  area.update('node', node.id)
  return node
}

async function mirrorInternalEdgesToPeer(sourceGid, peerGid, sourceToPeer) {
  const peerGroup = state.groups.get(peerGid)
  for (const c of editor.getConnections()) {
    const src = editor.getNode(c.source)
    const tgt = editor.getNode(c.target)
    if (src?.groupId !== sourceGid || tgt?.groupId !== sourceGid) continue

    const peerSrc = sourceToPeer.get(src.id)
    const peerTgt = sourceToPeer.get(tgt.id)
    if (!peerSrc || !peerTgt) continue

    const exists = editor.getConnections().some(
      (ec) =>
        ec.source === peerSrc.id &&
        ec.sourceOutput === c.sourceOutput &&
        ec.target === peerTgt.id &&
        ec.targetInput === c.targetInput
    )
    if (exists) continue

    const before = new Set(editor.getConnections().map((ec) => ec.id))
    const ok = await safeAddConnection(peerSrc.id, c.sourceOutput, peerTgt.id, c.targetInput)
    if (!ok) continue
    if (!peerGroup?.collapsed) continue
    const added = editor.getConnections().find((ec) => !before.has(ec.id))
    if (added) setConnectionHidden(added.id, true)
  }
}

async function alignPeerChildToSource(sourceChild, peerChild) {
  const srcTag = String(sourceChild.tag ?? '').trim()
  if (srcTag && nodeTagKey(peerChild.tag) !== nodeTagKey(srcTag)) {
    applyNodeTagOnNode(peerChild, srcTag)
    registerNodeMember(state.tagAtlas, peerChild)
  }
  copyNodeValues(sourceChild, peerChild)
  copyExposedParamsFromSource(sourceChild, peerChild)
  area.update('node', peerChild.id)
  applyTagStyle(peerChild)
}

async function removePeerGroupChild(peerGid, childId) {
  const peerGroup = state.groups.get(peerGid)
  for (const c of [...editor.getConnections()]) {
    if (c.source === childId || c.target === childId) {
      await editor.removeConnection(c.id)
    }
  }
  const child = editor.getNode(childId)
  if (child) unregisterMember(state.tagAtlas, child.tag, childId)
  await editor.removeNode(childId)
  if (peerGroup?.childOffsets) delete peerGroup.childOffsets[childId]
}

/**
 * Mirror children + internal wiring from one tagged group instance onto every
 * other group that shares the facade tag (weight-shared subgraph copies).
 */
async function syncPeerGroupStructure(sourceGid) {
  if (syncingPeerGroups || state.restoring) return
  const sourceGroup = state.groups.get(sourceGid)
  if (!sourceGroup) return
  const tag = groupTag(sourceGroup)
  if (!tag || !taggedGroupHasPeers(sourceGid)) return

  syncingPeerGroups = true
  try {
    const sig = signatureForGroup(sourceGid)
    const sourceChildren = getGroupChildren(sourceGid)

    for (const [peerGid, peerGroup] of state.groups) {
      if (peerGid === sourceGid) continue
      if (groupTag(peerGroup).toLowerCase() !== tag.toLowerCase()) continue

      const { map: sourceToPeer, orphanPeerIds } = buildSourceToPeerChildMap(sourceGid, peerGid)

      for (const sourceChild of sourceChildren) {
        let peerChild = sourceToPeer.get(sourceChild.id)
        if (!peerChild) {
          peerChild = await replicateGroupChildToPeer(
            sourceChild,
            peerGid,
            peerGroup,
            sourceGroup
          )
          if (peerChild) sourceToPeer.set(sourceChild.id, peerChild)
        } else {
          await alignPeerChildToSource(sourceChild, peerChild)
        }
      }

      for (const orphanId of orphanPeerIds) {
        await removePeerGroupChild(peerGid, orphanId)
      }

      await mirrorInternalEdgesToPeer(sourceGid, peerGid, sourceToPeer)
    }

    if (sig) await syncPeerGroupBoundaries(sourceGid, sig)
  } finally {
    syncingPeerGroups = false
    applyAllGroupStyles()
    applyAllConnectionStyles()
  }
}

/** Push the fullest tagged group's structure to every instance with that tag. */
async function syncAllTaggedGroupInstances(tag) {
  const canonical = canonicalTaggedGroupId(tag)
  if (!canonical) return
  await syncPeerGroupStructure(canonical)
}

async function removePeerChildrenByTags(sourceGid, tagKeys) {
  if (syncingPeerGroups || state.restoring || tagKeys.size === 0) return
  const sourceGroup = state.groups.get(sourceGid)
  const tag = groupTag(sourceGroup)
  if (!tag) return

  syncingPeerGroups = true
  try {
    for (const [peerGid, peerGroup] of state.groups) {
      if (peerGid === sourceGid) continue
      if (groupTag(peerGroup).toLowerCase() !== tag.toLowerCase()) continue

      const peerByTag = groupChildrenByTag(peerGid)
      for (const key of tagKeys) {
        const child = peerByTag.get(key)
        if (!child) continue
        for (const c of [...editor.getConnections()]) {
          if (c.source === child.id || c.target === child.id) {
            await editor.removeConnection(c.id)
          }
        }
        unregisterMember(state.tagAtlas, child.tag, child.id)
        await editor.removeNode(child.id)
        if (peerGroup.childOffsets) delete peerGroup.childOffsets[child.id]
      }

      const sig = signatureForGroup(sourceGid)
      if (sig) await syncPeerGroupBoundaries(sourceGid, sig)
    }
  } finally {
    syncingPeerGroups = false
    applyAllGroupStyles()
    applyAllConnectionStyles()
  }
}

/** Inspect every connection and classify it w.r.t. the given child set. */
function classifyEdges(childIds) {
  const internal = []
  const inputBoundary = [] // external -> child
  const outputBoundary = [] // child -> external
  for (const c of editor.getConnections()) {
    const srcIn = childIds.has(c.source)
    const tgtIn = childIds.has(c.target)
    if (srcIn && tgtIn) internal.push(c)
    else if (!srcIn && tgtIn) inputBoundary.push(c)
    else if (srcIn && !tgtIn) outputBoundary.push(c)
  }
  return { internal, inputBoundary, outputBoundary }
}

/**
 * Given a set of child node ids, compute the facade port layout. Every child
 * port that is not already consumed by an internal edge becomes a facade port:
 *   - Inputs:  external -> child + dangling tensor inputs.
 *   - Outputs: child -> external + dangling outputs with no consumer.
 *   - Params:  external constant -> child.__param__ + dangling exposed params.
 *
 * Exposing dangling ports means the group's interface always covers every
 * receiver/outlet a child needs/produces, even before the user wires it.
 */
function computeBoundary(childIds, sub) {
  const { inputBoundary, outputBoundary, internal } = classifyEdges(childIds)
  const inputs = []
  const outputs = []
  const params = []
  const seenIn = new Map() // `${childId}/${childInput}` -> facade index
  const seenOut = new Map()
  const seenParam = new Map()

  const pushInputPort = (child, portName, portSpec) => {
    const key = `${child.id}/${portName}`
    if (seenIn.has(key) || seenParam.has(key)) return
    seenIn.set(key, inputs.length)
    const shape = child.freshenedShape?.(portName, 'in') ?? ['...']
    inputs.push({
      childNodeId: child.id,
      childPort: portName,
      shape,
      dtype: portSpec?.dtype ?? 'any',
      optional: Boolean(portSpec?.optional),
    })
  }

  const pushParamPort = (child, portName, portSpec) => {
    const key = `${child.id}/${portName}`
    if (seenParam.has(key) || seenIn.has(key)) return
    seenParam.set(key, params.length)
    params.push({
      childNodeId: child.id,
      childPort: portName,
      paramName: portSpec.paramName,
      paramType: portSpec.paramType ?? 'int',
    })
  }

  const pushOutputPort = (child, portName, portSpec) => {
    const key = `${child.id}/${portName}`
    if (seenOut.has(key)) return
    seenOut.set(key, outputs.length)
    const shape = child.freshenedShape?.(portName, 'out') ?? ['...']
    outputs.push({
      childNodeId: child.id,
      childPort: portName,
      shape,
      dtype: portSpec?.dtype ?? 'any',
    })
  }

  // 1. External-edge boundary ports (always come first so port order is
  //    stable for users that wired things up before adding new members).
  for (const c of inputBoundary) {
    const child = editor.getNode(c.target)
    if (!child) continue
    const portSpec = child.inputs?.[c.targetInput]?.portSpec
    if (portSpec?.kind === 'param') pushParamPort(child, c.targetInput, portSpec)
    else pushInputPort(child, c.targetInput, portSpec)
  }
  for (const c of outputBoundary) {
    const child = editor.getNode(c.source)
    if (!child) continue
    const portSpec = child.outputs?.[c.sourceOutput]?.portSpec
    pushOutputPort(child, c.sourceOutput, portSpec)
  }

  // 2. Track child ports already saturated by an internal edge so we don't
  //    surface them as dangling.
  const internalIn = new Set()
  const internalOut = new Set()
  for (const c of internal) {
    internalIn.add(`${c.target}/${c.targetInput}`)
    internalOut.add(`${c.source}/${c.sourceOutput}`)
  }

  // 3. Dangling ports - declared on each child but never wired anywhere.
  //    Tensor inputs/outputs + any exposed __param__ ports.
  for (const id of childIds) {
    const child = editor.getNode(id)
    if (!child) continue

    for (const port of child.entry?.inputs ?? []) {
      const key = `${id}/${port.name}`
      if (internalIn.has(key) || seenIn.has(key)) continue
      const portSpec = child.inputs?.[port.name]?.portSpec ?? port
      pushInputPort(child, port.name, portSpec)
    }

    // Exposed __param__ inputs are user-driven; honor them on the facade so
    // external constants / Input(...) values can still parameterize the group.
    for (const portName of Object.keys(child.inputs ?? {})) {
      if (!portName.startsWith('__param__')) continue
      const key = `${id}/${portName}`
      if (internalIn.has(key) || seenParam.has(key)) continue
      const portSpec = child.inputs[portName].portSpec
      if (portSpec?.kind !== 'param') continue
      pushParamPort(child, portName, portSpec)
    }

    for (const port of child.entry?.outputs ?? []) {
      const key = `${id}/${port.name}`
      if (internalOut.has(key) || seenOut.has(key)) continue
      const portSpec = child.outputs?.[port.name]?.portSpec ?? port
      pushOutputPort(child, port.name, portSpec)
    }
  }

  return { inputs, outputs, params, internal, inputBoundary, outputBoundary, seenIn, seenOut, seenParam }
}

function centroid(nodes) {
  if (nodes.length === 0) return { x: 0, y: 0 }
  let sx = 0
  let sy = 0
  let count = 0
  for (const n of nodes) {
    const p = nodePosition(n)
    if (!p) continue
    sx += p.x
    sy += p.y
    count++
  }
  if (count === 0) return { x: 0, y: 0 }
  return { x: sx / count, y: sy / count }
}

function setNodeHidden(nodeId, hidden) {
  const el = area?.nodeViews?.get(nodeId)?.element
  if (!el) return
  el.classList.toggle('group-hidden', hidden)
}

function markFacadeElement(nodeId) {
  const el = area?.nodeViews?.get(nodeId)?.element
  if (!el) return
  el.classList.add('group-facade')
}

function applyAllGroupStyles() {
  for (const n of editor.getNodes()) {
    const el = area?.nodeViews?.get(n.id)?.element
    if (!el) continue
    if (isGroupFacade(n)) el.classList.add('group-facade')
    if (n.groupId) {
      const g = state.groups.get(n.groupId)
      if (g?.collapsed) el.classList.add('group-hidden')
      else el.classList.remove('group-hidden')
    }
  }
  for (const c of editor.getConnections()) {
    const s = editor.getNode(c.source)
    const t = editor.getNode(c.target)
    if (s?.groupId && s.groupId === t?.groupId) {
      const g = state.groups.get(s.groupId)
      setConnectionHidden(c.id, Boolean(g?.collapsed))
    }
  }
}

function setConnectionHidden(connectionId, hidden) {
  const view = area?.connectionViews?.get(connectionId)
  const el = view?.element
  if (!el) return
  el.classList.toggle('group-hidden', hidden)
}

/**
 * Reroute every boundary edge of the given group from its child endpoint to
 * the corresponding facade endpoint (or the reverse). Returns silently on
 * any partial failure since the validator picks up the resulting state.
 */
async function rerouteBoundaryEdges(group, direction /* 'to-facade' | 'to-children' */) {
  const { facadeNodeId, portMap } = group
  const facade = facadeNodeId ? editor.getNode(facadeNodeId) : null
  if (!facade && direction === 'to-facade') return
  const inputByChild = new Map()
  for (const m of portMap.inputs) {
    inputByChild.set(`${m.childNodeId}/${m.childPort}`, m.facadePort)
  }
  for (const m of portMap.params ?? []) {
    inputByChild.set(`${m.childNodeId}/${m.childPort}`, m.facadePort)
  }
  const outputByChild = new Map()
  for (const m of portMap.outputs) {
    outputByChild.set(`${m.childNodeId}/${m.childPort}`, m.facadePort)
  }
  const inputByFacade = new Map([
    ...portMap.inputs.map((m) => [m.facadePort, m]),
    ...(portMap.params ?? []).map((m) => [m.facadePort, m]),
  ])
  const outputByFacade = new Map(portMap.outputs.map((m) => [m.facadePort, m]))

  for (const c of [...editor.getConnections()]) {
    if (direction === 'to-facade') {
      // child boundary -> facade
      const inFacadePort = inputByChild.get(`${c.target}/${c.targetInput}`)
      const outFacadePort = outputByChild.get(`${c.source}/${c.sourceOutput}`)
      if (inFacadePort && c.source !== facadeNodeId) {
        await editor.removeConnection(c.id)
        await safeAddConnection(c.source, c.sourceOutput, facadeNodeId, inFacadePort)
      } else if (outFacadePort && c.target !== facadeNodeId) {
        await editor.removeConnection(c.id)
        await safeAddConnection(facadeNodeId, outFacadePort, c.target, c.targetInput)
      }
    } else {
      // facade -> child
      if (c.target === facadeNodeId) {
        const m = inputByFacade.get(c.targetInput)
        if (!m) continue
        await editor.removeConnection(c.id)
        await safeAddConnection(c.source, c.sourceOutput, m.childNodeId, m.childPort)
      } else if (c.source === facadeNodeId) {
        const m = outputByFacade.get(c.sourceOutput)
        if (!m) continue
        await editor.removeConnection(c.id)
        await safeAddConnection(m.childNodeId, m.childPort, c.target, c.targetInput)
      }
    }
  }
}

async function safeAddConnection(source, sourceOutput, target, targetInput) {
  const srcNode = editor.getNode(source)
  const tgtNode = editor.getNode(target)
  if (!srcNode || !tgtNode) return false
  try {
    const conn = new ClassicPreset.Connection(srcNode, sourceOutput, tgtNode, targetInput)
    await editor.addConnection(conn)
    return true
  } catch {
    return false
  }
}

/**
 * Group the current selection into a new collapsed subgraph node. Resolves
 * the selection from the rete selector by default; tests can pass an
 * explicit id set to bypass the selector.
 */
async function groupSelected(explicitIds) {
  const ids = explicitIds ?? selectedNodeIds()
  if (ids.size < 1) {
    flashDiagnostic('Select at least one node to group')
    return
  }
  for (const id of ids) {
    const n = editor.getNode(id)
    if (isGroupFacade(n)) {
      flashDiagnostic('Cannot nest a group inside another group (yet)')
      return
    }
    if (n?.groupId) {
      flashDiagnostic('Some selected nodes are already in a group')
      return
    }
    if (n?.entry?.kind === 'input' || n?.entry?.kind === 'output') {
      flashDiagnostic('Input/Output nodes cannot be inside a group')
      return
    }
  }

  const groupId = freshGroupId()
  const childNodes = [...ids].map((id) => editor.getNode(id)).filter(Boolean)
  for (const n of childNodes) {
    n.groupId = groupId
    if (!String(n.tag ?? '').trim()) {
      applyNodeTag(n, randomChildTag())
      area.update('node', n.id)
    }
    registerNodeMember(state.tagAtlas, n)
  }

  const boundary = computeBoundary(ids)
  const center = centroid(childNodes)
  const name = `Group${state.groups.size + 1}`
  const facadeEntry = makeGroupEntry(groupId, name, boundary)
  const facade = makeNode(facadeEntry)
  await editor.addNode(facade)
  await area.translate(facade.id, center)
  markFacadeElement(facade.id)

  const group = {
    id: groupId,
    name,
    description: '',
    facadeTag: '',
    collapsed: true,
    facadeNodeId: facade.id,
    portMap: facadeEntry.portMap,
    savedPosition: center,
    childOffsets: {},
  }
  captureChildOffsets(group, childNodes)
  state.groups.set(groupId, group)

  await rerouteBoundaryEdges(group, 'to-facade')

  // Hide children & internal edges visually.
  for (const n of childNodes) setNodeHidden(n.id, true)
  for (const c of editor.getConnections()) {
    if (
      childNodes.some((n) => n.id === c.source) &&
      childNodes.some((n) => n.id === c.target)
    ) {
      setConnectionHidden(c.id, true)
    }
  }

  // Paint the group color immediately (children + facade) instead of waiting
  // for the next debounced validation pass.
  applyTagStyle(facade)
  for (const child of childNodes) applyTagStyle(child)

  // Select the facade so the user immediately sees it as a unit.
  if (selector) {
    for (const n of editor.getNodes()) selector.remove({ label: 'node', id: n.id })
    selector.add({
      label: 'node',
      id: facade.id,
      translate: () => {},
      unselect: () => {},
    })
  }
  state.selectedNodeId = facade.id
  refreshInspector()
  queueValidation()
  queueAutosave()
}

/**
 * Resolve which group should receive new members from the current selection.
 * Uses any grouped node or facade in the selection; falls back to focusedGroupId().
 */
function resolveTargetGroupIdForAdd() {
  const ids = selectedNodeIds()
  const groupIds = new Set()
  for (const id of ids) {
    const n = editor.getNode(id)
    if (!n) continue
    if (isGroupFacade(n)) groupIds.add(n.entry.groupId)
    else if (n.groupId) groupIds.add(n.groupId)
  }
  if (groupIds.size > 1) return { error: 'Selection spans multiple groups' }
  if (groupIds.size === 1) return { gid: [...groupIds][0] }
  const gid = focusedGroupId()
  if (gid) return { gid }
  return { error: 'Select a group member (or the group) along with nodes to add' }
}

/** Assign groupId on nodes and reveal them while the group is expanded. */
async function addNodesToGroup(groupId, explicitIds) {
  const group = state.groups.get(groupId)
  if (!group) {
    flashDiagnostic('Group not found')
    return false
  }
  if (group.collapsed) await expandGroup(groupId)

  const ids = explicitIds ?? selectedNodeIds()
  const candidates = [...ids]
    .map((id) => editor.getNode(id))
    .filter((n) => n && !isGroupFacade(n) && n.groupId !== groupId)

  if (candidates.length === 0) {
    flashDiagnostic('Select ungrouped nodes to add to the group')
    return false
  }

  for (const n of candidates) {
    if (n.groupId && n.groupId !== groupId) {
      flashDiagnostic('Cannot add nodes that already belong to another group')
      return false
    }
    if (n.entry?.kind === 'input' || n.entry?.kind === 'output') {
      flashDiagnostic('Input/Output nodes cannot be inside a group')
      return false
    }
  }

  const anchor = groupFacadeAnchor(group) ?? centroid(getGroupChildren(groupId))
  for (const n of candidates) {
    n.groupId = groupId
    if (!String(n.tag ?? '').trim()) {
      applyNodeTag(n, randomChildTag())
      area.update('node', n.id)
    }
    registerNodeMember(state.tagAtlas, n)
    setNodeHidden(n.id, false)
    applyTagStyle(n)
    const p = nodePosition(n)
    if (p && anchor) {
      group.childOffsets = group.childOffsets ?? {}
      group.childOffsets[n.id] = { dx: p.x - anchor.x, dy: p.y - anchor.y }
    }
  }

  for (const c of editor.getConnections()) {
    const s = editor.getNode(c.source)
    const t = editor.getNode(c.target)
    if (s?.groupId === groupId && t?.groupId === groupId) setConnectionHidden(c.id, false)
  }

  await syncPeerGroupStructure(groupId)

  refreshInspector()
  queueValidation()
  queueAutosave()
  flashDiagnostic(`Added ${candidates.length} node(s) to ${group.name}`)
  return true
}

async function addToGroupFocused() {
  const resolved = resolveTargetGroupIdForAdd()
  if (resolved.error) {
    flashDiagnostic(resolved.error)
    return
  }
  await addNodesToGroup(resolved.gid)
}

/**
 * Re-show the children of a collapsed group and remove the facade. The group
 * association is preserved so the user can re-collapse later.
 */
async function expandGroup(groupId) {
  const group = state.groups.get(groupId)
  if (!group || !group.collapsed) return
  const facade = group.facadeNodeId ? editor.getNode(group.facadeNodeId) : null

  // Anchor the expansion to wherever the facade is RIGHT NOW (user may have
  // dragged it since collapse) so the children re-materialise around it
  // instead of jumping back to their pre-collapse absolute coords.
  const facadePos =
    (facade && nodePosition(facade)) ?? group.savedPosition ?? { x: 0, y: 0 }
  group.savedPosition = facadePos

  // Reroute boundary edges from facade back to children before deleting the
  // facade (so we don't lose the wiring).
  await rerouteBoundaryEdges(group, 'to-children')

  // Drop the facade node + its incident connections to it that didn't reroute.
  if (facade) {
    group.facadeTag = String(facade.tag ?? group.facadeTag ?? '')
    for (const c of [...editor.getConnections()]) {
      if (c.source === facade.id || c.target === facade.id) {
        await editor.removeConnection(c.id)
      }
    }
    await editor.removeNode(facade.id)
  }
  group.facadeNodeId = null
  group.collapsed = false

  // Translate children FIRST, then reveal them - this avoids a one-frame flash
  // at the stale coordinates.
  await applyChildOffsets(group, facadePos)
  for (const n of getGroupChildren(groupId)) setNodeHidden(n.id, false)
  for (const c of editor.getConnections()) {
    const s = editor.getNode(c.source)
    const t = editor.getNode(c.target)
    if (s?.groupId === groupId && t?.groupId === groupId) setConnectionHidden(c.id, false)
  }

  refreshInspector()
  queueValidation()
  queueAutosave()
}

/** Collapse an already-grouped (but currently expanded) set of children. */
async function collapseGroup(groupId) {
  const group = state.groups.get(groupId)
  if (!group || group.collapsed) return
  const children = getGroupChildren(groupId)
  if (children.length === 0) {
    // Group has no children left; just drop it.
    state.groups.delete(groupId)
    refreshInspector()
    return
  }
  const childIds = new Set(children.map((n) => n.id))
  let boundary = computeBoundary(childIds)
  const tag = groupTag(group)
  let template = group.pendingBoundarySignature ?? (tag ? getTagBoundaryTemplate(groupId, tag) : null)
  delete group.pendingBoundarySignature
  if (!template && tag) template = boundarySignatureFromBoundary(boundary)
  if (template) boundary = applySignatureToBoundary(boundary, template, group.portMap)
  const center = group.savedPosition ?? centroid(children)
  const facadeEntry = makeGroupEntry(groupId, group.name, boundary)
  const facade = makeNode(facadeEntry)
  await editor.addNode(facade)
  await area.translate(facade.id, center)
  markFacadeElement(facade.id)
  if (group.facadeTag) restoreNodeTag(facade, group.facadeTag)

  group.facadeNodeId = facade.id
  group.portMap = facadeEntry.portMap
  group.collapsed = true
  // Refresh offsets - children may have been dragged while the group was
  // expanded, so the previously-captured offsets are stale. captureChildOffsets
  // also writes back group.savedPosition using the facade's *actual* placed
  // coords (after any grid-snap applied during area.translate).
  captureChildOffsets(group, children)

  await rerouteBoundaryEdges(group, 'to-facade')

  for (const n of children) setNodeHidden(n.id, true)
  for (const c of editor.getConnections()) {
    const s = editor.getNode(c.source)
    const t = editor.getNode(c.target)
    if (s?.groupId === groupId && t?.groupId === groupId) setConnectionHidden(c.id, true)
  }
  applyTagStyle(facade)
  for (const child of children) applyTagStyle(child)
  state.selectedNodeId = facade.id
  if (tag) {
    const sig = boundarySignatureFromBoundary(boundary)
    await syncAllTaggedGroupInstances(tag)
    registerGroupMember(state.tagAtlas, group, facade)
    recordGroupMeta(state.tagAtlas, group, { boundarySignature: sig })
  }
  refreshInspector()
  queueValidation()
  queueAutosave()
}

/** Collapse every currently-expanded group. No-op for already-collapsed ones. */
async function collapseAllGroups() {
  const ids = [...state.groups.keys()].filter((gid) => !state.groups.get(gid)?.collapsed)
  for (const gid of ids) await collapseGroup(gid)
}

/** Expand every currently-collapsed group. No-op for already-expanded ones. */
async function expandAllGroups() {
  const ids = [...state.groups.keys()].filter((gid) => state.groups.get(gid)?.collapsed)
  for (const gid of ids) await expandGroup(gid)
}

/**
 * Dissolve a group permanently. If currently collapsed, expand first; then
 * clear groupId on every child and forget the group state.
 */
async function ungroup(groupId) {
  const group = state.groups.get(groupId)
  if (!group) return
  if (group.collapsed) await expandGroup(groupId)
  for (const n of getGroupChildren(groupId)) n.groupId = null
  unregisterMember(state.tagAtlas, group.facadeTag, groupId)
  state.groups.delete(groupId)
  refreshInspector()
  queueValidation()
  queueAutosave()
}

/** Resolve "the group currently being focused", whether via a facade or a child selection. */
function focusedGroupId() {
  const node = state.selectedNodeId ? editor.getNode(state.selectedNodeId) : null
  if (!node) return null
  if (isGroupFacade(node)) return node.entry.groupId
  if (node.groupId) return node.groupId
  return null
}

async function ungroupFocused() {
  const gid = focusedGroupId()
  if (!gid) {
    flashDiagnostic('Select a group facade or a grouped node first')
    return
  }
  await ungroup(gid)
}

async function toggleCollapseFocused() {
  const gid = focusedGroupId()
  if (!gid) return
  const g = state.groups.get(gid)
  if (!g) return
  if (g.collapsed) await expandGroup(gid)
  else await collapseGroup(gid)
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
  applyAllTagStyles()
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
    runtimeNumParams: state.runtimeNumParams,
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
    const { shapes, numParams } = await runShapeCheck(editor, state.framework, state.batchSize)
    state.runtimeShapes = shapes
    state.runtimeNumParams = numParams
    state.runtimeErrorNodeId = null
  } catch (e) {
    state.runtimeShapes = null
    state.runtimeNumParams = null
    state.runtimeError = e.message || String(e)
    state.runtimeErrorNodeId = e.nodeId || null
  } finally {
    state.runtimeRunning = false
    refreshRuntimePanel()
    applyRuntimeErrorHighlight()
    refreshInspector()
  }
}

function refreshInspector(options = {}) {
  const node = state.selectedNodeId ? editor.getNode(state.selectedNodeId) : null
  renderInspector(
    document.getElementById('inspector-body'),
    node,
    state.lastResult?.sub ?? new Map(),
    () => {
      const n = state.selectedNodeId ? editor.getNode(state.selectedNodeId) : null
      if (n) syncTaggedNodePeers(n)
      state.runtimeShapes = null
      state.runtimeNumParams = null
      state.runtimeError = null
      state.runtimeErrorNodeId = null
      queueValidation()
      queueAutosave()
    },
    state.runtimeShapes,
    state.blockInfo,
    async (n, newTag) => {
      const oldTag = String(n.tag ?? '').trim()
      applyNodeTagOnNode(n, newTag)
      if (n.entry?.kind === 'group') {
        const gid = n.entry.groupId
        const g = state.groups.get(gid)
        unregisterMember(state.tagAtlas, oldTag, gid)
        if (g) {
          persistGroupFacadeTag(g, newTag)
          if (nodeTagKey(newTag)) registerGroupMember(state.tagAtlas, g, n)
        }
        forEachPeerGroup(gid, oldTag, (_peerGid, peer) => applyGroupTag(peer, newTag))
        const newKey = String(newTag ?? '').trim()
        if (newKey) void alignTaggedGroupToPeers(gid, newKey)
        applyAllTagStyles()
      } else {
        unregisterMember(state.tagAtlas, oldTag, n.id)
        const newKey = String(newTag ?? '').trim()
        if (newKey) {
          // Register first so we can either seed the atlas (first member) or
          // adopt the canonical values from an existing peer.
          const entry = registerNodeMember(state.tagAtlas, n)
          const isFirstMember = entry?.members.size === 1
          if (isFirstMember) {
            // Push this node's values out as canonical to anything that
            // already shares the tag (paste / re-tag race).
            syncTaggedNodePeers(n)
          } else {
            await adoptTaggedPeerValues(n)
          }
        }
      }
      area.update('node', n.id)
      applyTagStyle(n)
      state.runtimeShapes = null
      state.runtimeNumParams = null
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
      await syncTaggedPeerParamPort(targetNode, param, shouldExpose)
      await area.update('node', targetNode.id)
      queueValidation()
      queueAutosave()
      refreshInspector({ forceRebuild: true })
    },
    {
      getName: (gid) => state.groups.get(gid)?.name ?? '',
      getDescription: (gid) => state.groups.get(gid)?.description ?? '',
      setDescription: (gid, value) => {
        const g = state.groups.get(gid)
        if (!g) return
        const text = String(value ?? '')
        g.description = text
        const peers = recordGroupMeta(state.tagAtlas, g, { description: text })
        for (const peerGid of peers) {
          const peer = state.groups.get(peerGid)
          if (peer) peer.description = text
        }
        queueAutosave()
      },
      isCollapsed: (gid) => Boolean(state.groups.get(gid)?.collapsed),
      rename: (gid, value) => {
        const g = state.groups.get(gid)
        if (!g) return
        const newName = String(value ?? '').trim() || g.name
        applyGroupName(g, newName)
        const peers = recordGroupMeta(state.tagAtlas, g, { name: newName })
        for (const peerGid of peers) {
          const peer = state.groups.get(peerGid)
          if (peer) applyGroupName(peer, newName)
        }
        queueAutosave()
      },
      toggle: (gid) => expandGroupOrCollapse(gid),
      ungroup: (gid) => ungroup(gid),
      addSelection: (gid) => void addNodesToGroup(gid),
    },
    options
  )
}

// The inspector's toggle action needs to act on a specific gid (not the
// currently-focused selection), so we route through a small helper.
async function expandGroupOrCollapse(groupId) {
  const g = state.groups.get(groupId)
  if (!g) return
  if (g.collapsed) await expandGroup(groupId)
  else await collapseGroup(groupId)
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
    nodes: editor.getNodes().map((n) => {
      const view = area?.nodeViews?.get(n.id)
      const p = view?.position
      const position =
        p && Number.isFinite(p.x) && Number.isFinite(p.y) ? { x: p.x, y: p.y } : undefined
      // Two flavours of "groupId" in the spec, distinguished by `kind`:
      //   - kind === 'group'  -> this is a facade; groupId is its identity
      //   - everything else   -> groupId is "I am a member of group X"
      const base = {
        id: n.id,
        name: n.entry.name,
        kind: n.entry.kind,
        tag: n.tag ?? '',
        values: n.values,
        exposedParams: Object.keys(n.inputs || {})
          .filter((k) => k.startsWith('__param__'))
          .map((k) => k.replace(/^__param__/, '')),
        position,
      }
      if (n.entry.kind === 'group') {
        base.groupId = n.entry.groupId
        base.portMap = n.entry.portMap
      } else if (n.groupId) {
        base.groupId = n.groupId
      }
      return base
    }),
    connections: editor.getConnections().map((c) => ({
      source: c.source,
      sourceOutput: c.sourceOutput,
      target: c.target,
      targetInput: c.targetInput,
    })),
    groups: [...state.groups.values()].map((g) => {
      const facade = g.facadeNodeId ? editor.getNode(g.facadeNodeId) : null
      return {
        id: g.id,
        name: g.name,
        description: g.description ?? '',
        tag: facade?.tag ?? g.facadeTag ?? '',
        collapsed: g.collapsed,
        facadeNodeId: g.facadeNodeId,
        savedPosition: g.savedPosition,
        childOffsets: g.childOffsets ?? {},
      }
    }),
  }
}

function runCodegen(options = {}) {
  let testCase
  if (options.withTest) {
    try {
      testCase = resolveInputSpecs(editor, state.framework, state.batchSize)
    } catch (err) {
      alert(`Cannot build test case: ${err.message}`)
      return
    }
  }
  const code = generateCode(editor.getNodes(), editor.getConnections(), state.framework, {
    trace: !!options.trace,
    testCase,
  })
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
    get selector() {
      return selector
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
    applyNodeTag: applyNodeTagOnNode,
    syncPeerGroupStructure,
    syncAllTaggedGroupInstances,
    refreshTagAtlas,
    get tagAtlasSummary() {
      return atlasSummary(state.tagAtlas)
    },
    AUTOSAVE_KEY,
    CLIPBOARD_KEY,
    groupNodes: (ids) => groupSelected(new Set(ids)),
    expandGroup,
    collapseGroup,
    collapseAllGroups,
    expandAllGroups,
    ungroup,
    addNodesToGroup,
    addToGroupFocused,
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
