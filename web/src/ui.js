/**
 * DOM-facing UI bits: palette, inspector, diagnostics panel, code dialog.
 *
 * Keeps the Rete editor decoupled from raw DOM. main.js wires the events.
 */

import mermaid from 'mermaid'

import { groupByModule } from './nodes.js'
import { prettyShape } from './shape.js'

// ---------- mermaid helpers (used by the inspector Info tab) ----------

let _mermaidReady = false
function ensureMermaid() {
  if (_mermaidReady) return
  mermaid.initialize({
    startOnLoad: false,
    theme: 'default',
    securityLevel: 'loose',
    flowchart: { useMaxWidth: true, htmlLabels: false },
  })
  _mermaidReady = true
}

let _mermaidCounter = 0
const _mermaidCache = new Map() // block name -> rendered svg string

async function renderMermaidInto(container, blockName, definition) {
  if (!definition) {
    container.innerHTML = '<p class="muted">No diagram.</p>'
    return
  }
  const cached = _mermaidCache.get(blockName)
  if (cached) {
    container.innerHTML = cached
    return
  }
  ensureMermaid()
  container.innerHTML = '<p class="muted">Rendering diagram…</p>'
  try {
    const id = `mmd-${++_mermaidCounter}`
    const { svg } = await mermaid.render(id, definition)
    _mermaidCache.set(blockName, svg)
    container.innerHTML = svg
  } catch (e) {
    container.innerHTML = `<pre class="err">Mermaid render failed: ${escapeHtml(
      e?.message ?? String(e)
    )}</pre>`
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

/**
 * Minimal inline Markdown for section items: convert ``[label](url)`` to a
 * safe anchor and escape everything else. Only http(s) URLs are linkified so
 * a malicious feed can't slip in javascript: URLs.
 */
function renderInlineMd(text) {
  const re = /\[([^\]]+?)\]\(([^)]+?)\)/g
  const out = []
  let last = 0
  let m
  while ((m = re.exec(text)) !== null) {
    out.push(escapeHtml(text.slice(last, m.index)))
    const safe = /^https?:\/\//i.test(m[2].trim()) ? m[2].trim() : ''
    if (safe) {
      out.push(
        `<a href="${escapeHtml(safe)}" target="_blank" rel="noopener">${escapeHtml(
          m[1]
        )}</a>`
      )
    } else {
      out.push(escapeHtml(m[0]))
    }
    last = re.lastIndex
  }
  out.push(escapeHtml(text.slice(last)))
  return out.join('')
}

// ---------- palette ----------

const PALETTE_EXPANDED_BY_DEFAULT = new Set(['built-in', 'utility'])
const paletteCollapsed = new Map()

function paletteSectionCollapsed(name) {
  if (paletteCollapsed.has(name)) return paletteCollapsed.get(name)
  return !PALETTE_EXPANDED_BY_DEFAULT.has(name)
}

function setPaletteSectionCollapsed(name, collapsed) {
  paletteCollapsed.set(name, collapsed)
}

function syncSectionCollapsed(section) {
  const name = section.dataset.group
  const collapsed = paletteSectionCollapsed(name)
  section.classList.toggle('collapsed', collapsed)
  section.querySelector('.palette-section-header')?.setAttribute(
    'aria-expanded',
    String(!collapsed)
  )
}

export function renderPalette(rootEl, entries, onCreate) {
  rootEl.replaceChildren()
  const groups = groupByModule(entries)
  for (const [moduleName, list] of groups) {
    const section = document.createElement('div')
    section.className = 'palette-section'
    section.dataset.group = moduleName
    if (paletteSectionCollapsed(moduleName)) section.classList.add('collapsed')

    const header = document.createElement('button')
    header.type = 'button'
    header.className = 'palette-section-header'
    header.setAttribute('aria-expanded', String(!section.classList.contains('collapsed')))
    header.innerHTML =
      `<span class="palette-section-chevron" aria-hidden="true"></span>` +
      `<span class="palette-section-label">${moduleName}</span>` +
      `<span class="palette-section-count">${list.length}</span>`
    header.addEventListener('click', () => {
      const collapsed = !section.classList.contains('collapsed')
      section.classList.toggle('collapsed', collapsed)
      header.setAttribute('aria-expanded', String(!collapsed))
      setPaletteSectionCollapsed(moduleName, collapsed)
    })

    const body = document.createElement('div')
    body.className = 'palette-section-body'
    for (const entry of list) {
      const item = document.createElement('div')
      item.className = 'block-item'
      item.draggable = true
      item.dataset.name = entry.name
      const meta = describePorts(entry)
      item.innerHTML = `<div>${entry.name}</div><div class="meta">${meta}</div>`
      item.addEventListener('click', () => onCreate(entry))
      item.addEventListener('dragstart', (e) => {
        e.dataTransfer.effectAllowed = 'copy'
        e.dataTransfer.setData('application/x-block-name', entry.name)
      })
      body.appendChild(item)
    }

    section.appendChild(header)
    section.appendChild(body)
    rootEl.appendChild(section)
  }
}

function describePorts(entry) {
  const i = entry.inputs.length
  const o = entry.outputs.length
  const k =
    entry.kind === 'function'
      ? 'fn'
      : entry.kind === 'rearrange' ||
          entry.kind === 'reshape' ||
          entry.kind === 'concat' ||
          entry.kind === 'stack' ||
          entry.kind === 'pool' ||
          entry.kind === 'upsample' ||
          entry.kind === 'const'
        ? 'op'
        : entry.kind === 'learnable'
          ? 'mod'
        : entry.kind === 'input'
          ? 'in'
          : entry.kind === 'output'
            ? 'out'
          : 'mod'
  return `${k} · ${i} in / ${o} out`
}

export function filterPalette(rootEl, query) {
  const q = query.trim().toLowerCase()
  rootEl.querySelectorAll('.palette-section').forEach((section) => {
    let any = false
    section.querySelectorAll('.block-item').forEach((el) => {
      const show = !q || el.dataset.name.toLowerCase().includes(q)
      el.style.display = show ? '' : 'none'
      if (show) any = true
    })
    if (q) {
      section.style.display = any ? '' : 'none'
      if (any) {
        section.classList.remove('collapsed')
        section.querySelector('.palette-section-header')?.setAttribute('aria-expanded', 'true')
      }
    } else {
      section.style.display = ''
      syncSectionCollapsed(section)
    }
  })
}

// ---------- inspector ----------

// Track which node is currently displayed so we can do incremental refreshes
// after validation runs without blowing away focused <input> elements.
let _currentNodeId = null
let _activeTab = 'params' // 'params' | 'info' — persists across selections

function buildGroupPanel(node, actions) {
  const wrap = document.createElement('div')
  wrap.className = 'group-panel'

  const isFacade = node.entry.kind === 'group'
  const gid = isFacade ? node.entry.groupId : node.groupId

  const title = document.createElement('div')
  title.className = 'group-panel-title'
  title.textContent = isFacade ? 'Group' : 'Member of group'
  wrap.appendChild(title)

  const nameRow = document.createElement('div')
  nameRow.className = 'row'
  const label = document.createElement('label')
  label.textContent = 'Name'
  const input = document.createElement('input')
  input.type = 'text'
  input.spellcheck = false
  input.value = String(actions.getName?.(gid) ?? node.entry.name ?? '')
  input.placeholder = 'Encoder'
  input.addEventListener('input', () => actions.rename?.(gid, input.value))
  nameRow.appendChild(label)
  nameRow.appendChild(input)
  wrap.appendChild(nameRow)

  const btnRow = document.createElement('div')
  btnRow.className = 'group-panel-buttons'

  const toggleBtn = document.createElement('button')
  toggleBtn.type = 'button'
  toggleBtn.className = 'mini-btn'
  const collapsed = actions.isCollapsed?.(gid)
  toggleBtn.textContent = collapsed ? 'Expand' : 'Collapse'
  toggleBtn.title = collapsed
    ? 'Show the contained nodes for editing'
    : 'Collapse into a single facade node'
  toggleBtn.addEventListener('click', () => actions.toggle?.(gid))
  btnRow.appendChild(toggleBtn)

  const ungroupBtn = document.createElement('button')
  ungroupBtn.type = 'button'
  ungroupBtn.className = 'mini-btn danger'
  ungroupBtn.textContent = 'Ungroup'
  ungroupBtn.title = 'Dissolve this group (children stay)'
  ungroupBtn.addEventListener('click', () => actions.ungroup?.(gid))
  btnRow.appendChild(ungroupBtn)

  wrap.appendChild(btnRow)
  return wrap
}

function buildPortsSection(node, sub, runtimeShapes) {
  const ports = document.createElement('div')
  ports.className = 'ports'
  // Output nodes sink a tensor on input `x`; label that section "Output" so
  // runtime shapes (keyed as nodeId/x) read as the graph's output shape.
  if (node.entry.kind === 'output' && node.entry.inputs.length > 0) {
    ports.appendChild(
      portList('Output', node.entry.inputs, node, sub, 'in', runtimeShapes)
    )
    return ports
  }
  ports.appendChild(portList('Inputs', node.entry.inputs, node, sub, 'in', runtimeShapes))
  ports.appendChild(portList('Outputs', node.entry.outputs, node, sub, 'out', runtimeShapes))
  return ports
}

function buildParamsPanel(node, sub, runtimeShapes, onChange, onAddConst, onToggleParamPort) {
  const panel = document.createElement('div')
  panel.className = 'tab-panel'
  panel.dataset.tab = 'params'

  if (node.entry.ctor.length === 0) {
    const none = document.createElement('p')
    none.className = 'muted'
    none.textContent = 'No constructor parameters.'
    panel.appendChild(none)
  } else {
    for (const param of node.entry.ctor) {
      const row = document.createElement('div')
      row.className = 'row'
      const id = `ctrl-${node.id}-${param.name}`
      const label = document.createElement('label')
      label.htmlFor = id
      label.textContent = `${param.name}${param.required ? ' *' : ''}`
      row.appendChild(label)
      const controls = document.createElement('div')
      controls.className = 'param-controls'
      const ctrl = makeControl(param, node.values[param.name], (v) => {
        node.values[param.name] = v
        onChange()
      })
      ctrl.id = id
      controls.appendChild(ctrl)
      if (typeof onAddConst === 'function' && node.inputs?.[`__param__${param.name}`]) {
        const addBtn = document.createElement('button')
        addBtn.type = 'button'
        addBtn.className = 'mini-btn'
        addBtn.textContent = '+ const'
        addBtn.title = `Create Constant and connect to 🔴 ${param.name}`
        addBtn.addEventListener('click', () => onAddConst(node, param))
        controls.appendChild(addBtn)
      }
      if (typeof onToggleParamPort === 'function') {
        const key = `__param__${param.name}`
        const exposed = Boolean(node.inputs?.[key])
        const toggleBtn = document.createElement('button')
        toggleBtn.type = 'button'
        toggleBtn.className = 'mini-btn'
        toggleBtn.textContent = exposed ? 'Hide port' : 'Expose port'
        toggleBtn.title = `${exposed ? 'Hide' : 'Show'} 🔴 ${param.name}`
        toggleBtn.addEventListener('click', () => onToggleParamPort(node, param, !exposed))
        controls.appendChild(toggleBtn)
      }
      row.appendChild(controls)
      panel.appendChild(row)
    }
  }

  panel.appendChild(buildPortsSection(node, sub, runtimeShapes))
  return panel
}

function buildGroupInfoPanel(gid, actions) {
  const panel = document.createElement('div')
  panel.className = 'tab-panel info-panel group-info-panel'
  panel.dataset.tab = 'info'

  const name = String(actions.getName?.(gid) ?? 'Group').trim() || 'Group'
  const lead = document.createElement('p')
  lead.className = 'info-desc'
  lead.textContent =
    'Notes for this subgraph. Groups that share the same tag keep name, tag, and description in sync.'
  panel.appendChild(lead)

  const title = document.createElement('p')
  title.className = 'group-info-name'
  title.innerHTML = `<strong>${escapeHtml(name)}</strong>`
  panel.appendChild(title)

  const row = document.createElement('div')
  row.className = 'row group-desc-row'
  const label = document.createElement('label')
  label.textContent = 'Description'
  label.htmlFor = `group-desc-${gid}`
  row.appendChild(label)

  const ta = document.createElement('textarea')
  ta.id = `group-desc-${gid}`
  ta.className = 'group-desc-input'
  ta.rows = 8
  ta.spellcheck = true
  ta.placeholder =
    'e.g. Downsample path: two 3×3 convs with stride 2. Output channels must match the skip branch before Add.'
  ta.value = String(actions.getDescription?.(gid) ?? '')
  ta.addEventListener('input', () => actions.setDescription?.(gid, ta.value))
  row.appendChild(ta)
  panel.appendChild(row)

  return panel
}

function buildInfoPanel(node, blockInfo, groupActions) {
  const gid =
    node.entry.kind === 'group' ? node.entry.groupId : node.groupId ?? null
  if (gid && groupActions) {
    return buildGroupInfoPanel(gid, groupActions)
  }

  const panel = document.createElement('div')
  panel.className = 'tab-panel info-panel'
  panel.dataset.tab = 'info'

  const info = blockInfo?.get?.(node.entry.name) ?? null
  if (!info) {
    const p = document.createElement('p')
    p.className = 'muted'
    p.innerHTML = `No reference diagram for <code>${escapeHtml(
      node.entry.name
    )}</code>. Run <code>python tools/fetch_block_diagrams.py</code> to refresh.`
    panel.appendChild(p)
    return panel
  }

  if (info.description) {
    const d = document.createElement('p')
    d.className = 'info-desc'
    d.textContent = info.description
    panel.appendChild(d)
  }
  if (info.shapes) {
    const s = document.createElement('p')
    s.className = 'info-shapes'
    s.innerHTML = `<strong>Shapes:</strong> <code>${escapeHtml(info.shapes)}</code>`
    panel.appendChild(s)
  }
  if (info.source) {
    const a = document.createElement('a')
    a.href = info.source
    a.target = '_blank'
    a.rel = 'noopener'
    a.className = 'info-source'
    a.textContent = 'View source on GitHub →'
    panel.appendChild(a)
  }
  if (info.mermaid) {
    const m = document.createElement('div')
    m.className = 'info-mermaid'
    panel.appendChild(m)
    // Fire-and-forget; renderMermaidInto handles its own error states.
    renderMermaidInto(m, node.entry.name, info.mermaid)
  }
  if (Array.isArray(info.sections)) {
    for (const sec of info.sections) {
      panel.appendChild(buildCollapsibleSection(sec))
    }
  }
  return panel
}

function buildCollapsibleSection(sec) {
  const details = document.createElement('details')
  details.className = 'info-section'
  // "See also" is the most useful at a glance; open it by default.
  if (sec.heading?.toLowerCase() === 'see also') details.open = true

  const summary = document.createElement('summary')
  summary.textContent = `${sec.heading}  (${sec.items?.length ?? 0})`
  details.appendChild(summary)

  const ul = document.createElement('ul')
  for (const item of sec.items ?? []) {
    const li = document.createElement('li')
    li.innerHTML = renderInlineMd(String(item))
    ul.appendChild(li)
  }
  details.appendChild(ul)
  return details
}

export function renderInspector(
  rootEl,
  node,
  sub,
  onChange,
  runtimeShapes,
  blockInfo,
  onTagChange,
  onAddConst,
  onToggleParamPort,
  groupActions,
  options = {}
) {
  if (!node) {
    _currentNodeId = null
    rootEl.replaceChildren()
    const p = document.createElement('p')
    p.className = 'muted'
    p.textContent = 'Select a node to edit its parameters.'
    rootEl.appendChild(p)
    return
  }

  // Same node selected: just refresh the params panel's ports section so the
  // user's focus on a control stays put across validation runs. The Info tab
  // doesn't depend on validation state and is left untouched. Structural
  // changes (expose/hide param ports) pass forceRebuild to redo param rows too.
  if (node.id === _currentNodeId && !options.forceRebuild) {
    const paramsPanel = rootEl.querySelector('.tab-panel[data-tab="params"]')
    if (paramsPanel) {
      const oldPorts = paramsPanel.querySelector('.ports')
      const fresh = buildPortsSection(node, sub, runtimeShapes)
      if (oldPorts) oldPorts.replaceWith(fresh)
      else paramsPanel.appendChild(fresh)
    }
    return
  }

  // Different node (or first render): full rebuild.
  _currentNodeId = node.id
  rootEl.replaceChildren()

  const header = document.createElement('div')
  header.className = 'header'
  header.innerHTML = `<strong>${escapeHtml(
    node.entry.name
  )}</strong><span class="module">${escapeHtml(node.entry.module)} · ${escapeHtml(
    node.entry.kind
  )}</span>`
  rootEl.appendChild(header)

  // Group controls. Shown when the node is either the group facade itself or
  // a member of a group (groupId set). The action callbacks come from main.js.
  if (groupActions && (node.entry.kind === 'group' || node.groupId)) {
    rootEl.appendChild(buildGroupPanel(node, groupActions))
  }

  // Tag row. For module-kind nodes the tag also acts as a weight-sharing key:
  // two ConvBlocks tagged "down1" share one Python attribute.
  const tagRow = document.createElement('div')
  tagRow.className = 'row tag-row'
  const tagLabel = document.createElement('label')
  const tagId = `ctrl-${node.id}-__tag`
  tagLabel.htmlFor = tagId
  tagLabel.textContent = 'Tag'
  const tagHelp =
    node.entry.kind === 'module' || node.entry.kind === 'group'
      ? 'label · same tag = shared weights'
      : 'label only'
  tagLabel.title = tagHelp
  tagRow.appendChild(tagLabel)
  const tagInput = document.createElement('input')
  tagInput.id = tagId
  tagInput.type = 'text'
  tagInput.spellcheck = false
  tagInput.placeholder =
    node.entry.kind === 'module' || node.entry.kind === 'group' ? 'down1' : 'note'
  tagInput.value = String(node.tag ?? '')
  tagInput.addEventListener('input', () => {
    if (typeof onTagChange === 'function') onTagChange(node, tagInput.value)
  })
  tagRow.appendChild(tagInput)
  rootEl.appendChild(tagRow)

  const tabs = document.createElement('div')
  tabs.className = 'tabs'
  const paramsBtn = document.createElement('button')
  paramsBtn.type = 'button'
  paramsBtn.className = 'tab-btn'
  paramsBtn.dataset.tab = 'params'
  paramsBtn.textContent = 'Params'
  const infoBtn = document.createElement('button')
  infoBtn.type = 'button'
  infoBtn.className = 'tab-btn'
  infoBtn.dataset.tab = 'info'
  infoBtn.textContent = 'Info'
  tabs.appendChild(paramsBtn)
  tabs.appendChild(infoBtn)
  rootEl.appendChild(tabs)

  const paramsPanel = buildParamsPanel(
    node,
    sub,
    runtimeShapes,
    onChange,
    onAddConst,
    onToggleParamPort
  )
  const infoPanel = buildInfoPanel(node, blockInfo, groupActions)
  rootEl.appendChild(paramsPanel)
  rootEl.appendChild(infoPanel)

  const activate = (tab) => {
    _activeTab = tab
    paramsBtn.classList.toggle('active', tab === 'params')
    infoBtn.classList.toggle('active', tab === 'info')
    paramsPanel.hidden = tab !== 'params'
    infoPanel.hidden = tab !== 'info'
  }
  paramsBtn.addEventListener('click', () => activate('params'))
  infoBtn.addEventListener('click', () => activate('info'))
  activate(_activeTab)
}

function portList(title, list, node, sub, side, runtimeShapes) {
  const wrap = document.createElement('div')
  const h = document.createElement('h4')
  h.textContent = title
  wrap.appendChild(h)
  if (list.length === 0) {
    const e = document.createElement('div')
    e.className = 'muted'
    e.textContent = '(none)'
    wrap.appendChild(e)
    return wrap
  }
  for (const p of list) {
    const row = document.createElement('div')
    const resolved = node.freshenedShape(p.name, side)
    const shape = resolved ? prettyShape(resolved, sub) : '·'
    // Read the live portSpec.dtype off the rete port (InputNode rewrites it
    // when the user picks a different dtype in the inspector).
    const rete = side === 'in' ? node.inputs[p.name] : node.outputs[p.name]
    const dtype = rete?.portSpec?.dtype ?? p.dtype
    const key = `${node.id}/${p.name}`
    const runtime = runtimeShapes?.get(key)
    const runtimeTag =
      runtime
        ? ` <span class="runtime-shape">runtime: ${runtime.join(' ')}</span>`
        : ''
    row.innerHTML = `<code>${p.name}</code> : ${shape} <span class="muted">[${dtype}${p.optional ? ', opt' : ''}${p.variadic ? ', var' : ''}]</span>${runtimeTag}`
    wrap.appendChild(row)
  }
  return wrap
}

function makeControl(param, value, onChange) {
  const type = param.type
  if (Array.isArray(param.choices) && param.choices.length > 0) {
    const el = document.createElement('select')
    for (const c of param.choices) {
      const opt = document.createElement('option')
      opt.value = c
      opt.textContent = c
      el.appendChild(opt)
    }
    el.value = value ?? param.choices[0]
    el.addEventListener('change', () => onChange(el.value))
    return el
  }
  if (type === 'bool') {
    const el = document.createElement('select')
    for (const o of ['true', 'false']) {
      const opt = document.createElement('option')
      opt.value = o
      opt.textContent = o
      el.appendChild(opt)
    }
    el.value = String(Boolean(value))
    el.addEventListener('change', () => onChange(el.value === 'true'))
    return el
  }
  if (type === 'int' || type === 'float') {
    const el = document.createElement('input')
    el.type = 'number'
    if (type === 'int') el.step = '1'
    el.value = value ?? ''
    el.addEventListener('input', () => {
      const n = Number(el.value)
      if (el.value === '') onChange(null)
      else if (Number.isFinite(n)) onChange(type === 'int' ? Math.trunc(n) : n)
    })
    return el
  }
  if (type === 'list') {
    const el = document.createElement('input')
    el.type = 'text'
    el.value = Array.isArray(value) ? value.join(',') : (value ?? '')
    el.addEventListener('input', () => {
      const parts = el.value
        .split(',')
        .map((s) => s.trim())
        .filter((s) => s.length > 0)
        .map((s) => {
          const n = Number(s)
          return Number.isFinite(n) ? n : s
        })
      onChange(parts)
    })
    return el
  }
  // string / any
  const el = document.createElement('input')
  el.type = 'text'
  el.value = value ?? ''
  el.addEventListener('input', () => onChange(el.value === '' ? null : el.value))
  return el
}

// ---------- diagnostics ----------

export function renderDiagnostics(rootEl, result) {
  rootEl.replaceChildren()
  const { errors, warnings } = result
  if (errors.length === 0 && warnings.length === 0) {
    const li = document.createElement('li')
    li.className = 'ok'
    li.textContent = 'Graph is valid.'
    rootEl.appendChild(li)
    return
  }
  for (const e of errors) {
    const li = document.createElement('li')
    li.className = 'err'
    li.textContent = e.message
    rootEl.appendChild(li)
  }
  for (const w of warnings) {
    const li = document.createElement('li')
    li.className = 'warn'
    li.textContent = w.message
    rootEl.appendChild(li)
  }
}

// ---------- code dialog ----------

export function showCode(code) {
  const dlg = document.getElementById('codegen-dialog')
  document.getElementById('codegen-output').textContent = code
  if (typeof dlg.showModal === 'function') dlg.showModal()
  else dlg.setAttribute('open', '')
}

export function wireCodeDialog() {
  const dlg = document.getElementById('codegen-dialog')
  document.getElementById('close-code-btn').addEventListener('click', () => dlg.close())
  document.getElementById('copy-code-btn').addEventListener('click', async () => {
    const txt = document.getElementById('codegen-output').textContent
    try {
      await navigator.clipboard.writeText(txt)
    } catch {
      /* ignore */
    }
  })
}

export function updateRuntimePanel({ framework, lastResult, batchSize, runtimeShapes, running, lastError }) {
  const btn = document.getElementById('run-shapes-btn')
  const status = document.getElementById('runtime-status')
  const batchInput = document.getElementById('batch-size')
  if (!btn || !status) return

  if (batchInput && document.activeElement !== batchInput) {
    batchInput.value = String(batchSize ?? 2)
  }

  if (framework !== 'pytorch') {
    btn.disabled = true
    status.textContent = 'Switch to pytorch_blocks to run shape checks.'
    status.className = 'muted'
    return
  }

  btn.disabled =
    running ||
    !(lastResult?.ok ?? false) ||
    Boolean(lastResult && !isGraphRunnable(lastResult, batchSize))

  if (running) {
    status.textContent = 'Running forward pass…'
    status.className = 'muted'
  } else if (lastError) {
    status.textContent = lastError
    status.className = 'err'
  } else if (runtimeShapes?.size) {
    status.textContent = `Runtime shapes captured for ${runtimeShapes.size} port(s).`
    status.className = 'ok'
  } else if (!(lastResult?.ok ?? false)) {
    status.textContent = 'Fix graph errors before running.'
    status.className = 'muted'
  } else if (lastResult && !isGraphRunnable(lastResult, batchSize)) {
    status.textContent =
      lastResult.concreteReason ||
      'Set Input shape (literals or B) and wire all required inputs.'
    status.className = 'muted'
  } else {
    status.textContent = 'Ready — graph is valid (unresolved axes are back-solved at runtime).'
    status.className = 'ok'
  }
}

function isGraphRunnable(lastResult, batchSize) {
  // Button stays disabled until runtime.js isFullyConcrete passes; main.js sets
  // state.canRunShapes from that on each validation tick.
  return Boolean(lastResult?.canRunShapes)
}
