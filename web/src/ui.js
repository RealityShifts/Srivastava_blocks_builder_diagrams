/**
 * DOM-facing UI bits: palette, inspector, diagnostics panel, code dialog.
 *
 * Keeps the Rete editor decoupled from raw DOM. main.js wires the events.
 */

import { groupByModule } from './nodes.js'
import { prettyShape } from './shape.js'

// ---------- palette ----------

export function renderPalette(rootEl, entries, onCreate) {
  rootEl.replaceChildren()
  const groups = groupByModule(entries)
  for (const [moduleName, list] of groups) {
    const title = document.createElement('div')
    title.className = 'group-title'
    title.textContent = moduleName
    rootEl.appendChild(title)
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
      rootEl.appendChild(item)
    }
  }
}

function describePorts(entry) {
  const i = entry.inputs.length
  const o = entry.outputs.length
  const k = entry.kind === 'function' ? 'fn' : 'mod'
  return `${k} · ${i} in / ${o} out`
}

export function filterPalette(rootEl, query) {
  const q = query.trim().toLowerCase()
  rootEl.querySelectorAll('.block-item').forEach((el) => {
    el.style.display = !q || el.dataset.name.toLowerCase().includes(q) ? '' : 'none'
  })
  rootEl.querySelectorAll('.group-title').forEach((tl) => {
    let next = tl.nextElementSibling
    let any = false
    while (next && !next.classList.contains('group-title')) {
      if (next.style.display !== 'none') {
        any = true
        break
      }
      next = next.nextElementSibling
    }
    tl.style.display = any ? '' : 'none'
  })
}

// ---------- inspector ----------

export function renderInspector(rootEl, node, sub, onChange) {
  rootEl.replaceChildren()
  if (!node) {
    rootEl.innerHTML = '<p class="muted">Select a node to edit its parameters.</p>'
    return
  }

  const header = document.createElement('div')
  header.className = 'header'
  header.innerHTML = `<strong>${node.entry.name}</strong><span class="module">${node.entry.module} · ${node.entry.kind}</span>`
  rootEl.appendChild(header)

  if (node.entry.ctor.length === 0) {
    const none = document.createElement('p')
    none.className = 'muted'
    none.textContent = 'No constructor parameters.'
    rootEl.appendChild(none)
  } else {
    for (const param of node.entry.ctor) {
      const row = document.createElement('div')
      row.className = 'row'
      const id = `ctrl-${node.id}-${param.name}`
      const label = document.createElement('label')
      label.htmlFor = id
      label.textContent = `${param.name}${param.required ? ' *' : ''}`
      row.appendChild(label)
      const ctrl = makeControl(param, node.values[param.name], (v) => {
        node.values[param.name] = v
        onChange()
      })
      ctrl.id = id
      row.appendChild(ctrl)
      rootEl.appendChild(row)
    }
  }

  // Ports + resolved shapes
  const ports = document.createElement('div')
  ports.className = 'ports'
  ports.appendChild(portList('Inputs', node.entry.inputs, node, sub, 'in'))
  ports.appendChild(portList('Outputs', node.entry.outputs, node, sub, 'out'))
  rootEl.appendChild(ports)
}

function portList(title, list, node, sub, side) {
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
    row.innerHTML = `<code>${p.name}</code> : ${shape} <span class="muted">[${dtype}${p.optional ? ', opt' : ''}${p.variadic ? ', var' : ''}]</span>`
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
