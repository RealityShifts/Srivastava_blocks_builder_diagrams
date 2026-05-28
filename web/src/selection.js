/**
 * Node selection + canvas navigation:
 * - preserve multi-select when dragging a selected node
 * - drag-to-marquee on empty canvas
 * - Space + drag to pan (grab mode)
 * - two-finger scroll / wheel to pan; pinch or Ctrl/Cmd+scroll to zoom
 * - middle mouse or Alt+drag to pan
 */

import { AreaExtensions, Drag, Zoom } from 'rete-area-plugin'

const MARQUEE_MIN_DRAG_PX = 5
/** Default Rete wheel/pinch step is 0.1; lower = slower zoom. */
const ZOOM_INTENSITY = 0.045
/** Pinch uses raw finger-distance ratio; dampen separately from wheel. */
const PINCH_ZOOM_SCALE = 0.4
/** Scroll / drag pan multiplier (1.2 = +20%). */
const PAN_SPEED = 1.2

function isTypingTarget(target) {
  const el = target instanceof Element ? target : null
  if (!el) return false
  return Boolean(el.closest('input, textarea, select, [contenteditable="true"]'))
}

function isGraphInteractiveTarget(target) {
  if (!target?.closest) return false
  return Boolean(
    target.closest('rete-node') ||
      target.closest('.node') ||
      target.closest('rete-connection') ||
      target.closest('button') ||
      target.closest('input') ||
      target.closest('select') ||
      target.closest('textarea') ||
      target.closest('label') ||
      target.closest('.socket')
  )
}

function rectsIntersect(a, b) {
  return a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top
}

function setupCanvasPan(area, nav) {
  const container = area.container
  const panDrag = new Drag({
    down(e) {
      if (nav.spacePressed) {
        if (e.button === 0) {
          container.classList.add('space-panning')
          return true
        }
        return false
      }
      if (e.pointerType === 'mouse' && e.button !== 0 && e.button !== 1) return false
      if (e.button === 1) return true
      if (e.button === 0 && e.altKey && !isGraphInteractiveTarget(e.target)) return true
      return false
    },
    move() {
      return true
    },
  })

  const onPointerUp = () => container.classList.remove('space-panning')
  container.addEventListener('pointerup', onPointerUp)
  container.addEventListener('pointercancel', onPointerUp)

  area.area.setDragHandler(panDrag)
  // Rete divides pointer delta by getZoom(); use 1/PAN_SPEED for +20% pan.
  area.area.dragHandler.destroy()
  panDrag.initialize(
    container,
    {
      getCurrentPosition: () => area.area.transform,
      getZoom: () => 1 / PAN_SPEED,
    },
    {
      start: () => null,
      translate: area.area.onTranslate.bind(area.area),
      drag: () => null,
    }
  )
  area.area.dragHandler = panDrag

  return {
    destroy() {
      container.removeEventListener('pointerup', onPointerUp)
      container.removeEventListener('pointercancel', onPointerUp)
      area.area.setDragHandler(null)
    },
  }
}

/** Pinch (2-touch) uses Rete Zoom; wheel without Ctrl pans, with Ctrl zooms. */
function setupCanvasZoom(area) {
  const container = area.container
  const origOnZoom = area.area.onZoom.bind(area.area)

  area.area.setZoomHandler(new Zoom(ZOOM_INTENSITY))
  area.area.onZoom = (delta, ox, oy, source) => {
    if (source === 'touch') {
      origOnZoom(delta * PINCH_ZOOM_SCALE, ox * PINCH_ZOOM_SCALE, oy * PINCH_ZOOM_SCALE, source)
      return
    }
    origOnZoom(delta, ox, oy, source)
  }

  const zoomHandler = area.area.zoomHandler
  if (!zoomHandler) return { destroy() {} }

  const element = area.area.content.holder
  const onzoom = area.area.onZoom.bind(area.area)

  const onWheel = (e) => {
    e.preventDefault()
    if (e.ctrlKey || e.metaKey) {
      const rect = element.getBoundingClientRect()
      const isNegative = e.deltaY < 0
      const delta = isNegative ? ZOOM_INTENSITY : -ZOOM_INTENSITY
      const ox = (rect.left - e.clientX) * delta
      const oy = (rect.top - e.clientY) * delta
      onzoom(delta, ox, oy, 'wheel')
      return
    }
    const t = area.area.transform
    void area.area.translate(
      t.x - e.deltaX * PAN_SPEED,
      t.y - e.deltaY * PAN_SPEED
    )
  }

  container.removeEventListener('wheel', zoomHandler.wheel)
  container.addEventListener('wheel', onWheel, { passive: false })

  return {
    destroy() {
      container.removeEventListener('wheel', onWheel)
      area.area.onZoom = origOnZoom
      area.area.setZoomHandler(new Zoom(0.1))
    },
  }
}

function setupKeyboardNav(area, nav) {
  const container = area.container

  const clearSpace = () => {
    nav.spacePressed = false
    container.classList.remove('space-pan', 'space-panning')
  }

  const onKeyDown = (e) => {
    if (e.key === 'Shift') nav.shiftPressed = true
    if (e.code !== 'Space' || nav.spacePressed) return
    if (isTypingTarget(e.target)) return
    nav.spacePressed = true
    container.classList.add('space-pan')
    e.preventDefault()
  }

  const onKeyUp = (e) => {
    if (e.key === 'Shift') nav.shiftPressed = false
    if (e.code === 'Space') clearSpace()
  }

  const onBlur = () => clearSpace()

  document.addEventListener('keydown', onKeyDown)
  document.addEventListener('keyup', onKeyUp)
  window.addEventListener('blur', onBlur)

  return {
    destroy() {
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('keyup', onKeyUp)
      window.removeEventListener('blur', onBlur)
      clearSpace()
    },
  }
}

function setupMarqueeSelection({ area, editor, selectableNodes, onSelectionChanged, nav }) {
  const container = area.container
  let overlay = null
  let pointerId = null
  let start = null
  let active = false

  function ensureOverlay() {
    if (!overlay) {
      overlay = document.createElement('div')
      overlay.className = 'selection-marquee'
      overlay.setAttribute('aria-hidden', 'true')
      container.appendChild(overlay)
    }
    return overlay
  }

  function hideOverlay() {
    if (overlay) overlay.style.display = 'none'
  }

  function updateOverlay(x1, y1, x2, y2) {
    const el = ensureOverlay()
    const rect = container.getBoundingClientRect()
    el.style.display = 'block'
    el.style.left = `${Math.min(x1, x2) - rect.left}px`
    el.style.top = `${Math.min(y1, y2) - rect.top}px`
    el.style.width = `${Math.abs(x2 - x1)}px`
    el.style.height = `${Math.abs(y2 - y1)}px`
  }

  function nodesInMarquee(x1, y1, x2, y2) {
    const box = {
      left: Math.min(x1, x2),
      top: Math.min(y1, y2),
      right: Math.max(x1, x2),
      bottom: Math.max(y1, y2),
    }
    const ids = []
    for (const node of editor.getNodes()) {
      const view = area.nodeViews.get(node.id)
      const el = view?.element
      if (!el || el.classList.contains('group-hidden')) continue
      const r = el.getBoundingClientRect()
      if (rectsIntersect(box, r)) ids.push(node.id)
    }
    return ids
  }

  async function applyMarqueeSelection(ids, additive) {
    if (!additive) {
      for (const n of editor.getNodes()) await selectableNodes.unselect(n.id)
    }
    for (const id of ids) await selectableNodes.select(id, true)
    if (ids.length > 0) onSelectionChanged?.(ids[ids.length - 1])
    else if (!additive) onSelectionChanged?.(null)
  }

  function resetPointer() {
    pointerId = null
    start = null
    active = false
    hideOverlay()
  }

  function onPointerDown(e) {
    if (nav.spacePressed) return
    if (e.button !== 0) return
    if (isGraphInteractiveTarget(e.target)) return
    pointerId = e.pointerId
    start = { x: e.clientX, y: e.clientY }
    active = false
  }

  function onPointerMove(e) {
    if (pointerId == null || e.pointerId !== pointerId || !start) return
    const dx = e.clientX - start.x
    const dy = e.clientY - start.y
    if (!active && Math.hypot(dx, dy) < MARQUEE_MIN_DRAG_PX) return
    active = true
    e.preventDefault()
    updateOverlay(start.x, start.y, e.clientX, e.clientY)
  }

  async function onPointerUp(e) {
    if (pointerId == null || e.pointerId !== pointerId) return
    const wasActive = active
    const origin = start
    const additive = e.shiftKey || e.ctrlKey || e.metaKey
    resetPointer()
    if (!wasActive || !origin) return
    await applyMarqueeSelection(
      nodesInMarquee(origin.x, origin.y, e.clientX, e.clientY),
      additive
    )
  }

  container.addEventListener('pointerdown', onPointerDown)
  container.addEventListener('pointermove', onPointerMove)
  container.addEventListener('pointerup', onPointerUp)
  container.addEventListener('pointercancel', onPointerUp)

  return {
    destroy() {
      container.removeEventListener('pointerdown', onPointerDown)
      container.removeEventListener('pointermove', onPointerMove)
      container.removeEventListener('pointerup', onPointerUp)
      container.removeEventListener('pointercancel', onPointerUp)
      overlay?.remove()
      overlay = null
    },
  }
}

/**
 * @param {import('rete-area-plugin').AreaPlugin} area
 * @param {import('rete').NodeEditor} editor
 * @param {{ onSelectionChanged?: (nodeId: string | null) => void }} options
 */
export function setupNodeSelection(area, editor, options = {}) {
  const { onSelectionChanged } = options
  const nav = { shiftPressed: false, spacePressed: false }
  const selector = AreaExtensions.selector()
  const ctrlAccum = AreaExtensions.accumulateOnCtrl()
  let preserveSelectionOnPick = false

  area.addPipe((context) => {
    if (context.type === 'nodepicked') {
      preserveSelectionOnPick = selector.isSelected({
        label: 'node',
        id: context.data.id,
      })
    }
    return context
  })

  const accumulating = {
    active() {
      return ctrlAccum.active() || nav.shiftPressed || preserveSelectionOnPick
    },
    destroy() {
      ctrlAccum.destroy()
    },
  }

  const selectableNodes = AreaExtensions.selectableNodes(area, selector, {
    accumulating,
  })

  const keyboard = setupKeyboardNav(area, nav)
  const pan = setupCanvasPan(area, nav)
  const zoom = setupCanvasZoom(area)
  const marquee = setupMarqueeSelection({
    area,
    editor,
    selectableNodes,
    onSelectionChanged,
    nav,
  })

  return {
    selector,
    selectableNodes,
    destroy() {
      accumulating.destroy()
      keyboard.destroy()
      pan.destroy()
      zoom.destroy()
      marquee.destroy()
    },
  }
}
