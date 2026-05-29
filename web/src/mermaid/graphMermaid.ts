/**
 * Export the current editor graph as a Mermaid `flowchart` definition, colored
 * to match the canvas. Kept in its own folder so it never touches the core
 * codegen path. Pure + dependency-light: the caller supplies a `colorOf`
 * resolver (so we can read the live `--tag-color` the canvas is showing) and a
 * `labelOf` resolver for display names.
 */

import type { Connection, NodeLike } from '../types.ts'

export interface GraphMermaidOptions {
  /** Resolve a node's current display color (hex/hsl) or null for default. */
  colorOf?: (node: NodeLike) => string | null
  /** Resolve a node's display label (defaults to instance name or block type). */
  labelOf?: (node: NodeLike) => string
  /** Diagram direction. Default 'TD' (top-down). */
  direction?: 'TD' | 'LR'
  /** Wrap each group's children in a Mermaid subgraph (expanded groups only). */
  groupSubgraphs?: boolean
}

/** A Mermaid-safe node id: alphanumerics + underscore, prefixed so it's valid. */
function safeId(rawId: string): string {
  return 'n_' + String(rawId).replace(/[^A-Za-z0-9_]/g, '_')
}

/** Escape a label for use inside a Mermaid `["..."]` node. */
function escapeLabel(text: string): string {
  return String(text)
    .replace(/"/g, "'")
    .replace(/\n/g, ' ')
    .trim()
}

function defaultLabel(n: NodeLike): string {
  const kind = n.entry?.kind
  // Input/Output/Constant carry their user-facing name in values, not `name`.
  if (kind === 'input' || kind === 'output') {
    const vn = String((n as any).values?.name ?? '').trim()
    if (vn) return vn
  }
  if (kind === 'const') {
    const v = (n as any).values?.value
    if (v !== undefined && v !== '') return String(v)
  }
  const inst = String((n as any).name ?? '').trim()
  const type = n.entry?.name ?? 'node'
  const tag = String((n as any).tag ?? '').trim()
  const base = inst || type
  return tag && tag !== base ? `${base} · ${tag}` : base
}

/** Convert `hsl(h, s%, l%)` to `#rrggbb`. Mermaid's `style` grammar rejects the
 *  commas/parens inside an `hsl(...)` literal, so colors must be hex. */
function hslToHex(h: number, s: number, l: number): string {
  s /= 100
  l /= 100
  const k = (n: number) => (n + h / 30) % 12
  const a = s * Math.min(l, 1 - l)
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))
  const to255 = (x: number) => Math.round(255 * x)
  const hex = (x: number) => to255(x).toString(16).padStart(2, '0')
  return `#${hex(f(0))}${hex(f(8))}${hex(f(4))}`
}

/** Normalize any supported color (hsl / hex / #rgb) to a `#rrggbb` literal that
 *  is safe to drop into a Mermaid `style` directive. Returns null if unparseable. */
function normalizeColor(color: string | null): string | null {
  if (!color) return null
  const c = color.trim()
  const hsl = /hsl\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\)/i.exec(c)
  if (hsl) return hslToHex(Number(hsl[1]), Number(hsl[2]), Number(hsl[3]))
  if (/^#[0-9a-f]{6}$/i.test(c)) return c.toLowerCase()
  if (/^#[0-9a-f]{3}$/i.test(c)) {
    const r = c[1], g = c[2], b = c[3]
    return `#${r}${r}${g}${g}${b}${b}`.toLowerCase()
  }
  return null
}

/** Readable text color (hex) for a given hex background so labels stay legible. */
function textColorForHex(hex: string): string {
  const n = parseInt(hex.slice(1), 16)
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
  return lum >= 0.6 ? '#10131f' : '#f5f6fa'
}

/**
 * Build the Mermaid flowchart text for the given nodes + connections.
 * Group facades render as a single styled node; their hidden children are not
 * emitted (they're represented by the facade). Constants/inputs/outputs render
 * with shape hints so the diagram reads like the canvas.
 */
export function graphToMermaid(
  nodes: NodeLike[],
  connections: Connection[],
  options: GraphMermaidOptions = {}
): string {
  const { colorOf, labelOf = defaultLabel, direction = 'TD', groupSubgraphs = false } = options
  const byId = new Map(nodes.map((n) => [n.id, n]))

  // A facade hides its members; drop grouped children that belong to a facade
  // present in this view so the diagram matches the collapsed canvas.
  const facadeGids = new Set(
    nodes.filter((n) => n.entry?.kind === 'group').map((n) => (n.entry as any).groupId)
  )
  const visible = nodes.filter((n) => !(n.groupId && facadeGids.has(n.groupId)))
  const visibleIds = new Set(visible.map((n) => n.id))

  const lines: string[] = [`flowchart ${direction}`]
  const styleLines: string[] = []

  // Node shape per kind: inputs/outputs as stadiums, consts as rounded, groups
  // as subroutine boxes, everything else as plain rectangles.
  const renderNode = (n: NodeLike): string => {
    const id = safeId(n.id)
    const label = escapeLabel(labelOf(n))
    const kind = n.entry?.kind
    if (kind === 'input' || kind === 'output') return `${id}(["${label}"])`
    if (kind === 'const') return `${id}("${label}")`
    if (kind === 'group') return `${id}[["${label}"]]`
    return `${id}["${label}"]`
  }

  const emitNode = (n: NodeLike) => {
    lines.push(`    ${renderNode(n)}`)
    // Mermaid's style grammar requires hex colors (it rejects hsl(...) commas),
    // so normalize whatever the canvas reports before emitting the directive.
    const fill = normalizeColor(colorOf?.(n) ?? null)
    if (fill) {
      const text = textColorForHex(fill)
      styleLines.push(
        `    style ${safeId(n.id)} fill:${fill},stroke:#0b0e16,stroke-width:1px,color:${text}`
      )
    }
  }

  if (groupSubgraphs) {
    // Group expanded children under a Mermaid subgraph by their groupId.
    const grouped = new Map<string, NodeLike[]>()
    const loose: NodeLike[] = []
    for (const n of visible) {
      if (n.groupId) {
        if (!grouped.has(n.groupId)) grouped.set(n.groupId, [])
        grouped.get(n.groupId)!.push(n)
      } else loose.push(n)
    }
    for (const n of loose) emitNode(n)
    for (const [gid, members] of grouped) {
      lines.push(`    subgraph ${safeId(gid)}["${escapeLabel(gid)}"]`)
      for (const n of members) emitNode(n)
      lines.push('    end')
    }
  } else {
    for (const n of visible) emitNode(n)
  }

  // Edges (only between visible nodes). Param wires (__param__*) render dashed.
  for (const c of connections) {
    if (!visibleIds.has(c.source) || !visibleIds.has(c.target)) continue
    const s = safeId(c.source)
    const t = safeId(c.target)
    const isParam = String(c.targetInput ?? '').startsWith('__param__')
    const port = String(c.targetInput ?? '').replace(/^__param__/, '')
    if (isParam) lines.push(`    ${s} -. ${escapeLabel(port)} .-> ${t}`)
    else lines.push(`    ${s} --> ${t}`)
  }

  return [...lines, ...styleLines].join('\n')
}
