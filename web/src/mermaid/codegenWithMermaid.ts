/**
 * Python codegen that embeds each block's Mermaid diagram as a docstring, so an
 * exported module is self-documenting and usable as a real custom block.
 *
 * IMPORTANT: this does NOT modify the core codegen. It calls the existing
 * `generate()` and post-processes its string output - injecting a module-level
 * overview docstring plus a per-block ```mermaid fenced block into each emitted
 * class docstring. Kept in src/mermaid/ to avoid cluttering the codegen path.
 */

import { generate } from '../codegen.ts'
import { graphToMermaid, type GraphMermaidOptions } from './graphMermaid.ts'
import type { Connection, NodeLike } from '../types.ts'

export interface MermaidCodegenOptions {
  /** Resolve a block type name -> its mermaid definition (repo or docstring). */
  mermaidForBlock?: (blockName: string) => string | null | undefined
  /** Options forwarded to the whole-graph mermaid overview. */
  graphMermaid?: GraphMermaidOptions
  /** Forwarded to generate() (e.g. trace/testCase). */
  framework?: string
}

const INDENT = '    '

/** Wrap mermaid text as an indented Python docstring body. */
function mermaidDocstring(mermaid: string, indent: string): string {
  const fenced = ['```mermaid', mermaid.trim(), '```'].join('\n')
  const body = fenced
    .split('\n')
    .map((l) => (l ? `${indent}${l}` : indent.trimEnd()))
    .join('\n')
  return `${indent}"""\n${body}\n${indent}"""`
}

/**
 * Insert a mermaid docstring as the first line of each class body. We match
 * `class Name(...):` and, if `mermaidForBlock(Name)` returns text, splice a
 * docstring right after the class header (before the existing body), preserving
 * indentation. Classes without a mermaid are left untouched.
 */
function injectClassDocstrings(
  code: string,
  mermaidForBlock: (name: string) => string | null | undefined
): string {
  const lines = code.split('\n')
  const out: string[] = []
  const classRe = /^(\s*)class\s+([A-Za-z_]\w*)\s*\(/
  for (const line of lines) {
    out.push(line)
    const m = classRe.exec(line)
    if (!m) continue
    if (!line.trimEnd().endsWith(':')) continue // skip multi-line class headers
    const indent = m[1] + INDENT
    const name = m[2]
    const mermaid = mermaidForBlock(name)
    if (mermaid && mermaid.trim()) {
      out.push(mermaidDocstring(mermaid, indent))
    }
  }
  return out.join('\n')
}

/** Module-level overview: the whole graph as one mermaid block, as a comment. */
function moduleOverview(graphMermaid: string): string {
  if (!graphMermaid.trim()) return ''
  const body = ['```mermaid', graphMermaid.trim(), '```'].join('\n')
  const quoted = body
    .split('\n')
    .map((l) => l)
    .join('\n')
  return `"""Generated model.\n\nArchitecture overview:\n\n${quoted}\n"""\n\n`
}

/**
 * Generate Python with embedded Mermaid. The core code is produced by the
 * existing `generate()`; we only decorate it with diagrams.
 */
export function generateWithMermaid(
  nodes: NodeLike[],
  connections: Connection[],
  framework: string,
  options: MermaidCodegenOptions = {}
): string {
  const { mermaidForBlock = () => null, graphMermaid = {} } = options
  const base = generate(nodes, connections, framework)

  // Per-class block diagrams (leaf blocks + group classes that have a diagram).
  const withClasses = injectClassDocstrings(base, mermaidForBlock)

  // Module-level whole-graph overview, prepended after the __future__ import so
  // it stays valid Python (the future import must be the first statement).
  const overview = moduleOverview(graphToMermaid(nodes, connections, graphMermaid))
  if (!overview) return withClasses

  const lines = withClasses.split('\n')
  // `from __future__ import annotations` must remain the first statement; insert
  // the overview docstring right after it (and its trailing blank line).
  let insertAt = 0
  if (lines[0]?.startsWith('from __future__')) {
    insertAt = 1
    while (insertAt < lines.length && lines[insertAt].trim() === '') insertAt++
  }
  const head = lines.slice(0, insertAt).join('\n')
  const tail = lines.slice(insertAt).join('\n')
  return `${head}${head ? '\n' : ''}${overview}${tail}`
}
