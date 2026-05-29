/**
 * Graph validator: walks the rete graph, runs unification across every edge,
 * surfaces a list of diagnostics, and returns the resolved substitution so
 * the UI can display concrete per-port shapes.
 *
 * Returns:
 *   {
 *     ok: boolean,
 *     errors: [{ kind, message, where, ... }],
 *     warnings: [...],
 *     sub: Map<string, string|number>,   // substitution after all edges
 *     portShapes: Map<"nodeId/portName/side", token[]>
 *   }
 */

import { unifyShape, UnifyError, cloneSub } from './unify.ts'
import { isVariable, isRest } from './shape.ts'
import { boundarySignatureFromEntry, boundarySignaturesMatch } from './groupBoundary.ts'
import type { GraphEditor, NodeLike } from './types.ts'
import type { Shape, Token, Substitution } from './shape.ts'

// `isVariable`/`isRest` are imported to keep parity with the original module
// surface (re-exported convenience for downstream callers); reference them so
// erasable-only type-stripping leaves them as live value imports.
void isVariable
void isRest

/**
 * A single validation diagnostic (error or warning). All diagnostics carry a
 * machine-readable `kind` plus a human-readable `message`. Individual kinds may
 * attach extra fields (e.g. `connection`, `source`, `target`) — the index
 * signature keeps those open without losing the required core fields.
 */
export interface Diagnostic {
  /** Machine-readable category, e.g. `'shape'`, `'dtype'`, `'tag-conflict'`. */
  kind: string
  /** Human-readable description shown in the UI. */
  message: string
  /** Optional location hint (some diagnostics use this instead of structured fields). */
  where?: string
  /** Forward-compatible extra fields (connection id, source/target endpoints, ...). */
  [key: string]: unknown
}

/**
 * The result of a full graph validation pass.
 */
export interface ValidationResult {
  /** True when there are no hard errors (warnings are allowed). */
  ok: boolean
  /** Hard errors that block codegen. */
  errors: Diagnostic[]
  /** Non-blocking warnings surfaced to the user. */
  warnings: Diagnostic[]
  /** The resolved substitution after unifying across every edge. */
  sub: Substitution
  /** Per-port resolved shapes keyed by `"nodeId/portName/side"`. */
  portShapes: Map<string, Token[]>
}

/** The outcome of a {@link dryRunEdge} prediction. */
export interface DryRunResult {
  /** Whether the candidate edge would be accepted. */
  ok: boolean
  /** When present, why the edge was rejected or flagged (e.g. `'untyped'`, dtype/shape reason). */
  reason?: string
}

export function validate(editor: GraphEditor): ValidationResult {
  const nodes = editor.getNodes()
  const connections = editor.getConnections()
  const sub: Substitution = new Map()
  const errors: Diagnostic[] = []
  const warnings: Diagnostic[] = []

  // 1. Seed the substitution with ctor-param-derived bindings.
  for (const n of nodes) {
    if (typeof (n as any).applyParamBindings === 'function') {
      (n as any).applyParamBindings(sub)
    }
  }

  // 2. Process every connection: unify the producer's output shape with the
  //    consumer's input shape.
  for (const c of connections) {
    const src = editor.getNode?.(c.source) as NodeLike | null | undefined
    const tgt = editor.getNode?.(c.target) as NodeLike | null | undefined
    if (!src || !tgt) continue
    const targetSpec = (tgt.inputs as any)?.[c.targetInput]?.portSpec
    if (targetSpec?.kind === 'param') continue
    const outShape = (src as any).freshenedShape(c.sourceOutput, 'out') as Shape | undefined
    const inShape = (tgt as any).freshenedShape(c.targetInput, 'in') as Shape | undefined
    if (!outShape || !inShape) {
      warnings.push({
        kind: 'missing-shape',
        connection: c.id,
        message: `Edge ${describeNode(src)}:${c.sourceOutput} -> ${describeNode(tgt)}:${c.targetInput} has untyped endpoint`,
      })
      continue
    }
    // Dtype compatibility: dtypes from the manifest are coarse labels.
    const outPort = (src.outputs as any)[c.sourceOutput]?.portSpec
    const inPort = (tgt.inputs as any)[c.targetInput]?.portSpec
    if (
      outPort &&
      inPort &&
      outPort.dtype !== 'any' &&
      inPort.dtype !== 'any' &&
      outPort.dtype !== inPort.dtype
    ) {
      errors.push({
        kind: 'dtype',
        connection: c.id,
        message: `dtype mismatch on ${describeNode(src)}:${c.sourceOutput} (${outPort.dtype}) -> ${describeNode(tgt)}:${c.targetInput} (${inPort.dtype})`,
      })
    }
    try {
      unifyShape(outShape, inShape, sub)
    } catch (e) {
      if (e instanceof UnifyError) {
        errors.push({
          kind: 'shape',
          connection: c.id,
          source: { node: src.id, port: c.sourceOutput, label: (src as any).label },
          target: { node: tgt.id, port: c.targetInput, label: (tgt as any).label },
          message: `Shape mismatch: ${describeNode(src)}:${c.sourceOutput} -> ${describeNode(tgt)}:${c.targetInput} — ${e.message}`,
        })
      } else throw e
    }
  }

  // 3a. Tag-based weight sharing: module nodes and group facades sharing a
  //     non-empty tag must agree on block/group type (and ctor values for
  //     modules), otherwise codegen would emit one self.<attr> backed by a
  //     single instance yet called from sites that semantically expect another.
  const tagGroups = new Map<string, NodeLike[]>()
  for (const n of nodes) {
    if (n.entry.kind !== 'module' && n.entry.kind !== 'group') continue
    const t = String(n.tag ?? '').trim()
    if (!t) continue
    if (!tagGroups.has(t)) tagGroups.set(t, [])
    tagGroups.get(t)!.push(n)
  }
  // Explicit per-instance name (blank names don't participate in the check -
  // they neither sync params nor force a shared name).
  const explicitName = (n: NodeLike) => String(n.name ?? '').trim()
  for (const [tag, group] of tagGroups) {
    if (group.length < 2) continue
    const head = group[0]
    for (const other of group.slice(1)) {
      // Name implies tag: weight-shared instances that carry explicit names must
      // share them, since the name both syncs their params and names the single
      // self.<attr>. (If either name is blank, fall through to the type/ctor
      // checks below.)
      const headName = explicitName(head)
      const otherName = explicitName(other)
      if (headName && otherName && headName.toLowerCase() !== otherName.toLowerCase()) {
        errors.push({
          kind: 'tag-conflict',
          message: `Tag "${tag}" shared by nodes with different names ("${headName}" vs "${otherName}"). Weight-shared instances must share a name.`,
        })
        continue
      }
      if (other.entry.name !== head.entry.name) {
        errors.push({
          kind: 'tag-conflict',
          message:
            head.entry.kind === 'module' && other.entry.kind === 'module'
              ? `Tag "${tag}" reused across different block types: ${head.entry.name} vs ${other.entry.name}. Same tag = shared weights, so types must match.`
              : `Tag "${tag}" reused across different types: ${head.entry.name} (${head.entry.kind}) vs ${other.entry.name} (${other.entry.kind}). Same tag = shared weights, so types must match.`,
        })
        continue
      }
      if (head.entry.kind === 'group' && other.entry.kind === 'group') {
        const sigA = boundarySignatureFromEntry(head.entry)
        const sigB = boundarySignatureFromEntry(other.entry)
        if (!boundarySignaturesMatch(sigA, sigB)) {
          const headIn = sigA.inputs.length
          const headOut = sigA.outputs.length
          const headP = sigA.params.length
          const otherIn = sigB.inputs.length
          const otherOut = sigB.outputs.length
          const otherP = sigB.params.length
          errors.push({
            kind: 'tag-conflict',
            message: `Tag "${tag}" on group "${other.entry.name}": boundary interface differs (${headIn}in/${headOut}out/${headP}param vs ${otherIn}in/${otherOut}out/${otherP}param). Weight-shared groups must have matching receivers and outlets.`,
          })
        }
        continue
      }
      if (head.entry.kind !== other.entry.kind) {
        errors.push({
          kind: 'tag-conflict',
          message: `Tag "${tag}" reused across a block and a group (${head.entry.name}). Same tag = shared weights, so kinds must match.`,
        })
        continue
      }
      const diff = ctorValueDiff(head, other)
      if (diff) {
        errors.push({
          kind: 'tag-conflict',
          message: `Tag "${tag}" on ${other.entry.name}: ctor "${diff.param}" differs (${JSON.stringify(diff.a)} vs ${JSON.stringify(diff.b)}). Weight-shared instances must use identical ctor values.`,
        })
      }
    }
  }

  // 3. Detect required-but-unconnected input ports.
  const incoming = new Map<string, number>()
  for (const c of connections) {
    const key = `${c.target}/${c.targetInput}`
    incoming.set(key, (incoming.get(key) ?? 0) + 1)
  }
  // Child ports whose boundary edge was rerouted to a (collapsed) group
  // facade no longer have a direct incoming edge - that's expected, not a
  // warning. We pull the ownership map out of every facade's portMap.
  const ownedByFacade = collectFacadeOwnership(nodes)
  for (const n of nodes) {
    for (const port of n.entry.inputs) {
      if (port.optional || port.variadic) continue
      const key = `${n.id}/${port.name}`
      if (!incoming.get(key) && !ownedByFacade.inputs.has(key)) {
        warnings.push({
          kind: 'unconnected',
          message: `${describeNode(n)}:${port.name} is required but has no input`,
        })
      }
    }
  }

  // Concat / Stack need at least two wires on their variadic port.
  for (const n of nodes) {
    if (n.entry.kind !== 'concat' && n.entry.kind !== 'stack') continue
    for (const port of n.entry.inputs) {
      if (!port.variadic) continue
      const count = connections.filter(
        (c) => c.target === n.id && c.targetInput === port.name
      ).length
      if (count < 2) {
        warnings.push({
          kind: 'variadic-min',
          message: `${describeNode(n)}: needs ≥2 inputs on ${port.name}, got ${count}`,
        })
      }
    }
  }

  // 4. Build per-port resolved shapes for hover/inspector display.
  const portShapes = new Map<string, Token[]>()
  for (const n of nodes) {
    for (const port of n.entry.inputs) {
      portShapes.set(
        `${n.id}/${port.name}/in`,
        (n as any).freshenedShape(port.name, 'in')
      )
    }
    for (const port of n.entry.outputs) {
      portShapes.set(
        `${n.id}/${port.name}/out`,
        (n as any).freshenedShape(port.name, 'out')
      )
    }
  }

  return { ok: errors.length === 0, errors, warnings, sub, portShapes }
}

/**
 * Predict whether a candidate edge would be accepted *without* mutating the
 * live substitution. Used to gate connection creation in the editor pipe.
 */
export function dryRunEdge(
  editor: GraphEditor,
  sourceNode: NodeLike,
  sourcePort: string,
  targetNode: NodeLike,
  targetPort: string
): DryRunResult {
  const targetSpec = (targetNode.inputs as any)?.[targetPort]?.portSpec
  if (targetSpec?.kind === 'param') return { ok: true }
  const { sub } = validate(editor)
  const trial = cloneSub(sub)
  const outShape = (sourceNode as any).freshenedShape(sourcePort, 'out') as Shape | undefined
  const inShape = (targetNode as any).freshenedShape(targetPort, 'in') as Shape | undefined
  if (!outShape || !inShape) return { ok: true, reason: 'untyped' }
  // dtype check
  const outDtype = (sourceNode.outputs as any)[sourcePort]?.portSpec?.dtype
  const inDtype = (targetNode.inputs as any)[targetPort]?.portSpec?.dtype
  if (
    outDtype &&
    inDtype &&
    outDtype !== 'any' &&
    inDtype !== 'any' &&
    outDtype !== inDtype
  ) {
    return { ok: false, reason: `dtype ${outDtype} -> ${inDtype}` }
  }
  try {
    unifyShape(outShape, inShape, trial)
    return { ok: true }
  } catch (e) {
    return { ok: false, reason: (e as Error).message }
  }
}

function describeNode(n: NodeLike): string {
  return `${(n as any).label}#${shortId(n.id)}`
}

/** First ctor param whose value differs between two nodes, or null. */
function ctorValueDiff(
  a: NodeLike,
  b: NodeLike
): { param: string; a: unknown; b: unknown } | null {
  for (const p of a.entry.ctor || []) {
    const va = a.values?.[p.name]
    const vb = b.values?.[p.name]
    if (!ctorValueEqual(va, vb)) return { param: p.name, a: va, b: vb }
  }
  return null
}

function ctorValueEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true
  // Treat null/undefined/'' as the same "unset" sentinel - common for
  // implicit-inferred ctor params like in_ch that get filled in later.
  const isUnset = (v: unknown) => v === null || v === undefined || v === ''
  if (isUnset(a) && isUnset(b)) return true
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((x, i) => ctorValueEqual(x, b[i]))
  }
  return false
}

function shortId(id: string): string {
  return String(id).slice(0, 6)
}

/**
 * Build the set of (childNodeId/childPort) keys that are currently proxied by
 * a group facade. Used to suppress false "required input is unconnected"
 * warnings: when the boundary edge has been rerouted to the facade, the
 * child's own input port looks dangling but is in fact wired through.
 */
function collectFacadeOwnership(
  nodes: NodeLike[]
): { inputs: Set<string>; outputs: Set<string> } {
  const inputs = new Set<string>()
  const outputs = new Set<string>()
  for (const n of nodes) {
    if (n.entry?.kind !== 'group') continue
    const portMap = n.entry.portMap as any
    for (const m of portMap?.inputs || []) {
      inputs.add(`${m.childNodeId}/${m.childPort}`)
    }
    for (const m of portMap?.params || []) {
      inputs.add(`${m.childNodeId}/${m.childPort}`)
    }
    for (const m of portMap?.outputs || []) {
      outputs.add(`${m.childNodeId}/${m.childPort}`)
    }
  }
  return { inputs, outputs }
}
