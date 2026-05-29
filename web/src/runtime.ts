/**
 * Runtime shape checking via a local Python HTTP runner.
 *
 * Requires `python tools/shape_runner.py` (PyTorch only for now).
 * Sends traced codegen output + concrete input shapes; receives per-port
 * tensor.shape lists from a real forward pass.
 */

import { generate, planGraph } from './codegen.ts'
import { parseShapeString } from './nodes.ts'
import { resolve } from './shape.ts'
import type { Substitution } from './shape.ts'
import type { GraphEditor, NodeLike } from './types.ts'

export const RUNNER_URL = 'http://127.0.0.1:8765/run'

/** Result of the {@link isFullyConcrete} readiness check. */
export interface ConcreteCheck {
  ok: boolean
  reason?: string
}

/** A resolved input tensor spec, ordered to match generated forward() args. */
export interface InputSpec {
  nodeId: string
  arg: string | undefined
  shape: number[]
  dtype: string
}

/** The POST body sent to the Python shape runner. */
export interface RunPayload {
  framework: string
  code: string
  inputs: InputSpec[]
  batch_size: number
}

/** Parsed runner response surfaced to the UI. */
export interface RunResult {
  shapes: Map<string, number[]>
  numParams: number | null
  payload: RunPayload
}

/** An error carrying the offending node id, when the runner blames one. */
interface RunnerError extends Error {
  nodeId?: string
}

/**
 * True when the graph is ready for a Python forward pass.
 *
 * Output ports are intentionally *not* required to be concrete - axes like
 * ConvBlock's H_out/W_out are computed by PyTorch at runtime, which is what
 * the shape runner is for.
 */
export function isFullyConcrete(editor: GraphEditor, sub: Substitution, batchSize = 2): ConcreteCheck {
  const nodes = editor.getNodes()
  if (nodes.length === 0) return { ok: false, reason: 'Graph is empty.' }

  const hasInput = nodes.some((n) => n.entry.kind === 'input')
  if (!hasInput) return { ok: false, reason: 'Add at least one Input node.' }

  for (const n of nodes) {
    if (n.entry.kind === 'input') {
      // Input can remain symbolic here; the runtime payload builder back-solves
      // via substitution constraints and fills remaining unresolved axes with
      // deterministic defaults.
      const dims = inputShapeToDims(n, batchSize)
      if (!dims) continue
      continue
    }
    for (const port of n.entry.ctor) {
      if (!port.required) continue
      const v = n.values?.[port.name]
      const wired = editor
        .getConnections()
        .some((c) => c.target === n.id && c.targetInput === `__param__${port.name}`)
      if (wired) continue
      if (v === null || v === undefined || v === '') {
        return {
          ok: false,
          reason: `${n.entry.name}: required param "${port.name}" is unset.`,
        }
      }
    }
  }

  return { ok: true }
}

function inputShapeToDims(inputNode: NodeLike, batchSize: number): number[] | null {
  const tokens = parseShapeString(inputNode.values?.shape)
  if (tokens.length === 0) return null
  const dims: number[] = []
  for (const tok of tokens) {
    if (tok === 'B') dims.push(batchSize)
    else if (/^-?\d+$/.test(tok)) dims.push(Number(tok))
    else return null
  }
  return dims
}

function defaultForAxis(base: string): number {
  if (base === 'B') return 2
  if (base.startsWith('H') || base.startsWith('W')) return 32
  if (base.startsWith('T')) return 16
  if (base.startsWith('C')) return 16
  if (base.startsWith('D')) return 64
  if (base.startsWith('N')) return 16
  if (base.startsWith('K')) return 8
  return 8
}

function backSolveInputShape(
  inputNode: NodeLike,
  sub: Substitution,
  batchSize: number,
  axisDefaults: Map<string, number>
): number[] | null {
  const tokens = parseShapeString(inputNode.values?.shape)
  if (tokens.length === 0) return null
  const dims: number[] = []
  for (const tok of tokens) {
    if (tok === 'B') {
      dims.push(batchSize)
      continue
    }
    if (/^-?\d+$/.test(tok)) {
      dims.push(Number(tok))
      continue
    }
    // Try the node-freshened axis first (same naming convention as validator).
    const fresh = `${tok}#${inputNode.id}`
    let r = resolve(fresh, sub)
    if (typeof r === 'string') r = resolve(r, sub)
    if (typeof r === 'number' && Number.isFinite(r)) {
      dims.push(Math.trunc(r))
      continue
    }
    const base = String(r ?? tok).split('#')[0]!
    const known = axisDefaults.get(base)
    if (known != null) {
      dims.push(known)
      continue
    }
    const fallback = base === 'B' ? batchSize : defaultForAxis(base)
    axisDefaults.set(base, fallback)
    dims.push(fallback)
  }
  return dims
}

/**
 * Resolve concrete input shapes for the current graph using the validator's
 * solved substitution + axis defaults. Returns input specs ordered to match
 * the generated forward()'s argument order.
 *
 * Exposed so codegen can reuse the same back-solver when emitting an embedded
 * `test_GeneratedModel()` function - the test calls forward() with the same
 * shapes the runtime shape-runner would.
 */
export function resolveInputSpecs(editor: GraphEditor, _framework: string, batchSize = 2): InputSpec[] {
  const nodes = editor.getNodes()
  const connections = editor.getConnections()
  const plan = planGraph(nodes, connections)
  if (!plan) throw new Error('Graph is empty.')

  const sub: Substitution = editor.__lastValidationSub ?? new Map()
  const axisDefaults = new Map<string, number>([['B', batchSize]])
  const inputs: InputSpec[] = []
  for (const n of plan.ordered) {
    if (n.entry.kind !== 'input') continue
    const dims = backSolveInputShape(n, sub, batchSize, axisDefaults)
    if (!dims) throw new Error(`Input "${n.values?.name}" shape is empty.`)
    inputs.push({
      nodeId: n.id,
      arg: plan.inputArgFor.get(n.id),
      shape: dims,
      dtype: String(n.values?.dtype ?? 'float'),
    })
  }
  return inputs
}

/** Build the POST body for the Python runner. */
export function buildRunPayload(editor: GraphEditor, framework: string, batchSize = 2): RunPayload {
  const inputs = resolveInputSpecs(editor, framework, batchSize)
  const code = generate(editor.getNodes(), editor.getConnections(), framework, {
    trace: true,
  })
  return { framework, code, inputs, batch_size: batchSize }
}

/** POST to the local runner; returns the parsed per-port shapes + param count. */
export async function runShapeCheck(editor: GraphEditor, framework: string, batchSize = 2): Promise<RunResult> {
  if (framework !== 'pytorch') {
    throw new Error('Runtime shape check is PyTorch-only for now.')
  }
  const payload = buildRunPayload(editor, framework, batchSize)
  const res = await fetch(RUNNER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  const body = await res.json().catch(() => ({}) as Record<string, unknown>)
  if (!res.ok) {
    const err: RunnerError = new Error(String(body?.error ?? `Runner HTTP ${res.status}`))
    if (body?.node_id) err.nodeId = String(body.node_id)
    throw err
  }
  const shapes = new Map<string, number[]>(Object.entries((body.shapes ?? {}) as Record<string, number[]>))
  const numParams =
    typeof body.num_params === 'number' && Number.isFinite(body.num_params)
      ? Math.trunc(body.num_params)
      : null
  return { shapes, numParams, payload }
}
