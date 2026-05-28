/**
 * Runtime shape checking via a local Python HTTP runner.
 *
 * Requires `python tools/shape_runner.py` (PyTorch only for now).
 * Sends traced codegen output + concrete input shapes; receives per-port
 * tensor.shape lists from a real forward pass.
 */

import { generate, planGraph } from './codegen.js'
import { parseShapeString } from './nodes.js'
import { resolve } from './shape.js'

export const RUNNER_URL = 'http://127.0.0.1:8765/run'

/** True when the graph is ready for a Python forward pass.

Output ports are intentionally *not* required to be concrete — axes like
ConvBlock's H_out/W_out are computed by PyTorch at runtime, which is what
the shape runner is for.
*/
export function isFullyConcrete(editor, sub, batchSize = 2) {
  const nodes = editor.getNodes()
  if (nodes.length === 0) return { ok: false, reason: 'Graph is empty.' }

  const hasInput = nodes.some((n) => n.entry.kind === 'input')
  if (!hasInput) return { ok: false, reason: 'Add at least one Input node.' }

  for (const n of nodes) {
    if (n.entry.kind === 'input') {
      // Input can remain symbolic here; runtime payload builder back-solves via
      // substitution constraints and fills remaining unresolved axes with
      // deterministic defaults.
      const dims = inputShapeToDims(n, batchSize)
      if (!dims) continue
      continue
    }
    for (const port of n.entry.ctor) {
      if (!port.required) continue
      const v = n.values[port.name]
      const wired = editor.getConnections().some(
        (c) => c.target === n.id && c.targetInput === `__param__${port.name}`
      )
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

function inputShapeToDims(inputNode, batchSize) {
  const tokens = parseShapeString(inputNode.values?.shape)
  if (tokens.length === 0) return null
  const dims = []
  for (const tok of tokens) {
    if (tok === 'B') dims.push(batchSize)
    else if (/^-?\d+$/.test(tok)) dims.push(Number(tok))
    else return null
  }
  return dims
}

function defaultForAxis(base) {
  if (base === 'B') return 2
  if (base.startsWith('H') || base.startsWith('W')) return 32
  if (base.startsWith('T')) return 16
  if (base.startsWith('C')) return 16
  if (base.startsWith('D')) return 64
  if (base.startsWith('N')) return 16
  if (base.startsWith('K')) return 8
  return 8
}

function backSolveInputShape(inputNode, sub, batchSize, axisDefaults) {
  const tokens = parseShapeString(inputNode.values?.shape)
  if (tokens.length === 0) return null
  const dims = []
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
    const base = String(r ?? tok).split('#')[0]
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
 * solved substitution + axis defaults. Returns `[{nodeId, arg, shape, dtype}]`
 * ordered to match the generated forward()'s argument order.
 *
 * Exposed so codegen can reuse the same back-solver when emitting an
 * embedded `test_GeneratedModel()` function — the test calls forward() with
 * the same shapes the runtime shape-runner would.
 */
export function resolveInputSpecs(editor, framework, batchSize = 2) {
  const nodes = editor.getNodes()
  const connections = editor.getConnections()
  const plan = planGraph(nodes, connections)
  if (!plan) throw new Error('Graph is empty.')

  const sub = (editor && editor.__lastValidationSub) || new Map()
  const axisDefaults = new Map([['B', batchSize]])
  const inputs = []
  for (const n of plan.ordered) {
    if (n.entry.kind !== 'input') continue
    const dims = backSolveInputShape(n, sub, batchSize, axisDefaults)
    if (!dims) throw new Error(`Input "${n.values?.name}" shape is empty.`)
    inputs.push({
      nodeId: n.id,
      arg: plan.inputArgFor.get(n.id),
      shape: dims,
      dtype: n.values?.dtype ?? 'float',
    })
  }
  return inputs
}

/** Build POST body for the Python runner. */
export function buildRunPayload(editor, framework, batchSize = 2) {
  const inputs = resolveInputSpecs(editor, framework, batchSize)
  const code = generate(editor.getNodes(), editor.getConnections(), framework, {
    trace: true,
  })
  return { framework, code, inputs, batch_size: batchSize }
}

/** POST to the local runner; returns Map<"nodeId/port", number[]>. */
export async function runShapeCheck(editor, framework, batchSize = 2) {
  if (framework !== 'pytorch') {
    throw new Error('Runtime shape check is PyTorch-only for now.')
  }
  const payload = buildRunPayload(editor, framework, batchSize)
  const res = await fetch(RUNNER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    const err = new Error(body?.error ?? `Runner HTTP ${res.status}`)
    if (body?.node_id) err.nodeId = body.node_id
    throw err
  }
  const shapes = new Map(Object.entries(body.shapes ?? {}))
  const numParams =
    typeof body.num_params === 'number' && Number.isFinite(body.num_params)
      ? Math.trunc(body.num_params)
      : null
  return { shapes, numParams, payload }
}
