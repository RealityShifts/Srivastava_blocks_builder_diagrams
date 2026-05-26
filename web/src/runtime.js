/**
 * Runtime shape checking via a local Python HTTP runner.
 *
 * Requires `python tools/shape_runner.py` (PyTorch only for now).
 * Sends traced codegen output + concrete input shapes; receives per-port
 * tensor.shape lists from a real forward pass.
 */

import { generate, planGraph } from './codegen.js'
import { parseShapeString } from './nodes.js'
import { isLiteral, isRest, isVariable, resolve } from './shape.js'

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
      const dims = inputShapeToDims(n, batchSize)
      if (!dims) {
        return {
          ok: false,
          reason: `Input "${n.values?.name || 'x'}" shape is not fully numeric (use literals or B for batch).`,
        }
      }
      continue
    }
    for (const port of n.entry.ctor) {
      if (!port.required) continue
      const v = n.values[port.name]
      if (v === null || v === undefined || v === '') {
        return {
          ok: false,
          reason: `${n.entry.name}: required param "${port.name}" is unset.`,
        }
      }
    }
  }

  // Only input ports must be concrete — output axes like H_out/W_out are
  // derived by the blocks themselves and filled in by the runtime runner.
  for (const n of nodes) {
    if (n.entry.kind === 'input') continue
    for (const port of n.entry.inputs) {
      const shape = n.freshenedShape(port.name, 'in')
      if (!shape) continue
      if (hasRest(shape)) {
        return { ok: false, reason: 'Variadic/rest (…) shapes cannot be run yet.' }
      }
      const dims = resolveShapeToDims(shape, sub, batchSize)
      if (!dims) {
        return {
          ok: false,
          reason: `${n.entry.name}:${port.name} still has unresolved axes.`,
        }
      }
    }
  }

  return { ok: true }
}

function hasRest(shape) {
  return shape.some(isRest)
}

function resolveShapeToDims(shape, sub, batchSize) {
  const dims = []
  for (const tok of shape) {
    if (isRest(tok)) return null
    if (isLiteral(tok)) {
      dims.push(tok)
      continue
    }
    let r = resolve(tok, sub)
    if (isVariable(r)) {
      const base = String(r).split('#')[0]
      if (base === 'B') r = batchSize
      else return null
    }
    if (typeof r === 'number' && Number.isFinite(r)) dims.push(Math.trunc(r))
    else return null
  }
  return dims
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

/** Build POST body for the Python runner. */
export function buildRunPayload(editor, framework, batchSize = 2) {
  const nodes = editor.getNodes()
  const connections = editor.getConnections()
  const plan = planGraph(nodes, connections)
  if (!plan) throw new Error('Graph is empty.')

  const code = generate(nodes, connections, framework, { trace: true })
  const inputs = []
  for (const n of plan.ordered) {
    if (n.entry.kind !== 'input') continue
    const dims = inputShapeToDims(n, batchSize)
    if (!dims) throw new Error(`Input "${n.values?.name}" shape is not concrete.`)
    inputs.push({
      nodeId: n.id,
      arg: plan.inputArgFor.get(n.id),
      shape: dims,
      dtype: n.values?.dtype ?? 'float',
    })
  }

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
    throw new Error(body.error || `Runner HTTP ${res.status}`)
  }
  const shapes = new Map(Object.entries(body.shapes ?? {}))
  return { shapes, payload }
}
