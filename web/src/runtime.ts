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

/**
 * Restrict the graph to `stopAtNodeId` and everything UPSTREAM of it (its
 * ancestor closure over the connection graph), dropping all downstream nodes.
 * The target node then has no consumer, so codegen's terminal detection turns
 * its output into the model's return value - i.e. the forward pass runs only up
 * to (and including) the selected node. This lets you click the node just
 * before a runtime break and see every shape that resolved up to that point.
 *
 * Returns a {nodes, connections} view; the editor itself is never mutated.
 */
export function subgraphUpTo(
  editor: GraphEditor,
  stopAtNodeId: string
): { nodes: NodeLike[]; connections: any[] } {
  const allNodes = editor.getNodes()
  const allConns = editor.getConnections()
  const byId = new Map(allNodes.map((n) => [n.id, n]))
  if (!byId.has(stopAtNodeId)) return { nodes: allNodes, connections: allConns }

  // Incoming edges per node, then BFS upstream from the target.
  const incoming = new Map<string, any[]>()
  for (const c of allConns) {
    if (!incoming.has(c.target)) incoming.set(c.target, [])
    incoming.get(c.target)!.push(c)
  }
  const keep = new Set<string>([stopAtNodeId])
  const queue = [stopAtNodeId]
  while (queue.length) {
    const id = queue.shift()!
    for (const c of incoming.get(id) ?? []) {
      if (!keep.has(c.source)) {
        keep.add(c.source)
        queue.push(c.source)
      }
    }
  }
  // Input nodes are always kept so forward() still has its arguments, even when
  // the target's branch doesn't transitively reach every Input.
  for (const n of allNodes) if (n.entry?.kind === 'input') keep.add(n.id)

  // A kept group facade needs ALL its members (and recursively any nested group
  // members), since codegen partitions children by groupId, not by connection.
  // Likewise a kept grouped child needs its facade so the group still compiles.
  let grew = true
  while (grew) {
    grew = false
    for (const n of allNodes) {
      if (keep.has(n.id)) continue
      // member of a kept facade's group?
      const facadeKept =
        n.groupId &&
        allNodes.some(
          (f) => keep.has(f.id) && f.entry?.kind === 'group' && (f.entry as any).groupId === n.groupId
        )
      // facade whose group has a kept member?
      const facadeOfKeptMember =
        n.entry?.kind === 'group' &&
        allNodes.some((m) => keep.has(m.id) && m.groupId === (n.entry as any).groupId)
      if (facadeKept || facadeOfKeptMember) {
        keep.add(n.id)
        grew = true
      }
    }
  }

  const nodes = allNodes.filter((n) => keep.has(n.id))
  const connections = allConns.filter((c) => keep.has(c.source) && keep.has(c.target))
  return { nodes, connections }
}

/**
 * Build the POST body for the Python runner. When `stopAtNodeId` is given, the
 * graph is truncated to that node's ancestor closure so the forward pass runs
 * only up to the selected node (see {@link subgraphUpTo}).
 */
export function buildRunPayload(
  editor: GraphEditor,
  framework: string,
  batchSize = 2,
  stopAtNodeId?: string
): RunPayload {
  const view = stopAtNodeId
    ? subgraphUpTo(editor, stopAtNodeId)
    : { nodes: editor.getNodes(), connections: editor.getConnections() }
  // resolveInputSpecs only reads Input nodes + the validation sub, both still
  // valid on the truncated view, so reuse the editor for input back-solving.
  const inputs = resolveInputSpecs(editor, framework, batchSize).filter((spec) =>
    view.nodes.some((n) => n.id === spec.nodeId)
  )
  const code = generate(view.nodes, view.connections, framework, { trace: true })
  return { framework, code, inputs, batch_size: batchSize }
}

/**
 * POST to the local runner; returns the parsed per-port shapes + param count.
 * Pass `stopAtNodeId` to run only up to a selected node (incremental debugging).
 */
export async function runShapeCheck(
  editor: GraphEditor,
  framework: string,
  batchSize = 2,
  stopAtNodeId?: string
): Promise<RunResult> {
  if (framework !== 'pytorch') {
    throw new Error('Runtime shape check is PyTorch-only for now.')
  }
  const payload = buildRunPayload(editor, framework, batchSize, stopAtNodeId)
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
