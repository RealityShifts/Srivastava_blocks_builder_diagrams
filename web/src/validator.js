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

import { unifyShape, UnifyError, cloneSub } from './unify.js'
import { isVariable, isRest } from './shape.js'

export function validate(editor) {
  const nodes = editor.getNodes()
  const connections = editor.getConnections()
  const sub = new Map()
  const errors = []
  const warnings = []

  // 1. Seed the substitution with ctor-param-derived bindings.
  for (const n of nodes) {
    if (typeof n.applyParamBindings === 'function') {
      n.applyParamBindings(sub)
    }
  }

  // 2. Process every connection: unify the producer's output shape with the
  //    consumer's input shape.
  for (const c of connections) {
    const src = editor.getNode(c.source)
    const tgt = editor.getNode(c.target)
    if (!src || !tgt) continue
    const outShape = src.freshenedShape(c.sourceOutput, 'out')
    const inShape = tgt.freshenedShape(c.targetInput, 'in')
    if (!outShape || !inShape) {
      warnings.push({
        kind: 'missing-shape',
        connection: c.id,
        message: `Edge ${describeNode(src)}:${c.sourceOutput} -> ${describeNode(tgt)}:${c.targetInput} has untyped endpoint`,
      })
      continue
    }
    // Dtype compatibility: dtypes from the manifest are coarse labels.
    const outPort = src.outputs[c.sourceOutput]?.portSpec
    const inPort = tgt.inputs[c.targetInput]?.portSpec
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
          source: { node: src.id, port: c.sourceOutput, label: src.label },
          target: { node: tgt.id, port: c.targetInput, label: tgt.label },
          message: `Shape mismatch: ${describeNode(src)}:${c.sourceOutput} -> ${describeNode(tgt)}:${c.targetInput} — ${e.message}`,
        })
      } else throw e
    }
  }

  // 3. Detect required-but-unconnected input ports.
  const incoming = new Map()
  for (const c of connections) {
    const key = `${c.target}/${c.targetInput}`
    incoming.set(key, (incoming.get(key) ?? 0) + 1)
  }
  for (const n of nodes) {
    for (const port of n.entry.inputs) {
      if (port.optional || port.variadic) continue
      const key = `${n.id}/${port.name}`
      if (!incoming.get(key)) {
        warnings.push({
          kind: 'unconnected',
          message: `${describeNode(n)}:${port.name} is required but has no input`,
        })
      }
    }
  }

  // 4. Build per-port resolved shapes for hover/inspector display.
  const portShapes = new Map()
  for (const n of nodes) {
    for (const port of n.entry.inputs) {
      portShapes.set(
        `${n.id}/${port.name}/in`,
        n.freshenedShape(port.name, 'in')
      )
    }
    for (const port of n.entry.outputs) {
      portShapes.set(
        `${n.id}/${port.name}/out`,
        n.freshenedShape(port.name, 'out')
      )
    }
  }

  return { ok: errors.length === 0, errors, warnings, sub, portShapes }
}

/**
 * Predict whether a candidate edge would be accepted *without* mutating the
 * live substitution. Used to gate connection creation in the editor pipe.
 */
export function dryRunEdge(editor, sourceNode, sourcePort, targetNode, targetPort) {
  const { sub } = validate(editor)
  const trial = cloneSub(sub)
  const outShape = sourceNode.freshenedShape(sourcePort, 'out')
  const inShape = targetNode.freshenedShape(targetPort, 'in')
  if (!outShape || !inShape) return { ok: true, reason: 'untyped' }
  // dtype check
  const outDtype = sourceNode.outputs[sourcePort]?.portSpec?.dtype
  const inDtype = targetNode.inputs[targetPort]?.portSpec?.dtype
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
    return { ok: false, reason: e.message }
  }
}

function describeNode(n) {
  return `${n.label}#${shortId(n.id)}`
}

function shortId(id) {
  return String(id).slice(0, 6)
}
