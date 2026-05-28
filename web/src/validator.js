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
    const targetSpec = tgt.inputs?.[c.targetInput]?.portSpec
    if (targetSpec?.kind === 'param') continue
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

  // 3a. Tag-based weight sharing: module nodes and group facades sharing a
  //     non-empty tag must agree on block/group type (and ctor values for
  //     modules), otherwise codegen would emit one self.<attr> backed by a
  //     single instance yet called from sites that semantically expect another.
  const tagGroups = new Map()
  for (const n of nodes) {
    if (n.entry.kind !== 'module' && n.entry.kind !== 'group') continue
    const t = String(n.tag ?? '').trim()
    if (!t) continue
    if (!tagGroups.has(t)) tagGroups.set(t, [])
    tagGroups.get(t).push(n)
  }
  for (const [tag, group] of tagGroups) {
    if (group.length < 2) continue
    const head = group[0]
    for (const other of group.slice(1)) {
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
        const headIn = head.entry.inputs?.length ?? 0
        const headOut = head.entry.outputs?.length ?? 0
        const otherIn = other.entry.inputs?.length ?? 0
        const otherOut = other.entry.outputs?.length ?? 0
        if (headIn !== otherIn || headOut !== otherOut) {
          errors.push({
            kind: 'tag-conflict',
            message: `Tag "${tag}" on group "${other.entry.name}": boundary ports differ (${headIn}in/${headOut}out vs ${otherIn}in/${otherOut}out). Weight-shared groups must have matching interfaces.`,
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
  const incoming = new Map()
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
  const targetSpec = targetNode.inputs?.[targetPort]?.portSpec
  if (targetSpec?.kind === 'param') return { ok: true }
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

/** First ctor param whose value differs between two nodes, or null. */
function ctorValueDiff(a, b) {
  for (const p of a.entry.ctor || []) {
    const va = a.values?.[p.name]
    const vb = b.values?.[p.name]
    if (!ctorValueEqual(va, vb)) return { param: p.name, a: va, b: vb }
  }
  return null
}

function ctorValueEqual(a, b) {
  if (a === b) return true
  // Treat null/undefined/'' as the same "unset" sentinel - common for
  // implicit-inferred ctor params like in_ch that get filled in later.
  const isUnset = (v) => v === null || v === undefined || v === ''
  if (isUnset(a) && isUnset(b)) return true
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((x, i) => ctorValueEqual(x, b[i]))
  }
  return false
}

function shortId(id) {
  return String(id).slice(0, 6)
}

/**
 * Build the set of (childNodeId/childPort) keys that are currently proxied by
 * a group facade. Used to suppress false "required input is unconnected"
 * warnings: when the boundary edge has been rerouted to the facade, the
 * child's own input port looks dangling but is in fact wired through.
 */
function collectFacadeOwnership(nodes) {
  const inputs = new Set()
  const outputs = new Set()
  for (const n of nodes) {
    if (n.entry?.kind !== 'group') continue
    for (const m of n.entry.portMap?.inputs || []) {
      inputs.add(`${m.childNodeId}/${m.childPort}`)
    }
    for (const m of n.entry.portMap?.outputs || []) {
      outputs.add(`${m.childNodeId}/${m.childPort}`)
    }
  }
  return { inputs, outputs }
}
