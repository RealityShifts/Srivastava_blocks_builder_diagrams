/**
 * Generate a runnable Python module from the rete graph.
 *
 * Emits a single nn.Module subclass:
 *   - imports each referenced block from its origin module
 *   - in __init__, instantiates each module-kind node with its ctor values
 *   - in forward(), traverses nodes in topological order, wiring outputs to
 *     inputs by referenced variable names
 *   - functions (kind=="function") are called inline in forward()
 *
 * The generator is framework-agnostic; PyTorch and Flax differ only in the
 * base class name and the entry method (forward vs __call__).
 */

function topoSort(nodes, connections) {
  const indeg = new Map()
  const out = new Map()
  for (const n of nodes) {
    indeg.set(n.id, 0)
    out.set(n.id, [])
  }
  for (const c of connections) {
    indeg.set(c.target, (indeg.get(c.target) ?? 0) + 1)
    out.get(c.source).push(c.target)
  }
  const queue = [...indeg.entries()].filter(([, d]) => d === 0).map(([k]) => k)
  const order = []
  while (queue.length) {
    const id = queue.shift()
    order.push(id)
    for (const tgt of out.get(id) ?? []) {
      indeg.set(tgt, indeg.get(tgt) - 1)
      if (indeg.get(tgt) === 0) queue.push(tgt)
    }
  }
  if (order.length !== nodes.length) {
    throw new Error('graph contains a cycle')
  }
  return order.map((id) => nodes.find((n) => n.id === id))
}

function pyRepr(value) {
  if (value === null || value === undefined) return 'None'
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (typeof value === 'number') return String(value)
  if (typeof value === 'string') return JSON.stringify(value)
  if (Array.isArray(value)) return `[${value.map(pyRepr).join(', ')}]`
  return JSON.stringify(value)
}

function snake(name, suffix) {
  const base = name
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1_$2')
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .toLowerCase()
  return `${base}_${suffix}`
}

/**
 * @param {Array} nodes      array of BlockNode
 * @param {Array} connections array of {source, sourceOutput, target, targetInput}
 * @param {'pytorch'|'flax'} framework
 */
export function generate(nodes, connections, framework) {
  if (nodes.length === 0) return '# empty graph\n'
  const ordered = topoSort(nodes, connections)

  // Track which Python variable names are taken so Input names and snake_case
  // attribute names never collide.
  const usedNames = new Set()
  const allocate = (base) => {
    let candidate = base
    let i = 2
    while (usedNames.has(candidate)) candidate = `${base}${i++}`
    usedNames.add(candidate)
    return candidate
  }

  // Input nodes' output vars come from their user-typed `name` (sanitized).
  // They become forward() arguments rather than `self.xxx` attributes.
  const inputArgFor = new Map() // nodeId -> python arg name
  for (const n of ordered) {
    if (n.entry.kind === 'input') {
      const raw = sanitizePyIdent(n.values?.name, 'x')
      inputArgFor.set(n.id, allocate(raw))
    }
  }

  // Stable per-node Python attribute names for module/function nodes.
  const attrName = new Map()
  ordered.forEach((n, i) => {
    if (n.entry.kind === 'input') return
    attrName.set(n.id, allocate(snake(n.entry.name, i)))
  })

  // Imports: group by module. Skip built-in Input.
  const imports = new Map()
  for (const n of ordered) {
    if (n.entry.kind === 'input') continue
    if (!imports.has(n.entry.module)) imports.set(n.entry.module, new Set())
    imports.get(n.entry.module).add(n.entry.name)
  }

  // Connections, indexed by (target, targetInput) -> [(source, sourceOutput)]
  const incoming = new Map()
  for (const c of connections) {
    const key = `${c.target}/${c.targetInput}`
    if (!incoming.has(key)) incoming.set(key, [])
    incoming.get(key).push(c)
  }

  // For each node, record the Python variable holding each of its outputs.
  const outputVarFor = new Map() // `${nodeId}/${portName}` -> python var
  for (const n of ordered) {
    if (n.entry.kind === 'input') {
      outputVarFor.set(`${n.id}/out`, inputArgFor.get(n.id))
      continue
    }
    const multi = n.entry.outputs.length > 1
    for (const port of n.entry.outputs) {
      outputVarFor.set(
        `${n.id}/${port.name}`,
        multi ? `${attrName.get(n.id)}_${port.name}` : attrName.get(n.id)
      )
    }
  }

  // ------------------- imports -------------------
  const lines = ['from __future__ import annotations', '']
  if (framework === 'pytorch') {
    lines.push('import torch')
    lines.push('import torch.nn as nn')
  } else {
    lines.push('import jax')
    lines.push('import jax.numpy as jnp')
    lines.push('import flax.nnx as nnx')
  }
  for (const [mod, names] of imports) {
    lines.push(`from ${mod} import ${[...names].sort().join(', ')}`)
  }
  lines.push('')

  // ------------------- class -------------------
  const baseClass = framework === 'pytorch' ? 'nn.Module' : 'nnx.Module'
  lines.push(`class GeneratedModel(${baseClass}):`)

  // ------- __init__ -------
  if (framework === 'pytorch') {
    lines.push('    def __init__(self):')
    lines.push('        super().__init__()')
  } else {
    lines.push('    def __init__(self, *, rngs: nnx.Rngs):')
  }

  const moduleNodes = ordered.filter((n) => n.entry.kind === 'module')
  if (moduleNodes.length === 0) lines.push('        pass')
  for (const n of moduleNodes) {
    const args = ctorArgs(n)
    if (framework === 'flax') args.push('rngs=rngs')
    lines.push(`        self.${attrName.get(n.id)} = ${n.entry.name}(${args.join(', ')})`)
  }
  lines.push('')

  // ------- forward / __call__ -------
  // Explicit Input nodes (in topo order) come first; dangling required inputs
  // on regular nodes still get auto-promoted to forward() args as a fallback.
  const entryInputs = findEntryInputs(ordered, connections, usedNames)
  const inputArgs = ordered
    .filter((n) => n.entry.kind === 'input')
    .map((n) => inputArgFor.get(n.id))
  const allArgs = [...inputArgs, ...entryInputs.map((e) => e.portArg)]
  const fwdName = framework === 'pytorch' ? 'forward' : '__call__'
  lines.push(`    def ${fwdName}(self${allArgs.length ? ', ' + allArgs.join(', ') : ''}):`)

  // Map each entry-input port to a function argument name.
  const argFor = new Map()
  for (const e of entryInputs) argFor.set(`${e.nodeId}/${e.portName}`, e.argName)

  for (const n of ordered) {
    // Input nodes are pure forward() args, not statements in the body.
    if (n.entry.kind === 'input') continue
    // Build the argument expression for each input port of this node.
    const callArgs = []
    for (const port of n.entry.inputs) {
      const key = `${n.id}/${port.name}`
      const incomingEdges = incoming.get(key) ?? []

      if (incomingEdges.length === 0) {
        if (argFor.has(key)) {
          callArgs.push(`${port.name}=${argFor.get(key)}`)
        } else if (port.optional) {
          // skip
        } else {
          callArgs.push(`${port.name}=None  # TODO: dangling required input`)
        }
        continue
      }

      if (port.variadic) {
        const refs = incomingEdges
          .map((c) => outputVarFor.get(`${c.source}/${c.sourceOutput}`))
          .filter(Boolean)
        callArgs.push(`${port.name}=[${refs.join(', ')}]`)
      } else {
        const c = incomingEdges[0]
        callArgs.push(
          `${port.name}=${outputVarFor.get(`${c.source}/${c.sourceOutput}`)}`
        )
      }
    }

    const callExpr =
      n.entry.kind === 'module'
        ? `self.${attrName.get(n.id)}(${callArgs.join(', ')})`
        : `${n.entry.name}(${callArgs.join(', ')})`

    const multi = n.entry.outputs.length > 1
    if (multi) {
      const targets = n.entry.outputs
        .map((p) => outputVarFor.get(`${n.id}/${p.name}`))
        .join(', ')
      lines.push(`        ${targets} = ${callExpr}`)
    } else {
      lines.push(`        ${attrName.get(n.id)} = ${callExpr}`)
    }
  }

  // Return the terminal node(s)' outputs.
  const terminals = findTerminals(ordered, connections)
  if (terminals.length === 0) {
    lines.push('        return None')
  } else if (terminals.length === 1) {
    const t = terminals[0]
    const v = outputVarFor.get(`${t.nodeId}/${t.portName}`)
    lines.push(`        return ${v}`)
  } else {
    const vs = terminals.map((t) => outputVarFor.get(`${t.nodeId}/${t.portName}`))
    lines.push(`        return (${vs.join(', ')})`)
  }
  lines.push('')

  return lines.join('\n')
}

function ctorArgs(node) {
  const out = []
  for (const p of node.entry.ctor) {
    const v = node.values[p.name]
    if (v === null || v === undefined || v === '') continue
    // Skip if it equals the default to keep the output tidy.
    if (deepEqual(v, p.default)) continue
    out.push(`${p.name}=${pyRepr(coerce(v, p.type))}`)
  }
  return out
}

function coerce(v, type) {
  if (type === 'int' && typeof v === 'string') {
    const n = Number(v)
    if (Number.isFinite(n) && Number.isInteger(n)) return n
  }
  if (type === 'float' && typeof v === 'string') {
    const n = Number(v)
    if (Number.isFinite(n)) return n
  }
  if (type === 'bool' && typeof v === 'string') {
    if (v === 'true') return true
    if (v === 'false') return false
  }
  return v
}

function deepEqual(a, b) {
  if (a === b) return true
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((x, i) => deepEqual(x, b[i]))
  }
  return false
}

/** Input ports with no incoming edges that aren't optional ⇒ surfaced as forward() args. */
function findEntryInputs(nodes, connections, usedNames = new Set()) {
  const incomingKeys = new Set(connections.map((c) => `${c.target}/${c.targetInput}`))
  const out = []
  const allocate = (base) => {
    let candidate = base
    let i = 2
    while (usedNames.has(candidate)) candidate = `${base}${i++}`
    usedNames.add(candidate)
    return candidate
  }
  for (const n of nodes) {
    if (n.entry.kind === 'input') continue
    for (const port of n.entry.inputs) {
      if (port.optional || port.variadic) continue
      const key = `${n.id}/${port.name}`
      if (!incomingKeys.has(key)) {
        const argName = allocate(port.name)
        out.push({
          nodeId: n.id,
          portName: port.name,
          argName,
          portArg: argName,
        })
      }
    }
  }
  return out
}

/** Coerce arbitrary user-typed text into a valid Python identifier. */
function sanitizePyIdent(raw, fallback = 'x') {
  const s = String(raw ?? '').trim()
  if (!s) return fallback
  // Replace non-word chars with underscore, prefix digit-starting with underscore.
  let out = s.replace(/[^A-Za-z0-9_]/g, '_')
  if (/^[0-9]/.test(out)) out = '_' + out
  return out || fallback
}

/** Output ports that have no outgoing edges ⇒ return values. */
function findTerminals(nodes, connections) {
  const sourceKeys = new Set(connections.map((c) => `${c.source}/${c.sourceOutput}`))
  const out = []
  for (const n of nodes) {
    for (const port of n.entry.outputs) {
      const key = `${n.id}/${port.name}`
      if (!sourceKeys.has(key)) {
        out.push({ nodeId: n.id, portName: port.name })
      }
    }
  }
  return out
}
