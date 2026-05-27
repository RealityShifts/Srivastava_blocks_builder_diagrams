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

/** Coerce arbitrary user-typed text into a valid Python identifier. */
export function sanitizePyIdent(raw, fallback = 'x') {
  const s = String(raw ?? '').trim()
  if (!s) return fallback
  let out = s.replace(/[^A-Za-z0-9_]/g, '_')
  if (/^[0-9]/.test(out)) out = '_' + out
  return out || fallback
}

/**
 * Shared graph planning used by generate() and the runtime shape runner.
 * Returns naming maps, wiring tables, and topo order.
 */
export function planGraph(nodes, connections) {
  if (nodes.length === 0) return null
  const ordered = topoSort(nodes, connections)

  // Two separate Python namespaces: `self.*` attributes vs. forward()-scope
  // locals. Splitting them lets `self.shared` and local `shared` coexist
  // (Python is fine with this) so weight-shared twins get nice names like
  // `shared = self.shared(x)` / `shared2 = self.shared(shared)`.
  const localPool = new Set()
  const attrPool = new Set()
  const allocLocal = (base) => {
    let c = base
    let i = 2
    while (localPool.has(c)) c = `${base}${i++}`
    localPool.add(c)
    return c
  }
  const allocAttr = (base) => {
    let c = base
    let i = 2
    while (attrPool.has(c)) c = `${base}${i++}`
    attrPool.add(c)
    return c
  }

  const inputArgFor = new Map()
  for (const n of ordered) {
    if (n.entry.kind === 'input') {
      inputArgFor.set(n.id, allocLocal(sanitizePyIdent(n.values?.name, 'x')))
    }
  }

  // Module-kind nodes sharing a (sanitized, non-empty) tag get the *same*
  // attribute - that's how weight tying works: one `self.<attr> = Block(...)`
  // in __init__, multiple call sites in forward. Non-module kinds (input,
  // rearrange, reshape) never share.
  const attrName = new Map()
  const sharedKeyToAttr = new Map() // tag -> attr name
  ordered.forEach((n, i) => {
    if (n.entry.kind === 'input') return
    const tag =
      n.entry.kind === 'module' ? sanitizePyIdent(n.tag ?? '', '') : ''
    if (tag) {
      if (!sharedKeyToAttr.has(tag)) {
        sharedKeyToAttr.set(tag, allocAttr(tag))
      }
      attrName.set(n.id, sharedKeyToAttr.get(tag))
      return
    }
    attrName.set(n.id, allocAttr(snake(n.entry.name, i)))
  })

  // Local variable per call site - always unique within the forward scope.
  // For non-shared nodes this matches the attr name exactly; for shared-
  // weight twins each call site gets a bumped suffix so the first call's
  // output isn't clobbered by the second.
  const localName = new Map()
  ordered.forEach((n) => {
    if (n.entry.kind === 'input') return
    localName.set(n.id, allocLocal(attrName.get(n.id)))
  })

  const imports = new Map()
  for (const n of ordered) {
    if (n.entry.kind === 'input') continue
    if (n.entry.module?.startsWith('__')) continue // synthetic nodes wire imports separately
    if (!imports.has(n.entry.module)) imports.set(n.entry.module, new Set())
    imports.get(n.entry.module).add(n.entry.name)
  }

  const incoming = new Map()
  for (const c of connections) {
    const key = `${c.target}/${c.targetInput}`
    if (!incoming.has(key)) incoming.set(key, [])
    incoming.get(key).push(c)
  }

  const outputVarFor = new Map()
  for (const n of ordered) {
    if (n.entry.kind === 'input') {
      outputVarFor.set(`${n.id}/out`, inputArgFor.get(n.id))
      continue
    }
    const multi = n.entry.outputs.length > 1
    for (const port of n.entry.outputs) {
      outputVarFor.set(
        `${n.id}/${port.name}`,
        multi ? `${localName.get(n.id)}_${port.name}` : localName.get(n.id)
      )
    }
  }

  const entryInputs = findEntryInputs(ordered, connections, localPool)
  return {
    ordered,
    inputArgFor,
    attrName,
    localName,
    imports,
    incoming,
    outputVarFor,
    entryInputs,
  }
}

/**
 * @param {Array} nodes      array of BlockNode
 * @param {Array} connections array of {source, sourceOutput, target, targetInput}
 * @param {'pytorch'|'flax'} framework
 * @param {{ trace?: boolean }} options  trace=true records tensor.shape per port
 */
export function generate(nodes, connections, framework, options = {}) {
  const trace = options.trace === true
  const plan = planGraph(nodes, connections)
  if (!plan) return '# empty graph\n'

  const {
    ordered,
    inputArgFor,
    attrName,
    localName,
    imports,
    incoming,
    outputVarFor,
    entryInputs,
  } = plan

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
  if (ordered.some((n) => n.entry.kind === 'rearrange')) {
    lines.push('from einops import rearrange')
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
  // Shared-weight nodes (same tag) collapse into one __init__ slot. Iterate
  // over module nodes and emit one assignment per unique attr name.
  const emittedAttrs = new Set()
  for (const n of moduleNodes) {
    const attr = attrName.get(n.id)
    if (emittedAttrs.has(attr)) continue
    emittedAttrs.add(attr)
    const args = ctorArgs(n)
    if (framework === 'flax') args.push('rngs=rngs')
    lines.push(`        self.${attr} = ${n.entry.name}(${args.join(', ')})`)
  }
  lines.push('')

  // ------- forward / __call__ -------
  // Explicit Input nodes (in topo order) come first; dangling required inputs
  // on regular nodes still get auto-promoted to forward() args as a fallback.
  const inputArgs = ordered
    .filter((n) => n.entry.kind === 'input')
    .map((n) => inputArgFor.get(n.id))
  const allArgs = [...inputArgs, ...entryInputs.map((e) => e.portArg)]
  const fwdName = framework === 'pytorch' ? 'forward' : '__call__'
  lines.push(`    def ${fwdName}(self${allArgs.length ? ', ' + allArgs.join(', ') : ''}):`)

  if (trace) lines.push('        _runtime_shapes = {}')

  const argFor = new Map()
  for (const e of entryInputs) argFor.set(`${e.nodeId}/${e.portName}`, e.argName)

  for (const n of ordered) {
    if (n.entry.kind === 'input') {
      if (trace) {
        const arg = inputArgFor.get(n.id)
        lines.push(
          `        _runtime_shapes[${JSON.stringify(`${n.id}/out`)}] = list(${arg}.shape)`
        )
      }
      continue
    }
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

    const callExpr = buildCallExpr(n, callArgs, attrName, framework)

    const multi = n.entry.outputs.length > 1
    if (trace) {
      lines.push('        try:')
      if (multi) {
        const targets = n.entry.outputs
          .map((p) => outputVarFor.get(`${n.id}/${p.name}`))
          .join(', ')
        lines.push(`            ${targets} = ${callExpr}`)
        for (const port of n.entry.outputs) {
          const v = outputVarFor.get(`${n.id}/${port.name}`)
          lines.push(
            `            _runtime_shapes[${JSON.stringify(`${n.id}/${port.name}`)}] = list(${v}.shape)`
          )
        }
      } else {
        const v = localName.get(n.id)
        const portName = n.entry.outputs[0]?.name ?? 'out'
        lines.push(`            ${v} = ${callExpr}`)
        lines.push(
          `            _runtime_shapes[${JSON.stringify(`${n.id}/${portName}`)}] = list(${v}.shape)`
        )
      }
      lines.push('        except Exception as _e:')
      lines.push(
        `            raise RuntimeError(${JSON.stringify(`NODE_ERROR::${n.id}::${n.entry.name}`)} + ': ' + str(_e)) from _e`
      )
    } else if (multi) {
      const targets = n.entry.outputs
        .map((p) => outputVarFor.get(`${n.id}/${p.name}`))
        .join(', ')
      lines.push(`        ${targets} = ${callExpr}`)
    } else {
      const v = localName.get(n.id)
      lines.push(`        ${v} = ${callExpr}`)
    }
  }

  const terminals = findTerminals(ordered, connections)
  if (trace) {
    lines.push('        return _runtime_shapes')
  } else if (terminals.length === 0) {
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

/**
 * Decide what `<var> = ...` expression to emit for the forward pass.
 * Built-in shape ops (rearrange, reshape) use bespoke call shapes;
 * everything else follows the standard module/function path.
 */
function buildCallExpr(node, callArgs, attrName, framework) {
  if (node.entry.kind === 'rearrange') {
    const xVar = positionalSource(callArgs, 'x')
    const pattern = JSON.stringify(String(node.values?.pattern ?? ''))
    const lenKwargs = parseLengthsKwargs(node.values?.lengths)
    const tail = lenKwargs ? `, ${lenKwargs}` : ''
    return `rearrange(${xVar}, ${pattern}${tail})`
  }
  if (node.entry.kind === 'reshape') {
    const xVar = positionalSource(callArgs, 'x')
    const dims = parseReshapeDims(node.values?.shape)
    if (framework === 'pytorch') return `${xVar}.reshape(${dims.join(', ')})`
    return `jnp.reshape(${xVar}, (${dims.join(', ')}${dims.length === 1 ? ',' : ''}))`
  }
  if (node.entry.kind === 'concat') {
    const tensors = variadicList(callArgs, 'xs')
    const dim = parseAxisDim(node.values?.dim, 1)
    if (framework === 'pytorch') return `torch.cat(${tensors}, dim=${dim})`
    return `jnp.concatenate(${tensors}, axis=${dim})`
  }
  if (node.entry.kind === 'stack') {
    const tensors = variadicList(callArgs, 'xs')
    const dim = parseAxisDim(node.values?.dim, 0)
    if (framework === 'pytorch') return `torch.stack(${tensors}, dim=${dim})`
    return `jnp.stack(${tensors}, axis=${dim})`
  }
  return node.entry.kind === 'module'
    ? `self.${attrName.get(node.id)}(${callArgs.join(', ')})`
    : `${node.entry.name}(${callArgs.join(', ')})`
}

function positionalSource(callArgs, portName) {
  const prefix = `${portName}=`
  const found = callArgs.find((a) => a.startsWith(prefix))
  if (!found) return 'None'  // dangling input; runtime will fail loudly
  return found.slice(prefix.length)
}

/** Pull `[a, b, c]` from `xs=[a, b, c]` in callArgs; default `[]`. */
function variadicList(callArgs, portName) {
  const prefix = `${portName}=`
  const found = callArgs.find((a) => a.startsWith(prefix))
  if (!found) return '[]'
  return found.slice(prefix.length)
}

function parseAxisDim(v, fallback = 0) {
  if (v === null || v === undefined || v === '') return fallback
  const n = Number(v)
  return Number.isFinite(n) ? Math.trunc(n) : fallback
}

/** "h=8, w=16; b=2" -> "h=8, w=16, b=2" (only valid name=int kwargs). */
function parseLengthsKwargs(s) {
  if (!s) return ''
  const out = []
  for (const part of String(s).split(/[,;\n]/)) {
    const m = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+)\s*$/.exec(part)
    if (m) out.push(`${m[1]}=${m[2]}`)
  }
  return out.join(', ')
}

/** Space/comma-separated int (or -1) tokens. Defaults to "-1" if empty. */
function parseReshapeDims(s) {
  const toks = String(s ?? '')
    .trim()
    .split(/[\s,]+/)
    .filter(Boolean)
  const dims = []
  for (const t of toks) {
    if (/^-?\d+$/.test(t)) dims.push(Number(t))
  }
  return dims.length ? dims : [-1]
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
