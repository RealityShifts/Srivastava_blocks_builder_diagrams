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
 * @param {Array} nodes      array of BlockNode (top-level + grouped + facades)
 * @param {Array} connections array of {source, sourceOutput, target, targetInput}
 * @param {'pytorch'|'flax'} framework
 * @param {{ trace?: boolean }} options  trace=true records tensor.shape per port
 *
 * If the graph contains group facades (kind === 'group'), every group is
 * emitted as its own nn.Module / nnx.Module subclass, and the main class
 * instantiates the subclasses at the facade locations.
 */
export function generate(nodes, connections, framework, options = {}) {
  const partition = partitionByGroup(nodes, connections)
  const { facadesByGid, childrenByGid, internalByGid } = partition

  // Assign Python class names to groups up front so both the main class and
  // any cross-references (e.g. nested calls) agree.
  const classNames = new Map()
  for (const [gid, facade] of facadesByGid) {
    classNames.set(gid, groupClassName(facade.entry.name, gid))
  }

  // Top-level view excludes group children entirely and treats facades as
  // ordinary module-kind nodes (their class is defined below in the same file).
  const topNodes = nodes.filter((n) => !n.groupId)
  const topConnections = connections.filter((c) => {
    const src = nodes.find((n) => n.id === c.source)
    const tgt = nodes.find((n) => n.id === c.target)
    return !src?.groupId && !tgt?.groupId
  })

  // Imports are emitted once across all classes, then we emit subclass bodies
  // followed by the main class.
  const lines = ['from __future__ import annotations', '']
  emitFrameworkImports(lines, framework)
  emitUserImports(lines, nodes)
  if (nodes.some((n) => n.entry.kind === 'rearrange')) {
    lines.push('from einops import rearrange')
  }
  lines.push('')

  const subClassSections = []
  for (const [gid, facade] of facadesByGid) {
    const children = childrenByGid.get(gid) || []
    const internals = internalByGid.get(gid) || []
    const view = buildSubgraphView(facade, children, internals)
    const subLines = []
    emitClassBody(
      subLines,
      view.nodes,
      view.connections,
      framework,
      classNames.get(gid),
      classNames,
      options
    )
    subClassSections.push(subLines.join('\n'))
  }
  if (subClassSections.length > 0) {
    lines.push(subClassSections.join('\n\n'))
    lines.push('')
  }

  // Main class. If there's truly nothing top-level (e.g. user grouped the
  // entire graph), still emit a hollow main class so the file is valid.
  if (topNodes.length === 0) return lines.join('\n') + '# empty graph\n'
  emitClassBody(
    lines,
    topNodes,
    topConnections,
    framework,
    'GeneratedModel',
    classNames,
    options
  )
  return lines.join('\n')
}

function emitFrameworkImports(lines, framework) {
  if (framework === 'pytorch') {
    lines.push('import torch')
    lines.push('import torch.nn as nn')
  } else {
    lines.push('import jax')
    lines.push('import jax.numpy as jnp')
    lines.push('import flax.nnx as nnx')
  }
}

function emitUserImports(lines, allNodes) {
  const imports = new Map()
  for (const n of allNodes) {
    if (n.entry.kind === 'input') continue
    if (n.entry.kind === 'group') continue // facades are defined locally
    if (n.entry.module?.startsWith('__')) continue
    if (!imports.has(n.entry.module)) imports.set(n.entry.module, new Set())
    imports.get(n.entry.module).add(n.entry.name)
  }
  for (const [mod, names] of imports) {
    lines.push(`from ${mod} import ${[...names].sort().join(', ')}`)
  }
}

function emitClassBody(lines, nodes, connections, framework, className, classNames, options) {
  // Subclasses (anything other than the main entry point) MUST always return
  // real tensors so the main class's wiring keeps working - even in trace
  // mode. Without this, the subclass would emit `return _runtime_shapes` and
  // the caller would receive a dict, then `hasattr(dict, 'shape')` is false
  // and every downstream port comes back with an empty shape list. Trace
  // recording is the main class's job only; expand a group if you want to
  // see internal child shapes for that subgraph.
  const isSubclass = className !== 'GeneratedModel'
  const trace = options.trace === true && !isSubclass
  const plan = planGraph(nodes, connections)
  if (!plan) {
    lines.push(`# empty graph for ${className}`)
    return
  }

  const {
    ordered,
    inputArgFor,
    attrName,
    localName,
    incoming,
    outputVarFor,
  } = plan

  const baseClass = framework === 'pytorch' ? 'nn.Module' : 'nnx.Module'
  lines.push(`class ${className}(${baseClass}):`)

  if (framework === 'pytorch') {
    lines.push('    def __init__(self):')
    lines.push('        super().__init__()')
  } else {
    lines.push('    def __init__(self, *, rngs: nnx.Rngs):')
  }

  const moduleNodes = ordered.filter(
    (n) => n.entry.kind === 'module' || n.entry.kind === 'group'
  )
  if (moduleNodes.length === 0) lines.push('        pass')
  const emittedAttrs = new Set()
  for (const n of moduleNodes) {
    const attr = attrName.get(n.id)
    if (emittedAttrs.has(attr)) continue
    emittedAttrs.add(attr)
    if (n.entry.kind === 'group') {
      const sub = classNames.get(n.entry.groupId) || groupClassName(n.entry.name, n.entry.groupId)
      const args = framework === 'flax' ? ['rngs=rngs'] : []
      lines.push(`        self.${attr} = ${sub}(${args.join(', ')})`)
    } else {
      const args = ctorArgs(n)
      if (framework === 'flax') args.push('rngs=rngs')
      lines.push(`        self.${attr} = ${n.entry.name}(${args.join(', ')})`)
    }
  }
  lines.push('')

  // ------- forward / __call__ -------
  // Only explicit Input nodes become forward() arguments.
  const inputArgs = ordered
    .filter((n) => n.entry.kind === 'input')
    .map((n) => inputArgFor.get(n.id))
  const allArgs = [...inputArgs]
  const fwdName = framework === 'pytorch' ? 'forward' : '__call__'
  lines.push(`    def ${fwdName}(self${allArgs.length ? ', ' + allArgs.join(', ') : ''}):`)

  if (trace) lines.push('        _runtime_shapes = {}')

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
    if (n.entry.kind === 'output') {
      if (trace) {
        const key = `${n.id}/x`
        const incomingEdges = incoming.get(key) ?? []
        if (incomingEdges.length > 0) {
          const c = incomingEdges[0]
          const srcVar = outputVarFor.get(`${c.source}/${c.sourceOutput}`)
          if (srcVar) {
            lines.push(
              `        _runtime_shapes[${JSON.stringify(`${n.id}/x`)}] = (list(${srcVar}.shape) if hasattr(${srcVar}, 'shape') else [])`
            )
          }
        }
      }
      continue
    }
    // Build the argument expression for each input port of this node.
    const callArgs = []
    for (const port of n.entry.inputs) {
      const key = `${n.id}/${port.name}`
      const incomingEdges = incoming.get(key) ?? []

      if (incomingEdges.length === 0) {
        if (port.optional) {
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
            `            _runtime_shapes[${JSON.stringify(`${n.id}/${port.name}`)}] = (list(${v}.shape) if hasattr(${v}, 'shape') else [])`
          )
        }
      } else {
        const v = localName.get(n.id)
        const portName = n.entry.outputs[0]?.name ?? 'out'
        lines.push(`            ${v} = ${callExpr}`)
        lines.push(
          `            _runtime_shapes[${JSON.stringify(`${n.id}/${portName}`)}] = (list(${v}.shape) if hasattr(${v}, 'shape') else [])`
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

  // Sub-classes have their boundary outputs represented as virtual Output
  // nodes named `out0`, `out1`, ...; the return tuple must match that order
  // because the caller in the main class unpacks positionally. Plain topo
  // order can interleave them, so we sort by the trailing index for subclasses.
  const explicitOutputs = isSubclass
    ? findExplicitOutputsOrdered(ordered, incoming, outputVarFor)
    : findExplicitOutputs(ordered, incoming, outputVarFor)
  const terminals = findTerminals(ordered, connections)
  if (trace) {
    lines.push('        return _runtime_shapes')
  } else if (explicitOutputs.length > 0) {
    if (explicitOutputs.length === 1) {
      lines.push(`        return ${explicitOutputs[0]}`)
    } else {
      lines.push(`        return (${explicitOutputs.join(', ')})`)
    }
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
}

/**
 * Decide what `<var> = ...` expression to emit for the forward pass.
 * Built-in shape ops (rearrange, reshape) use bespoke call shapes;
 * everything else follows the standard module/function path.
 */
function buildCallExpr(node, callArgs, attrName, framework) {
  if (node.entry.kind === 'const') {
    return constLiteral(node)
  }
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
  if (node.entry.kind === 'module' || node.entry.kind === 'group') {
    return `self.${attrName.get(node.id)}(${callArgs.join(', ')})`
  }
  return `${node.entry.name}(${callArgs.join(', ')})`
}

function constLiteral(node) {
  const type = String(node.values?.value_type ?? 'int')
  const raw = node.values?.value
  if (type === 'bool') {
    if (raw === true || raw === 'true' || raw === '1' || raw === 1) return 'True'
    if (raw === false || raw === 'false' || raw === '0' || raw === 0) return 'False'
    return 'False'
  }
  if (type === 'str') return pyRepr(String(raw ?? ''))
  const n = Number(raw)
  if (!Number.isFinite(n)) return '0'
  return type === 'int' ? String(Math.trunc(n)) : String(n)
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
    if (n.entry.kind === 'input' || n.entry.kind === 'output') continue
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

/** Return vars wired into explicit Output nodes (topo order). */
function findExplicitOutputs(ordered, incoming, outputVarFor) {
  const out = []
  for (const n of ordered) {
    if (n.entry.kind !== 'output') continue
    const key = `${n.id}/x`
    const edges = incoming.get(key) ?? []
    if (edges.length === 0) continue
    const c = edges[0]
    const v = outputVarFor.get(`${c.source}/${c.sourceOutput}`)
    if (v) out.push(v)
  }
  return out
}

/**
 * Like findExplicitOutputs, but sorts the Output nodes by the trailing
 * numeric index in `values.name` (e.g. "out0" < "out1" < "out10"). Used for
 * sub-class returns so the tuple order matches the facade's portMap order
 * regardless of topo sort.
 */
function findExplicitOutputsOrdered(ordered, incoming, outputVarFor) {
  const outNodes = ordered.filter((n) => n.entry.kind === 'output')
  outNodes.sort((a, b) => {
    const ai = parseInt(String(a.values?.name ?? '').replace(/^\D+/, ''), 10)
    const bi = parseInt(String(b.values?.name ?? '').replace(/^\D+/, ''), 10)
    const ax = Number.isFinite(ai) ? ai : 0
    const bx = Number.isFinite(bi) ? bi : 0
    return ax - bx
  })
  const out = []
  for (const n of outNodes) {
    const key = `${n.id}/x`
    const edges = incoming.get(key) ?? []
    if (edges.length === 0) continue
    const c = edges[0]
    const v = outputVarFor.get(`${c.source}/${c.sourceOutput}`)
    if (v) out.push(v)
  }
  return out
}

// ---------------------------------------------------------------------------
// Subgraph helpers - turn a (facade, children, internal edges) triple into a
// self-contained node + connection list that planGraph/emitClassBody can
// chew on like any other graph. Virtual Input/Output nodes stand in for the
// facade's boundary ports so the subclass gets a clean forward(self, in0, ...)
// signature.
// ---------------------------------------------------------------------------

function partitionByGroup(nodes, connections) {
  const facadesByGid = new Map()
  for (const n of nodes) {
    if (n.entry?.kind === 'group' && n.entry.groupId) {
      facadesByGid.set(n.entry.groupId, n)
    }
  }
  const childrenByGid = new Map()
  const internalByGid = new Map()
  for (const gid of facadesByGid.keys()) {
    childrenByGid.set(gid, [])
    internalByGid.set(gid, [])
  }
  const byId = new Map(nodes.map((n) => [n.id, n]))
  for (const n of nodes) {
    if (n.entry?.kind === 'group') continue
    if (n.groupId && childrenByGid.has(n.groupId)) {
      childrenByGid.get(n.groupId).push(n)
    }
  }
  for (const c of connections) {
    const src = byId.get(c.source)
    const tgt = byId.get(c.target)
    if (src?.groupId && src.groupId === tgt?.groupId && internalByGid.has(src.groupId)) {
      internalByGid.get(src.groupId).push(c)
    }
  }
  return { facadesByGid, childrenByGid, internalByGid }
}

function groupClassName(facadeName, gid) {
  const sane = sanitizePyIdent(facadeName || 'Group', 'Group')
  const pascal = sane.charAt(0).toUpperCase() + sane.slice(1)
  // A short suffix from the gid disambiguates two groups that the user happens
  // to name identically; without it Python would refuse the duplicate class.
  const tag = String(gid || '').replace(/[^A-Za-z0-9]/g, '').slice(-4)
  return tag ? `${pascal}_${tag}` : pascal
}

/**
 * Construct a virtual graph that represents one group as a standalone class:
 * synthetic Input nodes feed each boundary input, synthetic Output nodes
 * collect each boundary output, and the internal edges are the rest of the
 * subgraph's structure.
 */
function buildSubgraphView(facade, children, internalConnections) {
  const inputs = facade.entry.portMap?.inputs || []
  const outputs = facade.entry.portMap?.outputs || []

  const makeVirtual = (id, entry, values) => ({
    id,
    entry,
    label: entry.name,
    tag: '',
    groupId: null,
    values,
    inputs: Object.fromEntries(
      (entry.inputs || []).map((p) => [p.name, { portSpec: p }])
    ),
    outputs: Object.fromEntries(
      (entry.outputs || []).map((p) => [p.name, { portSpec: p }])
    ),
    freshenedShape() { return null },
    applyParamBindings() {},
  })

  const virtualInputs = inputs.map((m, i) =>
    makeVirtual(
      `__sg_in_${facade.id}_${i}`,
      {
        kind: 'input',
        name: 'Input',
        module: '__builtin__',
        ctor: [],
        inputs: [],
        outputs: [
          { name: 'out', shape: m.shape ?? ['...'], dtype: 'any', optional: false, variadic: false },
        ],
        bindings: {},
      },
      { name: `in${i}`, shape: '', dtype: 'any' }
    )
  )
  const virtualOutputs = outputs.map((m, i) =>
    makeVirtual(
      `__sg_out_${facade.id}_${i}`,
      {
        kind: 'output',
        name: 'Output',
        module: '__builtin__',
        ctor: [],
        inputs: [
          { name: 'x', shape: ['...'], dtype: 'any', optional: false, variadic: false },
        ],
        outputs: [],
        bindings: {},
      },
      { name: `out${i}` }
    )
  )

  const inputEdges = inputs.map((m, i) => ({
    source: `__sg_in_${facade.id}_${i}`,
    sourceOutput: 'out',
    target: m.childNodeId,
    targetInput: m.childPort,
  }))
  const outputEdges = outputs.map((m, i) => ({
    source: m.childNodeId,
    sourceOutput: m.childPort,
    target: `__sg_out_${facade.id}_${i}`,
    targetInput: 'x',
  }))

  return {
    nodes: [...virtualInputs, ...children, ...virtualOutputs],
    connections: [...inputEdges, ...internalConnections, ...outputEdges],
  }
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
