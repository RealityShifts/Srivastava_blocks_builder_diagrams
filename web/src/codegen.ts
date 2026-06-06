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

import type { NodeLike, Connection, ManifestEntry } from './types.ts'
import type { Substitution } from './shape.ts'
import { resolve } from './shape.ts'
import { unifyShape } from './unify.ts'

/**
 * Replay the validator's shape unification over the whole graph so codegen can
 * read the *resolved* value of a ctor-bound axis (e.g. MultiHeadAttention's
 * `D_val` -> `vdim`). Ctor params the user left blank but which are bound to an
 * axis can then be auto-filled from the dim that actually flows into the port -
 * otherwise the param would silently fall back to its default and the emitted
 * model would mis-size that layer.
 */
function buildAxisSub(nodes: NodeLike[], connections: Connection[]): Substitution {
  const sub: Substitution = new Map()
  const byId = new Map<string, NodeLike>(nodes.map((n) => [n.id, n]))
  for (const n of nodes) {
    const fn = (n as any).applyParamBindings
    if (typeof fn === 'function') fn.call(n, sub)
  }
  for (const c of connections) {
    const src = byId.get(c.source) as any
    const tgt = byId.get(c.target) as any
    if (!src || !tgt) continue
    if (typeof src.freshenedShape !== 'function' || typeof tgt.freshenedShape !== 'function') continue
    const out = src.freshenedShape(c.sourceOutput, 'out')
    const inp = tgt.freshenedShape(c.targetInput, 'in')
    if (!out || !inp) continue
    try {
      unifyShape(out, inp, sub)
    } catch {
      // A genuine shape conflict surfaces in the validator; codegen just skips
      // auto-fill for that edge rather than aborting the whole export.
    }
  }
  return sub
}

/** The naming maps, wiring tables, and topo order returned by {@link planGraph}. */
export interface PlanGraph {
  /** Nodes in topological order. */
  ordered: NodeLike[]
  /** Input-node id -> forward() argument name. */
  inputArgFor: Map<string, string>
  /** Output-node id -> return variable name. */
  outputReturnFor: Map<string, string>
  /** Node id -> `self.<attr>` attribute name. */
  attrName: Map<string, string>
  /** Node id -> forward()-scope local variable name. */
  localName: Map<string, string>
  /** Python module -> set of referenced block names to import. */
  imports: Map<string, Set<string>>
  /** `<targetId>/<targetInput>` -> incoming connection(s). */
  incoming: Map<string, Connection[]>
  /** `<nodeId>/<portName>` -> the variable holding that port's value. */
  outputVarFor: Map<string, string>
  /** Unwired required input ports surfaced as forward() args. */
  entryInputs: any[]
}

/** Options accepted by {@link generate}. */
export interface GenerateOptions {
  /** trace=true records tensor.shape per port and forward() returns the dict. */
  trace?: boolean
  /** When provided, appends a `test_GeneratedModel()` function plus an
   *  `if __name__ == "__main__":` entrypoint built from these input specs. */
  testCase?: Array<{ arg: string; shape: number[]; dtype?: string }>
}

function topoSort(nodes: NodeLike[], connections: Connection[]): NodeLike[] {
  const indeg = new Map<string, number>()
  const out = new Map<string, string[]>()
  for (const n of nodes) {
    indeg.set(n.id, 0)
    out.set(n.id, [])
  }
  for (const c of connections) {
    if (!out.has(c.source) || !indeg.has(c.target)) continue
    indeg.set(c.target, (indeg.get(c.target) as number) + 1)
    ;(out.get(c.source) as string[]).push(c.target)
  }
  const queue = [...indeg.entries()].filter(([, d]) => d === 0).map(([k]) => k)
  const order: string[] = []
  while (queue.length) {
    const id = queue.shift() as string
    order.push(id)
    for (const tgt of out.get(id) ?? []) {
      indeg.set(tgt, (indeg.get(tgt) as number) - 1)
      if (indeg.get(tgt) === 0) queue.push(tgt)
    }
  }
  if (order.length !== nodes.length) {
    throw new Error('graph contains a cycle')
  }
  return order.map((id) => nodes.find((n) => n.id === id) as NodeLike)
}

function pyRepr(value: any): string {
  if (value === null || value === undefined) return 'None'
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (typeof value === 'number') return String(value)
  if (typeof value === 'string') return JSON.stringify(value)
  if (Array.isArray(value)) return `[${value.map(pyRepr).join(', ')}]`
  return JSON.stringify(value)
}

function snake(name: string, suffix: any): string {
  const base = name
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1_$2')
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .toLowerCase()
  return `${base}_${suffix}`
}

/** Coerce arbitrary user-typed text into a valid Python identifier. */
export function sanitizePyIdent(raw: unknown, fallback: string = 'x'): string {
  const s = String(raw ?? '').trim()
  if (!s) return fallback
  let out = s.replace(/[^A-Za-z0-9_]/g, '_')
  if (/^[0-9]/.test(out)) out = '_' + out
  return out || fallback
}

/**
 * Python forward() argument for an Input node. A custom `name` (when not the
 * default `x`) wins; otherwise a non-empty tag is used; else `x`.
 */
export function inputForwardArgName(node: NodeLike): string {
  const rawName = String(node.values?.name ?? '').trim()
  if (rawName && rawName !== 'x') return sanitizePyIdent(rawName, 'x')
  const tag = sanitizePyIdent(node.tag ?? '', '')
  if (tag) return tag
  return sanitizePyIdent(rawName, 'x')
}

/**
 * Return variable for an explicit Output node. Custom `name` (when not the
 * default `y`) wins; otherwise a non-empty tag is used; else `y`.
 */
export function outputReturnArgName(node: NodeLike): string {
  const rawName = String(node.values?.name ?? '').trim()
  if (rawName && rawName !== 'y') return sanitizePyIdent(rawName, 'y')
  const tag = sanitizePyIdent(node.tag ?? '', '')
  if (tag) return tag
  return sanitizePyIdent(rawName, 'y')
}

// ---------------------------------------------------------------------------
// jaxtyping annotations
//
// jaxtyping wants `Float[Tensor, "B C H W"]`-style annotations. We synthesize
// them from each node's declared shape + dtype so the generated module can be
// dropped into another codebase as a real, typed custom block.
//
// Notes:
//   - Axis names with a "#<nodeId>" freshen suffix are stripped (jaxtyping
//     reserves `#` for broadcasting axes, so we'd otherwise misuse it).
//   - Axis tokens that aren't a bare identifier or an integer literal are
//     replaced with `_` (jaxtyping's "any size, no checking" axis).
//   - Empty / missing shape -> `"..."` (any rank, any sizes).
//   - dtype 'any' / unknown -> `Shaped` (matches any numeric dtype).
// ---------------------------------------------------------------------------
const JAXTYPING_BY_DTYPE: Record<string, string> = {
  float: 'Float',
  float16: 'Float',
  float32: 'Float',
  float64: 'Float',
  bfloat16: 'Float',
  int: 'Int',
  int8: 'Int',
  int16: 'Int',
  int32: 'Int',
  int64: 'Int',
  long: 'Int',
  uint: 'UInt',
  uint8: 'UInt',
  bool: 'Bool',
  complex: 'Complex',
  any: 'Shaped',
  '': 'Shaped',
}

function jaxtypingDtype(dtype: any): string {
  const key = String(dtype ?? '').toLowerCase().trim()
  return JAXTYPING_BY_DTYPE[key] || 'Shaped'
}

function sanitizeAxis(tok: any): string {
  const s = String(tok ?? '').trim()
  if (!s) return '_'
  if (s === '...') return '...'
  // Strip freshen suffix appended by shape.js (`C_in#node-7` -> `C_in`).
  const base = s.split('#')[0]
  if (/^-?\d+$/.test(base)) return base
  if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(base)) return base
  return '_'
}

/** Normalize a shape (array of tokens OR space-separated string) into the
 *  axis string jaxtyping expects, e.g. ["B","C","H","W"] -> "B C H W". */
function jaxtypingShape(shape: any): string {
  let toks: any[] = []
  if (Array.isArray(shape)) toks = shape
  else if (typeof shape === 'string') toks = shape.split(/[\s,]+/)
  toks = toks.map((t) => String(t ?? '').trim()).filter((t) => t.length > 0)
  if (toks.length === 0) return '...'
  toks = toks.map(sanitizeAxis)
  // jaxtyping permits at most one `...`; collapse any duplicates.
  let seenEllipsis = false
  toks = toks.filter((t) => {
    if (t !== '...') return true
    if (seenEllipsis) return false
    seenEllipsis = true
    return true
  })
  return toks.join(' ')
}

/** `Float[Tensor, "B C H W"]` style annotation. */
function jaxtypingAnno(shape: any, dtype: any, tensorType: string, usedDtypes?: Set<string>): string {
  const cls = jaxtypingDtype(dtype)
  usedDtypes?.add(cls)
  return `${cls}[${tensorType}, "${jaxtypingShape(shape)}"]`
}

/**
 * Shared graph planning used by generate() and the runtime shape runner.
 * Returns naming maps, wiring tables, and topo order.
 */
export function planGraph(nodes: NodeLike[], connections: Connection[]): PlanGraph | null {
  if (nodes.length === 0) return null
  const nodeIds = new Set(nodes.map((n) => n.id))
  const wired = connections.filter((c) => nodeIds.has(c.source) && nodeIds.has(c.target))
  const ordered = topoSort(nodes, wired)

  // Two separate Python namespaces: `self.*` attributes vs. forward()-scope
  // locals. Splitting them lets `self.shared` and local `shared` coexist
  // (Python is fine with this) so weight-shared twins get nice names like
  // `shared = self.shared(x)` / `shared2 = self.shared(shared)`.
  const localPool = new Set<string>()
  const attrPool = new Set<string>()
  const allocLocal = (base: string) => {
    let c = base
    let i = 2
    while (localPool.has(c)) c = `${base}${i++}`
    localPool.add(c)
    return c
  }
  const allocAttr = (base: string) => {
    let c = base
    let i = 2
    while (attrPool.has(c)) c = `${base}${i++}`
    attrPool.add(c)
    return c
  }

  const inputArgFor = new Map<string, string>()
  for (const n of ordered) {
    if (n.entry.kind === 'input') {
      inputArgFor.set(n.id, allocLocal(inputForwardArgName(n)))
    }
  }

  const outputReturnFor = new Map<string, string>()
  for (const n of ordered) {
    if (n.entry.kind === 'output') {
      outputReturnFor.set(n.id, allocLocal(outputReturnArgName(n)))
    }
  }

  // Attribute naming: the editable per-node name drives `self.<name>`. When no
  // name is set we fall back to the weight-sharing tag (so a tagged-but-unnamed
  // node still reads self.<tag>, and group facades - which have no per-instance
  // name - name their slot after the group tag), and finally to the snake-cased
  // block type. Module- and group-kind nodes sharing a (sanitized, non-empty)
  // tag get the *same* attribute - that's how weight tying works: one
  // `self.<attr> = ...` in __init__, multiple call sites in forward. Weight-
  // shared instances are validated to share a name, so the first member names
  // the shared slot. Non-module kinds (input, learnable, ...) never share.
  const attrBaseFor = (n: NodeLike, i: any) => {
    const explicit = String(n.name ?? '').trim()
    if (explicit) return sanitizePyIdent(explicit, snake(n.entry.name, i))
    const tag = sanitizePyIdent(n.tag ?? '', '')
    if (tag) return tag
    return snake(n.entry.name, i)
  }
  const attrName = new Map<string, string>()
  const sharedKeyToAttr = new Map<string, string>() // tag -> attr name
  ordered.forEach((n, i) => {
    if (n.entry.kind === 'input') return
    if (n.entry.kind === 'learnable') {
      attrName.set(n.id, allocAttr(sanitizePyIdent(n.values?.name, 'param')))
      return
    }
    const tag =
      n.entry.kind === 'module' || n.entry.kind === 'group'
        ? sanitizePyIdent(n.tag ?? '', '')
        : ''
    if (tag) {
      if (!sharedKeyToAttr.has(tag)) {
        sharedKeyToAttr.set(tag, allocAttr(attrBaseFor(n, i)))
      }
      attrName.set(n.id, sharedKeyToAttr.get(tag) as string)
      return
    }
    attrName.set(n.id, allocAttr(attrBaseFor(n, i)))
  })

  // Local variable per call site - always unique within the forward scope.
  // For non-shared nodes this matches the attr name exactly; for shared-
  // weight twins each call site gets a bumped suffix so the first call's
  // output isn't clobbered by the second.
  const localName = new Map<string, string>()
  ordered.forEach((n) => {
    if (n.entry.kind === 'input' || n.entry.kind === 'learnable') return
    localName.set(n.id, allocLocal(attrName.get(n.id) as string))
  })

  const imports = new Map<string, Set<string>>()
  for (const n of ordered) {
    if (n.entry.kind === 'input') continue
    if (n.entry.module?.startsWith('__')) continue // synthetic nodes wire imports separately
    if (!imports.has(n.entry.module)) imports.set(n.entry.module, new Set())
    ;(imports.get(n.entry.module) as Set<string>).add(n.entry.name)
  }

  const incoming = new Map<string, Connection[]>()
  for (const c of wired) {
    const key = `${c.target}/${c.targetInput}`
    if (!incoming.has(key)) incoming.set(key, [])
    ;(incoming.get(key) as Connection[]).push(c)
  }

  const outputVarFor = new Map<string, string>()
  for (const n of ordered) {
    if (n.entry.kind === 'input') {
      outputVarFor.set(`${n.id}/out`, inputArgFor.get(n.id) as string)
      continue
    }
    if (n.entry.kind === 'learnable') {
      outputVarFor.set(`${n.id}/out`, `self.${attrName.get(n.id)}`)
      continue
    }
    const multi = n.entry.outputs.length > 1
    for (const port of n.entry.outputs) {
      outputVarFor.set(
        `${n.id}/${port.name}`,
        multi ? `${localName.get(n.id)}_${port.name}` : (localName.get(n.id) as string)
      )
    }
  }

  const entryInputs = findEntryInputs(ordered, wired, localPool)
  return {
    ordered,
    inputArgFor,
    outputReturnFor,
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
 * @param {{ trace?: boolean, testCase?: Array<{arg:string, shape:number[], dtype?:string}> }} options
 *   trace=true records tensor.shape per port and forward() returns the dict.
 *   testCase, when provided, appends a `test_GeneratedModel()` function plus
 *   an `if __name__ == "__main__":` entrypoint that instantiates the model
 *   with dummy tensors built from the supplied input specs.
 *
 * If the graph contains group facades (kind === 'group'), every group is
 * emitted as its own nn.Module / nnx.Module subclass, and the main class
 * instantiates the subclasses at the facade locations.
 */
export function generate(
  nodes: NodeLike[],
  connections: Connection[],
  framework: string,
  options: GenerateOptions = {}
): string {
  const partition = partitionByGroup(nodes, connections)
  const { facadesByGid, childrenByGid, internalByGid } = partition

  // Assign Python class names to groups up front so both the main class and
  // any cross-references (e.g. nested calls) agree.
  const classNames = new Map<string, string>()
  for (const [gid, facade] of facadesByGid) {
    classNames.set(gid, groupClassName(facade.entry.name, gid))
  }

  // A child counts as "grouped" only when its facade is actually present in
  // the graph. When a group is expanded the facade is gone but children keep
  // their groupId; treating those orphans as top-level keeps the generated
  // model intact (otherwise children silently disappear from forward()).
  const isGroupedAway = (n: any) => Boolean(n?.groupId && facadesByGid.has(n.groupId))
  const topNodes = nodes.filter((n) => !isGroupedAway(n))
  const byIdAll = new Map<string, NodeLike>(nodes.map((n) => [n.id, n]))
  // Resolved axis values (shared across all classes) so ctor params bound to an
  // axis can be auto-filled from the dims that actually flow through the graph.
  const axisSub = buildAxisSub(nodes, connections)
  const topConnections = connections.filter((c) => {
    const src = byIdAll.get(c.source)
    const tgt = byIdAll.get(c.target)
    if (!src || !tgt) return false
    return !isGroupedAway(src) && !isGroupedAway(tgt)
  })

  // Imports are emitted once across all classes, then we emit subclass bodies
  // followed by the main class. We don't yet know which jaxtyping dtype
  // classes will be referenced, so we reserve a placeholder line and patch
  // it once every emitClassBody() has filled `usedDtypes`.
  const usedDtypes = new Set<string>()
  const lines: string[] = ['from __future__ import annotations', '']
  emitFrameworkImports(lines, framework)
  const jaxtypingLineIdx = lines.length
  lines.push('') // placeholder for `from jaxtyping import ...`
  emitUserImports(lines, nodes)
  if (nodes.some((n) => n.entry.kind === 'rearrange')) {
    lines.push('from einops import rearrange')
  }
  if (framework === 'flax' && nodes.some((n) => n.entry.kind === 'pool')) {
    lines.push('from flax.linen.pooling import avg_pool, max_pool')
  }
  if (framework === 'flax' && nodes.some((n) => n.entry.kind === 'upsample')) {
    lines.push('import jax.image')
  }
  lines.push('')

  // Emit each unique class name once. When two groups share a name (e.g.
  // because the user duplicated a group), the second one reuses the first's
  // class definition rather than emitting a redefinition. Each facade still
  // gets its own `self.<attr> = ClassName()` instance in the main class.
  //
  // Nested groups mean a class can *contain* another group as a member, so the
  // inner class must be defined ABOVE the outer one - `self.x = Inner()` runs at
  // construction time, and `from __future__ import annotations` only defers type
  // *annotations*, not runtime instantiation. We therefore emit in containment
  // order (deepest first). The dependency graph is expressed in CLASS NAMES, not
  // gids, so two same-name groups (weight-shared / duplicated) collapse to a
  // single class node and a single emission.
  const gidsForClass = new Map<string, string>() // class name -> one representative gid
  const classDeps = new Map<string, Set<string>>() // class -> classes it instantiates
  for (const [gid] of facadesByGid) {
    const cls = classNames.get(gid) as string
    if (!gidsForClass.has(cls)) gidsForClass.set(cls, gid)
    if (!classDeps.has(cls)) classDeps.set(cls, new Set())
    for (const child of childrenByGid.get(gid) || []) {
      if (child.entry?.kind === 'group' && child.entry.groupId) {
        const childCls = classNames.get(child.entry.groupId as string)
        if (childCls && childCls !== cls) (classDeps.get(cls) as Set<string>).add(childCls)
      }
    }
  }
  const orderedClasses = topoSortClasses(classDeps)
  const subClassSections: string[] = []
  for (const cls of orderedClasses) {
    const gid = gidsForClass.get(cls) as string
    const facade = facadesByGid.get(gid) as NodeLike
    const children = childrenByGid.get(gid) || []
    const internals = internalByGid.get(gid) || []
    const view = buildSubgraphView(facade, children, internals)
    const subLines: string[] = []
    emitClassBody(
      subLines,
      view.nodes,
      view.connections,
      framework,
      cls,
      classNames,
      options,
      { facade, usedDtypes, allById: byIdAll, axisSub }
    )
    subClassSections.push(subLines.join('\n'))
  }
  if (subClassSections.length > 0) {
    lines.push(subClassSections.join('\n\n'))
    lines.push('')
  }

  // Main class. If there's truly nothing top-level (e.g. user grouped the
  // entire graph), still emit a hollow main class so the file is valid.
  if (topNodes.length === 0) {
    lines[jaxtypingLineIdx] = renderJaxtypingImport(usedDtypes, framework)
    return lines.join('\n') + '# empty graph\n'
  }
  emitClassBody(
    lines,
    topNodes,
    topConnections,
    framework,
    'GeneratedModel',
    classNames,
    options,
    { usedDtypes, allById: byIdAll, axisSub }
  )
  if (Array.isArray(options.testCase) && options.testCase.length > 0) {
    emitTestCase(lines, framework, 'GeneratedModel', options.testCase, !!options.trace)
  }
  lines[jaxtypingLineIdx] = renderJaxtypingImport(usedDtypes, framework)
  return lines.join('\n')
}

/**
 * Emit `def test_<ClassName>() ...` and an `if __name__ == "__main__":` block
 * that instantiates the model with dummy tensors and prints the output shape.
 */
function emitTestCase(lines: string[], framework: string, className: string, inputs: any[], trace: boolean): void {
  const shapeTuple = (shape: number[]) => {
    const dims = shape.map((d) => (Number.isFinite(d) ? String(Math.trunc(d)) : '1'))
    return `(${dims.join(', ')}${dims.length === 1 ? ',' : ''})`
  }
  const tensorExpr = (shape: number[], dtype: any) => {
    const cat = jaxtypingDtype(dtype)
    const s = shapeTuple(shape)
    if (framework === 'pytorch') {
      if (cat === 'Int' || cat === 'UInt') return `torch.randint(0, 100, ${s})`
      if (cat === 'Bool') return `torch.zeros(${s}, dtype=torch.bool)`
      return `torch.randn(${s})`
    }
    if (cat === 'Int' || cat === 'UInt') return `jnp.zeros(${s}, dtype=jnp.int32)`
    if (cat === 'Bool') return `jnp.zeros(${s}, dtype=jnp.bool_)`
    return `jnp.zeros(${s}, dtype=jnp.float32)`
  }

  lines.push('')
  lines.push(`def test_${className}() -> None:`)
  if (framework === 'pytorch') {
    lines.push(`    model = ${className}()`)
    lines.push('    model.eval()')
  } else {
    lines.push(`    model = ${className}(rngs=nnx.Rngs(0))`)
  }
  for (const spec of inputs) {
    lines.push(`    ${spec.arg} = ${tensorExpr(spec.shape, spec.dtype)}`)
  }
  const callKwargs = inputs.map((s) => `${s.arg}=${s.arg}`).join(', ')
  if (framework === 'pytorch') {
    lines.push('    with torch.no_grad():')
    lines.push(`        out = model(${callKwargs})`)
  } else {
    lines.push(`    out = model(${callKwargs})`)
  }
  if (trace) {
    lines.push('    assert isinstance(out, dict), "trace=True forward must return a dict"')
    lines.push('    print("Runtime shapes:")')
    lines.push('    for k, v in out.items():')
    lines.push('        print(f"  {k}: {v}")')
  } else {
    lines.push('    if isinstance(out, dict):')
    lines.push('        for k, v in out.items():')
    lines.push('            print(f"{k}: {v}")')
    lines.push('    elif isinstance(out, (tuple, list)):')
    lines.push('        for i, t in enumerate(out):')
    lines.push('            print(f"out[{i}]: {tuple(t.shape)}")')
    lines.push('    else:')
    lines.push('        print(f"out: {tuple(out.shape)}")')
  }
  lines.push('')
  lines.push('')
  lines.push('if __name__ == "__main__":')
  lines.push(`    test_${className}()`)
  lines.push('')
}

function renderJaxtypingImport(usedDtypes: Set<string>, framework: string): string {
  // Always import the Tensor/Array symbol so signatures don't depend on whether
  // any input had a recognized dtype.
  const sorted = [...usedDtypes].sort()
  if (sorted.length === 0) sorted.push('Shaped')
  const jaxtypingLine = `from jaxtyping import ${sorted.join(', ')}`
  const tensorLine =
    framework === 'pytorch' ? 'from torch import Tensor' : 'from jax import Array'
  return `${jaxtypingLine}\n${tensorLine}`
}

/** Walk back from a port reference and return its declared shape/dtype.
 *  Returns `null` if we can't resolve (no edge, unknown port). */
function resolveSourceShapeDtype(ref: any, ordered: NodeLike[], incoming: Map<string, Connection[]>): any {
  // ref is { nodeId, portName } - the port itself is a SOURCE (output port).
  const owner = ordered.find((n) => n.id === ref.nodeId)
  if (!owner) return null
  if (owner.entry.kind === 'input' || owner.entry.kind === 'learnable') {
    // Explicit Input / LearnableTensor nodes carry their declared shape on
    // values.{shape,dtype}; fall back to the manifest entry if values are blank.
    return {
      shape: owner.values?.shape ?? owner.entry.outputs?.[0]?.shape ?? ['...'],
      dtype: owner.values?.dtype ?? owner.entry.outputs?.[0]?.dtype ?? 'any',
    }
  }
  const port = owner.entry.outputs?.find((p) => p.name === ref.portName)
  if (!port) return null
  // Group facade ports may have a child-derived dtype in the live editor, but
  // only shape survives autosave (computeBoundary copies dtype from the child
  // but the persisted portMap drops it). Force `any` here so the generated
  // type signature is identical before/after autosave roundtrips.
  const dtype = owner.entry.kind === 'group' ? 'any' : port.dtype ?? 'any'
  return { shape: port.shape ?? ['...'], dtype }
}

/** Resolve the source (shape, dtype) of an Output node's incoming edge. */
function resolveOutputSourceShapeDtype(outNode: NodeLike, ordered: NodeLike[], incoming: Map<string, Connection[]>): any {
  const edges = incoming.get(`${outNode.id}/x`) ?? []
  if (edges.length === 0) return null
  const c = edges[0]
  return resolveSourceShapeDtype(
    { nodeId: c.source, portName: c.sourceOutput },
    ordered,
    incoming
  )
}

function buildReturnAnnotation({ ordered, connections, incoming, isSubclass, facade, tensorType, usedDtypes }: any): string {
  // Sub-class return matches its facade's portMap.outputs (one virtual Output
  // per boundary, ordered out0..outN). Read shapes directly from the facade
  // so we don't depend on whether buildSubgraphView set virtual port shapes.
  let entries: any[] = []
  if (isSubclass && facade) {
    entries = (facade.entry.portMap?.outputs ?? []).map((m: any) => ({
      shape: m.shape ?? ['...'],
      dtype: 'any',
    }))
  } else {
    // Main class: prefer explicit Output nodes; otherwise terminal output
    // ports become the return tuple (same logic the body uses for `return`).
    const outNodes = ordered.filter((n: NodeLike) => n.entry.kind === 'output')
    if (outNodes.length > 0) {
      for (const o of outNodes) {
        const sd = resolveOutputSourceShapeDtype(o, ordered, incoming)
        if (sd) entries.push(sd)
      }
    } else {
      const terminals = findTerminals(ordered, connections)
      for (const t of terminals) {
        const sd = resolveSourceShapeDtype(
          { nodeId: t.nodeId, portName: t.portName },
          ordered,
          incoming
        )
        if (sd) entries.push(sd)
      }
    }
  }
  if (entries.length === 0) return 'None'
  if (entries.length === 1) {
    return jaxtypingAnno(entries[0].shape, entries[0].dtype, tensorType, usedDtypes)
  }
  const annos = entries.map((e) =>
    jaxtypingAnno(e.shape, e.dtype, tensorType, usedDtypes)
  )
  return `tuple[${annos.join(', ')}]`
}

function emitFrameworkImports(lines: string[], framework: string): void {
  if (framework === 'pytorch') {
    lines.push('import torch')
    lines.push('import torch.nn as nn')
  } else {
    lines.push('import jax')
    lines.push('import jax.numpy as jnp')
    lines.push('import flax.nnx as nnx')
  }
}

function emitUserImports(lines: string[], allNodes: NodeLike[]): void {
  const imports = new Map<string, Set<string>>()
  for (const n of allNodes) {
    if (n.entry.kind === 'input') continue
    if (n.entry.kind === 'group') continue // facades are defined locally
    if (n.entry.module?.startsWith('__')) continue
    if (!imports.has(n.entry.module)) imports.set(n.entry.module, new Set())
    ;(imports.get(n.entry.module) as Set<string>).add(n.entry.name)
  }
  for (const [mod, names] of imports) {
    lines.push(`from ${mod} import ${[...names].sort().join(', ')}`)
  }
}

function emitClassBody(
  lines: string[],
  nodes: NodeLike[],
  connections: Connection[],
  framework: string,
  className: string,
  classNames: Map<string, string>,
  options: GenerateOptions,
  typing: any = {}
): void {
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
    outputReturnFor,
    attrName,
    localName,
    incoming,
    outputVarFor,
  } = plan

  const tensorType = framework === 'pytorch' ? 'Tensor' : 'Array'
  const usedDtypes = typing.usedDtypes
  const facade = typing.facade // populated for subclass emission only

  const baseClass = framework === 'pytorch' ? 'nn.Module' : 'nnx.Module'
  lines.push(`class ${className}(${baseClass}):`)

  const { initParams, paramRef } = analyzeConstWiring(nodes, connections, attrName)
  if (facade) mergeGroupFacadeInitParams(facade, ordered, initParams, paramRef, typing.allById)
  // Bubble exposed params of CHILD group facades up into THIS class's __init__,
  // so a group's exposed knob (e.g. Decoder's style_dim) becomes a parameter of
  // the containing class and is threaded down - instead of being frozen at the
  // group's default. Params already driven by a local Constant wire (present in
  // paramRef) are left alone; same-named exposed params share one outer arg.
  bubbleChildGroupParams(ordered, initParams, paramRef, typing.allById)

  const initSigParts: string[] = []
  if (framework === 'flax') initSigParts.push('*, rngs: nnx.Rngs')
  for (const p of initParams) {
    initSigParts.push(`${p.initName}: ${p.pyType} = ${p.defaultLit}`)
  }
  const initSig =
    initSigParts.length > 0
      ? `self, ${initSigParts.join(', ')}`
      : framework === 'flax'
        ? 'self, *, rngs: nnx.Rngs'
        : 'self'

  lines.push(`    def __init__(${initSig}):`)
  if (framework === 'pytorch') {
    lines.push('        super().__init__()')
  }

  const moduleNodes = ordered.filter(
    (n) => n.entry.kind === 'module' || n.entry.kind === 'group'
  )
  const learnableNodes = ordered.filter((n) => n.entry.kind === 'learnable')
  if (moduleNodes.length === 0 && learnableNodes.length === 0) lines.push('        pass')
  const emittedAttrs = new Set<string>()
  for (const n of moduleNodes) {
    const attr = attrName.get(n.id) as string
    if (emittedAttrs.has(attr)) continue
    emittedAttrs.add(attr)
    if (n.entry.kind === 'group') {
      const sub = classNames.get(n.entry.groupId as string) || groupClassName(n.entry.name, n.entry.groupId)
      const args = groupCtorArgs(n, paramRef)
      if (framework === 'flax') args.push('rngs=rngs')
      lines.push(`        self.${attr} = ${sub}(${args.join(', ')})`)
    } else {
      const args = ctorArgs(n, paramRef, typing.axisSub)
      if (framework === 'flax') args.push('rngs=rngs')
      lines.push(`        self.${attr} = ${n.entry.name}(${args.join(', ')})`)
    }
  }
  for (const n of learnableNodes) {
    const attr = attrName.get(n.id) as string
    if (emittedAttrs.has(attr)) continue
    emittedAttrs.add(attr)
    lines.push(`        ${learnableInitLine(n, attr, framework)}`)
  }
  lines.push('')

  // ------- forward / __call__ -------
  // Only explicit Input nodes become forward() arguments.
  const inputNodes = ordered.filter((n) => n.entry.kind === 'input')
  const inputArgs = inputNodes.map((n) => inputArgFor.get(n.id) as string)

  // Resolve a jaxtyping annotation for each forward arg. For the main class
  // we use the user's Input ctor (shape/dtype). For a subclass we read the
  // facade's portMap so the signature reflects what the parent passes in.
  const argAnnotations = inputNodes.map((n, i) => {
    let shape: any
    let dtype: any = 'any'
    if (facade) {
      const m = facade.entry?.portMap?.inputs?.[i]
      if (m) shape = m.shape
    } else {
      shape = n.values?.shape ?? n.entry.outputs?.[0]?.shape
      dtype = n.values?.dtype ?? n.entry.outputs?.[0]?.dtype ?? 'any'
    }
    return jaxtypingAnno(shape, dtype, tensorType, usedDtypes)
  })

  // Resolve return-type annotation. Subclasses always return tensor(s) (we
  // disable trace there). The main class in trace mode returns a dict, so we
  // intentionally skip annotating the return to keep the type honest.
  const returnAnno = trace
    ? 'dict[str, list[int]]'
    : buildReturnAnnotation({
        ordered,
        connections,
        incoming,
        isSubclass,
        facade,
        tensorType,
        usedDtypes,
      })

  // Optional facade inputs get a `= None` default so callers may omit them.
  // Python requires defaulted params to be a suffix, so only the trailing run
  // of optional inputs is defaulted; an optional input that precedes a required
  // one stays positional (rare, but keeps the signature syntactically valid).
  const canDefault = inputNodes.map(() => false)
  for (let i = inputNodes.length - 1; i >= 0; i--) {
    if (!inputNodes[i]?.values?.optional) break
    canDefault[i] = true
  }
  const annotatedArgs = inputArgs.map(
    (name, i) => `${name}: ${argAnnotations[i]}${canDefault[i] ? ' = None' : ''}`
  )
  const fwdName = framework === 'pytorch' ? 'forward' : '__call__'
  const sigHead = `    def ${fwdName}(self${annotatedArgs.length ? ', ' + annotatedArgs.join(', ') : ''})`
  lines.push(returnAnno ? `${sigHead} -> ${returnAnno}:` : `${sigHead}:`)

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
    if (n.entry.kind === 'learnable') {
      if (trace) {
        const attr = attrName.get(n.id)
        lines.push(
          `        _runtime_shapes[${JSON.stringify(`${n.id}/out`)}] = list(self.${attr}.shape)`
        )
      }
      continue
    }
    // Constants are init-time only; wired ones feed ctor kwargs via __init__ params.
    if (n.entry.kind === 'const') continue
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
    const callArgs: string[] = []
    const danglingPorts: string[] = []
    for (const port of n.entry.inputs) {
      const key = `${n.id}/${port.name}`
      const incomingEdges = incoming.get(key) ?? []

      if (incomingEdges.length === 0) {
        if (port.optional) {
          // skip
        } else {
          // Emit a valid `=None` arg and flag it in a trailing comment on the
          // whole statement. An inline `# ...` here would comment out the rest
          // of the call (including the closing paren) and break the file.
          callArgs.push(`${port.name}=None`)
          danglingPorts.push(port.name)
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
    const danglingComment = danglingPorts.length
      ? `  # TODO: wire dangling required input${danglingPorts.length > 1 ? 's' : ''}: ${danglingPorts.join(', ')}`
      : ''

    const multi = n.entry.outputs.length > 1
    if (trace) {
      lines.push('        try:')
      if (multi) {
        const targets = n.entry.outputs
          .map((p) => outputVarFor.get(`${n.id}/${p.name}`))
          .join(', ')
        lines.push(`            ${targets} = ${callExpr}${danglingComment}`)
        for (const port of n.entry.outputs) {
          const v = outputVarFor.get(`${n.id}/${port.name}`)
          lines.push(
            `            _runtime_shapes[${JSON.stringify(`${n.id}/${port.name}`)}] = (list(${v}.shape) if hasattr(${v}, 'shape') else [])`
          )
        }
      } else {
        const v = localName.get(n.id)
        const portName = n.entry.outputs[0]?.name ?? 'out'
        lines.push(`            ${v} = ${callExpr}${danglingComment}`)
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
      lines.push(`        ${targets} = ${callExpr}${danglingComment}`)
    } else {
      const v = localName.get(n.id)
      lines.push(`        ${v} = ${callExpr}${danglingComment}`)
    }
  }

  // Sub-classes have their boundary outputs represented as virtual Output
  // nodes named `out0`, `out1`, ...; the return tuple must match that order
  // because the caller in the main class unpacks positionally. Plain topo
  // order can interleave them, so we sort by the trailing index for subclasses.
  const explicitReturnEntries = isSubclass
    ? collectExplicitReturnsOrdered(ordered, incoming, outputVarFor, outputReturnFor)
    : collectExplicitReturns(ordered, incoming, outputVarFor, outputReturnFor)
  if (!trace) {
    for (const { alias, srcVar } of explicitReturnEntries) {
      if (alias !== srcVar) lines.push(`        ${alias} = ${srcVar}`)
    }
  }
  const explicitOutputs = explicitReturnEntries.map((e) => e.alias)
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
function buildCallExpr(node: NodeLike, callArgs: string[], attrName: Map<string, string>, framework: string): string {
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
  if (node.entry.kind === 'unbind') {
    const xVar = positionalSource(callArgs, 'x')
    const dim = parseAxisDim(node.values?.dim, 0)
    // Count comes from values (the per-instance entry's outputs aren't visible
    // on the catalogue entry codegen reconstructs). Clamp to >= 1.
    const raw = Number(node.values?.count ?? node.entry.outputs.length)
    const count = Number.isFinite(raw) && raw >= 1 ? Math.trunc(raw) : 1
    // JAX has no unbind: moveaxis the split axis to front, then iterate (which
    // drops it) — same ordering and rank-reduction as torch.unbind.
    const tuple = framework === 'pytorch'
      ? `torch.unbind(${xVar}, dim=${dim})`
      : `tuple(jnp.moveaxis(${xVar}, ${dim}, 0))`
    // count===1 takes codegen's single-output path (`v = <expr>`), which would
    // assign the whole tuple; index element 0 so it's a tensor.
    if (count === 1) {
      return framework === 'pytorch'
        ? `torch.unbind(${xVar}, dim=${dim})[0]`
        : `jnp.moveaxis(${xVar}, ${dim}, 0)[0]`
    }
    return tuple
  }
  if (node.entry.kind === 'pool') {
    const xVar = positionalSource(callArgs, 'x')
    const { kernel, stride, padding, mode } = parsePoolParams(node.values)
    if (framework === 'pytorch') {
      const fn = mode === 'avg' ? 'avg_pool2d' : 'max_pool2d'
      return `torch.nn.functional.${fn}(${xVar}, kernel_size=${kernel}, stride=${stride}, padding=${padding})`
    }
    const poolFn = mode === 'avg' ? 'avg_pool' : 'max_pool'
    const pad =
      padding > 0 ? `((${padding}, ${padding}), (${padding}, ${padding}))` : "'VALID'"
    return `jnp.transpose(${poolFn}(jnp.transpose(${xVar}, (0, 2, 3, 1)), window_shape=(${kernel}, ${kernel}), strides=(${stride}, ${stride}), padding=${pad}), (0, 3, 1, 2))`
  }
  if (node.entry.kind === 'eltwise') {
    // Fold the variadic operands with an infix operator. `+`/`*` broadcast
    // identically in PyTorch and JAX, so the expression is framework-agnostic.
    const tensors = variadicList(callArgs, 'xs') // e.g. "[a, b, c]"
    const items = tensors.replace(/^\[|\]$/g, '').trim()
    const parts = items ? items.split(',').map((s) => s.trim()).filter(Boolean) : []
    const op = String(node.values?.op ?? 'add') === 'multiply' ? '*' : '+'
    if (parts.length === 0) return framework === 'pytorch' ? 'None' : 'None'
    if (parts.length === 1) return parts[0]
    return `(${parts.join(` ${op} `)})`
  }
  if (node.entry.kind === 'upsample') {
    const xVar = positionalSource(callArgs, 'x')
    const scale = parseScaleFactor(node.values?.scale_factor, 2)
    const align = parseBoolValue(node.values?.align_corners, false)
    if (framework === 'pytorch') {
      return `torch.nn.functional.interpolate(${xVar}, scale_factor=${scale}, mode='bilinear', align_corners=${align ? 'True' : 'False'})`
    }
    return `jnp.transpose(jax.image.resize(jnp.transpose(${xVar}, (0, 2, 3, 1)), (${xVar}.shape[0], int(${xVar}.shape[2] * ${scale}), int(${xVar}.shape[3] * ${scale}), ${xVar}.shape[1]), method='linear'), (0, 3, 1, 2))`
  }
  if (node.entry.kind === 'module' || node.entry.kind === 'group') {
    return `self.${attrName.get(node.id)}(${callArgs.join(', ')})`
  }
  return `${node.entry.name}(${callArgs.join(', ')})`
}

function constLiteral(node: NodeLike): string {
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

/** Map every Constant node to an __init__ parameter defaulting to its UI
 *  value. Wired constants name the param after the target ctor slot; unwired
 *  ones use the codegen attr name (constant_0, …). Module ctors reference the
 *  __init__ parameter directly — no self.* storage. */
function analyzeConstWiring(nodes: NodeLike[], connections: Connection[], attrName: Map<string, string>): { initParams: any[]; paramRef: Map<string, string> } {
  const byId = new Map<string, NodeLike>(nodes.map((n) => [n.id, n]))
  const constInitByNodeId = new Map<string, any>()
  const paramRef = new Map<string, string>()
  const usedInitNames = new Set<string>()

  const allocInitName = (base: any) => {
    let c = sanitizePyIdent(base, 'param')
    let i = 2
    while (usedInitNames.has(c)) c = `${sanitizePyIdent(base, 'param')}${i++}`
    usedInitNames.add(c)
    return c
  }

  const pyTypeForConst = (constNode: NodeLike, paramDef: any) => {
    const vt = constNode.values?.value_type || paramDef?.type || 'int'
    if (vt === 'bool') return 'bool'
    if (vt === 'str') return 'str'
    if (vt === 'float') return 'float'
    return 'int'
  }

  for (const c of connections) {
    const source = byId.get(c.source)
    const target = byId.get(c.target)
    if (!source || !target) continue
    if (source.entry?.kind !== 'const') continue
    const spec = (target.inputs as any)?.[c.targetInput]?.portSpec
    if (spec?.kind !== 'param') continue
    const paramName = spec.paramName
    const paramDef = (target.entry.ctor || []).find((p) => p.name === paramName)

    if (!constInitByNodeId.has(source.id)) {
      constInitByNodeId.set(source.id, {
        initName: allocInitName(paramName),
        pyType: pyTypeForConst(source, paramDef),
        defaultLit: constLiteral(source),
      })
    }
    paramRef.set(`${target.id}::${paramName}`, constInitByNodeId.get(source.id).initName)
  }

  // Every remaining Constant node becomes an __init__ param even if unwired.
  for (const n of nodes) {
    if (n.entry?.kind !== 'const') continue
    if (constInitByNodeId.has(n.id)) continue
    constInitByNodeId.set(n.id, {
      initName: allocInitName(attrName.get(n.id) || 'constant'),
      pyType: pyTypeForConst(n, null),
      defaultLit: constLiteral(n),
    })
  }

  return { initParams: [...constInitByNodeId.values()], paramRef }
}

/**
 * Resolve the *effective default value* of a ctor param exposed on a facade
 * port. When the facade child is a real block, that's its `values[param]` (or
 * the manifest default). When the child is itself a **nested group facade**, the
 * param has no manifest default (groups carry `ctor: []`), so we recurse through
 * the inner group's portMap to the underlying block that actually owns the
 * param - whose value (or internal Constant) is the real default. Without this,
 * a param re-exposed through one or more nested groups defaults to 0, which then
 * flows down as e.g. `out_ch=0` and produces a zero-sized weight at runtime.
 */
function resolveExposedParamDefault(
  childNodeId: string,
  paramName: string,
  allById: Map<string, NodeLike> | undefined,
  seen: Set<string> = new Set()
): { raw: any; paramDef: any } {
  const child = allById?.get(childNodeId)
  if (!child) return { raw: undefined, paramDef: undefined }
  const key = `${childNodeId}::${paramName}`
  if (seen.has(key)) return { raw: undefined, paramDef: undefined }
  seen.add(key)

  // Real block: the value lives directly on the node (or its manifest ctor).
  const paramDef = child.entry?.ctor?.find((p: any) => p.name === paramName)
  if (paramDef) {
    return { raw: (child.values as any)?.[paramName] ?? paramDef.default, paramDef }
  }

  // Nested group facade: descend through its portMap param of the same name.
  if (child.entry?.kind === 'group') {
    const inner = (child.entry as any).portMap?.params?.find((p: any) => p.paramName === paramName)
    if (inner) return resolveExposedParamDefault(inner.childNodeId, paramName, allById, seen)
  }
  return { raw: undefined, paramDef: undefined }
}

/** Subclass __init__ params for facade param ports (external constants wire here). */
function mergeGroupFacadeInitParams(
  facade: any,
  nodes: NodeLike[],
  initParams: any[],
  paramRef: Map<string, string>,
  allById?: Map<string, NodeLike>
): void {
  const byId = new Map<string, NodeLike>(nodes.map((n) => [n.id, n]))
  const used = new Set(initParams.map((p) => p.initName))
  for (const m of facade.entry?.portMap?.params ?? []) {
    const child = byId.get(m.childNodeId)
    const initName = sanitizePyIdent(m.paramName, 'param')
    if (!used.has(initName)) {
      // Prefer the locally-visible child, but fall back to a global recursive
      // resolve so a param exposed THROUGH a nested group still gets the real
      // default rather than 0.
      const local = child?.entry?.ctor?.find((p) => p.name === m.paramName)
      const resolved = local
        ? { raw: (child?.values as any)?.[m.paramName] ?? local.default, paramDef: local }
        : resolveExposedParamDefault(m.childNodeId, m.paramName, allById)
      const paramDef = resolved.paramDef
      const raw = resolved.raw
      initParams.push({
        initName,
        pyType: pyTypeForParamDef(paramDef, m.paramType),
        defaultLit:
          raw === null || raw === undefined || raw === ''
            ? pyRepr(coerce(0, m.paramType ?? 'int'))
            : pyRepr(coerce(raw, paramDef?.type ?? m.paramType ?? 'int')),
      })
      used.add(initName)
    }
    paramRef.set(`${m.childNodeId}::${m.paramName}`, initName)
  }
}

/**
 * Bubble exposed params of child GROUP facades up into the containing class.
 *
 * For each group node in `ordered`, every entry in its `portMap.params` is an
 * exposed knob. If it isn't already driven by a local Constant wire (i.e. no
 * `paramRef[facade.id::paramName]` yet), we mint a shared `__init__` parameter
 * on the containing class - one per paramName (same-named exposed params across
 * sibling groups share a single outer arg) - and register the paramRef so
 * `groupCtorArgs` threads it down as `paramName=<outerParam>`. Defaults resolve
 * recursively through nested groups via `resolveExposedParamDefault`.
 */
function bubbleChildGroupParams(
  ordered: NodeLike[],
  initParams: any[],
  paramRef: Map<string, string>,
  allById?: Map<string, NodeLike>
): void {
  const used = new Set(initParams.map((p) => p.initName))
  // Reuse one outer param per paramName so two groups exposing `out_ch` share it.
  const sharedByName = new Map<string, string>()
  for (const n of ordered) {
    if (n.entry?.kind !== 'group') continue
    const params = (n.entry as any)?.portMap?.params ?? []
    for (const m of params) {
      const key = `${n.id}::${m.paramName}`
      if (paramRef.has(key)) continue // already wired by a local Constant
      let initName = sharedByName.get(m.paramName)
      if (!initName) {
        initName = sanitizePyIdent(m.paramName, 'param')
        // Avoid clobbering an unrelated existing init param of the same name.
        let i = 2
        while (used.has(initName)) initName = `${sanitizePyIdent(m.paramName, 'param')}${i++}`
        used.add(initName)
        sharedByName.set(m.paramName, initName)
        const { raw, paramDef } = resolveExposedParamDefault(m.childNodeId, m.paramName, allById)
        initParams.push({
          initName,
          pyType: pyTypeForParamDef(paramDef, m.paramType),
          defaultLit:
            raw === null || raw === undefined || raw === ''
              ? pyRepr(coerce(0, m.paramType ?? 'int'))
              : pyRepr(coerce(raw, paramDef?.type ?? m.paramType ?? 'int')),
        })
      }
      paramRef.set(key, initName)
    }
  }
}

function pyTypeForParamDef(paramDef: any, fallbackType: any): string {
  const t = paramDef?.type ?? fallbackType ?? 'int'
  if (t === 'bool') return 'bool'
  if (t === 'str') return 'str'
  if (t === 'float') return 'float'
  return 'int'
}

/** Pass wired constants through to a group facade's __init__. A facade's
 *  portMap may list the same paramName more than once (one entry per child
 *  that shares an exposed param), so dedupe by paramName to avoid emitting a
 *  repeated keyword argument. */
function groupCtorArgs(facadeNode: NodeLike, paramRef: Map<string, string>): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const m of (facadeNode.entry as any)?.portMap?.params ?? []) {
    if (seen.has(m.paramName)) continue
    const wired = paramRef.get(`${facadeNode.id}::${m.paramName}`)
    if (wired) {
      out.push(`${m.paramName}=${wired}`)
      seen.add(m.paramName)
    }
  }
  return out
}

function positionalSource(callArgs: string[], portName: string): string {
  const prefix = `${portName}=`
  const found = callArgs.find((a) => a.startsWith(prefix))
  if (!found) return 'None'  // dangling input; runtime will fail loudly
  return found.slice(prefix.length)
}

/** Pull `[a, b, c]` from `xs=[a, b, c]` in callArgs; default `[]`. */
function variadicList(callArgs: string[], portName: string): string {
  const prefix = `${portName}=`
  const found = callArgs.find((a) => a.startsWith(prefix))
  if (!found) return '[]'
  return found.slice(prefix.length)
}

function parseAxisDim(v: any, fallback: number = 0): number {
  if (v === null || v === undefined || v === '') return fallback
  const n = Number(v)
  return Number.isFinite(n) ? Math.trunc(n) : fallback
}

/** "h=8, w=16; b=2" -> "h=8, w=16, b=2" (only valid name=int kwargs). */
function parseLengthsKwargs(s: any): string {
  if (!s) return ''
  const out: string[] = []
  for (const part of String(s).split(/[,;\n]/)) {
    const m = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+)\s*$/.exec(part)
    if (m) out.push(`${m[1]}=${m[2]}`)
  }
  return out.join(', ')
}

/** Space/comma-separated int (or -1) tokens. Defaults to "-1" if empty. */
function parseReshapeDims(s: any): number[] {
  const toks = String(s ?? '')
    .trim()
    .split(/[\s,]+/)
    .filter(Boolean)
  const dims: number[] = []
  for (const t of toks) {
    if (/^-?\d+$/.test(t)) dims.push(Number(t))
  }
  return dims.length ? dims : [-1]
}

/** Numeric-only shape tokens for LearnableTensor __init__ (defaults to scalar). */
function parseLearnableDims(s: any): number[] {
  const toks = String(s ?? '')
    .trim()
    .split(/[\s,]+/)
    .filter(Boolean)
  const dims: number[] = []
  for (const t of toks) {
    if (/^\d+$/.test(t)) dims.push(Number(t))
  }
  return dims.length ? dims : [1]
}

function learnableInitLine(node: NodeLike, attr: string, framework: string): string {
  const dims = parseLearnableDims(node.values?.shape)
  const init = String(node.values?.init ?? 'randn').toLowerCase()
  const shapeArgs = dims.join(', ')
  const shapeTuple = dims.length === 1 ? `(${shapeArgs},)` : `(${shapeArgs})`
  if (framework === 'pytorch') {
    let tensor
    if (init === 'zeros') tensor = `torch.zeros${shapeTuple}`
    else if (init === 'ones') tensor = `torch.ones${shapeTuple}`
    else tensor = `torch.randn${shapeTuple}`
    return `self.${attr} = nn.Parameter(${tensor})`
  }
  let arr
  if (init === 'zeros') arr = `jnp.zeros${shapeTuple}`
  else if (init === 'ones') arr = `jnp.ones${shapeTuple}`
  else arr = `jax.random.normal(rngs.params(), ${shapeTuple})`
  return `self.${attr} = nnx.Param(${arr})`
}

function parseBoolValue(v: any, fallback: boolean = false): boolean {
  if (v === true || v === 'true' || v === '1' || v === 1) return true
  if (v === false || v === 'false' || v === '0' || v === 0) return false
  return fallback
}

function parseScaleFactor(v: any, fallback: number = 2): number {
  if (v === null || v === undefined || v === '') return fallback
  const n = Number(v)
  return Number.isFinite(n) && n > 0 ? n : fallback
}

/** stride=0 means "same as kernel_size" (PyTorch default behaviour). */
function parsePoolParams(values: any = {}): { kernel: number; stride: number; padding: number; mode: string } {
  const kernel = parseAxisDim(values.kernel_size, 2)
  const strideRaw = values.stride
  const stride =
    strideRaw === null || strideRaw === undefined || strideRaw === '' || Number(strideRaw) === 0
      ? kernel
      : parseAxisDim(strideRaw, kernel)
  const padding = parseAxisDim(values.padding, 0)
  const mode = String(values.mode ?? 'max').toLowerCase() === 'avg' ? 'avg' : 'max'
  return { kernel, stride, padding, mode }
}

function ctorArgs(
  node: NodeLike,
  paramRef: Map<string, string> = new Map(),
  axisSub?: Substitution
): string[] {
  // paramName -> axis, for ctor params the manifest binds to a shape axis
  // (e.g. {"D_val": "vdim"} yields vdim -> D_val).
  const paramAxis = new Map<string, string>()
  for (const [axis, pname] of Object.entries(node.entry.bindings || {})) {
    paramAxis.set(pname, axis)
  }
  const out: string[] = []
  for (const p of node.entry.ctor) {
    const wired = paramRef.get(`${node.id}::${p.name}`)
    if (wired) {
      out.push(`${p.name}=${wired}`)
      continue
    }
    const v = (node.values as any)[p.name]
    if (v !== null && v !== undefined && v !== '') {
      // Skip if it equals the default to keep the output tidy.
      if (deepEqual(v, p.default)) continue
      out.push(`${p.name}=${pyRepr(coerce(v, p.type))}`)
      continue
    }
    // Unset: if this param is bound to an axis whose value the graph resolved,
    // emit that concrete dim so the layer is sized to the data actually wired
    // in (e.g. a 256-wide value stream -> vdim=256) instead of the default.
    // Suppress it when it merely restates the model `dim`, so equal-width
    // graphs emit exactly as before.
    const axis = paramAxis.get(p.name)
    if (axis && axisSub) {
      const r = resolve(`${axis}#${node.id}`, axisSub)
      const dimVal = Number((node.values as any)?.dim)
      const redundant = Number.isFinite(dimVal) && r === dimVal
      if (typeof r === 'number' && Number.isInteger(r) && !redundant) {
        out.push(`${p.name}=${pyRepr(coerce(r, p.type))}`)
      }
    }
  }
  return out
}

function coerce(v: any, type: any): any {
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

function deepEqual(a: any, b: any): boolean {
  if (a === b) return true
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((x, i) => deepEqual(x, b[i]))
  }
  return false
}

/** Input ports with no incoming edges that aren't optional ⇒ surfaced as forward() args. */
function findEntryInputs(nodes: NodeLike[], connections: Connection[], usedNames: Set<string> = new Set()): any[] {
  const incomingKeys = new Set(connections.map((c) => `${c.target}/${c.targetInput}`))
  const out: any[] = []
  const allocate = (base: string) => {
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

function explicitReturnEntry(outNode: NodeLike, incoming: Map<string, Connection[]>, outputVarFor: Map<string, string>, outputReturnFor: Map<string, string>): any {
  const key = `${outNode.id}/x`
  const edges = incoming.get(key) ?? []
  if (edges.length === 0) return null
  const c = edges[0]
  const srcVar = outputVarFor.get(`${c.source}/${c.sourceOutput}`)
  if (!srcVar) return null
  const alias = outputReturnFor.get(outNode.id) ?? srcVar
  return { alias, srcVar, node: outNode }
}

/** Return aliases for explicit Output nodes (topo order). */
function collectExplicitReturns(ordered: NodeLike[], incoming: Map<string, Connection[]>, outputVarFor: Map<string, string>, outputReturnFor: Map<string, string>): any[] {
  const out: any[] = []
  for (const n of ordered) {
    if (n.entry.kind !== 'output') continue
    const entry = explicitReturnEntry(n, incoming, outputVarFor, outputReturnFor)
    if (entry) out.push(entry)
  }
  return out
}

/**
 * Like collectExplicitReturns, but sorts Output nodes by the trailing numeric
 * index in `values.name` (e.g. "out0" < "out1" < "out10").
 */
function collectExplicitReturnsOrdered(ordered: NodeLike[], incoming: Map<string, Connection[]>, outputVarFor: Map<string, string>, outputReturnFor: Map<string, string>): any[] {
  const outNodes = ordered.filter((n) => n.entry.kind === 'output')
  outNodes.sort((a, b) => {
    const ai = parseInt(String(a.values?.name ?? '').replace(/^\D+/, ''), 10)
    const bi = parseInt(String(b.values?.name ?? '').replace(/^\D+/, ''), 10)
    const ax = Number.isFinite(ai) ? ai : 0
    const bx = Number.isFinite(bi) ? bi : 0
    return ax - bx
  })
  const out: any[] = []
  for (const n of outNodes) {
    const entry = explicitReturnEntry(n, incoming, outputVarFor, outputReturnFor)
    if (entry) out.push(entry)
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

function partitionByGroup(nodes: NodeLike[], connections: Connection[]): { facadesByGid: Map<string, NodeLike>; childrenByGid: Map<string, NodeLike[]>; internalByGid: Map<string, Connection[]> } {
  const facadesByGid = new Map<string, NodeLike>()
  for (const n of nodes) {
    if (n.entry?.kind === 'group' && n.entry.groupId) {
      facadesByGid.set(n.entry.groupId, n)
    }
  }
  const childrenByGid = new Map<string, NodeLike[]>()
  const internalByGid = new Map<string, Connection[]>()
  for (const gid of facadesByGid.keys()) {
    childrenByGid.set(gid, [])
    internalByGid.set(gid, [])
  }
  const byId = new Map<string, NodeLike>(nodes.map((n) => [n.id, n]))
  // Membership is recursive: a node belongs to the group named by `node.groupId`
  // regardless of whether it is itself a facade. A *nested* group facade carries
  // both `entry.groupId` (its own identity, registered above) and `node.groupId`
  // (the outer group it is a member of) - so it lands here as a child of the
  // outer group while still being the facade of its own. A top-level facade has
  // `node.groupId == null` and is a member of nothing.
  for (const n of nodes) {
    if (n.groupId && childrenByGid.has(n.groupId)) {
      ;(childrenByGid.get(n.groupId) as NodeLike[]).push(n)
    }
  }
  for (const c of connections) {
    const src = byId.get(c.source)
    const tgt = byId.get(c.target)
    if (src?.groupId && src.groupId === tgt?.groupId && internalByGid.has(src.groupId)) {
      ;(internalByGid.get(src.groupId) as Connection[]).push(c)
      continue
    }
    // Self-referential boundary edge: a child wired straight into its OWN
    // group's facade input (e.g. a residual skip where an attention output
    // feeds the group's `in_i`, which the portMap then routes into a sibling
    // child). Such an edge is NOT same-group (the facade's node.groupId differs),
    // so it would otherwise be dropped here while buildSubgraphView only emits a
    // synthetic Input for that port - severing the child->child link and, for a
    // variadic target like an Elementwise `xs`, silently losing one operand.
    // Rewrite it to the direct internal edge the portMap implies.
    const srcGid = src?.groupId
    if (
      srcGid &&
      tgt?.entry?.kind === 'group' &&
      tgt.entry.groupId === srcGid &&
      internalByGid.has(srcGid)
    ) {
      const m = (tgt.entry as any).portMap?.inputs?.find(
        (pm: any) => pm.facadePort === c.targetInput
      )
      if (m?.childNodeId && m.childPort) {
        ;(internalByGid.get(srcGid) as Connection[]).push({
          id: `__self_boundary_${c.id ?? `${c.source}_${c.targetInput}`}`,
          source: c.source,
          sourceOutput: c.sourceOutput,
          target: m.childNodeId,
          targetInput: m.childPort,
        })
      }
    }
  }
  return { facadesByGid, childrenByGid, internalByGid }
}

/**
 * Order group classes so every class is emitted AFTER the classes it
 * instantiates as members (deepest containment first). `deps` maps a class name
 * to the set of class names it contains. Returns a flat list in emit order.
 * Throws on a containment cycle (structurally impossible in the editor, but a
 * corrupt file could produce one).
 */
function topoSortClasses(deps: Map<string, Set<string>>): string[] {
  const order: string[] = []
  const state = new Map<string, 'visiting' | 'done'>()
  const visit = (cls: string) => {
    const s = state.get(cls)
    if (s === 'done') return
    if (s === 'visiting') throw new Error('graph contains a group-containment cycle')
    state.set(cls, 'visiting')
    for (const dep of deps.get(cls) ?? []) visit(dep)
    state.set(cls, 'done')
    order.push(cls)
  }
  for (const cls of deps.keys()) visit(cls)
  return order
}

function groupClassName(facadeName: any, _gid: any): string {
  // Class name comes purely from the user-facing group name so that two
  // groups sharing a name (e.g. duplicating "Encoder") map to the *same*
  // generated class instead of `Encoder_aaaa` and `Encoder_bbbb`. Multiple
  // facades that resolve to the same class are deduped in `generate()` so
  // Python doesn't see a redefinition.
  const sane = sanitizePyIdent(facadeName || 'Group', 'Group')
  return sane.charAt(0).toUpperCase() + sane.slice(1)
}

/**
 * Construct a virtual graph that represents one group as a standalone class:
 * synthetic Input nodes feed each boundary input, synthetic Output nodes
 * collect each boundary output, and the internal edges are the rest of the
 * subgraph's structure.
 */
function buildSubgraphView(facade: any, children: NodeLike[], internalConnections: Connection[]): { nodes: any[]; connections: any[] } {
  const inputs = facade.entry.portMap?.inputs || []
  const outputs = facade.entry.portMap?.outputs || []

  const makeVirtual = (id: string, entry: any, values: any): any => ({
    id,
    entry,
    label: entry.name,
    tag: '',
    groupId: null,
    values,
    inputs: Object.fromEntries(
      (entry.inputs || []).map((p: any) => [p.name, { portSpec: p }])
    ),
    outputs: Object.fromEntries(
      (entry.outputs || []).map((p: any) => [p.name, { portSpec: p }])
    ),
    freshenedShape() { return null },
    applyParamBindings() {},
  })

  const virtualInputs = inputs.map((m: any, i: number) =>
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
      // Carry the facade port's optionality so the subclass forward() can give
      // it a `= None` default (an optional child port, e.g. an attention mask,
      // must not become a required positional arg of the generated class).
      { name: `in${i}`, shape: '', dtype: 'any', optional: Boolean(facade.entry?.inputs?.[i]?.optional) }
    )
  )
  const virtualOutputs = outputs.map((m: any, i: number) =>
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

  const inputEdges = inputs
    .map((m: any, i: number) => ({
      source: `__sg_in_${facade.id}_${i}`,
      sourceOutput: 'out',
      target: m.childNodeId,
      targetInput: m.childPort,
    }))
    .filter((c: any) => c.target && c.targetInput)
  const outputEdges = outputs
    .map((m: any, i: number) => ({
      source: m.childNodeId,
      sourceOutput: m.childPort,
      target: `__sg_out_${facade.id}_${i}`,
      targetInput: 'x',
    }))
    .filter((c: any) => c.source && c.sourceOutput)

  return {
    nodes: [...virtualInputs, ...children, ...virtualOutputs],
    connections: [...inputEdges, ...internalConnections, ...outputEdges],
  }
}

/** Output ports that have no outgoing edges ⇒ return values. */
function findTerminals(nodes: NodeLike[], connections: Connection[]): any[] {
  const sourceKeys = new Set(connections.map((c) => `${c.source}/${c.sourceOutput}`))
  const out: any[] = []
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
