/**
 * Build Rete nodes from manifest entries.
 *
 * Each `BlockNode` is a `ClassicPreset.Node` enriched with:
 *   - `entry`:   the original manifest entry (frozen)
 *   - `values`:  per-instance ctor-param values (mutated by Inspector)
 *   - `axisShape(port, side)`: returns a *freshened* shape, with axis variables
 *     suffixed by the node id so different node instances don't collide.
 *
 * A single global `tensor` Socket is used for all connections; shape
 * validation is performed by the validator on top of the type system.
 */

import { ClassicPreset } from 'rete'
import { normalize, freshen } from './shape.js'

export const tensorSocket = new ClassicPreset.Socket('tensor')

export class BlockNode extends ClassicPreset.Node {
  constructor(entry) {
    super(entry.name)
    this.entry = entry
    // Free-form user annotation. Doubles as a *weight-sharing identity* for
    // module-kind nodes: two ConvBlocks tagged "down1" emit one self.down1 in
    // codegen and call it from each forward-pass site.
    this.tag = ''
    this.values = Object.fromEntries(
      entry.ctor.map((p) => [p.name, p.default])
    )
    // Cache normalized shapes (raw, without per-node freshening).
    this._inputShapes = Object.fromEntries(
      entry.inputs.map((p) => [p.name, normalize(p.shape)])
    )
    this._outputShapes = Object.fromEntries(
      entry.outputs.map((p) => [p.name, normalize(p.shape)])
    )
    this._paramSpecs = new Map()

    for (const p of entry.inputs) {
      const input = new ClassicPreset.Input(
        tensorSocket,
        labelFor(p),
        Boolean(p.variadic)
      )
      input.multipleConnections = Boolean(p.variadic)
      // Stash the manifest port spec for later inspection.
      input.portSpec = p
      this.addInput(p.name, input)
    }
    // Ctor parameters are init-time values (not runtime tensor flow). They are
    // hidden by default and can be explicitly exposed per-param from inspector.
    if (entry.kind !== 'input' && entry.kind !== 'const') {
      for (const p of entry.ctor) this._paramSpecs.set(p.name, p)
    }
    for (const p of entry.outputs) {
      const output = new ClassicPreset.Output(tensorSocket, labelFor(p))
      output.portSpec = p
      this.addOutput(p.name, output)
    }
  }

  /** Freshened shape for the given port, scoped to this node's id. */
  freshenedShape(portName, side /* 'in' | 'out' */) {
    const raw =
      side === 'in' ? this._inputShapes[portName] : this._outputShapes[portName]
    if (!raw) return null
    return freshen(raw, this.id)
  }

  /** Apply ctor-param -> axis bindings to the substitution map. */
  applyParamBindings(sub) {
    const bindings = this.entry.bindings || {}
    for (const [axis, paramName] of Object.entries(bindings)) {
      const value = this.values[paramName]
      if (value === null || value === undefined || value === '') continue
      const v = Number(value)
      if (!Number.isFinite(v) || !Number.isInteger(v)) continue
      const freshAxis = `${axis}#${this.id}`
      // Set the binding directly. The unifier walks substitutions, so a later
      // unifyShape() will see this as the resolved value of freshAxis.
      const existing = sub.get(freshAxis)
      if (existing !== undefined && existing !== v) {
        // Conflict between user-provided ctor value and an already-derived binding.
        // We still write it - the validator surfaces the resulting mismatch.
      }
      sub.set(freshAxis, v)
    }
  }

  paramInputKey(paramName) {
    return `__param__${paramName}`
  }

  isParamExposed(paramName) {
    return Boolean(this.inputs[this.paramInputKey(paramName)])
  }

  exposeParam(paramName) {
    if (!this._paramSpecs.has(paramName) || this.isParamExposed(paramName)) return
    const p = this._paramSpecs.get(paramName)
    const key = this.paramInputKey(paramName)
    const input = new ClassicPreset.Input(tensorSocket, `🔴 ${p.name}`, false)
    input.multipleConnections = false
    input.portSpec = {
      kind: 'param',
      paramName: p.name,
      required: Boolean(p.required),
      paramType: p.type,
    }
    this.addInput(key, input)
  }

  hideParam(paramName) {
    const key = this.paramInputKey(paramName)
    if (!this.inputs[key]) return
    this.removeInput(key)
  }
}

function labelFor(port) {
  const tag = port.variadic ? '[*]' : port.optional ? '?' : ''
  return `${port.name}${tag}`
}

/** Display title for a node: "<BlockName>" or "<BlockName> · <tag>". */
export function computeNodeLabel(node) {
  const base = node.entry.name
  const t = String(node.tag ?? '').trim()
  return t ? `${base} · ${t}` : base
}

/**
 * Update `node.tag` and refresh its displayed label in place. The rete area
 * still has to be told to re-render the node afterwards (`area.update('node',
 * id)`); the caller does that to keep this module free of area concerns.
 */
export function applyNodeTag(node, newTag) {
  node.tag = String(newTag ?? '')
  node.label = computeNodeLabel(node)
}

// Registry of tag (case-insensitive) -> assigned HSL colour. Persists for the
// lifetime of the page so weight-tied twins stay visually grouped across
// re-renders. Colours are handed out along the golden angle so every new tag
// lands maximally far from every existing one in hue space - no collisions,
// no near-duplicates even with many tags.
const GOLDEN_ANGLE_DEG = 137.5077640500378
const _tagColors = new Map()

/**
 * Unique, stable colour per tag. Two different tags are guaranteed to receive
 * different (and perceptually well-separated) hues.
 */
export function colorForTag(tag) {
  const s = String(tag ?? '').trim().toLowerCase()
  if (!s) return null
  const cached = _tagColors.get(s)
  if (cached) return cached
  const idx = _tagColors.size
  const hue = (idx * GOLDEN_ANGLE_DEG) % 360
  // Mild S/L jitter keyed off the index keeps hues near 0/360 from looking
  // identical to the eye as the palette wraps around.
  const sat = 68 + (idx % 3) * 6   // 68 / 74 / 80
  const lit = 58 + (idx % 2) * 6   // 58 / 64
  const color = `hsl(${hue.toFixed(2)}, ${sat}%, ${lit}%)`
  _tagColors.set(s, color)
  return color
}

// ---------------------------------------------------------------------------
// Input node - synthetic source node with user-defined shape & dtype.
// ---------------------------------------------------------------------------

/**
 * Built-in palette entry for an Input node. Mixed literals and named axes are
 * fine ("B 3 224 224", "B C H W", "2 3 64 64"). Not tied to any framework.
 */
export const INPUT_ENTRY = {
  name: 'Input',
  module: '__builtin__',
  framework: 'any',
  kind: 'input',
  ctor: [
    { name: 'name', type: 'str', default: 'x', required: false },
    { name: 'shape', type: 'str', default: 'B C H W', required: false },
    {
      name: 'dtype',
      type: 'str',
      default: 'float',
      required: false,
      choices: ['float', 'int', 'bool', 'complex', 'any'],
    },
  ],
  inputs: [],
  outputs: [
    {
      name: 'out',
      shape: ['B', 'C', 'H', 'W'],
      dtype: 'float',
      optional: false,
      variadic: false,
    },
  ],
  bindings: {},
}

/**
 * Explicit graph sink. Generated forward() returns only values routed into
 * Output nodes (in topo order), ignoring ordinary leaf outputs when present.
 */
export const OUTPUT_ENTRY = {
  name: 'Output',
  module: '__builtin__',
  framework: 'any',
  kind: 'output',
  ctor: [
    { name: 'name', type: 'str', default: 'y', required: false },
  ],
  inputs: [
    {
      name: 'x',
      shape: ['...'],
      dtype: 'any',
      optional: false,
      variadic: false,
    },
  ],
  outputs: [],
  bindings: {},
}

/**
 * Scalar constant for init-time parameter wiring.
 * Connect `out` to a node's `🔴 <param>` input.
 */
export const CONST_ENTRY = {
  name: 'Constant',
  module: '__builtin__',
  framework: 'any',
  kind: 'const',
  ctor: [
    {
      name: 'value_type',
      type: 'str',
      default: 'int',
      required: false,
      choices: ['int', 'float', 'bool', 'str'],
    },
    { name: 'value', type: 'str', default: '1', required: true },
  ],
  inputs: [],
  outputs: [
    {
      name: 'out',
      shape: [],
      dtype: 'any',
      optional: false,
      variadic: false,
    },
  ],
  bindings: {},
}

/** "B, 3,224 224" -> ["B", "3", "224", "224"] */
export function parseShapeString(s) {
  if (s == null) return []
  return String(s)
    .trim()
    .split(/[\s,]+/)
    .filter((tok) => tok.length > 0)
}

/**
 * Source node whose output shape and dtype come from user-typed values rather
 * than a manifest annotation. Inherits everything else from BlockNode so the
 * inspector / palette / unifier treat it like a normal entry.
 */
export class InputNode extends BlockNode {
  freshenedShape(portName, side) {
    if (side !== 'out') return null
    const tokens = normalize(parseShapeString(this.values.shape))
    // Keep the rete port's dtype tag in sync so the validator's dtype check
    // sees the latest value the user typed.
    const port = this.outputs[portName]
    if (port?.portSpec) port.portSpec.dtype = this.values.dtype || 'any'
    return freshen(tokens, this.id)
  }
}

/** Sink node with user-typed label (`name`) and unconstrained input shape. */
export class OutputNode extends BlockNode {
  freshenedShape(portName, side) {
    if (side !== 'in') return null
    return freshen(normalize(['...']), this.id)
  }
}

// ---------------------------------------------------------------------------
// Rearrange / Reshape - framework-agnostic shape ops driven by string patterns.
// Both nodes pipe through to einops.rearrange / .reshape() at codegen time.
// ---------------------------------------------------------------------------

/**
 * Built-in einops.rearrange wrapper. The pattern string drives both the codegen
 * call and the static shape inference: each top-level LHS group becomes an
 * axis on the input port, each RHS group an axis on the output port. Groups
 * shared verbatim between sides propagate their binding through unification.
 */
export const REARRANGE_ENTRY = {
  name: 'Rearrange',
  module: '__utility__',
  framework: 'any',
  kind: 'rearrange',
  ctor: [
    {
      name: 'pattern',
      type: 'str',
      default: 'b c h w -> b (h w) c',
      required: true,
    },
    {
      name: 'lengths',
      type: 'str',
      default: '',
      required: false,
    },
  ],
  inputs: [
    { name: 'x', shape: ['...'], dtype: 'float', optional: false, variadic: false },
  ],
  outputs: [
    { name: 'out', shape: ['...'], dtype: 'float', optional: false, variadic: false },
  ],
  bindings: {},
}

/**
 * Built-in .reshape() / jnp.reshape() wrapper. The shape string accepts ints
 * and -1; symbolic axes are intentionally not supported here (use Rearrange).
 */
export const RESHAPE_ENTRY = {
  name: 'Reshape',
  module: '__utility__',
  framework: 'any',
  kind: 'reshape',
  ctor: [
    {
      name: 'shape',
      type: 'str',
      default: '-1',
      required: true,
    },
  ],
  inputs: [
    { name: 'x', shape: ['...'], dtype: 'float', optional: false, variadic: false },
  ],
  outputs: [
    { name: 'out', shape: ['...'], dtype: 'float', optional: false, variadic: false },
  ],
  bindings: {},
}

/**
 * Concatenate variadic inputs along `dim` (PyTorch dim / JAX axis).
 * Wire two or more tensors into the `xs[*]` port.
 */
export const CONCAT_ENTRY = {
  name: 'Concat',
  module: '__utility__',
  framework: 'any',
  kind: 'concat',
  ctor: [
    {
      name: 'dim',
      type: 'int',
      default: 1,
      required: false,
    },
  ],
  inputs: [
    { name: 'xs', shape: ['...'], dtype: 'float', optional: false, variadic: true },
  ],
  outputs: [
    { name: 'out', shape: ['...'], dtype: 'float', optional: false, variadic: false },
  ],
  bindings: {},
}

/**
 * Stack variadic inputs along a new axis at `dim`.
 * All inputs must match shape; output rank is input rank + 1.
 */
export const STACK_ENTRY = {
  name: 'Stack',
  module: '__utility__',
  framework: 'any',
  kind: 'stack',
  ctor: [
    {
      name: 'dim',
      type: 'int',
      default: 0,
      required: false,
    },
  ],
  inputs: [
    { name: 'xs', shape: ['...'], dtype: 'float', optional: false, variadic: true },
  ],
  outputs: [
    { name: 'out', shape: ['...'], dtype: 'float', optional: false, variadic: false },
  ],
  bindings: {},
}

/**
 * Split one side of an einops pattern into top-level groups. Each output item
 * is either a bare identifier ("b"), the rest token "...", a literal int
 * ("1"), or a normalized parenthesized group ("(h w)"). Throws on unbalanced
 * parens; whitespace-only sides return [].
 */
export function splitEinopsSide(side) {
  const s = String(side).trim()
  const out = []
  let i = 0
  while (i < s.length) {
    if (/\s/.test(s[i])) {
      i++
      continue
    }
    if (s[i] === '(') {
      const close = s.indexOf(')', i)
      if (close < 0) throw new Error('unbalanced "(" in einops pattern')
      const inner = s
        .slice(i + 1, close)
        .trim()
        .split(/\s+/)
        .filter(Boolean)
        .join(' ')
      out.push(`(${inner})`)
      i = close + 1
    } else {
      let j = i
      while (j < s.length && !/[\s()]/.test(s[j])) j++
      out.push(s.slice(i, j))
      i = j
    }
  }
  return out
}

/** Parse "b c h w -> b (h w) c" into { lhs, rhs }. */
export function parseEinopsPattern(pattern) {
  const idx = String(pattern ?? '').indexOf('->')
  if (idx < 0) throw new Error('einops pattern must contain "->"')
  return {
    lhs: splitEinopsSide(String(pattern).slice(0, idx)),
    rhs: splitEinopsSide(String(pattern).slice(idx + 2)),
  }
}

/** "h=8, w=16" -> Map { h:8, w:16 }. Silently skips malformed entries. */
export function parseLengthsString(s) {
  const out = new Map()
  if (!s) return out
  for (const part of String(s).split(/[,;\n]/)) {
    const m = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+)\s*$/.exec(part)
    if (m) out.set(m[1], Number(m[2]))
  }
  return out
}

/**
 * Resolve a single einops-side token to a shape value:
 *   - "..."   -> "..." (passes through normalize/freshen as the rest marker)
 *   - "5"     -> 5 (literal)
 *   - "h" with lengths.has("h") -> numeric value
 *   - "(h w)" with all components in lengths -> product of values
 *   - everything else -> the token text as a symbolic axis (groups keep parens)
 */
function einopsItemToToken(item, lengths) {
  if (item === '...') return '...'
  if (/^-?\d+$/.test(item)) return Number(item)
  if (!item.startsWith('(')) {
    return lengths.has(item) ? lengths.get(item) : item
  }
  const components = item.slice(1, -1).split(/\s+/).filter(Boolean)
  let product = 1
  let allKnown = true
  for (const c of components) {
    if (/^-?\d+$/.test(c)) {
      product *= Number(c)
    } else if (lengths.has(c)) {
      product *= lengths.get(c)
    } else {
      allKnown = false
      break
    }
  }
  return allKnown ? product : item
}

/** Source/sink node for einops.rearrange. */
export class RearrangeNode extends BlockNode {
  freshenedShape(portName, side) {
    let parts
    try {
      const { lhs, rhs } = parseEinopsPattern(this.values.pattern)
      parts = side === 'in' ? lhs : rhs
    } catch {
      return null // invalid pattern; validator will report it elsewhere
    }
    const lengths = parseLengthsString(this.values.lengths)
    const tokens = parts.map((p) => einopsItemToToken(p, lengths))
    return freshen(tokens, this.id)
  }
}

/** Sugar over tensor.reshape() / jnp.reshape(); numeric-only shape literal. */
export class ReshapeNode extends BlockNode {
  freshenedShape(portName, side) {
    if (side === 'in') {
      // Accept anything coming in - reshape doesn't care about input rank.
      return freshen(normalize(['...']), this.id)
    }
    const tokens = normalize(parseShapeString(this.values.shape))
    return freshen(tokens, this.id)
  }
}

/** Pick the right node class for a manifest entry. */
export function makeNode(entry) {
  if (entry.kind === 'input') return new InputNode(entry)
  if (entry.kind === 'output') return new OutputNode(entry)
  if (entry.kind === 'rearrange') return new RearrangeNode(entry)
  if (entry.kind === 'reshape') return new ReshapeNode(entry)
  return new BlockNode(entry)
}

/** Palette section for a manifest entry. */
export function paletteGroup(entry) {
  if (entry.module === '__builtin__') return 'built-in'
  if (entry.module === '__utility__') return 'utility'
  return entry.module.split('.').slice(-1)[0]
}

/** Group manifest entries by their submodule for the palette UI. */
export function groupByModule(entries) {
  const groups = new Map()
  for (const e of entries) {
    const tail = paletteGroup(e)
    if (!groups.has(tail)) groups.set(tail, [])
    groups.get(tail).push(e)
  }
  for (const list of groups.values())
    list.sort((a, b) => a.name.localeCompare(b.name))
  const order = { 'built-in': 0, utility: 1 }
  return [...groups.entries()].sort(([a], [b]) => {
    const oa = order[a] ?? 99
    const ob = order[b] ?? 99
    if (oa !== ob) return oa - ob
    return a.localeCompare(b)
  })
}
