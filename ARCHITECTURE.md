# Architecture

A guided tour of the **Srivastava Blocks Builder** codebase, with code snippets.

> Visual model builder for `models/blocks` with strict jaxtyping shape inference.
> Drag neural-network blocks onto a canvas, wire them up, and the editor refuses
> invalid connections, continuously type-checks the whole graph, and generates a
> runnable PyTorch / Flax `nn.Module`.

---

## 1. The big picture

There are three cooperating pieces:

```
models/blocks/            tools/build_manifest.py        web/  (Vite app)
┌────────────────┐        ┌──────────────────────┐       ┌────────────────────────┐
│ pytorch_blocks │  ───▶  │ introspect each block │ ───▶  │ public/manifests/*.json │
│ flax_blocks    │        │ (ctor, ports, shapes, │       │                         │
│ customblocks/  │        │  jaxtyping → axes)    │       │  Rete v2 graph editor   │
└────────────────┘        └──────────────────────┘       │  shape unifier + codegen│
                                                          └────────────────────────┘
```

1. **`models/blocks/`** — a git submodule of real PyTorch/Flax block libraries
   (the "Srivastava book of Blocks"). Each block is a normal `nn.Module` with
   `jaxtyping` annotations like `Float[Tensor, "B C_in H W"]`.
2. **`tools/build_manifest.py`** — a Python introspection script that turns each
   block into a JSON **manifest entry**: constructor params, input/output port
   shapes (as symbolic axis arrays), dtypes, and **bindings** (which ctor param
   sets which axis). Output lands in `web/public/manifests/{pytorch,flax}.json`.
3. **`web/`** — a Vite + plain-ES-module single-page app built on
   [Rete.js v2](https://retejs.org). It loads the manifest, renders a palette,
   and is the heart of the project.

The web app enforces correctness at **three** layers (see `web/README.md`):

| Layer | When | Where |
|-------|------|-------|
| Edge dry-run | every time you draw a wire | `unify.js` via `validator.dryRunEdge` |
| Whole-graph validation | after every change (debounced) | `validator.validate` |
| Codegen | on "Generate" | `codegen.generate` (topo-sort detects cycles) |

---

## 2. Repository layout

```
models/blocks/          # submodule: pytorch_blocks / flax_blocks libraries
models/customblocks/    # user-authored blocks (pytorch/ and flax/)
tools/
  build_manifest.py     # blocks  -> public/manifests/*.json   (the contract)
  shape_runner.py       # local HTTP server: runs traced codegen for real shapes
  fetch_block_diagrams.py / generate_mermaid_blocks.py   # docs/diagrams
diagrams/               # per-block Mermaid architecture diagrams (pulled in repo)
web/
  index.html            # toolbar + #palette + #editor + #sidebar layout
  public/manifests/*.json   # generated block catalog
  public/block_info.json    # per-block docs (description, mermaid) for Info tab
  src/                  # the application — see §4
  test-*.mjs            # puppeteer / node test suites (see §8)
```

The manifest is the **contract** between Python and JS. Schema (from
`tools/build_manifest.py`):

```python
{
  "name": "ConvBlock",
  "module": "pytorch_blocks.core_blocks",
  "framework": "pytorch",
  "kind": "module" | "function",
  "ctor": [{"name": "in_ch", "type": "int", "default": null, "required": true}, ...],
  "inputs":  [{"name": "x",   "shape": ["B", "C_in", "H", "W"], "dtype": "float", ...}],
  "outputs": [{"name": "out", "shape": ["B", "C_out", "H_out", "W_out"], "dtype": "float"}],
  "bindings": {"C_in": "in_ch", "C_out": "out_ch"}   # axis  <-  ctor param
}
```

`bindings` is the magic link: setting `out_ch = 256` in the inspector binds the
axis `C_out` to `256`, which then flows through the unifier into every
downstream block.

---

## 3. The data model in one paragraph

A **node** is a Rete `ClassicPreset.Node` wrapping a manifest entry (`nodes.js`).
A **shape** is an array of *tokens* — literal integers (`128`), symbolic axis
variables (`"B"`, `"C_out"`), or the variadic rest marker `"..."` (`shape.js`).
A node's axes are *freshened* per instance (`"B"` → `"B#<nodeId>"`) so two
copies of the same block don't collide. The **substitution** (`sub`) is a `Map`
recording every axis's resolved value; the **unifier** grows it edge by edge
(`unify.js`). The **validator** seeds `sub` from ctor `bindings`, unifies all
edges, and reports errors (`validator.js`). **Groups** are subgraphs collapsed
behind a *facade* node; their **tag** drives weight-sharing and their **name**
drives the generated Python class. **Codegen** topologically sorts the graph and
emits a `nn.Module` (`codegen.js`).

---

## 4. The `web/src/` modules

Ordered roughly bottom-up (smallest/most foundational first).

| File | Lines | Role |
|------|------:|------|
| `tagSync.js` | 27 | helpers: are two nodes the same "tag family"? copy ctor values |
| `shape.js` | 94 | the token model: normalize / freshen / resolve / pretty-print shapes |
| `unify.js` | 130 | Hindley-Milner-style shape unification over a substitution Map |
| `groupBoundary.js` | 142 | group facade "boundary signatures" (must match across peers) |
| `runtime.js` | 177 | talk to `shape_runner.py` for real PyTorch forward-pass shapes |
| `tagAtlas.js` | 229 | single source of truth for everything sharing a tag |
| `validator.js` | 287 | whole-graph validation + single-edge dry-run |
| `selection.js` | 355 | pan / zoom / marquee / keyboard selection on the canvas |
| `nodes.js` | 810 | `BlockNode` / `GroupNode` classes + built-in entries + `makeGroupEntry` |
| `ui.js` | 842 | palette, inspector, diagnostics, code dialog (all DOM rendering) |
| `codegen.js` | 1414 | graph → runnable Python (`nn.Module` / `nnx.Module`) |
| `main.js` | 2987 | the orchestrator: bootstrap, pipes, groups, clipboard, history, autosave |

---

## 5. Shape inference — `shape.js` + `unify.js` + `validator.js`

This is the type system. It's a small Hindley-Milner-style unifier.

### 5.1 Shapes are arrays of tokens (`shape.js`)

A token is a **literal** (`Number`), a **variable** (`String` like `"B"`), or the
**rest** marker `"..."`. `normalize` coerces manifest strings to numbers where
possible; `freshen` scopes a variable to a node instance so two nodes' `B` axes
are distinct until an edge unifies them:

```js
// shape.js
export function freshen(shape, nodeId) {
  const tag = `#${nodeId}`
  return shape.map((t) => (isVariable(t) ? `${t}${tag}` : t))   // "B" -> "B#node12"
}

// resolve walks the substitution chain to a terminal literal/variable (cycle-safe)
export function resolve(tok, sub) {
  let cur = tok
  const seen = new Set()
  while (sub && sub.has(cur) && !seen.has(cur)) {
    seen.add(cur)
    cur = sub.get(cur)
  }
  return cur
}
```

### 5.2 The substitution `sub`

`sub` is a `Map<string, number|string>` — the running type environment:

```
sub = Map {
  "C_out#node1" => 256,          // bound to a literal (from a ctor param)
  "B#node2"     => "B#node1",    // aliased to another variable
}
```

`resolve("B#node2", sub)` follows the chain to its terminal value.

### 5.3 Unification (`unify.js`)

`unifyShape(a, b, sub)` walks two token arrays in lockstep and mutates `sub`;
`unifyToken` is the core constraint:

```js
// unify.js — unifyToken (a, b already walked to their current binding)
if (isRest(a) || isRest(b)) return                 // rest matches anything as a single axis
if (isLiteral(a) && isLiteral(b)) {
  if (a !== b) throw new UnifyError(`axis literal mismatch: ${a} vs ${b}`)  // 128 vs 256 → fail
  return
}
if (isVariable(a) && isLiteral(b)) { bind(a, b, sub); return }   // bind variable -> literal
if (isLiteral(a) && isVariable(b)) { bind(b, a, sub); return }
if (isVariable(a) && isVariable(b)) {              // var vs var: alias newer -> older
  if (a < b) bind(b, a, sub); else bind(a, b, sub) // so the substitution forms a DAG
}
```

The rest token `"..."` (`isRest`) is a wildcard; in the full-array `unifyShape`,
when one side has a `...`, the known-length prefix and suffix are anchored and
the middle axes are absorbed.

### 5.4 The validator (`validator.js`)

`validate(editor)` does the whole-graph pass:

1. For every node, `applyParamBindings(sub)` writes ctor → axis bindings (this is
   where `out_ch=256` becomes `C_out#node1 → 256`):

   ```js
   // nodes.js — BlockNode.applyParamBindings
   for (const [axis, paramName] of Object.entries(this.entry.bindings || {})) {
     const v = Number(this.values[paramName])
     if (Number.isInteger(v)) sub.set(`${axis}#${this.id}`, v)
   }
   ```

2. For every connection, unify the producer's freshened output with the
   consumer's freshened input — accumulating into the *same* `sub`.
3. Collect dtype mismatches, tag conflicts, unconnected required inputs.
4. Return `{ ok, errors, warnings, sub, portShapes }` (the last drives hover/inspector shapes).

`dryRunEdge` is the same idea but on a **clone**, so drawing a wire can be
vetoed without mutating live state:

```js
// validator.js — dryRunEdge (gates connectioncreate in main.js)
const { sub } = validate(editor)
const trial = cloneSub(sub)
try {
  unifyShape(outShape, inShape, trial)
  return { ok: true }
} catch (e) {
  return { ok: false, reason: e.message }
}
```

It is wired into the editor as a connection pipe (`main.js`):

```js
// main.js — bootstrap()
editor.addPipe((context) => {
  if (context.type === 'connectioncreate') {
    const check = dryRunEdge(editor, srcNode, sourceOutput, tgtNode, targetInput)
    if (!check.ok) { flashDiagnostic(`refused edge: ${check.reason}`); return /* cancel */ }
  }
  return context
})
```

---

## 6. Nodes — `nodes.js`

`BlockNode extends ClassicPreset.Node` adds three things to a manifest entry:
`entry` (frozen manifest), `values` (per-instance ctor params, edited by the
inspector), and `tag` (a free-form label that doubles as a **weight-sharing
identity**):

```js
// nodes.js — BlockNode
this.tag = ''         // two ConvBlocks tagged "down1" -> one self.down1 in codegen
this.groupId = null   // set when this node belongs to a collapsed subgraph
this.values = Object.fromEntries(entry.ctor.map((p) => [p.name, p.default]))
```

Constructor params can be *exposed as input ports* (`exposeParam`) so a
`Constant` node can drive them — that's the `__param__<name>` machinery.

`nodes.js` also defines the **built-in** entries that aren't real Python blocks
(`Input`, `Output`, `Constant`, `LearnableTensor`, `Rearrange`, `Reshape`,
`Concat`, `Stack`, `Pool2d`, `Upsample`) and — importantly — `makeGroupEntry`,
which turns a computed boundary into a facade node's ports + `portMap` (see §7).

---

## 7. Groups — `groupBoundary.js` + `makeGroupEntry` + the sync in `main.js`

A **group** is a subgraph you collapse behind a single **facade** node. Two
orthogonal identities matter, and conflating them caused real bugs:

> **Group NAME → generated Python class. Group TAG → weight-shared instance.**

- **Name** (`g.name`): codegen emits **one `class <Name>`** per unique name. Two
  groups named `Encoder` dedupe to a single class body (first wins).
- **Tag** (`g.facadeTag`): selects the `self.<attr>` instance; same tag = shared
  weights.

### 7.1 Boundary signatures (`groupBoundary.js`)

A facade's **boundary** is which child ports it exposes (shapes, dtypes,
optionality, params). A **signature** is the comparable, node-id-stripped form
used to check that peer groups have matching interfaces:

```js
// groupBoundary.js — shapeKey strips the "#nodeId" freshening so two
// structurally-identical facades from different children compare equal
function shapeKey(shape) {
  return JSON.stringify((shape ?? ['...']).map((t) => {
    const s = String(t); const h = s.indexOf('#')
    return typeof t === 'number' ? t : (h >= 0 ? s.slice(0, h) : s)
  }))
}

export function boundarySignaturesMatch(a, b) {
  return portListsMatch(a.inputs, b.inputs)   // count + shapeKey + dtype + optional
      && portListsMatch(a.outputs, b.outputs)
      && paramListsMatch(a.params, b.params)
}
```

### 7.2 The facade entry (`makeGroupEntry` in `nodes.js`)

`makeGroupEntry(groupId, name, boundary)` builds the facade's `in0/in1/…` and
`out0/…` ports and a `portMap` that links each facade port back to its child
port — the "see-through" map used by the validator and codegen:

```js
// nodes.js — makeGroupEntry
const inputs = boundary.inputs.map((b, i) => ({
  name: `in${i}`, shape: b.shape ?? ['...'], dtype: b.dtype ?? 'any',
  optional: Boolean(b.optional), variadic: false,           // <- optionality preserved
}))
return {
  name: name || 'Group', kind: 'group', module: '__group__', groupId,
  inputs, outputs, ctor: [], bindings: {},
  portMap: {
    inputs:  boundary.inputs.map((b, i)  => ({ facadePort: `in${i}`,  childNodeId: b.childNodeId, childPort: b.childPort, shape: b.shape })),
    outputs: boundary.outputs.map((b, i) => ({ facadePort: `out${i}`, childNodeId: b.childNodeId, childPort: b.childPort, shape: b.shape })),
    params:  (boundary.params||[]).map((b)=> ({ facadePort: `__param__${b.paramName}`, childNodeId: b.childNodeId, childPort: b.childPort, paramName: b.paramName, paramType: b.paramType ?? 'int' })),
  },
}
```

> ⚠️ The saved `portMap` only stores **shape**, not `dtype`/`optional`. When a
> facade is rebuilt on import or paste, those must be re-derived from the live
> child port (`childPortInterface` in `main.js`) — otherwise an optional child
> port (e.g. an attention `mask`) wrongly becomes a *required* facade input.

### 7.3 Structural sync across peers (`main.js`)

When two groups are "structural peers" (**same name OR same tag**), their
internals are kept identical so the single generated class is correct for every
instance:

```js
// main.js — structural peers share the generated class, so internals must match
function groupsAreStructuralPeers(a, b) {
  const an = groupName(a).toLowerCase(), bn = groupName(b).toLowerCase()
  if (an && an === bn) return true                    // same NAME  -> same class
  const at = groupTag(a).toLowerCase(),  bt = groupTag(b).toLowerCase()
  return Boolean(at && at === bt)                     // same TAG   -> shared weights
}
```

`syncStructuralGroupPeers(gid)` picks the *fullest* peer as canonical and mirrors
its children, internal edges and boundary onto the others. The child-to-child
mapping (`buildSourceToPeerChildMap`) pairs children **positionally within each
tag bucket, in topological order** — important because a group can legitimately
hold several children with the same tag, and a naive one-per-tag map produced
spurious cyclic edges on copy.

---

## 8. Codegen — `codegen.js`

Graph → runnable Python. The pipeline:

1. **`partitionByGroup`** splits nodes into per-group buckets + top-level nodes.
2. **`groupClassName`** assigns a Python class name from the group **name**
   (`Encoder`, not `Encoder_ab12`); duplicates dedupe.
3. For each unique class, **`buildSubgraphView`** fabricates a standalone graph
   (synthetic `Input`/`Output` nodes for each boundary port) and **`emitClassBody`**
   emits it.
4. **`emitClassBody`** for the main graph instantiates the subclasses.

```js
// codegen.js — one class per group NAME; same-name groups reuse the body
const classNames = new Map()
for (const [gid, facade] of facadesByGid) classNames.set(gid, groupClassName(facade.entry.name, gid))
...
for (const [gid, facade] of facadesByGid) {
  const cls = classNames.get(gid)
  if (emittedClassNames.has(cls)) continue   // duplicate group -> reuse first class
  emittedClassNames.add(cls)
  emitClassBody(subLines, view.nodes, view.connections, framework, cls, classNames, options, { facade, usedDtypes })
}
```

`planGraph` allocates two namespaces — `self.<attr>` instances vs `forward()`
locals — and is where **tags collapse to one attribute**:

```js
// codegen.js — planGraph: same tag (non-empty) -> a single self.<attr>
// (so two ConvBlocks tagged "down1" share one weight-bearing instance,
//  but get distinct call-site locals down1 / down1_2)
```

`buildCallExpr` maps each node kind to a Python expression (`self.attr(...)` for
modules/groups, `rearrange(x, "...")`, `torch.cat([...], dim=)`, pooling, etc.).
Dangling **required** inputs are emitted as a valid `=None` arg plus a *trailing*
TODO comment on the statement (an inline `#` comment inside the call would
comment out the closing paren). Optional facade inputs get a `= None` **default**
in the subclass `forward` signature — only for the trailing run, since Python
requires defaulted params to be a suffix.

A trace mode (`options.trace`) instruments every call to record real tensor
shapes — that output is sent to `shape_runner.py`.

---

## 9. Tags & the Tag Atlas — `tagSync.js` + `tagAtlas.js`

The **tag atlas** (`tagAtlas.js`) is the single source of truth for everything
sharing a tag: canonical ctor values, exposed params, and members (node ids or
group ids). When you retag a node it *adopts* the family's canonical values;
when you edit a tagged node's param it *records* the change to all peers.
`tagSync.js` holds the tiny predicates:

```js
// tagSync.js
export function nodesInSameTagFamily(a, b) {
  const key = nodeTagKey(a?.tag)
  return Boolean(key) && key === nodeTagKey(b?.tag)
      && a.entry?.kind === b.entry?.kind && a.entry?.name === b.entry?.name
}
```

This is distinct from group structural sync (§7.3): tag-family sync keeps
*values* aligned across weight-shared instances; group sync keeps group
*structure* aligned.

---

## 10. The orchestrator — `main.js`

`main.js` owns the live editor `state` (framework, entries, `groups` map,
`tagAtlas`, clipboard, history) and wires everything together in `bootstrap()`:

- builds the Rete editor + Area + Connection + Lit render plugins,
- installs the **connectioncreate** pipe (edge dry-run, §5.4),
- installs a **structural-signal** pipe that debounces validation + autosave on
  any node/edge add/remove,
- loads the manifest and restores the last autosave.

Notable subsystems living here:

- **Groups lifecycle** — `groupSelected`, `collapseGroup`, `expandGroup`,
  `addNodesToGroup`, plus the peer structural sync (§7.3).
- **Clipboard** — `copySelection` / `pasteClipboard` / `duplicateSelection`.
  Paste re-maps ids, rebuilds facades (re-deriving dtype/optional from children),
  restores intra-selection edges, and **selects the pasted nodes**.
- **Undo / redo** — snapshot-based: each settled change records
  `getGraphData()`; `undo`/`redo` replay a snapshot through `importGraph`. This
  is used (rather than a rete-history plugin) because editor state also lives in
  `state.groups`/tags/collapse flags, which only the full serializer captures.

  ```js
  // main.js — snapshot history (Ctrl+Z / Ctrl+Shift+Z)
  function recordHistory() {
    const snap = snapshotGraph()                  // JSON of getGraphData()
    if (snap === h.stack[h.index]) return         // no-op if unchanged
    h.stack = h.stack.slice(0, h.index + 1); h.stack.push(snap); h.index++
  }
  ```

- **Autosave** — debounced `localStorage` write of the full graph; restored on
  reload.

`getGraphData()` / `importGraph()` are the serialization pair (also the undo and
autosave substrate) and capture nodes (with positions, tags, exposed params),
connections, and the `groups` descriptors.

---

## 11. UI — `ui.js`

Pure DOM rendering, no framework. Three regions (`index.html`): the left
`#palette`, the centre `#editor` (Rete canvas), and the right `#sidebar` with
`#diagnostics`, the **Inspector**, and the runtime panel.

`renderInspector` builds the **Params** tab (one control per ctor param) and an
**Info** tab (block docs + Mermaid diagram from `block_info.json`). A key detail:
when the *same* node stays selected across re-renders it takes a fast path that
**reconciles each control's value with `node.values`** (so programmatic changes —
paste, inferred `in_ch`, tag-sync — show up) while **skipping the control the
user is actively editing** so the caret isn't disturbed:

```js
// ui.js — same-node fast path: update controls without stealing focus
for (const param of node.entry.ctor ?? []) {
  const ctrl = paramsPanel.querySelector(`[id="ctrl-${node.id}-${param.name}"]`)
  if (!ctrl || ctrl === document.activeElement) continue
  const next = String(controlDisplayValue(param, node.values[param.name]))
  if (String(ctrl.value) !== next) ctrl.value = next
}
```

The generated-code dialog (`showCode`) renders each line as a `<span>` so a CSS
counter draws a line-number gutter — the numbers are `::before` content, so they
never leak into Copy or text selection.

---

## 12. Runtime shape checking — `runtime.js` + `tools/shape_runner.py`

Symbolic inference can't compute everything (e.g. a conv's `H_out` depends on
padding/stride). For ground truth, `runtime.js`:

1. checks the graph is "concrete enough" (`isFullyConcrete`),
2. builds concrete input tensors (back-solving axes from `sub`),
3. asks `codegen.generate(..., { trace: true })` for instrumented code,
4. POSTs it to the local `tools/shape_runner.py` HTTP server (`RUNNER_URL`),
5. gets back each port's real `tensor.shape` and overlays them in the UI.

---

## 13. Tests

All under `web/`, run with `node test-*.mjs` (most spin up headless puppeteer
against `npm run dev` on port 5173):

| Script | Covers |
|--------|--------|
| `test:logic` | pure-JS shape parse + unify + codegen (no browser) |
| `test:e2e` | end-to-end editor smoke |
| `test:info-tab` | Info tab + Mermaid + collapsibles |
| `test:runtime` | traced codegen + `shape_runner.py` |
| `test:autosave` | localStorage save/restore |
| `test:tags-clipboard` | tags + copy/paste/duplicate + shared-tag codegen |
| `test:groups` | grouping/collapse/expand + facade ports |
| `test:group-copy-cycle` | duplicate-tag group copy must not create cyclic edges |
| `test:group-name-sync` | same-name (different-tag) groups sync structurally |
| `test:group-optional-ports` | optional child ports stay optional through import |
| `test:history-paste-select` | undo/redo + paste keeps selection |
| `test:inspector-reactivity` | params reflect programmatic value changes |

---

## 14. Regenerate & run

```bash
# 1. (Re)build the manifest from the block libraries (needs torch/jax/flax/jaxtyping)
python tools/build_manifest.py

# 2. Web app
cd web
npm install
npm run dev          # http://127.0.0.1:5173/
npm run build        # static bundle in web/dist/

# 3. (optional) runtime shape checks
python tools/shape_runner.py   # then click "Run shape check" in the UI
```

---

## 15. Reading order for newcomers

1. `web/README.md` — the user-facing intent and the three correctness layers.
2. `shape.js` then `unify.js` — the type system in ~220 lines.
3. `validator.js` — how edges + ctor bindings become errors.
4. `nodes.js` (`BlockNode`, `makeGroupEntry`) — the in-memory node model.
5. `codegen.js` (`generate`, `planGraph`, `emitClassBody`) — graph → Python.
6. `main.js` `bootstrap()` and the groups/clipboard/history sections — how it's
   all driven.
