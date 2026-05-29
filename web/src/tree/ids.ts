/**
 * Id generation for the tree model. Kept separate so the pure model/signature
 * modules and their tests don't pull in `main.ts`.
 */

let _counter = 0

/**
 * A fresh, monotonic + random node id. Monotonic prefix keeps ids
 * deterministic-ish in order of creation; the random suffix avoids collisions
 * when importing on top of an existing forest.
 */
export function freshNodeId(prefix = 'n'): string {
  return `${prefix}${++_counter}_${Math.random().toString(36).slice(2, 8)}`
}
