// Run from any working directory: node labs/s05-cpu/check_demos.cjs
const assert = require('node:assert/strict');
const gemm = require('../../chapters/25/demo.js');
for (const mode of ['ijk', 'ikj', 'tile']) {
  assert.equal(gemm.trace(mode).length, 18);
  assert.deepEqual(gemm.state(mode, 18).values, [30, 36, 42, 66, 81, 96]);
  assert.equal(new Set(gemm.trace(mode).map(v => `${v.r},${v.k},${v.c}`)).size, 18);
}
const mapper = require('../../chapters/27/demo.js');
assert.deepEqual(mapper.partition(7, 3), [{ start: 0, end: 3, sum: 14 }, { start: 3, end: 6, sum: 77 }, { start: 6, end: 7, sum: 49 }]);
for (const threads of [1, 3, 8]) {
  assert.equal(mapper.partition(7, threads).reduce((n, s) => n + s.end - s.start, 0), 7);
  assert.equal(mapper.partition(7, threads).reduce((n, s) => n + s.sum, 0), 140);
}
assert.deepEqual(mapper.lanes(19, 8).sums, [5, 6, 7, 8, 9, 10, 11, 12]);
for (const width of [4, 8, 16]) assert.equal(mapper.lanes(19, width).total, 95);
assert.deepEqual(mapper.rounding(), [1, 0]);
assert.match(mapper.present('threads', 3, 8, 'none').map, /<mn>26\.44<\/mn>/);
assert.match(mapper.present('rounding', 3, 8, 'none').output, /<mn>100000000<\/mn>/, 'the f32 grouping example keeps its exact integer');
assert.equal(mapper.dispatch('avx2'), 'scalar');
assert.equal(mapper.dispatch('both'), 'AVX2+FMA');
console.log('Both numerical tools: deterministic calculations passed');
