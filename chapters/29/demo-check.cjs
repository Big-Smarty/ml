// Run: node chapters/29/demo-check.cjs
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const ctx = { document: { getElementById: () => null } };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(__dirname + '/demo.js', 'utf8'), ctx);
const plain = value => JSON.parse(JSON.stringify(value));
assert.deepEqual(plain(ctx.s06Dispatch(67)), { groups: 2, lanes: 128, unused: 61, last: 66 });
assert.deepEqual(plain(ctx.s06Dispatch(64)), { groups: 1, lanes: 64, unused: 0, last: 63 });
assert.deepEqual(plain(ctx.s06Reduction(0)), [3,1,4,1,5,9,2,6]);
assert.deepEqual(plain(ctx.s06Reduction(1)), [8,10,6,7]);
assert.deepEqual(plain(ctx.s06Reduction(2)), [14,17]);
assert.deepEqual(plain(ctx.s06Reduction(3)), [31]);
assert.deepEqual(plain(ctx.s06Tile(17)), { rounds: 2, padding: 15, a: 50, b: 83, c: 13 });
assert.equal(ctx.s06Tile(16).rounds, 1);
assert.deepEqual(plain(ctx.s06Transfers(800)), { upload: 164, parameters: 68, history: 3204, resident: 3436, eachStep: 57768 });
assert.equal(ctx.s06Transfers(1).resident, ctx.s06Transfers(1).eachStep);
console.log('S06 illustration: dispatch, reduction, tile addresses/barriers and transfer counts passed.');
