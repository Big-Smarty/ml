// Dependency-free checks execute the actual scoped browser scripts with a tiny DOM stub.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
function load(chapter, id, defaults) {
  const nodes = new Map();
  const root = { querySelector(selector) {
    if (!nodes.has(selector)) nodes.set(selector, { value: defaults[selector] || '', textContent: '', innerHTML: '', listeners: {}, addEventListener(event, fn) { this.listeners[event] = fn; } });
    return nodes.get(selector);
  } };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, `../../chapters/${chapter}/demo.js`), 'utf8'), { document: { getElementById: key => key === id ? root : null } });
  return { node: key => root.querySelector(key), fire(key, event, value) { const n = root.querySelector(key); if (value !== undefined) n.value = value; n.listeners[event](); }, output: () => root.querySelector('[data-output]').textContent, chart: () => root.querySelector('[data-chart]').innerHTML };
}
const sampling = load(12, 'sampling-tool', { '#s12-mode': 'bootstrap', '#s12-unit': 'machine', '#s12-sharpness': '1' });
assert.match(sampling.output(), /\[0.5000, 0.8333\]/);
const initial = sampling.output();
sampling.fire('[data-action="draw"]', 'click');
assert.match(sampling.output(), /Seed 8/);
sampling.fire('[data-action="reset"]', 'click');
assert.equal(sampling.output(), initial);
sampling.fire('#s12-unit', 'change', 'row');
assert.match(sampling.output(), /18 paired rows/);
sampling.fire('#s12-mode', 'change', 'calibration');
assert.match(sampling.output(), /Brier 0.1859/);
assert.match(sampling.chart(), /<td>5<\/td><td>0.1431<\/td><td>0.2000/);
sampling.fire('#s12-sharpness', 'input', '2');
assert.match(sampling.output(), /accuracy 0.6667/);
assert.doesNotMatch(sampling.output(), /Brier 0.1859/);
const split = load(13, 'split-tool', {});
for (const [policy, train, valid, shared, result] of [['row',108,36,18,'26/36 = .722'],['group',96,48,0,'33/48 = .688'],['time',72,54,18,'36/54 = .667'],['strict',48,18,0,'12/18 = .667'],['leak',48,18,0,'18/18 = 1.000']]) {
  split.fire('#s13-policy', 'change', policy);
  assert.ok(split.output().includes(`training ${train} rows; validation ${valid}; shared machines ${shared}`));
  assert.ok(split.chart().includes(result));
}
split.fire('[data-action="reset"]', 'click');
assert.equal(split.node('#s13-policy').value, 'strict');
const geometry = load(14, 'geometry-tool', { '#s14-view': 'distance' });
assert.match(geometry.output(), /distances are 2.0000/);
geometry.fire('#s14-value', 'input', '2');
assert.match(geometry.output(), /distances are 5.0000/);
geometry.fire('#s14-view', 'change', 'tree');
assert.match(geometry.output(), /goes left/);
geometry.fire('#s14-value', 'input', '.9');
assert.match(geometry.output(), /goes right/);
geometry.fire('#s14-view', 'change', 'kernel');
assert.match(geometry.output(), /0.3679/);
geometry.fire('#s14-view', 'change', 'projection');
assert.match(geometry.output(), /lost norm 1.0000/);
geometry.fire('#s14-value', 'input', '45');
assert.match(geometry.output(), /lost norm 0.0000/);
geometry.fire('#s14-view', 'change', 'cluster');
assert.match(geometry.output(), /2.0000 and 2.0000/);
geometry.fire('#s14-value', 'input', '1');
assert.match(geometry.output(), /1.0000 and 2.0000/);
geometry.fire('#s14-value', 'input', '2.5');
assert.match(geometry.output(), /center 2 wins/);
geometry.fire('[data-action="reset"]', 'click');
assert.equal(geometry.node('#s14-view').value, 'distance');
assert.match(geometry.output(), /distances are 2.0000/);
console.log('Three actual scoped demos: bootstrap/calibration, five split cases, five geometry modes and deterministic resets passed.');
