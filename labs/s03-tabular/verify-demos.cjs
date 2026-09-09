// Dependency-free checks execute the actual scoped browser scripts with a tiny DOM stub.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const CourseNumbers = require('../../site/assets/numbers.js');
// Read mathematical display values from actual output markup while preserving fraction structure.
// The stub's innerHTML/textContent setters model the DOM dependency used by these demos.
function readableMath(markup) {
  return markup.replace(/<math\b[^>]*>([\s\S]*?)<\/math>/g, (_, formula) => formula
    .replace(/<mo>×<\/mo><msup><mn>10<\/mn>([\s\S]*?)<\/msup>/g, (_, power) => 'e' + power.replace(/<[^>]*>/g, '').replace('−', '-'))
    .replace(/<mfrac><mn>([^<]+)<\/mn><mn>([^<]+)<\/mn><\/mfrac>/g, '$1/$2')
    .replace(/<mo>,<\/mo>/g, ', ')
    .replace(/<mo>=<\/mo>/g, ' = ')
    .replace(/<[^>]*>/g, ''));
}
function domNode(value) {
  let markup = '';
  return {
    value, listeners: {},
    get innerHTML() { return markup; },
    set innerHTML(html) { markup = html; },
    get textContent() { return readableMath(markup).replace(/<[^>]*>/g, ''); },
    set textContent(text) { markup = String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;'); },
    addEventListener(event, fn) { this.listeners[event] = fn; }
  };
}
function load(chapter, id, defaults) {
  const nodes = new Map();
  const root = { querySelector(selector) {
    if (!nodes.has(selector)) nodes.set(selector, domNode(defaults[selector] || ''));
    return nodes.get(selector);
  } };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, `../../chapters/${chapter}/demo.js`), 'utf8'), { CourseNumbers, document: { getElementById: key => key === id ? root : null } });
  return { node: key => root.querySelector(key), fire(key, event, value) { const n = root.querySelector(key); if (value !== undefined) n.value = value; n.listeners[event](); }, output: () => root.querySelector('[data-output]').textContent, chart: () => readableMath(root.querySelector('[data-chart]').innerHTML) };
}
const sampling = load(12, 'sampling-tool', { '#s12-mode': 'bootstrap', '#s12-unit': 'machine', '#s12-sharpness': '1' });
assert.match(sampling.output(), /\[0.5, 0.8333\]/);
const initial = sampling.output();
sampling.fire('[data-action="draw"]', 'click');
assert.match(sampling.output(), /Seed 8/);
sampling.fire('[data-action="reset"]', 'click');
assert.equal(sampling.output(), initial);
sampling.fire('#s12-unit', 'change', 'row');
assert.match(sampling.output(), /18 paired rows/);
sampling.fire('#s12-mode', 'change', 'calibration');
assert.match(sampling.output(), /Brier 0.1859/);
assert.match(sampling.chart(), /<td>5<\/td><td>0.1431<\/td><td>0.2/);
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
assert.match(geometry.output(), /distances are 2/);
geometry.fire('#s14-value', 'input', '2');
assert.match(geometry.output(), /distances are 5/);
geometry.fire('#s14-view', 'change', 'tree');
assert.match(geometry.output(), /goes left/);
geometry.fire('#s14-value', 'input', '.9');
assert.match(geometry.output(), /goes right/);
geometry.fire('#s14-view', 'change', 'kernel');
assert.match(geometry.output(), /0.3679/);
geometry.fire('#s14-value', 'input', '5');
assert.match(geometry.output(), /4.54e-5/, 'small RBF similarity stays nonzero');
geometry.fire('#s14-view', 'change', 'projection');
assert.match(geometry.output(), /lost norm 1/);
geometry.fire('#s14-value', 'input', '45');
assert.match(geometry.output(), /lost norm 4.93e-32/);
geometry.fire('#s14-view', 'change', 'cluster');
assert.match(geometry.output(), /2 and 2/);
geometry.fire('#s14-value', 'input', '1');
assert.match(geometry.output(), /1 and 2/);
geometry.fire('#s14-value', 'input', '2.5');
assert.match(geometry.output(), /center 2 wins/);
geometry.fire('[data-action="reset"]', 'click');
assert.equal(geometry.node('#s14-view').value, 'distance');
assert.match(geometry.output(), /distances are 2/);
console.log('Three actual scoped demos: bootstrap/calibration, five split cases, five geometry modes and deterministic resets passed.');
