// Run actual chapter scripts in a tiny DOM double; no browser, framework or copied formulas.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
class Element {
  constructor(value = '') { this.value = value; this.textContent = ''; this.innerHTML = ''; this.events = {}; this.children = []; }
  setAttribute(name, value) { this[name] = value; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(type, callback) { this.events[type] = callback; }
  fire(type) { assert.equal(typeof this.events[type], 'function'); this.events[type](); }
}
function load(chapter, container, defaults) {
  const elements = new Map(Object.entries(defaults).map(([selector, value]) => [selector, new Element(value)]));
  const root = new Element();
  root.querySelector = selector => { assert.ok(elements.has(selector), `unexpected selector ${selector}`); return elements.get(selector); };
  const document = { querySelector: selector => selector === container ? root : null, createElementNS: () => new Element() };
  const source = fs.readFileSync(path.join(__dirname, `../../chapters/${chapter}/demo.js`), 'utf8');
  vm.runInNewContext(source, { document, Math, Number, String, Object, Array });
  return { root, get: selector => elements.get(selector) };
}
const fit = load('01', '#neuron-fit', { '[data-w]': '0', '[data-b]': '0', svg: '', '[data-readout]': '', '[data-table]': '', '[data-reset]': '' });
assert.match(fit.get('[data-readout]').textContent, /MSE = 9\.0000/);
fit.get('[data-w]').value = '1.5'; fit.get('[data-b]').value = '0.5'; fit.root.fire('input');
assert.match(fit.get('[data-readout]').textContent, /MSE = 0\.7500/);
assert.match(fit.get('[data-table]').innerHTML, /<td>2<\/td><td>5<\/td><td>3\.50<\/td><td>-1\.50<\/td>/);
fit.get('[data-w]').value = '-3'; fit.get('[data-b]').value = '-3'; fit.root.fire('input');
assert.match(fit.get('[data-readout]').textContent, /MSE = 66\.0000/);
fit.get('[data-reset]').fire('click'); assert.match(fit.get('[data-readout]').textContent, /MSE = 9\.0000/);

const gradient = load('02', '#loss-gradient', { '[data-w]': '0', '[data-h]': '0.00001', svg: '', '[data-readout]': '', '[data-step]': '', '[data-reset]': '' });
assert.match(gradient.get('[data-readout]').textContent, /analytical gradient \[-8\.000000, -2\.000000\]/);
const discrepancy = () => Number(gradient.get('[data-readout]').textContent.match(/discrepancy=([^.]*(?:\.[^.]*)?)\.$/)[1]);
assert.ok(discrepancy() < 1e-6);
gradient.get('[data-w]').value = '1.3'; gradient.get('[data-w]').fire('change');
gradient.get('[data-h]').value = '1e-16'; gradient.get('[data-h]').fire('change');
assert.ok(discrepancy() > 0.1, 'tiny h should expose loss of precision at this representable setting');
gradient.get('[data-reset]').fire('click'); gradient.get('[data-step]').fire('click');
assert.match(gradient.get('[data-readout]').textContent, /w=0\.8000, b=0\.2000/);
assert.match(gradient.get('[data-readout]').textContent, /MSE=3\.520000/);

const classifier = load('04', '#classifier-evidence', { '[data-mode]': 'probability', '[data-probability]': '', '[data-threshold-panel]': '', '[data-logit]': '0', '[data-target]': '1', '[data-threshold]': '0.5', '[data-reset]': '', svg: '', '[data-readout]': '', '[data-rows]': '' });
assert.match(classifier.get('[data-readout]').textContent, /p=0\.500000.*BCE=0\.693147/);
classifier.get('[data-logit]').value = '8'; classifier.get('[data-target]').value = '0'; classifier.root.fire('input');
assert.match(classifier.get('[data-readout]').textContent, /p=0\.999665.*BCE=8\.000335/);
classifier.get('[data-logit]').value = '-8'; classifier.get('[data-target]').value = '1'; classifier.root.fire('input');
assert.match(classifier.get('[data-readout]').textContent, /p=0\.000335.*BCE=8\.000335/);
classifier.get('[data-mode]').value = 'threshold'; classifier.root.fire('input');
assert.match(classifier.get('[data-readout]').textContent, /TP=1, FP=1, TN=1, FN=1/);
classifier.get('[data-threshold]').value = '0.4'; classifier.root.fire('input');
assert.match(classifier.get('[data-readout]').textContent, /TP=2, FP=1, TN=1, FN=0/);
classifier.get('[data-threshold]').value = '1'; classifier.root.fire('input');
assert.match(classifier.get('[data-readout]').textContent, /TP=0, FP=0, TN=2, FN=2.*precision=undefined \(no predicted positives\)/);
classifier.get('[data-threshold]').value = '0'; classifier.root.fire('input');
assert.match(classifier.get('[data-readout]').textContent, /TP=2, FP=2, TN=0, FN=0/);
classifier.get('[data-reset]').fire('click');
assert.equal(classifier.get('[data-mode]').value, 'probability');
assert.equal(Number(classifier.get('[data-threshold]').value), 0.5);
assert.match(classifier.get('[data-readout]').textContent, /p=0\.500000.*BCE=0\.693147/);
console.log('01/02/04 demo calculations, control boundaries and deterministic resets pass.');
