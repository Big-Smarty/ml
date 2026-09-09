// Dependency-free numerical/browser-wiring smoke check for the two authored illustrations.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
function element(value = '') {
  return {
    value, textContent: '', children: [], events: {},
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = nodes; },
    addEventListener(name, fn) { this.events[name] = fn; }
  };
}
function load(chapter, id, defaults) {
  const nodes = new Map(Object.entries(defaults).map(([key, value]) => [key, element(value)]));
  const root = {querySelector(selector) {
    if (!nodes.has(selector)) nodes.set(selector, element());
    return nodes.get(selector);
  }};
  const document = {getElementById: name => name === id ? root : null, createElement: () => element()};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../chapters', chapter, 'demo.js'), 'utf8'), {document});
  return {
    text: () => root.querySelector('[data-output]').textContent,
    click: key => root.querySelector(key).events.click(),
    change(key, value) { const node = root.querySelector(key); node.value = value; node.events.change(); },
    node: key => root.querySelector(key)
  };
}
const conv = load('20', 'convolution-microscope', {'[data-geometry]': 'valid'});
assert.match(conv.text(), /output 3×3/);
assert.match(conv.text(), /nine products \[0.00, 0.00, 0.20, 0.00, 0.00, 0.40, 0.00, 0.00, 0.60\], sum 1.20, ReLU 1.20/);
assert.match(conv.text(), /max pool is 1.80, winner \(1,0\)/);
assert.match(conv.text(), /kernel-gradient contributions are \[0.00, 0.00, 0.24, 0.00, 0.00, 0.36, 0.00, 0.00, 0.48\]/);
conv.click('[data-next]'); conv.click('[data-next]');
assert.match(conv.text(), /Selected output \(0,2\).*sum -1.20, ReLU 0.00/);
conv.change('[data-geometry]', 'same');
assert.match(conv.text(), /padding 1, stride 1: output 5×5/);
conv.change('[data-geometry]', 'stride');
assert.match(conv.text(), /padding 1, stride 2: output 3×3/);
assert.match(conv.text(), /winning pre-activation receives 0.00/);
conv.click('[data-reset]');
assert.match(conv.text(), /Selected output \(0,0\).*sum 1.20/);
assert.equal(conv.node('[data-geometry]').value, 'valid');
const embedding = load('22', 'embedding-neighborhood', {'[data-query]': 'horizontal', '[data-scale]': '1'});
assert.match(embedding.text(), /candidate order after excluding observed A: B, C, D/);
assert.match(embedding.text(), /Recall@2 1, Precision@2 0.5, nDCG@2 1.0000/);
const row = index => embedding.node('[data-ranking]').children[index].children.map(cell => cell.textContent);
assert.deepEqual(row(1), ['B: held-out positive', '[0.8, 0.6]', '0.800', '0.800', '0.632']);
embedding.change('[data-query]', 'vertical');
assert.match(embedding.text(), /C, B, D.*rank 2.*nDCG@2 0.6309/);
embedding.change('[data-scale]', '3');
assert.deepEqual(row(2), ['C: explicit negative', '[0, 3]', '3.000', '1.000', '2.000']);
assert.match(embedding.text(), /reconstructs query code as \[0, 1, 1, -1\]/);
embedding.click('[data-reset]');
assert.match(embedding.text(), /Query \[1, 0\].*nDCG@2 1.0000/);
assert.equal(embedding.node('[data-scale]').value, '1');
console.log('PASS: convolution geometry/products/ReLU/pool/backward and embedding geometry/ranking/metrics/reset');
