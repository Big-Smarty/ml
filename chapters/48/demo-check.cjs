const assert = require('node:assert/strict');
const {calculate} = require('./demo.js');
const balanced = calculate(0, 1, 0.02);
assert.deepEqual(balanced.attempted, [4, 4, 4]);
assert.deepEqual(balanced.accepted, [4, 4, 4]);
assert.equal(balanced.dropped, 0);
assert.ok(Math.abs(balanced.rawBalance - 1) < 1e-12);
assert.ok(Math.abs(balanced.balance - 0.02) < 1e-12);
const low = calculate(0, 0.5, 0.02);
assert.equal(low.capacity, 2);
assert.deepEqual(low.accepted, [2, 2, 2]);
assert.equal(low.dropped, 6);
assert.equal(low.rawBalance, balanced.rawBalance);
const collapsed = calculate(4, 1, 0.02);
assert.deepEqual(collapsed.attempted, [12, 0, 0]);
assert.equal(collapsed.dropped, 8);
const zero = calculate(4, 1, 0);
assert.equal(zero.balance, 0);
assert.deepEqual(zero.rows, collapsed.rows);
const tie = calculate(2, 1, 0.02);
assert.deepEqual(tie.attempted, [12, 0, 0]);
assert.ok(balanced.mean.every(value => Math.abs(value - 1 / 3) < 1e-12));
assert.ok(Math.abs(collapsed.rawBalance - 3 * collapsed.mean[0]) < 1e-12);
assert.ok(Math.abs(calculate(4, 1, 0.04).balance - 2 * collapsed.balance) < 1e-12);
console.log('chapter 48 routing illustration: all deterministic checks passed');

// Render the actual demo with native MathML nodes and exercise its controls.
const fs = require('node:fs');
const vm = require('node:vm');
const CourseNumbers = require('../../site/assets/numbers.js');
class Node {
  constructor(name, namespace = null) { this.name = name; this.namespace = namespace; this.children = []; this.events = {}; this.text = ''; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; this.text = ''; }
  setAttribute() {}
  addEventListener(name, callback) { this.events[name] = callback; }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return this.text + this.children.map(child => typeof child === 'string' ? child : child.textContent).join(''); }
}
const elements = Object.fromEntries(['[data-experts]', '[data-tokens]', '[data-summary]', '[data-reset]'].map(selector => [selector, new Node('div')]));
const controls = ['concentration', 'factor', 'alpha'].map((name, index) => {
  const control = new Node('input'); control.value = [0, 1, 0.02][index]; control.dataset = {control: name};
  elements[`[data-control="${name}"]`] = control; elements[`[data-value="${name}"]`] = new Node('output');
  return control;
});
const document = {
  getElementById: name => name === 'routing-capacity' ? {querySelector: selector => elements[selector]} : null,
  createElement: name => new Node(name),
  createElementNS: (namespace, name) => new Node(name, namespace),
  createTextNode: text => String(text)
};
vm.runInNewContext(fs.readFileSync(require.resolve('./demo.js'), 'utf8'), {document, CourseNumbers});
assert.equal(elements['[data-value="alpha"]'].textContent, '0.02');
assert.equal(elements['[data-experts]'].children[0].children[4].textContent, '0.3333');
assert.equal(elements['[data-experts]'].children[0].children[4].children[0].namespace, 'http://www.w3.org/1998/Math/MathML');
assert.match(elements['[data-summary]'].textContent, /Unweighted balance 1; alpha-weighted balance 0.02/);
controls[0].value = 4; controls[0].events.input();
assert.equal(elements['[data-experts]'].children[0].children[1].textContent, '12');
assert.equal(elements['[data-experts]'].children[0].children[3].textContent, '8');
assert.match(elements['[data-tokens]'].textContent, /12→E0 dropped/);
elements['[data-reset]'].events.click();
assert.equal(elements['[data-experts]'].children[0].children[1].textContent, '4');
console.log('chapter 48 MathML DOM: concise numbers, exact counts, variation and reset passed');
