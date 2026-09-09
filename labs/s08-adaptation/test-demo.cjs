'use strict';
const assert = require('node:assert/strict');
const { inferenceBudgetS08: budget } = require('../../chapters/40/demo.js');
const four = budget(5, 4);
assert.equal(four.cacheBytes, 640);
assert.equal(four.fullRows, 30);
assert.equal(four.cachedRows, 10);
assert.equal(four.squareScoreBytes, 200);
// Row/tile workspace is for a single query/head; the square tensor covers both heads.
assert.equal(four.rowScoreBytes, 20);
assert.equal(four.tileScoreBytes, 8);
assert.deepEqual(four.codes, [-7, -2, 1, 7]);
assert.equal(four.weightBytes, 6);
assert.ok(Math.abs(four.error - 0.05714285714285716) < 1e-12);
const eight = budget(5, 8);
assert.deepEqual(eight.codes, [-127, -38, 25, 127]);
assert.equal(eight.weightBytes, 8);
assert.equal(eight.cacheBytes, four.cacheBytes);
assert.ok(Math.abs(eight.error - 0.0031496062992125984) < 1e-12);
const longer = budget(6, 4);
assert.equal(longer.cacheBytes, 768);
assert.equal(longer.fullRows, 42);
assert.equal(longer.cachedRows, 12);
assert.equal(budget(1, 4).tileScoreBytes, 4);
for (const count of [0, 13, 1.5, NaN, Infinity]) assert.throws(() => budget(count, 4), RangeError);
for (const bits of [0, 16, '4', NaN]) assert.throws(() => budget(5, bits), RangeError);
console.log('Chapter 40 demo: cache/work/score memory, signed quantization, storage/error and invalid controls pass');

// Exercise the generated MathML, control changes and reset without a browser dependency.
const fs = require('node:fs');
const vm = require('node:vm');
class Node {
  constructor(name = '#text', namespace = null, text = '') {
    this.name = name; this.namespace = namespace; this.children = []; this.text = text;
    this.attributes = {}; this.events = {};
  }
  append(...children) { this.children.push(...children.map(child => typeof child === 'string' ? new Node('#text', null, child) : child)); }
  replaceChildren(...children) { this.children = []; this.text = ''; this.append(...children); }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(name, callback) { this.events[name] = callback; }
  set textContent(value) { this.children = []; this.text = String(value); }
  get textContent() { return this.text + this.children.map(child => child.textContent).join(''); }
}
const positions = new Node('input'); positions.value = '5';
const precision = new Node('select'); precision.value = '4';
const result = new Node('p'); const picture = new Node('svg'); const reset = new Node('button');
const controls = { '[name="positions"]': positions, '[name="precision"]': precision, '[data-result]': result, '[data-cache]': picture, '[data-reset]': reset };
const document = {
  querySelector: selector => selector === '#inference-budget' ? { querySelector: name => controls[name] } : null,
  createElementNS: (namespace, name) => new Node(name, namespace),
  createTextNode: text => new Node('#text', null, text)
};
vm.runInNewContext(fs.readFileSync(require.resolve('../../chapters/40/demo.js'), 'utf8'), { document });
function descendants(node) { return [node, ...node.children.flatMap(descendants)]; }
const mathNodes = descendants(result).filter(node => node.name === 'math');
assert.equal(mathNodes.length, 14);
assert.ok(mathNodes.every(node => node.namespace === 'http://www.w3.org/1998/Math/MathML'));
assert.ok(descendants(result).filter(node => node.name === 'mn').every(node => /^\d+(\.\d+)?$/.test(node.textContent)));
assert.match(result.textContent, /K\/V cache 640 bytes/);
assert.match(result.textContent, /codes \[−7,−2,1,7\]/);
assert.match(result.textContent, /maximum weight error 0.057143/);
positions.value = '6'; positions.events.change();
assert.match(result.textContent, /K\/V cache 768 bytes/);
assert.match(result.textContent, /processes 42 layer-token rows; caching processes 12/);
precision.value = '8'; precision.events.change();
assert.match(result.textContent, /codes \[−127,−38,25,127\]/);
assert.match(result.textContent, /maximum weight error 0.003150/);
positions.value = '1.5'; positions.events.change();
assert.equal(result.textContent, 'Use an integer prefix length from 1 to 12 and select int8 or int4.');
reset.events.click();
assert.equal(positions.value, '5'); assert.equal(precision.value, '4');
assert.match(result.textContent, /K\/V cache 640 bytes/);
assert.equal(picture.children[0].name, 'title'); assert.equal(picture.children[1].name, 'desc');
console.log('Chapter 40 MathML DOM: namespace, numeric tokens, variation, invalid input and reset pass');
