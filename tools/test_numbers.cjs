// Run with: node tools/test_numbers.cjs
const assert = require('node:assert/strict');
const numbers = require('../site/assets/numbers.js');
const cases = [
  [0, '0'], [-0, '0'], [2, '2'], ['2.0000', '2'], [0.693147, '0.6931'],
  [-0.0003353501, '−3.354 × 10⁻⁴'], [1.3e-43, '1.3 × 10⁻⁴³'],
  [1e-16, '1 × 10⁻¹⁶'], [0.001, '0.001'], [100000000, '100000000'],
  [1.234567e21, '1.235 × 10²¹'], [Number.MIN_VALUE, '4.941 × 10⁻³²⁴']
];
// A minimal native-node double also checks that both renderers produce the same MathML.
const document = { createElementNS(namespace, tag) {
  assert.equal(namespace, 'http://www.w3.org/1998/Math/MathML');
  return { tag, children: [], textContent: '', append(...children) { this.children.push(...children); } };
} };
const markup = node => `<${node.tag}>${node.textContent}${node.children.map(markup).join('')}</${node.tag}>`;
for (const [value, expected] of cases) {
  assert.equal(numbers.text(value), expected);
  assert.equal(markup(numbers.node(document, value)), `<math>${numbers.mathml(value)}</math>`);
  assert.doesNotMatch(numbers.mathml(value), /[eE][+-]?\d/);
}
assert.equal(numbers.text(8.999920000199998, 17), '8.999920000199998');
assert.equal(numbers.text(Number.MAX_SAFE_INTEGER), String(Number.MAX_SAFE_INTEGER));
for (const value of [NaN, Infinity, -Infinity, '', ' ', null, undefined, false, [], '<img>']) {
  assert.throws(() => numbers.mathml(value), TypeError);
}
for (const digits of [0, 18, 2.5]) assert.throws(() => numbers.text(1, digits), RangeError);
console.log('Number formatting: compact values, native exponents, precision, exact integers and invalid inputs pass.');
