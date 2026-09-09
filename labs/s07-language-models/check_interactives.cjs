// Run with Node.js; tests the exact pure calculations used by both browser tools.
const assert = require('node:assert/strict');
const token = require('../../chapters/33/demo.js').calculate;
const attention = require('../../chapters/35/demo.js').calculate;
const close = (a,b,tolerance=1e-9) => assert.ok(Math.abs(a-b)<tolerance, `${a} != ${b}`);
const byte = token();
assert.equal(byte.ids.length,11);assert.equal(byte.target,97);assert.equal(byte.vocabulary,256);
close(byte.loss,3.569828349260124);close(byte.probability,0.02816068706823159);
const scalar = token('scalar');assert.equal(scalar.ids.length,9);assert.equal(scalar.vocabulary,5);
const bpe = token('bpe');assert.deepEqual(bpe.ids,[99,256,195,169,32,99,256,195,169]);assert.equal(bpe.target,256);
assert.ok(token('bytes',3).loss<byte.loss);
const masked = attention();
close(masked.probabilities[0],0.7310585786300049);assert.equal(masked.probabilities[2],0);
close(masked.output[0],3.0757656854799804);close(masked.output[1],0.4621171572600098);
assert.deepEqual(masked,attention(1,100,true));
assert.deepEqual(masked.output,attention(1,-5,true).output);
assert.deepEqual(attention(0).output,[2,1]);
close(attention(1,100,false).output[0],20);
close(masked.normalized.reduce((a,b)=>a+b,0),0);
console.log('2 interactive calculations pass: units, target loss, one BPE merge, causal counterfactual, residual and normalization');

// Verify the actual readouts, including unmasked probabilities far below 0.000001.
const fs = require('node:fs');
const vm = require('node:vm');
const CourseNumbers = require('../../site/assets/numbers.js');
function display(chapter, id, defaults) {
  const nodes = new Map();
  const element = value => ({value, checked: false, innerHTML: '', children: [], style: {}, events: {},
    addEventListener(name, fn) {this.events[name] = fn;}, append(child) {this.children.push(child);}, replaceChildren() {this.children = [];}});
  const get = selector => {
    if (!nodes.has(selector)) nodes.set(selector, element(defaults[selector] || ''));
    return nodes.get(selector);
  };
  const document = {getElementById: key => key === id ? {querySelector: get} : null, createElement: () => element('')};
  vm.runInNewContext(fs.readFileSync(require.resolve(`../../chapters/${chapter}/demo.js`), 'utf8'), {document, CourseNumbers});
  return name => get(`[data-field="${name}"]`);
}
const tokenDisplay = display('33', 'token-loss-lab', {'[data-field="score"]': '2', '[data-field="mode"]': 'bytes'});
assert.match(tokenDisplay('result').innerHTML, /<mn>0.02816<\/mn>/);
assert.match(tokenDisplay('result').innerHTML, /<mn>3.57<\/mn>/);
const attentionDisplay = display('35', 'attention-lab', {'[data-field="query"]': '1', '[data-field="future"]': '100'});
assert.match(attentionDisplay('bars').children[0].innerHTML, /<mn>2.749<\/mn>.*<mn>43<\/mn>/);
attentionDisplay('causal').checked = true;
attentionDisplay('causal').events.change();
assert.match(attentionDisplay('bars').children[2].innerHTML, /<mn>0<\/mn>.*masked/);
console.log('Token/attention numeric renderers pass: compact values, tiny nonzero and exact masked zero.');
