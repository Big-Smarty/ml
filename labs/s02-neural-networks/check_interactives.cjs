// Run from any directory: node labs/s02-neural-networks/check_interactives.cjs
const assert=require('node:assert/strict');
const xor=require('../../chapters/07/demo.js');
const tape=require('../../chapters/08/demo.js');
const tensor=require('../../chapters/09/demo.js');
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-8,`${a} != ${b}`);
assert.deepEqual(xor.calculate('hidden',.5).map(r=>r.prediction),[0,1,1,0]);
assert.deepEqual(xor.calculate('hidden',1.5).map(r=>r.prediction),[0,0,0,0]);
assert.equal(xor.calculate('linear',.5).filter(r=>r.prediction===r.target).length,3);
assert.deepEqual(tape.calculate(3,0).gradients,[0,0,1]);
assert.deepEqual(tape.calculate(3,1).gradients,[1,1,1]);
assert.deepEqual(tape.calculate(3,2).gradients,[7,1,1]);
assert.deepEqual(tape.calculate(-2,2).gradients,[-3,1,1]);
assert.deepEqual(tensor.dense(1),[-1.5,3.5,-1.5,12.5]);
assert.deepEqual(tensor.dense(2),[-3.5,7.5,-3.5,25.5]);
for(const d of ['0','1','8','blank']){
 const r=tensor.digit(d,1);close(r.probabilities.reduce((a,b)=>a+b,0),1);
 if(d!=='blank')assert.equal(r.prediction,Number(d));
}
const rows=tensor.optimizer(1);close(rows[0].momentum,.8);close(rows[1].momentum,.62);
close(rows[0].adam,.9000000005);close(rows[1].m,.18);close(rows[1].v,.003996);
assert.ok(tensor.optimizer(0).every(r=>r.momentum===1 && r.adam===1 && r.m===0 && r.v===0));
console.log('Three interactive tools: deterministic reset, changed input, shape, softmax and optimizer calculations pass.');

// Run the actual renderers too: tiny probabilities must survive display formatting.
const fs = require('node:fs');
const vm = require('node:vm');
const CourseNumbers = require('../../site/assets/numbers.js');
function display(chapter, id, defaults) {
 const nodes = new Map();
 const get = selector => {
  if (!nodes.has(selector)) nodes.set(selector, {value: defaults[selector] || '', innerHTML: '', events: {}, addEventListener(name, fn) {this.events[name] = fn;}, setAttribute() {}});
  return nodes.get(selector);
 };
 const document = {querySelector: selector => selector === id ? {querySelector: get} : null};
 vm.runInNewContext(fs.readFileSync(require.resolve(`../../chapters/${chapter}/demo.js`), 'utf8'), {document, CourseNumbers});
 return get;
}
const xorDisplay = display('07', '#xor-hidden', {'[data-model]': 'hidden', '[data-threshold]': '3'});
assert.match(xorDisplay('[data-table]').innerHTML, /<mn>3.775<\/mn>.*<mn>11<\/mn>/, 'hidden sigmoid tail stays nonzero');
const tapeDisplay = display('08', '#autodiff-tape', {'[data-x]': '0.1'});
assert.match(tapeDisplay('svg').innerHTML, /m=x×x: 0.01</);
assert.doesNotMatch(tapeDisplay('[data-table]').innerHTML, /0.1000|0.0100/);
const tensorDisplay = display('09', '#tensor-digits', {'[data-stage]': 'digits', '[data-scale]': '2', '[data-digit]': '0'});
assert.match(tensorDisplay('[data-table]').innerHTML, /<msup><mn>10<\/mn>/, 'small digit probabilities use scientific notation');
tensorDisplay('[data-reset]').events.click();
assert.match(tensorDisplay('[data-readout]').innerHTML, /<mn>12.5<\/mn>/);
console.log('Three numeric renderers pass: compact labels, tiny sigmoid/softmax values and reset.');
