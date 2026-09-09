// Run: node tools/test_glossary.cjs. Exercise the actual shared-reader handlers.
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
class Element {
  constructor() {
    this.events = {}; this.dataset = {}; this.style = {}; this.attrs = {};
    this.hidden = true; this.offsetHeight = 80; this.hover = false; this.focused = false;
    const classes = new Set();
    this.classList = {add: value => classes.add(value), remove: value => classes.delete(value), contains: value => classes.has(value), toggle(value) {if (classes.has(value)) {classes.delete(value); return false;} classes.add(value); return true;}};
  }
  addEventListener(name, fn) {this.events[name] = fn;}
  fire(name, event = {}) {return this.events[name]?.(event);}
  setAttribute(name, value) {this.attrs[name] = value;}
  removeAttribute(name) {delete this.attrs[name];}
  matches() {return this.hover || this.focused;}
  getBoundingClientRect() {return {left: 100, bottom: 120};}
  focus() {this.focused = true; return this.fire('focus');}
}
const anchor = new Element(), preview = new Element(), main = new Element(), menu = new Element(), sidebar = new Element(), search = new Element();
anchor.dataset.term = 'demo';
const elements = {'#term-preview': preview, '#main': main, '#menu-button': menu, '#sidebar': sidebar, '#course-search': search};
const handlers = {};
const document = {documentElement: new Element(), body: new Element(),
  querySelector: selector => elements[selector] || null,
  querySelectorAll: selector => selector === 'a.term[data-term]' ? [anchor] : [],
  addEventListener: (name, fn) => {handlers[name] = fn;}};
const mobile = {matches: true, addEventListener() {}};
vm.runInNewContext(fs.readFileSync('site/assets/app.js', 'utf8'), {
  document, localStorage: {getItem: () => null, setItem() {}}, matchMedia: () => mobile,
  fetch: async () => ({json: async () => ({demo: {name: 'Example', definition: 'A definition.'}})}),
  innerWidth: 800, innerHeight: 600
});
(async () => {
  assert.equal(sidebar.inert, true);
  menu.fire('click');
  assert.equal(sidebar.inert, false); assert.equal(main.inert, true); assert.equal(search.focused, true);
  handlers.keydown({key: 'Escape'});
  assert.equal(sidebar.inert, true); assert.equal(main.inert, false); assert.equal(menu.focused, true);
  await anchor.focus();
  assert.equal(preview.hidden, false); assert.equal(preview.textContent, 'Example: A definition.');
  assert.equal(anchor.attrs['aria-describedby'], 'term-preview');
  anchor.fire('keydown', {key: 'Escape'});
  assert.equal(preview.hidden, true); assert.equal(anchor.attrs['aria-describedby'], undefined);
  anchor.focused = false; anchor.hover = true; await anchor.fire('mouseenter');
  assert.equal(preview.hidden, false);
  anchor.hover = false; anchor.fire('mouseleave'); assert.equal(preview.hidden, true);
  assert.equal(anchor.events.click, undefined, 'ordinary glossary links remain navigable');
  console.log('PASS: glossary focus/hover, Escape, ordinary links, and mobile navigation focus/inert state.');
})().catch(error => {console.error(error); process.exitCode = 1;});
