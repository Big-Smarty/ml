const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const handlers = {};
class Element {
  constructor() { this.events={}; this.dataset={term:'demo'}; this.style={}; this.hidden=true; this.attrs={}; this.value=''; this.offsetWidth=280; this.offsetHeight=80; this.classList={toggle(){},remove(){},contains(){return false;}}; }
  addEventListener(name, fn) { this.events[name]=fn; }
  fire(name,event={}) { this.events[name]?.(event); }
  setAttribute(k,v) {this.attrs[k]=v;}
  removeAttribute(k) {delete this.attrs[k];}
  replaceChildren() {}
  append() {}
  getBoundingClientRect() {return {left:100,top:100,bottom:120};}
  focus() {document.activeElement=this;this.fire('focus');}
}
const anchor = new Element(), preview = new Element(), elements = {};
const document={body:{dataset:{},classList:new Element().classList},activeElement:null,
 querySelector(s){if(s==='#mark-complete'||s==='.chapter-link.current')return null;if(s==='#term-preview')return preview;return elements[s]??=new Element();},
 querySelectorAll(s){return s==='a.term'?[anchor]:[];},createElement(){return new Element();},createTextNode(s){return s;},addEventListener(n,f){handlers[n]=f;}};
const context={document,localStorage:{getItem(){return null;},setItem(){}},matchMedia(){return {matches:false};},
 fetch:async url=>({ok:true,json:async()=>url.includes('terms')?{demo:{name:'Example',definition:'A definition.'}}:[]}),
 addEventListener(n,f){handlers[n]=f;},innerWidth:800,innerHeight:600,setTimeout, navigator:{}};
vm.runInNewContext(fs.readFileSync('site/assets/app.js','utf8'),context);
setImmediate(()=>{
 function tap(){let prevented=false;anchor.fire('mouseenter');anchor.fire('pointerdown',{pointerType:'touch'});anchor.focus();anchor.fire('click',{preventDefault(){prevented=true;}});return prevented;}
 assert.equal(tap(),true,'first touch previews instead of navigating');
 assert.equal(preview.hidden,false);assert.equal(anchor.attrs['aria-describedby'],'term-preview');
 assert.equal(tap(),false,'second touch follows glossary link');
 handlers.keydown({key:'Escape'});assert.equal(preview.hidden,true);assert.equal(anchor.attrs['aria-describedby'],undefined);
 assert.equal(tap(),true,'dismissed preview opens again on touch');
 anchor.fire('blur');anchor.fire('pointerdown',{pointerType:'mouse'});anchor.focus();let prevented=false;anchor.fire('click',{preventDefault(){prevented=true;}});assert.equal(prevented,false,'mouse link remains normal');
 console.log('PASS: actual glossary handlers support touch preview, second-tap navigation, focus description, Escape, and ordinary mouse links.');
});
