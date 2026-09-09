(() => {
  // Fixed MathML templates receive only formatted finite numbers from this illustration.
  const mn = value => {
    const match = String(value).match(/^(-?)(\d+(?:\.\d+)?)(?:e([+-]?\d+))?$/i);
    if (!match) throw new Error('Expected a finite mathematical value');
    const mantissa = `${match[1] ? '<mo>−</mo>' : ''}<mn>${match[2]}</mn>`;
    if (match[3] === undefined) return `<mrow>${mantissa}</mrow>`;
    const exponent = Number(match[3]);
    return `<mrow>${mantissa}<mo>×</mo><msup><mn>10</mn><mrow>${exponent < 0 ? '<mo>−</mo>' : ''}<mn>${Math.abs(exponent)}</mn></mrow></msup></mrow>`;
  };
  const math = body => `<math>${body}</math>`;
  const value = number => math(mn(number));
  const equal = (name, number) => math(`<mrow><mi${name.length > 1 ? ' mathvariant="normal"' : ''}>${name}</mi><mo>=</mo>${mn(number)}</mrow>`);
 const root=document.querySelector('#loss-gradient');if(!root)return;const q=s=>root.querySelector(s);let w=0,b=0;
 const loss=(w,b)=>[-2,-1,0,1,2].reduce((s,x)=>s+(w*x+b-(2*x+1))**2,0)/5;
 const node=(tag,attrs,text)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;return n;};
 function draw(){const h=Number(q('[data-h]').value),plus=loss(w+h,b),minus=loss(w-h,b),gw=(plus-minus)/(2*h),gb=(loss(w,b+h)-loss(w,b-h))/(2*h);q('[data-readout]').innerHTML=`${equal('w',w.toFixed(4))}, ${equal('b',b.toFixed(4))}; ${math(`<mrow><mi>L</mi><mo>(</mo><mi>w</mi><mo>+</mo><mi>h</mi><mo>)</mo><mo>=</mo>${mn(plus.toPrecision(16))}</mrow>`)}, ${math(`<mrow><mi>L</mi><mo>(</mo><mi>w</mi><mo>−</mo><mi>h</mi><mo>)</mo><mo>=</mo>${mn(minus.toPrecision(16))}</mrow>`)}; analytical gradient ${math(`<mrow><mo>[</mo>${mn((4*(w-2)).toPrecision(7))}<mo>,</mo>${mn((2*(b-1)).toPrecision(7))}<mo>]</mo></mrow>`)}; numerical ${math(`<mrow><mo>[</mo>${mn(gw.toPrecision(7))}<mo>,</mo>${mn(gb.toPrecision(7))}<mo>]</mo></mrow>`)}; ${equal('MSE',loss(w,b).toPrecision(7))}. Absolute weight-slope discrepancy: ${value(Math.abs(gw-4*(w-2)).toPrecision(4))}.`;
 const svg=q('svg'),sx=x=>55+(x+1)*100,sy=y=>240-y*10;svg.replaceChildren(node('title',{id:'gradient-title'},'MSE as weight varies'),node('desc',{id:'gradient-desc'},'Current bias is held fixed. Dashed finite-difference slope guide and solid quadratic loss. Text reports all numbers.'));
 for(let x=-1;x<=4;x++)svg.append(node('text',{x:sx(x),y:265,fill:'currentColor','text-anchor':'middle'},String(x)));
 for(let y=0;y<=20;y+=5)svg.append(node('text',{x:40,y:sy(y)+5,fill:'currentColor','text-anchor':'end'},String(y)));
 const d=Array.from({length:101},(_,i)=>{const x=-1+i*0.05;return `${i?'L':'M'}${sx(x)},${sy(loss(x,b))}`;}).join(' ');svg.append(node('path',{d,fill:'none',stroke:'var(--accent)','stroke-width':3}));svg.append(node('line',{x1:sx(w-0.4),x2:sx(w+0.4),y1:sy(loss(w,b)-0.4*gw),y2:sy(loss(w,b)+0.4*gw),stroke:'currentColor','stroke-width':2,'stroke-dasharray':'5 4'}),node('circle',{cx:sx(w),cy:sy(loss(w,b)),r:5,fill:'currentColor'}));
 }
 q('[data-w]').addEventListener('change',()=>{w=Number(q('[data-w]').value);b=0;draw();});q('[data-h]').addEventListener('change',draw);
 q('[data-step]').addEventListener('click',()=>{const gw=4*(w-2),gb=2*(b-1);w-=0.1*gw;b-=0.1*gb;draw();});q('[data-reset]').addEventListener('click',()=>{w=0;b=0;q('[data-w]').value='0';q('[data-h]').value='0.00001';draw();});draw();
})();
