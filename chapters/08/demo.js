(() => {
  'use strict';
  // Fixed MathML templates; dynamic values are finite numbers, never input markup.
  const math = body => '<math xmlns="http://www.w3.org/1998/Math/MathML">'+body+'</math>';
  const scalar = value => { const text=String(value); return text.startsWith('-')?'<mrow><mo>−</mo><mn>'+text.slice(1)+'</mn></mrow>':'<mn>'+text+'</mn>'; };
  const number = value => math(scalar(value));
  const vector = values => math('<mrow><mo>[</mo>'+values.map(scalar).join('<mo>,</mo>')+'<mo>]</mo></mrow>');
  const formula = {"x": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>x</mi></mrow></math>", "y": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>y</mi></mrow></math>", "m": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>m</mi></mrow></math>", "mul": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>m</mi><mo>=</mo><mi>x</mi><mo>×</mo><mi>x</mi></mrow></math>", "add": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mi>y</mi><mo>=</mo><mi>m</mi><mo>+</mo><mi>x</mi></mrow></math>", "dym": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mfrac><mrow><mi>d</mi><mi>y</mi></mrow><mrow><mi>d</mi><mi>m</mi></mrow></mfrac><mo>=</mo><mn>1</mn></mrow></math>", "dyx": "<math display=\"inline\" xmlns=\"http://www.w3.org/1998/Math/MathML\"><mrow><mfrac><mrow><mi>d</mi><mi>y</mi></mrow><mrow><mi>d</mi><mi>x</mi></mrow></mfrac></mrow></math>"};
  function calculate(x,step) {
    return {values:[x,x*x,x*x+x],gradients:[step===0?0:step===1?1:1+2*x,step===0?0:1,1]};
  }
  if (typeof module === 'object') module.exports={calculate};
  if (typeof document === 'undefined') return;
  const root=document.querySelector('#autodiff-tape');if(!root)return;
  const input=root.querySelector('[data-x]'),next=root.querySelector('[data-step]');let step=0;
  function render(){
    const x=Number(input.value);
    if(!input.value || !Number.isFinite(x) || x< -5 || x>5){root.querySelector('[data-status]').innerHTML='Enter '+formula.x+' between '+number(-5)+' and '+number(5)+'.';return;}
    const {values,gradients}=calculate(x,step);
    root.querySelector('[data-status]').innerHTML=[`Forward complete. Seed ${formula.y} with ${number(1)}; the adjoints of ${formula.x} and ${formula.m} start at ${number(0)}.`,`Addition: ${formula.dym} and the direct ${formula.dyx} contribution is ${number(1)}.`,`Multiplication: add ${number(x)} twice to ${formula.x}. Final ${math('<mrow><mfrac><mrow><mi>d</mi><mi>y</mi></mrow><mrow><mi>d</mi><mi>x</mi></mrow></mfrac><mo>=</mo><mn>1</mn><mo>+</mo><mo>(</mo>'+scalar(x)+'<mo>)</mo><mo>+</mo><mo>(</mo>'+scalar(x)+'<mo>)</mo><mo>=</mo>'+scalar(gradients[0])+'</mrow>')}.`][step];
    root.querySelector('svg').setAttribute('viewBox','0 0 580 195');
    root.querySelector('svg').innerHTML=`<title id="tape-title">Shared scalar graph at reverse step ${step}</title><desc id="tape-desc">x has value ${values[0]} and adjoint ${gradients[0]}. Its two multiplication uses each contribute ${step===2?x:0}; the direct addition path contributes ${step>0?1:0}. m has value ${values[1]} and adjoint ${gradients[1]}. y has value ${values[2]} and adjoint one. The table provides the same values.</desc><defs><marker id="tape-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="currentColor"/></marker></defs><path d="M115 83Q170 50 235 83M115 107Q170 140 235 107M355 95H450M80 65Q265 -15 500 65" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#tape-arrow)"/><text x="140" y="53" fill="currentColor" font-size="12">use 1: +${step===2?x:0}</text><text x="140" y="148" fill="currentColor" font-size="12">use 2: +${step===2?x:0}</text><text x="275" y="24" fill="currentColor" font-size="12">direct x path: +${step>0?1:0}</text>${[55,285,500].map((cx,i)=>`<rect x="${cx-50}" y="65" width="110" height="66" rx="5" fill="none" stroke="currentColor"/><text x="${cx-42}" y="87" fill="currentColor" font-size="13">${['x','m=x×x','y=m+x'][i]}: ${values[i].toFixed(2)}</text><text x="${cx-42}" y="111" fill="currentColor" font-size="13">adjoint: ${gradients[i].toFixed(2)}</text>`).join('')}<text x="12" y="180" fill="currentColor" font-size="12">Arrows show forward dependencies. Reverse contributions flow toward x and add.</text>`;
    root.querySelector('[data-table]').innerHTML='<table><caption>Forward values and current reverse adjoints</caption><thead><tr><th>Node</th><th>Value</th><th>Adjoint</th></tr></thead><tbody>'+[formula.x,formula.mul,formula.add].map((name,i)=>`<tr><td>${name}</td><td>${number(values[i].toFixed(4))}</td><td>${number(gradients[i].toFixed(4))}</td></tr>`).join('')+'</tbody></table>';
    next.disabled=step===2;
  }
  input.addEventListener('change',()=>{step=0;render();});
  next.addEventListener('click',()=>{step=Math.min(2,step+1);render();});
  root.querySelector('[data-reset]').addEventListener('click',()=>{input.value='3';step=0;render();});render();
})();
