(() => {
  'use strict';
  function calculate(x,step) {
    return {values:[x,x*x,x*x+x],gradients:[step===0?0:step===1?1:1+2*x,step===0?0:1,1]};
  }
  if (typeof module === 'object') module.exports={calculate};
  if (typeof document === 'undefined') return;
  const root=document.querySelector('#autodiff-tape');if(!root)return;
  const input=root.querySelector('[data-x]'),next=root.querySelector('[data-step]');let step=0;
  function render(){
    const x=Number(input.value);
    if(!input.value || !Number.isFinite(x) || x< -5 || x>5){root.querySelector('[data-status]').textContent='Enter x between −5 and 5.';return;}
    const {values,gradients}=calculate(x,step);
    root.querySelector('[data-status]').textContent=[`Forward complete. Seed y with one; the adjoints of x and m start at zero.`,`Addition: dy/dm=1 and the direct dy/dx contribution is 1.`,`Multiplication: add ${x} twice to x. Final dy/dx = 1 + ${x} + ${x} = ${gradients[0]}.`][step];
    root.querySelector('svg').setAttribute('viewBox','0 0 580 195');
    root.querySelector('svg').innerHTML=`<title id="tape-title">Shared scalar graph at reverse step ${step}</title><desc id="tape-desc">x has value ${values[0]} and adjoint ${gradients[0]}. Its two multiplication uses each contribute ${step===2?x:0}; the direct addition path contributes ${step>0?1:0}. m has value ${values[1]} and adjoint ${gradients[1]}. y has value ${values[2]} and adjoint one. The table provides the same values.</desc><defs><marker id="tape-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="currentColor"/></marker></defs><path d="M115 83Q170 50 235 83M115 107Q170 140 235 107M355 95H450M80 65Q265 -15 500 65" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#tape-arrow)"/><text x="140" y="53" fill="currentColor" font-size="12">use 1: +${step===2?x:0}</text><text x="140" y="148" fill="currentColor" font-size="12">use 2: +${step===2?x:0}</text><text x="275" y="24" fill="currentColor" font-size="12">direct x path: +${step>0?1:0}</text>${[55,285,500].map((cx,i)=>`<rect x="${cx-50}" y="65" width="110" height="66" rx="5" fill="none" stroke="currentColor"/><text x="${cx-42}" y="87" fill="currentColor" font-size="13">${['x','m=x×x','y=m+x'][i]}: ${values[i].toFixed(2)}</text><text x="${cx-42}" y="111" fill="currentColor" font-size="13">adjoint: ${gradients[i].toFixed(2)}</text>`).join('')}<text x="12" y="180" fill="currentColor" font-size="12">Arrows show forward dependencies. Reverse contributions flow toward x and add.</text>`;
    root.querySelector('[data-table]').innerHTML='<table><caption>Forward values and current reverse adjoints</caption><thead><tr><th>Node</th><th>Value</th><th>Adjoint</th></tr></thead><tbody>'+['x','m = x × x','y = m + x'].map((name,i)=>`<tr><td>${name}</td><td>${values[i].toFixed(4)}</td><td>${gradients[i].toFixed(4)}</td></tr>`).join('')+'</tbody></table>';
    next.disabled=step===2;
  }
  input.addEventListener('change',()=>{step=0;render();});
  next.addEventListener('click',()=>{step=Math.min(2,step+1);render();});
  root.querySelector('[data-reset]').addEventListener('click',()=>{input.value='3';step=0;render();});render();
})();
