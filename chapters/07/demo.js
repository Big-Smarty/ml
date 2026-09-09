(() => {
  'use strict';
  const sigmoid = z => z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  function calculate(mode, threshold) {
    return [[0,0,0],[0,1,1],[1,0,1],[1,1,0]].map(([a,b,target]) => {
      const hidden = [sigmoid(8*(a+b-threshold)),sigmoid(8*(a+b-1.5))];
      const logit = mode === 'linear' ? a+b-threshold : 8*(hidden[0]-hidden[1]-0.5);
      const probability = sigmoid(logit);
      return {a,b,target,hidden,logit,probability,prediction:Number(probability>=0.5)};
    });
  }
  if (typeof module === 'object') module.exports = {calculate};
  if (typeof document === 'undefined') return;
  const root=document.querySelector('#xor-hidden'); if (!root) return;
  const model=root.querySelector('[data-model]'), threshold=root.querySelector('[data-threshold]');
  function render() {
    const value=Number(threshold.value);
    if (!threshold.value || !Number.isFinite(value) || value < -1 || value > 3) {root.querySelector('[data-values]').textContent='Enter a threshold between −1 and 3.';return;}
    const rows=calculate(model.value,value), correct=rows.filter(r=>r.target===r.prediction).length;
    root.querySelector('[data-values]').textContent=`${model.value === 'linear' ? 'Linear z = x₀+x₁−threshold' : 'Hidden slopes = 8; second threshold = 1.5; output z = 8(h₀−h₁−0.5)'}. First threshold ${value}. ${correct}/4 correct Boolean decisions. These are hand-chosen parameters, not a trained model.`;
    const shape=(x,y,label,target)=>`${target?`<rect x="${x-5}" y="${y-5}" width="10" height="10" fill="currentColor"/>`:`<circle cx="${x}" cy="${y}" r="5" fill="none" stroke="currentColor" stroke-width="2"/>`}<text x="${x+9}" y="${y-8}" fill="currentColor" font-size="13">${label}</text>`;
    const axes=(origin,name,xlabel,ylabel)=>`<path d="M${origin} 35V220H${origin+190}" fill="none" stroke="currentColor"/><text x="${origin}" y="20" fill="currentColor" font-size="15">${name}</text><text x="${origin+165}" y="244" fill="currentColor" font-size="13">${xlabel}</text><text x="${origin-26}" y="40" fill="currentColor" font-size="13">${ylabel}</text><text x="${origin-12}" y="236" fill="currentColor" font-size="12">0</text><text x="${origin+175}" y="236" fill="currentColor" font-size="12">1</text>`;
    const px=(u,origin)=>origin+10+170*u,py=v=>210-170*v;
    const inputPoints=rows.map(r=>shape(px(r.a,35),py(r.b),`${r.a}${r.b}`,r.target)).join('');
    const hiddenPoints=[rows[0],rows[1],rows[3]].map(r=>shape(px(r.hidden[0],340),py(r.hidden[1]),r.a!==r.b?'01 / 10':`${r.a}${r.b}`,r.target)).join('');
    const boundary=model.value==='hidden'?`<path d="M${px(.5,340)} ${py(0)}L${px(1,340)} ${py(.5)}" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="5 4"/>`:(value>=0&&value<=2?`<path d="M${px(Math.max(0,value-1),35)} ${py(Math.min(1,value))}L${px(Math.min(1,value),35)} ${py(Math.max(0,value-1))}" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="5 4"/>`:'');
    root.querySelector('[data-scatter]').innerHTML=`<title id="xor-scatter-title">Original inputs and hidden coordinates</title><desc id="xor-scatter-desc">Circles denote target zero; filled squares denote target one. Labels identify the original Boolean inputs. The adjacent table gives each hidden coordinate and prediction. Dashed line is the selected model's zero-logit boundary.</desc>${axes(35,'Original inputs','x₀','x₁')}${axes(340,'Hidden representation','h₀','h₁')}${boundary}${inputPoints}${hiddenPoints}<text x="35" y="266" fill="currentColor" font-size="12">Circle: target 0. Filled square: target 1. Dashed: decision boundary.</text>`;
    root.querySelector('[data-table]').innerHTML='<table><caption>All four inputs and hidden coordinates</caption><thead><tr><th>Input</th><th>Target</th><th>h₀</th><th>h₁</th><th>Probability</th><th>Class</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${r.a},${r.b}</td><td>${r.target}</td><td>${r.hidden[0].toFixed(4)}</td><td>${r.hidden[1].toFixed(4)}</td><td>${r.probability.toFixed(4)}</td><td>${r.prediction}</td></tr>`).join('')+'</tbody></table>';
  }
  model.addEventListener('change',render);threshold.addEventListener('change',render);
  root.querySelector('[data-reset]').addEventListener('click',()=>{model.value='hidden';threshold.value='0.5';render();});
  render();
})();
