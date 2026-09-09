(() => {
  function calculate(query = 1, future = 100, causal = true) {
    const scores = [2,1,future];
    const count = causal ? query + 1 : 3;
    const max = Math.max(...scores.slice(0,count));
    const weights = scores.map((s,i) => i < count ? Math.exp(s-max) : 0);
    const sum = weights.reduce((a,b)=>a+b,0);
    const probabilities = weights.map(v=>v/sum);
    const values = [[2,1],[6,-1],[20,3]];
    const output = [0,1].map(z=>probabilities.reduce((n,p,j)=>n+p*values[j][z],0));
    const residual = [output[0]+1,output[1]-1];
    const mean = (residual[0]+residual[1])/2;
    const variance = residual.reduce((n,v)=>n+(v-mean)**2,0)/2;
    const normalized = residual.map(v=>(v-mean)/Math.sqrt(variance+1e-5));
    return {scores,raw:scores.map(v=>v*Math.sqrt(2)),probabilities,output,residual,mean,variance,normalized};
  }
  if (typeof module !== 'undefined') module.exports = {calculate};
  if (typeof document === 'undefined') return;
  const root = document.getElementById('attention-lab');
  if (!root) return;
  const get = name => root.querySelector(`[data-field="${name}"]`);
  const vector = xs => `[${xs.map(v=>v.toFixed(6)).join(', ')}]`;
  function draw() {
    const query = Number(get('query').value);
    const value = Number(get('future').value);
    const future = Number.isFinite(value) ? Math.max(-5,Math.min(100,value)) : 100;
    get('future').value=future;
    const causal = get('causal').checked;
    const r = calculate(query,future,causal);
    get('scores').textContent = `Q=[1,0], head width=2. Raw Q·K=${vector(r.raw)}; scaled scores=${vector(r.scores)}. Allowed keys: ${causal?Array.from({length:query+1},(_,i)=>i).join(', '):'0, 1, 2 (leaking future where present)'}.`;
    get('bars').replaceChildren();
    r.probabilities.forEach((p,j)=>{
      const row=document.createElement('p');
      row.textContent=`Key ${j}: probability ${p.toFixed(6)}${causal&&j>query?' — masked':''}`;
      row.style.borderInlineStart=`${1+Math.round(p*24)}px solid var(--sage)`;
      row.style.paddingInlineStart='0.5rem';get('bars').append(row);
    });
    get('result').textContent = `Weighted values O=${vector(r.output)}. Identity output projection, then residual input [1,−1] gives ${vector(r.residual)}. LayerNorm mean=${r.mean.toFixed(6)}, variance=${r.variance.toFixed(6)}, ε=0.00001, gain=[1,1], bias=[0,0] → ${vector(r.normalized)}.`;
    get('explain').textContent = causal && query<2 ? 'Changing the future key score cannot affect this earlier query. Masking happens before the maximum and normalizer.' : causal ? 'At the last query, all keys are already in its available history.' : 'Unmasked attention can read later values. A favorable training loss from this leakage would not establish usable generation.';
  }
  ['query','future','causal'].forEach(name=>get(name).addEventListener('change',draw));
  get('reset').addEventListener('click',()=>{get('query').value='1';get('future').value='100';get('causal').checked=true;get('prediction').value='';draw();});
  draw();
})();
