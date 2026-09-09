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
  const numeric = (value, digits = null) => {
    if (!Number.isFinite(value)) throw new Error('Nonfinite illustration result');
    const magnitude = digits === null ? String(Math.abs(value)) : Math.abs(value).toFixed(digits);
    return `${value < 0 ? '<mo>−</mo>' : ''}<mn>${magnitude}</mn>`;
  };
  const vector = (values, digits = null) => `<mrow><mo>[</mo>${values.map(value => numeric(value, digits)).join('<mo>,</mo>')}<mo>]</mo></mrow>`;
  const math = (body, display = false) => `<math xmlns="http://www.w3.org/1998/Math/MathML"${display ? ' display="block"' : ''}><mrow>${body}</mrow></math>`;
  const scalar = (value, digits = null) => math(numeric(value, digits));
  function draw() {
    const selectedQuery = Number(get('query').value);
    const query = Number.isInteger(selectedQuery) && selectedQuery >= 0 && selectedQuery <= 2 ? selectedQuery : 1;
    get('query').value = String(query);
    const value = Number(get('future').value);
    const future = Number.isFinite(value) ? Math.max(-5,Math.min(100,value)) : 100;
    get('future').value=future;
    const causal = get('causal').checked;
    const r = calculate(query,future,causal);
    get('scores').innerHTML = `${math('<mi>Q</mi><mo>=</mo>' + vector([1,0]))}, head width ${scalar(2)}. Raw dot products ${math('<mi>Q</mi><mo>·</mo><mi>K</mi><mo>=</mo>' + vector(r.raw,6))}; scaled scores ${math(vector(r.scores,6))}. Allowed keys: ${math(vector(Array.from({length:causal ? query+1 : 3},(_,i)=>i)))}${causal ? '' : ' (leaking future where present)'}.`;
    get('bars').replaceChildren();
    r.probabilities.forEach((p,j)=>{
      const row=document.createElement('p');
      row.innerHTML=`Key ${scalar(j)}: probability ${scalar(p,6)}${causal&&j>query?' — masked':''}`;
      row.style.borderInlineStart=`${1+Math.round(p*24)}px solid var(--sage)`;
      row.style.paddingInlineStart='0.5rem';get('bars').append(row);
    });
    get('result').innerHTML = `Weighted values ${math('<mi>O</mi><mo>=</mo>' + vector(r.output,6))}. Identity output projection, then residual input ${math(vector([1,-1]))} gives ${math(vector(r.residual,6))}. LayerNorm mean ${scalar(r.mean,6)}, variance ${scalar(r.variance,6)}, ${math('<mi>ε</mi><mo>=</mo><mn>0.00001</mn>')}, gain ${math(vector([1,1]))}, bias ${math(vector([0,0]))}; normalized output ${math(vector(r.normalized,6))}.`;
    get('explain').textContent = causal && query<2 ? 'Changing the future key score cannot affect this earlier query. Masking happens before the maximum and normalizer.' : causal ? 'At the last query, all keys are already in its available history.' : 'Unmasked attention can read later values. A favorable training loss from this leakage would not establish usable generation.';
  }
  ['query','future','causal'].forEach(name=>get(name).addEventListener('change',draw));
  get('reset').addEventListener('click',()=>{get('query').value='1';get('future').value='100';get('causal').checked=true;get('prediction').value='';draw();});
  draw();
})();
