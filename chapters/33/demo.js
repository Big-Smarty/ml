(() => {
  // Fixed illustration: one explicit BPE merge (a,f)->256, not full tokenizer training.
  function calculate(mode = 'bytes', score = 2) {
    const bytes = [99,97,102,195,169,32,99,97,102,195,169];
    const scalars = [99,97,102,233,32,99,97,102,233];
    const ids = mode === 'scalar' ? scalars : mode === 'bpe' ? [99,256,195,169,32,99,256,195,169] : bytes;
    const vocabulary = mode === 'scalar' ? 5 : mode === 'bpe' ? 257 : 256;
    const target = ids[1];
    let currentCount = 0, pairCount = 0;
    for (let i = 0; i + 1 < ids.length; i++) {
      if (ids[i] === ids[0]) { currentCount++; if (ids[i + 1] === target) pairCount++; }
    }
    const maximum = Math.max(0, score);
    const denominator = Math.exp(score - maximum) + (vocabulary - 1) * Math.exp(-maximum);
    const probability = Math.exp(score - maximum) / denominator;
    const loss = Math.log(denominator) + maximum - score;
    return {ids, vocabulary, target, currentCount, pairCount, probability, loss};
  }
  if (typeof module !== 'undefined') module.exports = {calculate};
  if (typeof document === 'undefined') return;
  const root = document.getElementById('token-loss-lab');
  if (!root) return;
  const get = name => root.querySelector(`[data-field="${name}"]`);
  const numeric = value => CourseNumbers.mathml(value);
  const math = (body, display = false) => `<math xmlns="http://www.w3.org/1998/Math/MathML"${display ? ' display="block"' : ''}><mrow>${body}</mrow></math>`;
  const scalar = value => math(numeric(value));
  function draw() {
    const raw = Number(get('score').value);
    const score = Number.isFinite(raw) ? Math.max(-5, Math.min(5, raw)) : 2;
    get('score').value = score;
    const mode = get('mode').value;
    const r = calculate(mode, score);
    get('ids').innerHTML = `<p>IDs/labels: <code>[${r.ids.join(', ')}]</code>. ${r.ids.length} tokens, ${r.ids.length - 1} next-token targets; vocabulary ${r.vocabulary}.</p>`;
    get('counts').innerHTML = `First input/target <code>99 → ${r.target}</code> (input <code>c</code>). This pair occurs ${r.pairCount} times among ${r.currentCount} successors of <code>c</code>. In this fixture its unsmoothed count probability is ${scalar(1)}.`;
    get('result').innerHTML = `Toy classifier: target logit ${scalar(score)}, every other logit ${scalar(0)}. Target probability ${scalar(r.probability)}; next-token loss ${scalar(r.loss)} nats. This illustrative score is supplied, not fitted from the counts.`;
    get('units').innerHTML = mode === 'scalar' ? 'Scalar labels are Unicode code points, mapped to five compact class addresses in this tiny illustrative vocabulary. This vocabulary does not cover unseen scalars.' : mode === 'bpe' ? `One learned rule: <code>(97,102) → 256</code>, bytes <code>af</code>. Ordered decoding concatenates bytes; all 256 byte fallbacks remain.` : `Byte IDs equal byte values. UTF-8 <code>é</code> consists of bytes <code>[195,169]</code>; byte coverage includes every possible byte.`;
  }
  get('mode').addEventListener('change', draw);
  get('score').addEventListener('change', draw);
  get('merge').addEventListener('click', () => {get('mode').value = 'bpe'; draw();});
  get('reset').addEventListener('click', () => {get('mode').value = 'bytes';get('score').value = '2';get('prediction').value = '';draw();});
  draw();
})();
