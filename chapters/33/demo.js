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
  function draw() {
    const raw = Number(get('score').value);
    const score = Number.isFinite(raw) ? Math.max(-5, Math.min(5, raw)) : 2;
    get('score').value = score;
    const mode = get('mode').value;
    const r = calculate(mode, score);
    get('ids').textContent = `IDs/labels: [${r.ids.join(', ')}]. ${r.ids.length} tokens, ${r.ids.length - 1} next-token targets; vocabulary ${r.vocabulary}.`;
    get('counts').textContent = `First input 99 (c) → target ${r.target}. This pair occurs ${r.pairCount} times among ${r.currentCount} successors of c. In this fixture its unsmoothed count probability is 1.`;
    get('result').textContent = `Toy classifier: target logit ${score}, every other logit 0. Target probability ${r.probability.toFixed(6)}; next-token loss ${r.loss.toFixed(6)} nats. This illustrative score is supplied, not fitted from the counts.`;
    get('units').textContent = mode === 'scalar' ? 'Scalar labels are Unicode code points, mapped to five compact class addresses in this tiny illustrative vocabulary. This vocabulary does not cover unseen scalars.' : mode === 'bpe' ? 'One learned rule: (97,102) → 256, bytes “af”. Ordered decoding concatenates bytes; all 256 byte fallbacks remain.' : 'Byte IDs equal byte values. UTF-8 é consists of bytes 195 and 169; byte coverage includes every possible byte.';
  }
  get('mode').addEventListener('change', draw);
  get('score').addEventListener('change', draw);
  get('merge').addEventListener('click', () => {get('mode').value = 'bpe'; draw();});
  get('reset').addEventListener('click', () => {get('mode').value = 'bytes';get('score').value = '2';get('prediction').value = '';draw();});
  draw();
})();
