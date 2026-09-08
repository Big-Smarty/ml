(() => {
  const get = id => document.getElementById(id);
  if (!get('attention-lab')) return;
  function draw() {
    const scores = [2, 1, Number(get('future-score').value)];
    const count = get('causal-enabled').checked ? 2 : 3;
    const max = Math.max(...scores.slice(0, count));
    const probabilities = scores.map((s, i) => i < count ? Math.exp(s - max) : 0);
    const sum = probabilities.reduce((a, b) => a + b, 0);
    probabilities.forEach((_, i) => probabilities[i] /= sum);
    get('future-score-value').textContent = scores[2];
    get('attention-bars').replaceChildren();
    probabilities.forEach((p, i) => {
      const row = document.createElement('div');
      row.style.cssText = 'margin:8px 0;padding:7px 10px;border:1px solid var(--line);position:relative;isolation:isolate;font-size:13px';
      const bar = document.createElement('span');
      bar.style.cssText = `position:absolute;inset:0 auto 0 0;width:${p * 100}%;background:var(--sage);z-index:-1`;
      row.append(bar, document.createTextNode(`Key ${i}${i === 2 ? ' (future)' : ''}: ${(100 * p).toFixed(2)}%`));
      get('attention-bars').append(row);
    });
    const output = probabilities.reduce((n, p, i) => n + p * [2, 6, 20][i], 0);
    get('attention-readout').textContent = `Probabilities [${probabilities.map(p => p.toFixed(6)).join(', ')}] · weighted value ${output.toFixed(6)}`;
  }
  get('future-score').addEventListener('input', draw);
  get('causal-enabled').addEventListener('change', draw);
  draw();
})();
