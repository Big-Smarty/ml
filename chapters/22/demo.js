(() => {
  const root = document.getElementById('embedding-neighborhood');
  if (!root) return;
  const query = root.querySelector('[data-query]');
  const scale = root.querySelector('[data-scale]');
  const output = root.querySelector('[data-output]');
  const body = root.querySelector('[data-ranking]');
  const vectors = [[1, 0], [0.8, 0.6], [0, 1], [-1, 0]];
  const names = ['A: observed positive', 'B: held-out positive', 'C: explicit negative', 'D: explicit negative'];
  function render() {
    const q = query.value === 'horizontal' ? [1, 0] : [0, 1];
    const multiplier = Number(scale.value);
    const items = vectors.map((v, i) => ({
      i,
      vector: v.map(x => x * (i === 2 ? multiplier : 1))
    })).map(item => {
      const [x, y] = item.vector;
      return {...item, dot: q[0] * x + q[1] * y,
        cosine: (q[0] * x + q[1] * y) / Math.hypot(x, y),
        distance: Math.hypot(q[0] - x, q[1] - y)};
    });
    const candidates = items.filter(item => item.i !== 0).sort((a, b) => b.dot - a.dot || a.i - b.i);
    body.replaceChildren();
    for (const item of items) {
      const tr = document.createElement('tr');
      for (const value of [names[item.i], `[${item.vector.join(', ')}]`, item.dot.toFixed(3), item.cosine.toFixed(3), item.distance.toFixed(3)]) {
        const td = document.createElement('td'); td.textContent = value; tr.append(td);
      }
      body.append(tr);
    }
    const rank = candidates.findIndex(item => item.i === 1);
    const dcg = rank < 2 ? 1 / Math.log2(rank + 2) : 0;
    const reconstruction = [q[0], q[1], q[0] + q[1], q[0] - q[1]];
    output.textContent = `Query [${q.join(', ')}]. Only C's length is scaled by ${multiplier}; other item vectors are fixed. Dot-product candidate order after excluding observed A: ${candidates.map(item => String.fromCharCode(65 + item.i)).join(', ')}. Held-out B has one-based rank ${rank + 1}, Recall@2 ${rank < 2 ? 1 : 0}, Precision@2 ${rank < 2 ? 0.5 : 0}, nDCG@2 ${dcg.toFixed(4)}. The fixed decoder [z0,z1,z0+z1,z0−z1] reconstructs query code as [${reconstruction.join(', ')}]. Its reconstruction role differs from ranking. Cosine ignores C's positive scale; dot product and distance can change. Explain why a long vector is not automatically a better neighbor.`;
  }
  query.addEventListener('change', render);
  scale.addEventListener('change', render);
  root.querySelector('[data-reset]').addEventListener('click', () => {
    query.value = 'horizontal'; scale.value = '1';
    root.querySelector('[data-prediction]').value = '';
    render();
  });
  render();
})();
