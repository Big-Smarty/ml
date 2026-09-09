(() => {
  const root = document.getElementById('embedding-neighborhood');
  if (!root) return;
  const query = root.querySelector('[data-query]');
  const scale = root.querySelector('[data-scale]');
  const output = root.querySelector('[data-output]');
  const body = root.querySelector('[data-ranking]');
  const vectors = [[1, 0], [0.8, 0.6], [0, 1], [-1, 0]];
  const names = ['A: observed positive', 'B: held-out positive', 'C: explicit negative', 'D: explicit negative'];
  // Templates contain fixed notation and numeric values from this illustration only.
  const number = value => CourseNumbers.mathml(value);
  const math = content => `<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>${content}</mrow></math>`;
  const scalar = value => math(number(value));
  const vector = values => math(`<mo>[</mo><mtable><mtr>${values.map(value => `<mtd>${number(value)}</mtd>`).join('')}</mtr></mtable><mo>]</mo>`);
  const metric = (name, value) => math(`<mi mathvariant="normal">${name}</mi><mo>@</mo><mn>2</mn><mo>=</mo>${number(value)}`);
  const decoder = math('<mo>[</mo><msub><mi>z</mi><mn>0</mn></msub><mo>,</mo><msub><mi>z</mi><mn>1</mn></msub><mo>,</mo><msub><mi>z</mi><mn>0</mn></msub><mo>+</mo><msub><mi>z</mi><mn>1</mn></msub><mo>,</mo><msub><mi>z</mi><mn>0</mn></msub><mo>−</mo><msub><mi>z</mi><mn>1</mn></msub><mo>]</mo>');
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
      const label = document.createElement('td');
      label.textContent = names[item.i];
      tr.append(label);
      for (const value of [vector(item.vector), scalar(item.dot), scalar(item.cosine), scalar(item.distance)]) {
        const td = document.createElement('td'); td.innerHTML = value; tr.append(td);
      }
      body.append(tr);
    }
    const rank = candidates.findIndex(item => item.i === 1);
    const dcg = rank < 2 ? 1 / Math.log2(rank + 2) : 0;
    const reconstruction = [q[0], q[1], q[0] + q[1], q[0] - q[1]];
    output.innerHTML = `Query ${vector(q)}. Only C's length is scaled by ${scalar(multiplier)}; other item vectors are fixed. Dot-product candidate order after excluding observed A: ${candidates.map(item => String.fromCharCode(65 + item.i)).join(', ')}. Held-out B has one-based rank ${scalar(rank + 1)}, ${metric('Recall', rank < 2 ? 1 : 0)}, ${metric('Precision', rank < 2 ? 0.5 : 0)}, ${metric('nDCG', dcg)}. The fixed decoder ${decoder} reconstructs query code as ${vector(reconstruction)}. Its reconstruction role differs from ranking. Cosine ignores C's positive scale; dot product and distance can change. Explain why a long vector is not automatically a better neighbor.`;
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
