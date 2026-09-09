(() => {
  'use strict';
  const root = document.getElementById('geometry-tool');
  if (!root) return;
  const view = root.querySelector('#s14-view');
  const input = root.querySelector('#s14-value');
  const label = root.querySelector('[data-parameter-label]');
  const output = root.querySelector('[data-output]');
  const chart = root.querySelector('[data-chart]');
  const points = [[0, 0], [2, 0], [0, 2], [2, 2]];
  const names = ['A', 'B', 'C', 'D'];
  // Templates are fixed; only validated numbers enter mathematical tokens.
  const mn = value => `<mn>${Number(value).toFixed(4)}</mn>`;
  const mi = value => `<mi>${value}</mi>`;
  const math = body => `<math xmlns="http://www.w3.org/1998/Math/MathML">${body}</math>`;
  const scalar = value => math(mn(value));
  const vector = values => math(`<mrow><mo>[</mo>${values.map(mn).join('<mo>,</mo>')}<mo>]</mo></mrow>`);
  const equal = (name, value) => math(`<mrow>${mi(name)}<mo>=</mo>${mn(value)}</mrow>`);
  const settings = {
    distance: ['Horizontal scale', 1, .1, 5, .1],
    tree: ['Horizontal threshold', 1, 0, 2, .1],
    kernel: ['RBF gamma', .5, .05, 5, .05],
    projection: ['Direction angle in degrees', 0, 0, 90, 5],
    cluster: ['First center horizontal position', 0, 0, 2.5, .1]
  };
  const sx = x => 40 + 110 * x;
  const sy = y => 255 - 110 * y;
  function render() {
    const v = Number(input.value);
    const spec = settings[view.value];
    if (!Number.isFinite(v) || v < spec[2] || v > spec[3]) { output.innerHTML = `Enter a value from ${scalar(spec[2])} through ${scalar(spec[3])}.`; return; }
    let evidence = [];
    let extra = '';
    let explanation = '';
    if (view.value === 'distance') {
      evidence = points.map((p, i) => [names[i], ((p[0] - 1) * v) ** 2 + (p[1] - 1) ** 2]);
      explanation = `All four squared distances are ${scalar(v * v + 1)}. The symmetric query remains tied at every horizontal scale; named order selects ${math(mi('A'))} for ${equal('k', 1)}. Scaling can change distances without changing this particular ranking.`;
    } else if (view.value === 'tree') {
      evidence = points.map((p, i) => [names[i], p[0] <= v ? 'left' : 'right']);
      explanation = `Query ${equal('x', 1)} goes ${1 <= v ? 'left' : 'right'} at threshold ${scalar(v)}. At threshold ${scalar(1)} each leaf contains one positive and one negative corner, so each Gini impurity is ${scalar(.5)}. A second split is needed for XOR.`;
      extra = `<path d="M${sx(v)} 25V255" stroke="currentColor" stroke-dasharray="5 3"/>`;
    } else if (view.value === 'kernel') {
      evidence = points.map((p, i) => [names[i], Math.exp(-v * ((p[0] - 1) ** 2 + (p[1] - 1) ** 2))]);
      explanation = `Squared distance is ${scalar(2)} for every corner, so RBF similarity ${math(`<mrow><mi mathvariant="normal">exp</mi><mo>⁡</mo><mo>(</mo><mo>−</mo>${mn(v)}<mo>×</mo>${mn(2)}<mo>)</mo><mo>=</mo>${mn(Math.exp(-2 * v))}</mrow>`)}. This changes influence width; it does not fit kernel coefficients or separate this tied query.`;
    } else if (view.value === 'projection') {
      const a = v * Math.PI / 180;
      const q = [Math.cos(a), Math.sin(a)];
      const score = q[0] + q[1];
      const reconstructed = q.map(x => x * score);
      const loss = reconstructed.reduce((s, x) => s + (1 - x) ** 2, 0);
      evidence = [[`Direction ${math(mi('x'))}`, q[0]], [`Direction ${math(mi('y'))}`, q[1]], ['Projection coordinate', score], [`Reconstruction ${math(mi('x'))}`, reconstructed[0]], [`Reconstruction ${math(mi('y'))}`, reconstructed[1]], ['Squared lost norm', loss]];
      explanation = `Unit direction at ${math(`<mrow>${mn(v)}<mo>°</mo></mrow>`)} projects query ${vector([1, 1])} to coordinate ${scalar(score)} and reconstructs ${vector(reconstructed)}. Squared lost norm ${scalar(loss)}. Origin is fixed for this calculation; fitted PCA also subtracts its training mean.`;
      extra = `<path d="M40 255L${sx(q[0] * 2)} ${sy(q[1] * 2)} M150 145L${sx(reconstructed[0])} ${sy(reconstructed[1])}" stroke="currentColor" fill="none" stroke-dasharray="5 3"/><rect x="${sx(reconstructed[0]) - 4}" y="${sy(reconstructed[1]) - 4}" width="8" height="8" fill="none" stroke="currentColor"/>`;
    } else {
      const d1 = (1 - v) ** 2 + 1;
      const d2 = 2;
      evidence = [['Center 1 squared distance', d1], ['Center 2 squared distance', d2]];
      explanation = `Centers are ${vector([v, 0])} and ${vector([2, 2])}. Query distances are ${scalar(d1)} and ${scalar(d2)}; center ${d1 <= d2 ? '1' : '2'} wins${d1 === d2 ? ' by the fixed tie rule' : ''}. This is an assignment calculation; the Rust learner must also update means.`;
      extra = `<rect x="${sx(v) - 6}" y="${sy(0) - 6}" width="12" height="12" fill="none" stroke="currentColor"/><rect x="${sx(2) - 6}" y="${sy(2) - 6}" width="12" height="12" fill="none" stroke="currentColor"/>`;
    }
    output.innerHTML = explanation;
    chart.innerHTML = `<svg viewBox="0 0 330 295" role="img" aria-labelledby="s14-title s14-desc"><title id="s14-title">Fixed four-corner geometry</title><desc id="s14-desc">A 0,0; B 2,0; C 0,2; D 2,2; query 1,1. ${view.value === "tree" ? "A dashed vertical line marks the threshold." : view.value === "projection" ? "Dashed lines mark projection and reconstruction; a square marks the reconstructed query." : view.value === "cluster" ? "Squares mark the two centers." : "The coordinates stay fixed; calculated distances or similarities appear in the table."} Exact results are in the table.</desc><path d="M40 25V255H285" stroke="currentColor" fill="none"/>${points.map((p, i) => `<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="4" fill="currentColor"/><text x="${sx(p[0]) + 7}" y="${sy(p[1]) - 6}" fill="currentColor">${names[i]} (${p.join(',')})</text>`).join('')}<path d="M144 139L156 151M144 151L156 139" stroke="currentColor"/><text x="158" y="150" fill="currentColor">query (1,1)</text>${extra}<text x="145" y="285" fill="currentColor">x coordinate</text></svg><table><caption>Calculated values for ${spec[0]}</caption><thead><tr><th>Quantity</th><th>Value</th></tr></thead><tbody>${evidence.map(([name, value]) => `<tr><td>${names.includes(name) ? math(mi(name)) : name}</td><td>${typeof value === 'number' ? scalar(value) : value}</td></tr>`).join('')}</tbody></table><p>All views reuse the same coordinates. The picture illustrates one mechanism and never runs or verifies your Rust code.</p>`;
  }
  function configure() {
    const [name, initial, min, max, step] = settings[view.value];
    label.innerHTML = view.value === 'tree'
      ? `Threshold on ${math(mi('x'))}; left means ${math(`<mrow>${mi('x')}<mo>≤</mo><mi mathvariant="normal">threshold</mi></mrow>`)}`
      : view.value === 'cluster' ? `${name}; center 2 stays at ${vector([2, 2])}`
      : view.value === 'kernel' ? `RBF ${math(mi('γ'))}` : name;
    Object.assign(input, { value: String(initial), min: String(min), max: String(max), step: String(step) });
    render();
  }
  view.addEventListener('change', configure);
  input.addEventListener('input', render);
  root.querySelector('[data-action="reset"]').addEventListener('click', () => { view.value = 'distance'; configure(); });
  configure();
})();
