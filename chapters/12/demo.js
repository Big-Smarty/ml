(() => {
  'use strict';
  const root = document.getElementById('sampling-tool');
  if (!root) return;
  // Frozen chapter-12 development predictions: paired values from maintenance-v1.
  const rows = [{"id":149,"group":12,"p":0.161044,"y":0},{"id":150,"group":12,"p":0.323852,"y":0},{"id":151,"group":12,"p":0.523256,"y":0},{"id":161,"group":13,"p":0.456924,"y":1},{"id":162,"group":13,"p":0.162271,"y":0},{"id":163,"group":13,"p":0.124308,"y":0},{"id":173,"group":14,"p":0.061348,"y":0},{"id":174,"group":14,"p":0.206629,"y":1},{"id":175,"group":14,"p":0.301752,"y":0},{"id":185,"group":15,"p":0.740419,"y":0},{"id":186,"group":15,"p":0.465891,"y":1},{"id":187,"group":15,"p":0.471546,"y":0},{"id":197,"group":16,"p":0.406413,"y":0},{"id":198,"group":16,"p":0.751604,"y":1},{"id":199,"group":16,"p":0.731464,"y":1},{"id":209,"group":17,"p":0.899401,"y":1},{"id":210,"group":17,"p":0.638296,"y":1},{"id":211,"group":17,"p":0.620813,"y":0}];
  const mode = root.querySelector('#s12-mode');
  const unit = root.querySelector('#s12-unit');
  const sharpness = root.querySelector('#s12-sharpness');
  const draw = root.querySelector('[data-action="draw"]');
  const output = root.querySelector('[data-output]');
  const chart = root.querySelector('[data-chart]');
  let seed = 7n;
  function rng(initial) {
    let state = initial;
    return n => {
      if (state === 0n) state = 1n;
      state = BigInt.asUintN(64, state ^ (state << 13n));
      state ^= state >> 7n;
      state = BigInt.asUintN(64, state ^ (state << 17n));
      return Number(state % BigInt(n));
    };
  }
  const accuracy = sample => sample.reduce((s, r) => s + Number((r.p >= .5) === Boolean(r.y)), 0) / sample.length;
  function render() {
    const calibration = mode.value === 'calibration';
    unit.disabled = calibration;
    sharpness.disabled = !calibration;
    draw.disabled = calibration;
    if (!calibration) {
      const units = unit.value === 'row' ? rows.map(r => [r]) : Array.from({ length: 6 }, (_, i) => rows.filter(r => r.group === i + 12));
      const next = rng(seed);
      const values = [];
      let selected;
      for (let b = 0; b < 2000; b++) {
        const ids = Array.from({ length: units.length }, () => next(units.length));
        if (b === 0) selected = ids;
        values.push(accuracy(ids.flatMap(i => units[i])));
      }
      const first = values[0];
      values.sort((a, b) => a - b);
      output.textContent = `Seed ${seed}; ${unit.value === 'row' ? '18 paired rows' : '6 whole machines'} per replicate; 2000 replicates. Original accuracy ${accuracy(rows).toFixed(4)}. Percentile endpoints [${values[50].toFixed(4)}, ${values[1950].toFixed(4)}]. First replicate accuracy ${first.toFixed(4)}.`;
      const counts = new Map();
      selected.forEach(i => counts.set(i, (counts.get(i) || 0) + 1));
      chart.innerHTML = `<table><caption>First replicate: repeats stay repeated</caption><thead><tr><th>Selected ${unit.value === 'row' ? 'row ID' : 'machine'}</th><th>Times drawn</th><th>Rows contributed</th></tr></thead><tbody>${[...counts].map(([i, n]) => `<tr><td>${unit.value === 'row' ? units[i][0].id : units[i][0].group}</td><td>${n}</td><td>${n * units[i].length}</td></tr>`).join('')}</tbody></table><p>This recomputes paired resampling in JavaScript using the disclosed 64-bit simulation RNG. It illustrates the Rust algorithm; repetitions add no independent machines.</p>`;
    } else {
      const raw = Number(sharpness.value);
      if (!Number.isFinite(raw) || raw < .5 || raw > 3) { output.textContent = 'Enter a confidence multiplier from 0.5 through 3.'; return; }
      const transformed = rows.map(r => ({ ...r, p: 1 / (1 + Math.exp(-raw * Math.log(r.p / (1 - r.p)))) }));
      const bins = Array.from({ length: 4 }, () => ({ n: 0, p: 0, y: 0 }));
      transformed.forEach(r => { const b = bins[Math.min(3, Math.floor(r.p * 4))]; b.n++; b.p += r.p; b.y += r.y; });
      const brier = transformed.reduce((s, r) => s + (r.p - r.y) ** 2, 0) / rows.length;
      output.textContent = `Odds confidence multiplier ${raw}; fixed 18 rows; accuracy ${accuracy(transformed).toFixed(4)}, Brier ${brier.toFixed(4)}. Counts below matter: these are noisy frequency estimates.`;
      const visible = bins.map((b, i) => ({ ...b, i })).filter(b => b.n);
      chart.innerHTML = `<svg viewBox="0 0 330 270" role="img" aria-labelledby="s12-title s12-desc"><title id="s12-title">Reliability diagram</title><desc id="s12-desc">Mean probability is horizontal, observed failure frequency vertical; all values and bin counts appear in the table.</desc><path d="M40 20V230H290 M40 230L290 20" fill="none" stroke="currentColor" stroke-dasharray="4 3"/>${visible.map(b => `<circle cx="${40 + 250 * b.p / b.n}" cy="${230 - 210 * b.y / b.n}" r="5" fill="currentColor"/>`).join('')}<text x="110" y="258" fill="currentColor">Mean probability 0 → 1</text><text x="45" y="15" fill="currentColor">Failure frequency: top 1, bottom 0</text></svg><table><caption>Nonempty reliability bins; four fixed probability intervals</caption><thead><tr><th>Bin</th><th>Count</th><th>Mean probability</th><th>Failure frequency</th></tr></thead><tbody>${visible.map(b => `<tr><td>${b.i}</td><td>${b.n}</td><td>${(b.p / b.n).toFixed(4)}</td><td>${(b.y / b.n).toFixed(4)}</td></tr>`).join('')}</tbody></table>`;
    }
  }
  [mode, unit].forEach(control => control.addEventListener('change', render));
  sharpness.addEventListener('input', render);
  draw.addEventListener('click', () => { seed++; render(); });
  root.querySelector('[data-action="reset"]').addEventListener('click', () => { seed = 7n; mode.value = 'bootstrap'; unit.value = 'machine'; sharpness.value = '1'; render(); });
  render();
})();
