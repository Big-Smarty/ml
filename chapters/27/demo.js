(() => {
  'use strict';
  function partition(rows, threads) {
    const size = Math.ceil(rows / Math.min(rows, threads)), shards = [];
    for (let start = 0; start < rows; start += size) {
      const end = Math.min(rows, start + size);
      let sum = 0;
      for (let r = start; r < end; r++) sum += (r + 1) ** 2;
      shards.push({ start, end, sum });
    }
    return shards;
  }
  function lanes(n, width) {
    const prefix = Math.floor(n / width) * width, sums = Array(width).fill(0);
    for (let i = 0; i < prefix; i++) sums[i % width] += (i + 1) * 0.5;
    let tail = 0;
    for (let i = prefix; i < n; i++) tail += (i + 1) * 0.5;
    return { prefix, sums, tail, total: sums.reduce((a, b) => a + b, 0) + tail };
  }
  function rounding() {
    const f = Math.fround;
    return [f(f(100000000 + -100000000) + 1), f(100000000 + f(-100000000 + 1))];
  }
  function dispatch(features) { return features === 'both' || features === 'wide' ? 'AVX2+FMA' : 'scalar'; }
  const number = value => {
    if (!Number.isFinite(value)) throw new Error('Expected a finite illustration value');
    return value < 0 ? `<mrow><mo>−</mo><mn>${-value}</mn></mrow>` : `<mn>${value}</mn>`;
  };
  const math = body => `<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>${body}</mrow></math>`;
  const vector = values => `<mrow><mo>[</mo>${values.map(number).join('<mo>,</mo>')}<mo>]</mo></mrow>`;
  const interval = (a, b) => `<mrow><mo>[</mo>${number(a)}<mo>,</mo>${number(b)}<mo>)</mo></mrow>`;
  function present(mode, count, width, feature) {
    if (mode === 'threads') {
      const shards = partition(7, count);
      const means = shards.reduce((sum, shard) => sum + shard.sum / (shard.end - shard.start), 0) / shards.length;
      return {
        output: `7 rows, ${count} requested workers: ${math(shards.map(shard => interval(shard.start, shard.end)).join('<mo>,</mo>'))}. All 7 rows covered exactly once.`,
        map: `<p>Private squared-error sums: ${math(shards.map(shard => number(shard.sum)).join('<mo>+</mo>') + '<mo>=</mo>' + number(140))}.</p><p>Creation-order reduction, then divide by all 7 rows: ${math('<mi mathvariant="normal">MSE</mi><mo>=</mo><mfrac>' + number(140) + number(7) + '</mfrac><mo>=</mo>' + number(20))}.</p><p>Mean of shard means would be ${math(number(Number(means.toFixed(6))))}; it is wrong when unequal shards carry different means.</p>`
      };
    }
    if (mode === 'lanes') {
      const s = lanes(19, width), total = s.sums.reduce((a, b) => a + b, 0);
      return {
        output: `19 paired values, width ${width}: ${s.prefix / width} full vectors, tail length ${19 - s.prefix}, ${math('<mi mathvariant="normal">dot</mi><mo>=</mo>' + number(s.total))}.`,
        map: `<p>${math('<mi>a</mi><mo>=</mo>' + vector(Array.from({ length: 19 }, (_, i) => i + 1)))}; ${math('<msub><mi>b</mi><mi>i</mi></msub><mo>=</mo>' + number(0.5))}. Lane sums: ${math(vector(s.sums))}.</p><p>Horizontal sum: ${math(number(total))}; tail: ${math(number(s.tail))}.</p><p>Full vector prefix: ${math(interval(0, s.prefix))}; safe scalar tail: ${math(interval(s.prefix, 19))}.</p>`
      };
    }
    if (mode === 'rounding') {
      const [left, right] = rounding();
      return {
        output: `Illustrated <code>f32</code> grouping: ${math('<mo>(</mo>' + number(100000000) + '<mo>−</mo>' + number(100000000) + '<mo>)</mo><mo>+</mo>' + number(1) + '<mo>=</mo>' + number(left))}; ${math(number(100000000) + '<mo>+</mo><mo>(</mo>' + number(-100000000) + '<mo>+</mo>' + number(1) + '<mo>)</mo><mo>=</mo>' + number(right))}.`,
        map: '<p><code>Math.fround</code> rounds after each indicated addition. Fixed worker order fixes this grouping for one partition; changing grouping can change the result. This is a numerical illustration, not a claim about executed CPU instructions.</p>'
      };
    }
    return {
      output: `Illustrated default backend: ${dispatch(feature)}. These are selected hypothetical capabilities, not this browser’s detected CPU.`,
      map: `<p>Default requires AVX2 AND FMA. Explicit AVX-512F extension: ${feature === 'wide' ? 'supported in this hypothetical case' : 'unsupported in this hypothetical case'}.</p><p>Real Rust uses <code>cfg</code> architecture gates and runtime detection. No speed is predicted.</p>`
    };
  }
  if (typeof module !== 'undefined') module.exports = { partition, lanes, rounding, dispatch, present };
  if (typeof document === 'undefined') return;
  const root = document.getElementById('threads-lanes');
  if (!root) return;
  const mode = root.querySelector('#map-mode'), count = root.querySelector('#map-count'), width = root.querySelector('#map-width'), feature = root.querySelector('#map-feature');
  const output = root.querySelector('output'), map = root.querySelector('[data-map]');
  function render() {
    for (const [control, visible] of [[count, mode.value === 'threads'], [width, mode.value === 'lanes'], [feature, mode.value === 'dispatch']]) {
      control.hidden = !visible;
      root.querySelector(`label[for="${control.id}"]`).hidden = !visible;
    }
    const view = present(mode.value, Number(count.value), Number(width.value), feature.value);
    // Templates are fixed; numerical slots pass a finite-number check and backend prose uses fixed choices.
    output.innerHTML = view.output;
    map.innerHTML = view.map;
  }
  for (const control of [mode, count, width, feature]) control.addEventListener('change', render);
  root.querySelector('[data-action="reset"]').addEventListener('click', () => { mode.value = 'threads'; count.value = '3'; width.value = '8'; feature.value = 'none'; render(); });
  render();
})();
