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
  if (typeof module !== 'undefined') module.exports = { partition, lanes, rounding, dispatch };
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
    if (mode.value === 'threads') {
      const shards = partition(7, Number(count.value));
      output.textContent = `7 rows, ${count.value} requested workers: ${shards.map(s => `[${s.start},${s.end})`).join(', ')}. All 7 rows covered exactly once.`;
      map.textContent = `Private squared-error sums: ${shards.map(s => s.sum).join(' + ')} = 140.\nCreation-order reduction, then divide by all 7 rows: MSE=20.\nMean of shard means would be ${(shards.reduce((sum, s) => sum + s.sum / (s.end - s.start), 0) / shards.length).toFixed(6)}; it is wrong when unequal shards carry different means.`;
    } else if (mode.value === 'lanes') {
      const s = lanes(19, Number(width.value));
      output.textContent = `19 paired values, width ${width.value}: ${s.prefix / Number(width.value)} full vectors, tail length ${19 - s.prefix}, dot=${s.total}.`;
      map.textContent = `a=1..19; b=0.5. Lane sums=[${s.sums.join(',')}].\nHorizontal sum=${s.sums.reduce((a, b) => a + b, 0)}; tail=${s.tail}.\nFull vector prefix=[0,${s.prefix}); safe scalar tail=[${s.prefix},19).`;
    } else if (mode.value === 'rounding') {
      const [left, right] = rounding();
      output.textContent = `Illustrated f32 grouping: (100000000−100000000)+1=${left}; 100000000+(−100000000+1)=${right}.`;
      map.textContent = 'Math.fround rounds after each indicated addition. Fixed worker order fixes this grouping for one partition; changing grouping can change the result. This is a numerical illustration, not a claim about executed CPU instructions.';
    } else {
      output.textContent = `Illustrated default backend: ${dispatch(feature.value)}. These are selected hypothetical capabilities, not this browser’s detected CPU.`;
      map.textContent = `Default requires AVX2 AND FMA. Explicit AVX-512F extension: ${feature.value === 'wide' ? 'supported in this hypothetical case' : 'unsupported in this hypothetical case'}.\nReal Rust uses cfg architecture gates and runtime detection. No speed is predicted.`;
    }
  }
  for (const control of [mode, count, width, feature]) control.addEventListener('change', render);
  root.querySelector('[data-action="reset"]').addEventListener('click', () => { mode.value = 'threads'; count.value = '3'; width.value = '8'; feature.value = 'none'; render(); });
  render();
})();
