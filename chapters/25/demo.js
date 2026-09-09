(() => {
  'use strict';
  // Source-level multiply trace only. No cache replacement, timing, or hardware model.
  function trace(order) {
    const visits = [];
    const add = (r, k, c) => visits.push({ r, k, c, a: r * 3 + k, b: k * 3 + c, out: r * 3 + c });
    if (order === 'ijk') {
      for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) add(r, k, c);
    } else if (order === 'ikj') {
      for (let r = 0; r < 2; r++) for (let k = 0; k < 3; k++) for (let c = 0; c < 3; c++) add(r, k, c);
    } else {
      for (let r0 = 0; r0 < 2; r0 += 2) for (let k0 = 0; k0 < 3; k0 += 2) for (let c0 = 0; c0 < 3; c0 += 2)
        for (let r = r0; r < Math.min(r0 + 2, 2); r++) for (let k = k0; k < Math.min(k0 + 2, 3); k++) for (let c = c0; c < Math.min(c0 + 2, 3); c++) add(r, k, c);
    }
    return visits;
  }
  function state(order, count) {
    const visits = trace(order), values = Array(6).fill(0), a = [1, 2, 3, 4, 5, 6], b = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    for (const v of visits.slice(0, count)) values[v.out] += a[v.a] * b[v.b];
    return { visits, values, next: visits[count], last: count ? visits[count - 1] : null };
  }
  if (typeof module !== 'undefined') module.exports = { trace, state };
  if (typeof document === 'undefined') return;
  const root = document.getElementById('memory-gemm');
  if (!root) return;
  const order = root.querySelector('#gemm-order'), output = root.querySelector('output'), log = root.querySelector('[data-trace]');
  let count = 0;
  function render() {
    const s = state(order.value, count);
    output.textContent = `${count} of 18 products; ${count * 2} counted FLOPs; ${count * 2} explicit A/B reads; C=[${s.values.join(',')}].`;
    const next = s.next ? `Next: A[${s.next.a}] × B[${s.next.b}] → C[${s.next.out}].` : 'Complete: all 18 products contributed once.';
    const seen = s.visits.slice(0, count).map(v => v.b);
    log.textContent = `${next}\nB offsets visited: ${seen.join(', ') || '(none)'}\nA/B source loads are counted explicitly; compiler reuse and actual cache traffic are not measured. Output allocation is one reusable six-value buffer.`;
    root.querySelector('[data-action="step"]').disabled = count === 18;
    root.querySelector('[data-action="finish"]').disabled = count === 18;
  }
  order.addEventListener('change', () => { count = 0; render(); });
  root.querySelector('[data-action="step"]').addEventListener('click', () => { count = Math.min(18, count + 1); render(); });
  root.querySelector('[data-action="finish"]').addEventListener('click', () => { count = 18; render(); });
  root.querySelector('[data-action="reset"]').addEventListener('click', () => { order.value = 'ijk'; count = 0; render(); });
  render();
})();
