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
  const number = value => {
    if (!Number.isFinite(value)) throw new Error('Expected a finite illustration value');
    return `<mn>${value}</mn>`;
  };
  const math = body => `<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>${body}</mrow></math>`;
  const vector = values => `<mrow><mo>[</mo>${values.map(number).join('<mo>,</mo>')}<mo>]</mo></mrow>`;
  const index = (name, value) => `<msub><mi>${name}</mi>${number(value)}</msub>`;
  function present(order, count) {
    const s = state(order, count);
    const output = `${count} of 18 products; ${count * 2} counted FLOPs; ${count * 2} explicit input reads; ${math(`<mi>C</mi><mo>=</mo>${vector(s.values)}`)}.`;
    const next = s.next ? `Next: ${math(`${index('A', s.next.a)}<mo>×</mo>${index('B', s.next.b)}<mo>→</mo>${index('C', s.next.out)}`)}.` : 'Complete: all 18 products contributed once.';
    const seen = s.visits.slice(0, count).map(v => v.b);
    const log = `<p>${next}</p><p>${math('<mi>B</mi>')} offsets visited: ${seen.length ? math(vector(seen)) : '(none)'}</p><p>Source loads from ${math('<mi>A</mi>')} and ${math('<mi>B</mi>')} are counted explicitly; compiler reuse and actual cache traffic are not measured. Output allocation is one reusable six-value buffer.</p>`;
    return { output, log };
  }
  if (typeof module !== 'undefined') module.exports = { trace, state, present };
  if (typeof document === 'undefined') return;
  const root = document.getElementById('memory-gemm');
  if (!root) return;
  const order = root.querySelector('#gemm-order'), output = root.querySelector('output'), log = root.querySelector('[data-trace]');
  let count = 0;
  function render() {
    const view = present(order.value, count);
    // Only fixed MathML templates and finite internally calculated numbers enter these fragments.
    output.innerHTML = view.output;
    log.innerHTML = view.log;
    root.querySelector('[data-action="step"]').disabled = count === 18;
    root.querySelector('[data-action="finish"]').disabled = count === 18;
  }
  order.addEventListener('change', () => { count = 0; render(); });
  root.querySelector('[data-action="step"]').addEventListener('click', () => { count = Math.min(18, count + 1); render(); });
  root.querySelector('[data-action="finish"]').addEventListener('click', () => { count = 18; render(); });
  root.querySelector('[data-action="reset"]').addEventListener('click', () => { order.value = 'ijk'; count = 0; render(); });
  render();
})();
