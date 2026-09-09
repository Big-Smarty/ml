/* Analytical counts only. This illustration neither executes WGSL nor measures time. */
function s06Dispatch(n) {
  const groups = Math.ceil(n / 64);
  return { groups, lanes: groups * 64, unused: groups * 64 - n, last: n - 1 };
}
function s06Tile(k) {
  return { rounds: Math.ceil(k / 16), padding: Math.ceil(k / 16) * 16 - k, a: 2 * k + 16, b: 16 * 5 + 3, c: 2 * 5 + 3 };
}
function s06Transfers(steps) {
  const upload = (8 * 2 + 8 + 17) * 4;
  const parameters = 17 * 4;
  const history = (steps + 1) * 4;
  return { upload, parameters, history, resident: upload + parameters + history, eachStep: upload + steps * parameters + history };
}
function s06Reduction(level) {
  let values = [3, 1, 4, 1, 5, 9, 2, 6];
  for (let pass = 0; pass < level; pass += 1) {
    const half = values.length / 2;
    values = values.slice(0, half).map((value, i) => value + values[i + half]);
  }
  return values;
}
(() => {
  const root = document.getElementById('gpu-workgroups');
  if (!root) return;
  const find = selector => root.querySelector(selector);
  // Only fixed MathML and numeric values enter these templates.
  const number = value => `<mn>${Number(value)}</mn>`;
  const math = body => `<math><mrow>${body}</mrow></math>`;
  const dimension = (symbol, value) => math(`<mi>${symbol}</mi><mo>=</mo>${number(value)}`);
  const vector = values => math(`<mo>[</mo>${values.map(number).join('<mo>,</mo>')}<mo>]</mo>`);
  const address = (symbol, row, column) => math(`<mi>${symbol}</mi><mo>[</mo>${number(row)}<mo>,</mo>${number(column)}<mo>]</mo>`);
  let level = 0;
  function update() {
    const n = Number(find('#gpu-n').value);
    const d = s06Dispatch(n);
    const rows = Array.from({ length: d.groups }, (_, group) => {
      const valid = Math.min(64, n - group * 64);
      const y = group * 62;
      return `<text x="10" y="${y + 20}" fill="currentColor">Group ${group}: ${valid} valid, ${64 - valid} unused</text><rect x="10" y="${y + 30}" width="${valid * 7}" height="22" fill="#218978"/><rect x="${10 + valid * 7}" y="${y + 30}" width="${(64 - valid) * 7}" height="22" fill="url(#gpu-guard-pattern)" stroke="currentColor"/>`;
    });
    const geometry = find('#gpu-geometry');
    geometry.setAttribute('viewBox', `0 0 540 ${d.groups * 62 + 36}`);
    geometry.innerHTML = `<title id="gpu-geometry-title">${d.groups} workgroups for ${n} values</title><desc id="gpu-geometry-desc">${n} valid stores and ${d.unused} guarded lanes. Solid regions are valid; striped regions are unused. Exact IDs are in the text below.</desc><defs><pattern id="gpu-guard-pattern" width="8" height="8" patternUnits="userSpaceOnUse"><path d="M0 8L8 0" stroke="currentColor" stroke-width="1"/></pattern></defs>${rows.join('')}<text x="10" y="${d.groups * 62 + 19}" fill="currentColor">Solid = valid stores; stripes = guarded lanes</text>`;
    find('[data-gpu-output="dispatch"]').innerHTML = `${dimension('N', n)}: ${d.groups} groups launch ${d.lanes} lanes. IDs 0–${d.last} are valid; ${d.unused ? `IDs ${n}–${d.lanes - 1} are guarded off` : 'no lanes are unused'}. Last value: group ${Math.floor(d.last / 64)}, local ID ${d.last % 64}, global ID ${d.last}.`;
    const values = s06Reduction(level);
    find('[data-gpu-output="reduction"]').innerHTML = `${level === 0 ? 'Loaded shared slots' : `After stride ${8 / 2 ** level} and its barrier`}: ${vector(values)}. ${level === 3 ? `Lane 0 stores ${math(number(31))}. Next returns to the initial load.` : 'Every lane reaches the next barrier, including lanes no longer adding.'}`;
    const k = Number(find('#gpu-k').value);
    const t = s06Tile(k);
    find('[data-gpu-output="tile"]').innerHTML = `${dimension('M', 3)}, ${dimension('K', k)}, ${dimension('N', 5)}: ${t.rounds} ${math('<mi>K</mi>')} round(s); ${t.padding} zero inner positions in the last round. ${address('C', 2, 3)} is offset ${t.c}. ${k > 16 ? `At inner index 16, ${address('A', 2, 16)} is offset ${t.a} and ${address('B', 16, 3)} offset ${t.b}.` : 'Inner index 16 is absent, so no second tile is loaded.'} Each round: guarded loads → barrier → 16 multiply-adds → barrier. Edge lanes skip only the final store.`;
    const steps = Number(find('#gpu-steps').value);
    const bytes = s06Transfers(steps);
    find('[data-gpu-output="transfers"]').textContent = `${steps} step${steps === 1 ? '' : 's'}: initial upload ${bytes.upload} bytes; final parameters ${bytes.parameters} bytes; loss history ${bytes.history} bytes; resident total ${bytes.resident} bytes. Reading parameters after every step instead totals ${bytes.eachStep} bytes. Counts include only these application buffers, not driver metadata, physical bus traffic, or time.`;
  }
  function selectPanel() {
    const selected = find('#gpu-stage').value;
    root.querySelectorAll('[data-gpu-panel]').forEach(panel => {
      panel.hidden = panel.dataset.gpuPanel !== selected;
    });
  }
  find('#gpu-stage').addEventListener('change', selectPanel);
  ['#gpu-n', '#gpu-k', '#gpu-steps'].forEach(selector => find(selector).addEventListener('change', update));
  find('#gpu-reduce-next').addEventListener('click', () => { level = (level + 1) % 4; update(); });
  find('#gpu-reset').addEventListener('click', () => {
    find('#gpu-stage').value = 'dispatch';
    find('#gpu-n').value = '67';
    find('#gpu-k').value = '17';
    find('#gpu-steps').value = '800';
    level = 0;
    update();
    selectPanel();
  });
  update();
  selectPanel();
})();
