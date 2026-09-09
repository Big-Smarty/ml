(() => {
  'use strict';
  const root = document.getElementById('split-tool');
  if (!root) return;
  const policy = root.querySelector('#s13-policy');
  const output = root.querySelector('[data-output]');
  const chart = root.querySelector('[data-chart]');
  const cases = {
    row: { name: 'Row holdout', train: 108, valid: 36, shared: 18, result: '26/36 = .722', rule: 'Hold IDs divisible by 4. Machines and time ranges overlap; this does not measure future new machines.' },
    group: { name: 'Group holdout', train: 96, valid: 48, shared: 0, result: '33/48 = .688', rule: 'Hold machine IDs whose remainder modulo 3 is 2. Machines differ, but days overlap.' },
    time: { name: 'Past to future', train: 72, valid: 54, shared: 18, result: '36/54 = .667', rule: 'Train through day 3; validate days 5–7. Same machines can appear on both sides.' },
    strict: { name: 'New machines in the future', train: 48, valid: 18, shared: 0, result: '12/18 = .667', rule: 'Train machines 0–11 through day 3; validate machines 12–17 on days 5–7.' },
    leak: { name: 'Forbidden repair-after feature', train: 48, valid: 18, shared: 0, result: '18/18 = 1.000', rule: 'Strict membership cannot fix a field recorded after the predicted event. This perfect rule is unavailable at prediction time.' }
  };
  function render() {
    const c = cases[policy.value];
    output.textContent = `${c.name}: training ${c.train} rows; validation ${c.valid}; shared machines ${c.shared}. ${c.rule}`;
    chart.innerHTML = `<table><caption>Frozen development diagnostic; completed kNN except the explicit leakage rule</caption><thead><tr><th>Training rows</th><th>Validation rows</th><th>Shared machines</th><th>Correct / total</th></tr></thead><tbody><tr><td>${c.train}</td><td>${c.valid}</td><td>${c.shared}</td><td>${c.result}</td></tr></tbody></table><p>Different policies change the validation population. These measured results are stored illustrations from the completed Rust lab, not browser training or a ranking of split quality.</p>`;
  }
  policy.addEventListener('change', render);
  root.querySelector('[data-action="reset"]').addEventListener('click', () => { policy.value = 'strict'; render(); });
  policy.value = 'strict';
  render();
})();
