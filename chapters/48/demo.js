/* Fixed arithmetic illustration: no training, Rust execution, or network requests. */
(() => {
  function calculate(concentration, factor, alpha) {
    const experts = 3, tokens = 12;
    const capacity = Math.max(1, Math.ceil(factor * tokens / experts));
    const attempted = [0, 0, 0], accepted = [0, 0, 0], probabilitySums = [0, 0, 0];
    const rows = [];
    for (let token = 0; token < tokens; token++) {
      const preferred = Math.floor(token / 4);
      const logits = [0, 1, 2].map(expert => (expert === preferred ? 2 : 0) + (expert === 0 ? concentration : 0));
      const maximum = Math.max(...logits), exp = logits.map(value => Math.exp(value - maximum));
      const sum = exp.reduce((a, b) => a + b, 0), probabilities = exp.map(value => value / sum);
      let route = 0;
      for (let expert = 1; expert < experts; expert++) if (probabilities[expert] > probabilities[route]) route = expert;
      attempted[route]++;
      const admitted = accepted[route] < capacity;
      if (admitted) accepted[route]++;
      probabilities.forEach((value, expert) => { probabilitySums[expert] += value; });
      rows.push({token: token + 1, route, admitted, gate: probabilities[route]});
    }
    const mean = probabilitySums.map(value => value / tokens);
    const rawBalance = experts * attempted.reduce((sum, count, expert) => sum + count / tokens * mean[expert], 0);
    return {capacity, attempted, accepted, mean, rows, rawBalance, balance: alpha * rawBalance, dropped: tokens - accepted.reduce((a, b) => a + b, 0)};
  }
  if (typeof module !== 'undefined') module.exports = {calculate};
  if (typeof document === 'undefined') return;
  const root = document.getElementById('routing-capacity');
  if (!root) return;
  const controls = ['concentration', 'factor', 'alpha'].map(name => root.querySelector(`[data-control="${name}"]`));
  const body = root.querySelector('[data-experts]');
  const tokenList = root.querySelector('[data-tokens]');
  const summary = root.querySelector('[data-summary]');
  const mathNamespace = 'http://www.w3.org/1998/Math/MathML';
  function mathNode(name, ...children) {
    const node = document.createElementNS(mathNamespace, name);
    node.append(...children);
    return node;
  }
  function mathNumber(value) {
    return mathNode('math', mathNode('mn', String(value)));
  }
  function mathFraction(numerator, denominator) {
    return mathNode('math', mathNode('mfrac', mathNode('mn', String(numerator)), mathNode('mn', String(denominator))));
  }
  function mathRoute(token, expert) {
    return mathNode('math', mathNode('mrow', mathNode('mn', String(token)), mathNode('mo', '→'), mathNode('msub', mathNode('mi', 'E'), mathNode('mn', String(expert)))));
  }
  function render() {
    const [concentration, factor, alpha] = controls.map(control => Number(control.value));
    controls.forEach(control => {
      root.querySelector(`[data-value="${control.dataset.control}"]`).replaceChildren(mathNumber(Number(control.value).toFixed(2)));
    });
    const result = calculate(concentration, factor, alpha);
    body.replaceChildren();
    for (let expert = 0; expert < 3; expert++) {
      const row = document.createElement('tr');
      [expert, result.attempted[expert], result.accepted[expert], result.attempted[expert] - result.accepted[expert], result.mean[expert].toFixed(4)].forEach(value => {
        const cell = document.createElement('td');
        cell.append(mathNumber(value));
        row.append(cell);
      });
      body.append(row);
    }
    tokenList.replaceChildren();
    result.rows.forEach((row, index) => {
      if (index) tokenList.append('; ');
      tokenList.append(mathRoute(row.token, row.route), row.admitted ? ' admitted' : ' dropped');
    });
    summary.replaceChildren(
      'Capacity ', mathNumber(result.capacity), ' per expert; ',
      mathFraction(12 - result.dropped, 12), ' admitted; ',
      mathFraction(result.dropped, 12), ' dropped. Unweighted balance ',
      mathNumber(result.rawBalance.toFixed(4)), '; alpha-weighted balance ',
      mathNumber(result.balance.toFixed(4)),
      '. Capacity changes admission only; alpha scales the displayed auxiliary loss only. No router is trained here.'
    );
  }
  controls.forEach(control => control.addEventListener('input', render));
  root.querySelector('[data-reset]').addEventListener('click', () => { controls.forEach((control, index) => {control.value = [0, 1, 0.02][index];}); render(); });
  render();
})();
