(() => {
  const root = document.getElementById('convolution-microscope');
  if (!root) return;
  const image = Array.from({length: 25}, (_, i) => i % 5 === 2 ? (Math.floor(i / 5) + 1) / 5 : 0);
  const kernel = [-1, 0, 1, -1, 0, 1, -1, 0, 1];
  const geometry = root.querySelector('[data-geometry]');
  const output = root.querySelector('[data-output]');
  const table = root.querySelector('[data-map]');
  // Templates contain fixed notation and numeric values from this illustration only.
  const number = value => CourseNumbers.mathml(value);
  const math = content => `<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow>${content}</mrow></math>`;
  const scalar = value => math(number(value));
  const vector = values => math(`<mo>[</mo><mtable><mtr>${values.map(value => `<mtd>${number(value)}</mtd>`).join('')}</mtr></mtable><mo>]</mo>`);
  const shape = side => math(`${number(side)}<mo>×</mo>${number(side)}`);
  const coordinate = (row, column) => math(`<mo>(</mo>${number(row)}<mo>,</mo>${number(column)}<mo>)</mo>`);
  let position = 0;
  function compute(padding, stride) {
    const side = Math.floor((5 + 2 * padding - 3) / stride) + 1;
    const values = [], products = [];
    for (let r = 0; r < side; r++) {
      for (let c = 0; c < side; c++) {
        const terms = kernel.map((w, j) => {
          const ir = r * stride + Math.floor(j / 3) - padding;
          const ic = c * stride + j % 3 - padding;
          return w * (ir >= 0 && ir < 5 && ic >= 0 && ic < 5 ? image[ir * 5 + ic] : 0);
        });
        products.push(terms);
        values.push(terms.reduce((a, b) => a + b, 0));
      }
    }
    return {side, values, products};
  }
  function render() {
    const presets = {valid: [0, 1], same: [1, 1], stride: [1, 2]};
    const [padding, stride] = presets[geometry.value];
    const {side, values, products} = compute(padding, stride);
    position %= values.length;
    const relu = values.map(v => Math.max(0, v));
    const pool = [0, 1, side, side + 1];
    const winner = pool.reduce((a, b) => relu[b] > relu[a] ? b : a);
    const r = Math.floor(position / side), c = position % side;
    const throughRelu = values[winner] > 0 ? 0.6 : 0;
    const kernelGradient = kernel.map((_, j) => {
      const ir = Math.floor(winner / side) * stride + Math.floor(j / 3) - padding;
      const ic = (winner % side) * stride + j % 3 - padding;
      return throughRelu * (ir >= 0 && ir < 5 && ic >= 0 && ic < 5 ? image[ir * 5 + ic] : 0);
    });
    output.innerHTML = `Input ${shape(5)}, kernel ${shape(3)}, padding ${scalar(padding)}, stride ${scalar(stride)}: output ${shape(side)}. Selected output ${coordinate(r, c)}, nine products ${vector(products[position])}, sum ${scalar(values[position])}, ReLU ${scalar(relu[position])}. First ${shape(2)} max pool is ${scalar(relu[winner])}, winner ${coordinate(Math.floor(winner / side), winner % side)}. Incoming gradient ${scalar('0.60')} routes only to this winner; all other map cells receive zero. After ReLU, the winning pre-activation receives ${scalar(throughRelu)}; its nine kernel-gradient contributions are ${vector(kernelGradient)}. Ties choose the first row-major cell. Why does changing stride alter which evidence survives?`;
    table.replaceChildren();
    const caption = document.createElement('caption');
    caption.textContent = 'ReLU feature map; selected convolution location marked “selected”; pooling winner marked “winner”.';
    table.append(caption);
    const body = document.createElement('tbody');
    for (let row = 0; row < side; row++) {
      const tr = document.createElement('tr');
      for (let col = 0; col < side; col++) {
        const at = row * side + col, td = document.createElement('td');
        td.innerHTML = `${scalar(relu[at])}${at === position ? ' selected' : ''}${at === winner ? ' winner' : ''}`;
        tr.append(td);
      }
      body.append(tr);
    }
    table.append(body);
  }
  root.querySelector('[data-next]').addEventListener('click', () => {position++; render();});
  root.querySelector('[data-previous]').addEventListener('click', () => {position = Math.max(0, position - 1); render();});
  root.querySelector('[data-reset]').addEventListener('click', () => {
    geometry.value = 'valid'; position = 0;
    root.querySelector('[data-prediction]').value = '';
    render();
  });
  geometry.addEventListener('change', () => {position = 0; render();});
  render();
})();
