function inferenceBudgetS08(count, bits) {
  if (!Number.isInteger(count) || count < 1 || count > 12 || ![4, 8].includes(bits)) {
    throw new RangeError('Use 1–12 integer positions and int4 or int8.');
  }
  const weights = [-1, -0.3, 0.2, 1];
  const maximumCode = bits === 8 ? 127 : 7;
  const scale = 1 / maximumCode;
  const codes = weights.map(value => Math.sign(value) * Math.round(Math.abs(value / scale)));
  const restored = codes.map(value => value * scale);
  const error = Math.max(...weights.map((value, index) => Math.abs(value - restored[index])));
  const cacheBytes = 2 * 2 * count * 8 * 4;
  const fullRows = 2 * count * (count + 1) / 2;
  const cachedRows = 2 * count;
  const weightBytes = Math.ceil(weights.length * bits / 8) + 4;
  return { count, bits, codes, scale, restored, error, cacheBytes, fullRows, cachedRows, weightBytes,
    squareScoreBytes: 2 * count * count * 4, rowScoreBytes: count * 4,
    tileScoreBytes: Math.min(count, 2) * 4 };
}
if (typeof module !== 'undefined' && module.exports) module.exports = { inferenceBudgetS08 };
(() => {
  if (typeof document === 'undefined') return;
  const root = document.querySelector('#inference-budget');
  if (!root) return;
  const positions = root.querySelector('[name="positions"]');
  const precision = root.querySelector('[name="precision"]');
  const result = root.querySelector('[data-result]');
  const picture = root.querySelector('[data-cache]');
  const update = () => {
    const count = Number(positions.value);
    const bits = Number(precision.value);
    if (!Number.isInteger(count) || count < 1 || count > 12 || ![4, 8].includes(bits)) {
      result.textContent = 'Use an integer prefix length from 1 to 12 and select int8 or int4.';
      return;
    }
    const { codes, scale, restored, error, cacheBytes, fullRows, cachedRows, weightBytes } = inferenceBudgetS08(count, bits);
    // Build MathML from fixed element names and validated numeric values.
    // Text nodes preserve screen-reader access without parsing interpolated HTML.
    const mathNamespace = 'http://www.w3.org/1998/Math/MathML';
    const element = (name, ...children) => {
      const node = document.createElementNS(mathNamespace, name);
      node.append(...children.map(child => typeof child === 'string' ? document.createTextNode(child) : child));
      return node;
    };
    const number = value => {
      const text = String(value);
      return text.startsWith('-')
        ? element('mrow', element('mo', '−'), element('mn', text.slice(1)))
        : element('mn', text);
    };
    const scalar = value => element('math', number(value));
    const vector = values => element('math', element('mrow',
      element('mo', '['),
      ...values.flatMap((value, index) => index === 0 ? [number(value)] : [element('mo', ','), number(value)]),
      element('mo', ']')));
    result.replaceChildren(
      'Prefix ', scalar(count), ' tokens: K/V cache ', scalar(cacheBytes), ' bytes (f32), ',
      scalar(2 * 2 * count), ' stored width-eight vectors. Full-prefix recomputation processes ',
      scalar(fullRows), ' layer-token rows; caching processes ', scalar(cachedRows),
      '. A square two-head score tensor would use ', scalar(2 * count * count * 4),
      ' bytes; one row for a single query/head uses at most ', scalar(count * 4),
      ' score bytes; a two-key tile for a single query/head uses at most ', scalar(Math.min(count, 2) * 4),
      ' score bytes, excluding numerator/state and inputs. Int', String(bits), ' weight codes ', vector(codes),
      ', scale ', scalar(scale.toFixed(6)), ', reconstructed ', vector(restored.map(x => x.toFixed(6))),
      '. Weights plus one f32 scale: ', scalar(weightBytes), ' bytes versus ', scalar(16),
      ' dense bytes; maximum weight error ', scalar(error.toFixed(6)),
      '. Weight precision does not change this f32 cache.'
    );
    const ns = 'http://www.w3.org/2000/svg';
    picture.replaceChildren();
    const title = document.createElementNS(ns, 'title');
    title.textContent = `Two decoder layers each retain keys and values for ${count} positions.`;
    picture.append(title);
    const desc = document.createElementNS(ns, 'desc');
    desc.textContent = `Each numbered column is a token position. Four rows are layer 1 keys, layer 1 values, layer 2 keys, layer 2 values. Each cell contains eight f32 values, 32 bytes. Total ${cacheBytes} bytes.`;
    picture.append(desc);
    ['L1 K', 'L1 V', 'L2 K', 'L2 V'].forEach((label, row) => {
      const text = document.createElementNS(ns, 'text');
      text.setAttribute('x', '4'); text.setAttribute('y', String(28 + row * 36));
      text.setAttribute('fill', 'currentColor'); text.textContent = label; picture.append(text);
      for (let column = 0; column < count; column += 1) {
        const box = document.createElementNS(ns, 'rect');
        box.setAttribute('x', String(55 + column * 34)); box.setAttribute('y', String(8 + row * 36));
        box.setAttribute('width', '29'); box.setAttribute('height', '28');
        box.setAttribute('fill', 'none'); box.setAttribute('stroke', 'currentColor'); picture.append(box);
        const value = document.createElementNS(ns, 'text');
        value.setAttribute('x', String(63 + column * 34)); value.setAttribute('y', String(28 + row * 36));
        value.setAttribute('fill', 'currentColor'); value.textContent = String(column); picture.append(value);
      }
    });
  };
  positions.addEventListener('change', update);
  precision.addEventListener('change', update);
  root.querySelector('[data-reset]').addEventListener('click', () => { positions.value = '5'; precision.value = '4'; update(); });
  update();
})();
