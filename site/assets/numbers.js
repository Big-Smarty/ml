/* Shared presentation only: calculations always receive the original numbers. */
const CourseNumbers = (() => {
  function parts(input, significantDigits = 4) {
    if (typeof input !== 'number' && (typeof input !== 'string' || !input.trim())) throw new TypeError('Expected a finite number');
    const value = Number(input);
    if (!Number.isFinite(value)) throw new TypeError('Expected a finite number');
    if (!Number.isInteger(significantDigits) || significantDigits < 1 || significantDigits > 17) throw new RangeError('Use 1–17 significant digits');
    const magnitude = Math.abs(value);
    const scientific = magnitude !== 0 && !Number.isSafeInteger(value) && (magnitude < 0.001 || magnitude >= 1e6);
    const formatted = scientific
      ? magnitude.toExponential(significantDigits - 1)
      : Number.isSafeInteger(value) ? String(magnitude) : String(Number(magnitude.toPrecision(significantDigits)));
    const [mantissa, exponent] = formatted.split('e');
    return {
      negative: value < 0,
      mantissa: mantissa.includes('.') ? mantissa.replace(/0+$/, '').replace(/\.$/, '') : mantissa,
      exponent: exponent === undefined ? null : Number(exponent)
    };
  }
  const superscripts = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  function text(value, significantDigits = 4) {
    const p = parts(value, significantDigits);
    return (p.negative ? '−' : '') + p.mantissa + (p.exponent === null ? '' : ' × 10' + String(p.exponent).split('').map(c => superscripts[c]).join(''));
  }
  function mathml(value, significantDigits = 4) {
    const p = parts(value, significantDigits);
    const exponent = p.exponent === null ? '' : '<mo>×</mo><msup><mn>10</mn><mrow>' + (p.exponent < 0 ? '<mo>−</mo>' : '') + '<mn>' + Math.abs(p.exponent) + '</mn></mrow></msup>';
    return '<mrow>' + (p.negative ? '<mo>−</mo>' : '') + '<mn>' + p.mantissa + '</mn>' + exponent + '</mrow>';
  }
  function node(document, value, significantDigits = 4) {
    const p = parts(value, significantDigits);
    const element = (tag, content) => {
      const result = document.createElementNS('http://www.w3.org/1998/Math/MathML', tag);
      if (content !== undefined) result.textContent = String(content);
      return result;
    };
    const root = element('math'), row = element('mrow');
    if (p.negative) row.append(element('mo', '−'));
    row.append(element('mn', p.mantissa));
    if (p.exponent !== null) {
      const power = element('msup'), exponent = element('mrow');
      if (p.exponent < 0) exponent.append(element('mo', '−'));
      exponent.append(element('mn', Math.abs(p.exponent)));
      power.append(element('mn', 10), exponent);
      row.append(element('mo', '×'), power);
    }
    root.append(row);
    return root;
  }
  return { text, mathml, node };
})();
if (typeof module !== 'undefined') module.exports = CourseNumbers;
