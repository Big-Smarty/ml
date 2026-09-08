// Run with node tools/test_highlighting.cjs; no npm packages or browser required.
const assert = require('node:assert/strict');
const Prism = require('../site/assets/prism.js');
const cases = {
  rust: `// keep < and & literal\nfn view<'a>(s: &'a str) -> &'a str { println!(r#"<hello>&"#); s }`,
  wgsl: '@group(0) @binding(0) var<storage, read> input: array<f32>;\n@compute @workgroup_size(64) fn main() { let x = 1u; }',
  bash: 'cargo run --manifest-path "projects/ch01/Cargo.toml"\n# a comment\ngit commit -m "learn < & >"',
  toml: '[package]\nname = "ch01"\npublish = false',
  python: 'def predict(x):\n    return 2 * x + 1',
  javascript: 'const predict = x => 2 * x + 1;',
};
for (const [language, source] of Object.entries(cases)) {
  assert.ok(Prism.languages[language], `bundled ${language} grammar`);
  const output = Prism.highlight(source, Prism.languages[language], language);
  assert.match(output, /class="token /, `${language} has colored tokens`);
  const plain = output.replace(/<[^>]*>/g, '').replace(/&lt;/g, '<').replace(/&amp;/g, '&');
  assert.equal(plain, source, `${language} highlighting preserves copyable source`);
}
const rust = Prism.highlight(cases.rust, Prism.languages.rust, 'rust');
assert.match(rust, /token lifetime/, 'Rust lifetime is not a character literal');
assert.match(rust, /token macro/, 'Rust macro recognized');
assert.match(rust, /token string/, 'Rust raw string recognized');
console.log('PASS: six local grammars, Rust lifetimes/macros/raw strings, and exact text preservation.');
