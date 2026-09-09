# Visual and command revision: validation

Validated 2026-09-09 against the completed revision following a217c48.

## What changed

The dark theme uses neutral gray surfaces (#111111, #191919, #222222) with light gray text. The light theme and persisted reader choice remain available. Body text is 18 px with 1.7 leading and a 70ch lesson measure. Numeric tables scroll independently; API contracts use operation names, separate signatures, and explanations in definition lists. Worked examples, exercises, callouts, and equations have distinct structural treatments.

All 56 lessons now contain native Presentation MathML: 5,631 static expressions, including 103 display equations. All 18 demos render mathematical results with the same semantics and typography. Literal Rust code, source excerpts, serialized token IDs, paths, and terminal output retain their code/data form. Glossary definitions describe mathematical ideas in words rather than emitting unformatted formulas.

Active learner instructions use just. Added dependency-fetch, optimized-build, lint, assembly, real-GPU-test, and reference-GPU-test recipes; test and reference commands preserve release/offline execution and literal argument forwarding. There are no direct Cargo run/test/check/build/fmt/clippy/rustc/fetch commands in lesson HTML, site pages, the root README, or section lab READMEs.

The design rationale, 27 primary empirical studies, standards, and limits are in [VISUAL_LEARNING_RESEARCH.md](VISUAL_LEARNING_RESEARCH.md). These changes have not been tested for learning gains with course participants.

## Executed checks

| Check | Result |
|---|---|
| Full content, metadata, source excerpt, local link and step audit | 56 chapters, 426 steps, zero findings |
| Independent mathematical review across chapters 01–19, 20–39, and 40–56 | All fractions, script bases, display formulas, code boundaries, and reviewed demo diffs pass; no unresolved P1/P2 findings |
| MathML regression validation | Balanced/XML-valid roots, token leaves, required arity, no nested math roots, and complete scripted bases |
| Python course regression checks | Pass: safe metadata, command dispatch, source archive, non-mutating build, table/math wrappers, and MathML regression cases |
| Nine existing Node demo harnesses | All pass against the actual scripts and calculation helpers |
| JavaScript syntax | All 18 demo scripts and shared reader script pass |
| Browser full-chapter layout | All 56 chapters at viewport widths 320, 390, 768, and 1366 px; no page-level horizontal overflow after correcting the long chapter-title case |
| Browser inline math geometry | No vertical clipping at 390 or 1366 px in the tested full-chapter state |
| Expanded answers and references | Chapters 01, 09, 11, 33, 35, 40, 48: all 78 main-content disclosures opened at 390 px; no page overflow or clipped inline math |
| Interactive rendering | All 18 tools visited and controls exercised; changed results retain native math wrappers. Stepped traces and explicit numeric/pointer interactions were checked where changing a selector alone did not alter result text |
| Keyboard scrolling | A wide Chapter 07 numeric table accepts focus and ArrowRight advances its horizontal scroll position |
| Script-free reading | Temporary copies of Chapters 09, 11, 40 with every page script removed show all 6/9/9 learning steps and 122/140/90 native math roots, with no page overflow at 390 px |
| Visual inspection | Neutral dark and light pages, inline variables, fractions, roots, matrices, worked answers, numeric tables, and the replacement API reference layout inspected in the available browser |
| Diff hygiene | Pass |

Recipe execution in this revision passed: just recipes-test, just lab 01, just solution 01 --check, just lab-test for representatives 01/07/12/20/25/29/33/40/47 covering all nine packages, just lint 01, just asm 28, just lab-build 25, just deps 29, just reference 12, and just reference-test 12. Ordinary section tests total 172 passing tests. The commands just gpu-test 29 and just reference-gpu-test 36 passed on the available AMD Radeon RX 6950 XT; this verifies the new command paths on that device. No Rust algorithm source changed in this revision.

## Limits of this verification

The executed browser checks use the available Chromium-based in-app browser. Screen-reader/browser combinations, actual browser zoom, forced colors, and operating-system text-spacing overrides were not tested here. The 320 px layout check is a narrow-layout test, not a substitute for all zoom or assistive-technology checks. The script-free copies test static reading without app/demo execution; they are not an audit of every browser's JavaScript-disable mode. Contrast ratios are calculated for the documented palette pairs, not a claim of complete WCAG conformance. Mathematical rendering and spoken output can vary with the browser, font, operating system, and assistive technology.

