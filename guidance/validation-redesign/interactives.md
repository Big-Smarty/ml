# Independent validation of the 18 focused interactions

**Review date:** 2026-09-09
**Scope:** source-level review of the final browser-demo JavaScript, chapter markup, metadata, fixed arithmetic, exposed controls, evidence labels, readable alternatives, and dependency-free checks. Rendered layout, browser compatibility, hosting, and Rust goal checks belong to the root integration review.

## Result

**Pass after the corrections listed below.** The chapter metadata declares exactly 18 unique interaction IDs, and those IDs match the approved catalog in `guidance/ASSIGNMENTS.md`. Every assigned concept has one home interaction. None of the reviewed scripts writes progress, gates navigation, claims to execute Rust, or treats control movement as evidence of completion.

The final source provides native labeled controls, a deterministic reset, a numerical text or table representation of the important result, and an explicit boundary between browser arithmetic and Rust or hardware evidence. Dynamic results use an `output`, `aria-live`, or status region. SVGs provide a title and description and are accompanied by exact text or tables; tools without SVGs use tables or traces directly. The initial markup, authored worked examples, captions, and `noscript` text keep the relevant calculation readable when JavaScript is unavailable.

## Catalog match and executable coverage

| Section | Home chapter and interaction ID | Approved focus checked | Dependency-free command | What the command exercises |
|---|---|---|---|---|
| 01 | 01 `neuron-fit`; 02 `loss-gradient`; 04 `classifier-evidence` | neuron fitting; loss/gradient; probability/threshold/confusion | `node labs/s01-foundations/demo_checks.js` | Actual scripts in a small DOM: MSE and residuals, central differences and tiny-h degradation, simultaneous update, stable sigmoid/BCE, confusion counts including undefined precision, and resets |
| 02 | 07 `xor-hidden`; 08 `autodiff-tape`; 09 `tensor-digits` | XOR/hidden representation; autodiff graph; tensor/digit predictions | `node labs/s02-neural-networks/check_interactives.cjs` | Exported calculations: XOR decisions, reverse adjoint accumulation, dense shapes, prototype-score softmax, momentum/Adam state, zero-gradient boundary |
| 03 | 12 `sampling-tool`; 13 `split-tool`; 14 `geometry-tool` | sampling/bootstrap/calibration; split/leakage; classical-model geometry | `node labs/s03-tabular/verify-demos.cjs` | Actual scripts in a small DOM: seeded row/machine bootstrap endpoints, calibration/Brier/bin counts, all five split cases, distance/tree/RBF/projection/cluster views, assignment transition, and resets |
| 04 | 20 `convolution-microscope`; 22 `embedding-neighborhood` | convolution/pooling; embedding/recommendation neighbors | `node labs/s04-deep-learning/check-demos.cjs` | Actual scripts in a small DOM: convolution geometry/products, ReLU, pooling winner and backward contributions; dot/cosine/distance rankings, Recall@2, Precision@2, nDCG@2, decoder result, and resets |
| 05 | 25 `memory-gemm`; 27 `threads-lanes` | memory/GEMM tracing; threads/SIMD lanes | `node labs/s05-cpu/check_demos.cjs` | Exported `trace`, `state`, `partition`, `lanes`, `rounding`, and `dispatch` calculations across traversal orders, shard counts, lane widths, rounding groups, and feature cases |
| 06 | 29 `gpu-workgroups` | workgroups/bounds/tiling/barriers/transfers | `node chapters/29/demo-check.cjs` | Actual global helpers through `vm`: dispatch counts, reduction levels, tiled offsets/padding, and resident/readback byte counts |
| 07 | 33 `token-loss-lab`; 35 `attention-lab` | tokens/next-token loss; causal attention/decoder flow | `node labs/s07-language-models/check_interactives.cjs` | Exported calculations: byte/scalar/BPE units, stable target loss, causal counterfactual, attention values, residual path, and normalization |
| 08 | 40 `inference-budget` | inference memory/work and quantization | `node labs/s08-adaptation/test-demo.cjs` | Exported `inferenceBudgetS08`: cache/work/score bytes, signed int4/int8 codes, storage and error, one-token tile, and invalid controls |
| 09 | 48 `routing-capacity` | expert routing/capacity/overflow/balance | `node chapters/48/demo-check.cjs` | Exported routing calculation: balanced loads, capacity drops, collapse, exact tie-first routing, mean probabilities, raw balance, and alpha scaling |

The nine commands above passed together. `node --check` also passed for all 18 `demo.js` files. A separate metadata assertion found 18 entries, 18 unique IDs, and exact set equality with the approved catalog.

## Independent arithmetic review

The executable checks were supplemented by direct formula review. The following boundaries were specifically confirmed:

- Chapter 01 uses the five-point line `y=2x+1`, giving MSE 9 at `[w,b]=[0,0]`, gradient `[-8,-2]`, and MSE 3.52 after the displayed rate-0.1 update.
- Chapter 02's analytical gradient is `[4(w-2),2(b-1)]`; the central difference uses independent `w±h` and `b±h` probes.
- Chapter 04 uses stable sigmoid and logit-form BCE. Threshold decisions use `p≥threshold`, and precision is now reported as undefined when no positive decisions exist.
- Chapter 07's two hand-chosen hidden features implement XOR at the reset threshold while the single linear boundary reaches only three of four corners. Chapter 08 accumulates both uses of `x` in `y=x²+x`, giving `dy/dx=1+2x`. Chapter 09's prototype logits are negative squared distances with the common input norm omitted; its softmax is maximum-shifted, and both optimizer states remain fixed when both displayed gradients are zero.
- Chapter 12 resamples 18 rows or six whole machines with replacement, reports the disclosed seeded percentile endpoints, applies confidence scaling in log-odds space, and recomputes four reliability bins without fabricating empty-bin observations. Chapter 13's row/group/time/strict membership counts and frozen diagnostic denominators agree with the stated split predicates. Chapter 14's distance, RBF, projection/reconstruction, and center-distance equations are consistent; the cluster control now reaches a case where center 2 wins.
- Chapter 20 performs neural-network cross-correlation with the documented padding/stride convention. The displayed first pool winner, ReLU derivative convention, and nine shared-kernel gradient contributions agree. Chapter 22's candidate exclusion, dot/cosine/distance values, single-relevant-item retrieval metrics, and fixed decoder are internally consistent.
- Chapter 25 visits each of 18 matrix products exactly once in all three traversal orders and produces `[30,36,42,66,81,96]`. Chapter 27 covers all rows once, preserves whole-batch weighting for unequal shards, handles scalar tails, and distinguishes hypothetical dispatch features from detected hardware.
- Chapter 29's ceiling dispatch, guarded lanes, reduction strides, row-major GEMM offsets, padding counts, and application-buffer transfer totals agree. Its SVG now redraws with the selected vector length instead of retaining the original 67-element picture.
- Chapter 33's UTF-8 byte, Unicode-scalar, and one-rule BPE sequences and vocabulary-dependent target loss agree. Chapter 35 excludes masked scores before the maximum and softmax normalizer, then applies the stated value sum, residual, and per-token normalization.
- Chapter 40 counts `2LtD` K/V values, cumulative full-prefix versus cached layer-token rows, per-head row/tile score workspace, two-head square score storage, and signed symmetric int4/int8 weight storage consistently.
- Chapter 48 counts attempted routes before capacity admission, uses deterministic first-index ties, holds attempted frequencies in the Switch-style balance term, and changes alpha without retroactively changing fixed routes.

## Corrections made during review

Owners corrected each actionable issue before this report was finalized:

1. Chapter 02 relabeled its drawn line as a finite-difference slope guide; it was not the actual secant segment between the displayed `w±h` probes.
2. Chapter 04 stopped displaying undefined precision as numeric zero when no positive prediction exists.
3. Chapter 08 clarified that the reverse adjoints of `x` and `m`, rather than their forward values, start at zero.
4. Chapter 09 made the optimizer explanation correct at multiplier zero and changed the numeric grid's text colors to retain readable contrast on mid-gray cells.
5. Chapter 14 extended the center control beyond the all-center-1 region and made its SVG description match the selected view.
6. Chapter 27 replaced “No x86 SIMD features” with the narrower and accurate “Neither AVX2 nor FMA.”
7. Chapter 29 made its workgroup SVG, title, and description update with `N`; the earlier fixed 67-element picture contradicted other selected values.
8. Chapter 40 labeled row and tile score-memory counts as single-query, single-head quantities so they are not confused with the displayed two-head square tensor total.

## Validation boundary

These checks establish deterministic arithmetic, metadata identity, source-level control wiring, reset behavior, and the availability of readable equivalents. They do not establish rendered layout, zoom/reflow, keyboard behavior in a real browser, color rendering under every theme, or successful Rust/GPU execution. Those are separate integration checks and should remain separate from claims made by the browser illustrations.
