# Sensor learning lab: chapters 01–06

Six working experiments share one small, offline Rust package with no dependencies. Later chapters use completed earlier algorithms as supplied reference machinery; editing an earlier learner file does not silently break a later baseline. Original projects remain preserved separately.

From the repository root:

```sh
just lab 01
just lab-check 01
just solution 01 --check
just lab-test 01
node labs/s01-foundations/demo_checks.js
```

Substitute any chapter 01–06. `just lab NN` and `just lab-check NN` are shortcuts supplied by the course root.

The baseline runs and ordinary tests pass. A learner `--check` returns exit1 with `GOAL_NOT_MET:` until the substantial chapter goal is met. Ordinary runtime/input errors do not use that marker. `--solution --check` calls the separate completed implementation and passes. Experiment options are for runs; goal checks always use fixed, readable fixtures and reject experiment options. Passing numerics does not prove that a particular derivation was authored, nor does it grade your explanation.

| Chapter | Supplied working baseline | Your meaningful algorithm change |
|---|---|---|
| 01 | Compare three manually listed calibration rules | Retain a candidate-grid comparison; build central-difference gradients and repeated simultaneous updates |
| 02 | Completed numerical trainer and operation counter | Derive both MSE components in one data pass; report truthful work and check independent numerical agreement |
| 03 | Raw-sensor-only fit with complete three-feature/scaling plumbing | Accumulate all feature gradients; preserve training-only scaling through changed units and inference |
| 04 | Smoothed class-prior probability | Build the full mean-BCE logistic trainer; inspect logits, probabilities and thresholded decisions |
| 05 | Row split with an explicit repeated-machine leakage warning and fixed cutoff | Partition complete machines; select the minimum-cost validation threshold; audit held-out residuals |
| 06 | Full-batch SGD with useful/slow/unstable rates | Shuffled uneven minibatches, L2 and saved-best early stopping |

Edit `src/chNN.rs`; all checks call its real functions. The counterpart `src/solutions/chNN.rs` explains the completed algorithm. Supplied parsing, finite-input checks, fixtures, shuffle and basic scoring remain complete. Chapter01's `candidate_search` is retained and tested independently of numerical training. Later chapters provide exact executable controls for variations; options never require installing a CLI framework.

```sh
# Numerical trainer: rate, number of updates, or a different data condition.
just lab 01 --rate 0.01 --steps 100
just lab 01 --variant alternate
just lab 01 --variant outlier
# Analytical counterpart supports the same controls.
just lab 02 --variant alternate --steps 150
# Consistent training/inference unit change; optional --rate and --steps.
just lab 03 --load-scale 1000000
# Change only policy, labels, or add one inconsistent duplicate; optional rate/steps.
just lab 04 --threshold 0.8
just lab 04 --variant reversed
just lab 04 --variant contradiction
# Validation policy controls and a whole-machine partition variation.
just lab 05 --miss-cost 10
just lab 05 --candidate 0.45
just lab 05 --variant extra-visit
# Default06 compares five configurations; any options choose one custom run.
just lab 06 --rate 0.003
just lab 06 --batch 5 --seed 29 --l2 0.2
just lab 06 --rows 5 --batch 2 --epochs 1
just lab 06 --batch 5 --patience 5
just lab 06 --variant rotated
just lab 06 --variant shifted
```

Add `--solution` to run each completed comparison before your own implementation supports it. The supplied01 candidate baseline does not use training steps/rate to change parameters; the numerical replacement does. The supplied06 baseline explicitly rejects unsupported minibatch/L2/stopping requests. Zero epochs deliberately return the untouched model and epoch0 losses; invalid batch/rate/L2 settings still fail. Inputs are validated, options are bounded, and no default computation downloads data or runs a long workload.

## Fixtures and numerical conventions

01–02: five exact `(input,target)` calibration pairs. `alternate` changes targets to−0.5x+3; `outlier` adds3 to the original line's final target.03: eight combinations of raw sensor±1, load±1000 and vibration±1. Feature order is fixed.04: eight separable two-feature binary fault rows.05:60 synthetic machines with two visits each, a hidden effect on some machines, disclosed flipped labels on every thirteenth machine, and a forbidden post-inspection field.06:12 noisy regression training rows and6 validation rows; shuffled labels and saturation are explicit changed conditions.

Scalar models store `[weight,bias]`;03 uses three weights then bias;04 uses two weights then bias. MSE averages squared errors without one half. Finite differences use f64/h=1e-5 and small smooth fixtures, compared with absolute1e-6 plus relative1e-4. Stable BCE uses logits; extreme confident wrong logits±1000 give loss1000. Train/validation curves omit L2 even though the optimized objective includes it. L2 penalizes weight only; update count and RNG state persist across batches. Early stopping returns the saved model with the lowest observed validation MSE.

The05 Rust metric report uses zero as a stated display convention for empty ratio denominators and retains counts; the04 browser explicitly labels precision undefined when no positives are predicted. Neither represents evidence for a population metric. All fixtures test implementation behavior, not production performance. Work counts are arithmetic counts, not runtime benchmarks; probabilities are not calibrated risk estimates.

The browser check executes the actual01/02/04 scripts in Node's built-in `vm` against a small DOM double, asserting numbers, parameter bounds and resets. Visual layout and assistive-technology behavior also need the course's browser review.
