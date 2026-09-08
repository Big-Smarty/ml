# Chapter 18 research notes

## Route and verification

The Sol High author delegated a bounded primary-source research task to GPT-5.6 Luna High. The researcher verified the source landing pages on 2026-09-08 and returned derivations, numerical examples, and standard-library implementation risks. The author implemented and tested the final algorithms and recalculated all displayed fixture results.

## Evidence used

- MacQueen, [“Some Methods for Classification and Analysis of Multivariate Observations”](https://digicoll.lib.berkeley.edu/record/113015?v=pdf), is the original source for k-means terminology and alternating/online mean updates.
- Lloyd, [“Least Squares Quantization in PCM”](https://doi.org/10.1109/TIT.1982.1056489), supports the least-squares quantization interpretation and alternating nearest-center procedure.
- Dempster, Laird, and Rubin, [“Maximum Likelihood from Incomplete Data via the EM Algorithm”](https://academic.oup.com/jrsssb/article/39/1/1/7027539), supplies the general EM framework and likelihood behavior.
- Wolfe, [“Pattern Clustering by Multivariate Mixture Analysis”](https://doi.org/10.1207/s15327906mbr0503_6), supplies early multivariate mixture clustering context.

## Arithmetic and implementation decisions

The k-means worked example `[1,2,8,9]` with centers 1 and 9 updates to 1.5 and 8.5, with inertia one. The responsibility example uses standard-normal density values 0.39894 and 0.05399, giving 0.8808 and 0.1192. Those values then update symmetric means to 0.2384 and 1.7616.

The GMM uses diagonal variance to keep covariance inversion out of the assigned lesson. Component terms remain in log space and `responsibilities` returns their normalized per-point probabilities. These differ from fitted component weights. A variance floor prevents single-point collapse. A dead component receives zero weight and weights are renormalized. K-means recomputes assignments after its last center update, preventing stale returned assignments. Tests check final summed inertia, normalized responsibilities and weights, nondecreasing summed fixture log-likelihood within tolerance, a variance floor, and a larger score for a distant point.

## Excluded extensions

Full covariance, k-means++ randomness, multiple restarts, automatic k selection, LOF, and calibrated thresholds were excluded. The chapter names each limitation and retains a complete deterministic demonstration of the assigned mechanisms.



## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 15–19; the researcher supplied checks and source findings, while Astra implemented the corrections.

Computed normalized responsibilities directly from max-shifted relative component masses, avoiding loss of the normalizer at huge common log offsets. Added a final-center reassignment fixture and overflow rejection. The reference now uses `squared_distance` consistently with its Rustlings primitive, names the summed likelihood `log_likelihood_sum`, and calls the unlabeled fixture `FEATURE_POINTS`. Diagonal covariance, dead-component handling, and held-out threshold requirements remain explicit.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.
