# Chapter 23 research notes

Route: chapter authored by the assigned high-reasoning author after bounded source verification by `gpt-5.6-luna` at high reasoning. Verification date: 2026-09-08. The signal and Rust code are original.

- Elman, “Finding Structure in Time,” Cognitive Science 14(2) (1990), https://doi.org/10.1207/S15516709COG1402_1. Supports hidden-state recurrence as dynamic memory and context-dependent representations.
- Werbos, “Backpropagation through time,” Proceedings of the IEEE 78(10) (1990), https://doi.org/10.1109/5.58337. Supports differentiating an unrolled dynamic system and accumulating shared recurrent parameter gradients.
- Hochreiter and Schmidhuber, “Long Short-Term Memory,” Neural Computation 9(8) (1997), https://doi.org/10.1162/neco.1997.9.8.1735. Supports the decaying-error motivation and gated memory paths. The one-cell pedagogical implementation follows the modern input/forget/output/candidate equations rather than reproducing every historical detail.
- Tashman, “Out-of-sample tests of forecasting accuracy,” International Journal of Forecasting 16(4) (2000), https://doi.org/10.1016/S0169-2070(00)00065-0. Supports evaluation after the fitting period and fixed versus rolling origins.
- Bergmeir, Hyndman, and Koo, “A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction,” CSDA 120 (2018), https://doi.org/10.1016/j.csda.2017.11.003. Supports conditional validity of ordinary CV only under specific error assumptions; the lecture does not generalize that result.

Reported errors are teacher-forced one-step MSE with hidden state reset at the validation boundary, plus a persistence baseline. No multi-step rollout claim is made.


## Independent Astra review, 2026-09-08

GPT-6 Astra High independently read the full lesson, metadata, research, reference, starter, and Rustlings exercise/solution. Bounded GPT-5.6 Luna High primary-source and technical verification covered chapters 20–24; the researcher supplied checks and source findings, while Astra implemented the corrections.

Added an independent RNN recurrent-weight central-difference check alongside the LSTM candidate check, sequence shape/finite assertions, and qualified the tabular-order comparison. Added Gers, Schmidhuber and Cummins (2000), https://doi.org/10.1162/089976600300015015, for the modern forget gate. Chronological teacher-forced evaluation and the unfavorable LSTM-versus-persistence comparison remain explicit.

Final scoped formatting, offline strict Clippy, reference tests, and small release runs pass. The runnable starter passes its run and intentionally fails its new TODO test; the corresponding solved Rustlings exercise passes. See `guidance/astra-review-15-28.md` for exact gates and limitations.
