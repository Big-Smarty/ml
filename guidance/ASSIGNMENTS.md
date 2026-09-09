# Active redesign assignments

The approved plan assigns exactly one GPT-6 Astra High implementation owner to each existing section. Root handles website, shared conventions, integration and private OpenAI Pages publication. Research uses GPT-5.6 Sol High. Each owner controls only the listed chapter directories, its independent lab package and section report. Original projects remain preserved reference code. Historical assignment/review reports describe earlier revisions.

| Section | Chapters | Lab package | Project | Interactive tools (home chapter) |
|---|---|---|---|---|
| 01 | 01–06 | s01-foundations | Sensor calibration to fault classification and evaluation | neuron fitting (01); loss/gradient (02); probability/threshold/confusion (04) |
| 02 | 07–11 | s02-neural-networks | XOR to autodiff, tensors and digit classifiers | XOR/hidden representation (07); autodiff graph (08); tensor/digit predictions (09) |
| 03 | 12–19 | s03-tabular | One imperfect maintenance dataset with groups/time/leakage, compare ≥3 prior models and explain residual mistakes | sampling/bootstrap/calibration (12); split/leakage (13); classical-model geometry (14) |
| 04 | 20–24 | s04-deep-learning | Vision, representations, sequences and recommendations | convolution/pooling (20); embedding/recommendation neighbors (22) |
| 05 | 25–28 | s05-cpu | Measure and improve one correct dense kernel | memory/GEMM tracing (25); threads/SIMD lanes (27) |
| 06 | 29–32 | s06-gpu | Supplied GPU host to actual kernels, training and measured optimization | workgroups/bounds/tiling/barriers/transfers (29) |
| 07 | 33–39 | s07-language-models | Byte baseline, full BPE comparison, attention, decoder, data, training and checkpointed capstone | tokens/next-token loss (33); causal attention/decoder flow (35) |
| 08 | 40–46 | s08-adaptation | Caching, quantization, efficient attention, LoRA, retrieval, RL and preferences | inference memory/work and quantization (40) |
| 09 | 47–56 | s09-advanced | Architecture/generation/multimodal/deployment studies then actual sparse MoE LM | expert routing/capacity/overflow/balance (48) |

Lab CLI and metadata are fixed by AUTHORING.md. Tool home may move within a bundle if justified, but retain exactly the conceptual tools above (18 total) and report final homes. Later chapters link directly to the earlier tool or use a section-shared file with root coordination. Do not replace full substantive topics with mere tool references.
