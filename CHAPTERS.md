# Course map

The redesigned course keeps all 56 chapters and the nine original topic bundles. Each chapter has a working Rust baseline, a learner-owned algorithm, inspectable checks, hints and a separate completed solution. Sessions are approximate guides, not time limits.

Use `just lab NN` to run a baseline, `just lab-check NN` to check the goal, and `just solution NN --check` after an implementation attempt. Numerical agreement alone does not establish any source-review criteria listed in the lesson.

## 01. First principles

Project: Calibrate a sensor, then detect faults. Deliverable: A trained predictor and a report of its mistakes.

Lab: `labs/s01-foundations`. Design matrix: [section report](guidance/redesign/section-01.md).

### 01. One neuron learns from data

Learner source: [labs/s01-foundations/src/ch01.rs](labs/s01-foundations/src/ch01.rs). 2 sessions; 6 learning steps.

Goal: Replace candidate search with numerical-gradient training.

Topics: Prediction; weights and bias; squared loss; numerical slopes; training and inference.

Sessions: Compare calibration rules; Build numerical training.

### 02. Why learning works

Learner source: [labs/s01-foundations/src/ch02.rs](labs/s01-foundations/src/ch02.rs). 2 sessions; 6 learning steps.

Goal: Replace numerical probing with a one-pass analytical gradient and verify agreement.

Topics: Derivatives; gradients; analytical updates; finite-difference checks.

Sessions: Derive a cheaper gradient; Check agreement and preserve behavior.

### 03. Multiple inputs

Learner source: [labs/s01-foundations/src/ch03.rs](labs/s01-foundations/src/ch03.rs). 2 sessions; 6 learning steps.

Goal: Extend one-feature gradients to all three inputs and verify scaled inference.

Topics: Vectors; dot products; linear regression; feature scaling.

Sessions: Use all sensor measurements; Keep the model consistent when units change.

### 04. Classification

Learner source: [labs/s01-foundations/src/ch04.rs](labs/s01-foundations/src/ch04.rs). 2 sessions; 6 learning steps.

Goal: Replace a constant prior with a full stable logistic-gradient trainer.

Topics: Probability; sigmoid; likelihood; stable binary cross-entropy.

Sessions: Give outcomes probability; Train and inspect a stable classifier.

### 05. Measuring learning

Learner source: [labs/s01-foundations/src/ch05.rs](labs/s01-foundations/src/ch05.rs). 2 sessions; 7 learning steps.

Goal: Replace a contaminated row split and fixed cutoff with grouped validation selection.

Topics: Splits; leakage; metrics; thresholds; baselines.

Sessions: Define a trustworthy partition; Choose and audit a decision policy.

### 06. Reliable optimization

Learner source: [labs/s01-foundations/src/ch06.rs](labs/s01-foundations/src/ch06.rs). 2 sessions; 7 learning steps.

Goal: Extend full-batch SGD with shuffled minibatches, L2 and saved-best early stopping.

Topics: Learning rates; minibatches; regularization; generalization.

Sessions: Make update size and batches explicit; Compare regularization and saved models.

## 02. Build a neural network

Project: Build a neural network you can explain. Deliverable: A tested digit classifier with gradients you can inspect.

Lab: `labs/s02-neural-networks`. Design matrix: [section report](guidance/redesign/section-02.md).

### 07. From a neuron to XOR

Learner source: [labs/s02-neural-networks/src/ch07.rs](labs/s02-neural-networks/src/ch07.rs). 2 sessions; 6 learning steps.

Goal: Implement and verify full XOR backpropagation.

Topics: Nonlinearities; hidden layers; manual backpropagation.

Sessions: Learn both layers; Inspect what fitting means.

### 08. Automatic differentiation

Learner source: [labs/s02-neural-networks/src/ch08.rs](labs/s02-neural-networks/src/ch08.rs). 2 sessions; 6 learning steps.

Goal: Implement scalar reverse mode with shared dependencies and numerical transfer checks.

Topics: Computation graphs; reverse mode; accumulated gradients.

Sessions: Implement reverse accumulation; Use the engine.

### 09. Tensors and batches

Learner source: [labs/s02-neural-networks/src/ch09.rs](labs/s02-neural-networks/src/ch09.rs). 2 sessions; 6 learning steps.

Goal: Implement checked dense forward and backward with rectangular transfer tests.

Topics: Shapes; contiguous storage; matrix operations; batched gradients.

Sessions: Compute a batch; Differentiate a batch.

### 10. Recognizing digits with a linear model

Learner source: [labs/s02-neural-networks/src/ch10.rs](labs/s02-neural-networks/src/ch10.rs). 2 sessions; 6 learning steps.

Goal: Train a real ten-class linear classifier and inspect ambiguous held-out errors.

Topics: IDX format; MNIST; softmax; multiclass evaluation.

Sessions: Train pixel-to-class weights; Evaluate and transfer.

### 11. An MNIST neural network

Learner source: [labs/s02-neural-networks/src/ch11.rs](labs/s02-neural-networks/src/ch11.rs). 3 sessions; 9 learning steps.

Goal: Train a nonlinear digit MLP with momentum and Adam and verify exact next-update restore.

Topics: Initialization; momentum; Adam; debugging; checkpoints.

Sessions: Learn hidden features; Implement momentum and Adam; Diagnose and resume.

## 03. Beyond neural networks

Project: Investigate an imperfect maintenance dataset. Deliverable: A fair comparison of models and an error analysis.

Lab: `labs/s03-tabular`. Design matrix: [section report](guidance/redesign/section-03.md).

### 12. Statistics and uncertainty

Learner source: [labs/s03-tabular/src/ch12.rs](labs/s03-tabular/src/ch12.rs). 3 sessions; 6 learning steps.

Goal: Implement whole-machine percentile bootstrap and counted reliability bins.

Topics: Sampling; confidence intervals; bootstrap; calibration.

Sessions: What can eighteen readings tell us?; Which observations must move together?; Does “0.8” behave like eight in ten?.

### 13. Real tabular data

Learner source: [labs/s03-tabular/src/ch13.rs](labs/s03-tabular/src/ch13.rs). 3 sessions; 6 learning steps.

Goal: Implement training-fitted median/scaling/vocabulary and four legal split policies.

Topics: Missing values; categories; grouped splits; temporal splits.

Sessions: What was known when the prediction was made?; Which column means “loaded”?; Which holdout matches deployment?.

### 14. Nearest neighbors and naive Bayes

Learner source: [labs/s03-tabular/src/ch14.rs](labs/s03-tabular/src/ch14.rs). 3 sessions; 7 learning steps.

Goal: Implement complete kNN voting and fitted Gaussian priors/variances.

Topics: Distance; scaling; nearest neighbors; Gaussian naive Bayes.

Sessions: Which machine reading is actually close?; Can each class be represented by a distribution?; Which assumption explains the mistakes?.

### 15. Trees, forests, and boosting

Learner source: [labs/s03-tabular/src/ch15.rs](labs/s03-tabular/src/ch15.rs). 4 sessions; 8 learning steps.

Goal: Implement recursive CART, bagging, random feature forests and residual boosting.

Topics: Decision trees; bagging; random forests; gradient boosting.

Sessions: Which question separates a mixed node?; What changes when trees see different rows?; What should the next tree try to correct?; Separate capacity from validation improvement.

### 16. Margins and kernels

Learner source: [labs/s03-tabular/src/ch16.rs](labs/s03-tabular/src/ch16.rs). 3 sessions; 6 learning steps.

Goal: Implement regularized hinge updates and kernel-perceptron coefficient fitting.

Topics: Hinge loss; linear SVM; kernel similarity.

Sessions: How far is a correct point from the boundary?; Implement the soft-margin updates; Can similarity make a boundary curve?.

### 17. PCA and numerical linear algebra

Learner source: [labs/s03-tabular/src/ch17.rs](labs/s03-tabular/src/ch17.rs). 4 sessions; 8 learning steps.

Goal: Implement full covariance, orthogonal power components and reconstruction diagnostics.

Topics: Covariance; eigenvectors; power iteration; projections; conditioning.

Sessions: How much independent variation is in three sensors?; Which direction keeps the most variance?; How do several coordinates reconstruct a row?; When can a correct answer still wobble?.

### 18. Clustering and density

Learner source: [labs/s03-tabular/src/ch18.rs](labs/s03-tabular/src/ch18.rs). 4 sessions; 8 learning steps.

Goal: Implement Lloyd updates, stable EM and inspect changed-input density scores.

Topics: K-means; Gaussian mixtures; EM; anomaly scores.

Sessions: Can unlabeled readings reveal operating regimes?; What if membership is uncertain?; Fit means, variances and weights from fractional counts; How unusual is this observation under the fitted density?.

### 19. Tabular ML capstone

Learner source: [labs/s03-tabular/src/ch19.rs](labs/s03-tabular/src/ch19.rs). 3 sessions; 6 learning steps.

Goal: Implement frozen group/time cross-validation and six-candidate pooled-cost selection.

Topics: Cross-validation; hyperparameter search; fair baselines; error analysis.

Sessions: What exactly will the final comparison claim?; Search the declared budget without moving the goalposts; Refit once and inspect the held-out result.

## 04. Broader deep learning

Project: Explore different kinds of data. Deliverable: Small vision, sequence and recommendation experiments.

Lab: `labs/s04-deep-learning`. Design matrix: [section report](guidance/redesign/section-04.md).

### 20. Convolutional networks

Learner source: [labs/s04-deep-learning/src/ch20.rs](labs/s04-deep-learning/src/ch20.rs). 4 sessions; 12 learning steps.

Goal: Replace the pointwise/subsampling model with shared convolution, max pooling and matched backward.

Topics: Convolution; padding and stride; pooling; gradients.

Sessions: Implement shared convolution; Pool values and route credit; Differentiate the whole classifier; Compare spatial reductions.

### 21. Deeper vision models

Learner source: [labs/s04-deep-learning/src/ch21.rs](labs/s04-deep-learning/src/ch21.rs). 3 sessions; 9 learning steps.

Goal: Replace centering/canonical-view baselines with standardization, its backward rule and translations.

Topics: Residual connections; normalization; augmentation.

Sessions: Normalize one image with named axes; Differentiate both residual routes; Implement and compare translations.

### 22. Learning representations

Learner source: [labs/s04-deep-learning/src/ch22.rs](labs/s04-deep-learning/src/ch22.rs). 3 sessions; 9 learning steps.

Goal: Train the autoencoder encoder, complete shared InfoNCE gradients and implement cosine selection.

Topics: Autoencoders; bottlenecks; contrastive learning.

Sessions: Train the bottleneck encoder; Differentiate both contrastive views; Choose and verify a neighborhood rule.

### 23. Sequences and forecasting

Learner source: [labs/s04-deep-learning/src/ch23.rs](labs/s04-deep-learning/src/ch23.rs). 4 sessions; 12 learning steps.

Goal: Extend one-lag/truncated/persistence baselines to full recurrent forward, BPTT and LSTM rollout.

Topics: RNNs; BPTT; LSTMs; temporal evaluation.

Sessions: Unroll an ordered hidden state; Implement full RNN BPTT; Extend LSTM credit through both states; Evaluate observed-input and free-running forecasts.

### 24. Recommendation systems

Learner source: [labs/s04-deep-learning/src/ch24.rs](labs/s04-deep-learning/src/ch24.rs). 3 sessions; 9 learning steps.

Goal: Train both factor tables with simultaneous pairwise gradients and replace nCG with binary nDCG.

Topics: Embeddings; matrix factorization; ranking metrics.

Sessions: Train a shared preference space; Implement ranking evaluation; Compare and defend a section experiment.

## 05. Make the CPU faster

Project: Make one correct kernel faster. Deliverable: A reproducible dense-kernel benchmark.

Lab: `labs/s05-cpu`. Design matrix: [section report](guidance/redesign/section-05.md).

### 25. Measure before optimizing

Learner source: [labs/s05-cpu/src/ch25.rs](labs/s05-cpu/src/ch25.rs). 2 sessions; 6 learning steps.

Goal: Produce a configured local measurement and explain its limits.

Topics: Floating point; profiling; memory costs; arithmetic intensity.

Sessions: Measure a correct dense call; Interpret memory and profiling evidence.

### 26. Fast matrix multiplication

Learner source: [labs/s05-cpu/src/ch26.rs](labs/s05-cpu/src/ch26.rs). 2 sessions; 6 learning steps.

Goal: Implement clipped tiles, verify unfamiliar shapes and report measured variants.

Topics: Loop ordering; cache locality; blocking; allocation reuse.

Sessions: Change traversal and preserve storage; Tile and measure the same product.

### 27. Multithreaded training and inference

Learner source: [labs/s05-cpu/src/ch27.rs](labs/s05-cpu/src/ch27.rs). 3 sessions; 9 learning steps.

Goal: Verify simultaneous training and implement a measured inference threshold.

Topics: Partitioning; scoped threads; gradient reduction; reproducibility.

Sessions: Partition dense inference; Reduce private training sums; Train and measure a policy.

### 28. SIMD

Learner source: [labs/s05-cpu/src/ch28.rs](labs/s05-cpu/src/ch28.rs). 3 sessions; 9 learning steps.

Goal: Implement guarded AVX-512F and report actual backend evidence.

Topics: Auto-vectorization; AVX2 and FMA; feature detection; tails; AVX-512 extension.

Sessions: Make lane grouping explicit; Guard explicit SIMD; Extend and measure.

## 06. Write GPU kernels

Project: Move a computation onto the GPU. Deliverable: A verified GPU pipeline and measured performance report.

Lab: `labs/s06-gpu`. Design matrix: [section report](guidance/redesign/section-06.md).

### 29. GPU fundamentals

Learner source: [labs/s06-gpu/src/ch29.rs](labs/s06-gpu/src/ch29.rs). 2 sessions; 8 learning steps.

Goal: Implement guarded parallel SAXPY and verify actual hardware output against its CPU oracle.

Topics: Adapter discovery; wgpu; WGSL; buffers; dispatch; readback.

Sessions: From serial chunks to independent invocations; Prove dispatch and synchronized readback.

### 30. Reductions and tiled matrix multiplication

Learner source: [labs/s06-gpu/src/ch30.rs](labs/s06-gpu/src/ch30.rs). 2 sessions; 7 learning steps.

Goal: Build cooperative reduction and tiled GEMM; combine parity with a barrier/ownership explanation.

Topics: Workgroups; shared memory; barriers; edge dimensions.

Sessions: Cooperate on a staged sum; Cooperate on rectangular matrix tiles.

### 31. Train on the GPU

Learner source: [labs/s06-gpu/src/ch31.rs](labs/s06-gpu/src/ch31.rs). 2 sessions; 8 learning steps.

Goal: Complete CPU/WGSL backpropagation and train the 2–4–1 network with resident buffers.

Topics: Forward and backward kernels; updates; residency.

Sessions: Derive and check full nonlinear training; Train with resident WGSL stages.

### 32. GPU performance

Learner source: [labs/s06-gpu/src/ch32.rs](labs/s06-gpu/src/ch32.rs). 2 sessions; 9 learning steps.

Goal: Implement a fused WGSL kernel and a warmed, alternating, correctness-backed hardware experiment.

Topics: Asynchronous execution; timing; fusion; mixed precision; Vulkan extension.

Sessions: Design an honest GPU measurement; Fuse and evaluate precision choices.

## 07. Train your language model

Project: Train a small language model. Deliverable: A byte-level decoder with reproducible training and checkpoints.

Lab: `labs/s07-language-models`. Design matrix: [section report](guidance/redesign/section-07.md).

### 33. Language modeling

Learner source: [labs/s07-language-models/src/ch33.rs](labs/s07-language-models/src/ch33.rs). 3 sessions; 5 learning steps.

Goal: Construct smoothed conditional successor counts and distinguish byte/scalar targets; Implement and gradient-check simultaneous embedding/output/bias learning; Compare fixed-budget widths on untouched text and interpret errors.

Topics: Byte models; n-grams; embeddings; next-token loss.

Sessions: What exactly follows the byte?; How can a learned row predict every byte?; Did learning help the reserved text?.

### 34. Tokenization

Learner source: [labs/s07-language-models/src/ch34.rs](labs/s07-language-models/src/ch34.rs). 3 sessions; 5 learning steps.

Goal: Fit deterministic document-aware merge rules with overlaps and ties; Implement ranked encoding and verify byte-exact serialization round trips; Measure byte/BPE compression on multilingual held-out text.

Topics: UTF-8; byte tokens; BPE training; serialization.

Sessions: Which text units does a model actually receive?; Can every learned ID return to the original bytes?; Is compression useful beyond the training snippet?.

### 35. Attention

Learner source: [labs/s07-language-models/src/ch35.rs](labs/s07-language-models/src/ch35.rs). 3 sessions; 5 learning steps.

Goal: Implement scaled causal attention and verify future invariance; Derive every QKV backward path and check all elements numerically; Compose two independent heads and explain feature packing.

Topics: Queries; keys; values; causal masking; gradients.

Sessions: What does one position ask another position for?; Where does the error travel through a weighted mixture?; How do independent heads fit inside a decoder?.

### 36. A decoder Transformer

Learner source: [labs/s07-language-models/src/ch36.rs](labs/s07-language-models/src/ch36.rs). 4 sessions; 7 learning steps.

Goal: Assemble token/position and pre-normalized attention residual paths; Derive and verify LayerNorm input/gain/bias backward; Complete FFN integration and match a full two-layer decoder; Overfit a tiny sequence and compare controlled depth variation.

Topics: Residual paths; normalization; positions; feed-forward blocks.

Sessions: How can attention become a complete next-token model?; Why normalize features separately for each token?; Which weights does the block actually own?; Can a complete decoder overfit a tiny pattern?.

### 37. Language-model data

Learner source: [labs/s07-language-models/src/ch37.rs](labs/s07-language-models/src/ch37.rs). 3 sessions; 6 learning steps.

Goal: Inspect cleaning and retain raw-source provenance through parsing; Implement near-duplicate Jaccard decisions with removal links; Review threshold choices and state document-split/contamination limits.

Topics: Provenance; cleaning; deduplication; document splits; contamination.

Sessions: What exactly are we allowed to call the training data?; Which changed copies should count as duplicates?; What does a clean split actually establish?.

### 38. Training at a larger scale

Learner source: [labs/s07-language-models/src/ch38.rs](labs/s07-language-models/src/ch38.rs). 4 sessions; 7 learning steps.

Goal: Implement complete AdamW moments and proposed parameter update; Combine unequal microbatch gradients then globally clip; Implement the schedule and prove next-update checkpoint equivalence; Compare optimizer configurations and account for excluded memory.

Topics: AdamW; schedules; accumulation; memory; resumable training.

Sessions: How can two differently scaled gradients use one learning rate?; Which gradients belong to the same update?; Which step selects the next learning rate?; How much storage is hidden behind the parameter count?.

### 39. Scratch-trained language-model capstone

Learner source: [labs/s07-language-models/src/ch39.rs](labs/s07-language-models/src/ch39.rs). 4 sessions; 8 learning steps.

Goal: Integrate actual document-safe accumulated training; Evaluate every target and generate from the trained decoder; Reproduce interrupted training with the same artifacts and call policy; Report controlled context/resource comparison with explicit limits.

Topics: Training; validation; checkpoints; generation; resource budgeting.

Sessions: What claim will this training experiment test?; Does the reported loss include every reserved target?; Can another run reproduce the next update?; Which larger run can we afford to claim?.

## 08. Efficient and adapted LLMs

Project: Adapt and evaluate a language model. Deliverable: Controlled experiments that compare quality, cost and behavior.

Lab: `labs/s08-adaptation`. Design matrix: [section report](guidance/redesign/section-08.md).

### 40. Inference systems

Learner source: [labs/s08-adaptation/src/ch40.rs](labs/s08-adaptation/src/ch40.rs). 2 sessions; 9 learning steps.

Goal: Implement incremental inference and prove every-prefix parity and work counts; Implement controlled sampling and fair event order on unequal requests.

Topics: KV cache; sampling; batching; streaming.

Sessions: Cache the decoder; Sample and serve.

### 41. Quantization

Learner source: [labs/s08-adaptation/src/ch41.rs](labs/s08-adaptation/src/ch41.rs). 2 sessions; 8 learning steps.

Goal: Implement row scales, signed codes and packed storage; Measure output/model error, bytes and local timing independently.

Topics: Int8; int4; dequantization; quality and speed.

Sessions: Represent weights; Evaluate approximation.

### 42. Efficient attention

Learner source: [labs/s08-adaptation/src/ch42.rs](labs/s08-adaptation/src/ch42.rs). 2 sessions; 8 learning steps.

Goal: Implement persistent online state and bounded tiles; Implement causal GQA indexing and verify every output coordinate.

Topics: Online softmax; tiled attention; grouped-query attention; context costs.

Sessions: Stream a normalization; Map complete attention.

### 43. Adaptation

Learner source: [labs/s08-adaptation/src/ch43.rs](labs/s08-adaptation/src/ch43.rs). 3 sessions; 10 learning steps.

Goal: Implement a response mask and compare unseen prompts; Implement both factor gradients and prove frozen-base/merge parity; Implement temperature-consistent gradients and evaluate unseen tokens.

Topics: Supervised fine-tuning; LoRA; distillation.

Sessions: Select response supervision; Train and merge an adapter; Distill a distribution.

### 44. Retrieval and RAG

Learner source: [labs/s08-adaptation/src/ch44.rs](labs/s08-adaptation/src/ch44.rs). 3 sessions; 10 learning steps.

Goal: Implement BM25 and explain length/saturation counterexamples; Run both-tower training and implement the exact inner-product index; Assemble source text and compute coverage/selective risk including unsupported overlap.

Topics: Retrieval baselines; embeddings; indexing; grounded evaluation.

Sessions: Build a lexical baseline; Train and search embeddings; Evaluate evidence.

### 45. Reinforcement learning foundations

Learner source: [labs/s08-adaptation/src/ch45.rs](labs/s08-adaptation/src/ch45.rs). 3 sessions; 12 learning steps.

Goal: Implement epsilon-greedy interaction and compare five-seed regret; Implement full Q-learning and check exact path values; Trace REINFORCE and implement frozen-batch clipped PPO.

Topics: Bandits; tabular control; policy gradients.

Sessions: Collect useful bandit evidence; Propagate delayed reward; Optimize a policy.

### 46. Preference and reward training

Learner source: [labs/s08-adaptation/src/ch46.rs](labs/s08-adaptation/src/ch46.rs). 3 sessions; 12 learning steps.

Goal: Fit paired reward scores and test confounded feature transfer; Implement stable mean DPO gradients and frozen reference; Implement expected-reward gradients and expose an incomplete checker.

Topics: DPO; reference policy; preference pairs; verifiable reward.

Sessions: Learn a reward model; Fit relative response odds; Optimize a verifier.

## 09. Advanced architectures & engineering

Project: Study architectures and engineer a system. Deliverable: A sparse MoE language model you can train, evaluate and serve.

Lab: `labs/s09-advanced`. Design matrix: [section report](guidance/redesign/section-09.md).

### 47. Sparsity and pruning

Learner source: [labs/s09-advanced/src/ch47.rs](labs/s09-advanced/src/ch47.rs). 2 sessions; 6 learning steps.

Goal: Prune deterministically, construct CSR, verify two-input parity, and distinguish byte and timing savings..

Topics: Sparse representations; pruning; sparse kernels; measured savings.

Sessions: Choose what to remove; Make zeros change execution.

### 48. Mixture of experts

Learner source: [labs/s09-advanced/src/ch48.rs](labs/s09-advanced/src/ch48.rs). 2 sessions; 6 learning steps.

Goal: Implement the full jointly trained sparse expert update with exact capacity and balancing accounting..

Topics: Routing; expert capacity; load balancing; conditional compute.

Sessions: Train a route, not only an expert; Give every attempted token an account.

### 49. State-space models and linear attention

Learner source: [labs/s09-advanced/src/ch49.rs](labs/s09-advanced/src/ch49.rs). 2 sessions; 6 learning steps.

Goal: Implement affine prefix scan and normalized feature-kernel attention with direct-oracle and causal checks..

Topics: Recurrence; scans; selective state; linear attention.

Sessions: Compose a sequence without changing its order; Compress a causal attention history.

### 50. Distributed training

Learner source: [labs/s09-advanced/src/ch50.rs](labs/s09-advanced/src/ch50.rs). 3 sessions; 9 learning steps.

Goal: Implement example-weighted reduction and full pipeline updates; verify serial, finite-difference, and recovery equivalence..

Topics: Data parallelism; tensor parallelism; pipeline stages; recovery.

Sessions: Give every example the same weight; Split a model and send gradients home; Continue the same training computation.

### 51. Generative modeling beyond text

Learner source: [labs/s09-advanced/src/ch51.rs](labs/s09-advanced/src/ch51.rs). 3 sessions; 9 learning steps.

Goal: Implement beta-VAE and GAN objectives and a complete tested deterministic reverse diffusion chain..

Topics: VAE; GAN; diffusion; sampling; objectives.

Sessions: Sample a latent variable without losing its gradient; Train two changing opponents; Learn noise and remove it in order.

### 52. Multimodal models

Learner source: [labs/s09-advanced/src/ch52.rs](labs/s09-advanced/src/ch52.rs). 2 sessions; 6 learning steps.

Goal: Implement and train symmetric contrastive dual encoders and distinguish narrow retrieval checks from generalization..

Topics: Image and text encoders; contrastive alignment; retrieval.

Sessions: Build a comparison between unlike inputs; Ask whether retrieval learned the intended relationship.

### 53. Production ML

Learner source: [labs/s09-advanced/src/ch53.rs](labs/s09-advanced/src/ch53.rs). 2 sessions; 6 learning steps.

Goal: Verify train/serve artifact parity; implement explicit canary promotion and a bounded recent-input drift window..

Topics: Pipelines; serving; monitoring; drift; versioning; rollback.

Sessions: Make training and serving share one contract; Decide when a release should change.

### 54. Responsible evaluation

Learner source: [labs/s09-advanced/src/ch54.rs](labs/s09-advanced/src/ch54.rs). 3 sessions; 9 learning steps.

Goal: Implement subgroup audit metrics and explain tested attribution, perturbation, membership, parser, and causal limits..

Topics: Interpretability; robustness; privacy; security; fairness; causal limits.

Sessions: Explain one decision and challenge nearby inputs; Count who receives which errors; Separate protected boundaries from causal evidence.

### 55. Working with established tools

Learner source: [labs/s09-advanced/src/ch55.rs](labs/s09-advanced/src/ch55.rs). 2 sessions; 6 learning steps.

Goal: Build independent runtime conversion, coordinate diagnostics, gradient and update parity, and validated file interchange..

Topics: Framework concepts; runtime interchange; parity checks; optional integration.

Sessions: Preserve the same affine function across layouts; Preserve training semantics and the artifact boundary.

### 56. Advanced capstone

Learner source: [labs/s09-advanced/src/ch56.rs](labs/s09-advanced/src/ch56.rs). 4 sessions; 12 learning steps.

Goal: Implement and verify the full sparse decoder block, train against a dense baseline, evaluate, resume, and serve a bounded real request..

Topics: Sparse MoE language model; dense baseline; training; serving; evaluation.

Sessions: Read the whole contextual model; Send the task gradient through routing; Compare actual learning under declared budgets; Resume and serve the trained artifact.
