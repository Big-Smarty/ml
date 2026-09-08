# Chapters

This map is generated from course.json and chapter metadata. Lectures are HTML, not Markdown.

## 01. One neuron learns from data

Part: First principles. Prerequisites: Programming basics only.

Project: Fit a line with one neuron. Dataset: Course-authored line samples.

Lessons: The shared neuron interface; Examples and targets; The prediction rule; A numerical loss; Numerical slopes and updates; Run and check the program; Practice and transfer.

Introduced concepts: Prediction; weights and bias; squared loss; numerical slopes; training and inference.

Outcomes:

- Organize a scalar neuron and its gradient with Rust structs and methods
- Train one scalar neuron without ML libraries
- Calculate prediction and mean squared error by hand
- Explain borrowing with &self and propagate numerical errors with Result and ?
- Diagnose the update sign, learning rate, and sequential-update bug

Completion checks:

- Starting MSE equals 9
- Initial numerical gradient is weight -8 and bias -2 within 1e-8
- First step is weight 0.8, bias 0.2 within 1e-8
- An asymmetric fixture proves weight and bias update from the same old model
- 100 steps yield MSE below 1e-12
- Held-out prediction at 0.5 is within 1e-7 of 2
- Invalid data and learning rates are rejected


## 02. Why learning works

Part: First principles. Prerequisites: 01.

Project: Check an analytical gradient. Dataset: Course-authored line samples.

Lessons: From numerical probes to derivatives; Derive the gradient; Check before trusting; Build and run the Rust project; Practice and transfer.

Introduced concepts: Derivatives; gradients; analytical updates; finite-difference checks.

Outcomes:

- Keep the Chapter 1 Neuron interface while changing only its gradient strategy
- Derive exact MSE gradients for a weight and bias
- Perform a simultaneous analytical update
- Check derivatives with central differences
- Diagnose sign, averaging, and stale-parameter bugs

Completion checks:

- Analytical gradient at zero equals (-8, -2)
- Central differences match at two parameter settings within atol 1e-6 plus rtol 1e-4
- One step at learning rate 0.1 produces weight 0.8 and bias 0.2
- Training for 100 steps reaches loss below 1e-12
- Empty and nonfinite data are rejected


## 03. Multiple inputs

Part: First principles. Prerequisites: 02.

Project: Predict from several measurements. Dataset: Synthetic multifeature regression.

Lessons: Why one input is insufficient; Vectors and dot products; Multifeature gradients; Feature scaling; Run, practice, and transfer.

Introduced concepts: Vectors; dot products; linear regression; feature scaling.

Outcomes:

- Represent one example as a fixed-length feature vector
- Compute an affine prediction with a dot product
- Derive multifeature linear-regression gradients
- Fit and reuse training-set scaling statistics

Completion checks:

- Each standardized feature has training mean near zero
- Constant features are rejected
- Training MSE falls below 2.1 on the deterministic fixture
- A three-feature held-out prediction lies between 240 and 260
- Every weight and bias derivative matches central differences


## 04. Classification

Part: First principles. Prerequisites: 03.

Project: Train a binary classifier. Dataset: Course-authored two-class points.

Lessons: What carries forward from linear regression; From logits to probabilities and classes; Likelihood and binary cross-entropy; Stable loss and gradients in Rust; Practice and transfer.

Introduced concepts: Probability; sigmoid; likelihood; stable binary cross-entropy.

Outcomes:

- Distinguish an unbounded logit, a positive-class probability, and a discrete class prediction
- Explain why binary classification replaces regression MSE with mean binary cross-entropy
- Compute stable binary cross-entropy directly from logits
- Train a binary logistic classifier with a separate parameter Gradient

Completion checks:

- logit, probability, and predict return a score, probability, and class respectively
- BCE at logit zero equals ln 2 and remains finite at logits plus and minus 1000
- The separate analytical Gradient matches central differences on the same mean BCE objective
- Training mean BCE falls below 0.04
- predict classifies every fixture point correctly at its documented 0.5 threshold


## 05. Measuring learning

Part: First principles. Prerequisites: 04.

Project: Audit a classifier evaluation. Dataset: Synthetic imbalanced classification.

Lessons: Code and terminology carried forward; What training performance cannot tell you; Splits and leakage; Confusion counts and metrics; Thresholds and baselines; Run the audit; Practice and transfer.

Introduced concepts: Splits; leakage; metrics; thresholds; baselines.

Outcomes:

- Separate training, validation, and test decisions
- Detect entity overlap and preprocessing leakage
- Compute confusion counts and derived metrics from probabilities and targets
- Choose a threshold on validation and compare a baseline

Completion checks:

- At threshold 0.5 the test confusion counts are TP 1, FP 1, TN 7, FN 1
- Accuracy equals 0.8 while precision and recall equal 0.5
- Validation selects threshold 0.6 from the documented candidates
- Duplicate entity identifiers across splits are detected


## 06. Reliable optimization

Part: First principles. Prerequisites: 05.

Project: Compare learning curves. Dataset: Synthetic noisy regression.

Lessons: When a correct gradient still fails; Learning rates and curves; Minibatches; Regularization and generalization; Run the comparison; Practice and transfer.

Introduced concepts: Learning rates; minibatches; regularization; generalization.

Outcomes:

- Read training and validation learning curves together
- Explain learning-rate failure modes
- Apply deterministic minibatch updates
- Distinguish optimizer steps from epochs
- Derive and implement L2 regularization
- Distinguish optimization from generalization

Completion checks:

- A reliable learning_rate lowers validation loss below 0.2
- The learned weight is within 0.15 of two
- L2 produces a smaller absolute weight than the unregularized run
- A batch of four examples gives three optimizer steps per epoch
- Invalid batch sizes and hyperparameters are rejected


## 07. From a neuron to XOR

Part: Build a neural network. Prerequisites: 06.

Project: Solve XOR with a small MLP. Dataset: XOR truth table.

Lessons: The XOR limit; Manual chain rule; Stable objective and checking; The Rust network; Backward-pass practice; Review and transfer.

Introduced concepts: Nonlinearities; hidden layers; manual backpropagation.

Outcomes:

- Explain why a linear neuron cannot represent XOR
- Distinguish a forward-pass cache, a probability, and a class prediction
- Derive and implement an averaged Gradient for both network layers
- Apply the Gradient as one simultaneous optimizer step
- Check all nine parameter gradients with central differences

Completion checks:

- 10,000 full-batch updates on explicit XOR data produce mean binary cross-entropy below 0.02
- predict classifies all four XOR examples correctly at threshold 0.5
- The averaged Gradient matches central differences for all nine parameters
- step applies that Gradient to every matching parameter
- Empty or nonfinite data, out-of-range targets, and nonpositive learning rates are rejected


## 08. Automatic differentiation

Part: Build a neural network. Prerequisites: 07.

Project: Build scalar autodiff. Dataset: Synthetic computation graphs.

Lessons: Computation graphs; Reverse accumulation; Topological ordering; The Rust engine; Graph practice; Review and transfer.

Introduced concepts: Computation graphs; reverse mode; accumulated gradients.

Outcomes:

- Represent scalar calculations as a directed acyclic graph
- Distinguish node adjoints from parameter gradients
- Accumulate adjoint contributions through shared nodes
- Implement and numerically check scalar autodiff

Completion checks:

- x*x+x at x=3 produces value 12 and adjoint 7 for x
- A tanh chain matches central differences
- Repeated backward calls clear reachable adjoints
- Shared nodes are traversed once but receive every operand contribution through add_gradient
- Nonfinite intermediates and overflowing adjoints return errors even with a finite root


## 09. Tensors and batches

Part: Build a neural network. Prerequisites: 08.

Project: Build a batched dense layer. Dataset: Hand-computed matrices.

Lessons: Shapes and contiguous layout; Dense-layer gradients; Reference kernels; Run the matrix checkpoint; Shape practice; Review and transfer.

Introduced concepts: Shapes; contiguous storage; matrix operations; batched gradients.

Outcomes:

- Interpret contiguous row-major data using shapes
- Compute a batched dense forward pass with weights stored [out_features,in_features]
- Derive input, weight, and bias gradients from supplied output gradients
- Reject incompatible shapes and check a weight gradient numerically

Completion checks:

- Forward fixture equals [-1.5,3.5,-1.5,12.5]
- Input, weight, and bias gradients match hand calculations
- A weight gradient matches central differences
- Invalid matrix and backward shapes return errors


## 10. Recognizing digits with a linear model

Part: Build a neural network. Prerequisites: 09.

Project: Train a digit classifier. Dataset: MNIST plus generated IDX fixture.

Lessons: IDX files; Softmax cross-entropy; Linear classifier training; Fixture and MNIST modes; Stable-classification practice; Review and transfer.

Introduced concepts: IDX format; MNIST; softmax; multiclass evaluation.

Outcomes:

- Validate and parse big-endian IDX image and label files
- Compute stable softmax probabilities and cross-entropy from logits
- Map a dense layer's weights and bias to a ten-class linear model
- Train with mean minibatch gradients
- Separate validation choices from final test evaluation

Completion checks:

- Generated bytes parse through the real IDX loader
- Malformed magic and payload sizes are rejected
- Fixture mean cross-entropy falls by at least 80 percent
- Fixture held-out accuracy reaches 100 percent
- Explicit MNIST mode requires readable paths
- cross_entropy_from_logits retains ln 10 for ten equal logits at a common offset of 1e16


## 11. An MNIST neural network

Part: Build a neural network. Prerequisites: 10.

Project: Train and restore a digit MLP. Dataset: MNIST plus tiny generated fixture.

Lessons: A nonlinear digit model; Initialization; Momentum and Adam; Complete checkpoints; Train and practice; Review and transfer.

Introduced concepts: Initialization; momentum; Adam; debugging; checkpoints.

Outcomes:

- Train a one-hidden-layer ReLU classifier with mean cross-entropy gradients
- Map packed parameters to W1, b1, W2, and b2 offsets
- Implement momentum and Adam optimizer steps
- Save and validate complete resumable optimizer checkpoints

Completion checks:

- Both momentum and Adam reduce fixture loss
- Fixture accuracy exceeds two thirds
- Packed parameter offsets match W1, b1, W2, and b2 shapes
- A hidden parameter gradient matches central differences away from a ReLU kink
- Save/restore preserves the exact next Adam update
- Corrupt or unsafe checkpoint dimensions and values are rejected
- Zero-epoch evaluation requires an existing checkpoint and preserves its bytes
- Nonfinite loss, gradients, and optimizer moments are rejected


## 12. Statistics and uncertainty

Part: Beyond neural networks. Prerequisites: 11.

Project: Measure uncertainty in evaluation. Dataset: Synthetic predictions.

Lessons: A sample gives an estimate; Resample the observed cases; Accuracy ignores the probability scale; Score probabilities and map the maths to Rust; Run and perturb the experiment; Practice, debug, and transfer.

Introduced concepts: Sampling; confidence intervals; bootstrap; calibration.

Outcomes:

- Distinguish a sample statistic from a population quantity
- Compute and interpret a bootstrap percentile interval
- Build calibration bins and a Brier score
- Choose a resampling unit that preserves groups or temporal dependence

Completion checks:

- Reference project reports accuracy, a deterministic bootstrap interval, Brier score, and nonempty calibration bins
- Lesson maps a frozen Chapter 11 class probability and observed class to Prediction without reusing training loss
- Bootstrap draws complete probability-outcome rows with replacement
- Tests reject empty or malformed predictions
- Bootstrap endpoints distinguish all-correct, all-wrong, and mixed observations; calibration bins match counts and means
- Lesson distinguishes interval procedure coverage from probability about a fixed parameter


## 13. Real tabular data

Part: Beyond neural networks. Prerequisites: 12.

Project: Build a leakage-safe preprocessor. Dataset: Tiny local tabular fixture; UCI Auto MPG extension.

Lessons: Code and terminology carried forward; Split before learning anything; Missing is information about measurement; Categories need a stable vocabulary; Groups and time define believable tests; Build the Rust preprocessor; Practice and review.

Introduced concepts: Missing values; categories; grouped splits; temporal splits.

Outcomes:

- Distinguish raw rows, encoded feature vectors, and discrete class labels
- Fit median imputation and a category vocabulary on training rows only
- Construct group-isolated and past-to-future splits
- Transform held-out rows with fixed fitted state and a stable feature shape

Completion checks:

- Reference Preprocessor::fit estimates median and categories from training RawRow values after the split
- Preprocessor::transform returns 2 + categories.len() encoded features and leaves the class label separate
- Held-out value 999 and category secret cannot alter fitted state
- Group and temporal split behaviors have runnable tests
- Even-count median remains finite at extreme finite inputs; missing indicators remain distinct


## 14. Nearest neighbors and naive Bayes

Part: Beyond neural networks. Prerequisites: 13.

Project: Compare two classifiers. Dataset: Local three-class fixture; UCI Iris extension.

Lessons: Distance turns similarity into arithmetic; Scale before measuring neighbors; k-nearest neighbors votes from memory; Gaussian naive Bayes models each class; Run a fair small comparison; Practice, debugging, and transfer.

Introduced concepts: Distance; scaling; nearest neighbors; Gaussian naive Bayes.

Outcomes:

- Compute squared Euclidean distance between two-feature vectors
- Fit a scaler on training features and transform later features without refitting
- Fit k-nearest neighbors and predict a discrete label by deterministic voting
- Fit Gaussian naive Bayes and distinguish its raw log scores from its predicted label

Completion checks:

- The squared-distance primitive receives the same two-feature arrays stored by Point and passed to prediction
- Knn::fit rejects invalid k or training labels and both predictors reject nonfinite inputs
- Both fitted estimators predict label 1 for the middle fixture cluster
- Gaussian likelihoods are accumulated as raw log scores
- Scaler::fit runs once on training rows and Scaler::transform reuses its state
- Overflowed fitted statistics, distances, or Gaussian scores return errors


## 15. Trees, forests, and boosting

Part: Beyond neural networks. Prerequisites: 14.

Project: Build a tabular ensemble. Dataset: Local tabular fixture.

Lessons: A tree partitions the feature space; Bag unstable trees into a random forest; Boosting corrects what remains; Trace the three Rust implementations; Run the ensemble and inspect capacity; Practice, debug, and transfer.

Introduced concepts: Decision trees; bagging; random forests; gradient boosting.

Outcomes:

- Choose a decision-tree split using weighted Gini impurity
- Explain how depth controls memorization
- Build a random forest with row and per-node feature randomness
- Fit squared-error gradient boosting to residuals

Completion checks:

- Baseline tree minimizes row-count-weighted child Gini impurity
- Forest bootstraps rows, samples a candidate feature at every recursive node, and predicts by majority vote
- Boosting stumps minimize residual SSE and the ensemble converges on the tiny regression fixture
- Starter verifies weighted impurity; Rustlings verifies the Gini primitive


## 16. Margins and kernels

Part: Beyond neural networks. Prerequisites: 15.

Project: Classify linear and curved boundaries. Dataset: Synthetic points.

Lessons: From a correct boundary to a confident boundary; Hinge loss spends effort near the margin; Turn the objective into Rust updates; Curve the boundary with similarity; Run, inspect, and practice; Review and transfer.

Introduced concepts: Hinge loss; linear SVM; kernel similarity.

Outcomes:

- Convert binary class labels to -1 and +1, then compute signed margins and hinge loss
- Train and evaluate a two-feature linear soft-margin SVM
- Explain how an RBF kernel turns distances into nonlinear influence
- Diagnose label-encoding and kernel-inference bugs

Completion checks:

- The zero model has regularized objective 1 on the linear fixture because its mean hinge loss is 1 and its L2 penalty is 0
- The trained linear SVM classifies all eight linearly separated fixture points
- The linear model reaches at most 75% on XOR while the RBF classifier reaches 100%
- Empty data and invalid gamma are rejected


## 17. PCA and numerical linear algebra

Part: Beyond neural networks. Prerequisites: 16.

Project: Compress correlated measurements. Dataset: Synthetic correlated data; UCI Wine extension.

Lessons: Turn correlated columns into a matrix; Variance chooses an eigenvector; Find the direction by repeated multiplication; Conditioning tells you when answers wobble; Run the compressor and practice; Review and transfer.

Introduced concepts: Covariance; eigenvectors; power iteration; projections; conditioning.

Outcomes:

- Center a two-feature dataset and compute sample covariance
- Find a leading eigenvector with checked power iteration
- Fit PCA state, transform rows into projection scores, and reconstruct them
- Interpret explained variance, eigengaps, and covariance conditioning

Completion checks:

- Power iteration recovers eigenvalue 3 for the matrix [[2,1],[1,2]]
- The returned component has unit norm and a small eigenpair residual
- Translating all rows preserves eigenvalues, directions, and reconstruction error
- Rank-one data reconstructs with negligible one-component error
- Mean squared reconstruction norm divides summed squared row norms by the number of rows
- Too few or nonfinite rows are rejected


## 18. Clustering and density

Part: Beyond neural networks. Prerequisites: 17.

Project: Discover synthetic clusters. Dataset: Synthetic Gaussian clusters.

Lessons: Alternate between centers and assignments; Replace a hard group with a probability; Fit the mixture with expectation-maximization; Turn density into an anomaly score; Run the models and practice.

Introduced concepts: K-means; Gaussian mixtures; EM; anomaly scores.

Outcomes:

- Fit deterministic k-means to unlabeled feature points and compute summed inertia
- Explain Gaussian-mixture responsibilities as normalized component probabilities
- Implement stable expectation-maximization with log-sum-exp
- Use negative log density as an anomaly score without confusing rarity with harm

Completion checks:

- K-means finds one negative and one positive fixture center with summed inertia below 3
- Final assignments are recomputed from the returned centers
- Mixture responsibilities sum to one and component weights remain normalized
- The distant point receives a larger anomaly score than a cluster-center point
- Later well-conditioned EM fits do not reduce summed fixture log-likelihood beyond tolerance


## 19. Tabular ML capstone

Part: Beyond neural networks. Prerequisites: 18.

Project: Run a reproducible model comparison. Dataset: Bundled tabular fixture; UCI extension.

Lessons: Write the evaluation contract before tuning; Cross-validation reuses development data safely; Fit every learned transformation inside the fold; Search a fixed budget and compare fairly; Finish with error analysis, not one score; Run the capstone and practice.

Introduced concepts: Cross-validation; hyperparameter search; fair baselines; error analysis.

Outcomes:

- Construct deterministic stratified folds from stable row IDs
- Keep scaling and all learned preprocessing inside each training fold
- Run a fixed-budget hyperparameter search with deterministic tie-breaking
- Compare a learned logistic model with a fair majority baseline
- Evaluate once on untouched test rows and inspect errors by ID

Completion checks:

- A fixed seed yields identical fold membership across runs
- Every development row receives one valid fold and each fold contains three rows per class
- Scaling is fitted separately on each fold's training rows
- Cross-validation accuracy pools correct and total held-out rows instead of averaging fold percentages
- Candidate ties resolve to the earliest frozen candidate
- The selected logistic model beats the majority baseline on the untouched fixture test


## 20. Convolutional networks

Part: Broader deep learning. Prerequisites: 19.

Project: Build a small convolutional digit model. Dataset: Tiny images; MNIST extension.

Lessons: Why images need shared local weights; Convolution arithmetic; Pooling and gradients; Build and run the classifier; Practice and debug; Review and transfer.

Introduced concepts: Convolution; padding and stride; pooling; gradients.

Outcomes:

- Compute convolution output shapes and values with padding and stride
- Trace max-pooling indices and route their gradients
- Backpropagate stable multiclass loss through a dense head and convolution
- Train the tiny digit fixture and optionally load prepared MNIST IDX files

Completion checks:

- Central-difference and analytic convolution gradients agree away from ReLU and max-pool ties
- Training reduces validation cross-entropy, reaches at least 70% fixture accuracy, and increments optimizer step_count once per image update
- Malformed IDX headers are rejected
- Default execution performs no network access


## 21. Deeper vision models

Part: Broader deep learning. Prerequisites: 20.

Project: Train a residual image block. Dataset: Course-authored images.

Lessons: Why deeper stacks become hard to optimize; Residual arithmetic; Normalization; Augmentation and training; Practice and debug; Review and transfer.

Introduced concepts: Residual connections; normalization; augmentation.

Outcomes:

- Explain an identity shortcut and trace named gradients through two blocks
- Normalize one row-major image over an explicit 36-pixel axis
- Generate label-preserving translations without crossing split boundaries
- Train two small pre-normalized residual convolutional blocks with per-example cross-entropy

Completion checks:

- First residual-block kernel analytic and numerical gradients agree through both blocks
- Normalized fixture images have mean approximately zero over all 36 pixels
- Stable cross-entropy from two logits retains log(2) under a large common offset
- Training reduces mean held-out loss and classifies all four canonical images
- Augmentation is deterministic and limited to training rows


## 22. Learning representations

Part: Broader deep learning. Prerequisites: 21.

Project: Learn a useful small representation. Dataset: Course-authored paired vectors.

Lessons: Targets can come from the input; Autoencoder backpropagation; Contrastive pairs and negatives; Run both experiments; Practice and debug; Review and transfer.

Introduced concepts: Autoencoders; bottlenecks; contrastive learning.

Outcomes:

- Train an autoencoder through a two-value bottleneck using coordinate MSE
- Distinguish reconstruction quality from representation usefulness
- Compute stable InfoNCE directly from similarity logits
- Train a contrastive encoder and evaluate paired retrieval

Completion checks:

- Autoencoder training reduces coordinate reconstruction MSE by the declared factor
- The bottleneck has exactly two coordinates
- InfoNCE uses stable log-sum-exp and averages over anchors
- Contrastive training reduces its distinct objective and reports paired retrieval separately


## 23. Sequences and forecasting

Part: Broader deep learning. Prerequisites: 22.

Project: Forecast a held-out signal. Dataset: Course-authored time series.

Lessons: Order changes the prediction problem; RNNs and BPTT; LSTM memory; Temporal evaluation; Practice and debug; Review and transfer.

Introduced concepts: RNNs; BPTT; LSTMs; temporal evaluation.

Outcomes:

- Unroll a recurrent state over time
- Derive and check BPTT gradients through shared recurrent weights
- Trace LSTM input, forget, output, and candidate paths
- Distinguish teacher-forced one-step evaluation from free-running inference on a later contiguous segment

Completion checks:

- RNN and LSTM loss_and_gradient parameter derivatives match central differences
- Both RNN and LSTM reduce later one-step validation MSE
- Training targets all precede validation targets
- Stable sigmoid avoids overflow for large negative inputs


## 24. Recommendation systems

Part: Broader deep learning. Prerequisites: 23.

Project: Recommend held-out items. Dataset: Course-authored interaction matrix.

Lessons: From sparse interactions to vectors; Pairwise ranking; Ranking metrics; Run the recommender; Practice and debug; Review and transfer.

Introduced concepts: Embeddings; matrix factorization; ranking metrics.

Outcomes:

- Compute raw preference scores from user and item embeddings
- Train matrix factors with a stable regularized pairwise ranking loss
- Keep held-out positives out of optimization
- Calculate Recall@k, Precision@k, and nDCG@k from ranked candidates

Completion checks:

- Pairwise analytic and numerical gradients agree
- Stable softplus remains finite at extreme margins
- Training lowers the mean regularized loss over eight training triples
- Ranking metrics average four query-level values over a declared candidate set that excludes training positives


## 25. Measure before optimizing

Part: Make the CPU faster. Prerequisites: 24.

Project: Write an honest CPU benchmark. Dataset: Deterministic matrices.

Lessons: Floating-point evidence; Controlled measurement; Work and throughput; Memory costs and arithmetic intensity; Benchmark practice; Review and transfer.

Introduced concepts: Floating point; profiling; memory costs; arithmetic intensity.

Outcomes:

- Explain why an f32 performance kernel needs tolerance-based comparison with an f64 numerical oracle
- Construct a release-mode benchmark with warmups and repeated samples
- Report latency with units, variability, work, and measured throughput
- Calculate and qualify an arithmetic-intensity estimate
- Use profiling evidence to choose a useful optimization target

Completion checks:

- Dense f32 outputs for inputs=[batch,in_features] and weights=[out_features,in_features] agree with an f64 accumulator on a deterministic odd-shaped fixture
- Bad dense shapes return an error
- No-argument benchmark uses three warmups and eleven measured repetitions
- Benchmark reports unit-bearing median and range values, checksum, GFLOP/s, cold-array traffic estimate, and estimated FLOP/byte
- Starter run succeeds while its guided median_ns test reaches the TODO


## 26. Fast matrix multiplication

Part: Make the CPU faster. Prerequisites: 25.

Project: Optimize a matrix kernel. Dataset: Deterministic rectangular matrices.

Lessons: Matrix traversal; Loop ordering; Cache blocking; Allocation reuse; Locality practice; Review and transfer.

Introduced concepts: Loop ordering; cache locality; blocking; allocation reuse.

Outcomes:

- Map GEMM notation and transposed dense weights to row-major offsets
- Explain how loop order changes spatial locality
- Implement a cache-blocked scalar matrix multiplication with edge tiles
- Reuse and correctly clear caller-owned output storage
- Benchmark multiple correct kernels without claiming a predetermined winner

Completion checks:

- Scalar row-col-inner, scalar row-inner-col, and blocked row-inner-col kernels agree with an f64 reference
- The correctness fixture uses m=3, k=5, n=7 and blocks 1, 4, 16, and usize::MAX
- Zero dimensions, mismatched storage, and a zero block return errors
- Each timed kernel reuses one caller-owned output allocation and reports median, range, and GFLOP/s
- Starter run produces the prior unchecked scalar result while its new loop-order test reaches the TODO


## 27. Multithreaded training and inference

Part: Make the CPU faster. Prerequisites: 26.

Project: Parallelize dense computation. Dataset: Deterministic batches.

Lessons: Dense-to-linear continuity; Batch partitioning; Private gradient sums; Deterministic reduction; Parallel project; Review and transfer.

Introduced concepts: Partitioning; scoped threads; gradient reduction; reproducibility.

Outcomes:

- Map a one-output linear model to the course's row-major dense convention
- Distinguish one-row predict from scalar and threaded batch-prediction wrappers
- Return loss and parameter derivatives together from scalar and parallel fused APIs
- Reduce private worker sums in fixed order and divide once by the complete batch size
- Run actual parallel training with one simultaneous update

Completion checks:

- predict_batch_parallel matches predict_batch for every row
- loss_and_gradient_parallel agrees with scalar loss_and_gradient within tolerance
- Workers return sums and 31 rows split across three unequal shards are divided by 31 exactly once
- Three-shard training on 31 rows lowers MSE by at least 1000 times
- Empty prediction succeeds while empty training, zero threads, and invalid shapes fail


## 28. SIMD

Part: Make the CPU faster. Prerequisites: 27.

Project: Dispatch a portable vector kernel. Dataset: Odd-length deterministic vectors.

Lessons: SIMD lanes and tails; Compiler auto-vectorization; Safe runtime dispatch; Stable AVX-512 extension; SIMD practice; Review and transfer.

Introduced concepts: Auto-vectorization; AVX2 and FMA; feature detection; tails; AVX-512 extension.

Outcomes:

- Explain SIMD lanes, vector prefixes, horizontal reduction, and scalar tails
- Distinguish compiler auto-vectorization from an inspected guarantee
- Implement runtime-guarded AVX2 and FMA dispatch with a scalar fallback
- Run an actual stable AVX-512F extension only on supported x86 CPUs
- Document unsafe invariants and test empty, short, and odd vector lengths

Completion checks:

- dot_dispatch and dot_avx512_checked reject unequal lengths before any intrinsic call
- dot_dispatch handles every length 0 through 39 and odd length 1003
- dot_scalar and every runtime-supported AVX2/FMA and AVX-512F backend are compared directly with dot_f64_reference
- The auto-vectorization candidate handles an odd length
- Explicit AVX-512 mode returns a clear unsupported error when detection fails
- Every unsafe call and load has documented feature and bounds invariants


## 29. GPU fundamentals

Part: Write GPU kernels. Prerequisites: 28.

Project: Run a checked GPU vector addition. Dataset: Deterministic vectors.

Lessons: Why a GPU needs a different execution model; Discover the hardware you actually run; Move a vector through buffers and bindings; Dispatch WGSL and read the answer back; Run, debug, and extend the project; Review and transfer.

Introduced concepts: Adapter discovery; wgpu; WGSL; buffers; dispatch; readback.

Outcomes:

- Select and report a hardware Vulkan adapter without accepting a software fallback
- Explain how Rust buffers and WGSL bindings represent the same contiguous [N] f32 vectors
- Dispatch enough workgroups for an arbitrary element count without dropping the tail
- Synchronize a real GPU result into a mapped readback buffer
- Compare GPU output with the same scalar primitive using an explicit absolute-plus-relative tolerance

Completion checks:

- The executable reports every discovered adapter and the selected hardware Vulkan adapter
- A five-element vector is dispatched and read back from a real compute shader
- Ordinary tests cover the scalar oracle, f32 tolerance, and exact and partial dispatch counts
- The ignored hardware test covers 67 elements, including a partial workgroup
- Empty, mismatched, and nonfinite inputs return errors before adapter discovery


## 30. Reductions and tiled matrix multiplication

Part: Write GPU kernels. Prerequisites: 29.

Project: Build GPU reduction and GEMM. Dataset: Deterministic rectangular matrices.

Lessons: Why neighboring invocations must cooperate; Reduce values inside one workgroup; Tile matrix multiplication; Handle edge dimensions and device limits; Run, debug, and extend the GPU library; Review and transfer.

Introduced concepts: Workgroups; shared memory; barriers; edge dimensions.

Outcomes:

- Trace a tree reduction through workgroup memory and barriers
- Derive a tiled row-major matrix multiplication for rectangular matrices
- Guard partial tiles without allowing any lane to skip a barrier
- Validate dimensions, checked products, device limits, and input lengths
- Use Gpu::matmul and Gpu::reduce_sum with their named scalar oracles

Completion checks:

- matmul_scalar handles a nonsquare 3×17 by 17×5 product
- The ignored hardware test matches that odd-tile product within absolute-plus-relative tolerance
- reduce_sum_scalar validates finite nonempty input, and a 777-value staged reduction matches it
- Zero, overflowing, malformed, nonfinite, or device-exceeding shapes return errors
- Gpu stores a persistent device, queue, matmul pipeline, and reduction pipeline
- The lesson distinguishes per-workgroup dimensions from the padded global dispatch grid


## 31. Train on the GPU

Part: Write GPU kernels. Prerequisites: 30.

Project: Train a GPU neural network. Dataset: Synthetic nonlinear points.

Lessons: Why repeated host transfers stop being training; Work a nonlinear forward and backward pass; Order three device-resident kernels; Check gradients and training behavior; Run, debug, and extend the project; Review and transfer.

Introduced concepts: Forward and backward kernels; updates; residency.

Outcomes:

- Map a nonlinear network's forward, backward, and update equations to separate GPU kernels
- Keep inputs, activations, gradients, losses, and parameters in device buffers across training steps
- Compute stable binary cross-entropy directly from logits
- Check selected analytical gradients with central differences
- Verify GPU training against the same scalar model operations and a falling-loss criterion

Completion checks:

- CPU analytical gradients match central differences for input, hidden, output, and bias parameters
- Stable binary cross-entropy remains finite for logits of magnitude 1000
- CPU training reduces nonlinear XOR loss by at least seventy percent
- The ignored hardware test checks GPU loss decrease and all parameters against the CPU oracle
- No training tensor is mapped between forward, backward, and update steps


## 32. GPU performance

Part: Write GPU kernels. Prerequisites: 31.

Project: Measure and improve GPU kernels. Dataset: Deterministic GPU workloads.

Lessons: Performance begins with a well-defined clock; Understand asynchronous submission; Fuse adjacent elementwise kernels; Choose mixed precision by capability and range; Connect wgpu to Vulkan extensions; Measure, debug, and transfer.

Introduced concepts: Asynchronous execution; timing; fusion; mixed precision; Vulkan extension.

Outcomes:

- Separate CPU wall-clock latency from device timestamp duration
- Explain asynchronous queue submission and explicit completion points
- Fuse an affine transform and ReLU while preserving a scalar oracle
- Query timestamp and shader-f16 support before requesting optional features
- Relate wgpu's f16 feature to concrete Vulkan capability extensions

Completion checks:

- Separate and fused f32 outputs from affine_relu_with_gpu match affine_relu_scalar for an odd invocation count
- The default run uses warmups and reports repeated CPU wall-clock median and range
- Device pass timestamps are reported only when TIMESTAMP_QUERY is supported
- The f16 shader is created only when SHADER_F16 is supported
- Inputs outside the safe teaching range use the f32 fallback


## 33. Language modeling

Part: Train your language model. Prerequisites: 32.

Project: Train a tiny text predictor. Dataset: Course-authored text.

Lessons: Code and terminology carried forward; Turn text into supervised examples; Counts are the first language model; Replace a token ID with learned coordinates; Train the Rust predictor; Practice and debug; Review and transfer.

Introduced concepts: Byte models; n-grams; embeddings; next-token loss.

Outcomes:

- Create byte and Unicode-character n-gram counts with explicit units
- Derive and train an embedding next-byte classifier with stable cross-entropy
- Distinguish byte class IDs, logits, probabilities, targets, and predictions
- Trace gradients into both the selected embedding row and output matrix
- Distinguish training fit from held-out language quality

Completion checks:

- Reference tests confirm byte and character units differ
- Model::loss, Model::step, and Model::predict retain their documented roles with explicit data and learning_rate arguments
- Embedding and output parameters both change while mean next-byte cross-entropy falls
- Tagged Rust excerpts match the reference source
- Default project runs offline without arguments
- Starter runs and its guided test fails at a literal TODO


## 34. Tokenization

Part: Train your language model. Prerequisites: 33.

Project: Build and test a BPE tokenizer. Dataset: Course-authored multilingual text.

Lessons: Code and terminology carried forward; Text arrives as UTF-8 bytes; Learn merges from adjacent pairs; Apply the learned ranks; Save enough information; Practice, debug, and extend; Review and transfer.

Introduced concepts: UTF-8; byte tokens; BPE training; serialization.

Outcomes:

- Explain UTF-8 byte sequences without conflating bytes, token IDs, or Unicode scalar values
- Fit deterministic non-overlapping BPE merge rules from adjacent counts
- Encode bytes to token IDs and decode token IDs to the exact original bytes
- Relate maximum merge count, learned rule count, and vocabulary size
- Serialize merge order and reject malformed tokenizer files

Completion checks:

- The default fixture requests at most 24 merge rules, learns 12, and produces a 268-entry vocabulary
- Multilingual byte round trips pass for ASCII, accented text, CJK, and emoji
- Saved and reloaded merge tables are equal
- Malformed headers, forward references, oversized expansion, and unknown token IDs return errors
- Starter prints Unicode scalar-value and UTF-8 byte counts before its guided test


## 35. Attention

Part: Train your language model. Prerequisites: 34.

Project: Implement causal self-attention. Dataset: Hand-computed sequence tensors.

Lessons: Code and terminology carried forward; Ask, match, and retrieve; Mask information from the future; Reverse the weighted mixture; Check the whole chain numerically; Practice and investigate; Review and transfer.

Introduced concepts: Queries; keys; values; causal masking; gradients.

Outcomes:

- Compute scaled causal query-key scores and probability-weighted values
- Explain queries, keys, and values by role and shape
- Derive manual gradients for Q, K, V, and row softmax
- Distinguish an attention derivative probe from next-token cross-entropy
- Use finite differences and causal invariance as complementary tests

Completion checks:

- causal_attention receives direct row-major query, key, and value arrays and returns output plus source-position probabilities with the documented shapes
- Every Q, K, and V element passes f64 central differences against the summed attention_probe_objective
- backward propagates supplied output gradients without an extra averaging denominator
- Future key and value changes cannot affect earlier outputs, and masked probabilities are exactly zero
- Default run traces probabilities, outputs, and nonzero gradient norms


## 36. A decoder Transformer

Part: Train your language model. Prerequisites: 35.

Project: Build a trainable decoder. Dataset: Course-authored text.

Lessons: Code and terminology carried forward; Assemble a pre-normalized decoder block; Normalize features within each token; Store all trainable state in one checked buffer; Derive the parameter count and the prediction contract; Backpropagate tensors in reverse order; Verify CPU and optional GPU execution; Practice and debug; Review and transfer.

Introduced concepts: Residual paths; normalization; positions; feed-forward blocks.

Outcomes:

- Assemble a pre-normalized causal decoder with residual and feed-forward paths
- Derive tensor-level backward operations through every component
- Inventory and train every parameter family
- Validate causality, normalization, gradients, loss reduction, and optional GPU GEMMs

Completion checks:

- Exact large configuration count equals 14,442,496
- Causal prefix logits ignore future token changes
- Attention, normalization, feed-forward, embedding, and output gradients pass representative f32 finite differences
- Representative identifiable parameter families change while loss falls
- Optional GPU feature routes forward and backward GEMMs through chapter 30


## 37. Language-model data

Part: Train your language model. Prerequisites: 36.

Project: Prepare an auditable text corpus. Dataset: Course-authored documents; licensed corpus extension.

Lessons: Know what each document is; Clean with explicit rules; Remove repeated documents; Split complete documents and bound contamination claims; Produce and inspect the audit; Practice, debug, and transfer.

Introduced concepts: Provenance; cleaning; deduplication; document splits; contamination.

Outcomes:

- Parse whole documents while preserving source and license provenance
- Normalize document text and compute a stable derived identity
- Remove exact and detected near duplicates with an auditable rule
- Assign each retained whole document to one stable split without constructing token windows

Completion checks:

- Parsing, cleaning, deduplication, and split assignment produce separate document artifacts
- Malformed metadata, empty cleaned text, and invalid thresholds return errors
- The near-duplicate fixture removes the intended copy and records its retained-document link
- The manifest retains source and license fields and gives each retained document exactly one split


## 38. Training at a larger scale

Part: Train your language model. Prerequisites: 37.

Project: Resume a reproducible LM run. Dataset: Course-authored text.

Lessons: Adapt each parameter's update scale; Change learning rate with progress; Accumulate microbatches; Account for memory; Resume the exact next step; Practice and diagnose; Review and transfer.

Introduced concepts: AdamW; schedules; accumulation; memory; resumable training.

Outcomes:

- Derive Adam moment and bias-correction updates while separating the data objective from decoupled weight decay
- Combine equal microbatch gradients before global clipping
- Compute cosine decay with warmup per optimizer step
- Save and validate every state item required for exact continuation
- Estimate persistent memory and define a throughput measurement

Completion checks:

- Save-load branch produces bit-identical next loss and parameters
- Checkpoint restores cursor, RNG, schedule step, model, and moments
- Accumulated return value is the mean data loss and excludes AdamW weight decay
- Nonfinite gradients and checkpoint values are rejected
- Default run reports calculated persistent memory and measured loss


## 39. Scratch-trained language-model capstone

Part: Train your language model. Prerequisites: 38.

Project: Train your own dense language model. Dataset: Tiny local corpus; user-supplied licensed corpus.

Lessons: Carry forward the decoder and trainer contracts; Define the experiment; Train the complete tiny decoder; Compute a target-weighted held-out metric; Generate a continuation; Save, load, and continue; Budget the larger option; Practice, debug, and extend; Review and transfer.

Introduced concepts: Training; validation; checkpoints; generation; resource budgeting.

Outcomes:

- Train and teacher-forced evaluate a dense causal decoder on byte IDs
- Use distinct licensed training and validation documents without silent fallback
- Save, load, resume additional optimizer steps, and generate from the actual trained model
- Report exact parameter and persistent-memory budgets
- Measure tokens per second without fabricating scale claims

Completion checks:

- Tiny test trains the complete decoder, lowers fixture loss, resumes the exact next public training update, rejects changed training IDs, and generates
- Default run performs 40 optimizer steps over 960 target tokens, evaluates a separate held-out document, and reports measured throughput
- Unequal held-out blocks are weighted by their target counts
- Default and large configurations have byte vocabulary 256 and exactly 10,896 and 14,442,496 parameters
- Large-info prints exactly 14,442,496 parameters without allocation
- Large mode instantiates the real Trainer and requires an explicit step budget
- After training binds a nonzero fingerprint, a resumed update rejects changed training IDs


## 40. Inference systems

Part: Efficient and adapted LLMs. Prerequisites: 39.

Project: Build a cached text generator. Dataset: Chapter 39 model and fixture.

Lessons: Code and terminology carried forward; Reuse the past without changing the model; Match full-prefix logits, layer by layer; Turn logits into a controlled choice; Schedule requests and stream completed tokens; Run and investigate; Before you move on.

Introduced concepts: KV cache; sampling; batching; streaming.

Outcomes:

- Implement per-layer decoder KV caching
- Verify cached logits against every full prefix
- Apply temperature and top-k sampling
- Schedule and stream multiple requests

Completion checks:

- Cached logits differ from full-prefix last-row logits by less than 2e-5
- Cache stores exactly two width-sized vectors per layer and position
- Top-k one always returns the maximum-logit token
- Context overflow returns an error


## 41. Quantization

Part: Efficient and adapted LLMs. Prerequisites: 40.

Project: Quantize and evaluate a linear layer. Dataset: Deterministic weights and activations.

Lessons: Code and terminology carried forward; Map real weights onto a finite grid; Put two int4 weights in one byte; Evaluate quantization error and model loss separately; Storage savings do not promise speed; Run, debug, and extend; Before you move on.

Introduced concepts: Int8; int4; dequantization; quality and speed.

Outcomes:

- Quantize weight rows to signed int8 and int4
- Pack and sign-extend two int4 values per byte
- Compare dequantized matrix-vector multiplication with a scalar f32 reference
- Separate storage, quantization error, decoder loss, and measured runtime

Completion checks:

- Packed int4 round-trips signed values including an odd tail
- Int8 and int4 matvec errors stay within fixture tolerances
- Int4 storage is smaller than int8 including scales
- Decoder comparison preserves the Chapter 36 [D,V] output-weight layout and reports logits, loss, and token decision
- Benchmark reports warmed median samples and checksums


## 42. Efficient attention

Part: Efficient and adapted LLMs. Prerequisites: 41.

Project: Compare attention algorithms. Dataset: Deterministic attention tensors.

Lessons: Keep a scalar causal-attention oracle; Update softmax state without all scores; Tile exact causal attention; Share keys and values across query heads; Run, diagnose, and transfer; Before you move on.

Introduced concepts: Online softmax; tiled attention; grouped-query attention; context costs.

Outcomes:

- Derive the online-softmax recurrence
- Implement exact causal attention over key tiles
- Verify tiled output against a scalar two-pass oracle
- Map grouped query heads to shared key/value heads
- Account for long-context arithmetic and intermediate storage

Completion checks:

- Tiled attention matches the scalar oracle within 2e-6 + 2e-6 times absolute reference across key-tile widths
- First causal output equals its only visible value
- Increasing maxima preserve a hand-computed weighted numerator
- Invalid or overflowing shapes and nonfinite scores are rejected
- GQA requires a divisible head mapping
- A maximum-sized tile does not overflow index arithmetic


## 43. Adaptation

Part: Efficient and adapted LLMs. Prerequisites: 42.

Project: Adapt a small model. Dataset: Course-authored instruction pairs.

Lessons: Supervised fine-tuning continues next-token training; Change a projection through a low-rank path; Learn a teacher’s distribution; Choose the adaptation boundary; Run, debug, and extend; Before you move on.

Introduced concepts: Supervised fine-tuning; LoRA; distillation.

Outcomes:

- Run all-parameter supervised fine-tuning on a decoder
- Apply and differentiate a zero-initialized LoRA projection
- Prove the base remains frozen
- Distill teacher probabilities into a smaller decoder
- Separate fit from held-out adaptation evidence

Completion checks:

- SFT lowers decoder training loss
- LoRA starts with exactly zero parameter change and lowers loss
- Base parameter values remain unchanged after adapter training
- One adapter gradient matches central differences
- Soft-target gradient matches direct cross-entropy finite differences
- Student KL to teacher falls and remains correct for a confidently wrong student


## 44. Retrieval and RAG

Part: Efficient and adapted LLMs. Prerequisites: 43.

Project: Answer from a local document collection. Dataset: Course-authored fact corpus.

Lessons: Begin with a lexical baseline; Learn embeddings and search them exactly; Assemble evidence before answering; Evaluate support and abstain; Run, debug, and extend; Before you move on.

Introduced concepts: Retrieval baselines; embeddings; indexing; grounded evaluation.

Outcomes:

- Implement and diagnose a BM25 lexical baseline
- Train query and document embeddings from explicit labeled pairs
- Search a dense index exactly
- Assemble source-labeled context and extract an answer
- Measure hit@1, attribution, coverage, and abstention

Completion checks:

- BM25 ranks an exact-term fact first
- Both query and document table gradients match direct-loss central differences
- Training receives explicit labeled data and uses the documented learning rate
- Dense exact search retrieves trained semantic pairs and held-out feature combinations
- Changing retrieved fact text changes the extracted answer without retraining
- Every answered fixture includes a matching source label
- Unknown-content unsupported query abstains


## 45. Reinforcement learning foundations

Part: Efficient and adapted LLMs. Prerequisites: 44.

Project: Learn a policy in a tiny environment. Dataset: Course-authored bandit and gridworld.

Lessons: Learning from consequences; Bandits and exploration; Tabular control; Policy gradients; Rust experiment; Practice and transfer.

Introduced concepts: Bandits; tabular control; policy gradients.

Outcomes:

- Estimate action values with epsilon-greedy bandit interaction
- Derive and implement the tabular Q-learning control update
- Train a categorical policy with an actual REINFORCE gradient
- Distinguish sampled evidence from convergence guarantees

Completion checks:

- Eight thousand seeded bandit steps identify arm 2 as best
- Q-learning makes right the greedy action in all four nonterminal states and matches exact path returns
- Terminal transitions ignore sentinel continuation values and cannot restart the episode
- Five thousand seeded REINFORCE episodes assign action 1 probability above 0.9
- Invalid epsilon and empty bandit runs are rejected


## 46. Preference and reward training

Part: Efficient and adapted LLMs. Prerequisites: 45.

Project: Improve a toy policy from feedback. Dataset: Course-authored preference pairs.

Lessons: From reward to preferences; DPO arithmetic; Reference policy; Verifiable reward; Rust experiment; Practice and transfer.

Introduced concepts: DPO; reference policy; preference pairs; verifiable reward.

Outcomes:

- Compute DPO loss from chosen, rejected, policy, and frozen-reference log probabilities
- Derive and check a mean DPO logit gradient stored separately from policy parameters
- Train a toy preference policy and a separate verifiable-reward policy
- Explain what the reference policy and beta control

Completion checks:

- DPO mean loss falls from ln 2 below 0.18
- Each chosen response has the highest learned logit for its prompt
- One DPO logit gradient matches a central difference within 1e-6
- Stable log-softmax retains normalization at large common offsets
- Reversing a shared-prompt pair batch preserves its update
- The fused loss_and_gradient returns a distinct Gradient with the policy-logit shape
- Invalid rows and overflowing arithmetic are rejected; failed overflowing updates preserve old logits
- Verifiable-reward training assigns the correct action probability above 0.99


## 47. Sparsity and pruning

Part: Advanced architectures & engineering. Prerequisites: 46.

Project: Prune and execute a sparse model. Dataset: Deterministic matrices.

Lessons: Why zeros need a representation; Magnitude pruning; CSR arithmetic; Correct benchmarking; Rust experiment; Practice and transfer.

Introduced concepts: Sparse representations; pruning; sparse kernels; measured savings.

Outcomes:

- Prune a dense matrix by global weight magnitude
- Construct and validate compressed sparse row storage
- Compare checked scalar f64 dense and CSR matrix-vector kernels within tolerance
- Measure kernel time and output error across several densities

Completion checks:

- CSR preserves row pointers [0,1,1,3], columns [1,0,2], and values [2,-3,4] for the worked fixture
- CSR and dense outputs agree within 1e-10 absolute plus 1e-8 relative tolerance
- Fifty-percent pruning keeps the two largest magnitudes and resolves exact ties by original index
- Default run reports 31-sample kernel median and range plus output RMSE at four densities


## 48. Mixture of experts

Part: Advanced architectures & engineering. Prerequisites: 47.

Project: Train a small expert layer. Dataset: Synthetic routed examples.

Lessons: Conditional computation; Routing gradients; Capacity and overflow; Balancing experts; Rust experiment; Practice and transfer.

Introduced concepts: Routing; expert capacity; load balancing; conditional compute.

Outcomes:

- Train experts and a router jointly through a selected full-softmax gate
- Derive why renormalized top-1 gating removes the task-loss router gradient
- Enforce per-expert capacity and account for overflow
- Compute a Switch-style differentiable load-balancing loss

Completion checks:

- Router and selected-expert gradients match central differences away from a routing tie
- Uncapped task-only half-MSE falls below one fifth of its initial value
- All three experts receive routes after training on the deterministic fixture
- Overflow count equals assignments beyond the computed per-expert capacity
- Accepted task gradients divide by all attempted examples even when capacity drops one


## 49. State-space models and linear attention

Part: Advanced architectures & engineering. Prerequisites: 48.

Project: Compare sequence computation methods. Dataset: Synthetic sequences.

Lessons: Sequence state without a cache; Selective recurrence; Associative scan; Linear attention; Rust comparison; Practice and transfer.

Introduced concepts: Recurrence; scans; selective state; linear attention.

Outcomes:

- Compute a selective state-space sequence as a recurrence
- Derive associative affine-transition composition and verify scan parity
- Implement normalized causal linear attention with recurrent summaries
- Compare recurrence, scan schedule, and a direct scalar oracle without false speed claims

Completion checks:

- Associative scan matches sequential recurrence within absolute-plus-relative tolerance
- Affine composition passes an explicit associativity check
- causal_linear_attention_recurrent matches causal_linear_attention_scalar
- Empty, mismatched, and nonfinite sequence inputs are rejected


## 50. Distributed training

Part: Advanced architectures & engineering. Prerequisites: 49.

Project: Reproduce a multi-worker training step. Dataset: Deterministic local worker batches.

Lessons: Why one worker stops being enough; Data-parallel gradients; Tensor and pipeline partitions; Recovery and equivalence; Run, debug, and transfer.

Introduced concepts: Data parallelism; tensor parallelism; pipeline stages; recovery.

Outcomes:

- Reproduce a serial gradient update with communicating data-parallel workers
- Trace tensor-sharded forward and backward reductions
- Send activations forward and gradients backward through two pipeline stages
- Resume minibatch training from validated state and explain the limits of equivalence

Completion checks:

- Data-parallel steps match serial steps within 1e-12 for batch sizes 1, 3, and 5
- Tensor-parallel prediction and parameter update match the serial reference within 1e-12
- Two-stage pipeline backward matches serial and finite-difference gradients for three- and four-example data slices
- Empty or nonfinite pipeline data, invalid learning rates, and nonfinite updates are rejected
- CH50v1 parameter order and cursor bytes remain stable, and a restored two-minibatch run equals the uninterrupted local run
- Malformed and nonfinite checkpoints are rejected


## 51. Generative modeling beyond text

Part: Advanced architectures & engineering. Prerequisites: 50.

Project: Generate a small synthetic distribution. Dataset: Course-authored distributions.

Lessons: Three ways to model a distribution; Variational autoencoders; Adversarial training; Diffusion and reverse denoising; Run, diagnose, and transfer.

Introduced concepts: VAE; GAN; diffusion; sampling; objectives.

Outcomes:

- Train and sample a one-dimensional beta-VAE with reparameterized latent noise
- Alternate discriminator and non-saturating generator updates in a trainable GAN
- Train a timestep-conditioned noise predictor and execute a complete deterministic reverse chain
- Distinguish each model's training objective from generated-sample quality

Completion checks:

- The beta-VAE objective falls by at least 30 percent
- GAN training changes both generator and discriminator parameters and yields finite samples
- Diffusion noise MSE falls by at least 20 percent
- A perfect supplied noise value maps each diffusion state to the exact previous deterministic state
- Explicit alternate data changes the VAE objective, and invalid empty, nonfinite, mismatched, or diffusion-schedule inputs are rejected
- Default execution trains all three models and reports 512-sample distribution summaries, distinguishing VAE means from full observations


## 52. Multimodal models

Part: Advanced architectures & engineering. Prerequisites: 51.

Project: Match small images and captions. Dataset: Course-authored image-caption pairs.

Lessons: Two modalities, one comparison space; Contrastive alignment by hand; Train the Rust dual encoder; Retrieval evaluation and failure modes; Practice and transfer.

Introduced concepts: Image and text encoders; contrastive alignment; retrieval.

Outcomes:

- Convert tiny pixels and caption words into modality-specific vectors
- Normalize embeddings and build an image-by-text similarity matrix
- Train both encoders with symmetric in-batch contrastive cross-entropy
- Measure image-to-text Recall@1 on perturbed paired examples

Completion checks:

- Symmetric contrastive loss falls below one quarter of its initial value
- Training updates both the image and text parameter matrices
- Training-pair Recall@1 reaches 100 percent
- Perturbed variants of the three same base patterns reach 100 percent Recall@1
- Alternate supplied images, candidates, and temperature change the corresponding calculations
- Zero, nonfinite, and out-of-vocabulary inputs are rejected


## 53. Production ML

Part: Advanced architectures & engineering. Prerequisites: 52.

Project: Package and monitor an inference service. Dataset: Versioned local model and requests.

Lessons: The model is a pipeline artifact; Version and validate checkpoints; Serve through a bounded contract; Monitor drift and roll back; Operate, debug, and transfer.

Introduced concepts: Pipelines; serving; monitoring; drift; versioning; rollback.

Outcomes:

- Train and atomically save a model with preprocessing, schema, version, and data fingerprint
- Reject malformed or ambiguous requests before inference
- Serve a validated checkpoint through a real local HTTP endpoint
- Detect a mean shift and roll back a failed canary to a prior artifact

Completion checks:

- A checkpoint round trip preserves every model and preprocessing field
- Unsupported formats, nonfinite values, duplicate Content-Length, Transfer-Encoding, extra fields, and oversized requests are rejected
- The default demo detects drift and selects model version one after version two fails its canary
- The explicit train command writes a loadable artifact
- The host-only HTTP test returns a successful response for one valid local POST


## 54. Responsible evaluation

Part: Advanced architectures & engineering. Prerequisites: 53.

Project: Audit a model and its evidence. Dataset: Course-authored grouped evaluation data.

Lessons: One score cannot audit every risk; Interpretability and robustness; Fairness and privacy measurements; Security boundaries and causal limits; Run, challenge, and transfer.

Introduced concepts: Interpretability; robustness; privacy; security; fairness; causal limits.

Outcomes:

- Compute feature ablation, perturbation, subgroup, and membership-attack measurements
- Name the denominator, threat model, and evidence boundary for each metric
- Explain why adding noise alone does not establish differential privacy
- Distinguish predictive association from a causal intervention claim

Completion checks:

- Group A and B metrics report counts, accuracy, positive rate, TPR, and FPR
- A radius-0.25 perturbation reports boundary flips and maximum probability change
- A loss-threshold membership attack evaluates a model trained only on marked member rows
- Model::loss returns one row's stable binary cross-entropy rather than a dataset mean
- Nonfinite, out-of-range, and wrong-schema inputs are rejected before scoring
- The Chapter 54 comma-separated two-feature input is explicitly distinguished from Chapter 53's form-encoded scalar request
- The crude association is +0.30 while the equal-stratum standardized association is -0.10


## 55. Working with established tools

Part: Advanced architectures & engineering. Prerequisites: 54.

Project: Export and verify a simple model. Dataset: Course-authored model fixture.

Lessons: Specify the computation contract; Work the affine update by hand; Separate autodiff from optimization; Export and restore inference state; Run and debug an optional integration; Retrieve the parity contract.

Introduced concepts: Framework concepts; runtime interchange; parity checks; optional integration.

Outcomes:

- Map an affine model between scratch [out,in] storage and framework matrix multiplication
- Check predictions, mean loss, gradients, both SGD updates, and restored inference
- Export actual scratch weights and biases and import them into the optional Burn CPU runtime
- Distinguish numerical parity, backend records, graph interchange, and training resume

Completion checks:

- Scratch predicts [1.15,0.65] with MSE 0.6725 within f32 tolerance
- Framework test executes real forward, MSE, gradient, weight and bias update, record and restored-output assertions
- Both updated biases match [0.035,-0.315] and updated output matches [1.04125,-0.18375]
- Scratch export is imported and compared on three input probes in Burn
- Wrong headers, length, and nonfinite artifact values are rejected
- Starter launches successfully and its derivative TODO fails the intended test


## 56. Advanced capstone

Part: Advanced architectures & engineering. Prerequisites: 55.

Project: Train and serve a small MoE LM. Dataset: Course-authored text; licensed corpus extension.

Lessons: Preserve the contextual decoder; Keep the selected router probability; Derive capacity and balancing; Run a controlled dense comparison; Train, resume, load and serve; Debug and challenge the capstone.

Introduced concepts: Sparse MoE language model; dense baseline; training; serving; evaluation.

Outcomes:

- Train an actual pre-normalized causal decoder with sparse expert feed-forward layers
- Derive and check task and auxiliary router gradients away from hard-routing ties
- Compare dense and sparse models at the same token budget with explicit parameter-count conventions
- Resume the identical next SGD step from validated corpus and model state
- Generate text from a saved model through a bounded local HTTP endpoint
- Separate tiny-fixture evidence from language quality and performance claims

Completion checks:

- One-expert logits agree with independent dense decoder oracle within 1e-12
- Future-token perturbation and prefix shortening preserve earlier inference logits
- All small-model parameter derivatives pass f64 finite differences away from ties
- Router task gradients are nonzero with auxiliary coefficient zero; unselected expert task gradients are zero
- Capacity counts attempted and accepted routes; dropped tokens keep task loss and shared gradients but have no expert gradient
- Public parameter span labels, order, ranges and the 3,303/2,879 total/active counts remain exact
- Checkpoint next update resumes exactly and rejects malformed or changed training state
- Fragmented requests yield complete 200 or 400 responses with correct byte length
- Default 160-step run updates router, all experts, embeddings, attention and normalization
- Custom corpus inputs require an explicit held-out file and reject equal/shared 32-byte passages

