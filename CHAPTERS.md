# Chapters

This map is generated from course.json and chapter metadata. Lectures are HTML, not Markdown.

## 01. One neuron learns from data

Part: First principles. Prerequisites: Programming basics only.

Project: Fit a line with one neuron. Dataset: Course-authored line samples.

Lessons: Examples and targets; The prediction rule; A numerical loss; Numerical slopes and updates; Run and check the program; Practice and transfer.

Introduced concepts: Prediction; weights and bias; squared loss; numerical slopes; training and inference.

Outcomes:

- Train one scalar neuron without ML libraries
- Calculate prediction and mean squared error by hand
- Explain training versus inference
- Diagnose the update sign and learning rate

Completion checks:

- Starting MSE equals 9
- First step is weight 0.8, bias 0.2 within 1e-8
- 100 steps yield MSE below 1e-12
- Held-out prediction at 0.5 is within 1e-7 of 2
- Invalid data and learning rates are rejected


## 02. Why learning works

Part: First principles. Prerequisites: 01.

Project: Check an analytical gradient. Dataset: Course-authored line samples.

Lessons: From numerical probes to derivatives; Derive the gradient; Check before trusting; Build and run the Rust project; Practice and transfer.

Introduced concepts: Derivatives; gradients; analytical updates; finite-difference checks.

Outcomes:

- Derive exact MSE gradients for a weight and bias
- Perform a simultaneous analytical update
- Check derivatives with central differences
- Diagnose sign, averaging, and stale-parameter bugs

Completion checks:

- Analytical gradient at zero equals (-8, -2)
- Central differences match at two parameter settings within atol 1e-6 plus rtol 1e-4
- One step at rate 0.1 produces weight 0.8 and bias 0.2
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

Lessons: From numbers to categories; Sigmoid and probability; Likelihood and cross-entropy; Stable Rust implementation; Practice and transfer.

Introduced concepts: Probability; sigmoid; likelihood; stable binary cross-entropy.

Outcomes:

- Convert a linear score into a probability with sigmoid
- Derive binary cross-entropy from Bernoulli likelihood
- Compute stable loss directly from logits
- Train and threshold a binary logistic classifier

Completion checks:

- BCE at logit zero equals ln 2
- Stable BCE remains finite at logits plus and minus 1000
- Analytical gradients match central differences on the stable logits objective
- Training loss falls below 0.04
- Every fixture point is classified correctly at threshold 0.5


## 05. Measuring learning

Part: First principles. Prerequisites: 04.

Project: Audit a classifier evaluation. Dataset: Synthetic imbalanced classification.

Lessons: What a training score cannot tell you; Splits and leakage; Confusion counts and metrics; Thresholds and baselines; Run the audit; Practice and transfer.

Introduced concepts: Splits; leakage; metrics; thresholds; baselines.

Outcomes:

- Separate training, validation, and test decisions
- Detect entity overlap and preprocessing leakage
- Compute confusion counts and derived metrics
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
- Derive and implement L2 regularization
- Distinguish optimization from generalization

Completion checks:

- A reliable rate lowers held-out MSE below 0.2
- The learned weight is within 0.15 of two
- L2 produces a smaller absolute weight than the unregularized run
- Invalid batch sizes and hyperparameters are rejected


## 07. From a neuron to XOR

Part: Build a neural network. Prerequisites: 06.

Project: Solve XOR with a small MLP. Dataset: XOR truth table.

Lessons: The XOR limit; Manual chain rule; Stable objective and checking; The Rust network; Backward-pass practice; Review and transfer.

Introduced concepts: Nonlinearities; hidden layers; manual backpropagation.

Outcomes:

- Explain why a linear neuron cannot represent XOR
- Trace a 2-2-1 network forward by hand
- Derive and implement both layers of backpropagation
- Check hidden and output gradients with central differences

Completion checks:

- 10,000 full-batch updates produce XOR loss below 0.02
- All four XOR labels are classified correctly
- Hidden and output weight gradients match central differences
- Nonpositive learning rates are rejected
- The actual update matches central differences for all nine parameters


## 08. Automatic differentiation

Part: Build a neural network. Prerequisites: 07.

Project: Build scalar autodiff. Dataset: Synthetic computation graphs.

Lessons: Computation graphs; Reverse accumulation; Topological ordering; The Rust engine; Graph practice; Review and transfer.

Introduced concepts: Computation graphs; reverse mode; accumulated gradients.

Outcomes:

- Represent scalar calculations as a directed acyclic graph
- Explain reverse-mode adjoints and topological order
- Accumulate gradients through shared nodes
- Implement and numerically check scalar autodiff

Completion checks:

- x*x+x at x=3 produces value 12 and gradient 7
- A tanh chain matches central differences
- Repeated backward calls clear reachable gradients
- Shared nodes are traversed once but receive every operand contribution
- Nonfinite intermediates and overflowing adjoints return errors even with a finite root


## 09. Tensors and batches

Part: Build a neural network. Prerequisites: 08.

Project: Build a batched dense layer. Dataset: Hand-computed matrices.

Lessons: Shapes and contiguous layout; Dense-layer gradients; Reference kernels; Run the matrix checkpoint; Shape practice; Review and transfer.

Introduced concepts: Shapes; contiguous storage; matrix operations; batched gradients.

Outcomes:

- Interpret contiguous row-major data using shapes
- Compute a batched dense forward pass
- Derive gradients for inputs, weights, and bias
- Reject incompatible shapes and check a weight gradient numerically

Completion checks:

- Forward fixture equals [-1.5,3.5,-1.5,12.5]
- dX, dW, and bias gradients match hand calculations
- A weight gradient matches central differences
- Invalid matrix and backward shapes return errors


## 10. Recognizing digits with a linear model

Part: Build a neural network. Prerequisites: 09.

Project: Train a digit classifier. Dataset: MNIST plus generated IDX fixture.

Lessons: IDX files; Softmax cross-entropy; Linear classifier training; Fixture and MNIST modes; Stable-classification practice; Review and transfer.

Introduced concepts: IDX format; MNIST; softmax; multiclass evaluation.

Outcomes:

- Validate and parse big-endian IDX image and label files
- Compute stable softmax and cross-entropy from logits
- Train a ten-class linear model with minibatch gradients
- Separate validation choices from final test evaluation

Completion checks:

- Generated bytes parse through the real IDX loader
- Malformed magic and payload sizes are rejected
- Fixture cross-entropy falls by at least 80 percent
- Fixture held-out accuracy reaches 100 percent
- Explicit MNIST mode requires readable paths
- Cross-entropy retains ln 10 for ten equal logits at a common offset of 1e16


## 11. An MNIST neural network

Part: Build a neural network. Prerequisites: 10.

Project: Train and restore a digit MLP. Dataset: MNIST plus tiny generated fixture.

Lessons: A nonlinear digit model; Initialization; Momentum and Adam; Complete checkpoints; Train and practice; Review and transfer.

Introduced concepts: Initialization; momentum; Adam; debugging; checkpoints.

Outcomes:

- Train a one-hidden-layer ReLU classifier with real gradients
- Explain initialization and diagnose inactive units
- Implement momentum and Adam updates
- Save and validate complete resumable optimizer checkpoints

Completion checks:

- Both momentum and Adam reduce fixture loss
- Fixture accuracy exceeds two thirds
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
- Bootstrap draws complete prediction-label pairs with replacement
- Tests reject empty or malformed predictions
- Bootstrap endpoints distinguish all-correct, all-wrong, and mixed observations; calibration bins match counts and means
- Lesson distinguishes interval procedure coverage from probability about a fixed parameter


## 13. Real tabular data

Part: Beyond neural networks. Prerequisites: 12.

Project: Build a leakage-safe preprocessor. Dataset: Tiny local tabular fixture; UCI Auto MPG extension.

Lessons: Split before learning anything; Missing is information about measurement; Categories need a stable vocabulary; Groups and time define believable tests; Build the Rust preprocessor; Practice and review.

Introduced concepts: Missing values; categories; grouped splits; temporal splits.

Outcomes:

- Identify preprocessing operations that learn fitted state
- Fit median imputation and a category vocabulary on training rows only
- Construct group-isolated and past-to-future splits
- Transform unseen categories with a stable feature shape

Completion checks:

- Reference fits median and categories after the split
- Held-out value 999 and category secret cannot alter fitted state
- Unknown categories preserve output dimension
- Group and temporal split behaviors have runnable tests
- Even-count median remains finite at extreme finite inputs; missing indicators remain distinct


## 14. Nearest neighbors and naive Bayes

Part: Beyond neural networks. Prerequisites: 13.

Project: Compare two classifiers. Dataset: Local three-class fixture; UCI Iris extension.

Lessons: Distance turns similarity into arithmetic; Scale before measuring neighbors; k-nearest neighbors votes from memory; Gaussian naive Bayes models each class; Run a fair small comparison; Practice, debugging, and transfer.

Introduced concepts: Distance; scaling; nearest neighbors; Gaussian naive Bayes.

Outcomes:

- Compute and interpret squared Euclidean distance
- Fit and apply training-only standardization
- Implement deterministic multiclass k-nearest-neighbor voting
- Fit Gaussian naive Bayes and compare log class scores

Completion checks:

- kNN rejects invalid k and nonfinite coordinates
- Both classifiers identify the middle fixture cluster
- Gaussian likelihoods are accumulated in log space
- Scaling state is fitted once and reused
- Nonfinite queries and overflowed fitted statistics, distances, or Gaussian scores return errors


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

- Baseline tree finds a Gini-reducing threshold
- Forest bootstraps rows and samples a candidate feature at every recursive node
- Boosting fits mean residuals and converges on the tiny regression fixture
- Starter and Rustlings exercises verify Gini arithmetic


## 16. Margins and kernels

Part: Beyond neural networks. Prerequisites: 15.

Project: Classify linear and curved boundaries. Dataset: Synthetic points.

Lessons: From a correct boundary to a confident boundary; Hinge loss spends effort near the margin; Turn the objective into Rust updates; Curve the boundary with similarity; Run, inspect, and practice; Review and transfer.

Introduced concepts: Hinge loss; linear SVM; kernel similarity.

Outcomes:

- Compute signed margins and hinge loss for labels encoded as -1 and +1
- Train and evaluate a two-feature linear soft-margin SVM
- Explain how an RBF kernel turns distances into nonlinear influence
- Diagnose label-encoding and kernel-inference bugs

Completion checks:

- The zero model has average hinge objective 1 on the linear fixture
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
- Project and reconstruct rows with one principal component
- Interpret explained variance, eigengaps, and covariance conditioning

Completion checks:

- Power iteration recovers eigenvalue 3 for the matrix [[2,1],[1,2]]
- The returned component has unit norm and a small eigenpair residual
- Translating all rows preserves eigenvalues, directions, and reconstruction error
- Rank-one data reconstructs with negligible one-component error
- Too few or nonfinite rows are rejected


## 18. Clustering and density

Part: Beyond neural networks. Prerequisites: 17.

Project: Discover synthetic clusters. Dataset: Synthetic Gaussian clusters.

Lessons: Alternate between centers and assignments; Replace a hard group with a probability; Fit the mixture with expectation-maximization; Turn density into an anomaly score; Run the models and practice.

Introduced concepts: K-means; Gaussian mixtures; EM; anomaly scores.

Outcomes:

- Run deterministic k-means and compute inertia
- Explain Gaussian-mixture responsibilities and diagonal covariance
- Implement stable expectation-maximization with log-sum-exp
- Use negative log density as an anomaly score without confusing rarity with harm

Completion checks:

- K-means finds one negative and one positive fixture center with inertia below 3
- Final assignments are recomputed from the returned centers
- Mixture responsibilities sum to one and component weights remain normalized
- The distant point receives a larger anomaly score than a cluster-center point
- Later well-conditioned EM fits do not reduce fixture log likelihood beyond tolerance


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
- Training reduces validation cross-entropy and reaches at least 70% fixture accuracy
- Malformed IDX headers are rejected
- Default execution performs no network access


## 21. Deeper vision models

Part: Broader deep learning. Prerequisites: 20.

Project: Train a residual image block. Dataset: Course-authored images.

Lessons: Why deeper stacks become hard to optimize; Residual arithmetic; Normalization; Augmentation and training; Practice and debug; Review and transfer.

Introduced concepts: Residual connections; normalization; augmentation.

Outcomes:

- Explain an identity shortcut and trace its gradient
- Normalize an image with explicit mean, variance, and epsilon
- Generate label-preserving translations without crossing split boundaries
- Train two small pre-normalized residual convolutional blocks

Completion checks:

- First residual-block kernel analytic and numerical gradients agree through both blocks
- Normalized fixture images have mean approximately zero
- Training reduces held-out loss and classifies all four canonical images
- Augmentation is deterministic and limited to training rows


## 22. Learning representations

Part: Broader deep learning. Prerequisites: 21.

Project: Learn a useful small representation. Dataset: Course-authored paired vectors.

Lessons: Targets can come from the input; Autoencoder backpropagation; Contrastive pairs and negatives; Run both experiments; Practice and debug; Review and transfer.

Introduced concepts: Autoencoders; bottlenecks; contrastive learning.

Outcomes:

- Train an autoencoder through a two-value bottleneck
- Distinguish reconstruction quality from representation usefulness
- Compute stable InfoNCE directly from similarity logits
- Train a contrastive encoder and evaluate paired retrieval

Completion checks:

- Autoencoder training reduces reconstruction MSE by the declared factor
- The bottleneck has exactly two coordinates
- InfoNCE uses stable log-sum-exp from logits
- Contrastive training reduces its objective and reports paired retrieval separately


## 23. Sequences and forecasting

Part: Broader deep learning. Prerequisites: 22.

Project: Forecast a held-out signal. Dataset: Course-authored time series.

Lessons: Order changes the prediction problem; RNNs and BPTT; LSTM memory; Temporal evaluation; Practice and debug; Review and transfer.

Introduced concepts: RNNs; BPTT; LSTMs; temporal evaluation.

Outcomes:

- Unroll a recurrent state over time
- Derive and check BPTT gradients through shared recurrent weights
- Trace LSTM input, forget, output, and candidate paths
- Evaluate one-step forecasts on a later contiguous segment

Completion checks:

- Analytic RNN recurrent-weight and LSTM candidate-weight BPTT match central differences
- Both RNN and LSTM reduce later one-step validation MSE
- Training targets all precede validation targets
- Stable sigmoid avoids overflow for large negative inputs


## 24. Recommendation systems

Part: Broader deep learning. Prerequisites: 23.

Project: Recommend held-out items. Dataset: Course-authored interaction matrix.

Lessons: From sparse interactions to vectors; Pairwise ranking; Ranking metrics; Run the recommender; Practice and debug; Review and transfer.

Introduced concepts: Embeddings; matrix factorization; ranking metrics.

Outcomes:

- Compute preference scores from user and item embeddings
- Train matrix factors with a stable pairwise ranking loss
- Keep held-out positives out of optimization
- Calculate Recall@k, Precision@k, and nDCG@k from ranked candidates

Completion checks:

- Pairwise analytic and numerical gradients agree
- Stable softplus remains finite at extreme margins
- Training lowers pairwise loss and improves held-out ranking
- Metrics exclude training positives and use a declared candidate set


## 25. Measure before optimizing

Part: Make the CPU faster. Prerequisites: 24.

Project: Write an honest CPU benchmark. Dataset: Deterministic matrices.

Lessons: Floating-point evidence; Controlled measurement; Work and throughput; Memory costs and arithmetic intensity; Benchmark practice; Review and transfer.

Introduced concepts: Floating point; profiling; memory costs; arithmetic intensity.

Outcomes:

- Explain why optimized f32 results need tolerance-based comparison
- Construct a release-mode benchmark with warmups and repeated samples
- Report latency, variability, work, and measured throughput
- Calculate and qualify an arithmetic-intensity estimate
- Use profiling evidence to choose a useful optimization target

Completion checks:

- Dense f32 output agrees with an f64 accumulator on a deterministic odd-shaped fixture
- Bad dense shapes return an error
- No-argument benchmark uses three warmups and eleven measured repetitions
- Benchmark reports median, range, checksum, GFLOP/s, cold-array traffic estimate, and estimated FLOP/byte
- Starter run succeeds while its guided median test reaches the TODO


## 26. Fast matrix multiplication

Part: Make the CPU faster. Prerequisites: 25.

Project: Optimize a matrix kernel. Dataset: Deterministic rectangular matrices.

Lessons: Matrix traversal; Loop ordering; Cache blocking; Allocation reuse; Locality practice; Review and transfer.

Introduced concepts: Loop ordering; cache locality; blocking; allocation reuse.

Outcomes:

- Map matrix notation to row-major offsets for nonsquare shapes
- Explain how loop order changes spatial locality
- Implement a cache-blocked scalar matrix multiplication with edge tiles
- Reuse and correctly clear output storage
- Benchmark multiple correct kernels without claiming a predetermined winner

Completion checks:

- i-j-k, i-k-j, and blocked kernels agree with an f64 reference
- The correctness fixture uses m=3, k=5, n=7 and blocks 1, 4, 16, and usize::MAX
- Zero dimensions, mismatched storage, and a zero block return errors
- Each timed kernel reuses one output allocation and reports median, range, and GFLOP/s
- Starter run produces the prior scalar result while its new loop-order test reaches the TODO


## 27. Multithreaded training and inference

Part: Make the CPU faster. Prerequisites: 26.

Project: Parallelize dense computation. Dataset: Deterministic batches.

Lessons: Batch partitioning; Private gradients; Deterministic reduction; Parallel project; Concurrency practice; Review and transfer.

Introduced concepts: Partitioning; scoped threads; gradient reduction; reproducibility.

Outcomes:

- Partition batch rows into complete nonoverlapping thread shards
- Use scoped threads with borrowed inputs and disjoint mutable inference outputs
- Compute private per-shard gradients and reduce them in fixed order
- Run actual parallel training with one simultaneous update
- State the practical boundary of floating-point reproducibility

Completion checks:

- Parallel inference matches scalar output for every row
- Four-shard loss and gradients agree with scalar accumulation within tolerance
- Three-shard training on 31 rows lowers MSE by at least 1000 times
- Empty inference succeeds while empty training, zero threads, and invalid shapes fail
- Gradient shards are reduced in creation order before one parameter update


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

- Portable dispatch rejects unequal lengths
- Default dispatch handles every length 0 through 39 and odd length 1003
- Every runtime-supported AVX2/FMA and AVX-512F backend is directly compared with an oracle
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
- Explain how Rust buffers and WGSL bindings represent contiguous tensors
- Dispatch enough workgroups for an arbitrary vector length
- Synchronize a real GPU result into a mapped readback buffer
- Compare GPU output with a scalar oracle using an explicit tolerance

Completion checks:

- The executable reports every discovered adapter and the selected hardware Vulkan adapter
- A five-element vector is dispatched and read back from a real compute shader
- The ignored hardware test covers 67 elements, including a partial workgroup
- Empty, mismatched, and nonfinite inputs return errors before dispatch


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
- Use the reusable Gpu matmul API and compare it with a scalar oracle

Completion checks:

- The scalar oracle handles a nonsquare 3×17 by 17×5 product
- The ignored hardware test matches that odd-tile product within absolute-plus-relative tolerance
- A 777-value staged reduction matches the scalar sum
- Zero, overflowing, malformed, nonfinite, or device-exceeding shapes return errors
- Gpu stores a persistent device, queue, matmul pipeline, and reduction pipeline


## 31. Train on the GPU

Part: Write GPU kernels. Prerequisites: 30.

Project: Train a GPU neural network. Dataset: Synthetic nonlinear points.

Lessons: Why repeated host transfers stop being training; Work a nonlinear forward and backward pass; Order three device-resident kernels; Check gradients and training behavior; Run, debug, and extend the project; Review and transfer.

Introduced concepts: Forward and backward kernels; updates; residency.

Outcomes:

- Map a nonlinear network's forward, backward, and update equations to separate GPU kernels
- Keep inputs, activations, gradients, losses, and weights in device buffers across training steps
- Compute stable binary cross-entropy directly from logits
- Check selected analytical gradients with central differences
- Verify GPU training against a scalar oracle and a falling-loss criterion

Completion checks:

- CPU analytical gradients match central differences for input, hidden, output, and bias parameters
- Stable logit loss remains finite for logits of magnitude 1000
- CPU training reduces nonlinear XOR loss by at least seventy percent
- The ignored hardware test checks GPU loss decrease and all weights against the CPU oracle
- No training tensor is mapped between forward, backward, and update steps


## 32. GPU performance

Part: Write GPU kernels. Prerequisites: 31.

Project: Measure and improve GPU kernels. Dataset: Deterministic GPU workloads.

Lessons: Performance begins with a well-defined clock; Understand asynchronous submission; Fuse adjacent elementwise kernels; Choose mixed precision by capability and range; Connect wgpu to Vulkan extensions; Measure, debug, and transfer.

Introduced concepts: Asynchronous execution; timing; fusion; mixed precision; Vulkan extension.

Outcomes:

- Separate host wall time from device timestamp duration
- Explain asynchronous queue submission and explicit completion points
- Fuse an affine transform and ReLU while preserving a scalar oracle
- Query timestamp and shader-f16 support before requesting optional features
- Relate wgpu's f16 feature to concrete Vulkan capability extensions

Completion checks:

- Separate and fused f32 outputs match the scalar oracle for an odd invocation count
- The default run uses warmups and reports repeated wall-clock median and range
- GPU timestamps are reported only when TIMESTAMP_QUERY is supported
- The f16 shader is created only when SHADER_F16 is supported
- Inputs outside the safe teaching range use the f32 fallback


## 33. Language modeling

Part: Train your language model. Prerequisites: 32.

Project: Train a tiny text predictor. Dataset: Course-authored text.

Lessons: Turn text into supervised examples; Counts are the first language model; Replace a token ID with learned coordinates; Train the Rust predictor; Practice and debug; Review and transfer.

Introduced concepts: Byte models; n-grams; embeddings; next-token loss.

Outcomes:

- Create byte and Unicode-character n-gram counts with explicit units
- Derive and train an embedding next-byte model with stable cross-entropy
- Trace gradients into both the selected embedding row and output matrix
- Distinguish training fit from held-out language quality

Completion checks:

- Reference tests confirm byte and character units differ
- Embedding and output parameters both change while loss falls
- Default project runs offline without arguments
- Starter runs and its guided test fails at a literal TODO


## 34. Tokenization

Part: Train your language model. Prerequisites: 33.

Project: Build and test a BPE tokenizer. Dataset: Course-authored multilingual text.

Lessons: Text arrives as UTF-8 bytes; Learn merges from adjacent pairs; Apply the learned ranks; Save enough information; Practice, debug, and extend; Review and transfer.

Introduced concepts: UTF-8; byte tokens; BPE training; serialization.

Outcomes:

- Explain UTF-8 byte sequences without conflating bytes and characters
- Train deterministic non-overlapping BPE merges
- Encode and decode arbitrary bytes exactly
- Serialize merge order and reject malformed tokenizer files

Completion checks:

- Multilingual byte round trips pass for ASCII, accented text, CJK, and emoji
- Saved and reloaded merge tables are equal
- Malformed headers and forward references return errors
- Starter prints UTF-8 unit counts before its guided test


## 35. Attention

Part: Train your language model. Prerequisites: 34.

Project: Implement causal self-attention. Dataset: Hand-computed sequence tensors.

Lessons: Ask, match, and retrieve; Mask information from the future; Reverse the weighted mixture; Check the whole chain numerically; Practice and investigate; Review and transfer.

Introduced concepts: Queries; keys; values; causal masking; gradients.

Outcomes:

- Compute scaled causal QKᵀ scores and weighted values
- Explain queries, keys, and values by role and shape
- Derive manual gradients for Q, K, V, and row softmax
- Use finite differences and causal invariance as complementary tests

Completion checks:

- Every Q, K, and V element passes f64 central differences
- Future changes cannot affect the first output
- Masked probabilities are exactly zero
- Default run traces weights, outputs, and nonzero gradient norms


## 36. A decoder Transformer

Part: Train your language model. Prerequisites: 35.

Project: Build a trainable decoder. Dataset: Course-authored text.

Lessons: Assemble a pre-normalized decoder block; Normalize features within each token; Store all trainable state in one checked buffer; Backpropagate tensors in reverse order; Verify CPU and optional GPU execution; Practice and debug; Review and transfer.

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

Lessons: Know what each document is; Clean with explicit rules; Remove repeated documents; Split complete documents and test contamination; Produce and inspect the audit; Practice, debug, and transfer.

Introduced concepts: Provenance; cleaning; deduplication; document splits; contamination.

Outcomes:

- Record source and license provenance for every document
- Apply explicit cleaning and inspect removed content
- Detect exact and near duplicates with a documented similarity rule
- Assign whole documents to stable splits and audit contamination

Completion checks:

- Malformed or empty documents return errors
- Near-duplicate fixture removes the intended copy
- Manifest retains source and license fields
- Document IDs are unique and assigned to exactly one split


## 38. Training at a larger scale

Part: Train your language model. Prerequisites: 37.

Project: Resume a reproducible LM run. Dataset: Course-authored text.

Lessons: Adapt each parameter's update scale; Change learning rate with progress; Accumulate microbatches; Account for memory; Resume the exact next step; Practice and diagnose; Review and transfer.

Introduced concepts: AdamW; schedules; accumulation; memory; resumable training.

Outcomes:

- Derive Adam moment, bias-correction, and decoupled-decay updates
- Combine equal microbatch gradients before global clipping
- Compute cosine decay with warmup per optimizer step
- Save and validate every state item required for exact continuation
- Estimate persistent memory and define a throughput measurement

Completion checks:

- Save-load branch produces bit-identical next loss and parameters
- Checkpoint restores cursor, RNG, schedule step, model, and moments
- Nonfinite gradients and checkpoint values are rejected
- Default run reports calculated persistent memory and measured loss


## 39. Scratch-trained language-model capstone

Part: Train your language model. Prerequisites: 38.

Project: Train your own dense language model. Dataset: Tiny local corpus; user-supplied licensed corpus.

Lessons: Define the experiment; Train the complete tiny decoder; Generate a continuation; Save, load, and continue; Budget the larger option; Practice, debug, and extend; Review and transfer.

Introduced concepts: Training; validation; checkpoints; generation; resource budgeting.

Outcomes:

- Train and evaluate a dense causal decoder on byte text
- Use distinct licensed training and validation documents without silent fallback
- Save, load, resume, and generate from the actual trained model
- Report exact parameter and persistent-memory budgets
- Measure tokens per second without fabricating scale claims

Completion checks:

- Tiny test trains the complete decoder, lowers fixture loss, saves, loads, and generates
- Default run evaluates a separate held-out document and reports measured throughput
- Large-info prints exactly 14,442,496 parameters without allocation
- Large mode instantiates the real Trainer and requires an explicit step budget
- Changed training data is rejected on resume


## 40. Inference systems

Part: Efficient and adapted LLMs. Prerequisites: 39.

Project: Build a cached text generator. Dataset: Chapter 39 model and fixture.

Lessons: Reuse the past without changing the model; Match full-prefix logits, layer by layer; Turn logits into a controlled choice; Batch iterations and stream completed tokens; Run and investigate; Before you move on.

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

Lessons: Map real weights onto a finite grid; Put two int4 weights in one byte; Evaluate error and model behavior separately; Storage savings do not promise speed; Run, debug, and extend; Before you move on.

Introduced concepts: Int8; int4; dequantization; quality and speed.

Outcomes:

- Quantize weight rows to signed int8 and int4
- Pack and sign-extend two int4 values per byte
- Run dequantized matrix-vector multiplication
- Separate storage, numerical error, decoder quality, and measured runtime

Completion checks:

- Packed int4 round-trips signed values including an odd tail
- Int8 and int4 matvec errors stay within fixture tolerances
- Int4 storage is smaller than int8 including scales
- Decoder comparison reports logits, loss, and token decision
- Benchmark reports warmed median samples and checksums


## 42. Efficient attention

Part: Efficient and adapted LLMs. Prerequisites: 41.

Project: Compare attention algorithms. Dataset: Deterministic attention tensors.

Lessons: Keep a simple full-attention oracle; Update softmax without seeing all scores first; Tile exact attention; Share keys and values across query heads; Run, diagnose, and transfer; Before you move on.

Introduced concepts: Online softmax; tiled attention; grouped-query attention; context costs.

Outcomes:

- Derive the online-softmax recurrence
- Implement exact causal attention over key tiles
- Verify tiled output against a full oracle
- Map grouped query heads to shared key/value heads
- Account for long-context arithmetic and intermediate storage

Completion checks:

- Tiled attention matches the full oracle within 2e-6 + 2e-6 times absolute reference across tile sizes
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
- Train query and document embeddings
- Search a dense index exactly
- Assemble source-labeled context and extract an answer
- Measure hit@1, attribution, coverage, and abstention

Completion checks:

- BM25 ranks an exact-term fact first
- Both query and document table gradients match direct-loss central differences
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
- Derive and check a DPO logit gradient
- Train a toy preference policy and a separate verifiable-reward policy
- Explain what the reference policy and beta control

Completion checks:

- DPO mean loss falls from ln 2 below 0.18
- Each chosen response has the highest learned logit for its prompt
- One DPO logit gradient matches a central difference within 1e-6
- Stable log-softmax retains normalization at large common offsets
- Reversing a shared-prompt pair batch preserves its update
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
- Prove scalar dense and CSR matrix-vector kernels agree within tolerance
- Measure kernel time and output error across several densities

Completion checks:

- CSR row pointers preserve an empty middle row
- CSR and dense outputs agree within 1e-10 absolute plus 1e-8 relative tolerance
- Fifty-percent pruning keeps the two largest magnitudes in the four-value fixture
- Default run reports kernel median and range plus output RMSE at four densities


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
- Mean uncapped task loss falls below one fifth of its initial value
- All three experts receive routes after training on the deterministic fixture
- Overflow count equals assignments beyond the computed per-expert capacity


## 49. State-space models and linear attention

Part: Advanced architectures & engineering. Prerequisites: 48.

Project: Compare sequence computation methods. Dataset: Synthetic sequences.

Lessons: Sequence state without a cache; Selective recurrence; Associative scan; Linear attention; Rust comparison; Practice and transfer.

Introduced concepts: Recurrence; scans; selective state; linear attention.

Outcomes:

- Compute a selective state-space sequence as a recurrence
- Derive associative affine-transition composition and verify scan parity
- Implement normalized causal linear attention with recurrent summaries
- Compare recurrence, scan schedule, and a direct prefix oracle without false speed claims

Completion checks:

- Associative scan matches sequential recurrence within absolute-plus-relative tolerance
- Affine composition passes an explicit associativity check
- Causal recurrent linear attention matches a direct prefix oracle
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
- Two-stage pipeline backward updates both stages and matches one serial update within 1e-12
- A restored two-minibatch run equals the uninterrupted local run
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
- Measure image-to-text recall at one on perturbed paired examples

Completion checks:

- Symmetric contrastive loss falls below one quarter of its initial value
- Training updates both the image and text parameter matrices
- Training-pair recall at one reaches 100 percent
- Perturbed variants of the three same base patterns reach 100 percent recall at one
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
- Nonfinite, out-of-range, and wrong-schema inputs are rejected before scoring
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
- Framework test executes real forward, loss, gradient, weight and bias update, record and restored-output assertions
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
- Capacity counts attempted and accepted routes and residual-only drops correctly
- Checkpoint next update resumes exactly and rejects malformed or changed training state
- Fragmented requests yield complete 200 or 400 responses with correct byte length
- Default 160-step run updates router, all experts, embeddings, attention and normalization
- Custom corpus inputs require an explicit held-out file and reject equal/shared 32-byte passages

