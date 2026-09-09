# Build a neural network: Chapters 07–11

One independent, std-only Rust package. Defaults are tiny, deterministic and offline. Every learner module runs before edits. The complete answers live separately in `src/solutions/`.

```bash
just lab 07
just lab-check 07
just solution 07 --check
# Repeat with 08, 09, 10 or 11; course aliases are just lab NN / just lab-check NN.
```

`--check` invokes the actual learner function. An unfinished algorithm returns exit 1 with `GOAL_NOT_MET:` and numerical evidence. Runtime, input, shape and I/O errors also return nonzero but retain their own messages. `just lab-test 07` protects the useful baselines, supplied boundaries and solution numerics; it does not assert that unfinished learning goals pass.

| Chapter | Working baseline | Implement in the learner module | Evidence and transfer |
|---|---|---|---|
| 07 | Fit output weights on frozen sigmoid features | Complete all nine mean backpropagation derivatives in `gradient` | Every parameter checked on XOR and signed soft-target rows; fit XOR from two specified initial states |
| 08 | Forward tape plus the root operation's immediate sensitivities | Full reverse sweep in `backward`; a tanh-neuron graph in `neuron_gradient` | Shared `x*x+x`, tanh chains, separate equal-valued leaves, unrelated branches, repeated backward calls, neuron training and changed-input slopes |
| 09 | Bias-only batch forward and its correct backward | Dense `forward` and `backward`, including all three gradient buffers | Hand-worked nonsquare arrays, all 18 derivatives in an unfamiliar rectangular case |
| 10 | Fit class biases without learning pixel weights | Complete mean class-by-pixel `gradient` and `per_class_recall` | Numerical slopes, label rotation, short final minibatch, loss and confusion/error reports |
| 11 | Frozen random ReLU features trained by SGD | Full MLP `gradient`; momentum and Adam in `update` | All 66 narrow-model derivatives, independent two-step optimizer arithmetic, real training, exact next-update checkpoint restoration |

Supply an unchanged model while computing its gradients. Clear accumulators intentionally. Loss and parameter gradients use means over examples; the dense kernel propagates the reduction already encoded in its incoming derivatives and never adds another average. Gradient checks use `f64` central differences with h=1e-5 and tolerance `1e-6 + 1e-4*abs(expected)`. ReLU checks avoid zero pre-activations.

Useful earlier checkpoints remain visible: Chapter 09's normal run reports forward values even before backward is complete; Chapter 11's goal reports derivative agreement, then momentum, then Adam. An expected later goal failure does not erase an earlier numerical result. The lessons also ask for explanations and experiments that a numerical check cannot grade.

## Controlled experiments

XOR supports explicit bounded rate, step-count and first-hidden-weight perturbations. Each invocation starts from the same initial model.

```bash
just solution 07 --steps 1000 --rate 0.1
just solution 07 --steps 1000 --rate 1
just solution 07 --shift 0.08
```

MLP supports optimizer selection and three initial states. `scaled` uses seed 7 and uniform width-scaled weights, `zero` zeros every parameter, and `large` multiplies initial weights by 20. Parameters are reset between invocations. The learner update initially uses SGD regardless of the requested optimizer; completing `update` adds the named stateful behavior.

```bash
just solution 11 --optimizer momentum --init scaled
just solution 11 --optimizer adam --init zero
just solution 11 --init large
```

Other lesson experiments name exact editable regions: Chapter 08’s `neuron_gradient` is invoked by normal reporting, twenty-step training and goal checks; `08 --input -0.7` changes its input. Chapter 09’s supplied `report` duplicates rows and scales upstream gradients through the learner kernel; `09 --scale 2` doubles input values. Chapter 10’s `per_class_recall` receives actual returned confusion counts, and `10 --brightness 0.5` scales only held-out inputs. Chapter 11’s supplied `report` prints first-layer gradient diagnostics; the named `resume_check` is invoked by every normal run, so the lesson’s deliberate moment-reset experiment is executable.

## Data and MNIST extension

`src/data.rs` holds ten course-authored 5×3 binary block glyphs. Training uses three brightness variants per class, 30 rows total. Held-out data contains one new brightness variant per class plus two identical blank images labelled 1 and 7. Those two targets deliberately conflict: a deterministic image-only classifier cannot get both right. These tiny synthetic examples are not MNIST and do not establish handwriting quality. Both fixture and file mode use the same checked big-endian IDX parser; bytes are divided by 255.

An optional explicit extension accepts uncompressed official-format IDX files. It does not download. The course preparation command is the separate, network-enabled choice:

```bash
just mnist --download
just solution 10 \
  --mnist datasets/downloads/mnist/fit-images-idx3-ubyte \
  datasets/downloads/mnist/fit-labels-idx1-ubyte \
  datasets/downloads/mnist/validation-images-idx3-ubyte \
  datasets/downloads/mnist/validation-labels-idx1-ubyte 1
```

For Chapter 11, use `11 --solution --optimizer adam --init scaled` before the same `--mnist` arguments. Epochs are explicit and bounded to 1..=100 in file mode. Keep validation within the original training set and reserve official test data for final evaluation. This redesign did not download or measure MNIST.

## Supplied support and checkpoints

`xor.rs` supplies fixed network state/cache/update plumbing. `tape.rs` supplies graph storage and valid reverse order. `tensor.rs` supplies contiguous arrays and shape validation. `data.rs` supplies IDX and fixtures. `linear.rs` supplies stable scoring and error reporting. `mlp.rs` supplies model storage, initialization, cache, and versioned checkpoint I/O. Arrays are row-major, inputs `[batch,in]`, weights `[out,in]`.

Chapter 11's normal run saves and restores inside a new temporary directory, compares the exact next update using the same seven rows, and removes only that directory. Both optimizers' full state is included. The same data/order is required; there is no shuffle or schedule state. Saving uses a new sibling `.tmp`, file sync and rename, but does not sync the containing directory. The preserved `projects/ch11` reference has the persistent-checkpoint CLI and zero-epoch evaluation mode; this lab's default demonstration remains temporary.

## Verify

```bash
just fmt-check 07
just lint 07
just lab-test 07
node labs/s02-neural-networks/check_interactives.cjs
```

The Node check uses only the built-in assertion library and verifies the three chapter-local tools' pure calculations. Browser tables, native controls, labels and deterministic resets complement the static worked examples. The tools neither execute Rust nor record mastery. See `guidance/redesign/section-02.md` for the authoring matrix and measured validation.
