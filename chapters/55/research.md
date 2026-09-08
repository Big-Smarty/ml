# Chapter 55 research

Route: GPT-5.6 Luna High, bounded primary-source verification for framework interoperability. Verified 2026-09-08. Chapter metadata in `course.json` names framework concepts, runtime interchange, parity checks, and optional integration; `projects/ch55/framework` pins Burn `=0.21.0`.

## Verified project facts

- `cargo test --manifest-path projects/ch55/Cargo.toml`: 2 tests passed.
- `cargo test --manifest-path projects/ch55/framework/Cargo.toml`: 1 test passed.
- `cargo run --manifest-path projects/ch55/framework/Cargo.toml` completed the Burn CPU forward, loss, gradients, direct SGD update, and Named MessagePack bytes round trip.
- The fixture computes `output = [1.15, 0.65]`, mean squared error `0.6725`, bias gradient `[0.15, 1.15]`, and Burn-layout weight gradient `[[0.225, 1.725], [-0.3, -2.3]]` up to f32 rounding. The updated Burn-layout weights are `[[0.1775, 0.5275], [-0.37, 0.33]]`.
- The scratch oracle stores weights as `[out, in]`; its forward loop computes `W_out,in x`. The Burn parity program stores the transpose `[in, out]` so `[batch, in] matmul [in, out]` produces `[batch, out]`. This is a declared interchange mapping, not a claim that all frameworks use the same parameter layout.

## Claim-to-source notes

### Burn tensors, shapes, and backend types

- Burn 0.21 exposes `Tensor<B, D, K>` with a backend type, compile-time rank, and tensor kind. Its `Shape::matmul` treats the last two dimensions as matrices and leading dimensions as broadcast dimensions. The API documents `matmul` as `C = AB`; the chapter’s `[1,2] x [2,2] -> [1,2]` fixture follows that contract. [Burn Tensor API, 0.21.0](https://burn.dev/docs/burn/prelude/struct.Tensor.html) and [Burn Shape API, 0.21.0](https://burn.dev/docs/burn/prelude/struct.Shape.html)
- Do not teach the transpose as a universal Burn or framework rule. PyTorch `nn.functional.linear` documents `weight` as `[out_features, in_features]` and computes `xA^T + b`, while JAX `matmul` takes `x1` with trailing `N` and `x2` with leading matrix dimension `N`; copying a state between implementations requires an explicit layout map. [PyTorch linear](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.linear.html), [JAX matmul](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.matmul.html)

### Autodiff semantics

- Burn enables autodiff by decorating a backend: `Autodiff<MyBackend>`. `backward()` returns a gradient container; gradients are retrieved with `tensor.grad(&gradients)`, and `inner()` removes autodiff information. Burn explicitly contrasts this with PyTorch, where `backward()` populates/accumulates leaf `.grad` fields. [Burn autodiff](https://burn.dev/books/burn/building-blocks/autodiff.html)
- The project’s `type Cpu = Autodiff<NdArray<f32>>`, `require_grad()`, `loss.backward()`, and `weights.grad(&gradients)` therefore use the documented API. The direct tensor subtraction is an intentionally transparent SGD step; it does not demonstrate Burn optimizer state or a full `Module` training loop.
- JAX presents a different model: `jax.grad` transforms a scalar-valued pure function, and transformations trace JAX operations. Side effects and data-dependent Python control flow are not interchangeable with array operations under transformations. [JAX transformations](https://docs.jax.dev/en/latest/101/transformations.html)

### Serialization and runtime interchange

- Burn’s record system is backend independent within Burn. The 0.21 API defines `NamedMpkBytesRecorder<S>` as an in-memory Named MessagePack recorder with `Vec<u8>` output/input; `FullPrecisionSettings` means f32 floating point and i32 integer serialization. The project correctly records the inner `NdArray<f32>` weights-and-bias tensor pair and loads both with the same recorder and a target device. [NamedMpkBytesRecorder, Burn 0.21.0](https://burn.dev/docs/burn/record/struct.NamedMpkBytesRecorder.html), [Burn record guide](https://burn.dev/books/burn/building-blocks/record.html)
- “Interchange” must be scoped: this bytes record is Burn’s format and is not a PyTorch `.pt`, JAX checkpoint, ONNX model, or generic wire format. Cross-framework interchange needs a separately specified format, tensor names, shape/layout map, dtype policy, preprocessing, and numerical tolerance. The current example serializes updated weights and bias, not a model architecture, optimizer state, RNG state, or training cursor.
- Burn’s record guide says precision conversion is supported and that the same recorder must be used for load/save. It also advises compressed formats for storage and warns that the binary format may not be backward compatible; do not turn this small in-memory demonstration into a general checkpoint recommendation.
- PyTorch’s official tutorial recommends saving a `state_dict`, recreating the model, then loading it; its device mapping and `eval()` requirements are part of the runtime contract. This supports teaching that parameter values alone do not define a portable model. [PyTorch saving/loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

### Burn 0.21 backend caveat

- Burn’s 0.21 release says `burn-ndarray` remains available for a transition window but is on a deprecation path; new development should target `burn-flex`. Keep the chapter project’s `ndarray` feature as a small, deterministic optional parity fixture, and label it as such rather than presenting it as the preferred new CPU backend. [Burn 0.21 release](https://burn.dev/blog/release-0.21.0/)

## Teaching pitfalls

- A matching scalar output is insufficient parity evidence. Compare shapes, dtype, parameter names/order, bias broadcasting, reduction convention, preprocessing, and representative gradients; use tolerances because f32 and backend kernels can differ.
- Mean squared error matters: with two outputs, `mean((output-target)^2)` gives `dL/doutput = output-target`; a sum reduction or a different batch denominator changes every gradient.
- Burn’s `Tensor<Cpu, 2>` rank and backend type are compile-time choices, but runtime tensor dimensions still need checks at framework boundaries.
- Do not claim Burn records can be loaded by PyTorch/JAX merely because Burn records are backend independent. Backend independence is within Burn’s record/type system.
- Do not imply autodiff gradients automatically update parameters. `backward()` computes gradients; the example applies explicit SGD and does not include optimizer policy, momentum, or checkpointed optimizer state.
- Do not compare Burn’s dynamic graph semantics and JAX’s traced pure-function transformations as if they were the same abstraction. They can compute equivalent derivatives on the fixture while imposing different constraints on program structure.

## Sources

1. [Burn Tensor API, 0.21.0](https://burn.dev/docs/burn/prelude/struct.Tensor.html)
2. [Burn Shape API, 0.21.0](https://burn.dev/docs/burn/prelude/struct.Shape.html)
3. [Burn Autodiff guide](https://burn.dev/books/burn/building-blocks/autodiff.html)
4. [Burn NamedMpkBytesRecorder, 0.21.0](https://burn.dev/docs/burn/record/struct.NamedMpkBytesRecorder.html)
5. [Burn Record guide](https://burn.dev/books/burn/building-blocks/record.html)
6. [Burn 0.21.0 release](https://burn.dev/blog/release-0.21.0/)
7. [PyTorch functional linear](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.linear.html)
8. [PyTorch saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
9. [JAX matmul](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.matmul.html)
10. [JAX grad/vmap transformations](https://docs.jax.dev/en/latest/101/transformations.html)

## Consistency integration, 2026-09-08

The scratch API keeps Chapter 9's `Dense::forward` and `[out,in]` `weights`, uses a distinct `Gradient`, and follows Chapter 36's `loss_and_gradient` plus `apply_sgd` split. The optional test runs real parity assertions, updates and records both weights and bias, and checks restored inference. The standard-library executable exports the actual updated affine artifact, and the Burn import CLI validates it and compares three input probes. All optional computation is isolated in `projects/ch55/framework` with Burn 0.21.0. The ONNX concepts page supports the narrow graph-format explanation; no ONNX exporter is implemented.
