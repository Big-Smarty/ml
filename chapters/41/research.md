# Chapter 41 research notes

Route: Sol High author with bounded GPT-5.6 Luna High primary-source research. Verified 2026-09-08.

- Jacob et al., [integer-arithmetic-only inference](https://ar5iv.labs.arxiv.org/html/1712.05877): affine relation `real = scale(code-zero_point)`, int32 accumulation, per-channel range problems, and hardware-dependent latency.
- [LiteRT int8 specification](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec): signed int8 weights use zero point zero and may use per-axis scales. Backend results need not be bit exact.
- [TorchAO overview](https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html): two int4 values can occupy one byte; storage includes quantization metadata and may use specialized layouts.
- [llama.cpp Q4 definitions](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h): concrete blocks combine packed nibbles with scales, demonstrating that effective bits per weight exceed four.
- Frantar et al., [GPTQ](https://ar5iv.labs.arxiv.org/html/2210.17323): data-aware post-training weight quantization and accelerator-specific measurements. No speedup is copied into the lesson.

The course kernel uses original symmetric signed codes −7..7 for int4, rather than copying llama.cpp's Q4_0 encoding. Its scalar reference and quantized kernel both use row-major weights [out_features,in_features]. The decoder-level evaluation preserves Chapter 36's actual `output_weight` span as [D,V]: it transposes to a [V,D] row view for per-vocabulary-output quantization, dequantizes, and transposes back to [D,V]. The optional benchmark separately measures the scalar packed kernel.

Independent review: GPT-6 Astra High owned the complete lesson/code/starter/exercise audit and corrections. Bounded GPT-5.6 Luna High primary-source and technical verification covered AdamW, checkpoint continuation, validation, caching, serving, and quantization claims. See `guidance/astra-review-29-41.md` for findings, executed gates, and inherited hardware evidence.
