# Section 06 — Write GPU kernels

Implemented by the single GPT-6 Astra High section owner. Scope: chapters 29–32,
`labs/s06-gpu`, and this report. Existing projects29–32 remain unchanged;
chapter 30's pinned library is reused as the scalar matrix/reduction oracle.
Research input: the supplied systems20–39 audit, interactive-tool brief and
learning-science synthesis. No additional author/research delegation occurred.

## Four-chapter design matrix

| Chapter / original topics | Observable objective and explanation map | Meaningful learner work | Evidence, transfer and sessions |
|---|---|---|---|
| **29 GPU fundamentals:** Adapter discovery; wgpu; WGSL; buffers; dispatch; readback | `question` starts with five known sums; `buffers` translates f32 elements/bytes/bindings; `dispatch` draws padded ownership; `vector-build` derives0.5×3.5−0.5=1.25; `adapter` distinguishes hardware from software; `readback` traces map/poll; `hardware-build` runs the actual route; `review` retrieves it | Replace serial64-value chunks with bounded64-lane WGSL SAXPY, and implement the same CPU operation. The host supplies setup/buffers/command recording. Whole entry signature, loop ownership and arithmetic change; this is more than filling a dispatch-count expression | Known five outputs; N=1,64,67,129, scales0.5/−2; actual dispatch/readback parity. Change WGSL workgroup to 32 and `host::VECTOR_WORKGROUP_SIZE` together, predict three groups for 67. **2 sessions:40+40min** |
| **30 Reductions and tiled matrix multiplication:** Workgroups; shared memory; barriers; edge dimensions | `question` starts from 31 and [58,64,139,154]; `reduction` works8→4→2→1; `reduce-build` extends to 600→2→1; `matmul` maps a lane to C; `edges` traces A[2,16]=offset50, B[16,3]=83, C[2,3]=13; `tile-build` challenges multi-group edges; `review` explains ordering | Replace serial subtotals with 256-lane shared reduction, direct dots with 16×16 cooperative tiles and two uniform barriers. Implement CPU stage traces as the session's no-hardware practice, not a GPU simulator | Numeric parity on3×17×5 and 17×19×33; sums atN=1,600,777. Learner checks return **GOAL_REVIEW_REQUIRED: exit3** after parity, with `lab.manual_checks` for actual cooperative source/ownership/barrier traces. Solution auto checks exit0; author source review recorded below. Transfer: odd identity B and cancellation. **2 sessions:40+44min** |
| **31 Train on the GPU:** Forward and backward kernels; updates; residency | `question` introduces actual2–4–1 model/layout; `math` computes one hidden derivative before notation; `checks` uses independent f64 differences; `cpu-build` learns all weights; `kernels` ports CPU chain rule; `residency-audit` labels old versions and lifetimes; `gpu-build` runs capstone; `review` states limits | Extend a working fixed-feature readout into full17-parameter CPU and WGSL backprop. Keep mean reduction and simultaneous updates. GPU allocation, layout and ordered passes remain supplied | All17 gradient probes; ≥70% BCE reduction; four unseen sign points; real hardware comparisons after 1/80/800 steps and 3 steps on a3-row batch. Transfer: duplicated-batch denominator and private [2,17] partial-gradient design. **2 sessions:42+43min** |
| **32 GPU performance:** Asynchronous execution; timing; fusion; mixed precision; Vulkan extension | `question` works [-1,0,2]→[0,.25,3.25]; `timing` separates three boundaries; `measurement-design` discusses warmups/noise; `async` explains completion; `experiment-build` implements schedule; `fusion` counts4N→2N; `precision` covers feature/range/Vulkan distinctions; `fused-build` measures; `review` separates capability/correctness/speed | Replace one separate sample with 3 warmups+7 alternating measured samples per plan; implement the complete fused WGSL entry before scheduling it. Supplied half/timestamp plumbing prevents incidental API work from becoming the exercise | Actual shader output checks on65,537 values,257 non-binary values, ReLU boundary and 50,000 range guard; real clock distributions, no speed threshold. Transfer: `PERFORMANCE_ELEMENT_COUNT=257` and clipping at 6. **2 sessions:41+43min** |

Every session contains a worked example, prediction, working baseline,
implementation, evidence and an unfamiliar input/transfer explanation. Math
follows concrete arithmetic or an ownership/buffer diagram. Native details
contain progressive hints and worked reasoning. Existing meaningful anchors
(`question`, `continuity`, `run`, `review`, `sources`, topic anchors) remain as
steps or nested headings. Metadata topic keys exactly match course.json and
step order matches the lesson DOM.

## One host and truthful completion statuses

`host.rs` is the one persistent hardware device/queue implementation, adapted
from the checked reference programs. `src/chNN.rs` owns learner functions and
WGSL, `src/solutions/chNN.rs` explains completed versions. Baselines all run CPU
by default, and all four explicit hardware baselines were also executed
successfully. No original reference files were edited.

Final CLI results:

| Chapter | CPU baseline | Learner `--check` | `--solution --check` | Actual solution `--gpu --check` |
|---|---:|---|---:|---:|
|29|0|1, `GOAL_NOT_MET:` actual2 versus expected3.5 on an unfamiliar scaled vector|0|0|
|30|0|3, `GOAL_REVIEW_REQUIRED:` correct numeric baseline still requires cooperative source/trace review|0|0|
|31|0|1, `GOAL_NOT_MET:` gradient[0]=0 versus independent0.0518540|0|0|
|32|0|1, `GOAL_NOT_MET:` zero warmups/one sample versus required3/7 per plan|0|0|

Once a learner's CPU criteria in29/31/32 are complete, CPU-only `--check`
returns `GOAL_REVIEW_REQUIRED:` (exit3) with shader/source/hardware criteria in
`lab.manual_checks`. Hardware checks can then succeed only after the selected
learner WGSL actually dispatches and matches the CPU oracle. Chapter 30 retains
its manual structural review even after numerical hardware parity. Completed
solution CPU checks exit0 for their stated automatic scope, with author
source/hardware review recorded separately.

Only intentional goal comparisons receive `GOAL_NOT_MET:`. Invalid CLI,
invalid data, setup and missing-device errors remain ordinary errors. A
controlled run with `VK_DRIVER_FILES` pointing at an absent driver file exited1
with `no hardware Vulkan adapter found; software fallback is intentionally
disabled`; it was not converted into a goal failure or success.

**Chapter 30 solution source review:** every one of 256 lanes writes its own
shared A/B slot, initializes both slots to zero, guards storage loads with
explicit ifs, reaches the load barrier, consumes16 products, and reaches the
second barrier before overwrite. Only the final C store is output-bound
conditional. For17×19 by 19×33,6 workgroups cover a3×2 output grid and 2 K
rounds; the final inner round has 3 real values and 13 zeros. The reduction has
256 unique scratch writers, guarded paired loads, uniform strides
128,64,32,16,8,4,2,1 and lane0 subtotal ownership. Host pass boundaries order
600→2→1. These properties were read in the actual solution, not inferred from
substring tests or output equality. Hardware parity independently passed.

## Interactive: one tool, home29

`chapters/29/demo.js` supplies exactly one focused workgroup/residency tool,
linked from 30/31 and conceptually reused by 32. Four stages show dispatch,
shared reduction, one fixed-size tile, and parameter readback. Native selectors
change one factor at a time; Step advances the reduction; Reset deterministically
restores dispatch/N67/K17/800steps/level0. All selectors have labels, outputs
are live text, and the dispatch SVG has title/description and striped guarded
regions in addition to fill colors. Static complete examples remain in HTML
without JavaScript. It never runs Rust/WGSL, estimates elapsed time or changes
progress state.

`node chapters/29/demo-check.cjs` checks:

- N67→2groups/128lanes/61unused; last ID66 is group1/local2.
- N64→1group/64lanes/no unused lanes; N129→3groups/192lanes in the browser.
- [3,1,4,1,5,9,2,6]→[8,10,6,7]→[14,17]→[31].
- K17→2rounds/15padding positions; offsets A50, B83, C13. K16→1round.
- 800steps: upload164B, final parameters68B, history3204B; resident3436B;
  reading parameters each step57768B. One step makes the two counts equal240B.

Hidden-tab browser inspection exercised N129, all reduction levels, K16,
800→1 transfer steps and Reset using native controls. After the final rebuild,
N129 updated the SVG to three groups and its accessible description to 129 valid
and 63 guarded lanes; Reset restored two groups/67 valid/61 guarded. Screenshot
inspection confirmed the three bars and striped unused region. The temporary
tab was closed. No browser viewport
settings were changed. The final diagram is a count visualization, not a
performance claim.

## Executed numerical and hardware evidence

Run date:2026-09-09. Stable Rust 1.96; wgpu 29.0.4; bytemuck 1.25.0;
pollster 0.4.0; release hardware build. Selected **AMD Radeon RX 6950XT (RADV
NAVI 21), Vulkan, DiscreteGpu, radv Mesa 26.2.2-arch3.2**. A second integrated
adapter was discovered and reported. TIMESTAMP_QUERY and SHADER_F16 were both
supported and requested. No software adapter was accepted.

- **29:** all 8 combinations of lengths1/64/67/129 and scales0.5/−2 had max
  CPU/GPU error0. Five-value solution [2.5,4,1.25,5,.875].
- **30:**3×17 by 17×5 max error5.960e−8;17×19 by 19×33 max error3.576e−7.
  N1/600/777 staged sum errors0. Tolerances respectively
  1e−5+1e−4|expected| and 1e−4+1e−5|expected|.
- **31:** CPU BCE 0.730554→0.013002; GPU0.730555→0.013002 after 800 steps at
  rate.3. Max parameter errors:1step2.980e−8;80steps1.341e−7;
  3rows/3steps2.980e−8;800steps2.861e−6. All17 analytic gradients differed
  from independent f64 central differences by at most7.993e−9. Four unseen
  probabilities were.0588,.9941,.9901,.0295 for targets 0,1,1,0. The fixed-feature
  baseline still works but finishes at BCE.684010 and misses two of these
  decisions, providing a concrete reason to learn the hidden features.
- **32:** bounded65,537-value binary-fraction fixture max error0 for all plans;
  a distinct257-value `i/29−4` fixture showed actual half-rounding error4.580e−3
  while both f32 plans had error0. Mixed tolerance0.01+1e−6|expected| is limited
  to these bounded fixtures. The50,000 range case selected f32 and passed.

One recorded warmed run (3warmups,7alternating measured samples per plan):

| Plan | Encoding-through-output-map wall median (range) | Summed device-pass median (range), ns |
|---|---|---|
|Separate f32|219.184µs (213.924–600.694µs)|6600 (6280–7080)|
|Fused f32|201.405µs (199.334–292.012µs)|3600 (3480–4080)|
|Mixed f16 arithmetic|204.084µs (180.075–213.804µs)|3640 (3520–4240)|

The wider allocation/upload-through-timestamp-decoding latency is separately
printed per sample. Device pass timestamps exclude host allocation/I/O and
inter-pass gaps; the narrower wall timer excludes initial bulk allocation but
includes bind-group creation, encode/submit, output copy, polling and output
map. The timing distributions overlap and are one host/run, not a universal
speedup, application-throughput claim, or precise prediction from 4N→2N.

## Validation and prerequisite audit

Executed and passed:

```text
cargo fmt --manifest-path labs/s06-gpu/Cargo.toml --check
cargo clippy --manifest-path labs/s06-gpu/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path labs/s06-gpu/Cargo.toml --all-targets
cargo test --release --manifest-path labs/s06-gpu/Cargo.toml -- --ignored --nocapture
node chapters/29/demo-check.cjs
```

Ordinary suite:2 passed,1 explicitly ignored hardware test. Explicit hardware
suite:1 passed, encompassing all four solutions, actual reduction/GEMM,
resident training, fusion, feature/range checks and timing. All4 CPU baselines,
all 4 solution CPU checks, expected learner statuses, and all 4 hardware baselines
were executed. Chapter metadata, local source excerpts and source paths passed all four
`chapter_audit` calls with no errors. A temporary copied-package probe replaced
learner29/31/32 with completed CPU implementations and verified that CPU-only
checks still exit3 requiring source/hardware review; the temporary copy was
removed. No temporary generators, logs or old lesson copies live in the lab.

Prerequisites remain29←28,30←29,31←30,32←31. Chapter 29 briefly retrieves lane
coverage and tolerance;30 retrieves row-major dot products;31 retrieves XOR,
mean BCE, chain rule and simultaneous updates;32 retrieves asynchronous queue
work and device residency. The lessons define invocation/group/tile/buffer
before formal use. Prior expert Rust knowledge is used for slices, loops,
Result and constants; prior GPU API expertise is not assumed. Chapters36/38/39
retain their preserved chapter 30 reference interface; no downstream migration
is required.

Limits remain explicit: hardware Vulkan only; fixed educational tile size;
serial training backward; tiny synthetic generalization fixture; host-recorded
steps; f32 storage in the mixed path; CPU evidence does not prove device
execution; parity does not prove tiling; no general simulator or browser timing.
