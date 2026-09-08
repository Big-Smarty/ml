# Chapter 50 research

Model route: GPT-5.6 Luna High. Sources were checked against the linked original paper or first-party PyTorch documentation on 2026-09-08.

## Claim-to-source notes

- **Data parallelism and all-reduce averaging.** PyTorch DDP initializes replicas from rank 0, computes each replica's gradients on its local input, and uses bucketed `allreduce` to calculate the mean gradient across processes. Every rank then applies its local optimizer to the same averaged gradients, so replicas remain synchronized when the optimizer and initial state match. The all-reduce calls must occur in the same order on every rank. For equally sized local batches whose gradients are local means, `g = (g_0 + g_1) / 2`. The course instead sends raw sums and counts, then divides the summed gradients by the summed counts; this remains correct for uneven worker batches. [PyTorch Distributed Data Parallel design note](https://docs.pytorch.org/docs/stable/notes/ddp.html)

- **Collective semantics for the local demo.** `all_reduce` applies an elementwise reduction and stores the result on every participating rank; `SUM` followed by division by world size is the documented gradient-averaging pattern. This supports using Rust channels to model a real synchronous collective while keeping the reduction convention visible in the reference implementation. [PyTorch distributed applications tutorial](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html)

- **Tensor/model parallel partitioning.** Megatron-LM's original model-parallel construction splits transformer GEMMs across GPUs: column-parallel first projections let nonlinearities run locally, row-parallel second projections require an all-reduce, and attention heads are partitioned across devices. The paper distinguishes this intra-layer (tensor) approach from pipeline model parallelism and shows that the two are complementary. Keep the lesson's matrix partitions small and name the required synchronization; a split is correct only if the local partial results are combined at the mathematically required boundary. [Shoeybi et al., *Megatron-LM*](https://arxiv.org/abs/1909.08053), especially §§3 and 5.1.

- **Pipeline stages and bubbles.** GPipe partitions a sequential network into `K` cells on `K` accelerators, divides a mini-batch of size `N` into `M` micro-batches, sends activations across neighboring boundaries, accumulates gradients, and applies one synchronous update at the end of the mini-batch. Its reported amortized idle/bubble overhead is `O((K−1)/(M+K−1))`; it found the overhead nearly negligible when `M ≥ 4K`, while `M = 1` leaves only one active device at a time. Use this to explain pipeline fill/drain and why more micro-batches reduce idle time, without claiming the local channel demo measures accelerator utilization. [Huang et al., *GPipe*](https://arxiv.org/abs/1811.06965), §§2.2–2.3 and 3.

- **Checkpoint and recovery equivalence.** PyTorch's general-checkpoint guidance says resuming training requires at least model state, optimizer state, and the progress marker; optimizer state contains buffers and parameters that affect future updates. PyTorch Distributed Checkpoint additionally supports saving/loading from multiple ranks and load-time resharding into a different cluster topology, but loading still requires the destination model's allocated state/sharding information. For chapter code, checkpoint the model parameters, optimizer buffers, step, and any data/RNG cursor that controls the next batch; compare uninterrupted and resumed runs from the same boundary. [PyTorch saving/loading tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) and [PyTorch Distributed Checkpoint documentation](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)

## Cautions for the author

- The local `std::thread`/channel implementation is real inter-worker communication and can verify message ordering, gradient aggregation, synchronization, and a multi-worker training step. It does **not** verify multi-GPU execution, accelerator kernels, NCCL/Gloo behavior, network bandwidth, or device memory placement.
- Floating-point addition is not associative. Different tree/ring/reduction orders, fused operations, or hardware can produce small differences, so checkpoint recovery and a single-worker reference should use tolerances; exact bitwise equivalence is not promised even when the mathematical update is equivalent.
- Averaging must match the loss convention and sample counts. Equal `g_r / R` is correct only when each worker contributes the same effective number of examples (or when each local gradient is already normalized consistently); uneven worker batches require a sample-weighted reduction.

## Source list

1. PyTorch, “Distributed Data Parallel” design note — https://docs.pytorch.org/docs/stable/notes/ddp.html
2. PyTorch, “Writing Distributed Applications with PyTorch” — https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html
3. Mohammad Shoeybi et al., “Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism” — https://arxiv.org/abs/1909.08053
4. Yanping Huang et al., “GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism” — https://arxiv.org/abs/1811.06965
5. PyTorch, “Saving and Loading Models” — https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
6. PyTorch, “Distributed Checkpoint” — https://docs.pytorch.org/docs/stable/distributed.checkpoint.html


## Astra High review and correction — 2026-09-08

The user-directed ownership is GPT-6 Astra High, reviewing the existing Sol High draft and implementing corrections. Bounded read-only mathematical/source verification was delegated to GPT-5.6 Luna High (`verify_math`), which browsed original Switch, S4, linear-attention, AEVB, GAN, and DDIM sources. The author integrated its evidence and independently inspected all chapter lecture, metadata, reference code, starters, and Rustlings exercise/solution files.

Verified real communicating data/tensor/pipeline workers and restored minibatch cursor. Added malformed/nonfinite/version/step-overflow cases and a finite-difference check reaching through the first pipeline stage. Clarified that recovery round-trips a string in one process and assumes unchanged data, batch size, and learning rate; it does not persist or relaunch.
