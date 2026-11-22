# LongCat Flash Experts Validation

Standalone validation and benchmark scripts for the isolated `hyper_multicore.longcat.LongcatFlashExperts` module.

## Architecture

- Completely isolated from `MegaMoeExperts` high-level routing/unpermute logic.
- Forward performs a single one-shot `npu_moe_init_routing_v2` permute + count over all real `[0, num_experts)` and zero/identity `[num_experts, num_experts + zero_experts)` experts.
- Only the contiguous real-expert prefix `all_permuted[:R]` enters the fused `HyperMegaMoe` kernel; all-zero-expert steps skip the kernel completely.
- Output stitches `[real_y, all_permuted[R:]]` and executes a single fused `npu_moe_token_unpermute(..., probs=router_weights.float())` covering both real and zero expert FP32 weighting and gradients.
- Supports both `--swiglu-precision fused` and `--swiglu-precision hf` (HF-compatible BF16 intermediate rounding).
- `native_reference.py` mirrors the same one-shot permute + fused unpermute semantics for exact precision alignment.

## Usage

```bash
# Full-width EP4 accuracy alignment (fused SwiGLU + edge routes: normal/hot/empty_source/identity)
torchrun --standalone --nproc_per_node=4 examples/longcat_flash/validate.py \
  --check accuracy --backend push --swiglu-precision fused \
  --ep 4 --tokens 2048 --hidden 6144 --intermediate 2048 --real 32 --zero 16 --top-k 12 \
  --updates 4 --exercise-edge-routes --output /tmp/longcat_acc_fused

# Full-width EP4 accuracy alignment (HF BF16 SwiGLU + edge routes)
torchrun --standalone --nproc_per_node=4 examples/longcat_flash/validate.py \
  --check accuracy --backend push --swiglu-precision hf \
  --ep 4 --tokens 2048 --hidden 6144 --intermediate 2048 --real 32 --zero 16 --top-k 12 \
  --updates 4 --exercise-edge-routes --output /tmp/longcat_acc_hf

# Benchmark push vs native
torchrun --standalone --nproc_per_node=4 examples/longcat_flash/validate.py \
  --check benchmark --backend push --swiglu-precision fused \
  --ep 4 --tokens 2048 --hidden 6144 --intermediate 2048 --real 32 --zero 16 --top-k 12 \
  --warmup 10 --samples 20 --output /tmp/longcat_bench_push
```
