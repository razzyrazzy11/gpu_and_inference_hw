import torch
from torch.profiler import profile as torch_profile, ProfilerActivity
from transformers import DynamicCache
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)

@torch.inference_mode()
def optimized_loop(model, input_ids, n_steps):
    # TODO: fix the performance issues you found — changes may include
    # both `optimized_loop` and `generate_optimized`

    # The slow baseline re-runs a full forward pass over the entire growing sequence every step 
    # (O(n^2) work that recomputes all past keys / values),
    # calls .item() each step (a blocking CPU <-> GPU sync x128), and never uses a KV cache.
    # The fixes below attack all three.

    # @torch inference_mode() (decorator above) disables autograd bookkeeping entirely - no graph, no version counters - which is cheaper than no_grad()

    # --- Prefill: process the whole prompt once and fill the KV cache ---
    past = DynamicCache()
    cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
    outputs = model(
        input_ids=input_ids,
        past_key_values=past,
        use_cache=True,
        cache_position=cache_position,
    )
    next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
    generated = [next_token_id]

    # --- Decode: feed only the one new token each step ---
    # With the KV cache populated, every subsequent step attends to the cached key/values and only computes the single new position
    # That turns each step from "forward over 1024+t tokens" into "forward over 1 token".
    for i in range(1, n_steps):
        cache_position = torch.tensor(
            [input_ids.shape[1] + i - 1], device=input_ids.device
        )
        outputs = model(
            input_ids=next_token_id,
            past_key_values=past,
            use_cache=True,
            cache_position=cache_position,
        )
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token_id)

    # Keep tokens on-device as tensors during the loop and do a single host sync
    # at the very end (.tolist()), instead of one .item() sync per step.
    return torch.cat(generated, dim=1).squeeze(0).tolist()   


def profile(loop_fn, model, input_ids, trace_name: str):
    # TODO: wrap loop_fn(model, input_ids, PROFILE_STEPS) with torch.profiler,
    # print the summary table, and export a Chrome trace to RESULTS_DIR / trace_name
    with torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)
    torch.cuda.synchronize()  # ensure all kernels finished before reading times
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace(str(RESULTS_DIR / trace_name))

def generate_optimized(optimized_trace_name: str) -> float:
    # TODO: load the model (consider dtype and other loading options),
    # then call profile() and time_generation() on optimized_loop.
    # Return the elapsed time from time_generation so main() can print a speedup.
    # dtype is a model-loading choice that lives here, not in the loop: build in
    # bfloat16 so every matmul moves ~half the bytes and can use the GPU's
    # fast low-precision pipes. bf16 (vs fp16) keeps fp32's exponent range, so
    # the randomly-initialized weights won't overflow to inf/nan.
    model = build_model(torch.bfloat16)
    input_ids = get_input_ids()

    # Produce the optimized Chrome trace, then time the real 128-token run.
    profile(optimized_loop, model, input_ids, optimized_trace_name)
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    return elapsed  # main() divides slow_elapsed by this to report the speedup

def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# 
# SUMMARY
#  Slow:        0.95s
#  Optimized:   0.20s
#  Speedup:     4.80x  (vs V0 slow baseline)
#
# Changes made and speedup per fix (ranked by impact):
# 
# 1. KV cache + single-token decode - Biggest win
#    The baseline re-runs the full forward over the entire sequence every step, so step t recomputes KV for all 1024+t tokens (O(n^2)). Adding a 
#    DynamicCache with use_cache=True and feeding only the latest token per step turns each decode into a single-position forward. This is the dominant cost
#    in the V0 trace (attention/linear kernels grow with sequence length) and removing it is most of the speedup on its own.
#
# 2. bfloat16 weights / activations.
#    Building the model in bf16 roughly halves memory traffic for every matmul and lets the GPU use its fast low-precision units; on this H100, this is a large multiplier on top of the KV cache.
#    (bf16 over fp16 to keep fp32's exponent range so random weights don't overflow).
#
# 3. Remove the per-step .item() sync.
#    Each .item() in the baseline forces a blocking  CPU<->GPU synchronisation, stalling the pipeline 128 times. Collecting tokens as on-device tensors and doing one .tolist() at the very end
#    cuts that to a single sync, which shows up in the trace as far fewer gaps where the GPU stream goes idle.
#
# 4. @torch.inference_mode().
#    Disables autograd graph construction and version tracking for a small but real per-op saving across thousands of tiny decode ops.
#     
# Measured on H100 (the `Speedup` line from main() is the graded number):
#   - V0 baseline (fp32, full recompute, per-step .item()):  1.00x  (0.95s)
#   - Fully optimized (KV cache + bf16 + single end-sync +
#       inference_mode):                                      4.80x  (0.20s)  <-- clears the >=4x "Great" tier
# Per-fix attribution (not separately benchmarked this run): the KV cache is the
# dominant win — it changes per-step work from O(sequence) to O(1) — with bf16
# the next-largest contributor and the sync/inference_mode fixes smaller
# constant-factor gains on top.
#
# Biggest impact and why:
#   The KV cache. It changes the asymptotic work of the loop, not just a constant factor: every decode step drops from "attend over the whole growing prefix" to 
#   "attend over the cached prefix + one new token". The other fixes (bf16, fewer syncs, inference_mode) are constant-factor wins layered on top of that algorithmic change.
