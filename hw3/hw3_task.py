"""
HW3: Mini Inference Engine
CacheManager · Continuous Batching · Prefix Caching

Edit only this file.  See README.md for background and implementation details.

Run:
    python hw3_inference_engine/hw3_task.py
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tqdm import tqdm

from engine_utils import (
    CacheHandle,
    Request,
    Batch,
    BatchPhase,
    StepMetrics,
    DummyLLM,
    SchedulingPolicy,
    RequestStatus,
    generate_workload,
    compute_stats,
    print_stats,
    plot_results,
    plot_policy_results,
    BLOCK_SIZE,
    NUM_BLOCKS,
    MAX_SEQS,
    TOKEN_BUDGET,
    PREFILL_CHUNK,
)


# ── Task 1: Cache Manager ─────────────────────────────────────────────────────


class CacheManager:
    """
    Unified block allocator, prefix cache, and LRU eviction.

    Ref-count semantics:
        allocate(n)    ref = 1   request owns the block
        lock(handle)   ref += 1  request also pins a cached block
        unlock(handle) ref -= 1  block is evictable once ref drops to 1
        free(ids)      ref -= 1  block goes to free pool when ref reaches 0
        _evict_blocks_from_kv_cache(n)      reclaims n LRU unlocked blocks from the prefix cache
    """

    def __init__(
        self, num_blocks: int = NUM_BLOCKS, block_size: int = BLOCK_SIZE
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks))  # available block IDs
        self._ref: list[int] = [0] * num_blocks  # reference counts
        # Prefix cache: token-tuple key → list of block IDs
        self._cache: dict[tuple[int, ...], list[int]] = {}
        # LRU order: index 0 = least-recently used; updated on every hit and insert
        self._lru: list[tuple[int, ...]] = []
        # Per-block count of how many cache entries reference it.
        # _ref is incremented only ONCE for cache ownership (when _cache_ref
        # goes from 0 → 1) and decremented when _cache_ref returns to 0.
        self._cache_ref: list[int] = [0] * num_blocks

    @property
    def num_free_blocks(self) -> int:
        # TODO
        # Blocks with ref == 0 live in the free pool.
        return len(self._free)

    @property
    def ref_counts(self) -> list[int]:
        """Snapshot of per-block effective ownership refs."""
        return list(self._ref)

    @property
    def cache_ref_counts(self) -> list[int]:
        """Snapshot of per-block cache-entry reference counts."""
        return list(self._cache_ref)

    @property
    def cache_entries(self) -> dict[tuple[int, ...], list[int]]:
        """Snapshot of cached prefix -> block mapping."""
        return {k: list(v) for k, v in self._cache.items()}

    @property
    def lru_keys(self) -> list[tuple[int, ...]]:
        """Snapshot of cache keys in LRU order (oldest first)."""
        return list(self._lru)

    def allocate(self, n: int) -> list[int] | None:
        """Claim n blocks (ref=1 each). Evicts LRU cache entries if needed.
        Returns None only when eviction cannot free enough blocks."""
        # TODO
        if n<= 0:
            return []
        # Not enough free blocks? Try to reclaim the shortfall from the LRU
        # cache. allocate() never touches blocks pinned by a live request
        if len(self._free) < n:
            self._evict_blocks_from_kv_cache(n - len(self._free))
        if len(self._free) < n:
            return None
        claimed = self._free[:n]
        self._free = self._free[n:]
        for b in claimed:
            # A free block has ref == 0; claiming it gives one request-owner.
            self._ref[b] = 1
        return claimed

    def free(self, block_ids: list[int]) -> None:
        """Decrement each block's ref; return to the free list when ref reaches 0."""
        # TODO
        # free() releases ONE ownership unit (a live request's claim). It must
        # not touch _cache_ref: cache ownership is released only by eviction.
        for b in block_ids:
            if self._ref[b] > 0:
                self._ref[b] -= 1
                if self._ref[b] == 0:
                    self._free.append(b)

    def lock(self, handle: CacheHandle) -> None:
        """Pin the matched blocks (incr ref). Must be called before using them."""
        # TODO
        # A live request that reuses cached blocks adds a second ownership unit,
        # pushing ref to >= 2 so eviction will skip these blocks. _cache_ref is
        # untouched: locking does not create a new cache entry.
        for b in handle.matched_blocks:
            self._ref[b] += 1


    def unlock(self, handle: CacheHandle) -> None:
        """Release the pin (decr ref). Blocks become evictable when ref drops to 1."""
        # TODO
        # Mirror of lock(): drop the live-request unit. Cache ownership keeps
        # ref >= 1, so a cached-but-unlocked block never returns to the free
        # pool here — only eviction can do that.
        for b in handle.matched_blocks:
            if self._ref[b] > 0:
                self._ref[b] -= 1

    def match_prefix(self, tokens: list[int]) -> CacheHandle:
        """Longest-prefix lookup. Returns a CacheHandle WITHOUT pinning.
        Updates LRU order on a hit. Returns CacheHandle(0, []) on a miss."""
        # TODO
        # Only complete blocks are cacheable, so candidate prefix lengths are
        # multiples of block_size. Probe longest-first and return the first hit:
        # a shorter sub-prefix may have been evicted independently while a longer
        # one survived, so we must not stop at the first miss.
        max_blocks = len(tokens) // self.block_size
        for k in range(max_blocks, 0, -1):
            key = tuple(tokens[: k * self.block_size])
            if key in self._cache:
                # Refresh ONLY the matched key in the LRU (move to most-recent).
                # Do not touch its sub-prefix keys — they keep their LRU rank.
                self._lru.remove(key)
                self._lru.append(key)
                return CacheHandle(
                    num_matched_tokens=k * self.block_size,
                    matched_blocks=list(self._cache[key]),  # copy: never alias
                )
        return CacheHandle(num_matched_tokens=0, matched_blocks=[])

    def insert_prefix(self, tokens: list[int], block_ids: list[int]) -> None:
        """Store every complete-block prefix not already cached.
        For each block in a new entry, increment _cache_ref. Only increment
        _ref when _cache_ref goes from 0 → 1 (first cache entry for that block)
        so that overlapping entries share a single ref-count for cache ownership."""
        # TODO
        # One cache key per complete-block prefix: with block_size 16, a 40-token
        # prompt caches the length-16 and length-32 prefixes (not 40). Keys are
        # added shortest-first so the LRU reflects insertion order.
        num_complete = len(tokens) // self.block_size
        for k in range(1, num_complete + 1):
            key = tuple(tokens[: k * self.block_size])
            if key in self._cache:
                # Idempotent: re-inserting an existing prefix must not duplicate
                # the LRU entry or inflate any ref counts.
                continue
            blocks = list(block_ids[:k])  # copy so the request can't mutate us
            self._cache[key] = blocks
            self._lru.append(key)
            for b in blocks:
                self._cache_ref[b] += 1
                # The cache holds exactly ONE ownership unit per block, taken
                # when the first cache entry references it. Overlapping entries
                # only bump _cache_ref, not _ref.
                if self._cache_ref[b] == 1:
                    self._ref[b] += 1

    def _evict_blocks_from_kv_cache(self, n: int) -> None:
        """Attempt to evict least-recently-used cache entries whose blocks are
        unlocked (`ref == 1`) to reclaim up to `n` blocks.
        Because cache entries can overlap on blocks, evicting an entry does not
        always free a block immediately. A block becomes free only when its
        cache ownership drops to zero."""
        # TODO
        freed = 0
        # Walk the LRU oldest-first. Snapshot the keys because we mutate _lru.
        for key in list(self._lru):
            if freed >= n:
                break  # reclaimed enough; stop (do not over-evict)
            blocks = self._cache.get(key)
            if blocks is None:
                continue
            # ref >= 2 means a live request has this block locked → not evictable.
            if any(self._ref[b] >= 2 for b in blocks):
                continue
            # Drop the cache entry itself.
            self._lru.remove(key)
            del self._cache[key]
            for b in blocks:
                self._cache_ref[b] -= 1
                # Release the single cache-ownership unit only when the LAST
                # cache entry for this block goes away. A block shared by a
                # longer/shorter overlapping key stays cache-owned (frees 0).
                if self._cache_ref[b] == 0:
                    self._ref[b] -= 1
                    if self._ref[b] == 0:
                        self._free.append(b)
                        freed += 1


# ── Task 2: Scheduler ─────────────────────────────────────────────────────────


class Scheduler:
    def __init__(
        self,
        cache_manager: CacheManager,
        block_size: int = BLOCK_SIZE,
        max_seqs: int = MAX_SEQS,
        token_budget: int = TOKEN_BUDGET,
        prefill_chunk: int = PREFILL_CHUNK,
        enable_prefix_caching: bool = True,
        scheduling_policy: SchedulingPolicy | str = SchedulingPolicy.PREFILL_FIRST,
    ) -> None:
        self.cache_manager = cache_manager
        self.block_size = block_size
        self.max_seqs = max_seqs
        self.token_budget = token_budget
        self.prefill_chunk = prefill_chunk
        self.enable_prefix_caching = enable_prefix_caching
        self.scheduling_policy = SchedulingPolicy(scheduling_policy)
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.step: int = 0

    def add(self, req: Request) -> None:
        req.status = RequestStatus.WAITING
        self.waiting.append(req)

    def _blocks_for(self, n_tokens: int) -> int:
        return (n_tokens + self.block_size - 1) // self.block_size

    def _preempt(self, req: Request, batch: Batch) -> None:
        """Free req's blocks (respecting lock state), reset its state, re-queue it."""
        if req.cache_handle is not None:
            n = len(req.cache_handle.matched_blocks)
            self.cache_manager.unlock(req.cache_handle)
            self.cache_manager.free(req.block_table[n:])
            req.cache_handle = None
        else:
            self.cache_manager.free(req.block_table)
        req.block_table = []
        req.num_computed_tokens = 0
        req.num_generated_tokens = 0
        req.prefix_tokens_saved = 0
        req.first_token_step = None
        req.num_preemptions += 1
        req.status = RequestStatus.WAITING
        self.running.remove(req)
        self.waiting.appendleft(req)
        batch.preempted.append(req)

    def schedule(self) -> Batch | None:
        """
        Return a single-phase Batch for this step, or None if idle
        (no waiting and no running requests).

        Phase selection policy:
          - PREFILL_FIRST:
              * If any prefill work exists (running prefills or waiting queue
                non-empty), try _schedule_prefill().
              * Otherwise, schedule decode.
          - DECODE_FIRST:
              * If any decode-ready running request exists, try
                _schedule_decode().
              * Otherwise, schedule prefill.

        Delegates to _schedule_prefill() / _schedule_decode().
        See README.md → Task 2 for the full algorithm.
        """
        # TODO
        # A step always "ticks", even when there is nothing runnable.
        if not self.running and not self.waiting:
            self.step += 1
            return None

        # Snapshot what kinds of work exist *before* scheduling mutates state.
        has_prefill_work = bool(self.waiting) or any(
            r.is_prefilling for r in self.running
        )
        has_decode_ready = any(not r.is_prefilling for r in self.running)

        if self.scheduling_policy == SchedulingPolicy.PREFILL_FIRST:
            if has_prefill_work:
                batch = self._schedule_prefill()
                # "Useful work" rule: if prefill could neither continue nor admit
                # anyone but decode-ready requests exist, run decode instead of
                # returning a useless empty prefill batch.
                if (
                    not batch.to_prefill
                    and not batch.newly_admitted
                    and has_decode_ready
                ):
                    batch = self._schedule_decode()
            else:
                batch = self._schedule_decode()
        else:  # DECODE_FIRST
            if has_decode_ready:
                batch = self._schedule_decode()
                # Symmetric fall-through: if no decode actually ran but prefill
                # work is available, do prefill this step.
                if not batch.to_decode and has_prefill_work:
                    batch = self._schedule_prefill()
            else:
                batch = self._schedule_prefill()

        self.step += 1
        return batch        

    def _schedule_prefill(self) -> Batch:
        """
        Build a prefill Batch.

        Step A — running requests still prefilling (iterate over a copy - list(self.running)):
          Compute chunk = min(remaining_prefill, prefill_chunk, budget).
          Allocate any new blocks the chunk needs (allocation may evict cache
          entries internally); _preempt on allocation failure.
          Add (req, chunk) to batch.to_prefill; deduct from budget.

        Step B — admit from waiting while budget > 0 and slots remain:
          If prefix caching: call match_prefix FIRST → if hit, lock the
          handle and reduce the number of blocks to allocate.
          Allocate the remaining blocks; on failure unlock the handle and break.
          Build block_table = matched_blocks + newly allocated blocks.
          Set num_computed_tokens, prefix_tokens_saved, cache_handle.
          If the entire prompt was cached, skip adding to to_prefill.
          Append to running and newly_admitted; add first chunk to batch.

        Note:
          Keep this batch phase-pure: populate only batch.to_prefill here.
        """
        # TODO
        batch = Batch(phase=BatchPhase.PREFILL)
        budget = self.token_budget
        # Step A: advance running requests that are still prefilling
        # Iterate over a copy: _preempt() mutates self.running.
        for req in list(self.running):
            if not req.is_prefilling:
                continue
            if budget <= 0:
                break
            # Full-admission invariant: a running prefill request must already own enough blocks for its whole prompt
            # If not, preempt it
            if len(req.block_table) < self._blocks_for(req.prompt_len):
                self._preempt(req, batch)
                continue
            chunk = min(req.remaining_prefill, self.prefill_chunk, budget)
            batch.to_prefill.append((req, chunk))
            budget -= chunk

        # Steb B: admit waiting requests in strict FCFS order
        while self.waiting and budget > 0 and len(self.running) < self.max_seqs:
            req = self.waiting[0]

            # Try the prefix cache first so we can skip already computed blocks
            handle: CacheHandle | None = None
            matched_blocks: list[int] = []
            num_matched = 0

            if self.enable_prefix_caching:
                handle = self.cache_manager.match_prefix(req.prompt_tokens)
                if handle.num_matched_tokens > 0:
                    num_matched = handle.num_matched_tokens
                    matched_blocks = handle.matched_blocks
                else:
                    handle = None  # a miss carries no blocks to lock/unlock

            total_blocks = self._blocks_for(req.prompt_len)
            blocks_to_alloc = total_blocks - len(matched_blocks)

            # Pin the matched blocks BEFORE allocating the rest, so eviction
            # triggered by allocate() can't reclaim the prefix we are reusing.
            if handle is not None:
                self.cache_manager.lock(handle)

            new_blocks = (
                self.cache_manager.allocate(blocks_to_alloc)
                if blocks_to_alloc > 0
                else []
            )
            if new_blocks is None:
                # Out of memory: roll back the lock and stop admitting. Leave the
                # request at the front of the queue (no FCFS bypass).
                if handle is not None:
                    self.cache_manager.unlock(handle)
                break

            # Commit the admission.
            self.waiting.popleft()
            req.status = RequestStatus.RUNNING
            # Fresh list — must not alias the cache's internal block list, or a
            # later block_table.extend() during decode would corrupt the cache.
            req.block_table = list(matched_blocks) + list(new_blocks)
            req.num_computed_tokens = num_matched
            req.prefix_tokens_saved = num_matched
            req.cache_handle = handle
            req.first_scheduled_step = self.step
            self.running.append(req)
            batch.newly_admitted.append(req)

            # Schedule the first prefill chunk — unless the prompt was fully
            # cached, in which case there is no prefill compute to do.
            remaining = req.remaining_prefill
            if remaining > 0:
                chunk = min(remaining, self.prefill_chunk, budget)
                if chunk > 0:
                    batch.to_prefill.append((req, chunk))
                    budget -= chunk

        return batch

    def _schedule_decode(self) -> Batch:
        """
        Build a decode Batch (iterate over a copy of running).

        For each request: if the next token crosses a block boundary
        (tokens_so_far + 1 needs a new block), allocate one block;
        _preempt on failure. Append to batch.to_decode.

        Note:
          Only include decode-ready requests (not still-prefilling ones).
        """
        # TODO
        batch = Batch(phase=BatchPhase.DECODE)
        # Iterate over a copy: a preemption removes a request from self.running,
        # and we must still consider the requests that come after it.
        for req in list(self.running):
            # Decode batches are phase-pure — never include still-prefilling reqs.
            if req.is_prefilling:
                continue
            tokens_so_far = req.num_computed_tokens + req.num_generated_tokens
            # Only a position that crosses a block boundary needs a new block.
            if self._blocks_for(tokens_so_far + 1) > len(req.block_table):
                new_block = self.cache_manager.allocate(1)
                if new_block is None:
                    # No memory for the next token → preempt and keep going.
                    self._preempt(req, batch)
                    continue
                req.block_table.extend(new_block)
            batch.to_decode.append(req)
        return batch

# ── MiniEngine (provided — do not modify) ────────────────────────────────────


class MiniEngine:
    def __init__(
        self,
        num_blocks: int = NUM_BLOCKS,
        block_size: int = BLOCK_SIZE,
        enable_prefix_caching: bool = True,
        scheduling_policy: SchedulingPolicy | str = SchedulingPolicy.PREFILL_FIRST,
    ) -> None:
        self.enable_prefix_caching = enable_prefix_caching
        self.cache_manager = CacheManager(num_blocks, block_size)
        self.model = DummyLLM(num_blocks, block_size)
        self.scheduler = Scheduler(
            self.cache_manager,
            block_size,
            enable_prefix_caching=enable_prefix_caching,
            scheduling_policy=scheduling_policy,
        )

    def run(
        self, workload: list[Request], label: str = ""
    ) -> tuple[list[Request], list[StepMetrics]]:
        requests = sorted([r.copy() for r in workload], key=lambda r: r.arrival_step)
        finished: list[Request] = []
        all_metrics: list[StepMetrics] = []
        next_idx, step = 0, 0
        prog = tqdm(desc=label, unit="step", mininterval=0.25)
        last_prog_ts = 0.0

        def refresh_progress(force: bool = False) -> None:
            nonlocal last_prog_ts
            now = time.monotonic()
            if force or (now - last_prog_ts >= 0.5):
                prog.update(step - prog.n)
                prog.set_postfix_str(
                    f"done={len(finished)}/{len(requests)} "
                    f"running={len(self.scheduler.running)} "
                    f"waiting={len(self.scheduler.waiting)}"
                )
                last_prog_ts = now

        while len(finished) < len(requests):
            # Admit newly arrived requests
            while next_idx < len(requests) and requests[next_idx].arrival_step <= step:
                self.scheduler.add(requests[next_idx])
                next_idx += 1

            if not self.scheduler.running and not self.scheduler.waiting:
                if next_idx < len(requests):
                    step = requests[next_idx].arrival_step
                    continue
                break

            batch = self.scheduler.schedule()
            if batch is None:
                step += 1
                refresh_progress()
                continue

            if batch.is_prefill:
                for req, chunk in batch.to_prefill:
                    req._next_token = self.model.prefill(
                        req.prompt_tokens,
                        req.block_table,
                        req.num_computed_tokens,
                        chunk,
                    )
                    req.num_computed_tokens += chunk
            else:
                for req in batch.to_decode:
                    input_tok = getattr(req, "_next_token", req.prompt_tokens[-1])
                    pos = req.num_computed_tokens + req.num_generated_tokens
                    req._next_token = self.model.decode(
                        input_tok,
                        req.block_table,
                        pos,
                    )
                    req.num_generated_tokens += 1
                    if req.num_generated_tokens == 1 and req.first_token_step is None:
                        req.first_token_step = step

            done_this_step = 0
            for req in list(self.scheduler.running):
                if req.is_done:
                    req.finish_step = step
                    req.status = RequestStatus.DONE
                    self.scheduler.running.remove(req)
                    if self.enable_prefix_caching:
                        self.cache_manager.insert_prefix(
                            req.prompt_tokens, req.block_table
                        )
                    if req.cache_handle is not None:
                        n = len(req.cache_handle.matched_blocks)
                        self.cache_manager.unlock(req.cache_handle)
                        self.cache_manager.free(req.block_table[n:])
                    else:
                        self.cache_manager.free(req.block_table)
                    finished.append(req)
                    done_this_step += 1
            all_metrics.append(
                StepMetrics(
                    step=step,
                    decode_tokens=len(batch.to_decode),
                    prefill_tokens=sum(c for _, c in batch.to_prefill),
                    num_running=len(self.scheduler.running),
                    num_waiting=len(self.scheduler.waiting),
                    kv_blocks_used=self.cache_manager.num_blocks
                    - self.cache_manager.num_free_blocks,
                    prefix_tokens_saved=sum(
                        r.prefix_tokens_saved for r in batch.newly_admitted
                    ),
                )
            )
            step += 1
            refresh_progress(force=done_this_step > 0)

        refresh_progress(force=True)
        prog.close()
        return finished, all_metrics


# ── Main (provided — do not modify) ──────────────────────────────────────────


def main():
    print("=" * 60)
    print("HW3: Mini Inference Engine")
    print("=" * 60)

    workload_configs = [
        (
            "Prefill-Heavy",
            dict(
                prompt_len_range=(64, 256),
                output_len_range=(30, 150),
                shared_prefix_len=256,
            ),
        ),
        (
            "Decode-Heavy",
            dict(
                num_requests=50,
                prompt_len_range=(48, 128),
                output_len_range=(150, 400),
                shared_prefix_len=32,
            ),
        ),
    ]

    all_results: list[tuple] = []
    policy_results: list[tuple] = []
    for label, wl_kwargs in workload_configs:
        wl = generate_workload(**wl_kwargs)
        print(f"\n{'─' * 60}")
        print(f"  {label}  ({len(wl)} requests)\n")

        eng_off = MiniEngine(enable_prefix_caching=False)
        fin_off, met_off = eng_off.run(wl, label="no-cache")
        stats_off = compute_stats(fin_off, met_off, len(met_off))
        print_stats("No prefix cache", stats_off)

        eng_on = MiniEngine(
            enable_prefix_caching=True,
            scheduling_policy=SchedulingPolicy.PREFILL_FIRST,
        )
        fin_on, met_on = eng_on.run(wl, label="cache-on")
        stats_on = compute_stats(fin_on, met_on, len(met_on))
        print_stats("Prefix cache ON", stats_on)

        speedup = stats_off["total_steps"] / max(stats_on["total_steps"], 1)
        print(
            f"\n    Steps: {stats_off['total_steps']} → {stats_on['total_steps']}  "
            f"({speedup:.2f}× fewer)"
        )
        print(f"    TTFT:  {stats_off['ttft_mean']} → {stats_on['ttft_mean']} steps")

        all_results.append((label, met_off, met_on, stats_off, stats_on))

        eng_decode_first = MiniEngine(
            enable_prefix_caching=True,
            scheduling_policy=SchedulingPolicy.DECODE_FIRST,
        )
        fin_df, met_df = eng_decode_first.run(wl, label="cache-on/decode-first")
        stats_df = compute_stats(fin_df, met_df, len(met_df))

        print("\n  Scheduling policy (cache ON)")
        print(
            f"    Prefill-first steps / TTFT / E2E : "
            f"{stats_on['total_steps']} / {stats_on['ttft_mean']} / {stats_on['e2e_mean']}"
        )
        print(
            f"    Decode-first  steps / TTFT / E2E : "
            f"{stats_df['total_steps']} / {stats_df['ttft_mean']} / {stats_df['e2e_mean']}"
        )
        policy_results.append((label, met_on, met_df, stats_on, stats_df))

    print(f"\n{'─' * 60}")
    plot_results(all_results)
    plot_policy_results(policy_results)


if __name__ == "__main__":
    main()


# ── Writeup ───────────────────────────────────────────────────────────────────
#
# Q1: Compare the prefix cache's impact on TTFT and E2E latency between the
#     two workloads.  Why is the speedup much larger for the prefill-heavy
#     workload?  Give specific numbers from your run.
#
# Q2: Trace the ref-count lifecycle of a shared prefix block from the moment
#     a first request finishes (insert_prefix) through a second request
#     using that block (match_prefix → lock → run → unlock) to the eventual
#     eviction.  What is the ref count at each stage, and what prevents the
#     block from being evicted while the second request is live?
#
# Q3: With prefix caching ON, why does eviction reduce preemptions compared
#     to the no-caching run?  Under what condition would eviction fail and
#     fall back to preemption?
#
# Q4: Compare the two scheduling policies (PREFILL_FIRST vs DECODE_FIRST)
#     using the numbers on your policy-comparison plot. On which workload
#     does the choice of policy matter a lot, and on which is it almost
#     a wash?  Explain what each policy optimises for, and name a
#     realistic scenario in which you would pick each one.
#
# Q1: From the summary panels: prefill-heavy TTFT 233 -> 48 steps and E2E 360 -> 152 
#     with caching on (50.1% hit); decode-heavy TTFT 738 -> 431 and E2E ~ 1084 -> 773 (10.5% hit).
#     The cache only removes prefill work, so the workload with long shared prompts (prefill-heavy, 
#     half its prompt tokens cached) skips huge prompt recompute, while decode heavy spends most of 
#     its life in per-token decode that caching can't shortcut - hence the far bigger speedup on prefill-heavy.
#
# Q2: ref-count lifecycle. insert_prefix while the first request still owns the block -> ref 2 (request + cache),
#     cache_ref ≥1. Request frees → ref 1 (cache-only, evictable). Second request match_prefix → ref unchanged at 1; 
#     lock -> ref 2 (pinned). Eviction skips any block with ref ≥2, so the live second request can't have its prefix evicted. 
#     unlock -> ref 1; later LRU eviction drops cache_ref to 0, ref to 0, block returns to free.
#
# Q3: From the summary, we see preemptions 91→18 (prefill-heavy) and 273→159 (decode-heavy). 
#     Two reasons: a cache hit means an admitted request allocates fewer fresh blocks, and allocate() reclaims cached-only blocks via LRU eviction before resorting to preemption. 
#     Eviction fails → falls back to preemption only when every cached block is locked by a live request (ref ≥2), so the LRU walk frees nothing.
#
# Q4: From the policy plot, prefill-heavy E2E 152 (prefill-first) vs 192 (decode-first), TTFT 48 vs 100 — policy matters a lot. 
#     Decode-heavy E2E ~773 vs ~731, TTFT ~431 vs ~434 — almost a wash (decode-first marginally better E2E). 
#     PREFILL_FIRST optimizes admission/throughput and TTFT (good for high-QPS chat with shared system prompts); DECODE_FIRST optimizes draining in-flight requests / E2E tail (good for long-generation batch jobs with few requests)
#
