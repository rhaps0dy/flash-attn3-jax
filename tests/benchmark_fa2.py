import time

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flash_attn_jax import flash_mha

TEST_CASES = [
    ((1, 20, 16, 32), 1e-3),
    ((1, 2, 1, 64), 5e-4),
    ((16, 100, 28, 64), 2e-4),
    ((16, 512, 32, 128), 1e-4),
    ((21, 50, 17, 160), 5e-4),
]


def bench(label, fwd, b_sz, seq_len, n_heads, dim, n_run=20, n_warmup=4, bwd=False):
    # the flops below only include the matmul of the forward pass
    flops = 4 * b_sz * seq_len * seq_len * n_heads * dim  # b.q.k.h.d
    if bwd:
        flops *= 3.5
    if bwd:

        def loss(q, k, v):
            return jnp.sum(fwd(q, k, v))

        f = jax.grad(loss, (0, 1, 2))
    else:
        f = fwd
    qkv_shape = b_sz, seq_len, n_heads, dim

    def normal(seed):
        rng = jax.random.PRNGKey(seed)
        return jax.random.normal(rng, qkv_shape, dtype=jnp.float16)

    dts = []
    for i in range(n_warmup + n_run):
        q = normal(3 * i)
        k = normal(3 * i + 1)
        v = normal(3 * i + 2) / seq_len
        start_time = time.perf_counter()
        res = f(q, k, v)
        if bwd:
            res = res[0]
        res = res.block_until_ready()
        res = float(res.sum())
        dt = time.perf_counter() - start_time
        dts.append(dt)
    dts = dts[n_warmup:]
    dts = np.array(dts)
    min_ms = np.min(dts) * 1000
    max_ms = np.max(dts) * 1000
    mean_ms = np.mean(dts) * 1000
    std_ms = np.std(dts) * 1000
    gflops = flops / np.mean(dts) / 1e12

    return {
        "label": label,
        "seq_len": seq_len,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "tflops": gflops,
        "mean_s": np.mean(dts),
    }


def print_comparison_table(results_pairs, title):
    print(f"\n{'=' * 110}")
    print(f"{title:^110}")
    print(f"{'=' * 110}")
    print(
        f"{'Batch':>5} {'SeqLen':>7} │ {'Flash Attn':>27} │ {'Native (Flax)':>27} │ {'Speedup':>8} {'TFLOPS Δ':>10}"
    )
    print(
        f"{'Size':>5} {'':>7} │ {'Time (ms)':>13} {'TFLOPS':>13} │ {'Time (ms)':>13} {'TFLOPS':>13} │ {'':>8} {'':>10}"
    )
    print(f"{'-' * 110}")

    for flash_res, flax_res in results_pairs:
        b_sz = flash_res["seq_len"] // 512 * 32
        if flash_res["seq_len"] == 512:
            b_sz = 32
        elif flash_res["seq_len"] == 1024:
            b_sz = 16
        elif flash_res["seq_len"] == 2048:
            b_sz = 8
        elif flash_res["seq_len"] == 4096:
            b_sz = 4
        elif flash_res["seq_len"] == 8192:
            b_sz = 2
        elif flash_res["seq_len"] == 16384:
            b_sz = 1

        speedup = flax_res["mean_s"] / flash_res["mean_s"]
        tflops_delta = flash_res["tflops"] - flax_res["tflops"]

        print(
            f"{b_sz:>5} {flash_res['seq_len']:>7} │ "
            f"{flash_res['mean_ms']:>13.2f} {flash_res['tflops']:>13.1f} │ "
            f"{flax_res['mean_ms']:>13.2f} {flax_res['tflops']:>13.1f} │ "
            f"{speedup:>7.2f}x {tflops_delta:>+9.1f}"
        )

    print(f"{'=' * 110}\n")


run_mha_jit = jax.jit(flash_mha)
attn_flax_jit = jax.jit(flax.linen.dot_product_attention)

# Values taken from:
# https://github.com/Dao-AILab/flash-attention/blob/2c3baba4a63c4007c8a132c5380edc9430f88a22/benchmarks/benchmark_flash_attention.py#L74C1-L77C11
BSIZE_SEQLEN_VALS = [
    (32, 512),
    (16, 1024),
    (8, 2048),
    (4, 4096),
    (2, 8192),
    (1, 16384),
]
HEADDIM = 128
DIM = 2048
n_heads = DIM // HEADDIM

print("\nRunning forward pass benchmarks...")
fwd_results = []
for b_sz, seqlen in BSIZE_SEQLEN_VALS:
    print(f"  Benchmarking batch={b_sz}, seqlen={seqlen}...")
    flash_res = bench("flash-attn", run_mha_jit, b_sz, seqlen, n_heads, HEADDIM)
    flax_res = bench("attn-flax", attn_flax_jit, b_sz, seqlen, n_heads, HEADDIM)
    fwd_results.append((flash_res, flax_res))

print("\nRunning backward pass benchmarks...")
bwd_results = []
for b_sz, seqlen in BSIZE_SEQLEN_VALS:
    print(f"  Benchmarking batch={b_sz}, seqlen={seqlen}...")
    flash_res = bench(
        "bwd flash-attn", run_mha_jit, b_sz, seqlen, n_heads, HEADDIM, bwd=True
    )
    flax_res = bench(
        "bwd attn-flax", attn_flax_jit, b_sz, seqlen, n_heads, HEADDIM, bwd=True
    )
    bwd_results.append((flash_res, flax_res))

print_comparison_table(fwd_results, "Forward Pass Comparison")
print_comparison_table(bwd_results, "Backward Pass Comparison")
