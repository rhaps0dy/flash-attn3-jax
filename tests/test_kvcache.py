import jax
import jax.numpy as jnp
import pytest

from flash_attn3_jax import flash_mha_with_kvcache

from .ref_mha import ref_mha
from .test_utils import check

jax.config.update("jax_default_matmul_precision", "highest")


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("query_seqlen", [1, 8])
@pytest.mark.parametrize("context_seqlen", [512, 1024, 2048])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize(
    "nheads_kv, gqa_ratio",
    [
        (2, 2),
        (4, 2),
        (1, 8),
    ],
)
def test_flash_attn_kvcache_basic(
    nheads_kv,
    gqa_ratio,
    batch_size,
    query_seqlen,
    context_seqlen,
    head_dim,
    is_causal,
    dtype,
):
    """Test basic KV cache functionality against reference implementation"""
    nheads_q = nheads_kv * gqa_ratio

    key = jax.random.PRNGKey(42)
    key, *subkeys = jax.random.split(key, 5)

    q = jax.random.normal(
        subkeys[0], (batch_size, query_seqlen, nheads_q, head_dim), dtype=jnp.float32
    )
    k_cache = jax.random.normal(
        subkeys[1], (batch_size, context_seqlen, nheads_kv, head_dim), dtype=jnp.float32
    )
    v_cache = jax.random.normal(
        subkeys[2], (batch_size, context_seqlen, nheads_kv, head_dim), dtype=jnp.float32
    )

    cache_seqlens = jnp.full((batch_size,), context_seqlen, dtype=jnp.int32)

    ref_out = ref_mha(q, k_cache, v_cache, is_causal=is_causal)

    q = q.astype(dtype)
    k_cache = k_cache.astype(dtype)
    v_cache = v_cache.astype(dtype)

    jax_out = ref_mha(q, k_cache, v_cache, is_causal=is_causal)

    out = flash_mha_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        is_causal=is_causal,
    )

    check(ref_out, jax_out, out)


@pytest.mark.parametrize("dtype", [jnp.bfloat16])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("query_seqlen", [1, 16])
@pytest.mark.parametrize("context_seqlen", [1024, 2048])
@pytest.mark.parametrize("head_dim", [128])
def test_flash_attn_kvcache_with_new_kv(
    query_seqlen, context_seqlen, head_dim, is_causal, dtype
):
    """Test KV cache with new KV."""
    batch_size = 2
    nheads_q = 8
    nheads_kv = 2
    seqlen_new = 8

    key = jax.random.PRNGKey(123)
    key, *subkeys = jax.random.split(key, 6)

    q = jax.random.normal(
        subkeys[0], (batch_size, query_seqlen, nheads_q, head_dim), dtype=jnp.float32
    )
    k_cache = jax.random.normal(
        subkeys[1], (batch_size, context_seqlen, nheads_kv, head_dim), dtype=jnp.float32
    )
    v_cache = jax.random.normal(
        subkeys[2], (batch_size, context_seqlen, nheads_kv, head_dim), dtype=jnp.float32
    )
    k_new = jax.random.normal(
        subkeys[3], (batch_size, seqlen_new, nheads_kv, head_dim), dtype=jnp.float32
    )
    v_new = jax.random.normal(
        subkeys[4], (batch_size, seqlen_new, nheads_kv, head_dim), dtype=jnp.float32
    )

    cache_seqlens = context_seqlen - seqlen_new

    k_ref = k_cache.at[:, cache_seqlens:, :, :].set(k_new.astype(dtype))
    v_ref = v_cache.at[:, cache_seqlens:, :, :].set(v_new.astype(dtype))

    ref_out = ref_mha(q, k_ref, v_ref, is_causal=is_causal)

    q = q.astype(dtype)
    k_cache = k_cache.astype(dtype)
    v_cache = v_cache.astype(dtype)
    k_new = k_new.astype(dtype)
    v_new = v_new.astype(dtype)
    k_ref = k_ref.astype(dtype)
    v_ref = v_ref.astype(dtype)

    jax_out = ref_mha(q, k_ref, v_ref, is_causal=is_causal)

    out = flash_mha_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k_new,
        v=v_new,
        cache_seqlens=cache_seqlens,
        is_causal=is_causal,
    )

    check(ref_out, jax_out, out)


@pytest.mark.parametrize("dtype", [jnp.bfloat16])
@pytest.mark.parametrize("is_causal", [True, False])
def test_flash_attn_kvcache_different_cache_lengths(is_causal, dtype):
    """Test KV cache with different cache lengths per batch item."""
    batch_size = 4
    query_seqlen = 8
    max_context_seqlen = 1024
    nheads_q = 8
    nheads_kv = 2
    head_dim = 128

    key = jax.random.PRNGKey(456)
    key, *subkeys = jax.random.split(key, 4)

    q = jax.random.normal(
        subkeys[0], (batch_size, query_seqlen, nheads_q, head_dim), dtype=jnp.float32
    )
    k_cache = jax.random.normal(
        subkeys[1],
        (batch_size, max_context_seqlen, nheads_kv, head_dim),
        dtype=jnp.float32,
    )
    v_cache = jax.random.normal(
        subkeys[2],
        (batch_size, max_context_seqlen, nheads_kv, head_dim),
        dtype=jnp.float32,
    )

    cache_seqlens = jnp.array([512, 768, 1024, 256], dtype=jnp.int32)
    out = flash_mha_with_kvcache(
        q=q.astype(dtype),
        k_cache=k_cache.astype(dtype),
        v_cache=v_cache.astype(dtype),
        cache_seqlens=cache_seqlens,
        is_causal=is_causal,
    )

    for i in range(batch_size):
        seqlen_i = cache_seqlens[i].item()
        q_i = q[i : i + 1]
        k_i = k_cache[i : i + 1, :seqlen_i, :, :]
        v_i = v_cache[i : i + 1, :seqlen_i, :, :]

        ref_out_i = ref_mha(q_i, k_i, v_i, is_causal=is_causal)

        q_i_dtype = q_i.astype(dtype)
        k_i_dtype = k_i.astype(dtype)
        v_i_dtype = v_i.astype(dtype)

        jax_out_i = ref_mha(q_i_dtype, k_i_dtype, v_i_dtype, is_causal=is_causal)

        check(ref_out_i, jax_out_i, out[i])


@pytest.mark.parametrize("dtype", [jnp.bfloat16])
@pytest.mark.parametrize("is_causal", [True, False])
def test_flash_attn_kvcache_batch_idx(is_causal, dtype):
    """Test KV cache with custom cache_batch_idx mapping."""
    batch_size = 3
    num_caches = 5
    query_seqlen = 4
    context_seqlen = 512
    nheads_q = 8
    nheads_kv = 2
    head_dim = 64

    key = jax.random.PRNGKey(789)
    key, *subkeys = jax.random.split(key, 4)

    q = jax.random.normal(
        subkeys[0], (batch_size, query_seqlen, nheads_q, head_dim), dtype=jnp.float32
    )
    k_cache = jax.random.normal(
        subkeys[1], (num_caches, context_seqlen, nheads_kv, head_dim), dtype=jnp.float32
    )
    v_cache = jax.random.normal(
        subkeys[2], (num_caches, context_seqlen, nheads_kv, head_dim), dtype=jnp.float32
    )

    cache_batch_idx = jnp.array([0, 2, 4], dtype=jnp.int32)
    cache_seqlens = jnp.full((batch_size,), context_seqlen, dtype=jnp.int32)

    out = flash_mha_with_kvcache(
        q=q.astype(dtype),
        k_cache=k_cache.astype(dtype),
        v_cache=v_cache.astype(dtype),
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        is_causal=is_causal,
    )

    for i in range(batch_size):
        cache_idx = cache_batch_idx[i].item()
        q_i = q[i : i + 1]
        k_i = k_cache[cache_idx : cache_idx + 1]
        v_i = v_cache[cache_idx : cache_idx + 1]

        ref_out_i = ref_mha(q_i, k_i, v_i, is_causal=is_causal)

        q_i_dtype = q_i.astype(dtype)
        k_i_dtype = k_i.astype(dtype)
        v_i_dtype = v_i.astype(dtype)

        jax_out_i = ref_mha(q_i_dtype, k_i_dtype, v_i_dtype, is_causal=is_causal)

        check(ref_out_i, jax_out_i, out[i])


@pytest.mark.parametrize("dtype", [jnp.bfloat16])
@pytest.mark.parametrize("window_size", [(-1, -1), (256, 0), (128, 128)])
@pytest.mark.parametrize("context_seqlen", [1024])
@pytest.mark.parametrize("is_causal", [True, False])
def test_flash_attn_kvcache_window(window_size, context_seqlen, is_causal, dtype):
    """Test KV cache with sliding window attention."""
    batch_size = 2
    query_seqlen = 8
    nheads_q = 4
    nheads_kv = 4
    head_dim = 64

    key = jax.random.PRNGKey(987)
    key, *subkeys = jax.random.split(key, 4)

    q = jax.random.normal(
        subkeys[0], (batch_size, query_seqlen, nheads_q, head_dim), dtype=jnp.float32
    )
    k_cache = jax.random.normal(
        subkeys[1], (batch_size, context_seqlen, nheads_kv, head_dim), dtype=jnp.float32
    )
    v_cache = jax.random.normal(
        subkeys[2], (batch_size, context_seqlen, nheads_kv, head_dim), dtype=jnp.float32
    )

    cache_seqlens = jnp.full((batch_size,), context_seqlen, dtype=jnp.int32)

    ref_out = ref_mha(q, k_cache, v_cache, is_causal=is_causal, window_size=window_size)

    q = q.astype(dtype)
    k_cache = k_cache.astype(dtype)
    v_cache = v_cache.astype(dtype)

    jax_out = ref_mha(q, k_cache, v_cache, is_causal=is_causal, window_size=window_size)

    out = flash_mha_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        is_causal=is_causal,
        window_size=window_size,
    )

    check(ref_out, jax_out, out)


@pytest.mark.parametrize("num_splits", [1, 4, 8, 16])
@pytest.mark.parametrize("is_causal", [True, False])
def test_flash_attn_kvcache_num_splits(num_splits, is_causal):
    """Test contiguous KV cache with different num_splits values."""
    batch_size = 2
    query_seqlen = 8
    context_seqlen = 1024
    nheads_q = 8
    nheads_kv = 2
    head_dim = 128
    dtype = jnp.bfloat16

    key = jax.random.PRNGKey(999)
    key, *subkeys = jax.random.split(key, 4)

    q = jax.random.normal(
        subkeys[0], (batch_size, query_seqlen, nheads_q, head_dim), dtype=jnp.float32
    )
    k_cache = jax.random.normal(
        subkeys[1], (batch_size, context_seqlen, nheads_kv, head_dim), dtype=jnp.float32
    )
    v_cache = jax.random.normal(
        subkeys[2], (batch_size, context_seqlen, nheads_kv, head_dim), dtype=jnp.float32
    )

    cache_seqlens = jnp.full((batch_size,), context_seqlen, dtype=jnp.int32)

    ref_out = ref_mha(q, k_cache, v_cache, is_causal=is_causal)

    q = q.astype(dtype)
    k_cache = k_cache.astype(dtype)
    v_cache = v_cache.astype(dtype)

    jax_out = ref_mha(q, k_cache, v_cache, is_causal=is_causal)

    out = flash_mha_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        is_causal=is_causal,
        num_splits=num_splits,
    )

    check(ref_out, jax_out, out)


# Paged KV cache tests
def generate_paged_kv_cache(
    seqlen_k, page_size, batch_size, nheads_k, head_dim, dtype, key
):
    """Generate paged KV cache matching PyTorch implementation.

    Generates random paged KV cache data first, then creates a contiguous view
    by gathering pages according to a random page table. This matches the
    hopper/test_flash_attn.py::_generate_block_kvcache.

    Note: seqlen_k must be evenly divisible by page_size.
    """

    assert seqlen_k % page_size == 0, (
        f"seqlen_k ({seqlen_k}) must be divisible by page_size ({page_size})"
    )

    num_pages_per_seq = seqlen_k // page_size
    num_blocks = num_pages_per_seq * batch_size * 3

    key, k_key, v_key, perm_key = jax.random.split(key, 4)

    k_cache_paged = jax.random.normal(
        k_key, (num_blocks, page_size, nheads_k, head_dim), dtype=dtype
    )
    v_cache_paged = jax.random.normal(
        v_key, (num_blocks, page_size, nheads_k, head_dim), dtype=dtype
    )

    perm = jax.random.permutation(perm_key, num_blocks)
    page_table = (
        perm[: batch_size * num_pages_per_seq]
        .reshape(batch_size, num_pages_per_seq)
        .astype(jnp.int32)
    )

    k_gathered = k_cache_paged[page_table.flatten()]
    v_gathered = v_cache_paged[page_table.flatten()]

    k_cache_contiguous = k_gathered.reshape(batch_size, seqlen_k, nheads_k, head_dim)
    v_cache_contiguous = v_gathered.reshape(batch_size, seqlen_k, nheads_k, head_dim)

    return (
        k_cache_contiguous,
        v_cache_contiguous,
        page_table,
        k_cache_paged,
        v_cache_paged,
    )


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("page_size", [64, 128, 256])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("query_seqlen", [1, 8, 32])
@pytest.mark.parametrize("context_seqlen", [512, 1024, 2048])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(
    "nheads_kv, gqa_ratio",
    [
        (2, 2),
        (4, 2),
        (1, 8),
    ],
)
def test_flash_attn_paged_kv_cache(
    page_size,
    nheads_kv,
    gqa_ratio,
    batch_size,
    query_seqlen,
    context_seqlen,
    head_dim,
    is_causal,
    dtype,
):
    """Test paged KV cache against contiguous KV cache reference.

    Note: seqlen_k must be divisible by page_size for paged KV cache to work.
    """
    if context_seqlen % page_size != 0:
        pytest.skip(
            f"context_seqlen ({context_seqlen}) must be divisible by page_size ({page_size})"
        )

    nheads_q = nheads_kv * gqa_ratio

    key = jax.random.PRNGKey(765)
    key, q_key, cache_key = jax.random.split(key, 3)

    q = jax.random.normal(
        q_key, (batch_size, query_seqlen, nheads_q, head_dim), dtype=jnp.float32
    )

    (
        k_cache_contiguous,
        v_cache_contiguous,
        page_table,
        k_cache_paged,
        v_cache_paged,
    ) = generate_paged_kv_cache(
        context_seqlen, page_size, batch_size, nheads_kv, head_dim, dtype, cache_key
    )

    ref_out = ref_mha(
        q,
        k_cache_contiguous.astype(jnp.float32),
        v_cache_contiguous.astype(jnp.float32),
        is_causal=is_causal,
    )

    q = q.astype(dtype)

    jax_out = ref_mha(
        q,
        k_cache_contiguous,
        v_cache_contiguous,
        is_causal=is_causal,
    )

    out_paged = flash_mha_with_kvcache(
        q=q,
        k_cache=k_cache_paged,
        v_cache=v_cache_paged,
        cache_seqlens=context_seqlen,
        page_table=page_table,
        is_causal=is_causal,
    )

    out_contiguous = flash_mha_with_kvcache(
        q=q,
        k_cache=k_cache_contiguous.astype(dtype),
        v_cache=v_cache_contiguous.astype(dtype),
        cache_seqlens=context_seqlen,
        is_causal=is_causal,
    )

    check(ref_out, jax_out, out_paged)
    check(ref_out, out_contiguous, out_paged)


@pytest.mark.parametrize("dtype", [jnp.bfloat16])
@pytest.mark.parametrize("page_size", [64, 128, 256])
@pytest.mark.parametrize("is_causal", [True, False])
def test_flash_attn_paged_kv_cache_variable_seqlens(page_size, is_causal, dtype):
    """Test paged KV cache with different sequence lengths per batch item."""
    batch_size = 4
    query_seqlen = 1
    max_context_seqlen = 2048
    nheads_q = 32
    nheads_kv = 8
    head_dim = 128

    key = jax.random.PRNGKey(543)
    key, q_key, cache_key = jax.random.split(key, 3)

    q = jax.random.normal(
        q_key, (batch_size, query_seqlen, nheads_q, head_dim), dtype=jnp.float32
    )

    (
        k_cache_contiguous,
        v_cache_contiguous,
        page_table,
        k_cache_paged,
        v_cache_paged,
    ) = generate_paged_kv_cache(
        max_context_seqlen, page_size, batch_size, nheads_kv, head_dim, dtype, cache_key
    )

    cache_seqlens = jnp.array([512, 1024, 2048, 1536], dtype=jnp.int32)
    out_paged = flash_mha_with_kvcache(
        q=q.astype(dtype),
        k_cache=k_cache_paged,
        v_cache=v_cache_paged,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        is_causal=is_causal,
    )
    for i in range(batch_size):
        seqlen_i = cache_seqlens[i].item()
        q_i = q[i : i + 1]
        k_i = k_cache_contiguous[i : i + 1, :seqlen_i, :, :]
        v_i = v_cache_contiguous[i : i + 1, :seqlen_i, :, :]

        ref_out_i = ref_mha(q_i, k_i, v_i, is_causal=is_causal)

        q_i_dtype = q_i.astype(dtype)
        k_i_dtype = k_i.astype(dtype)
        v_i_dtype = v_i.astype(dtype)

        jax_out_i = ref_mha(q_i_dtype, k_i_dtype, v_i_dtype, is_causal=True)

        check(ref_out_i, jax_out_i, out_paged[i])


@pytest.mark.parametrize("dtype", [jnp.bfloat16])
@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("query_seqlen", [1, 16])
@pytest.mark.parametrize("context_seqlen", [1024, 2048])
@pytest.mark.parametrize("head_dim", [128])
def test_flash_attn_paged_kv_cache_with_new_kv(
    query_seqlen, context_seqlen, page_size, head_dim, is_causal, dtype
):
    """Test paged KV cache with new KV."""
    batch_size = 2
    nheads_q = 8
    nheads_kv = 2
    seqlen_new = 16

    key = jax.random.PRNGKey(321)
    key, q_key, cache_key, k_new_key, v_new_key = jax.random.split(key, 5)

    q = jax.random.normal(
        q_key, (batch_size, query_seqlen, nheads_q, head_dim), dtype=jnp.float32
    )

    (
        k_cache_contiguous,
        v_cache_contiguous,
        page_table,
        k_cache_paged,
        v_cache_paged,
    ) = generate_paged_kv_cache(
        context_seqlen, page_size, batch_size, nheads_kv, head_dim, dtype, cache_key
    )

    k_new = jax.random.normal(
        k_new_key, (batch_size, seqlen_new, nheads_kv, head_dim), dtype=jnp.float32
    )
    v_new = jax.random.normal(
        v_new_key, (batch_size, seqlen_new, nheads_kv, head_dim), dtype=jnp.float32
    )

    cache_seqlens = context_seqlen - seqlen_new

    k_ref = k_cache_contiguous.at[:, cache_seqlens:, :, :].set(k_new.astype(dtype))
    v_ref = v_cache_contiguous.at[:, cache_seqlens:, :, :].set(v_new.astype(dtype))

    ref_out = ref_mha(q, k_ref, v_ref, is_causal=is_causal)

    q = q.astype(dtype)
    k_new = k_new.astype(dtype)
    v_new = v_new.astype(dtype)
    k_ref = k_ref.astype(dtype)
    v_ref = v_ref.astype(dtype)

    jax_out = ref_mha(q, k_ref, v_ref, is_causal=is_causal)

    out = flash_mha_with_kvcache(
        q=q,
        k_cache=k_cache_paged,
        v_cache=v_cache_paged,
        page_table=page_table,
        k=k_new,
        v=v_new,
        cache_seqlens=cache_seqlens,
        is_causal=is_causal,
    )

    check(ref_out, jax_out, out)


@pytest.mark.parametrize("dtype", [jnp.bfloat16])
@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("is_causal", [True, False])
def test_flash_attn_paged_kv_cache_batch_idx(page_size, is_causal, dtype):
    """Test paged KV cache with custom cache_batch_idx mapping."""
    batch_size = 3
    num_caches = 5
    query_seqlen = 4
    context_seqlen = 512
    nheads_q = 8
    nheads_kv = 2
    head_dim = 64

    if context_seqlen % page_size != 0:
        pytest.skip(
            f"context_seqlen ({context_seqlen}) must be divisible by page_size ({page_size})"
        )

    key = jax.random.PRNGKey(654)
    key, q_key, cache_key = jax.random.split(key, 3)

    q = jax.random.normal(
        q_key, (batch_size, query_seqlen, nheads_q, head_dim), dtype=jnp.float32
    )

    (
        k_cache_contiguous,
        v_cache_contiguous,
        page_table,
        k_cache_paged,
        v_cache_paged,
    ) = generate_paged_kv_cache(
        context_seqlen, page_size, num_caches, nheads_kv, head_dim, dtype, cache_key
    )

    cache_batch_idx = jnp.array([0, 2, 4], dtype=jnp.int32)
    cache_seqlens = jnp.full((batch_size,), context_seqlen, dtype=jnp.int32)

    out_paged = flash_mha_with_kvcache(
        q=q.astype(dtype),
        k_cache=k_cache_paged,
        v_cache=v_cache_paged,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        page_table=page_table,
        is_causal=is_causal,
    )

    for i in range(batch_size):
        cache_idx = cache_batch_idx[i].item()
        q_i = q[i : i + 1]
        k_i = k_cache_contiguous[cache_idx : cache_idx + 1]
        v_i = v_cache_contiguous[cache_idx : cache_idx + 1]

        ref_out_i = ref_mha(q_i, k_i, v_i, is_causal=is_causal)

        q_i_dtype = q_i.astype(dtype)
        k_i_dtype = k_i.astype(dtype)
        v_i_dtype = v_i.astype(dtype)

        jax_out_i = ref_mha(q_i_dtype, k_i_dtype, v_i_dtype, is_causal=is_causal)

        check(ref_out_i, jax_out_i, out_paged[i])


@pytest.mark.parametrize("dtype", [jnp.bfloat16])
@pytest.mark.parametrize("page_size", [64, 128, 256])
@pytest.mark.parametrize("window_size", [(-1, -1), (256, 0), (128, 128)])
@pytest.mark.parametrize("context_seqlen", [1024])
@pytest.mark.parametrize("is_causal", [True, False])
def test_flash_attn_paged_kv_cache_window(
    page_size, window_size, context_seqlen, is_causal, dtype
):
    """Test paged KV cache with sliding window attention."""
    batch_size = 2
    query_seqlen = 8
    nheads_q = 4
    nheads_kv = 4
    head_dim = 64

    if context_seqlen % page_size != 0:
        pytest.skip(
            f"context_seqlen ({context_seqlen}) must be divisible by page_size ({page_size})"
        )

    key = jax.random.PRNGKey(147)
    key, q_key, cache_key = jax.random.split(key, 3)

    q = jax.random.normal(
        q_key, (batch_size, query_seqlen, nheads_q, head_dim), dtype=jnp.float32
    )

    (
        k_cache_contiguous,
        v_cache_contiguous,
        page_table,
        k_cache_paged,
        v_cache_paged,
    ) = generate_paged_kv_cache(
        context_seqlen, page_size, batch_size, nheads_kv, head_dim, dtype, cache_key
    )

    cache_seqlens = jnp.full((batch_size,), context_seqlen, dtype=jnp.int32)

    ref_out = ref_mha(
        q,
        k_cache_contiguous,
        v_cache_contiguous,
        is_causal=is_causal,
        window_size=window_size,
    )

    q = q.astype(dtype)

    jax_out = ref_mha(
        q,
        k_cache_contiguous.astype(dtype),
        v_cache_contiguous.astype(dtype),
        is_causal=is_causal,
        window_size=window_size,
    )

    out_paged = flash_mha_with_kvcache(
        q=q,
        k_cache=k_cache_paged,
        v_cache=v_cache_paged,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        is_causal=is_causal,
        window_size=window_size,
    )

    check(ref_out, jax_out, out_paged)


@pytest.mark.parametrize("num_splits", [1, 4, 8, 16])
@pytest.mark.parametrize("is_causal", [True, False])
def test_flash_attn_paged_kv_cache_num_splits(num_splits, is_causal):
    """Test paged KV cache with different num_splits values."""
    batch_size = 2
    query_seqlen = 8
    context_seqlen = 1024
    page_size = 128
    nheads_q = 8
    nheads_kv = 2
    head_dim = 128
    dtype = jnp.bfloat16

    key = jax.random.PRNGKey(888)
    key, q_key, cache_key = jax.random.split(key, 3)

    q = jax.random.normal(
        q_key, (batch_size, query_seqlen, nheads_q, head_dim), dtype=jnp.float32
    )

    (
        k_cache_contiguous,
        v_cache_contiguous,
        page_table,
        k_cache_paged,
        v_cache_paged,
    ) = generate_paged_kv_cache(
        context_seqlen, page_size, batch_size, nheads_kv, head_dim, dtype, cache_key
    )

    ref_out = ref_mha(q, k_cache_contiguous, v_cache_contiguous, is_causal=is_causal)

    q = q.astype(dtype)

    jax_out = ref_mha(
        q,
        k_cache_contiguous.astype(dtype),
        v_cache_contiguous.astype(dtype),
        is_causal=is_causal,
    )

    out_paged = flash_mha_with_kvcache(
        q=q,
        k_cache=k_cache_paged,
        v_cache=v_cache_paged,
        cache_seqlens=context_seqlen,
        page_table=page_table,
        is_causal=is_causal,
        num_splits=num_splits,
    )

    check(ref_out, jax_out, out_paged)
