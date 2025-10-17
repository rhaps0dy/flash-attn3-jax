import math
from functools import partial

import jax
import jax._src.dispatch
import jax.numpy as jnp
from jax import dtypes
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import mlir, xla

_flash_mha_varlen_fwd_p = Primitive("flash_mha_varlen_fwd")
_flash_mha_varlen_fwd_p.multiple_results = True
_flash_mha_varlen_fwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_varlen_fwd_p))
jax._src.dispatch.prim_requires_devices_during_lowering.add(_flash_mha_varlen_fwd_p)


def flash_mha_varlen_fwd(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k=None,
    max_seqlen_q: int = -1,
    max_seqlen_k: int = -1,
    softmax_scale: float | None = None,
    is_causal: bool = False,
    window_size: tuple = (-1, -1),
):
    if max_seqlen_q == -1:
        max_seqlen_q = q.shape[0]
    if max_seqlen_k == -1:
        max_seqlen_k = k.shape[0]
    assert cu_seqlens_q.shape == cu_seqlens_k.shape, (
        "cu_seqlens_q and cu_seqlens_k must have the same shape."
    )
    d = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    has_seqused_k = seqused_k is not None
    if seqused_k is None:
        seqused_k = jnp.empty([], dtype=jnp.int32)
    kwargs = dict(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        has_seqused_k=has_seqused_k,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
    )
    return tuple(
        _flash_mha_varlen_fwd_p.bind(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_k, **kwargs
        )
    )


def _flash_mha_varlen_fwd_hlo_lowering(
    ctx,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    max_seqlen_q: int,
    max_seqlen_k: int,
    has_seqused_k: bool,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
):
    def fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_k):
        q_dtype = dtypes.canonicalize_dtype(q.dtype)
        k_dtype = dtypes.canonicalize_dtype(k.dtype)
        v_dtype = dtypes.canonicalize_dtype(v.dtype)
        [totalq, h, d] = q.shape
        b = cu_seqlens_q.shape[0] - 1
        assert q_dtype == k_dtype and q_dtype == v_dtype
        assert q_dtype in [jnp.bfloat16, jnp.float16]
        assert b >= 1

        dpad = (8 - d % 8) % 8
        if dpad > 0:
            q = jnp.pad(
                q, ((0, 0), (0, 0), (0, dpad)), mode="constant", constant_values=0
            )
            k = jnp.pad(
                k, ((0, 0), (0, 0), (0, dpad)), mode="constant", constant_values=0
            )
            v = jnp.pad(
                v, ((0, 0), (0, 0), (0, dpad)), mode="constant", constant_values=0
            )

        out_shape = [totalq, h, d + dpad]
        lse_shape = [b, h, max_seqlen_q]
        # Empty accumulators for non-split mode
        out_accum_shape = [1, 1, 1, 1]
        lse_accum_shape = [1, 1, 1]

        out_types = [
            jax.ShapeDtypeStruct(out_shape, q_dtype),
            jax.ShapeDtypeStruct(lse_shape, jnp.float32),
            jax.ShapeDtypeStruct(out_accum_shape, jnp.float32),
            jax.ShapeDtypeStruct(lse_accum_shape, jnp.float32),
        ]

        # Create empty arrays for optional inputs not used in varlen
        empty_i32 = jnp.array([], dtype=jnp.int32)
        empty_f32 = jnp.array([], dtype=jnp.float32)
        empty_like_q = jnp.zeros((0,), dtype=q_dtype)

        # For varlen: seqused_k is optional, seqlens_q/k are required
        seqused_q_buf = empty_i32  # Not used in varlen
        seqused_k_buf = seqused_k if has_seqused_k else empty_i32

        out, lse, _, _ = jax.ffi.ffi_call(
            "flash_mha_fwd",
            result_shape_dtypes=out_types,
            has_side_effect=False,
            input_layouts=[None] * 21,  # q, k, v, + 18 optional buffers
            output_layouts=[None] * 4,
        )(
            q,
            k,
            v,
            empty_like_q,  # k_new
            empty_like_q,  # v_new
            empty_like_q,  # q_v
            cu_seqlens_q,
            cu_seqlens_k,
            empty_i32,  # cu_seqlens_k_new
            seqused_q_buf,
            seqused_k_buf,
            empty_i32,  # page_table
            empty_i32,  # kv_batch_idx
            empty_i32,  # leftpad_k
            empty_like_q,  # rotary_cos
            empty_like_q,  # rotary_sin
            empty_i32,  # seqlens_rotary
            empty_f32,  # q_descale
            empty_f32,  # k_descale
            empty_f32,  # v_descale
            empty_i32,  # scheduler_metadata
            max_seqlen_q=max_seqlen_q,
            max_seqlen_q_has_value=True,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_k_has_value=True,
            softmax_scale_val=softmax_scale,
            softmax_scale_has_value=True,
            pack_gqa_val=False,
            pack_gqa_has_value=False,
            is_causal=is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            attention_chunk=0,
            softcap=0.0,
            is_rotary_interleaved=False,
            num_splits=1,
            sm_margin=0,
        )

        if dpad > 0:
            out = out[:, :, :d]

        return out, lse

    return mlir.lower_fun(fwd, multiple_results=True)(
        ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_k
    )


mlir.register_lowering(
    _flash_mha_varlen_fwd_p,
    _flash_mha_varlen_fwd_hlo_lowering,  # type: ignore
    platform="gpu",
)


def _flash_mha_varlen_fwd_abstract(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    has_seqused_k,
    softmax_scale=None,
    is_causal=None,
    window_size_left=None,
    window_size_right=None,
):
    q_dtype = dtypes.canonicalize_dtype(q.dtype)
    k_dtype = dtypes.canonicalize_dtype(k.dtype)
    v_dtype = dtypes.canonicalize_dtype(v.dtype)
    [totalq, h, d] = q.shape
    b = cu_seqlens_q.shape[0] - 1
    assert q_dtype == k_dtype and q_dtype == v_dtype
    assert q_dtype in [jnp.bfloat16, jnp.float16]
    assert b >= 1

    out_shape = [totalq, h, d]
    lse_shape = [b, h, max_seqlen_q]

    return (ShapedArray(out_shape, q_dtype), ShapedArray(lse_shape, jnp.float32))


_flash_mha_varlen_fwd_p.def_abstract_eval(_flash_mha_varlen_fwd_abstract)
