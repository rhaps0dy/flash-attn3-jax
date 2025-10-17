import math
from functools import partial

import jax
import jax._src.dispatch
import jax.numpy as jnp
from jax import dtypes
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import mlir, xla

# ==== Register primitives ====

_flash_mha_varlen_bwd_p = Primitive("flash_mha_varlen_bwd")
_flash_mha_varlen_bwd_p.multiple_results = True
_flash_mha_varlen_bwd_p.def_impl(partial(xla.apply_primitive, _flash_mha_varlen_bwd_p))
jax._src.dispatch.prim_requires_devices_during_lowering.add(_flash_mha_varlen_bwd_p)


def flash_mha_varlen_bwd(
    dout,
    q,
    k,
    v,
    o,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: int = -1,
    max_seqlen_k: int = -1,
    softmax_scale: float | None = None,
    is_causal: bool = False,
    window_size: tuple = (-1, -1),
    deterministic: bool = False,
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
    kwargs = dict(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        deterministic=deterministic,
    )
    return tuple(
        _flash_mha_varlen_bwd_p.bind(
            dout, q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, **kwargs
        )
    )


# ==== HLO lowering ====


def _flash_mha_varlen_bwd_hlo_lowering(
    ctx,
    dout,
    q,
    k,
    v,
    o,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
):
    def bwd(dout, q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k):
        q_dtype = dtypes.canonicalize_dtype(q.dtype)
        k_dtype = dtypes.canonicalize_dtype(k.dtype)
        v_dtype = dtypes.canonicalize_dtype(v.dtype)
        [totalq, h, d] = q.shape
        [totalk, hk, dk] = k.shape
        b = cu_seqlens_q.shape[0] - 1
        assert q_dtype == k_dtype and q_dtype == v_dtype
        assert q_dtype in [jnp.bfloat16, jnp.float16]
        assert b >= 1
        assert d == dk, "q and k must have the same head size."

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
            dout = jnp.pad(
                dout, ((0, 0), (0, 0), (0, dpad)), mode="constant", constant_values=0
            )
            o = jnp.pad(
                o, ((0, 0), (0, 0), (0, dpad)), mode="constant", constant_values=0
            )

        dq_shape = [totalq, h, d + dpad]
        dk_shape = [totalk, hk, d + dpad]
        dv_shape = [totalk, hk, d + dpad]

        # kBlockM is typically 128 for most architectures/dimensions
        # This ensures proper alignment for each sequence in the varlen batch
        kBlockM = 128
        total_q_padded_rounded = (
            (totalq + b * kBlockM + kBlockM - 1) // kBlockM
        ) * kBlockM

        out_types = [
            jax.ShapeDtypeStruct(dq_shape, q_dtype),
            jax.ShapeDtypeStruct(dk_shape, k_dtype),
            jax.ShapeDtypeStruct(dv_shape, v_dtype),
            jax.ShapeDtypeStruct([h, total_q_padded_rounded], jnp.float32),  # softmax_d
            jax.ShapeDtypeStruct(
                [h, total_q_padded_rounded], jnp.float32
            ),  # softmax_lse_log2
        ]

        # Create empty arrays for optional inputs not used in varlen
        empty_i32 = jnp.array([], dtype=jnp.int32)

        dq, dk, dv, _, _ = jax.ffi.ffi_call(
            "flash_mha_bwd",
            result_shape_dtypes=out_types,
            has_side_effect=False,
            input_layouts=[None] * 10,
            output_layouts=[None] * 5,
        )(
            dout,
            q,
            k,
            v,
            o,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            empty_i32,  # seqused_q
            empty_i32,  # seqused_k
            max_seqlen_q_val=max_seqlen_q,
            max_seqlen_q_has_value=True,
            max_seqlen_k_val=max_seqlen_k,
            max_seqlen_k_has_value=True,
            softmax_scale_val=softmax_scale,
            softmax_scale_has_value=True,
            is_causal=is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            softcap=0.0,
            deterministic=deterministic,
            sm_margin=0,
        )

        if dpad > 0:
            dq = dq[:, :, :d]
            dk = dk[:, :, :d]
            dv = dv[:, :, :d]

        return dq, dk, dv

    return mlir.lower_fun(bwd, multiple_results=True)(
        ctx, dout, q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k
    )


mlir.register_lowering(
    _flash_mha_varlen_bwd_p,
    _flash_mha_varlen_bwd_hlo_lowering,  # type: ignore
    platform="gpu",
)


def _flash_mha_varlen_bwd_abstract(
    dout,
    q,
    k,
    v,
    o,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
):
    q_dtype = dtypes.canonicalize_dtype(q.dtype)
    k_dtype = dtypes.canonicalize_dtype(k.dtype)
    v_dtype = dtypes.canonicalize_dtype(v.dtype)
    [totalq, h, d] = q.shape
    b = cu_seqlens_q.shape[0] - 1
    assert q_dtype == k_dtype and q_dtype == v_dtype
    assert q_dtype in [jnp.bfloat16, jnp.float16]
    assert b >= 1

    dq_shape = q.shape
    dk_shape = k.shape
    dv_shape = v.shape

    return (
        ShapedArray(dq_shape, q_dtype),
        ShapedArray(dk_shape, k_dtype),
        ShapedArray(dv_shape, v_dtype),
    )


_flash_mha_varlen_bwd_p.def_abstract_eval(_flash_mha_varlen_bwd_abstract)
