from functools import partial

import jax
from jax.experimental.custom_partitioning import (
    ArrayMapping,
    CompoundFactor,
    SdyShardingRule,
    custom_partitioning,
)
from jax.interpreters.mlir import ir
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .flash_hlo import _flash_mha_bwd_hlo, _flash_mha_fwd_hlo
from .ring_attention import ring_bwd, ring_fwd

_flash_mha_fwd_hlo_sharded = custom_partitioning(
    _flash_mha_fwd_hlo, static_argnums=(3, 4, 5)
)
_flash_mha_bwd_hlo_sharded = custom_partitioning(
    _flash_mha_bwd_hlo, static_argnums=(6, 7, 8)
)


def is_replicated(sharding):
    if isinstance(sharding, NamedSharding):
        return sharding.is_fully_replicated
    raise ValueError(f"Unsupported sharding type: {type(sharding)}")


def partition_fwd(
    softmax_scale,
    is_causal,
    window_size,
    mesh: Mesh,
    arg_shapes: list[jax.ShapeDtypeStruct],
    result_shape: list[jax.ShapeDtypeStruct],
):
    result_shardings = tuple([x.sharding for x in result_shape])
    arg_shardings = tuple([x.sharding for x in arg_shapes])

    q_sharding = arg_shardings[0]
    k_sharding = arg_shardings[1]
    v_sharding = arg_shardings[2]
    assert q_sharding == k_sharding and q_sharding == v_sharding, (
        "Only support q, k, v sharing the same sharding."
    )
    if is_replicated(q_sharding):
        result_shardings = (q_sharding, q_sharding)
    elif isinstance(q_sharding, NamedSharding):
        mesh = q_sharding.mesh
        [n, l, h, d] = q_sharding.spec
        assert d is None, (
            "Sharding across `d` won't be efficient, so it's not supported."
        )
        if l is not None:
            assert window_size == (-1, -1), (
                "Ring attention doesn't support local masking yet."
            )
            result_shardings = q_sharding, NamedSharding(mesh, P(n, h, l))
            arg_shardings = q_sharding, q_sharding, q_sharding
            axis_name = l
            axis_size = mesh.shape[axis_name]
            return (
                mesh,
                partial(
                    ring_fwd,
                    softmax_scale=softmax_scale,
                    is_causal=is_causal,
                    axis_name=axis_name,
                    axis_size=axis_size,
                    mha_fwd=_flash_mha_fwd_hlo,
                ),
                result_shardings,
                arg_shardings,
            )
        else:
            result_shardings = q_sharding, NamedSharding(mesh, P(n, h, l))
            arg_shardings = q_sharding, q_sharding, q_sharding

    def fwd(q, k, v):
        return _flash_mha_fwd_hlo(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            window_size=window_size,
        )

    return mesh, fwd, result_shardings, arg_shardings


def infer_sharding_fwd(
    softmax_scale,
    is_causal,
    window_size,
    mesh: Mesh,
    arg_shapes: list[jax.ShapeDtypeStruct],
    result_shape: list[jax.ShapeDtypeStruct],
):
    arg_shardings = jax.tree_util.tree_map(lambda x: x.sharding, arg_shapes)
    q_sharding = arg_shardings[0]
    k_sharding = arg_shardings[1]
    v_sharding = arg_shardings[2]
    assert q_sharding == k_sharding and q_sharding == v_sharding, (
        "Only support q, k, v sharing the same sharding."
    )
    if is_replicated(q_sharding):
        result_sharding = (q_sharding, q_sharding)
    elif isinstance(q_sharding, NamedSharding):
        [n, l, h, d] = q_sharding.spec
        result_sharding = (q_sharding, NamedSharding(q_sharding.mesh, P(n, h, l)))
    else:
        raise ValueError("Unsupported sharding type.", type(q_sharding))
    return result_sharding


def sharding_rule_fwd(
    softmax_scale,
    is_causal,
    window_size,
    mesh: Mesh,
    arg_shapes: list[ir.RankedTensorType],
    result_shape: list[ir.RankedTensorType],
):
    q_shape, k_shape, v_shape = arg_shapes
    group_size = q_shape.shape[-2] // k_shape.shape[-2]
    if group_size > 1:
        return SdyShardingRule(
            operand_mappings=(
                ArrayMapping("n", "l", CompoundFactor("g", "h"), "d"),
                ArrayMapping("n", "L", "h", "d"),
                ArrayMapping("n", "L", "h", "D"),
            ),
            result_mappings=(
                ArrayMapping("n", "l", CompoundFactor("g", "h"), "D"),
                ArrayMapping("n", CompoundFactor("g", "h"), "l"),
            ),
            g=group_size,
        )
    else:
        return SdyShardingRule(
            operand_mappings=(
                ArrayMapping("n", "l", "h", "d"),
                ArrayMapping("n", "L", "h", "d"),
                ArrayMapping("n", "L", "h", "D"),
            ),
            result_mappings=(
                ArrayMapping("n", "l", "h", "D"),
                ArrayMapping("n", "h", "l"),
            ),
        )


_flash_mha_fwd_hlo_sharded.def_partition(
    infer_sharding_from_operands=infer_sharding_fwd,
    partition=partition_fwd,
    sharding_rule=sharding_rule_fwd,  #'n l (g h) d, n L h d, n L h D -> n l (g h) D, n (g h) l'
)


def infer_sharding_bwd(
    softmax_scale, is_causal, window_size, mesh, arg_shapes, result_shape
):
    # args: dout, q, k, v, out, lse
    # outs: dq, dk, dv
    arg_shardings = jax.tree_util.tree_map(lambda x: x.sharding, arg_shapes)
    q_sharding = arg_shardings[1]
    k_sharding = arg_shardings[2]
    v_sharding = arg_shardings[3]
    return q_sharding, k_sharding, v_sharding


def partition_bwd(
    softmax_scale, is_causal, window_size, mesh, arg_shapes, result_shape
):
    result_shardings = jax.tree_util.tree_map(lambda x: x.sharding, result_shape)
    arg_shardings = jax.tree_util.tree_map(lambda x: x.sharding, arg_shapes)

    arg_shardings[0]
    q_sharding = arg_shardings[1]
    k_sharding = arg_shardings[2]
    v_sharding = arg_shardings[3]
    arg_shardings[4]
    lse_sharding = arg_shardings[5]
    assert q_sharding == k_sharding and q_sharding == v_sharding, (
        "Only support q, k, v sharing the same sharding."
    )
    if is_replicated(q_sharding):
        result_shardings = (q_sharding,) * 3
    elif isinstance(q_sharding, NamedSharding):
        mesh = q_sharding.mesh
        [n, l, h, d] = q_sharding.spec
        assert d is None, (
            "Sharding across `d` won't be efficient, so it's not supported."
        )
        if l is not None:
            assert window_size == (-1, -1), (
                "Ring attention doesn't support local masking yet."
            )
            result_shardings = q_sharding, q_sharding, q_sharding
            lse_sharding = NamedSharding(mesh, P(n, h, l))
            arg_shardings = (q_sharding,) * 5 + (lse_sharding,)
            axis_name = l
            axis_size = mesh.shape[axis_name]
            return (
                mesh,
                partial(
                    ring_bwd,
                    softmax_scale=softmax_scale,
                    is_causal=is_causal,
                    axis_name=axis_name,
                    axis_size=axis_size,
                    mha_bwd=_flash_mha_bwd_hlo,
                ),
                result_shardings,
                arg_shardings,
            )
        else:
            result_shardings = q_sharding, q_sharding, q_sharding
            lse_sharding = NamedSharding(mesh, P(n, h, l))
            arg_shardings = (q_sharding,) * 5 + (lse_sharding,)

    def fwd(*args):
        return _flash_mha_bwd_hlo(
            *args,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            window_size=window_size,
        )

    return mesh, fwd, result_shardings, arg_shardings


def sharding_rule_bwd(
    softmax_scale, is_causal, window_size, mesh, arg_shapes, result_shape
):
    do_shape, q_shape, k_shape, v_shape, o_shape, lse_shape = arg_shapes
    group_size = q_shape.shape[-2] // k_shape.shape[-2]
    if group_size > 1:
        return SdyShardingRule(
            operand_mappings=(
                ArrayMapping("n", "l", CompoundFactor("g", "h"), "d"),
                ArrayMapping("n", "l", CompoundFactor("g", "h"), "d"),
                ArrayMapping("n", "L", "h", "d"),
                ArrayMapping("n", "L", "h", "D"),
                ArrayMapping("n", "l", CompoundFactor("g", "h"), "D"),
                ArrayMapping("n", CompoundFactor("g", "h"), "l"),
            ),
            result_mappings=(
                ArrayMapping("n", "l", CompoundFactor("g", "h"), "d"),
                ArrayMapping("n", "L", "h", "d"),
                ArrayMapping("n", "L", "h", "D"),
            ),
            g=group_size,
        )
    else:
        return SdyShardingRule(
            operand_mappings=(
                ArrayMapping("n", "l", "h", "d"),
                ArrayMapping("n", "l", "h", "d"),
                ArrayMapping("n", "L", "h", "d"),
                ArrayMapping("n", "L", "h", "D"),
                ArrayMapping("n", "l", "h", "D"),
                ArrayMapping("n", "h", "l"),
            ),
            result_mappings=(
                ArrayMapping("n", "l", "h", "d"),
                ArrayMapping("n", "L", "h", "d"),
                ArrayMapping("n", "L", "h", "D"),
            ),
        )


_flash_mha_bwd_hlo_sharded.def_partition(
    infer_sharding_from_operands=infer_sharding_bwd,
    partition=partition_bwd,
    sharding_rule=sharding_rule_bwd,  #'n l (g h) D, n l (g h) d, n L h d, n L h D, n l (g h) D, n (g h) l -> n l (g h) d, n L h d, n L h D'
)
