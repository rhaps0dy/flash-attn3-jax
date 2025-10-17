#include <cuda_runtime_api.h>
#include <cutlass/numeric_types.h>
#include <pybind11/pybind11.h>
#include <stddef.h>

#include <cstdint>

#include "check.h"
#include "flash.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error
mha_fwd_impl(cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t device, ffi::AnyBuffer q,
             ffi::AnyBuffer k, ffi::AnyBuffer v, ffi::AnyBuffer k_new_, ffi::AnyBuffer v_new_,
             ffi::AnyBuffer q_v_, ffi::AnyBuffer cu_seqlens_q_, ffi::AnyBuffer cu_seqlens_k_,
             ffi::AnyBuffer cu_seqlens_k_new_, ffi::AnyBuffer seqused_q_, ffi::AnyBuffer seqused_k_,
             ffi::AnyBuffer page_table_, ffi::AnyBuffer kv_batch_idx_, ffi::AnyBuffer leftpad_k_,
             ffi::AnyBuffer rotary_cos_, ffi::AnyBuffer rotary_sin_, ffi::AnyBuffer seqlens_rotary_,
             ffi::AnyBuffer q_descale_, ffi::AnyBuffer k_descale_, ffi::AnyBuffer v_descale_,
             ffi::AnyBuffer scheduler_metadata_, int64_t max_seqlen_q_val,
             bool max_seqlen_q_has_value, int64_t max_seqlen_k_val, bool max_seqlen_k_has_value,
             double softmax_scale_val, bool softmax_scale_has_value, bool pack_gqa_val,
             bool pack_gqa_has_value, bool is_causal, int64_t window_size_left,
             int64_t window_size_right, int64_t attention_chunk, double softcap,
             bool is_rotary_interleaved, int64_t num_splits, int64_t sm_margin,
             ffi::Result<ffi::AnyBuffer> out, ffi::Result<ffi::AnyBuffer> softmax_lse,
             ffi::Result<ffi::AnyBuffer> out_accum, ffi::Result<ffi::AnyBuffer> softmax_lse_accum);

ffi::Error
mha_bwd_impl(cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t device,
             ffi::AnyBuffer dout, ffi::AnyBuffer q, ffi::AnyBuffer k, ffi::AnyBuffer v,
             ffi::AnyBuffer out, ffi::AnyBuffer softmax_lse, ffi::AnyBuffer cu_seqlens_q_,
             ffi::AnyBuffer cu_seqlens_k_, ffi::AnyBuffer seqused_q_, ffi::AnyBuffer seqused_k_,
             int64_t max_seqlen_q_val, bool max_seqlen_q_has_value, int64_t max_seqlen_k_val,
             bool max_seqlen_k_has_value, double softmax_scale_val, bool softmax_scale_has_value,
             bool is_causal, int64_t window_size_left, int64_t window_size_right, double softcap,
             bool deterministic, int64_t sm_margin, ffi::Result<ffi::AnyBuffer> dq,
             ffi::Result<ffi::AnyBuffer> dk, ffi::Result<ffi::AnyBuffer> dv,
             ffi::Result<ffi::AnyBuffer> softmax_d, ffi::Result<ffi::AnyBuffer> softmax_lse_log2);

ffi::Error mha_combine_impl(cudaStream_t stream, int32_t device, ffi::AnyBuffer out_partial,
                            ffi::AnyBuffer lse_partial, int64_t head_size_og,
                            ffi::Result<ffi::AnyBuffer> out,
                            ffi::Result<ffi::AnyBuffer> softmax_lse);

ffi::Error mha_fwd_get_scheduler_metadata_impl(
    cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t device, int64_t batch_size,
    int64_t max_seqlen_q, int64_t max_seqlen_k, int64_t num_heads, int64_t num_heads_k,
    int64_t headdim, int64_t headdim_v, int32_t qkv_dtype_int, ffi::AnyBuffer seqused_k,
    ffi::AnyBuffer cu_seqlens_q_, ffi::AnyBuffer cu_seqlens_k_, ffi::AnyBuffer cu_seqlens_k_new_,
    ffi::AnyBuffer seqused_q_, ffi::AnyBuffer leftpad_k_, int64_t page_size_val,
    bool page_size_has_value, int64_t max_seqlen_k_new, bool is_causal, int64_t window_size_left,
    int64_t window_size_right, int64_t attention_chunk, bool has_softcap, int64_t num_splits,
    bool pack_gqa_val, bool pack_gqa_has_value, int64_t sm_margin,
    ffi::Result<ffi::AnyBuffer> tile_count_semaphore);

namespace {

template <typename T> pybind11::capsule EncapsulateFfiCall(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be an XLA FFI handler");
  return pybind11::capsule(reinterpret_cast<void *>(fn));
}

XLA_FFI_DEFINE_HANDLER(mha_fwd, mha_fwd_impl,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Ctx<ffi::ScratchAllocator>()
                           .Ctx<ffi::DeviceOrdinal>()
                           .Arg<ffi::AnyBuffer>() // q
                           .Arg<ffi::AnyBuffer>() // k
                           .Arg<ffi::AnyBuffer>() // v
                           .Arg<ffi::AnyBuffer>() // k_new_
                           .Arg<ffi::AnyBuffer>() // v_new_
                           .Arg<ffi::AnyBuffer>() // q_v_
                           .Arg<ffi::AnyBuffer>() // cu_seqlens_q_
                           .Arg<ffi::AnyBuffer>() // cu_seqlens_k_
                           .Arg<ffi::AnyBuffer>() // cu_seqlens_k_new_
                           .Arg<ffi::AnyBuffer>() // seqused_q_
                           .Arg<ffi::AnyBuffer>() // seqused_k_
                           .Arg<ffi::AnyBuffer>() // page_table_
                           .Arg<ffi::AnyBuffer>() // kv_batch_idx_
                           .Arg<ffi::AnyBuffer>() // leftpad_k_
                           .Arg<ffi::AnyBuffer>() // rotary_cos_
                           .Arg<ffi::AnyBuffer>() // rotary_sin_
                           .Arg<ffi::AnyBuffer>() // seqlens_rotary_
                           .Arg<ffi::AnyBuffer>() // q_descale_
                           .Arg<ffi::AnyBuffer>() // k_descale_
                           .Arg<ffi::AnyBuffer>() // v_descale_
                           .Arg<ffi::AnyBuffer>() // scheduler_metadata_
                           .Attr<int64_t>("max_seqlen_q")
                           .Attr<bool>("max_seqlen_q_has_value")
                           .Attr<int64_t>("max_seqlen_k")
                           .Attr<bool>("max_seqlen_k_has_value")
                           .Attr<double>("softmax_scale_val")
                           .Attr<bool>("softmax_scale_has_value")
                           .Attr<bool>("pack_gqa_val")
                           .Attr<bool>("pack_gqa_has_value")
                           .Attr<bool>("is_causal")
                           .Attr<int64_t>("window_size_left")
                           .Attr<int64_t>("window_size_right")
                           .Attr<int64_t>("attention_chunk")
                           .Attr<double>("softcap")
                           .Attr<bool>("is_rotary_interleaved")
                           .Attr<int64_t>("num_splits")
                           .Attr<int64_t>("sm_margin")
                           .Ret<ffi::AnyBuffer>() // out
                           .Ret<ffi::AnyBuffer>() // softmax_lse
                           .Ret<ffi::AnyBuffer>() // out_accum
                           .Ret<ffi::AnyBuffer>() // softmax_lse_accum
);

XLA_FFI_DEFINE_HANDLER(mha_bwd, mha_bwd_impl,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Ctx<ffi::ScratchAllocator>()
                           .Ctx<ffi::DeviceOrdinal>()
                           .Arg<ffi::AnyBuffer>() // dout
                           .Arg<ffi::AnyBuffer>() // q
                           .Arg<ffi::AnyBuffer>() // k
                           .Arg<ffi::AnyBuffer>() // v
                           .Arg<ffi::AnyBuffer>() // out
                           .Arg<ffi::AnyBuffer>() // softmax_lse
                           .Arg<ffi::AnyBuffer>() // cu_seqlens_q_
                           .Arg<ffi::AnyBuffer>() // cu_seqlens_k_
                           .Arg<ffi::AnyBuffer>() // seqused_q_
                           .Arg<ffi::AnyBuffer>() // seqused_k_
                           .Attr<int64_t>("max_seqlen_q_val")
                           .Attr<bool>("max_seqlen_q_has_value")
                           .Attr<int64_t>("max_seqlen_k_val")
                           .Attr<bool>("max_seqlen_k_has_value")
                           .Attr<double>("softmax_scale_val")
                           .Attr<bool>("softmax_scale_has_value")
                           .Attr<bool>("is_causal")
                           .Attr<int64_t>("window_size_left")
                           .Attr<int64_t>("window_size_right")
                           .Attr<double>("softcap")
                           .Attr<bool>("deterministic")
                           .Attr<int64_t>("sm_margin")
                           .Ret<ffi::AnyBuffer>() // dq
                           .Ret<ffi::AnyBuffer>() // dk
                           .Ret<ffi::AnyBuffer>() // dv
                           .Ret<ffi::AnyBuffer>() // softmax_d
                           .Ret<ffi::AnyBuffer>() // softmax_lse_log2
);

XLA_FFI_DEFINE_HANDLER(mha_combine, mha_combine_impl,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Ctx<ffi::DeviceOrdinal>()
                           .Arg<ffi::AnyBuffer>() // out_partial
                           .Arg<ffi::AnyBuffer>() // lse_partial
                           .Attr<int64_t>("head_size_og")
                           .Ret<ffi::AnyBuffer>() // out
                           .Ret<ffi::AnyBuffer>() // softmax_lse
);

XLA_FFI_DEFINE_HANDLER(mha_fwd_get_scheduler_metadata, mha_fwd_get_scheduler_metadata_impl,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Ctx<ffi::ScratchAllocator>()
                           .Ctx<ffi::DeviceOrdinal>()
                           .Attr<int64_t>("batch_size")
                           .Attr<int64_t>("max_seqlen_q")
                           .Attr<int64_t>("max_seqlen_k")
                           .Attr<int64_t>("num_heads")
                           .Attr<int64_t>("num_heads_k")
                           .Attr<int64_t>("headdim")
                           .Attr<int64_t>("headdim_v")
                           .Attr<int32_t>("qkv_dtype_int")
                           .Arg<ffi::AnyBuffer>() // seqused_k
                           .Arg<ffi::AnyBuffer>() // cu_seqlens_q_
                           .Arg<ffi::AnyBuffer>() // cu_seqlens_k_
                           .Arg<ffi::AnyBuffer>() // cu_seqlens_k_new_
                           .Arg<ffi::AnyBuffer>() // seqused_q_
                           .Arg<ffi::AnyBuffer>() // leftpad_k_
                           .Attr<int64_t>("page_size_val")
                           .Attr<bool>("page_size_has_value")
                           .Attr<int64_t>("max_seqlen_k_new")
                           .Attr<bool>("is_causal")
                           .Attr<int64_t>("window_size_left")
                           .Attr<int64_t>("window_size_right")
                           .Attr<int64_t>("attention_chunk")
                           .Attr<bool>("has_softcap")
                           .Attr<int64_t>("num_splits")
                           .Attr<bool>("pack_gqa_val")
                           .Attr<bool>("pack_gqa_has_value")
                           .Attr<int64_t>("sm_margin")
                           .Ret<ffi::AnyBuffer>() // tile_count_semaphore
);

pybind11::dict FFIRegistrations() {
  pybind11::dict dict;
  dict["flash_mha_fwd"] = EncapsulateFfiCall(mha_fwd);
  dict["flash_mha_bwd"] = EncapsulateFfiCall(mha_bwd);
  dict["flash_mha_combine"] = EncapsulateFfiCall(mha_combine);
  dict["flash_mha_fwd_get_scheduler_metadata"] = EncapsulateFfiCall(mha_fwd_get_scheduler_metadata);
  return dict;
}

PYBIND11_MODULE(flash_api, m) {
  m.doc() = "XLA FFI bindings for FlashAttention";
  m.def("get_ffi_registrations", &FFIRegistrations);
}

} // namespace
