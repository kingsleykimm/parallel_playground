#include "../impls/sm90_layout.hpp"
#include <moe_cuda/kernels/common/sm90_utils.cuh>
#include <runtime/tensor.h>

// transpose any sf layouts - this was moved out of common.hpp
inline at::Tensor transform_sf_layout_mn(at::Tensor &sf_tensor,
                                         cudaStream_t &stream) {
  // ensure that the tensor is mn_major, if not
  if (major_of(sf_tensor) != Major::MN) {
    sf_tensor.dim() == 2 ? custom::unsqueeze(sf_tensor, 0) : void();
    size_t num_groups = sf_tensor.size(0);
    size_t mn = sf_tensor.size(1);
    size_t k = sf_tensor.size(2);
    size_t tma_aligned_mn = ti_align(mn, 16 / sizeof(float));

    at::Tensor transposed =
        custom::empty({num_groups, tma_aligned_mn, k}, dtype_of(sf_tensor),
                      to_int_device(sf_tensor), stream);
    auto strides = custom::to_i64_shape(transposed.strides());
    strides[1] = 1;
    strides[2] = static_cast<int64_t>(tma_aligned_mn);
    transposed = transposed.as_strided(transposed.sizes(), strides);
    sm90_transpose_sf(sf_tensor, transposed, stream);
    return std::move(transposed);
  } else {
    return std::move(sf_tensor);
  }
}
