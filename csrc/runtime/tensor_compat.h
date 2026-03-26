#pragma once

#include <moe_cuda/dtype.h>
#include <moe_cuda/error.hpp>
#include <moe_cuda/types.h>
#include <runtime/dtype_torch.h>
#include <runtime/utils.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/nn/utils/rnn.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace custom {

using Tensor = at::Tensor;

inline c10::Device to_c10_device(int dev) {
  return dev < 0 ? c10::kCPU : c10::Device(c10::kCUDA, dev);
}

inline int to_int_device(const at::Tensor &t) {
  if (t.is_cpu()) {
    return -1;
  }
  return static_cast<int>(t.get_device());
}

inline c10::ScalarType dtype_of(const at::Tensor &t) { return t.scalar_type(); }

inline bool is_mn_major(const at::Tensor &t) {
  return t.dim() >= 2 && t.stride(-2) == 1;
}

inline Major major_of(const at::Tensor &t) {
  return is_mn_major(t) ? Major::MN : Major::K;
}

inline std::vector<int64_t> to_i64_shape(const std::vector<size_t> &shape) {
  return std::vector<int64_t>(shape.begin(), shape.end());
}

inline std::vector<int64_t> to_i64_shape(const c10::IntArrayRef &shape) {
  return std::vector<int64_t>(shape.begin(), shape.end());
}

inline at::TensorOptions make_options(c10::ScalarType dtype, int device) {
  return torch::TensorOptions().dtype(dtype).device(to_c10_device(device));
}

template <typename Fn>
auto with_optional_stream(int device, cudaStream_t stream, Fn &&fn) {
  if (stream != nullptr && device >= 0) {
    c10::cuda::CUDAStreamGuard guard(
        c10::cuda::getStreamFromExternal(stream, device));
    return fn();
  }
  return fn();
}

inline std::string to_string(const at::Tensor &t, int n, int s = 0,
                             int e = -1) {
  (void)e;
  if (!t.defined()) {
    return "Tensor(undefined)";
  }

  std::ostringstream stream;
  stream << "Tensor(shape=" << c10::str(t.sizes())
         << ", dtype=" << type_to_string(dtype_of(t)) << ")\n";

  auto flat = t.flatten();
  const int64_t numel = flat.numel();
  const int64_t start = std::max<int64_t>(0, s);
  const int64_t count =
      std::max<int64_t>(0, std::min<int64_t>(n, numel - start));
  auto preview = flat.narrow(0, start, count).cpu();
  stream << preview;
  return stream.str();
}

inline bool tensor_is_finite(const at::Tensor &t) {
  return torch::all(torch::isfinite(t)).item<bool>();
}

template <typename T> inline std::vector<T> tolist(const at::Tensor &t) {
  auto flat = t.flatten().contiguous().cpu();
  std::vector<T> out(static_cast<size_t>(flat.numel()));
  if (!out.empty()) {
    const T *ptr = flat.data_ptr<T>();
    std::copy(ptr, ptr + out.size(), out.begin());
  }
  return out;
}

template <typename T> constexpr c10::ScalarType cpp_type_to_dtype() {
  if constexpr (std::is_same_v<T, float>) {
    return c10::ScalarType::Float;
  } else if constexpr (std::is_same_v<T, __half>) {
    return c10::ScalarType::Half;
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return c10::ScalarType::BFloat16;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return c10::ScalarType::Long;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return c10::ScalarType::Int;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return c10::ScalarType::Short;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return c10::ScalarType::Char;
  } else if constexpr (std::is_same_v<T, bool>) {
    return c10::ScalarType::Bool;
  } else {
    HOST_ERROR("tensor_compat.h: unsupported C++ type");
  }
}

inline Tensor empty(const std::vector<size_t> &shape, c10::ScalarType dtype,
                    int device, cudaStream_t stream = nullptr,
                    bool blocking = false) {
  auto out = with_optional_stream(device, stream, [&]() {
    return torch::empty(to_i64_shape(shape), make_options(dtype, device));
  });
  if (blocking && stream != nullptr && device >= 0) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  return out;
}

inline Tensor empty_like(const Tensor &self,
                         std::optional<c10::ScalarType> dtype = std::nullopt,
                         cudaStream_t stream = nullptr, bool blocking = false) {
  auto opts = dtype.has_value() ? self.options().dtype(*dtype) : self.options();
  auto out = with_optional_stream(to_int_device(self), stream, [&]() {
    return torch::empty_like(self, opts);
  });
  if (blocking && stream != nullptr && to_int_device(self) >= 0) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  return out;
}

inline Tensor zeros(const std::vector<size_t> &shape, c10::ScalarType dtype,
                    int device, cudaStream_t stream = nullptr,
                    bool blocking = false) {
  auto out = with_optional_stream(device, stream, [&]() {
    return torch::zeros(to_i64_shape(shape), make_options(dtype, device));
  });
  if (blocking && stream != nullptr && device >= 0) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  return out;
}

inline Tensor zeros_like(const Tensor &self,
                         std::optional<c10::ScalarType> dtype = std::nullopt,
                         cudaStream_t stream = nullptr, bool blocking = false) {
  auto opts = dtype.has_value() ? self.options().dtype(*dtype) : self.options();
  auto out = with_optional_stream(to_int_device(self), stream, [&]() {
    return torch::zeros_like(self, opts);
  });
  if (blocking && stream != nullptr && to_int_device(self) >= 0) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  return out;
}

inline Tensor ones(const std::vector<size_t> &shape, c10::ScalarType dtype,
                   int device, cudaStream_t stream = nullptr,
                   bool blocking = false) {
  auto out = with_optional_stream(device, stream, [&]() {
    return torch::ones(to_i64_shape(shape), make_options(dtype, device));
  });
  if (blocking && stream != nullptr && device >= 0) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  return out;
}

inline Tensor ones_like(const Tensor &self,
                        std::optional<c10::ScalarType> dtype = std::nullopt,
                        cudaStream_t stream = nullptr, bool blocking = false) {
  auto opts = dtype.has_value() ? self.options().dtype(*dtype) : self.options();
  auto out = with_optional_stream(to_int_device(self), stream, [&]() {
    return torch::ones_like(self, opts);
  });
  if (blocking && stream != nullptr && to_int_device(self) >= 0) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  return out;
}

template <typename T>
Tensor full(const std::vector<size_t> &shape, T fill_value, int device) {
  const auto dtype = cpp_type_to_dtype<T>();
  return torch::full(to_i64_shape(shape), fill_value,
                     make_options(dtype, device));
}

template <typename T> Tensor full_like(const Tensor &self, T fill_value) {
  const auto dtype = cpp_type_to_dtype<T>();
  return torch::full(to_i64_shape(self.sizes()), fill_value,
                     make_options(dtype, to_int_device(self)));
}

inline Tensor view(const Tensor &self, const std::vector<size_t> new_shape) {
  return self.view(to_i64_shape(new_shape));
}

inline Tensor deepcopy(Tensor &self, cudaStream_t stream = nullptr,
                       bool blocking = false) {
  auto out = with_optional_stream(to_int_device(self), stream,
                                  [&]() { return self.clone(); });
  if (blocking && stream != nullptr && to_int_device(self) >= 0) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  return out;
}

inline Tensor shallow_copy(Tensor &self) { return self; }

inline Tensor from_torch_tensor(torch::Tensor tensor, bool keep_alive = false) {
  (void)keep_alive;
  return tensor;
}

inline Tensor concat(const std::vector<Tensor *> &tensors, int dim,
                     cudaStream_t stream = nullptr) {
  HOST_ASSERT(!tensors.empty(),
              "tensor_compat.h: concat requires non-empty tensor list");
  std::vector<Tensor> copied;
  copied.reserve(tensors.size());
  for (const auto *t : tensors) {
    HOST_ASSERT(t != nullptr,
                "tensor_compat.h: concat does not accept null tensors");
    copied.push_back(*t);
  }
  const int device = to_int_device(copied.front());
  return with_optional_stream(device, stream,
                              [&]() { return torch::cat(copied, dim); });
}

inline Tensor concat(const std::vector<Tensor *> &tensors, int dim,
                     StreamPool &streams) {
  cudaStream_t stream = nullptr;
  int stream_id = streams.fetchStream(stream);
  auto out = concat(tensors, dim, stream);
  streams.returnStream(stream_id);
  return out;
}

inline Tensor concat(const std::vector<Tensor> &tensors, int dim,
                     cudaStream_t stream = nullptr) {
  HOST_ASSERT(!tensors.empty(),
              "tensor_compat.h: concat requires non-empty tensor list");
  const int device = to_int_device(tensors.front());
  return with_optional_stream(device, stream,
                              [&]() { return torch::cat(tensors, dim); });
}

inline Tensor stack(const std::vector<Tensor *> &tensors, int dim,
                    cudaStream_t stream = nullptr) {
  HOST_ASSERT(!tensors.empty(),
              "tensor_compat.h: stack requires non-empty tensor list");
  std::vector<Tensor> copied;
  copied.reserve(tensors.size());
  for (const auto *t : tensors) {
    HOST_ASSERT(t != nullptr,
                "tensor_compat.h: stack does not accept null tensors");
    copied.push_back(*t);
  }
  const int device = to_int_device(copied.front());
  return with_optional_stream(device, stream,
                              [&]() { return torch::stack(copied, dim); });
}

inline Tensor stack(const std::vector<Tensor *> &tensors, int dim,
                    StreamPool &streams) {
  cudaStream_t stream = nullptr;
  int stream_id = streams.fetchStream(stream);
  auto out = stack(tensors, dim, stream);
  streams.returnStream(stream_id);
  return out;
}

inline std::vector<Tensor> split(const Tensor &self, size_t split_size, int dim,
                                 cudaStream_t stream = nullptr,
                                 bool blocking = false) {
  auto out = with_optional_stream(to_int_device(self), stream, [&]() {
    return self.split(static_cast<int64_t>(split_size), dim);
  });
  if (blocking && stream != nullptr && to_int_device(self) >= 0) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  return out;
}

inline std::vector<Tensor> split(const Tensor &self,
                                 const std::vector<size_t> &split_sizes,
                                 int dim, cudaStream_t stream = nullptr,
                                 bool blocking = false) {
  std::vector<int64_t> sizes(split_sizes.begin(), split_sizes.end());
  auto out = with_optional_stream(to_int_device(self), stream, [&]() {
    return torch::split_with_sizes(self, sizes, dim);
  });
  if (blocking && stream != nullptr && to_int_device(self) >= 0) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  return out;
}

inline Tensor repeat(Tensor &self, int dim, size_t num_repeats,
                     cudaStream_t stream = nullptr, bool blocking = false) {
  const int64_t ndim = self.dim();
  if (dim < 0) {
    dim += static_cast<int>(ndim);
  }
  HOST_ASSERT(dim >= 0 && dim < ndim,
              "tensor_compat.h: repeat dim out of bounds");
  std::vector<int64_t> repeats(static_cast<size_t>(ndim), 1);
  repeats[dim] = static_cast<int64_t>(num_repeats);
  auto out = with_optional_stream(to_int_device(self), stream,
                                  [&]() { return self.repeat(repeats); });
  if (blocking && stream != nullptr && to_int_device(self) >= 0) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  return out;
}

inline Tensor repeat_interleave(Tensor &self, int dim, size_t num_repeats,
                                cudaStream_t stream = nullptr,
                                bool blocking = false) {
  const int64_t ndim = self.dim();
  if (dim < 0) {
    dim += static_cast<int>(ndim);
  }
  HOST_ASSERT(dim >= 0 && dim < ndim,
              "tensor_compat.h: repeat_interleave dim out of bounds");
  auto out = with_optional_stream(to_int_device(self), stream, [&]() {
    return torch::repeat_interleave(self, static_cast<int64_t>(num_repeats),
                                    dim);
  });
  if (blocking && stream != nullptr && to_int_device(self) >= 0) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  return out;
}

inline Tensor squeeze(Tensor other) { return other.squeeze(); }

inline void unsqueeze(Tensor &other, int dim) { other = other.unsqueeze(dim); }

inline void broadcast_to(Tensor &self, const Tensor &reference) {
  self = self.expand_as(reference);
}

template <typename T>
Tensor pad_sequence(std::vector<Tensor *> sequences, bool batch_first,
                    T padding_value, std::string padding_side,
                    StreamPool &streams) {
  (void)streams;
  HOST_ASSERT(!sequences.empty(),
              "tensor_compat.h: pad_sequence expects non-empty input");

  std::vector<Tensor> seq;
  seq.reserve(sequences.size());
  for (auto *t : sequences) {
    HOST_ASSERT(t != nullptr,
                "tensor_compat.h: pad_sequence got a null tensor");
    HOST_ASSERT(t->dim() <= 1,
                "tensor_compat.h: pad_sequence supports 1D tensors only");
    seq.push_back(*t);
  }

  if (padding_side == "right") {
    return torch::nn::utils::rnn::pad_sequence(seq, batch_first, padding_value);
  }

  HOST_ASSERT(padding_side == "left",
              "tensor_compat.h: padding_side must be left or right");
  int64_t max_len = 0;
  for (const auto &t : seq) {
    max_len = std::max<int64_t>(max_len, t.size(0));
  }

  std::vector<int64_t> shape =
      batch_first
          ? std::vector<int64_t>{static_cast<int64_t>(seq.size()), max_len}
          : std::vector<int64_t>{max_len, static_cast<int64_t>(seq.size())};

  auto out = torch::full(shape, padding_value, seq.front().options());
  for (size_t i = 0; i < seq.size(); ++i) {
    const auto len = seq[i].size(0);
    const auto start = max_len - len;
    if (batch_first) {
      out[i].narrow(0, start, len).copy_(seq[i]);
    } else {
      out.narrow(1, static_cast<int64_t>(i), 1)
          .squeeze(1)
          .narrow(0, start, len)
          .copy_(seq[i]);
    }
  }
  return out;
}

} // namespace custom

// Transitional global forwards for legacy call sites that used unqualified
// helpers.
inline c10::Device to_c10_device(int dev) { return custom::to_c10_device(dev); }

inline int to_int_device(const at::Tensor &t) {
  return custom::to_int_device(t);
}

inline c10::ScalarType dtype_of(const at::Tensor &t) {
  return custom::dtype_of(t);
}

inline bool is_mn_major(const at::Tensor &t) { return custom::is_mn_major(t); }

inline Major major_of(const at::Tensor &t) { return custom::major_of(t); }

inline bool tensor_is_finite(const at::Tensor &t) {
  return custom::tensor_is_finite(t);
}
