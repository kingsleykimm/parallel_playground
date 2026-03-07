#pragma once

#include <moe_cuda/dtype.h>
#include <runtime/tensor_compat.h>
#include <moe_cuda/types.h>

#include <optional>
#include <string>

namespace custom {

struct Weight {
    at::Tensor storage;
    std::optional<at::Tensor> scale_inv;
    std::string name;
    Major major = Major::K;
    ScaleFactor sf_type = ScaleFactor::None;

    Weight() = default;
    explicit Weight(at::Tensor init) : storage(std::move(init)) {}
    Weight(std::string weight_name, at::Tensor init)
        : storage(std::move(init)), name(std::move(weight_name)) {}
    Weight(at::Tensor init, at::Tensor scale_inv_tensor)
        : storage(std::move(init)), scale_inv(std::move(scale_inv_tensor)) {}
    Weight(std::string weight_name, at::Tensor init, at::Tensor scale_inv_tensor)
        : storage(std::move(init)), scale_inv(std::move(scale_inv_tensor)), name(std::move(weight_name)) {}

    int device() const;
    c10::ScalarType dtype() const;
    void to(int dev);
    void print();
};

}  // namespace custom
