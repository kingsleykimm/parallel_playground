#pragma once
// Torch dtype helpers kept for backward-compatible call sites.

#include <torch/headeronly/core/ScalarType.h>
#include <moe_cuda/dtype.h>

inline c10::ScalarType from_torch_type(c10::ScalarType type) { return type; }

inline c10::ScalarType to_torch_type(c10::ScalarType type) { return type; }
