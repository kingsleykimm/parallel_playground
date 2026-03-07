#pragma once
// Torch dtype helpers kept for backward-compatible call sites.

#include <moe_cuda/dtype.h>
#include <c10/core/ScalarType.h>

inline c10::ScalarType from_torch_type(c10::ScalarType type) {
    return type;
}

inline c10::ScalarType to_torch_type(c10::ScalarType type) {
    return type;
}
