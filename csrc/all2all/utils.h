#pragma once
#include <moe_cuda/error.hpp>
inline int get_num_experts(int num_experts) {
    for (auto expert : {32, 64, 128, 256, 512}) {
        if (num_experts == expert) {
            return expert;
        }
    }
    HOST_ERROR("Invalid number of experts");
    return 0;
}

inline int get_token_dim(int token_dim) {
    for (auto token_dim_value : {16, 32, 64, 128, 256, 512, 1024, 2048}) {
        if (token_dim == token_dim_value) {
            return token_dim_value;
        }
    }
    HOST_ERROR("Invalid token dimension");
    return 0;
}

inline int get_num_experts_per_token(int num_experts_per_token) {
    for (auto num_experts_per_token_value : {1, 2, 4, 8, 10, 12, 16, 32}) {
        if (num_experts_per_token == num_experts_per_token_value) {
            return num_experts_per_token_value;
        }
    }
    HOST_ERROR("Invalid number of experts per token");
    return 0;
}

