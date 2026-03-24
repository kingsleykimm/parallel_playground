/**
    @file : smaller file to benchmark naive moe forward pass as well as optimizations on top
 */
#include <cuda_runtime.h>
#include <runtime/parallel.h>
#include <kernels/internal_api.hpp>
#include <all2all/all2all_base.hpp>

#ifndef TORCH_H
#define TORCH_H
#include <torch/torch.h>
#endif








