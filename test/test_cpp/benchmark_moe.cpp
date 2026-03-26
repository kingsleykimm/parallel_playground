/**
    @file : smaller file to benchmark naive moe forward pass as well as
   optimizations on top
 */
#include <all2all/all2all_base.hpp>
#include <cuda_runtime.h>
#include <kernels/internal_api.hpp>
#include <runtime/parallel.h>

#ifndef TORCH_H
#define TORCH_H
#include <torch/torch.h>
#endif
