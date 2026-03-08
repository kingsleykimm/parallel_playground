#pragma once
#include <cstddef>

extern "C" size_t tk_globals_size(int bm, int bn, int bk);
extern "C" void tk_build_globals(int bm, int bn, int bk, void* out,
    void* A, void* B, void* C, void* scale_a, void* scale_b,
    size_t M, size_t N, size_t K);
