#pragma once

struct ParallelConfig {
    uint32_t tp_size;
    uint32_t dp_size;
    uint32_t ep_size;
    uint32_t node_size;
    uint32_t world_size;
};