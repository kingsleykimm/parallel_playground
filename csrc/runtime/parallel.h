#pragma once
#include <mpi.h>
struct ParallelGroup {
    uint32_t rank;
    uint32_t size;
    MPI_Comm comm;

    ParallelGroup() : rank (0), size(0) {}
    ParallelGroup (int global_rank, int size) : rank(global_rank % size), size(size) {
        MPI_Comm_split(MPI_COMM_WORLD, global_rank / size, global_rank % size, &comm);
    }



};

struct ParallelConfig {
    uint32_t tp_size;
    uint32_t dp_size;
    uint32_t ep_size;
};