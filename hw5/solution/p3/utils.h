// Do not edit!
#pragma once

#include <cstdio>
#include <cuda_runtime.h>

/// Print the error message if a CUDA API execution failed.
void _cudaCheck(const char *file, int line, const char *func, cudaError_t code) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d in %s:%s\n",
                file, line, func, cudaGetErrorString(code));
        exit(1);
    }
}

// Wrap every cuda API with this check.
#define CUDA_CHECK(cmd) _cudaCheck(__FILE__, __LINE__, __PRETTY_FUNCTION__, (cmd))

/// Replace
///     kernel<<<blocks, threads>>>(arg1, ...);
/// with
///     CUDA_LAUNCH(kernel, blocks, threads, arg1, ...);
#define CUDA_LAUNCH(kernel, blocks, threads, ...) do {                           \
        cudaGetLastError();                                                      \
        kernel<<<(blocks), (threads)>>>(__VA_ARGS__);                            \
        _cudaCheck(__FILE__, __LINE__, __PRETTY_FUNCTION__, cudaGetLastError()); \
    } while (0)
