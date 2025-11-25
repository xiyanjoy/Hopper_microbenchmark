#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "utils.h"


template <class TS, class TD = TS>
__device__ __forceinline__
void cp_async_copy_128b(TS const* gmem_ptr, TD* smem_ptr, bool pred=true) {
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    int src_size = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n"
        :: "r"(smem_int_ptr),
            "l"(gmem_ptr),
            "n"(16),
            "r"(src_size));
}

__device__ __forceinline__
void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__
void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}


template <typename Element, size_t TILE_M, size_t TILE_K, size_t UNROLL_ITER, size_t OUT_ITER, size_t LOAD_WIDTH, size_t NUM_THREADS_PER_ROW, size_t NUM_THREADS_PER_COLUMN, size_t NUM_STAGES>
__global__ void __launch_bounds__(128, 1) ldgsts_kernel(Element* out, const Element* in, uint64_t M, uint64_t K) {
    constexpr int NUM_ELEMENTS = LOAD_WIDTH / sizeof(Element);  // 16B / 4B = 4
    int tidx = threadIdx.x;
    int lane_idx = tidx % 32;
    int bidx = blockIdx.x;

    extern __shared__ __align__(1024) Element smem_buffer[];  // (NUM_STAGES, TILE_M * TILE_K * UNROLL_ITER)
    Element out_register[NUM_ELEMENTS] = {0};

    int m = bidx * TILE_M + tidx / NUM_THREADS_PER_ROW;
    int k = (tidx % NUM_THREADS_PER_ROW) * NUM_ELEMENTS;
    // int m = bidx * TILE_M + tidx / 32 * NUM_THREADS_PER_COLUMN + tidx % NUM_THREADS_PER_COLUMN;
    // int k = (lane_idx / NUM_THREADS_PER_COLUMN) * NUM_ELEMENTS;
    float4 cur_float4[UNROLL_ITER];
    #pragma unroll
    for (int out_idx = 0; out_idx < NUM_STAGES - 1; ++out_idx) {
        #pragma unroll
        for (int unroll_idx = 0; unroll_idx < UNROLL_ITER; ++unroll_idx) {
            int k_idx = out_idx * UNROLL_ITER + unroll_idx;
            cp_async_copy_128b(
                &in[m * K + k + k_idx * TILE_K],
                &smem_buffer[(out_idx % NUM_STAGES) * TILE_M * TILE_K * UNROLL_ITER +
                    (m - bidx * TILE_M) * (TILE_K * UNROLL_ITER) + TILE_K * unroll_idx + k],  // [m - bidx * TILE_M, TILE_K * unroll_idx + k]
                out_idx < OUT_ITER
            );
        }
        cp_async_fence();
    }
    for (int out_idx = 0; out_idx < OUT_ITER; ++out_idx) {
        cp_async_wait<NUM_STAGES - 2>();
        #pragma unroll
        for (int unroll_idx = 0; unroll_idx < UNROLL_ITER; ++unroll_idx) {
            int k_idx = (out_idx + NUM_STAGES - 1) * UNROLL_ITER + unroll_idx;
            cp_async_copy_128b(
                &in[m * K + k + k_idx * TILE_K],
                &smem_buffer[((out_idx + NUM_STAGES - 1) % NUM_STAGES) * TILE_M * TILE_K * UNROLL_ITER +
                    (m - bidx * TILE_M) * (TILE_K * UNROLL_ITER) + TILE_K * unroll_idx + k],  // [m - bidx * TILE_M, TILE_K * unroll_idx + k]
                out_idx + NUM_STAGES - 1 < OUT_ITER
            );
        }
        cp_async_fence();
        #pragma unroll
        for (int unroll_idx = 0; unroll_idx < UNROLL_ITER; ++unroll_idx) {
            cur_float4[unroll_idx] = *reinterpret_cast<float4*>(&smem_buffer[
                (out_idx % NUM_STAGES) * TILE_M * TILE_K * UNROLL_ITER +
                (m - bidx * TILE_M) * (TILE_K * UNROLL_ITER) + TILE_K * unroll_idx + k
            ]);
            out_register[0] += (cur_float4[unroll_idx].x + 1.f);
            out_register[1] += (cur_float4[unroll_idx].y + 1.f);
            out_register[2] += (cur_float4[unroll_idx].z + 1.f);
            out_register[3] += (cur_float4[unroll_idx].w + 1.f);
        }
    }

    if (out != nullptr) {
        if (tidx == 0 && bidx == 0) printf("out is not nullptr\n");
        *reinterpret_cast<float4*>(&out[m * TILE_K + k]) = *reinterpret_cast<float4*>(out_register);
    }
}


int main() {
    using Element = float;
    constexpr size_t NUM_STAGES = 2;
    int BENCH_ITER = 100;
    int NUM_WORKSPACE = 8;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
    printf("Shared memory per SM: %ld byte\n", deviceProp.sharedMemPerMultiprocessor);
    int sharedMemPerBlock = 0;
    cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    printf("Max Shared memory per SM: %d byte\n", sharedMemPerBlock);
    size_t grid = deviceProp.multiProcessorCount;
    constexpr size_t num_warps = 4;
    constexpr size_t UNROLL_ITER = 36;
    constexpr size_t OUT_ITER = 1024;
    constexpr size_t LOAD_WIDTH = 16; // Byte
    // constexpr size_t NUM_THREADS_PER_ROW = 4;
    // constexpr size_t NUM_THREADS_PER_COLUMN = 8;  // Per warp
    constexpr size_t NUM_THREADS_PER_ROW = 8;
    constexpr size_t NUM_THREADS_PER_COLUMN = 4;  // Per warp
    constexpr size_t TILE_M = num_warps * NUM_THREADS_PER_COLUMN;
    constexpr size_t TILE_K = NUM_THREADS_PER_ROW * (LOAD_WIDTH / sizeof(Element));
    constexpr size_t K_ITER = UNROLL_ITER * OUT_ITER;
    uint64_t M = grid * TILE_M;
    uint64_t K = K_ITER * TILE_K;
    size_t size_byte = M * K * sizeof(Element);
    double size_gb = (double)size_byte / (1 << 30);
    cudaStream_t stream = 0;
    std::vector<Element*> in_list(NUM_WORKSPACE);
    for (int i = 0; i < NUM_WORKSPACE; ++i) {
        cudaMalloc(&in_list[i], size_byte);
        cudaMemset(in_list[i], 0, size_byte);
    }

    auto kernel = &ldgsts_kernel<Element, TILE_M, TILE_K, UNROLL_ITER, OUT_ITER, LOAD_WIDTH, NUM_THREADS_PER_ROW, NUM_THREADS_PER_COLUMN, NUM_STAGES>;
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemPerBlock));

    printf("M=%lu; K=%lu\n", M, K);
    printf("All Tile: %0.3fKB\n", float(M * K * sizeof(Element)) / (1 << 10));
    printf("TILE_M=%lu; TILE_K=%lu\n", TILE_M, TILE_K);
    printf("Per Tile: %0.3fKB\n", float(TILE_M * TILE_K * sizeof(Element)) / (1 << 10));
    // warm up
    for (int i = 0; i < NUM_WORKSPACE; ++i) {
        kernel<<<grid, 128, sharedMemPerBlock, stream>>>(nullptr, in_list[i], M, K);
    }
    CHECK_CUDA_KERNEL_LAUNCH();

    // read
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_ms = 0.f;
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITER; ++i) {
        kernel<<<grid, 128, sharedMemPerBlock, stream>>>(nullptr, in_list[i % NUM_WORKSPACE], M, K);
    }
    cudaEventRecord(stop);
    CHECK_CUDA_KERNEL_LAUNCH();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("time %f us\n", 1000 * time_ms / BENCH_ITER);
    printf("read %fGB/s\n", size_gb * BENCH_ITER / ((double)time_ms / 1000));
    Element* out_device;
    Element* out_host;
    cudaMalloc(&out_device, M * TILE_K * sizeof(Element));
    cudaMemset(out_device, 0, M * TILE_K * sizeof(Element));
    out_host = (Element*)malloc(M * TILE_K * sizeof(Element));
    kernel<<<grid, 128, sharedMemPerBlock, stream>>>(out_device, in_list[0], M, K);
    CHECK_CUDA_KERNEL_LAUNCH();
    cudaDeviceSynchronize();
    cudaMemcpy(out_host, out_device, M * TILE_K * sizeof(Element), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * TILE_K; ++i) {
        if (abs(out_host[i] - K_ITER) > 1e-3) {
            printf("wrong in %d: %f vs %ld\n", i, out_host[i], K_ITER);
            return 0;
        }
    }
    printf("all pass\n");
    return 0;
}