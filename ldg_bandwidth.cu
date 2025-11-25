#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "utils.h"



enum class L1CacheHint {
    NO_ALLOCATE,
    EVICT_FIRST,
    EVICT_NORMAL,
    EVICT_LAST
};

enum class L2PrefetchHint {
    B64,
    B128,
    B256
};

template<
    typename T,
    L1CacheHint l1_cache_hint,
    L2PrefetchHint l2_prefetch_hint
>
__device__ __forceinline__
T load_128b_from_gmem(const void* addr) {
    static_assert(sizeof(T) == 128/8);
    int4 ret;

    #define EXEC(L1_HINT_STR, L2_HINT_STR) { \
        asm volatile("ld.global.nc.L1::" L1_HINT_STR ".L2::" L2_HINT_STR ".v4.s32 {%0, %1, %2, %3}, [%4];" \
            : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) \
            : "l"(addr)); \
    }

    #define DISPATCH_L2(L1_HINT_STR) { \
        if constexpr(l2_prefetch_hint == L2PrefetchHint::B64) \
            EXEC(L1_HINT_STR, "64B") \
        else if constexpr(l2_prefetch_hint == L2PrefetchHint::B128) \
            EXEC(L1_HINT_STR, "128B") \
        else if constexpr(l2_prefetch_hint == L2PrefetchHint::B256) \
            EXEC(L1_HINT_STR, "256B") \
    }

    if constexpr(l1_cache_hint == L1CacheHint::NO_ALLOCATE)
        DISPATCH_L2("no_allocate")
    else if constexpr(l1_cache_hint == L1CacheHint::EVICT_FIRST)
        DISPATCH_L2("evict_first")
    else if constexpr(l1_cache_hint == L1CacheHint::EVICT_NORMAL)
        DISPATCH_L2("evict_normal")
    else if constexpr(l1_cache_hint == L1CacheHint::EVICT_LAST)
        DISPATCH_L2("evict_last")

    #undef EXEC
    #undef DISPATCH_L2
    return *reinterpret_cast<T*>(&ret);
}


template <typename Element, size_t TILE_M, size_t TILE_K, size_t UNROLL_ITER, size_t OUT_ITER, size_t LOAD_WIDTH, size_t NUM_THREADS_PER_ROW, size_t NUM_THREADS_PER_COLUMN>
__global__ void __launch_bounds__(128, 1) ldg_kernel(Element* out, const Element* in, uint64_t M, uint64_t K) {
    constexpr int NUM_ELEMENTS = LOAD_WIDTH / sizeof(Element);  // 16B / 4B = 4
    int tidx = threadIdx.x;
    int lane_idx = tidx % 32;
    int bidx = blockIdx.x;

    Element out_register[NUM_ELEMENTS] = {0};

    int m = bidx * TILE_M + tidx / NUM_THREADS_PER_ROW;
    int k = (tidx % NUM_THREADS_PER_ROW) * NUM_ELEMENTS;
    // int m = bidx * TILE_M + tidx / 32 * NUM_THREADS_PER_COLUMN + tidx % NUM_THREADS_PER_COLUMN;
    // int k = (lane_idx / NUM_THREADS_PER_COLUMN) * NUM_ELEMENTS;
    float4 cur_float4[UNROLL_ITER];
    for (int out_idx = 0; out_idx < OUT_ITER; ++out_idx) {
        #pragma unroll
        for (int unroll_idx = 0; unroll_idx < UNROLL_ITER; ++unroll_idx) {
            int k_idx = out_idx * UNROLL_ITER + unroll_idx;
            cur_float4[unroll_idx] = load_128b_from_gmem<float4, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(&in[m * K + k + k_idx * TILE_K]);
        }
        #pragma unroll
        for (int unroll_idx = 0; unroll_idx < UNROLL_ITER; ++unroll_idx) {
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

    auto kernel = &ldg_kernel<Element, TILE_M, TILE_K, UNROLL_ITER, OUT_ITER, LOAD_WIDTH, NUM_THREADS_PER_ROW, NUM_THREADS_PER_COLUMN>;
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