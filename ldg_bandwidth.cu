#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>


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


template <typename Element, size_t TILE_M, size_t TILE_K, size_t K_ITER, size_t LOAD_WIDTH, size_t NUM_THREADS_PER_ROW, size_t NUM_THREADS_PER_COLUMN>
__global__ void __launch_bounds__(128, 1) ldg_kernel(Element* out, const Element* in, uint64_t M, uint64_t K) {
    constexpr int NUM_ELEMENTS = LOAD_WIDTH / sizeof(Element);  // 16B / 4B = 4
    // constexpr int NUM_THREADS_PER_ROW = TILE_K / NUM_ELEMENTS;  // 4
    int tidx = threadIdx.x;
    int lane_idx = tidx % 32;
    int bidx = blockIdx.x;

    Element out_register[NUM_ELEMENTS] = {0};

    int m = bidx * TILE_M + tidx / NUM_THREADS_PER_ROW;
    int k = (tidx % NUM_THREADS_PER_ROW) * NUM_ELEMENTS;
    // int m = bidx * TILE_M + tidx / 32 * NUM_THREADS_PER_COLUMN + tidx % NUM_THREADS_PER_COLUMN;
    // int k = (lane_idx / NUM_THREADS_PER_COLUMN) * NUM_ELEMENTS;
    // #pragma unroll
    for (int k_idx = 0; k_idx < K_ITER; ++k_idx) {
        float4 cur_float4 = load_128b_from_gmem<float4, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(&in[m * K + k + k_idx * TILE_K]);
        out_register[0] += (cur_float4.x + 1.f);
    }

    if (out != nullptr) {
        if (tidx == 0 && bidx == 0) printf("out is not nullptr\n");
        *reinterpret_cast<float4*>(&out[m * TILE_K + k]) = *reinterpret_cast<float4*>(out_register);
    }
}


int main() {
    using Element = float;
    int BENCH_ITER = 10;
    constexpr size_t grid = 132;
    constexpr size_t num_warps = 4;
    constexpr size_t K_ITER = 10 * 16;
    constexpr size_t LOAD_WIDTH = 16; // Byte
    // constexpr size_t NUM_THREADS_PER_ROW = 4;
    // constexpr size_t NUM_THREADS_PER_COLUMN = 8;  // Per warp
    constexpr size_t NUM_THREADS_PER_ROW = 8;
    constexpr size_t NUM_THREADS_PER_COLUMN = 4;  // Per warp
    constexpr size_t TILE_M = num_warps * NUM_THREADS_PER_COLUMN;  // 4 * 8
    constexpr size_t TILE_K = NUM_THREADS_PER_ROW * (LOAD_WIDTH / sizeof(Element));
    uint64_t M = grid * TILE_M;
    uint64_t K = K_ITER * TILE_K;
    size_t size_byte = M * K * sizeof(Element);
    double size_gb = (double)size_byte / (1 << 30);
    cudaStream_t stream = 0;
    std::vector<Element*> in_list(BENCH_ITER);
    for (int i = 0; i < BENCH_ITER; ++i) {
        cudaMalloc(&in_list[i], size_byte);
        cudaMemset(in_list[i], 0, size_byte);
    }

    constexpr size_t sm90_capacity = 232448;
    cudaFuncSetAttribute(ldg_kernel<Element, TILE_M, TILE_K, K_ITER, LOAD_WIDTH, NUM_THREADS_PER_ROW, NUM_THREADS_PER_COLUMN>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm90_capacity);

    printf("M=%lu; K=%lu\n", M, K);
    printf("TILE_M=%lu; TILE_K=%lu\n", TILE_M, TILE_K);
    printf("Per Tile: %0.3fKB\n", float(TILE_M * TILE_K * sizeof(Element)) / (1 << 10));
    auto kernel = &ldg_kernel<Element, TILE_M, TILE_K, K_ITER, LOAD_WIDTH, NUM_THREADS_PER_ROW, NUM_THREADS_PER_COLUMN>;
    // warm up
    for (int i = 0; i < 10; ++i) {
        kernel<<<grid, 128, sm90_capacity, stream>>>(nullptr, in_list[i], M, K);
    }

    // read
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_ms = 0.f;
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITER; ++i) {
        kernel<<<grid, 128, sm90_capacity, stream>>>(nullptr, in_list[i], M, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("time %f us\n", 1000 * time_ms / BENCH_ITER);
    printf("read %fGB/s\n", size_gb * BENCH_ITER / ((double)time_ms / 1000));
    Element* out_device;
    Element* out_host;
    cudaMalloc(&out_device, M * TILE_K * sizeof(Element));
    cudaMemset(out_device, 0, M * TILE_K * sizeof(Element));
    out_host = (Element*)malloc(M * TILE_K * sizeof(Element));
    kernel<<<grid, 128, sm90_capacity, stream>>>(out_device, in_list[0], M, K);
    cudaDeviceSynchronize();
    cudaMemcpy(out_host, out_device, M * TILE_K * sizeof(Element), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * TILE_K; ++i) {
        if ((i % 4 == 0 && abs(out_host[i] - K_ITER) > 1e-3) || (i % 4 != 0 && abs(out_host[i] > 1e-3))) {
            printf("wrong in %d: %f vs %ld\n", i, out_host[i], K_ITER);
            return 0;
        }
    }
    printf("all pass\n");
    return 0;
}