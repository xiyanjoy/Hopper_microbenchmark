#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cuda/barrier>

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/util/debug.hpp>

using Barrier = cuda::barrier<cuda::thread_scope_block>;


template <class T>
inline CUtensorMapDataType get_CUtensorMapDataType() {
	if constexpr (std::is_same<T, int8_t>::value) {
		return CU_TENSOR_MAP_DATA_TYPE_UINT8;
	} else if constexpr (std::is_same<T, uint8_t>::value) {
		return CU_TENSOR_MAP_DATA_TYPE_UINT8;
	} else if constexpr (std::is_same<T, __nv_fp8_e4m3>::value) {
		return CU_TENSOR_MAP_DATA_TYPE_UINT8;
	} else if constexpr (std::is_same<T, __nv_fp8_e5m2>::value) {
		return CU_TENSOR_MAP_DATA_TYPE_UINT8;
	} else if constexpr (std::is_same<T, uint16_t>::value) {
		return CU_TENSOR_MAP_DATA_TYPE_UINT16;
	} else if constexpr (std::is_same<T, uint32_t>::value) {
		return CU_TENSOR_MAP_DATA_TYPE_UINT32;
	} else if constexpr (std::is_same<T, uint64_t>::value) {
		return CU_TENSOR_MAP_DATA_TYPE_UINT64;
	} else if constexpr (std::is_same<T, int32_t>::value) {
		return CU_TENSOR_MAP_DATA_TYPE_INT32;
	} else if constexpr (std::is_same<T, int64_t>::value) {
		return CU_TENSOR_MAP_DATA_TYPE_INT64;
	} else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
		return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
	} else if constexpr (std::is_same<T, __half>::value) {
		return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
	} else if constexpr (std::is_same<T, float>::value) {
		return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
	} else if constexpr (std::is_same<T, double>::value) {
		return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
	} else {
		static_assert(sizeof(T) < 0, "Unknown TMA Format!");
	}
}

PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    // Get pointer to cuTensorMapEncodeTiled
    cudaDriverEntryPointQueryResult driver_status;
    void* cuTensorMapEncodeTiled_ptr = nullptr;
    #if (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 5)
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, \
                                        cudaEnableDefault, &driver_status);
    #else
        cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, \
                                cudaEnableDefault, &driver_status);
    #endif

    if (driver_status != cudaDriverEntryPointSuccess) {
        throw std::runtime_error("driver_status != cudaDriverEntryPointSuccess");
    }

    return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(cuTensorMapEncodeTiled_ptr);
    }

template <typename data_type>
CUtensorMap make_2d_tma_copy_desc(
    data_type* global_address, 
    uint64_t gmem_dim[2],
    uint64_t stride_in_bytes, 
    uint32_t smem_dim[2], 
    CUtensorMapSwizzle swizzle_type, 
    PFN_cuTensorMapEncodeTiled encode_func = nullptr
) {
    CUtensorMap tensor_map{};
    constexpr uint32_t rank = 2;
    uint64_t global_stride[rank - 1] = {stride_in_bytes};
    uint32_t elem_strides[rank] = {1, 1};

    if (encode_func == nullptr) encode_func = get_cuTensorMapEncodeTiled();

    CUresult res = encode_func(
        &tensor_map, get_CUtensorMapDataType<typename std::remove_cv<data_type>::type>(), rank,
        global_address, gmem_dim, global_stride, smem_dim, elem_strides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_type,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tensor_map;
}

__device__ __forceinline__ uint64_t mbarrier_arrive_1_expect_tx_cta(void* smem_ptr, uint32_t tx_count) {
  uint64_t state;
  asm("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2; // 8. "
    : "=l"(state)
    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr))), "r"(tx_count)
    : "memory");
  return state;
}

__device__ __host__ __forceinline__ constexpr int div_up(int a, int b) { return (a + b - 1) / b; }


template <typename Element, size_t TILE_M, size_t TILE_K, size_t NUM_STAGES>
__global__ void __launch_bounds__(256, 1) read_kernel_v0(Element* out, const __grid_constant__ CUtensorMap tensor_map, size_t M, size_t K) {
    constexpr uint64_t SMEM_SIZE_PER_STAGE = TILE_M * TILE_K * sizeof(Element);
    constexpr int BLOCK_SIZE = 256;
    constexpr int K_PER_ITER = NUM_STAGES * TILE_K;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    Element* smem[NUM_STAGES];
    Barrier* full_bars[NUM_STAGES];
    Barrier* empty_bars[NUM_STAGES];
    for (int i = 0; i < NUM_STAGES; ++i) {
        smem[i] = reinterpret_cast<Element*>(&smem_buffer[i * SMEM_SIZE_PER_STAGE]);
        full_bars[i] = reinterpret_cast<Barrier*>(&smem_buffer[NUM_STAGES * SMEM_SIZE_PER_STAGE + i * sizeof(Barrier)]);
        empty_bars[i] = full_bars[i] + NUM_STAGES;
    }
    int lane_predicate = cute::elect_one_sync();
    if (threadIdx.x < 32 && lane_predicate == 1) {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map));
        for (int i = 0; i < NUM_STAGES; i++) init(full_bars[i], 1);
        for (int i = 0; i < NUM_STAGES; i++) init(empty_bars[i], BLOCK_SIZE - 128);
        cutlass::arch::fence_view_async_shared();
    }
    __syncthreads();

    if (threadIdx.x < 128) {
        cutlass::arch::warpgroup_reg_dealloc<40>();
        int producer_phase = 0;
        #pragma unroll(NUM_STAGES)
        for (int i = 0; i < K / TILE_K; ++i) {
            if (threadIdx.x == 0) {
                int which_smem = i % NUM_STAGES;
                (*empty_bars[which_smem]).wait_parity((producer_phase + 1) & 1);
                cute::SM90_TMA_LOAD_2D::copy(
                    &tensor_map,
                    reinterpret_cast<uint64_t*>(full_bars[which_smem]),
                    static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                    smem[which_smem],
					i * TILE_K, 
					blockIdx.x * TILE_M);
                auto no_use = mbarrier_arrive_1_expect_tx_cta(full_bars[which_smem], SMEM_SIZE_PER_STAGE);
                if (which_smem == NUM_STAGES-1) ++producer_phase;
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<232>();
        int consumer_phase = 0;
        #pragma unroll(NUM_STAGES)
        for (int i = 0; i < K / TILE_K; ++i) {
            int which_smem = i % NUM_STAGES;
            (*full_bars[which_smem]).wait_parity(consumer_phase & 1);
            // dummy STG
            if (int(smem[which_smem][threadIdx.x - 128]) != 0) *out = smem[which_smem][threadIdx.x - 128];
            auto no_use_var = (*empty_bars[which_smem]).arrive();
            if (which_smem == NUM_STAGES-1) ++consumer_phase;
        }
    }
}


template <typename Element, size_t TILE_M, size_t TILE_K, size_t NUM_STAGES>
__global__ void __launch_bounds__(256, 1) read_kernel_v1(Element* out, const __grid_constant__ CUtensorMap tensor_map, size_t M, size_t K) {
    constexpr uint64_t SMEM_SIZE_PER_STAGE = TILE_M * TILE_K * sizeof(Element);
    constexpr int BLOCK_SIZE = 256;
    constexpr int K_PER_ITER = NUM_STAGES * TILE_K;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    Element* smem[NUM_STAGES];
    Barrier* full_bars[NUM_STAGES];
    Barrier* empty_bars[NUM_STAGES];
    for (int i = 0; i < NUM_STAGES; ++i) {
        smem[i] = reinterpret_cast<Element*>(&smem_buffer[i * SMEM_SIZE_PER_STAGE]);
        full_bars[i] = reinterpret_cast<Barrier*>(&smem_buffer[NUM_STAGES * SMEM_SIZE_PER_STAGE + i * sizeof(Barrier)]);
        empty_bars[i] = full_bars[i] + NUM_STAGES;
    }
    int lane_predicate = cute::elect_one_sync();
    if (threadIdx.x < 32 && lane_predicate == 1) {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map));
        for (int i = 0; i < NUM_STAGES; i++) init(full_bars[i], 1);
        for (int i = 0; i < NUM_STAGES; i++) init(empty_bars[i], BLOCK_SIZE - 128);
        cutlass::arch::fence_view_async_shared();
    }
    __syncthreads();

    if (threadIdx.x < 128) {
        cutlass::arch::warpgroup_reg_dealloc<40>();
		if (threadIdx.x == 0) {
        	for (int k_iter = 0; k_iter < div_up(K, K_PER_ITER); k_iter++) {
                int num_stages = (K - k_iter * K_PER_ITER) / TILE_K;  // assuming K is a multiple of TILE_K
                # pragma unroll
                for (int s = 0; s < NUM_STAGES; s++) {
                    if (s == num_stages) break;
                    (*empty_bars[s]).wait_parity(k_iter + 1 & 1);
                    cute::SM90_TMA_LOAD_2D::copy(
						&tensor_map,
						reinterpret_cast<uint64_t*>(full_bars[s]),
						static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
						smem[s],
						k_iter * K_PER_ITER + s * TILE_K,
						blockIdx.x * TILE_M);
                    auto no_use_var = mbarrier_arrive_1_expect_tx_cta(full_bars[s], SMEM_SIZE_PER_STAGE);
                }
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<232>();
		for (int k_iter = 0; k_iter < div_up(K, K_PER_ITER); k_iter++) {
			#pragma unroll
			for (int s = 0; s < NUM_STAGES; s++) {
				(*full_bars[s]).wait_parity(k_iter & 1);
				//dummy STG
				if (int(smem[s][threadIdx.x - 128]) != 0) *out = smem[s][threadIdx.x - 128];
				auto no_use = (*empty_bars[s]).arrive();
			}
		}
    }
}


int main() {
    using Element = __half;
    int BENCH_ITER = 100;
    constexpr size_t grid = 132;
    constexpr size_t NUM_STAGES = 8;
    constexpr size_t k_iter = 8 * NUM_STAGES;
    constexpr uint32_t TILE_M = 4*8;
    constexpr uint32_t TILE_K = 64;
    uint64_t M = grid * TILE_M;
    uint64_t K = k_iter * TILE_K;
    uint32_t smem_dim[2] = {TILE_K, TILE_M};
    uint64_t gmem_dim[2] = {K, M};
    size_t size_byte = M * K * sizeof(Element);
    double size_gb = (double)size_byte / (1 << 30);
    CUtensorMapSwizzle swizzle_type = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    cudaStream_t stream = 0;
    std::vector<Element*> in_list(BENCH_ITER);
    std::vector<CUtensorMap> desc_list(BENCH_ITER);
    for (int i = 0; i < BENCH_ITER; ++i) {
        cudaMalloc(&in_list[i], size_byte);
        cudaMemset(in_list[i], 0, size_byte);
        desc_list[i] = make_2d_tma_copy_desc<Element>(in_list[i], gmem_dim, K * sizeof(Element), smem_dim, swizzle_type);
    }

    constexpr size_t sm90_capacity = 232448;
    constexpr size_t smem_size = NUM_STAGES * (TILE_M * TILE_K * sizeof(Element) + 2 * sizeof(Barrier));
    static_assert(smem_size <= sm90_capacity, "The required shared memory size is too large");
    cudaFuncSetAttribute(read_kernel_v1<Element, TILE_M, TILE_K, NUM_STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    printf("M=%lu; K=%lu\n", M, K);
    printf("TILE_M=%u; TILE_K=%u\n", TILE_M, TILE_K);
    printf("Per TMA: %0.3fKB\n", float(TILE_M * TILE_K * sizeof(Element)) / (1 << 10));
    // warm up
    for (int i = 0; i < 10; ++i)
        read_kernel_v1<Element, TILE_M, TILE_K, NUM_STAGES><<<grid, 256, smem_size, stream>>>(nullptr, desc_list[i], M, K);

    // read
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_ms = 0.f;
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITER; ++i) {
        read_kernel_v1<Element, TILE_M, TILE_K, NUM_STAGES><<<grid, 256, smem_size, stream>>>(nullptr, desc_list[i], M, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("time %f us\n", 1000 * time_ms / BENCH_ITER);
    printf("read %fGB/s\n", size_gb * BENCH_ITER / ((double)time_ms / 1000));

    return 0;
}