#include "hip/hip_runtime.h"
/**
 * This file contains CUDA definitions for the macros and functions needed for the
 * common compute framework.
 */

#define KERNEL extern "C" __global__
#define DEVICE __device__
#define LOCAL __shared__
#define LOCAL_ARG
#define GLOBAL
#define RESTRICT __restrict__
#define LOCAL_ID threadIdx.x
#define LOCAL_SIZE blockDim.x
#define GLOBAL_ID (blockIdx.x*blockDim.x+threadIdx.x)
#define GLOBAL_SIZE (blockDim.x*gridDim.x)
#define GROUP_ID blockIdx.x
#define NUM_GROUPS gridDim.x
#define SYNC_THREADS __syncthreads();
#define MEM_FENCE __threadfence_block();
#define ATOMIC_ADD(dest, value) atomicAdd(dest, value)

typedef long long mm_long;
typedef unsigned long long mm_ulong;

#define SUPPORTS_64_BIT_ATOMICS __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__
#define SUPPORTS_DOUBLE_PRECISION __HIP_ARCH_HAS_DOUBLES__
