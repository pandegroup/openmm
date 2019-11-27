/**
 * This file contains CUDA definitions for the macros and functions needed for the
 * common compute framework.
 */

#define KERNEL extern "C" __global__
#define LOCAL __shared__
#define GLOBAL
#define RESTRICT __restrict__
#define LOCAL_ID threadIdx.x
#define LOCAL_SIZE blockDim.x
#define GLOBAL_ID blockIdx.x*blockDim.x+threadIdx.x
#define GLOBAL_SiZE blockDim.x*gridDim.x
#define GROUP_ID blockIdx.x
#define NUM_GROUPS gridDim.x
#define SYNC_THREADS __syncthreads();
