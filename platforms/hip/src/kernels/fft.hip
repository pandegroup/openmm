#include "hip/hip_runtime.h"
static __inline__ __device__ real2 multiplyComplex(real2 c1, real2 c2) {
    return make_real2(c1.x*c2.x-c1.y*c2.y, c1.x*c2.y+c1.y*c2.x);
}

/**
 * Load a value from the half-complex grid produces by a real-to-complex transform.
 */
static __inline__ __device__ real2 loadComplexValue(const real2* __restrict__ in, int x, int y, int z) {
    const int inputZSize = ZSIZE/2+1;
    if (z < inputZSize)
        return in[x*YSIZE*inputZSize+y*inputZSize+z];
    int xp = (x == 0 ? 0 : XSIZE-x);
    int yp = (y == 0 ? 0 : YSIZE-y);
    real2 value = in[xp*YSIZE*inputZSize+yp*inputZSize+(ZSIZE-z)];
    return make_real2(value.x, -value.y);
}

/**
 * Perform a 1D FFT on each row along one axis.
 */

extern "C" __global__ void execFFT(const INPUT_TYPE* __restrict__ in, OUTPUT_TYPE* __restrict__ out) {
    __shared__ real2 w[ZSIZE];
    __shared__ real2 data0[BLOCKS_PER_GROUP*ZSIZE];
    __shared__ real2 data1[BLOCKS_PER_GROUP*ZSIZE];
    for (int i = threadIdx.x; i < ZSIZE; i += blockDim.x)
        w[i] = make_real2(cos(-(SIGN)*i*2*M_PI/ZSIZE), sin(-(SIGN)*i*2*M_PI/ZSIZE));
    __syncthreads();
    
    const int block = threadIdx.x/THREADS_PER_BLOCK;
    for (int baseIndex = blockIdx.x*BLOCKS_PER_GROUP; baseIndex < XSIZE*YSIZE; baseIndex += gridDim.x*BLOCKS_PER_GROUP) {
        int index = baseIndex+block;
        int x = index/YSIZE;
        int y = index-x*YSIZE;
#if OUTPUT_IS_PACKED
        if (x < XSIZE/2+1) {
#endif
        if (index < XSIZE*YSIZE)
            for (int i = threadIdx.x-block*THREADS_PER_BLOCK; i < ZSIZE; i += THREADS_PER_BLOCK)
    #if INPUT_IS_REAL
                data0[i+block*ZSIZE] = make_real2(in[x*(YSIZE*ZSIZE)+y*ZSIZE+i], 0);
    #elif INPUT_IS_PACKED
                data0[i+block*ZSIZE] = loadComplexValue(in, x, y, i);
    #else
                data0[i+block*ZSIZE] = in[x*(YSIZE*ZSIZE)+y*ZSIZE+i];
    #endif
#if OUTPUT_IS_PACKED
        }
#endif
        __syncthreads();
        COMPUTE_FFT
    }
}
