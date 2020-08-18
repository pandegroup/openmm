#include "hip/hip_runtime.h"
/**
 * Combine the two halves of a real grid into a complex grid that is half as large.
 */
extern "C" __global__ void packForwardData(const real* __restrict__ in, real2* __restrict__ out) {
    const int gridSize = PACKED_XSIZE*PACKED_YSIZE*PACKED_ZSIZE;
    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < gridSize; index += blockDim.x*gridDim.x) {
        int x = index/(PACKED_YSIZE*PACKED_ZSIZE);
        int remainder = index-x*(PACKED_YSIZE*PACKED_ZSIZE);
        int y = remainder/PACKED_ZSIZE;
        int z = remainder-y*PACKED_ZSIZE;
#if PACKED_AXIS == 0
        real2 value = make_real2(in[2*x*YSIZE*ZSIZE+y*ZSIZE+z], in[(2*x+1)*YSIZE*ZSIZE+y*ZSIZE+z]);
#elif PACKED_AXIS == 1
        real2 value = make_real2(in[x*YSIZE*ZSIZE+2*y*ZSIZE+z], in[x*YSIZE*ZSIZE+(2*y+1)*ZSIZE+z]);
#else
        real2 value = make_real2(in[x*YSIZE*ZSIZE+y*ZSIZE+2*z], in[x*YSIZE*ZSIZE+y*ZSIZE+(2*z+1)]);
#endif
        out[index] = value;
    }
}

/**
 * Split the transformed data back into a full sized, symmetric grid.
 */
extern "C" __global__ void unpackForwardData(const real2* __restrict__ in, real2* __restrict__ out) {
    // Compute the phase factors.
    
#if PACKED_AXIS == 0
    __shared__ real2 w[PACKED_XSIZE];
    for (int i = threadIdx.x; i < PACKED_XSIZE; i += blockDim.x)
        w[i] = make_real2(sin(i*2*M_PI/XSIZE), cos(i*2*M_PI/XSIZE));
#elif PACKED_AXIS == 1
    __shared__ real2 w[PACKED_YSIZE];
    for (int i = threadIdx.x; i < PACKED_YSIZE; i += blockDim.x)
        w[i] = make_real2(sin(i*2*M_PI/YSIZE), cos(i*2*M_PI/YSIZE));
#else
    __shared__ real2 w[PACKED_ZSIZE];
    for (int i = threadIdx.x; i < PACKED_ZSIZE; i += blockDim.x)
        w[i] = make_real2(sin(i*2*M_PI/ZSIZE), cos(i*2*M_PI/ZSIZE));
#endif
    __syncthreads();

    // Transform the data.
    
    const int gridSize = PACKED_XSIZE*PACKED_YSIZE*PACKED_ZSIZE;
    const int outputZSize = ZSIZE/2+1;
    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < gridSize; index += blockDim.x*gridDim.x) {
        int x = index/(PACKED_YSIZE*PACKED_ZSIZE);
        int remainder = index-x*(PACKED_YSIZE*PACKED_ZSIZE);
        int y = remainder/PACKED_ZSIZE;
        int z = remainder-y*PACKED_ZSIZE;
        int xp = (x == 0 ? 0 : PACKED_XSIZE-x);
        int yp = (y == 0 ? 0 : PACKED_YSIZE-y);
        int zp = (z == 0 ? 0 : PACKED_ZSIZE-z);
        real2 z1 = in[x*PACKED_YSIZE*PACKED_ZSIZE+y*PACKED_ZSIZE+z];
        real2 z2 = in[xp*PACKED_YSIZE*PACKED_ZSIZE+yp*PACKED_ZSIZE+zp];
#if PACKED_AXIS == 0
        real2 wfac = w[x];
#elif PACKED_AXIS == 1
        real2 wfac = w[y];
#else
        real2 wfac = w[z];
#endif
        real2 output = make_real2((z1.x+z2.x - wfac.x*(z1.x-z2.x) + wfac.y*(z1.y+z2.y))/2, (z1.y-z2.y - wfac.y*(z1.x-z2.x) - wfac.x*(z1.y+z2.y))/2);
        if (z < outputZSize)
            out[x*YSIZE*outputZSize+y*outputZSize+z] = output;
        xp = (x == 0 ? 0 : XSIZE-x);
        yp = (y == 0 ? 0 : YSIZE-y);
        zp = (z == 0 ? 0 : ZSIZE-z);
        if (zp < outputZSize) {
#if PACKED_AXIS == 0
            if (x == 0)
                out[PACKED_XSIZE*YSIZE*outputZSize+yp*outputZSize+zp] = make_real2((z1.x-z1.y+z2.x-z2.y)/2, (-z1.x-z1.y+z2.x+z2.y)/2);
#elif PACKED_AXIS == 1
            if (y == 0)
                out[xp*YSIZE*outputZSize+PACKED_YSIZE*outputZSize+zp] = make_real2((z1.x-z1.y+z2.x-z2.y)/2, (-z1.x-z1.y+z2.x+z2.y)/2);
#else
            if (z == 0)
                out[xp*YSIZE*outputZSize+yp*outputZSize+PACKED_ZSIZE] = make_real2((z1.x-z1.y+z2.x-z2.y)/2, (-z1.x-z1.y+z2.x+z2.y)/2);
#endif
            else
                out[xp*YSIZE*outputZSize+yp*outputZSize+zp] = make_real2(output.x, -output.y);
        }
    }
}

/**
 * Load a value from the half-complex grid produced by a real-to-complex transform.
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
 * Repack the symmetric complex grid into one half as large in preparation for doing an inverse complex-to-real transform.
 */
extern "C" __global__ void packBackwardData(const real2* __restrict__ in, real2* __restrict__ out) {
    // Compute the phase factors.
    
#if PACKED_AXIS == 0
    __shared__ real2 w[PACKED_XSIZE];
    for (int i = threadIdx.x; i < PACKED_XSIZE; i += blockDim.x)
        w[i] = make_real2(cos(i*2*M_PI/XSIZE), sin(i*2*M_PI/XSIZE));
#elif PACKED_AXIS == 1
    __shared__ real2 w[PACKED_YSIZE];
    for (int i = threadIdx.x; i < PACKED_YSIZE; i += blockDim.x)
        w[i] = make_real2(cos(i*2*M_PI/YSIZE), sin(i*2*M_PI/YSIZE));
#else
    __shared__ real2 w[PACKED_ZSIZE];
    for (int i = threadIdx.x; i < PACKED_ZSIZE; i += blockDim.x)
        w[i] = make_real2(cos(i*2*M_PI/ZSIZE), sin(i*2*M_PI/ZSIZE));
#endif
    __syncthreads();

    // Transform the data.
    
    const int gridSize = PACKED_XSIZE*PACKED_YSIZE*PACKED_ZSIZE;
    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < gridSize; index += blockDim.x*gridDim.x) {
        int x = index/(PACKED_YSIZE*PACKED_ZSIZE);
        int remainder = index-x*(PACKED_YSIZE*PACKED_ZSIZE);
        int y = remainder/PACKED_ZSIZE;
        int z = remainder-y*PACKED_ZSIZE;
        int xp = (x == 0 ? 0 : PACKED_XSIZE-x);
        int yp = (y == 0 ? 0 : PACKED_YSIZE-y);
        int zp = (z == 0 ? 0 : PACKED_ZSIZE-z);
        real2 z1 = loadComplexValue(in, x, y, z);
#if PACKED_AXIS == 0
        real2 wfac = w[x];
        real2 z2 = loadComplexValue(in, PACKED_XSIZE-x, yp, zp);
#elif PACKED_AXIS == 1
        real2 wfac = w[y];
        real2 z2 = loadComplexValue(in, xp, PACKED_YSIZE-y, zp);
#else
        real2 wfac = w[z];
        real2 z2 = loadComplexValue(in, xp, yp, PACKED_ZSIZE-z);
#endif
        real2 even = make_real2((z1.x+z2.x)/2, (z1.y-z2.y)/2);
        real2 odd = make_real2((z1.x-z2.x)/2, (z1.y+z2.y)/2);
        odd = make_real2(odd.x*wfac.x-odd.y*wfac.y, odd.y*wfac.x+odd.x*wfac.y);
        out[x*PACKED_YSIZE*PACKED_ZSIZE+y*PACKED_ZSIZE+z] = make_real2(even.x-odd.y, even.y+odd.x);
    }
}

/**
 * Split the data back into a full sized, real grid after an inverse transform.
 */
extern "C" __global__ void unpackBackwardData(const real2* __restrict__ in, real* __restrict__ out) {
    const int gridSize = PACKED_XSIZE*PACKED_YSIZE*PACKED_ZSIZE;
    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < gridSize; index += blockDim.x*gridDim.x) {
        int x = index/(PACKED_YSIZE*PACKED_ZSIZE);
        int remainder = index-x*(PACKED_YSIZE*PACKED_ZSIZE);
        int y = remainder/PACKED_ZSIZE;
        int z = remainder-y*PACKED_ZSIZE;
        real2 value = 2*in[index];
#if PACKED_AXIS == 0
        out[2*x*YSIZE*ZSIZE+y*ZSIZE+z] = value.x;
        out[(2*x+1)*YSIZE*ZSIZE+y*ZSIZE+z] = value.y;
#elif PACKED_AXIS == 1
        out[x*YSIZE*ZSIZE+2*y*ZSIZE+z] = value.x;
        out[x*YSIZE*ZSIZE+(2*y+1)*ZSIZE+z] = value.y;
#else
        out[x*YSIZE*ZSIZE+y*ZSIZE+2*z] = value.x;
        out[x*YSIZE*ZSIZE+y*ZSIZE+(2*z+1)] = value.y;
#endif
    }
}
