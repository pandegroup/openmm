#include "hip/hip_runtime.h"
extern "C" {

/**
 * This is called by the various functions below to clear a buffer.
 */
__device__ void clearSingleBuffer(int* __restrict__ buffer, int size) {
    int index = blockDim.x*blockIdx.x+threadIdx.x;
    int4* buffer4 = (int4*) buffer;
    int sizeDiv4 = size/4;
    while (index < sizeDiv4) {
        buffer4[index] = make_int4(0);
        index += blockDim.x*gridDim.x;
    }
    if (blockDim.x*blockIdx.x+threadIdx.x == 0)
        for (int i = sizeDiv4*4; i < size; i++)
            buffer[i] = 0;
}

/**
 * Fill a buffer with 0.
 */
__global__ void clearBuffer(int* __restrict__ buffer, int size) {
    clearSingleBuffer(buffer, size);
}

/**
 * Fill two buffers with 0.
 */
__global__ void clearTwoBuffers(int* __restrict__ buffer1, int size1, int* __restrict__ buffer2, int size2) {
    clearSingleBuffer(buffer1, size1);
    clearSingleBuffer(buffer2, size2);
}

/**
 * Fill three buffers with 0.
 */
__global__ void clearThreeBuffers(int* __restrict__ buffer1, int size1, int* __restrict__ buffer2, int size2, int* __restrict__ buffer3, int size3) {
    clearSingleBuffer(buffer1, size1);
    clearSingleBuffer(buffer2, size2);
    clearSingleBuffer(buffer3, size3);
}

/**
 * Fill four buffers with 0.
 */
__global__ void clearFourBuffers(int* __restrict__ buffer1, int size1, int* __restrict__ buffer2, int size2, int* __restrict__ buffer3, int size3, int* __restrict__ buffer4, int size4) {
    clearSingleBuffer(buffer1, size1);
    clearSingleBuffer(buffer2, size2);
    clearSingleBuffer(buffer3, size3);
    clearSingleBuffer(buffer4, size4);
}

/**
 * Fill five buffers with 0.
 */
__global__ void clearFiveBuffers(int* __restrict__ buffer1, int size1, int* __restrict__ buffer2, int size2, int* __restrict__ buffer3, int size3, int* __restrict__ buffer4, int size4, int* __restrict__ buffer5, int size5) {
    clearSingleBuffer(buffer1, size1);
    clearSingleBuffer(buffer2, size2);
    clearSingleBuffer(buffer3, size3);
    clearSingleBuffer(buffer4, size4);
    clearSingleBuffer(buffer5, size5);
}

/**
 * Fill six buffers with 0.
 */
__global__ void clearSixBuffers(int* __restrict__ buffer1, int size1, int* __restrict__ buffer2, int size2, int* __restrict__ buffer3, int size3, int* __restrict__ buffer4, int size4, int* __restrict__ buffer5, int size5, int* __restrict__ buffer6, int size6) {
    clearSingleBuffer(buffer1, size1);
    clearSingleBuffer(buffer2, size2);
    clearSingleBuffer(buffer3, size3);
    clearSingleBuffer(buffer4, size4);
    clearSingleBuffer(buffer5, size5);
    clearSingleBuffer(buffer6, size6);
}

/**
 * Sum the energy buffer.
 */
__global__ void reduceEnergy(const mixed* __restrict__ energyBuffer, mixed* __restrict__ result, int bufferSize, int workGroupSize) {
    HIP_DYNAMIC_SHARED( mixed, tempBuffer)
    const unsigned int thread = threadIdx.x;
    mixed sum = 0;
    for (unsigned int index = thread; index < bufferSize; index += blockDim.x)
        sum += energyBuffer[index];
    tempBuffer[thread] = sum;
    for (int i = 1; i < workGroupSize; i *= 2) {
        __syncthreads();
        if (thread%(i*2) == 0 && thread+i < workGroupSize)
            tempBuffer[thread] += tempBuffer[thread+i];
    }
    if (thread == 0)
        *result = tempBuffer[0];
}

/**
 * Record the atomic charges into the posq array.
 */
__global__ void setCharges(real* __restrict__ charges, real4* __restrict__ posq, int* __restrict__ atomOrder, int numAtoms) {
    for (int i = blockDim.x*blockIdx.x+threadIdx.x; i < numAtoms; i += blockDim.x*gridDim.x)
        posq[i].w = charges[atomOrder[i]];
}
}
