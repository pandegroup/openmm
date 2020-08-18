#include "hip/hip_runtime.h"
/**
 * Compute the nonbonded parameters for particles and exceptions.
 */
extern "C" __global__ void computeParameters(mixed* __restrict__ energyBuffer, bool includeSelfEnergy, real* __restrict__ globalParams,
        int numAtoms, const float4* __restrict__ baseParticleParams, real4* __restrict__ posq, real* __restrict__ charge,
        float2* __restrict__ sigmaEpsilon, float4* __restrict__ particleParamOffsets, int* __restrict__ particleOffsetIndices
#ifdef HAS_EXCEPTIONS
        , int numExceptions, const float4* __restrict__ baseExceptionParams, float4* __restrict__ exceptionParams,
        float4* __restrict__ exceptionParamOffsets, int* __restrict__ exceptionOffsetIndices
#endif
        ) {
    mixed energy = 0;

    // Compute particle parameters.
    
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < numAtoms; i += blockDim.x*gridDim.x) {
        float4 params = baseParticleParams[i];
#ifdef HAS_OFFSETS
        int start = particleOffsetIndices[i], end = particleOffsetIndices[i+1];
        for (int j = start; j < end; j++) {
            float4 offset = particleParamOffsets[j];
            real value = globalParams[(int) offset.w];
            params.x += value*offset.x;
            params.y += value*offset.y;
            params.z += value*offset.z;
        }
#endif
#ifdef USE_POSQ_CHARGES
        posq[i].w = params.x;
#else
        charge[i] = params.x;
#endif
        sigmaEpsilon[i] = make_float2(0.5f*params.y, 2*SQRT(params.z));
#ifdef HAS_OFFSETS
    #ifdef INCLUDE_EWALD
        energy -= EWALD_SELF_ENERGY_SCALE*params.x*params.x;
    #endif
    #ifdef INCLUDE_LJPME
        real sig3 = params.y*params.y*params.y;
        energy += LJPME_SELF_ENERGY_SCALE*sig3*sig3*params.z;
    #endif
#endif
    }

    // Compute exception parameters.
    
#ifdef HAS_EXCEPTIONS
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < numExceptions; i += blockDim.x*gridDim.x) {
        float4 params = baseExceptionParams[i];
#ifdef HAS_OFFSETS
        int start = exceptionOffsetIndices[i], end = exceptionOffsetIndices[i+1];
        for (int j = start; j < end; j++) {
            float4 offset = exceptionParamOffsets[j];
            real value = globalParams[(int) offset.w];
            params.x += value*offset.x;
            params.y += value*offset.y;
            params.z += value*offset.z;
        }
#endif
        exceptionParams[i] = make_float4((float) (138.935456f*params.x), (float) params.y, (float) (4*params.z), 0);
    }
#endif
    if (includeSelfEnergy)
        energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] += energy;
}

/**
 * Compute parameters for subtracting the reciprocal part of excluded interactions.
 */
extern "C" __global__ void computeExclusionParameters(real4* __restrict__ posq, real* __restrict__ charge, float2* __restrict__ sigmaEpsilon,
        int numExclusions, const int2* __restrict__ exclusionAtoms, float4* __restrict__ exclusionParams) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < numExclusions; i += blockDim.x*gridDim.x) {
        int2 atoms = exclusionAtoms[i];
#ifdef USE_POSQ_CHARGES
        real chargeProd = posq[atoms.x].w*posq[atoms.y].w;
#else
        real chargeProd = charge[atoms.x]*charge[atoms.y];
#endif
#ifdef INCLUDE_LJPME
        float2 sigEps1 = sigmaEpsilon[atoms.x];
        float2 sigEps2 = sigmaEpsilon[atoms.y];
        float sigma = sigEps1.x*sigEps2.x;
        float epsilon = sigEps1.y*sigEps2.y;
#else
        float sigma = 0;
        float epsilon = 0;
#endif
        exclusionParams[i] = make_float4((float) (138.935456f*chargeProd), sigma, epsilon, 0);
    }
}