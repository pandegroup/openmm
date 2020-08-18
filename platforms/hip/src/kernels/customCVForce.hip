#include "hip/hip_runtime.h"
/**
 * Copy the positions and velocities to the inner context.
 */
extern "C" __global__ void copyState(real4* posq, real4* posqCorrection, mixed4* velm, int* __restrict__ atomOrder,
        real4* innerPosq, real4* innerPosqCorrection, mixed4* innerVelm, int* __restrict__ innerInvAtomOrder,
        int numAtoms) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < numAtoms; i += blockDim.x*gridDim.x) {
        int index = innerInvAtomOrder[atomOrder[i]];
        innerPosq[index] = posq[i];
        innerVelm[index] = velm[i];
#ifdef USE_MIXED_PRECISION
        innerPosqCorrection[index] = posqCorrection[i];
#endif
    }
}

/**
 * Copy the forces back to the main context.
 */
extern "C" __global__ void copyForces(long long* forces, int* __restrict__ invAtomOrder, long long* innerForces,
        int* __restrict__ innerAtomOrder, int numAtoms, int paddedNumAtoms) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < numAtoms; i += blockDim.x*gridDim.x) {
        int index = invAtomOrder[innerAtomOrder[i]];
        forces[index] = innerForces[i];
        forces[index+paddedNumAtoms] = innerForces[i+paddedNumAtoms];
        forces[index+paddedNumAtoms*2] = innerForces[i+paddedNumAtoms*2];
    }
}

/**
 * Add all the forces from the CVs.
 */
extern "C" __global__ void addForces(long long* forces, int bufferSize
    PARAMETER_ARGUMENTS) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < bufferSize; i += blockDim.x*gridDim.x) {
        ADD_FORCES
    }
}
