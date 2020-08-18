#include "hip/hip_runtime.h"
__device__ real2 multofReal2(real2 a, real2 b) {
    return make_real2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

/**
 * Precompute the cosine and sine sums which appear in each force term.
 */

extern "C" __global__ void calculateEwaldCosSinSums(mixed* __restrict__ energyBuffer, const real4* __restrict__ posq, real2* __restrict__ cosSinSum, real4 periodicBoxSize) {
    const unsigned int ksizex = 2*KMAX_X-1;
    const unsigned int ksizey = 2*KMAX_Y-1;
    const unsigned int ksizez = 2*KMAX_Z-1;
    const unsigned int totalK = ksizex*ksizey*ksizez;
    real3 reciprocalBoxSize = make_real3(2*M_PI/periodicBoxSize.x, 2*M_PI/periodicBoxSize.y, 2*M_PI/periodicBoxSize.z);
    real reciprocalCoefficient = ONE_4PI_EPS0*4*M_PI/(periodicBoxSize.x*periodicBoxSize.y*periodicBoxSize.z);
    unsigned int index = blockIdx.x*blockDim.x+threadIdx.x;
    mixed energy = 0;
    while (index < (KMAX_Y-1)*ksizez+KMAX_Z)
        index += blockDim.x*gridDim.x;
    while (index < totalK) {
        // Find the wave vector (kx, ky, kz) this index corresponds to.

        int rx = index/(ksizey*ksizez);
        int remainder = index - rx*ksizey*ksizez;
        int ry = remainder/ksizez;
        int rz = remainder - ry*ksizez - KMAX_Z + 1;
        ry += -KMAX_Y + 1;
        real kx = rx*reciprocalBoxSize.x;
        real ky = ry*reciprocalBoxSize.y;
        real kz = rz*reciprocalBoxSize.z;

        // Compute the sum for this wave vector.

        real2 sum = make_real2(0);
        for (int atom = 0; atom < NUM_ATOMS; atom++) {
            real4 apos = posq[atom];
            real phase = apos.x*kx;
            real2 structureFactor = make_real2(COS(phase), SIN(phase));
            phase = apos.y*ky;
            structureFactor = multofReal2(structureFactor, make_real2(COS(phase), SIN(phase)));
            phase = apos.z*kz;
            structureFactor = multofReal2(structureFactor, make_real2(COS(phase), SIN(phase)));
            sum += apos.w*structureFactor;
        }
        cosSinSum[index] = sum;

        // Compute the contribution to the energy.

        real k2 = kx*kx + ky*ky + kz*kz;
        real ak = EXP(k2*EXP_COEFFICIENT) / k2;
        energy += reciprocalCoefficient*ak*(sum.x*sum.x + sum.y*sum.y);
        index += blockDim.x*gridDim.x;
    }
    energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] += energy;
}

/**
 * Compute the reciprocal space part of the Ewald force, using the precomputed sums from the
 * previous routine.
 */

extern "C" __global__ void calculateEwaldForces(unsigned long long* __restrict__ forceBuffers, const real4* __restrict__ posq, const real2* __restrict__ cosSinSum, real4 periodicBoxSize) {
    unsigned int atom = blockIdx.x*blockDim.x+threadIdx.x;
    real3 reciprocalBoxSize = make_real3(2*M_PI/periodicBoxSize.x, 2*M_PI/periodicBoxSize.y, 2*M_PI/periodicBoxSize.z);
    real reciprocalCoefficient = ONE_4PI_EPS0*4*M_PI/(periodicBoxSize.x*periodicBoxSize.y*periodicBoxSize.z);
    while (atom < NUM_ATOMS) {
        real3 force = make_real3(0);
        real4 apos = posq[atom];

        // Loop over all wave vectors.

        int lowry = 0;
        int lowrz = 1;
        for (int rx = 0; rx < KMAX_X; rx++) {
            real kx = rx*reciprocalBoxSize.x;
            for (int ry = lowry; ry < KMAX_Y; ry++) {
                real ky = ry*reciprocalBoxSize.y;
                real phase = apos.x*kx;
                real2 tab_xy = make_real2(COS(phase), SIN(phase));
                phase = apos.y*ky;
                tab_xy = multofReal2(tab_xy, make_real2(COS(phase), SIN(phase)));
                for (int rz = lowrz; rz < KMAX_Z; rz++) {
                    real kz = rz*reciprocalBoxSize.z;

                    // Compute the force contribution of this wave vector.

                    int index = rx*(KMAX_Y*2-1)*(KMAX_Z*2-1) + (ry+KMAX_Y-1)*(KMAX_Z*2-1) + (rz+KMAX_Z-1);
                    real k2 = kx*kx + ky*ky + kz*kz;
                    real ak = EXP(k2*EXP_COEFFICIENT)/k2;
                    phase = apos.z*kz;
                    real2 structureFactor = multofReal2(tab_xy, make_real2(COS(phase), SIN(phase)));
                    real2 sum = cosSinSum[index];
                    real dEdR = 2*reciprocalCoefficient*ak*apos.w*(sum.x*structureFactor.y - sum.y*structureFactor.x);
                    force.x += dEdR*kx;
                    force.y += dEdR*ky;
                    force.z += dEdR*kz;
                    lowrz = 1 - KMAX_Z;
                }
                lowry = 1 - KMAX_Y;
            }
        }

        // Record the force on the atom.

        atomicAdd(&forceBuffers[atom], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
        atomicAdd(&forceBuffers[atom+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
        atomicAdd(&forceBuffers[atom+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.z*0x100000000)));
        atom += blockDim.x*gridDim.x;
    }
}
