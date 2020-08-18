#include "hip/hip_runtime.h"
extern "C" __global__ void applyPositionDeltas(int numAtoms, real4* __restrict__ posq, real4* __restrict__ posqCorrection, mixed4* __restrict__ posDelta) {
    for (unsigned int index = blockIdx.x*blockDim.x+threadIdx.x; index < numAtoms; index += blockDim.x*gridDim.x) {
#ifdef USE_MIXED_PRECISION
        real4 pos1 = posq[index];
        real4 pos2 = posqCorrection[index];
        mixed4 pos = make_mixed4(pos1.x+(mixed)pos2.x, pos1.y+(mixed)pos2.y, pos1.z+(mixed)pos2.z, pos1.w);
#else
        mixed4 pos = posq[index];
#endif
        pos.x += posDelta[index].x;
        pos.y += posDelta[index].y;
        pos.z += posDelta[index].z;
#ifdef USE_MIXED_PRECISION
        posq[index] = make_real4((real) pos.x, (real) pos.y, (real) pos.z, (real) pos.w);
        posqCorrection[index] = make_real4(pos.x-(real) pos.x, pos.y-(real) pos.y, pos.z-(real) pos.z, 0);
#else
        posq[index] = pos;
#endif
    }
}
