/**
 * Compute the center of each group.
 */
KERNEL void computeGroupCenters(int numParticleGroups, GLOBAL const real4* RESTRICT posq, GLOBAL const int* RESTRICT groupParticles,
        GLOBAL const real* RESTRICT groupWeights, GLOBAL const int* RESTRICT groupOffsets, GLOBAL real4* RESTRICT centerPositions) {
    //
    LOCAL volatile real3 temp[BLOCK_SIZE];
    for (int group = GROUP_ID; group < numParticleGroups; group += NUM_GROUPS) {
        // The threads in this block work together to compute the center one group.

        int firstIndex = groupOffsets[group];
        int lastIndex = groupOffsets[group+1];
        real3 center = make_real3(0);
        for (int index = LOCAL_ID; index < lastIndex-firstIndex; index += BLOCK_SIZE) {
            int atom = groupParticles[firstIndex+index];
            real weight = groupWeights[firstIndex+index];
            real4 pos = posq[atom];
            center.x += weight*pos.x;
            center.y += weight*pos.y;
            center.z += weight*pos.z;
        }

        // Sum the values.

        int thread = LOCAL_ID;
        temp[thread].x = center.x;
        temp[thread].y = center.y;
        temp[thread].z = center.z;
        // Easier to cope with varying block / wavefront sizes w/o perf. penalty if
        // expressed as a constexpr reduction
        #if BLOCK_SIZE > WARP_SIZE
        SYNC_THREADS;
        #pragma unroll
        for (int i = BLOCK_SIZE/2; i > WARP_SIZE/2; i /= 2) {
            if (thread < i) {
                temp[thread].x += temp[thread+i].x;
                temp[thread].y += temp[thread+i].y;
                temp[thread].z += temp[thread+i].z;
            }
            SYNC_THREADS;
        }
        #endif
        #pragma unroll
        for (int i = WARP_SIZE/2; i > 1; i /= 2) {
            if (thread < i) {
                temp[thread].x += temp[thread+i].x;
                temp[thread].y += temp[thread+i].y;
                temp[thread].z += temp[thread+i].z;
            }
            SYNC_WARPS;
        }
        if (thread == 0) {
            centerPositions[group] = make_real4(temp[0].x+temp[1].x, temp[0].y+temp[1].y, temp[0].z+temp[1].z, 0);
        }
    }
}

/**
 * Compute the difference between two vectors, setting the fourth component to the squared magnitude.
 */
DEVICE real4 delta(real4 vec1, real4 vec2, bool periodic, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    real4 result = make_real4(vec1.x-vec2.x, vec1.y-vec2.y, vec1.z-vec2.z, 0);
    if (periodic)
        APPLY_PERIODIC_TO_DELTA(result);
    result.w = result.x*result.x + result.y*result.y + result.z*result.z;
    return result;
}

/**
 * Compute the angle between two vectors.  The w component of each vector should contain the squared magnitude.
 */
DEVICE real computeAngle(real4 vec1, real4 vec2) {
    real dotProduct = vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
    real cosine = dotProduct*RSQRT(vec1.w*vec2.w);
    real angle;
    if (cosine > 0.99f || cosine < -0.99f) {
        // We're close to the singularity in acos(), so take the cross product and use asin() instead.

        real3 crossProduct = cross(trimTo3(vec1), trimTo3(vec2));
        real scale = vec1.w*vec2.w;
        angle = ASIN(SQRT(dot(crossProduct, crossProduct)/scale));
        if (cosine < 0)
            angle = M_PI-angle;
    }
    else
       angle = ACOS(cosine);
    return angle;
}

/**
 * Compute the cross product of two vectors, setting the fourth component to the squared magnitude.
 */
DEVICE real4 computeCross(real4 vec1, real4 vec2) {
    real3 cp = cross(trimTo3(vec1), trimTo3(vec2));
    return make_real4(cp.x, cp.y, cp.z, cp.x*cp.x+cp.y*cp.y+cp.z*cp.z);
}

/**
 * Compute the forces on groups based on the bonds.
 */
KERNEL void computeGroupForces(int numParticleGroups, GLOBAL mm_ulong* RESTRICT groupForce, GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real4* RESTRICT centerPositions,
        GLOBAL const int* RESTRICT bondGroups, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ
        EXTRA_ARGS) {
    mixed energy = 0;
    INIT_PARAM_DERIVS
    for (int index = GLOBAL_ID; index < NUM_BONDS; index += GLOBAL_SIZE) {
        COMPUTE_FORCE
    }
    energyBuffer[GLOBAL_ID] += energy;
    SAVE_PARAM_DERIVS
}

/**
 * Apply the forces from the group centers to the individual atoms.
 */
KERNEL void applyForcesToAtoms(int numParticleGroups, GLOBAL const int* RESTRICT groupParticles, GLOBAL const real* RESTRICT groupWeights, GLOBAL const int* RESTRICT groupOffsets,
        GLOBAL const mm_long* RESTRICT groupForce, GLOBAL mm_ulong* RESTRICT atomForce) {
    for (int group = GROUP_ID; group < numParticleGroups; group += NUM_GROUPS) {
        mm_long fx = groupForce[group];
        mm_long fy = groupForce[group+numParticleGroups];
        mm_long fz = groupForce[group+numParticleGroups*2];
        int firstIndex = groupOffsets[group];
        int lastIndex = groupOffsets[group+1];
        for (int index = LOCAL_ID; index < lastIndex-firstIndex; index += LOCAL_SIZE) {
            int atom = groupParticles[firstIndex+index];
            real weight = groupWeights[firstIndex+index];
            ATOMIC_ADD(&atomForce[atom], (mm_ulong) ((mm_long) (fx*weight)));
            ATOMIC_ADD(&atomForce[atom+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (fy*weight)));
            ATOMIC_ADD(&atomForce[atom+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (fz*weight)));
        }
    }
}
