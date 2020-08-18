/**
 * Calculate the center of mass momentum.
 */

KERNEL void calcCenterOfMassMomentum(int numAtoms, GLOBAL const mixed4* RESTRICT velm, GLOBAL float3* RESTRICT cmMomentum) {
    LOCAL float3 temp[BLOCK_SIZE];
    float3 cm = make_float3(0, 0, 0);
    for (int index = GLOBAL_ID; index < numAtoms; index += GLOBAL_SIZE) {
        mixed4 velocity = velm[index];
        if (velocity.w != 0) {
            mixed mass = RECIP(velocity.w);
            cm.x += (float) (velocity.x*mass);
            cm.y += (float) (velocity.y*mass);
            cm.z += (float) (velocity.z*mass);
        }
    }

    // Sum the threads in this group.

    int thread = LOCAL_ID;
    temp[thread] = cm;

    // Easier to cope with varying block / wavefront sizes w/o perf. penalty if
    // expressed as a constexpr reduction
    #if BLOCK_SIZE > WARP_SIZE
    SYNC_THREADS;
    #pragma unroll
    for (int i = BLOCK_SIZE/2; i > WARP_SIZE/2; i /= 2) {
        if (thread < i) {
            temp[thread] += temp[thread+i];
        }
        SYNC_THREADS;
    }
    #endif
    #pragma unroll
    for (int i = WARP_SIZE/2; i > 1; i /= 2) {
        if (thread < i) {
            temp[thread] += temp[thread+i];
        }
        SYNC_WARPS;
    }
    if (thread == 0)
        cmMomentum[GROUP_ID] = temp[thread]+temp[thread+1];
}

/**
 * Remove center of mass motion.
 */

KERNEL void removeCenterOfMassMomentum(int numAtoms, GLOBAL mixed4* RESTRICT velm, GLOBAL const float3* RESTRICT cmMomentum) {
    // First sum all of the momenta that were calculated by individual groups.

    LOCAL float3 temp[BLOCK_SIZE];
    float3 cm = make_float3(0, 0, 0);
    for (int index = LOCAL_ID; index < NUM_GROUPS; index += LOCAL_SIZE)
        cm += cmMomentum[index];
    int thread = LOCAL_ID;
    temp[thread] = cm;
     #if BLOCK_SIZE > WARP_SIZE
    SYNC_THREADS;
    #pragma unroll
    for (int i = BLOCK_SIZE/2; i > WARP_SIZE/2; i /= 2) {
        if (thread < i) {
            temp[thread] += temp[thread+i];
        }
        SYNC_THREADS;
    }
    #endif
    #pragma unroll
    for (int i = WARP_SIZE/2; i > 1; i /= 2) {
        if (thread < i) {
            temp[thread] += temp[thread+i];
        }
        SYNC_WARPS;
    }
    cm = make_float3(INVERSE_TOTAL_MASS*(temp[0].x+temp[1].x), INVERSE_TOTAL_MASS*(temp[0].y+temp[1].y), INVERSE_TOTAL_MASS*(temp[0].z+temp[1].z));

    // Now remove the center of mass velocity from each atom.

    for (int index = GLOBAL_ID; index < numAtoms; index += GLOBAL_SIZE) {
        velm[index].x -= cm.x;
        velm[index].y -= cm.y;
        velm[index].z -= cm.z;
    }
}
