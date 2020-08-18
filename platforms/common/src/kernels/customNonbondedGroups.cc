typedef struct {
    real x, y, z;
    real q;
    real fx, fy, fz;
    ATOM_PARAMETER_DATA
#ifndef PARAMETER_SIZE_IS_EVEN
    real padding;
#endif
} AtomData;

/**
 * Find the maximum of a value across all threads in a warp, and return that to
 * every thread.
 */
DEVICE tileflags_t reduceMax(tileflags_t val, LOCAL_ARG tileflags_t* temp) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    // CUDA lets us do this slightly more efficiently by using shuffle operations.
    for (int mask = TILE_SIZE/2; mask > 0; mask /= 2)
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
#elif defined(__HIP_DEVICE_COMPILE___)
    // note that the HIP/NVCC case would be handled by the above
    for (int mask = TILE_SIZE/2; mask > 0; mask /= 2)
        val = max(val, __shfl_xor(val, mask));
    return val;
#else
    int indexInWarp = LOCAL_ID%TILE_SIZE;
    temp[LOCAL_ID] = val;
    SYNC_WARPS;
    for (int offset = TILE_SIZE/2; offset > 0; offset /= 2) {
        if (indexInWarp < offset)
            temp[LOCAL_ID] = max(temp[LOCAL_ID], temp[LOCAL_ID+offset]);
        SYNC_WARPS;
    }
    return temp[LOCAL_ID-indexInWarp];
#endif
}

#ifndef SUPPORTS_64_BIT_ATOMICS
/**
 * This function is used on devices that don't support 64 bit atomics.  Multiple threads within
 * a single tile might have computed forces on the same atom.  This loops over them and makes sure
 * that only one thread updates the force on any given atom.
 */
void writeForces(GLOBAL real4* forceBuffers, LOCAL AtomData* localData, int atomIndex) {
    localData[LOCAL_ID].x = atomIndex;
    SYNC_WARPS;
    real4 forceSum = make_real4(0);
    int start = (LOCAL_ID/TILE_SIZE)*TILE_SIZE;
    int end = start+TILE_SIZE;
    bool isFirst = true;
    for (int i = start; i < end; i++)
        if (localData[i].x == atomIndex) {
            forceSum += (real4) (localData[i].fx, localData[i].fy, localData[i].fz, 0);
            isFirst &= (i >= LOCAL_ID);
        }
    const unsigned int warp = GLOBAL_ID/TILE_SIZE;
    unsigned int offset = atomIndex + warp*PADDED_NUM_ATOMS;
    if (isFirst)
        forceBuffers[offset] += forceSum;
    SYNC_WARPS;
}
#endif

KERNEL void computeInteractionGroups(
#ifdef SUPPORTS_64_BIT_ATOMICS
        GLOBAL mm_ulong* RESTRICT forceBuffers,
#else
        GLOBAL real4* RESTRICT forceBuffers,
#endif
        GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real4* RESTRICT posq, GLOBAL const excl4* RESTRICT groupData,
        GLOBAL const int* RESTRICT numGroupTiles, int useNeighborList,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ
        PARAMETER_ARGUMENTS) {
    const unsigned int totalWarps = GLOBAL_SIZE/TILE_SIZE;
    const unsigned int warp = GLOBAL_ID/TILE_SIZE; // global warpIndex
    const unsigned int tgx = LOCAL_ID & (TILE_SIZE-1); // index within the warp
    const unsigned int tbx = LOCAL_ID - tgx;           // block warpIndex
    mixed energy = 0;
    INIT_DERIVATIVES
    LOCAL AtomData localData[LOCAL_MEMORY_SIZE];
    LOCAL tileflags_t reductionBuffer[LOCAL_MEMORY_SIZE];

    const unsigned int startTile = (useNeighborList ? warp*numGroupTiles[0]/totalWarps : FIRST_TILE+warp*(LAST_TILE-FIRST_TILE)/totalWarps);
    const unsigned int endTile = (useNeighborList ? (warp+1)*numGroupTiles[0]/totalWarps : FIRST_TILE+(warp+1)*(LAST_TILE-FIRST_TILE)/totalWarps);
    for (int tile = startTile; tile < endTile; tile++) {
        const excl4 atomData = groupData[TILE_SIZE*tile+tgx];
        const int atom1 = atomData.x;
        const int atom2 = atomData.y;
        const tileflags_t rangeStart = atomData.z&SIMD_HALF_MASK;
        const tileflags_t rangeEnd = (atomData.z>>SIMD_HALF_WIDTH)&SIMD_HALF_MASK;
        const tileflags_t exclusions = atomData.w;
        real4 posq1 = posq[atom1];
        LOAD_ATOM1_PARAMETERS
        real3 force = make_real3(0);
        real4 posq2 = posq[atom2];
        localData[LOCAL_ID].x = posq2.x;
        localData[LOCAL_ID].y = posq2.y;
        localData[LOCAL_ID].z = posq2.z;
        localData[LOCAL_ID].q = posq2.w;
        LOAD_LOCAL_PARAMETERS
        localData[LOCAL_ID].fx = 0.0f;
        localData[LOCAL_ID].fy = 0.0f;
        localData[LOCAL_ID].fz = 0.0f;
        tileflags_t tj = tgx;
        tileflags_t rangeStop = rangeStart + reduceMax(rangeEnd-rangeStart, reductionBuffer);
        SYNC_WARPS;
        for (tileflags_t j = rangeStart; j < rangeStop; j++) {
            if (j < rangeEnd) {
                bool isExcluded = (((exclusions>>static_cast<tileflags_t>(tj))&1) == 0);
                int localIndex = tbx+tj;
                posq2 = make_real4(localData[localIndex].x, localData[localIndex].y, localData[localIndex].z, localData[localIndex].q);
                real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                if (!isExcluded && r2 < CUTOFF_SQUARED) {
#endif
                    real invR = RSQRT(r2);
                    real r = r2*invR;
                    LOAD_ATOM2_PARAMETERS
                    real dEdR = 0.0f;
                    real tempEnergy = 0.0f;
                    const real interactionScale = 1.0f;
                    COMPUTE_INTERACTION
                    energy += tempEnergy;
                    delta *= dEdR;
                    force.x -= delta.x;
                    force.y -= delta.y;
                    force.z -= delta.z;
                    localData[localIndex].fx += delta.x;
                    localData[localIndex].fy += delta.y;
                    localData[localIndex].fz += delta.z;
#ifdef USE_CUTOFF
                }
#endif
                tj = (tj == rangeEnd-1 ? rangeStart : tj+1);
            }
            SYNC_WARPS;
        }
#ifdef SUPPORTS_64_BIT_ATOMICS
        if (exclusions != 0) {
            ATOMIC_ADD(&forceBuffers[atom1], (mm_ulong) ((mm_long) (force.x*0x100000000)));
            ATOMIC_ADD(&forceBuffers[atom1+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force.y*0x100000000)));
            ATOMIC_ADD(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force.z*0x100000000)));
        }
        ATOMIC_ADD(&forceBuffers[atom2], (mm_ulong) ((mm_long) (localData[LOCAL_ID].fx*0x100000000)));
        ATOMIC_ADD(&forceBuffers[atom2+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (localData[LOCAL_ID].fy*0x100000000)));
        ATOMIC_ADD(&forceBuffers[atom2+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (localData[LOCAL_ID].fz*0x100000000)));
        SYNC_WARPS;
#else
        writeForces(forceBuffers, localData, atom2);
        localData[LOCAL_ID].fx = force.x;
        localData[LOCAL_ID].fy = force.y;
        localData[LOCAL_ID].fz = force.z;
        writeForces(forceBuffers, localData, atom1);
#endif
    }
    energyBuffer[GLOBAL_ID] += energy;
    SAVE_DERIVATIVES
}

/**
 * If the neighbor list needs to be rebuilt, reset the number of tiles to 0.  This is
 * executed by a single thread.
 */
KERNEL void prepareToBuildNeighborList(GLOBAL int* RESTRICT rebuildNeighborList, GLOBAL int* RESTRICT numGroupTiles) {
    if (rebuildNeighborList[0] == 1)
        numGroupTiles[0] = 0;
}

/**
 * Filter the list of tiles to include only ones that have interactions within the
 * padded cutoff.
 */
KERNEL void buildNeighborList(GLOBAL int* RESTRICT rebuildNeighborList, GLOBAL int* RESTRICT numGroupTiles,
        GLOBAL const real4* RESTRICT posq, GLOBAL const excl4* RESTRICT groupData, GLOBAL excl4* RESTRICT filteredGroupData,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {

    // If the neighbor list doesn't need to be rebuilt on this step, return immediately.

    if (rebuildNeighborList[0] == 0)
        return;

    const unsigned int totalWarps = GLOBAL_SIZE/TILE_SIZE;
    const unsigned int warp = GLOBAL_ID/TILE_SIZE; // global warpIndex
    const unsigned int local_warp = LOCAL_ID/TILE_SIZE; // local warpIndex
    const unsigned int tgx = LOCAL_ID & (TILE_SIZE-1); // index within the warp
    const unsigned int tbx = LOCAL_ID - tgx;           // block warpIndex
    LOCAL real4 localPos[LOCAL_MEMORY_SIZE];
    LOCAL volatile bool anyInteraction[WARPS_IN_BLOCK];
    LOCAL volatile int tileIndex[WARPS_IN_BLOCK];
    LOCAL tileflags_t reductionBuffer[LOCAL_MEMORY_SIZE];

    const unsigned int startTile = warp*NUM_TILES/totalWarps;
    const unsigned int endTile = (warp+1)*NUM_TILES/totalWarps;
    for (int tile = startTile; tile < endTile; tile++) {
        const excl4 atomData = groupData[TILE_SIZE*tile+tgx];
        const int atom1 = atomData.x;
        const int atom2 = atomData.y;
        const tileflags_t rangeStart = atomData.z&SIMD_HALF_MASK;
        const tileflags_t rangeEnd = (atomData.z>>SIMD_HALF_WIDTH)&SIMD_HALF_MASK;
        const tileflags_t exclusions = atomData.w;
        real4 posq1 = posq[atom1];
        localPos[LOCAL_ID] = posq[atom2];
        if (tgx == 0)
            anyInteraction[local_warp] = false;
        tileflags_t tj = tgx;
        tileflags_t rangeStop = rangeStart + reduceMax(rangeEnd-rangeStart, reductionBuffer);
        SYNC_WARPS;
        for (tileflags_t j = rangeStart; j < rangeStop && !anyInteraction[local_warp]; j++) {
            SYNC_WARPS;
            if (j < rangeEnd && tj < rangeEnd) {
                bool isExcluded = (((exclusions>>static_cast<tileflags_t>(tj))&1) == 0);
                int localIndex = tbx+tj;
                real3 delta = make_real3(localPos[localIndex].x-posq1.x, localPos[localIndex].y-posq1.y, localPos[localIndex].z-posq1.z);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                if (!isExcluded && r2 < PADDED_CUTOFF_SQUARED)
                    anyInteraction[local_warp] = true;
            }
            tj = (tj == rangeEnd-1 ? rangeStart : tj+1);
            SYNC_WARPS;
        }
        if (anyInteraction[local_warp]) {
            SYNC_WARPS;
            if (tgx == 0)
                tileIndex[local_warp] = ATOMIC_ADD(numGroupTiles, 1);
            SYNC_WARPS;
            filteredGroupData[TILE_SIZE*tileIndex[local_warp]+tgx] = atomData;
        }
    }
}
