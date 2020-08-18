/**
 * Compute a value based on pair interactions.
 */
KERNEL void computeN2Value(GLOBAL const real4* RESTRICT posq, GLOBAL const tileflags_t* RESTRICT exclusions,
        GLOBAL const ushort2* exclusionTiles,
#ifdef SUPPORTS_64_BIT_ATOMICS
        GLOBAL mm_ulong* RESTRICT global_value,
#else
        GLOBAL real* RESTRICT global_value,
#endif
#ifdef USE_CUTOFF
        GLOBAL const int* RESTRICT tiles, GLOBAL const unsigned int* RESTRICT interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, unsigned int maxTiles, GLOBAL const real4* RESTRICT blockCenter,
        GLOBAL const real4* RESTRICT blockSize, GLOBAL const int* RESTRICT interactingAtoms
#else
        unsigned int numTiles
#endif
        PARAMETER_ARGUMENTS) {
    const unsigned int totalWarps = GLOBAL_SIZE/TILE_SIZE;
    const unsigned int warp = GLOBAL_ID/TILE_SIZE;
    const unsigned int tgx = LOCAL_ID & (TILE_SIZE-1);
    const unsigned int tbx = LOCAL_ID - tgx;
    LOCAL real3 local_pos[LOCAL_BUFFER_SIZE];
    LOCAL real local_value[LOCAL_BUFFER_SIZE];
    ATOM_PARAMETER_DATA

    // First loop: process tiles that contain exclusions.

    const int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const ushort2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        real value = 0;
        unsigned int atom1 = x*TILE_SIZE + tgx;
        real3 pos1 = trimTo3(posq[atom1]);
        LOAD_ATOM1_PARAMETERS
#ifdef USE_EXCLUSIONS
        tileflags_t excl = exclusions[pos*TILE_SIZE+tgx];
#endif
        if (x == y) {
            // This tile is on the diagonal.

            const unsigned int localAtomIndex = LOCAL_ID;
            local_pos[localAtomIndex] = pos1;
            LOAD_LOCAL_PARAMETERS_FROM_1
            SYNC_WARPS;
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                int atom2 = tbx+j;
                real3 pos2 = local_pos[atom2];
                real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                if (r2 < CUTOFF_SQUARED) {
#endif
                    real invR = RSQRT(r2);
                    real r = r2*invR;
                    LOAD_ATOM2_PARAMETERS
                    atom2 = y*TILE_SIZE+j;
                    real tempValue1 = 0;
                    real tempValue2 = 0;
#ifdef USE_EXCLUSIONS
                    bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS || !(excl & static_cast<tileflags_t>(0x1)));
                    if (!isExcluded && atom1 != atom2) {
#else
                    if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 != atom2) {
#endif
                        COMPUTE_VALUE
                    }
                    value += tempValue1;
                    ADD_TEMP_DERIVS1
#ifdef USE_CUTOFF
                }
#endif
#ifdef USE_EXCLUSIONS
                excl >>= static_cast<tileflags_t>(1);
#endif
                SYNC_WARPS;
            }
        }
        else {
            // This is an off-diagonal tile.

            const unsigned int localAtomIndex = LOCAL_ID;
            unsigned int j = y*TILE_SIZE + tgx;
            local_pos[localAtomIndex] = trimTo3(posq[j]);
            LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
            local_value[localAtomIndex] = 0;
            SYNC_WARPS;
#ifdef USE_EXCLUSIONS
            excl = (excl >> static_cast<tileflags_t>(tgx)) | (excl << static_cast<tileflags_t>(TILE_SIZE - tgx));
#endif
            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = tbx+tj;
                real3 pos2 = local_pos[atom2];
                real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                if (r2 < CUTOFF_SQUARED) {
#endif
                    real invR = RSQRT(r2);
                    real r = r2*invR;
                    LOAD_ATOM2_PARAMETERS
                    atom2 = y*TILE_SIZE+tj;
                    real tempValue1 = 0;
                    real tempValue2 = 0;
#ifdef USE_EXCLUSIONS
                    bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS || !(excl & static_cast<tileflags_t>(0x1)));
                    if (!isExcluded) {
#else
                    if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
#endif
                        COMPUTE_VALUE
                    }
                    value += tempValue1;
                    local_value[tbx+tj] += tempValue2;
                    ADD_TEMP_DERIVS1
                    ADD_TEMP_DERIVS2
#ifdef USE_CUTOFF
                }
#endif
#ifdef USE_EXCLUSIONS
                excl >>= static_cast<tileflags_t>(1);
#endif
                tj = (tj + 1) & (TILE_SIZE - 1);
                SYNC_WARPS;
            }
        }

        // Write results.

#ifdef SUPPORTS_64_BIT_ATOMICS
        unsigned int offset1 = x*TILE_SIZE + tgx;
        ATOMIC_ADD(&global_value[offset1], (mm_ulong) ((mm_long) (value*0x100000000)));
        STORE_PARAM_DERIVS1
        if (x != y) {
            unsigned int offset2 = y*TILE_SIZE + tgx;
            ATOMIC_ADD(&global_value[offset2], (mm_ulong) ((mm_long) (local_value[LOCAL_ID]*0x100000000)));
            STORE_PARAM_DERIVS2
        }
#else
        unsigned int offset1 = x*TILE_SIZE + tgx + warp*PADDED_NUM_ATOMS;
        unsigned int offset2 = y*TILE_SIZE + tgx + warp*PADDED_NUM_ATOMS;
        global_value[offset1] += value;
        STORE_PARAM_DERIVS1
        if (x != y) {
            global_value[offset2] += local_value[LOCAL_ID];
            STORE_PARAM_DERIVS2
        }
#endif
    }

    // Second loop: tiles without exclusions, either from the neighbor list (with cutoff) or just enumerating all
    // of them (no cutoff).

#ifdef USE_CUTOFF
    unsigned int numTiles = interactionCount[0];
    if (numTiles > maxTiles)
        return; // There wasn't enough memory for the neighbor list.
    int pos = (int) (warp*(numTiles > maxTiles ? NUM_BLOCKS*((mm_long)NUM_BLOCKS+1)/2 : (mm_long)numTiles)/totalWarps);
    int end = (int) ((warp+1)*(numTiles > maxTiles ? NUM_BLOCKS*((mm_long)NUM_BLOCKS+1)/2 : (mm_long)numTiles)/totalWarps);
#else
    int pos = (int) (warp*(mm_long)numTiles/totalWarps);
    int end = (int) ((warp+1)*(mm_long)numTiles/totalWarps);
#endif
    int skipBase = 0;
    int currentSkipIndex = tbx;
    LOCAL int atomIndices[LOCAL_BUFFER_SIZE];
    LOCAL volatile int skipTiles[LOCAL_BUFFER_SIZE];
    skipTiles[LOCAL_ID] = -1;

    while (pos < end) {
        real value = 0;
        bool includeTile = true;

        // Extract the coordinates of this tile.

        int x, y;
        bool singlePeriodicCopy = false;
#ifdef USE_CUTOFF
        x = tiles[pos];
        real4 blockSizeX = blockSize[x];
        singlePeriodicCopy = (0.5f*periodicBoxSize.x-blockSizeX.x >= CUTOFF &&
                              0.5f*periodicBoxSize.y-blockSizeX.y >= CUTOFF &&
                              0.5f*periodicBoxSize.z-blockSizeX.z >= CUTOFF);
#else
        y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
        x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
        if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
            y += (x < y ? -1 : 1);
            x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
        }

        // Skip over tiles that have exclusions, since they were already processed.

        SYNC_WARPS;
        while (skipTiles[tbx+TILE_SIZE-1] < pos) {
            SYNC_WARPS;
            if (skipBase+tgx < NUM_TILES_WITH_EXCLUSIONS) {
                ushort2 tile = exclusionTiles[skipBase+tgx];
                skipTiles[LOCAL_ID] = tile.x + tile.y*NUM_BLOCKS - tile.y*(tile.y+1)/2;
            }
            else
                skipTiles[LOCAL_ID] = end;
            skipBase += TILE_SIZE;
            currentSkipIndex = tbx;
            SYNC_WARPS;
        }
        while (skipTiles[currentSkipIndex] < pos)
            currentSkipIndex++;
        includeTile = (skipTiles[currentSkipIndex] != pos);
#endif
        if (includeTile) {
            unsigned int atom1 = x*TILE_SIZE + tgx;

            // Load atom data for this tile.

            real3 pos1 = trimTo3(posq[atom1]);
            LOAD_ATOM1_PARAMETERS
            const unsigned int localAtomIndex = LOCAL_ID;
#ifdef USE_CUTOFF
            unsigned int j = interactingAtoms[pos*TILE_SIZE+tgx];
#else
            unsigned int j = y*TILE_SIZE + tgx;
#endif
            atomIndices[LOCAL_ID] = j;
            if (j < PADDED_NUM_ATOMS) {
                local_pos[localAtomIndex] = trimTo3(posq[j]);
                LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
                local_value[localAtomIndex] = 0;
            }
            SYNC_WARPS;
#ifdef USE_PERIODIC
            if (singlePeriodicCopy) {
                // The box is small enough that we can just translate all the atoms into a single periodic
                // box, then skip having to apply periodic boundary conditions later.

                real4 blockCenterX = blockCenter[x];
                APPLY_PERIODIC_TO_POS_WITH_CENTER(pos1, blockCenterX)
                APPLY_PERIODIC_TO_POS_WITH_CENTER(local_pos[LOCAL_ID], blockCenterX)
                SYNC_WARPS;
                unsigned int tj = tgx;
                for (j = 0; j < TILE_SIZE; j++) {
                    int atom2 = tbx+tj;
                    real3 pos2 = local_pos[atom2];
                    real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    if (r2 < CUTOFF_SQUARED) {
                        real invR = RSQRT(r2);
                        real r = r2*invR;
                        LOAD_ATOM2_PARAMETERS
                        atom2 = atomIndices[tbx+tj];
                        real tempValue1 = 0;
                        real tempValue2 = 0;
                        if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                            COMPUTE_VALUE
                        }
                        value += tempValue1;
                        local_value[tbx+tj] += tempValue2;
                        ADD_TEMP_DERIVS1
                        ADD_TEMP_DERIVS2
                    }
                    tj = (tj + 1) & (TILE_SIZE - 1);
                    SYNC_WARPS;
                }
            }
            else
#endif
            {
                // We need to apply periodic boundary conditions separately for each interaction.

                unsigned int tj = tgx;
                for (j = 0; j < TILE_SIZE; j++) {
                    int atom2 = tbx+tj;
                    real3 pos2 = local_pos[atom2];
                    real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#ifdef USE_PERIODIC
                    APPLY_PERIODIC_TO_DELTA(delta)
#endif
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                    if (r2 < CUTOFF_SQUARED) {
#endif
                        real invR = RSQRT(r2);
                        real r = r2*invR;
                        LOAD_ATOM2_PARAMETERS
                        atom2 = atomIndices[tbx+tj];
                        real tempValue1 = 0;
                        real tempValue2 = 0;
                        if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                            COMPUTE_VALUE
                        }
                        value += tempValue1;
                        local_value[tbx+tj] += tempValue2;
                        ADD_TEMP_DERIVS1
                        ADD_TEMP_DERIVS2
#ifdef USE_CUTOFF
                    }
#endif
                    tj = (tj + 1) & (TILE_SIZE - 1);
                    SYNC_WARPS;
                }
            }

            // Write results.

#ifdef USE_CUTOFF
            unsigned int atom2 = atomIndices[LOCAL_ID];
#else
            unsigned int atom2 = y*TILE_SIZE + tgx;
#endif
#ifdef SUPPORTS_64_BIT_ATOMICS
            unsigned int offset1 = atom1;
            ATOMIC_ADD(&global_value[offset1], (mm_ulong) ((mm_long) (value*0x100000000)));
            STORE_PARAM_DERIVS1
            if (atom2 < PADDED_NUM_ATOMS) {
                unsigned int offset2 = atom2;
                ATOMIC_ADD(&global_value[offset2], (mm_ulong) ((mm_long) (local_value[LOCAL_ID]*0x100000000)));
                STORE_PARAM_DERIVS2
            }
#else
            unsigned int offset1 = atom1 + warp*PADDED_NUM_ATOMS;
            global_value[offset1] += value;
            STORE_PARAM_DERIVS1
            if (atom2 < PADDED_NUM_ATOMS) {
                unsigned int offset2 = atom2 + warp*PADDED_NUM_ATOMS;
                global_value[offset2] += local_value[LOCAL_ID];
                STORE_PARAM_DERIVS2
            }
#endif
        }
        pos++;
    }
}
