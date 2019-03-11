__device__ void computeDirectFieldDampingFactors(real alpha, real r, real& fdamp3, real& fdamp5, real& fdamp7) {
    real ar = alpha*r;
    real ar2 = ar*ar;
    real ar3 = ar2*ar;
    real ar4 = ar2*ar2;
    real expAR = EXP(-ar);
    real one = 1;
    fdamp3 = 1 - (1 + ar + ar2*(one/2))*expAR;
    fdamp5 = 1 - (1 + ar + ar2*(one/2) + ar3*(one/6))*expAR;
    fdamp7 = 1 - (1 + ar + ar2*(one/2) + ar3*(one/6) + ar4*(one/30))*expAR;
}

__device__ void computeMutualFieldDampingFactors(real alphaI, real alphaJ, real r, real& fdamp3, real& fdamp5) {
    real arI = alphaI*r;
    real arI2 = arI*arI;
    real arI3 = arI2*arI;
    real expARI = EXP(-arI);
    real one = 1;
    real seven = 7;
    if (alphaI == alphaJ) {
        real arI4 = arI3*arI;
        real arI5 = arI4*arI;
        fdamp3 = 1 - (1 + arI + arI2*(one/2) + arI3*(seven/48) + arI4*(one/48))*expARI;
        fdamp5 = 1 - (1 + arI + arI2*(one/2) + arI3*(one/6) + arI4*(one/24) + arI5*(one/144))*expARI;
    }
    else {
        real arJ = alphaJ*r;
        real arJ2 = arJ*arJ;
        real arJ3 = arJ2*arJ;
        real expARJ = EXP(-arJ);
        real aI2 = alphaI*alphaI;
        real aJ2 = alphaJ*alphaJ;
        real A = aJ2/(aJ2-aI2);
        real B = aI2/(aI2-aJ2);
        real A2 = A*A;
        real B2 = B*B;
        fdamp3 = 1 - A2*(1 + arI + arI2*(one/2))*expARI -
                     B2*(1 + arJ + arJ2*(one/2))*expARJ -
                     2*A2*B*(1 + arI)*expARI -
                     2*B2*A*(1 + arJ)*expARJ;
        fdamp5 = 1 - A2*(1 + arI + arI2*(one/2) + arI3*(one/6))*expARI -
                     B2*(1 + arJ + arJ2*(one/2) + arJ3*(one/6))*expARJ -
                     2*A2*B*(1 + arI + arI2*(one/3))*expARI -
                     2*B2*A*(1 + arJ + arJ2*(one/3))*expARJ;
    }
}

typedef struct {
    real3 pos;
    real3 field;
    ATOM_PARAMETER_DATA
} AtomData;

/**
 * Compute a value based on pair interactions.
 */
extern "C" __global__ void computeField(const real4* __restrict__ posq, const unsigned int* __restrict__ exclusions,
        const ushort2* __restrict__ exclusionTiles, unsigned long long* __restrict__ fieldBuffers,
#ifdef USE_CUTOFF
        const int* __restrict__ tiles, const unsigned int* __restrict__ interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, unsigned int maxTiles, const real4* __restrict__ blockCenter,
        const real4* __restrict__ blockSize, const unsigned int* __restrict__ interactingAtoms
#else
        unsigned int numTiles
#endif
        PARAMETER_ARGUMENTS) {
    const unsigned int totalWarps = (blockDim.x*gridDim.x)/TILE_SIZE;
    const unsigned int warp = (blockIdx.x*blockDim.x+threadIdx.x)/TILE_SIZE;
    const unsigned int tgx = threadIdx.x & (TILE_SIZE-1);
    const unsigned int tbx = threadIdx.x - tgx;
    __shared__ AtomData localData[THREAD_BLOCK_SIZE];

    // First loop: process tiles that contain exclusions.
    
    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const ushort2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        real3 field = make_real3(0);
        unsigned int atom1 = x*TILE_SIZE + tgx;
        real4 pos1 = posq[atom1];
        LOAD_ATOM1_PARAMETERS
        unsigned int excl = exclusions[pos*TILE_SIZE+tgx];
        if (x == y) {
            // This tile is on the diagonal.

            const unsigned int localAtomIndex = threadIdx.x;
            localData[localAtomIndex].pos = make_real3(pos1.x, pos1.y, pos1.z);
            LOAD_LOCAL_PARAMETERS_FROM_1
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                int atom2 = tbx+j;
                real3 pos2 = localData[atom2].pos;
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
                    real3 tempField1 = make_real3(0);
                    real3 tempField2 = make_real3(0);
                    bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS || !(excl & 0x1));
                    if (!isExcluded && atom1 != atom2) {
                        COMPUTE_FIELD
                    }
                    field += tempField1;
#ifdef USE_CUTOFF
                }
#endif
                excl >>= 1;
            }
        }
        else {
            // This is an off-diagonal tile.

            const unsigned int localAtomIndex = threadIdx.x;
            unsigned int j = y*TILE_SIZE + tgx;
            real4 tempPosq = posq[j];
            localData[localAtomIndex].pos = make_real3(tempPosq.x, tempPosq.y, tempPosq.z);
            LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
            localData[localAtomIndex].field = make_real3(0);
            excl = (excl >> tgx) | (excl << (TILE_SIZE - tgx));
            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = tbx+tj;
                real3 pos2 = localData[atom2].pos;
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
                    real3 tempField1 = make_real3(0);
                    real3 tempField2 = make_real3(0);
                    bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS || !(excl & 0x1));
                    if (!isExcluded) {
                        COMPUTE_FIELD
                    }
                    field += tempField1;
                    localData[tbx+tj].field += tempField2;
#ifdef USE_CUTOFF
                }
#endif
                excl >>= 1;
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
        }

        // Write results.

        unsigned int offset1 = x*TILE_SIZE + tgx;
        atomicAdd(&fieldBuffers[offset1], static_cast<unsigned long long>((long long) (field.x*0x100000000)));
        atomicAdd(&fieldBuffers[offset1+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (field.y*0x100000000)));
        atomicAdd(&fieldBuffers[offset1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (field.z*0x100000000)));
        if (x != y) {
            unsigned int offset2 = y*TILE_SIZE + tgx;
            atomicAdd(&fieldBuffers[offset2], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.x*0x100000000)));
            atomicAdd(&fieldBuffers[offset2+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.y*0x100000000)));
            atomicAdd(&fieldBuffers[offset2+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.z*0x100000000)));
        }
    }

    // Second loop: tiles without exclusions, either from the neighbor list (with cutoff) or just enumerating all
    // of them (no cutoff).

#ifdef USE_CUTOFF
    unsigned int numTiles = interactionCount[0];
    if (numTiles > maxTiles)
        return; // There wasn't enough memory for the neighbor list.
    int pos = (int) (warp*(numTiles > maxTiles ? NUM_BLOCKS*((long long)NUM_BLOCKS+1)/2 : (long)numTiles)/totalWarps);
    int end = (int) ((warp+1)*(numTiles > maxTiles ? NUM_BLOCKS*((long long)NUM_BLOCKS+1)/2 : (long)numTiles)/totalWarps);
#else
    int pos = (int) (warp*(long long)numTiles/totalWarps);
    int end = (int) ((warp+1)*(long long)numTiles/totalWarps);
#endif
    int skipBase = 0;
    int currentSkipIndex = tbx;
    __shared__ int atomIndices[THREAD_BLOCK_SIZE];
    __shared__ volatile int skipTiles[THREAD_BLOCK_SIZE];
    skipTiles[threadIdx.x] = -1;
    
    while (pos < end) {
        real3 field = make_real3(0);
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

        while (skipTiles[tbx+TILE_SIZE-1] < pos) {
            if (skipBase+tgx < NUM_TILES_WITH_EXCLUSIONS) {
                ushort2 tile = exclusionTiles[skipBase+tgx];
                skipTiles[threadIdx.x] = tile.x + tile.y*NUM_BLOCKS - tile.y*(tile.y+1)/2;
            }
            else
                skipTiles[threadIdx.x] = end;
            skipBase += TILE_SIZE;            
            currentSkipIndex = tbx;
        }
        while (skipTiles[currentSkipIndex] < pos)
            currentSkipIndex++;
        includeTile = (skipTiles[currentSkipIndex] != pos);
#endif
        if (includeTile) {
            unsigned int atom1 = x*TILE_SIZE + tgx;

            // Load atom data for this tile.
            
            real4 pos1 = posq[atom1];
            LOAD_ATOM1_PARAMETERS
            const unsigned int localAtomIndex = threadIdx.x;
#ifdef USE_CUTOFF
            unsigned int j = interactingAtoms[pos*TILE_SIZE+tgx];
#else
            unsigned int j = y*TILE_SIZE + tgx;
#endif
            atomIndices[threadIdx.x] = j;
            if (j < PADDED_NUM_ATOMS) {
                real4 tempPosq = posq[j];
                localData[localAtomIndex].pos = make_real3(tempPosq.x, tempPosq.y, tempPosq.z);
                LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
                localData[localAtomIndex].field = make_real3(0);
            }
#ifdef USE_PERIODIC
            if (singlePeriodicCopy) {
                // The box is small enough that we can just translate all the atoms into a single periodic
                // box, then skip having to apply periodic boundary conditions later.

                real4 blockCenterX = blockCenter[x];
                APPLY_PERIODIC_TO_POS_WITH_CENTER(pos1, blockCenterX)
                APPLY_PERIODIC_TO_POS_WITH_CENTER(localData[threadIdx.x].pos, blockCenterX)
                unsigned int tj = tgx;
                for (unsigned int j = 0; j < TILE_SIZE; j++) {
                    int atom2 = tbx+tj;
                    real3 pos2 = localData[atom2].pos;
                    real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    if (r2 < CUTOFF_SQUARED) {
                        real invR = RSQRT(r2);
                        real r = r2*invR;
                        LOAD_ATOM2_PARAMETERS
                        atom2 = atomIndices[tbx+tj];
                        real3 tempField1 = make_real3(0);
                        real3 tempField2 = make_real3(0);
                        if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                            COMPUTE_FIELD
                        }
                        field += tempField1;
                        localData[tbx+tj].field += tempField2;
                    }
                    tj = (tj + 1) & (TILE_SIZE - 1);
                }
            }
            else
#endif
            {
                // We need to apply periodic boundary conditions separately for each interaction.

                unsigned int tj = tgx;
                for (unsigned int j = 0; j < TILE_SIZE; j++) {
                    int atom2 = tbx+tj;
                    real3 pos2 = localData[atom2].pos;
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
                        real3 tempField1 = make_real3(0);
                        real3 tempField2 = make_real3(0);
                        if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                            COMPUTE_FIELD
                        }
                        field += tempField1;
                        localData[tbx+tj].field += tempField2;
#ifdef USE_CUTOFF
                    }
#endif
                    tj = (tj + 1) & (TILE_SIZE - 1);
                }
            }
        
            // Write results.

            unsigned int offset1 = atom1;
            atomicAdd(&fieldBuffers[offset1], static_cast<unsigned long long>((long long) (field.x*0x100000000)));
            atomicAdd(&fieldBuffers[offset1+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (field.y*0x100000000)));
            atomicAdd(&fieldBuffers[offset1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (field.z*0x100000000)));
#ifdef USE_CUTOFF
            unsigned int atom2 = atomIndices[threadIdx.x];
#else
            unsigned int atom2 = y*TILE_SIZE + tgx;
#endif
            if (atom2 < PADDED_NUM_ATOMS) {
                unsigned int offset2 = atom2;
                atomicAdd(&fieldBuffers[offset2], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.x*0x100000000)));
                atomicAdd(&fieldBuffers[offset2+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.y*0x100000000)));
                atomicAdd(&fieldBuffers[offset2+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.z*0x100000000)));
            }
        }
        pos++;
    }
}
