/**
 * Record the force on an atom to global memory.
 */
inline DEVICE void storeForce(int atom, real3 force, GLOBAL mm_ulong* RESTRICT forceBuffers) {
    ATOMIC_ADD(&forceBuffers[atom], (mm_ulong) ((mm_long) (force.x*0x100000000)));
    ATOMIC_ADD(&forceBuffers[atom+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force.y*0x100000000)));
    ATOMIC_ADD(&forceBuffers[atom+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force.z*0x100000000)));
}

/**
 * Compute the difference between two vectors, taking periodic boundary conditions into account
 * and setting the fourth component to the squared magnitude.
 */
inline DEVICE real4 delta(real3 vec1, real3 vec2, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    real4 result = make_real4(vec1.x-vec2.x, vec1.y-vec2.y, vec1.z-vec2.z, 0.0f);
#ifdef USE_PERIODIC
    APPLY_PERIODIC_TO_DELTA(result)
#endif
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

        real3 crossProduct = trimTo3(cross(vec1, vec2));
        real scale = vec1.w*vec2.w;
        angle = ASIN(SQRT(dot(crossProduct, crossProduct)/scale));
        if (cosine < 0.0f)
            angle = M_PI-angle;
    }
    else
       angle = ACOS(cosine);
    return angle;
}

/**
 * Compute the cross product of two vectors, setting the fourth component to the squared magnitude.
 */
inline DEVICE real4 computeCross(real4 vec1, real4 vec2) {
    real3 cp = trimTo3(cross(vec1, vec2));
    return make_real4(cp.x, cp.y, cp.z, cp.x*cp.x+cp.y*cp.y+cp.z*cp.z);
}

/**
 * Determine whether a particular interaction is in the list of exclusions.
 */
inline DEVICE bool isInteractionExcluded(int atom1, int atom2, GLOBAL const int* RESTRICT exclusions, GLOBAL const int* RESTRICT exclusionStartIndex) {
    if (atom1 > atom2) {
        int temp = atom1;
        atom1 = atom2;
        atom2 = temp;
    }
    int first = exclusionStartIndex[atom1];
    int last = exclusionStartIndex[atom1+1];
    for (int i = last-1; i >= first; i--) {
        int excluded = exclusions[i];
        if (excluded == atom2)
            return true;
        if (excluded <= atom1)
            return false;
    }
    return false;
}

/**
 * Compute the interaction.
 */
KERNEL void computeInteraction(
        GLOBAL mm_ulong* RESTRICT forceBuffers, GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real4* RESTRICT posq,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ
#ifdef USE_CUTOFF
        , GLOBAL const int* RESTRICT neighbors, GLOBAL const int* RESTRICT neighborStartIndex
#endif
#ifdef USE_FILTERS
        , GLOBAL int* RESTRICT particleTypes, GLOBAL int* RESTRICT orderIndex, GLOBAL int* RESTRICT particleOrder
#endif
#ifdef USE_EXCLUSIONS
        , GLOBAL int* RESTRICT exclusions, GLOBAL int* RESTRICT exclusionStartIndex
#endif
        PARAMETER_ARGUMENTS) {
    mixed energy = 0;

    // Loop over particles to be the first one in the set.

    for (int p1 = GROUP_ID; p1 < NUM_ATOMS; p1 += NUM_GROUPS) {
#ifdef USE_CENTRAL_PARTICLE
        const int a1 = p1;
#else
        const int a1 = 0;
#endif
#ifdef USE_CUTOFF
        int firstNeighbor = neighborStartIndex[p1];
        int numNeighbors = neighborStartIndex[p1+1]-firstNeighbor;
#else
  #ifdef USE_CENTRAL_PARTICLE
        int numNeighbors = NUM_ATOMS;
  #else
        int numNeighbors = NUM_ATOMS-p1-1;
  #endif
#endif
        int numCombinations = NUM_CANDIDATE_COMBINATIONS;
        for (int index = LOCAL_ID; index < numCombinations; index += LOCAL_SIZE) {
            FIND_ATOMS_FOR_COMBINATION_INDEX;
            bool includeInteraction = IS_VALID_COMBINATION;
#ifdef USE_CUTOFF
            if (includeInteraction) {
                VERIFY_CUTOFF;
            }
#endif
#ifdef USE_FILTERS
            int order = orderIndex[COMPUTE_TYPE_INDEX];
            if (order == -1)
                includeInteraction = false;
#endif
#ifdef USE_EXCLUSIONS
            if (includeInteraction) {
                VERIFY_EXCLUSIONS;
            }
#endif
            if (includeInteraction) {
                PERMUTE_ATOMS;
                LOAD_PARTICLE_DATA;
                COMPUTE_INTERACTION;
            }
        }
    }
    energyBuffer[GLOBAL_ID] += energy;
}

/**
 * Find a bounding box for the atoms in each block.
 */
KERNEL void findBlockBounds(real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const real4* RESTRICT posq, GLOBAL real4* RESTRICT blockCenter, GLOBAL real4* RESTRICT blockBoundingBox, GLOBAL int* RESTRICT numNeighborPairs) {
    int index = GLOBAL_ID;
    int base = index*TILE_SIZE;
    while (base < NUM_ATOMS) {
        real4 pos = posq[base];
#ifdef USE_PERIODIC
        APPLY_PERIODIC_TO_POS(pos)
#endif
        real4 minPos = pos;
        real4 maxPos = pos;
        int last = min(base+TILE_SIZE, NUM_ATOMS);
        for (int i = base+1; i < last; i++) {
            pos = posq[i];
#ifdef USE_PERIODIC
            real4 center = 0.5f*(maxPos+minPos);
            APPLY_PERIODIC_TO_POS_WITH_CENTER(pos, center)
#endif
            minPos = make_real4(min(minPos.x,pos.x), min(minPos.y,pos.y), min(minPos.z,pos.z), 0);
            maxPos = make_real4(max(maxPos.x,pos.x), max(maxPos.y,pos.y), max(maxPos.z,pos.z), 0);
        }
        real4 blockSize = 0.5f*(maxPos-minPos);
        blockBoundingBox[index] = blockSize;
        blockCenter[index] = 0.5f*(maxPos+minPos);
        index += GLOBAL_SIZE;
        base = index*TILE_SIZE;
    }
    if (GROUP_ID == 0 && LOCAL_ID == 0)
        *numNeighborPairs = 0;
}

/**
 * Find a list of neighbors for each atom.
 */
KERNEL void findNeighbors(real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const real4* RESTRICT posq, GLOBAL const real4* RESTRICT blockCenter, GLOBAL const real4* RESTRICT blockBoundingBox, GLOBAL int2* RESTRICT neighborPairs,
        GLOBAL int* RESTRICT numNeighborPairs, GLOBAL int* RESTRICT numNeighborsForAtom, int maxNeighborPairs
#ifdef USE_EXCLUSIONS
        , GLOBAL const int* RESTRICT exclusions, GLOBAL const int* RESTRICT exclusionStartIndex
#endif
        ) {
    LOCAL real3 positionCache[FIND_NEIGHBORS_WORKGROUP_SIZE];
    int indexInWarp = LOCAL_ID%TILE_SIZE;
#if !defined(__CUDA_ARCH__) || !defined(__HIP_DEVICE_COMPILE__)
    LOCAL bool includeBlockFlags[FIND_NEIGHBORS_WORKGROUP_SIZE];
    int warpStart = LOCAL_ID-indexInWarp;
#endif
    for (int atom1 = GLOBAL_ID; atom1 < PADDED_NUM_ATOMS; atom1 += GLOBAL_SIZE) {
        // Load data for this atom.  Note that all threads in a warp are processing atoms from the same block.

        real3 pos1 = trimTo3(posq[atom1]);
        int block1 = atom1/TILE_SIZE;
        real4 blockCenter1 = blockCenter[block1];
        real4 blockSize1 = blockBoundingBox[block1];
        int totalNeighborsForAtom1 = 0;

        // Loop over atom blocks to search for neighbors.  The threads in a warp compare block1 against TILE_SIZE
        // other blocks in parallel.

#ifdef USE_CENTRAL_PARTICLE
        int startBlock = 0;
#else
        int startBlock = block1;
#endif
        for (int block2Base = startBlock; block2Base < NUM_BLOCKS; block2Base += TILE_SIZE) {
            int block2 = block2Base+indexInWarp;
            bool includeBlock2 = (block2 < NUM_BLOCKS);
            if (includeBlock2) {
                real4 blockCenter2 = blockCenter[block2];
                real4 blockSize2 = blockBoundingBox[block2];
                real4 blockDelta = blockCenter1-blockCenter2;
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(blockDelta)
#endif
                blockDelta.x = max((real) 0, fabs(blockDelta.x)-blockSize1.x-blockSize2.x);
                blockDelta.y = max((real) 0, fabs(blockDelta.y)-blockSize1.y-blockSize2.y);
                blockDelta.z = max((real) 0, fabs(blockDelta.z)-blockSize1.z-blockSize2.z);
                includeBlock2 &= (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z < CUTOFF_SQUARED);
            }

            // Loop over any blocks we identified as potentially containing neighbors.

#if defined(__CUDA_ARCH__) || defined (__HIP_DEVICE_COMPILE__)
            int includeBlockFlags = BALLOT(includeBlock2);
            while (includeBlockFlags != 0) {
                int i = __ffs(includeBlockFlags)-1;
                includeBlockFlags &= includeBlockFlags-1;
                {
#else
            includeBlockFlags[LOCAL_ID] = includeBlock2;
            SYNC_WARPS;
            for (int i = 0; i < TILE_SIZE; i++) {
                if (includeBlockFlags[warpStart+i]) {
#endif
                    int block2 = block2Base+i;

                    // Loop over atoms in this block.

                    int start = block2*TILE_SIZE;
                    int included[TILE_SIZE];
                    int numIncluded = 0;
                    SYNC_WARPS;
                    positionCache[LOCAL_ID] = trimTo3(posq[start+indexInWarp]);
                    SYNC_WARPS;
                    if (atom1 < NUM_ATOMS) {
                        for (int j = 0; j < TILE_SIZE; j++) {
                            int atom2 = start+j;
                            real3 pos2 = positionCache[LOCAL_ID-indexInWarp+j];

                            // Decide whether to include this atom pair in the neighbor list.

                            real4 atomDelta = delta(pos1, pos2, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#ifdef USE_CENTRAL_PARTICLE
                            bool includeAtom = (atom2 != atom1 && atom2 < NUM_ATOMS && atomDelta.w < CUTOFF_SQUARED);
#else
                            bool includeAtom = (atom2 > atom1 && atom2 < NUM_ATOMS && atomDelta.w < CUTOFF_SQUARED);
#endif
#ifdef USE_EXCLUSIONS
                            if (includeAtom)
                                includeAtom &= !isInteractionExcluded(atom1, atom2, exclusions, exclusionStartIndex);
#endif
                            if (includeAtom)
                                included[numIncluded++] = atom2;
                        }
                    }

                    // If we found any neighbors, store them to the neighbor list.

                    if (numIncluded > 0) {
                        int baseIndex = ATOMIC_ADD(numNeighborPairs, numIncluded);
                        if (baseIndex+numIncluded <= maxNeighborPairs)
                            for (int j = 0; j < numIncluded; j++)
                                neighborPairs[baseIndex+j] = make_int2(atom1, included[j]);
                        totalNeighborsForAtom1 += numIncluded;
                    }
                }
            }
        }
        if (atom1 < NUM_ATOMS)
            numNeighborsForAtom[atom1] = totalNeighborsForAtom1;
        SYNC_WARPS;
    }
}

/**
 * Sum the neighbor counts to compute the start position of each atom.  This kernel
 * is executed as a single work group.
 */
KERNEL void computeNeighborStartIndices(GLOBAL int* RESTRICT numNeighborsForAtom, GLOBAL int* RESTRICT neighborStartIndex,
            GLOBAL int* RESTRICT numNeighborPairs, int maxNeighborPairs) {
    LOCAL unsigned int posBuffer[COMPUTE_NEIGHBORS_WORKGROUP_SIZE];
    if (*numNeighborPairs > maxNeighborPairs) {
        // There wasn't enough memory for the neighbor list, so we'll need to rebuild it.  Set the neighbor start
        // indices to indicate no neighbors for any atom.

        for (int i = LOCAL_ID; i <= NUM_ATOMS; i += LOCAL_SIZE)
            neighborStartIndex[i] = 0;
        return;
    }
    unsigned int globalOffset = 0;
    for (unsigned int startAtom = 0; startAtom < NUM_ATOMS; startAtom += LOCAL_SIZE) {
        // Load the neighbor counts into local memory.

        unsigned int globalIndex = startAtom+LOCAL_ID;
        posBuffer[LOCAL_ID] = (globalIndex < NUM_ATOMS ? numNeighborsForAtom[globalIndex] : 0);
        SYNC_THREADS;

        // Perform a parallel prefix sum.

        for (unsigned int step = 1; step < LOCAL_SIZE; step *= 2) {
            unsigned int add = (LOCAL_ID >= step ? posBuffer[LOCAL_ID-step] : 0);
            SYNC_THREADS;
            posBuffer[LOCAL_ID] += add;
            SYNC_THREADS;
        }

        // Write the results back to global memory.

        if (globalIndex < NUM_ATOMS) {
            neighborStartIndex[globalIndex+1] = posBuffer[LOCAL_ID]+globalOffset;
            numNeighborsForAtom[globalIndex] = 0; // Clear this so the next kernel can use it as a counter
        }
        globalOffset += posBuffer[LOCAL_SIZE-1];
        SYNC_THREADS;
    }
    if (LOCAL_ID == 0)
        neighborStartIndex[0] = 0;
}

/**
 * Assemble the final neighbor list.
 */
KERNEL void copyPairsToNeighborList(GLOBAL const int2* RESTRICT neighborPairs, GLOBAL int* RESTRICT neighbors, GLOBAL int* RESTRICT numNeighborPairs,
            int maxNeighborPairs, GLOBAL int* RESTRICT numNeighborsForAtom, GLOBAL const int* RESTRICT neighborStartIndex) {
    int actualPairs = *numNeighborPairs;
    if (actualPairs > maxNeighborPairs)
        return; // There wasn't enough memory for the neighbor list, so we'll need to rebuild it.
    for (unsigned int index = GLOBAL_ID; index < actualPairs; index += GLOBAL_SIZE) {
        int2 pair = neighborPairs[index];
        int startIndex = neighborStartIndex[pair.x];
        int offset = ATOMIC_ADD(numNeighborsForAtom+pair.x, 1);
        neighbors[startIndex+offset] = pair.y;
    }
}
