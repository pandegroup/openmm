#define NEIGHBOR_BLOCK_SIZE TILE_SIZE

/**
 * Calculate the ellipsoid coordinate frames and associated matrices.
 */
KERNEL void computeEllipsoidFrames(int numParticles, GLOBAL const real4* RESTRICT posq, GLOBAL int2* const RESTRICT axisParticleIndices,
        GLOBAL const float4* RESTRICT sigParams, GLOBAL const float4* RESTRICT scale, GLOBAL real* RESTRICT aMatrix,
        GLOBAL real* RESTRICT bMatrix, GLOBAL real* RESTRICT gMatrix, GLOBAL const int* sortedParticles) {
    for (int sortedIndex = GLOBAL_ID; sortedIndex < numParticles; sortedIndex += GLOBAL_SIZE) {
        // Compute the local coordinate system of the ellipsoid;

        int originalIndex = sortedParticles[sortedIndex];
        real3 pos = trimTo3(posq[originalIndex]);
        int2 axisParticles = axisParticleIndices[originalIndex];
        real3 xdir, ydir, zdir;
        if (axisParticles.x == -1) {
            xdir = make_real3(1, 0, 0);
            ydir = make_real3(0, 1, 0);
        }
        else {
            xdir = pos-trimTo3(posq[axisParticles.x]);
            xdir = normalize(xdir);
            if (axisParticles.y == -1) {
                if (xdir.y > -0.5f && xdir.y < 0.5f)
                    ydir = make_real3(0, 1, 0);
                else
                    ydir = make_real3(1, 0, 0);
            }
            else
                ydir = pos-trimTo3(posq[axisParticles.y]);
            ydir -= xdir*dot(xdir, ydir);
            ydir = normalize(ydir);
        }
        zdir = cross(xdir, ydir);

        // Compute matrices we will need later.

        GLOBAL real (*a)[3] = (GLOBAL real (*)[3]) (aMatrix+sortedIndex*9);
        GLOBAL real (*b)[3] = (GLOBAL real (*)[3]) (bMatrix+sortedIndex*9);
        GLOBAL real (*g)[3] = (GLOBAL real (*)[3]) (gMatrix+sortedIndex*9);
        a[0][0] = xdir.x;
        a[0][1] = xdir.y;
        a[0][2] = xdir.z;
        a[1][0] = ydir.x;
        a[1][1] = ydir.y;
        a[1][2] = ydir.z;
        a[2][0] = zdir.x;
        a[2][1] = zdir.y;
        a[2][2] = zdir.z;
        float4 sig = sigParams[originalIndex];
        float3 r2 = make_float3(sig.y, sig.z, sig.w);
        float3 e2 = trimTo3(scale[originalIndex]);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                b[i][j] = a[0][i]*e2.x*a[0][j] + a[1][i]*e2.y*a[1][j] + a[2][i]*e2.z*a[2][j];
                g[i][j] = a[0][i]*r2.x*a[0][j] + a[1][i]*r2.y*a[1][j] + a[2][i]*r2.z*a[2][j];
            }
    }
}

/**
 * Find a bounding box for the atoms in each block.
 */
KERNEL void findBlockBounds(int numAtoms, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const int* sortedAtoms, GLOBAL const real4* RESTRICT posq, GLOBAL real4* RESTRICT sortedPos, GLOBAL real4* RESTRICT blockCenter,
        GLOBAL real4* RESTRICT blockBoundingBox, GLOBAL int* RESTRICT neighborBlockCount) {
    int index = GLOBAL_ID;
    int base = index*TILE_SIZE;
    while (base < numAtoms) {
        real4 pos = posq[sortedAtoms[base]];
        sortedPos[base] = pos;
#ifdef USE_PERIODIC
        APPLY_PERIODIC_TO_POS(pos)
#endif
        real4 minPos = pos;
        real4 maxPos = pos;
        int last = min(base+TILE_SIZE, numAtoms);
        for (int i = base+1; i < last; i++) {
            pos = posq[sortedAtoms[i]];
            sortedPos[i] = pos;
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
    if (GLOBAL_ID == 0)
        *neighborBlockCount = 0;
}

/**
 * This is called by findNeighbors() to write a block to the neighbor list.
 */
DEVICE void storeNeighbors(int atom1, int* neighborBuffer, int numAtomsInBuffer, int maxNeighborBlocks, GLOBAL int* RESTRICT neighbors,
        GLOBAL int* RESTRICT neighborIndex, GLOBAL int* RESTRICT neighborBlockCount) {
    int blockIndex = ATOMIC_ADD(neighborBlockCount, 1);
    if (blockIndex >= maxNeighborBlocks)
        return; // We don't have enough room for the neighbor list.
    neighborIndex[blockIndex] = atom1;
    int baseIndex = blockIndex*NEIGHBOR_BLOCK_SIZE;
    for (int i = 0; i < numAtomsInBuffer; i++)
        neighbors[baseIndex+i] = neighborBuffer[i];
    for (int i = numAtomsInBuffer; i < NEIGHBOR_BLOCK_SIZE; i++)
        neighbors[baseIndex+i] = -1;
}

/**
 * Build a list of neighbors for each atom.
 */
KERNEL void findNeighbors(int numAtoms, int maxNeighborBlocks, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL real4* RESTRICT sortedPos, GLOBAL real4* RESTRICT blockCenter, GLOBAL real4* RESTRICT blockBoundingBox, GLOBAL int* RESTRICT neighbors,
        GLOBAL int* RESTRICT neighborIndex, GLOBAL int* RESTRICT neighborBlockCount, GLOBAL const int* RESTRICT exclusions, GLOBAL const int* RESTRICT exclusionStartIndex) {
    const int numBlocks = (numAtoms+TILE_SIZE-1)/TILE_SIZE;
    int neighborBuffer[NEIGHBOR_BLOCK_SIZE];
    for (int atom1 = GLOBAL_ID; atom1 < numAtoms; atom1 += GLOBAL_SIZE) {
        int nextExclusion = exclusionStartIndex[atom1];
        int lastExclusion = exclusionStartIndex[atom1+1];
        real4 pos = sortedPos[atom1];
        int nextBufferIndex = 0;

        // Loop over atom blocks and compute the distance of this atom from each one's bounding box.

        for (int block = (atom1+1)/TILE_SIZE; block < numBlocks; block++) {
            real4 center = blockCenter[block];
            real4 blockSize = blockBoundingBox[block];
            real4 blockDelta = center-pos;
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(blockDelta)
#endif
            blockDelta.x = max((real) 0, fabs(blockDelta.x)-blockSize.x);
            blockDelta.y = max((real) 0, fabs(blockDelta.y)-blockSize.y);
            blockDelta.z = max((real) 0, fabs(blockDelta.z)-blockSize.z);
            if (blockDelta.x*blockDelta.x+blockDelta.y*blockDelta.y+blockDelta.z*blockDelta.z >= CUTOFF_SQUARED)
                continue;

            // Loop over atoms within this block.

            int first = max(block*TILE_SIZE, atom1+1);
            int last = min((block+1)*TILE_SIZE, numAtoms);
            for (int atom2 = first; atom2 < last; atom2++) {
                // Skip over excluded interactions.

                if (nextExclusion < lastExclusion && exclusions[nextExclusion] >= atom2) {
                    nextExclusion++;
                    continue;
                }
                real4 delta = pos-sortedPos[atom2];
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                if (r2 < CUTOFF_SQUARED) {
                    neighborBuffer[nextBufferIndex++] = atom2;
                    if (nextBufferIndex == NEIGHBOR_BLOCK_SIZE) {
                        storeNeighbors(atom1, neighborBuffer, nextBufferIndex, maxNeighborBlocks, neighbors, neighborIndex, neighborBlockCount);
                        nextBufferIndex = 0;
                    }
                }
            }
        }
        if (nextBufferIndex > 0)
            storeNeighbors(atom1, neighborBuffer, nextBufferIndex, maxNeighborBlocks, neighbors, neighborIndex, neighborBlockCount);
    }
}

typedef struct {
    float4 sig;
    float2 eps;
    real3 pos;
    real a[3][3], b[3][3], g[3][3];
} AtomData;

DEVICE void loadAtomData(AtomData* data, int sortedIndex, int originalIndex, GLOBAL const real4* RESTRICT pos, GLOBAL const float4* RESTRICT sigParams,
        GLOBAL const float2* RESTRICT epsParams, GLOBAL const real* RESTRICT aMatrix, GLOBAL const real* RESTRICT bMatrix, GLOBAL const real* RESTRICT gMatrix) {
    data->sig = sigParams[originalIndex];
    data->eps = epsParams[originalIndex];
    data->pos = trimTo3(pos[sortedIndex]);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            int k = 9*sortedIndex+3*i+j;
            data->a[i][j] = aMatrix[k];
            data->b[i][j] = bMatrix[k];
            data->g[i][j] = gMatrix[k];
        }
}

inline DEVICE real3 matrixVectorProduct(real (*m)[3], real3 v) {
    return make_real3(m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
                      m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
                      m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z);
}

inline DEVICE real3 vectorMatrixProduct(real3 v, real (*m)[3]) {
    return make_real3(m[0][0]*v.x + m[1][0]*v.y + m[2][0]*v.z,
                      m[0][1]*v.x + m[1][1]*v.y + m[2][1]*v.z,
                      m[0][2]*v.x + m[1][2]*v.y + m[2][2]*v.z);
}

inline DEVICE void matrixSum(real (*result)[3], real (*a)[3], real (*b)[3]) {
    result[0][0] = a[0][0]+b[0][0];
    result[0][1] = a[0][1]+b[0][1];
    result[0][2] = a[0][2]+b[0][2];
    result[1][0] = a[1][0]+b[1][0];
    result[1][1] = a[1][1]+b[1][1];
    result[1][2] = a[1][2]+b[1][2];
    result[2][0] = a[2][0]+b[2][0];
    result[2][1] = a[2][1]+b[2][1];
    result[2][2] = a[2][2]+b[2][2];
}

inline DEVICE real determinant(real (*m)[3]) {
    return (m[0][0]*m[1][1]*m[2][2] + m[0][1]*m[1][2]*m[2][0] + m[0][2]*m[1][0]*m[2][1] -
            m[0][0]*m[1][2]*m[2][1] - m[0][1]*m[1][0]*m[2][2] - m[0][2]*m[1][1]*m[2][0]);
}

inline DEVICE void matrixInverse(real (*result)[3], real (*m)[3]) {
    real invDet = RECIP(determinant(m));
    result[0][0] = invDet*(m[1][1]*m[2][2] - m[1][2]*m[2][1]);
    result[1][0] = -invDet*(m[1][0]*m[2][2] - m[1][2]*m[2][0]);
    result[2][0] = invDet*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
    result[0][1] = -invDet*(m[0][1]*m[2][2] - m[0][2]*m[2][1]);
    result[1][1] = invDet*(m[0][0]*m[2][2] - m[0][2]*m[2][0]);
    result[2][1] = -invDet*(m[0][0]*m[2][1] - m[0][1]*m[2][0]);
    result[0][2] = invDet*(m[0][1]*m[1][2] - m[0][2]*m[1][1]);
    result[1][2] = -invDet*(m[0][0]*m[1][2] - m[0][2]*m[1][0]);
    result[2][2] = invDet*(m[0][0]*m[1][1] - m[0][1]*m[1][0]);
}

DEVICE void computeOneInteraction(AtomData* data1, AtomData* data2, real sigma, real epsilon, real3 dr, real r2, real3* force1, real3* force2, real3* torque1, real3* torque2, mixed *totalEnergy) {
    real rInv = RSQRT(r2);
    real r = r2*rInv;
    real3 drUnit = dr*rInv;

    // Compute the switching function.

    real switchValue = 1, switchDeriv = 0;
    #if USE_SWITCH
    if (r > SWITCH_CUTOFF) {
        real x = r-SWITCH_CUTOFF;
        switchValue = 1+x*x*x*(SWITCH_C3+x*(SWITCH_C4+x*SWITCH_C5));
        switchDeriv = x*x*(3*SWITCH_C3+x*(4*SWITCH_C4+x*5*SWITCH_C5));
    }
    #endif

    // Compute vectors and matrices we'll be needing.

    real B12[3][3], G12[3][3], B12inv[3][3], G12inv[3][3];
    matrixSum(B12, data1->b, data2->b);
    matrixSum(G12, data1->g, data2->g);
    matrixInverse(B12inv, B12);
    matrixInverse(G12inv, G12);
    real detG12 = determinant(G12);

    // Estimate the distance between the ellipsoids and compute the first terms needed for the energy.

    real sigma12 = 1/SQRT(0.5f*dot(drUnit, matrixVectorProduct(G12inv, drUnit)));
    real h12 = r - sigma12;
    real rho = sigma/(h12+sigma);
    real rho2 = rho*rho;
    real rho6 = rho2*rho2*rho2;
    real u = 4*epsilon*(rho6*rho6-rho6);
    real eta = SQRT(2*data1->eps.y*data2->eps.y/detG12);
    real chi = 2*dot(drUnit, matrixVectorProduct(B12inv, drUnit));
    chi *= chi;
    real energy = u*eta*chi;

    // Compute the terms needed for the force.

    real3 kappa = matrixVectorProduct(G12inv, dr);
    real3 iota = matrixVectorProduct(B12inv, dr);
    real rInv2 = rInv*rInv;
    real dUSLJdr = 24*epsilon*(2*rho6-1)*rho6*rho/sigma;
    real temp = 0.5f*sigma12*sigma12*sigma12*rInv2;
    real3 dudr = (drUnit + (kappa-drUnit*dot(kappa, drUnit))*temp)*dUSLJdr;
    real3 dchidr = (iota-drUnit*dot(iota, drUnit))*(-8*rInv2*SQRT(chi));
    real3 force = (dchidr*u + dudr*chi)*(eta*switchValue) - drUnit*(energy*switchDeriv);
    *force1 += force;
    *force2 -= force;

    // Compute the terms needed for the torque.

    for (int j = 0; j < 2; j++) {
        real (*a)[3] = (j == 0 ? data1->a : data2->a);
        real (*b)[3] = (j == 0 ? data1->b : data2->b);
        real (*g)[3] = (j == 0 ? data1->g : data2->g);
        float4 sig = (j == 0 ? data1->sig : data2->sig);
        real3 dudq = cross(vectorMatrixProduct(kappa, g), kappa*(temp*dUSLJdr));
        real3 dchidq = cross(vectorMatrixProduct(iota, b), iota)*(-4*rInv2);
        real3 scale = make_real3(sig.y, sig.z, sig.w)*(-0.5f*eta/detG12);
        real d[3][3];
        d[0][0] = scale.x*(2*a[0][0]*(G12[1][1]*G12[2][2] - G12[1][2]*G12[2][1]) +
                             a[0][2]*(G12[1][2]*G12[0][1] + G12[1][0]*G12[2][1] - G12[1][1]*(G12[0][2] + G12[2][0])) +
                             a[0][1]*(G12[0][2]*G12[2][1] + G12[2][0]*G12[1][2] - G12[2][2]*(G12[0][1] + G12[1][0])));
        d[0][1] = scale.x*(  a[0][0]*(G12[0][2]*G12[2][1] + G12[2][0]*G12[1][2] - G12[2][2]*(G12[0][1] + G12[1][0])) +
                           2*a[0][1]*(G12[0][0]*G12[2][2] - G12[2][0]*G12[0][2]) +
                             a[0][2]*(G12[1][0]*G12[0][2] + G12[2][0]*G12[0][1] - G12[0][0]*(G12[1][2] + G12[2][1])));
        d[0][2] = scale.x*(  a[0][0]*(G12[0][1]*G12[1][2] + G12[1][0]*G12[2][1] - G12[1][1]*(G12[0][2] + G12[2][0])) +
                             a[0][1]*(G12[1][0]*G12[0][2] + G12[2][0]*G12[0][1] - G12[0][0]*(G12[1][2] + G12[2][1])) +
                           2*a[0][2]*(G12[1][1]*G12[0][0] - G12[1][0]*G12[0][1]));
        d[1][0] = scale.y*(2*a[1][0]*(G12[1][1]*G12[2][2] - G12[1][2]*G12[2][1]) +
                             a[1][1]*(G12[0][2]*G12[2][1] + G12[2][0]*G12[1][2] - G12[2][2]*(G12[0][1] + G12[1][0])) +
                             a[1][2]*(G12[1][2]*G12[0][1] + G12[1][0]*G12[2][1] - G12[1][1]*(G12[0][2] + G12[2][0])));
        d[1][1] = scale.y*(  a[1][0]*(G12[0][2]*G12[2][1] + G12[2][0]*G12[1][2] - G12[2][2]*(G12[0][1] + G12[1][0])) +
                           2*a[1][1]*(G12[2][2]*G12[0][0] - G12[2][0]*G12[0][2]) +
                             a[1][2]*(G12[1][0]*G12[0][2] + G12[0][1]*G12[2][0] - G12[0][0]*(G12[1][2] + G12[2][1])));
        d[1][2] = scale.y*(  a[1][0]*(G12[0][1]*G12[1][2] + G12[1][0]*G12[2][1] - G12[1][1]*(G12[0][2] + G12[2][0])) +
                             a[1][1]*(G12[1][0]*G12[0][2] + G12[0][1]*G12[2][0] - G12[0][0]*(G12[1][2] + G12[2][1])) +
                           2*a[1][2]*(G12[1][1]*G12[0][0] - G12[1][0]*G12[0][1]));
        d[2][0] = scale.z*(2*a[2][0]*(G12[1][1]*G12[2][2] - G12[2][1]*G12[1][2]) +
                             a[2][1]*(G12[0][2]*G12[2][1] + G12[1][2]*G12[2][0] - G12[2][2]*(G12[0][1] + G12[1][0])) +
                             a[2][2]*(G12[0][1]*G12[1][2] + G12[2][1]*G12[1][0] - G12[1][1]*(G12[0][2] + G12[2][0])));
        d[2][1] = scale.z*(  a[2][0]*(G12[0][2]*G12[2][1] + G12[1][2]*G12[2][0] - G12[2][2]*(G12[0][1] + G12[1][0])) +
                           2*a[2][1]*(G12[0][0]*G12[2][2] - G12[0][2]*G12[2][0]) +
                             a[2][2]*(G12[1][0]*G12[0][2] + G12[0][1]*G12[2][0] - G12[0][0]*(G12[1][2] + G12[2][1])));
        d[2][2] = scale.z*(  a[2][0]*(G12[0][1]*G12[1][2] + G12[2][1]*G12[1][0] - G12[1][1]*(G12[0][2] + G12[2][0])) +
                             a[2][1]*(G12[1][0]*G12[0][2] + G12[2][0]*G12[0][1] - G12[0][0]*(G12[1][2] + G12[2][1])) +
                           2*a[2][2]*(G12[1][1]*G12[0][0] - G12[1][0]*G12[0][1]));
        real3 detadq = make_real3(0);
        for (int i = 0; i < 3; i++)
            detadq += cross(make_real3(a[i][0], a[i][1], a[i][2]), make_real3(d[i][0], d[i][1], d[i][2]));
        real3 torque = (dchidq*(u*eta) + detadq*(u*chi) + dudq*(eta*chi))*switchValue;
        *(j == 0 ? torque1 : torque2) -= torque;
    }
    *totalEnergy += switchValue*energy;
}

/**
 * Compute the interactions.
 */
KERNEL void computeForce(
        GLOBAL mm_ulong* RESTRICT forceBuffers, GLOBAL mm_ulong* RESTRICT torqueBuffers,
        int numAtoms, int numExceptions, GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real4* RESTRICT pos,
        GLOBAL const float4* RESTRICT sigParams, GLOBAL const float2* RESTRICT epsParams, GLOBAL const int* RESTRICT sortedAtoms,
        GLOBAL const real* RESTRICT aMatrix, GLOBAL const real* RESTRICT bMatrix, GLOBAL const real* RESTRICT gMatrix,
        GLOBAL const int* RESTRICT exclusions, GLOBAL const int* RESTRICT exclusionStartIndex,
        GLOBAL const int4* RESTRICT exceptionParticles, GLOBAL const float2* RESTRICT exceptionParams
#ifdef USE_CUTOFF
        , int maxNeighborBlocks, GLOBAL int* RESTRICT neighbors, GLOBAL int* RESTRICT neighborIndex, GLOBAL int* RESTRICT neighborBlockCount,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ
#endif
        ) {
    const unsigned int warp = GLOBAL_ID/TILE_SIZE;
    mixed energy = 0;
#ifdef USE_CUTOFF
    const int numBlocks = *neighborBlockCount;
    if (numBlocks > maxNeighborBlocks)
        return; // There wasn't enough memory for the neighbor list.
    for (int block = GLOBAL_ID; block < numBlocks; block += GLOBAL_SIZE) {
        // Load parameters for atom1.

        int atom1 = neighborIndex[block];
        int index1 = sortedAtoms[atom1];
        AtomData data1;
        loadAtomData(&data1, atom1, index1, pos, sigParams, epsParams, aMatrix, bMatrix, gMatrix);
        real3 force1 = make_real3(0);
        real3 torque1 = make_real3(0);
        for (int indexInBlock = 0; indexInBlock < NEIGHBOR_BLOCK_SIZE; indexInBlock++) {
            // Load parameters for atom2.

            int atom2 = neighbors[NEIGHBOR_BLOCK_SIZE*block+indexInBlock];
            if (atom2 == -1)
                continue;
            int index2 = sortedAtoms[atom2];
            AtomData data2;
            loadAtomData(&data2, atom2, index2, pos, sigParams, epsParams, aMatrix, bMatrix, gMatrix);
            real3 force2 = make_real3(0);
            real3 torque2 = make_real3(0);

            // Compute the interaction.

            real3 delta = data1.pos-data2.pos;
#ifdef USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(delta)
#endif
            real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
            real sigma = data1.sig.x+data2.sig.x;
            real epsilon = data1.eps.x*data2.eps.x;
            computeOneInteraction(&data1, &data2, sigma, epsilon, delta, r2, &force1, &force2, &torque1, &torque2, &energy);
            ATOMIC_ADD(&forceBuffers[index2], (mm_ulong) ((mm_long) (force2.x*0x100000000)));
            ATOMIC_ADD(&forceBuffers[index2+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force2.y*0x100000000)));
            ATOMIC_ADD(&forceBuffers[index2+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force2.z*0x100000000)));
            ATOMIC_ADD(&torqueBuffers[index2], (mm_ulong) ((mm_long) (torque2.x*0x100000000)));
            ATOMIC_ADD(&torqueBuffers[index2+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (torque2.y*0x100000000)));
            ATOMIC_ADD(&torqueBuffers[index2+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (torque2.z*0x100000000)));
        }
        ATOMIC_ADD(&forceBuffers[index1], (mm_ulong) ((mm_long) (force1.x*0x100000000)));
        ATOMIC_ADD(&forceBuffers[index1+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force1.y*0x100000000)));
        ATOMIC_ADD(&forceBuffers[index1+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force1.z*0x100000000)));
        ATOMIC_ADD(&torqueBuffers[index1], (mm_ulong) ((mm_long) (torque1.x*0x100000000)));
        ATOMIC_ADD(&torqueBuffers[index1+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (torque1.y*0x100000000)));
        ATOMIC_ADD(&torqueBuffers[index1+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (torque1.z*0x100000000)));
    }
#else
    for (int atom1 = GLOBAL_ID; atom1 < numAtoms; atom1 += GLOBAL_SIZE) {
        // Load parameters for atom1.

        int index1 = sortedAtoms[atom1];
        AtomData data1;
        loadAtomData(&data1, atom1, index1, pos, sigParams, epsParams, aMatrix, bMatrix, gMatrix);
        real3 force1 = make_real3(0);
        real3 torque1 = make_real3(0);
        int nextExclusion = exclusionStartIndex[atom1];
        int lastExclusion = exclusionStartIndex[atom1+1];
        for (int atom2 = atom1+1; atom2 < numAtoms; atom2++) {
            // Skip over excluded interactions.

            if (nextExclusion < lastExclusion && exclusions[nextExclusion] == atom2) {
                nextExclusion++;
                continue;
            }

            // Load parameters for atom2.

            int index2 = sortedAtoms[atom2];
            AtomData data2;
            loadAtomData(&data2, atom2, index2, pos, sigParams, epsParams, aMatrix, bMatrix, gMatrix);
            real3 force2 = make_real3(0);
            real3 torque2 = make_real3(0);

            // Compute the interaction.

            real3 delta = data1.pos-data2.pos;
            real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
            real sigma = data1.sig.x+data2.sig.x;
            real epsilon = data1.eps.x*data2.eps.x;
            computeOneInteraction(&data1, &data2, sigma, epsilon, delta, r2, &force1, &force2, &torque1, &torque2, &energy);
            ATOMIC_ADD(&forceBuffers[index2], (mm_ulong) ((mm_long) (force2.x*0x100000000)));
            ATOMIC_ADD(&forceBuffers[index2+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force2.y*0x100000000)));
            ATOMIC_ADD(&forceBuffers[index2+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force2.z*0x100000000)));
            ATOMIC_ADD(&torqueBuffers[index2], (mm_ulong) ((mm_long) (torque2.x*0x100000000)));
            ATOMIC_ADD(&torqueBuffers[index2+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (torque2.y*0x100000000)));
            ATOMIC_ADD(&torqueBuffers[index2+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (torque2.z*0x100000000)));
        }
        ATOMIC_ADD(&forceBuffers[index1], (mm_ulong) ((mm_long) (force1.x*0x100000000)));
        ATOMIC_ADD(&forceBuffers[index1+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force1.y*0x100000000)));
        ATOMIC_ADD(&forceBuffers[index1+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force1.z*0x100000000)));
        ATOMIC_ADD(&torqueBuffers[index1], (mm_ulong) ((mm_long) (torque1.x*0x100000000)));
        ATOMIC_ADD(&torqueBuffers[index1+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (torque1.y*0x100000000)));
        ATOMIC_ADD(&torqueBuffers[index1+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (torque1.z*0x100000000)));
    }
#endif

    // Now compute exceptions.

    for (int index = GLOBAL_ID; index < numExceptions; index += GLOBAL_SIZE) {
        int4 atomIndices = exceptionParticles[index];
        float2 params = exceptionParams[index];
        int index1 = atomIndices.x, index2 = atomIndices.y;
        int atom1 = atomIndices.z, atom2 = atomIndices.w;
        AtomData data1, data2;
        loadAtomData(&data1, atom1, index1, pos, sigParams, epsParams, aMatrix, bMatrix, gMatrix);
        loadAtomData(&data2, atom2, index2, pos, sigParams, epsParams, aMatrix, bMatrix, gMatrix);
        real3 force1 = make_real3(0), force2 = make_real3(0);
        real3 torque1 = make_real3(0), torque2 = make_real3(0);
        real3 delta = data1.pos-data2.pos;
        real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
        if (r2 < CUTOFF_SQUARED) {
#endif
            computeOneInteraction(&data1, &data2, params.x, params.y, delta, r2, &force1, &force2, &torque1, &torque2, &energy);
            ATOMIC_ADD(&forceBuffers[index1], (mm_ulong) ((mm_long) (force1.x*0x100000000)));
            ATOMIC_ADD(&forceBuffers[index1+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force1.y*0x100000000)));
            ATOMIC_ADD(&forceBuffers[index1+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force1.z*0x100000000)));
            ATOMIC_ADD(&forceBuffers[index2], (mm_ulong) ((mm_long) (force2.x*0x100000000)));
            ATOMIC_ADD(&forceBuffers[index2+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force2.y*0x100000000)));
            ATOMIC_ADD(&forceBuffers[index2+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force2.z*0x100000000)));
            ATOMIC_ADD(&torqueBuffers[index1], (mm_ulong) ((mm_long) (torque1.x*0x100000000)));
            ATOMIC_ADD(&torqueBuffers[index1+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (torque1.y*0x100000000)));
            ATOMIC_ADD(&torqueBuffers[index1+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (torque1.z*0x100000000)));
            ATOMIC_ADD(&torqueBuffers[index2], (mm_ulong) ((mm_long) (torque2.x*0x100000000)));
            ATOMIC_ADD(&torqueBuffers[index2+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (torque2.y*0x100000000)));
            ATOMIC_ADD(&torqueBuffers[index2+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (torque2.z*0x100000000)));
#ifdef USE_CUTOFF
        }
#endif
    }
    energyBuffer[GLOBAL_ID] += energy;
}

/**
 * Convert the torques to forces on the connected particles.
 */
KERNEL void applyTorques(
        GLOBAL mm_ulong* RESTRICT forceBuffers, GLOBAL const mm_long* RESTRICT torqueBuffers,
        int numParticles, GLOBAL const real4* RESTRICT posq, GLOBAL int2* const RESTRICT axisParticleIndices,
        GLOBAL const int* sortedParticles) {
    const unsigned int warp = GLOBAL_ID/TILE_SIZE;
    for (int sortedIndex = GLOBAL_ID; sortedIndex < numParticles; sortedIndex += GLOBAL_SIZE) {
        int originalIndex = sortedParticles[sortedIndex];
        real3 pos = trimTo3(posq[originalIndex]);
        int2 axisParticles = axisParticleIndices[originalIndex];
        if (axisParticles.x != -1) {
            // Load the torque.

            real scale = 1/(real) 0x100000000;
            real3 torque = make_real3(scale*torqueBuffers[originalIndex], scale*torqueBuffers[originalIndex+PADDED_NUM_ATOMS], scale*torqueBuffers[originalIndex+2*PADDED_NUM_ATOMS]);
            real3 force = make_real3(0), xforce = make_real3(0), yforce = make_real3(0);

            // Apply a force to the x particle.

            real3 dx = trimTo3(posq[axisParticles.x])-pos;
            real dx2 = dot(dx, dx);
            real3 f = cross(torque, dx)/dx2;
            xforce += f;
            force -= f;
            if (axisParticles.y != -1) {
                // Apply a force to the y particle.  This is based on the component of the torque
                // that was not already applied to the x particle.

                real3 dy = trimTo3(posq[axisParticles.y])-pos;
                real dy2 = dot(dy, dy);
                real3 torque2 = dx*dot(torque, dx)/dx2;
                f = cross(torque2, dy)/dy2;
                yforce += f;
                force -= f;
            }
            ATOMIC_ADD(&forceBuffers[originalIndex], (mm_ulong) ((mm_long) (force.x*0x100000000)));
            ATOMIC_ADD(&forceBuffers[originalIndex+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force.y*0x100000000)));
            ATOMIC_ADD(&forceBuffers[originalIndex+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (force.z*0x100000000)));
            ATOMIC_ADD(&forceBuffers[axisParticles.x], (mm_ulong) ((mm_long) (xforce.x*0x100000000)));
            ATOMIC_ADD(&forceBuffers[axisParticles.x+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (xforce.y*0x100000000)));
            ATOMIC_ADD(&forceBuffers[axisParticles.x+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (xforce.z*0x100000000)));
            if (axisParticles.y != -1) {
                ATOMIC_ADD(&forceBuffers[axisParticles.y], (mm_ulong) ((mm_long) (yforce.x*0x100000000)));
                ATOMIC_ADD(&forceBuffers[axisParticles.y+PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (yforce.y*0x100000000)));
                ATOMIC_ADD(&forceBuffers[axisParticles.y+2*PADDED_NUM_ATOMS], (mm_ulong) ((mm_long) (yforce.z*0x100000000)));
            }
        }
    }
}
