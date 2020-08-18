#include "hip/hip_runtime.h"
extern "C" __global__ void findAtomGridIndex(const real4* __restrict__ posq, int2* __restrict__ pmeAtomGridIndex,
            real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
            real3 recipBoxVecX, real3 recipBoxVecY, real3 recipBoxVecZ) {
    // Compute the index of the grid point each atom is associated with.
    
    for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < NUM_ATOMS; atom += blockDim.x*gridDim.x) {
        real4 pos = posq[atom];
        APPLY_PERIODIC_TO_POS(pos)
        real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                             pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                             pos.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x) % GRID_SIZE_X,
                                   ((int) t.y) % GRID_SIZE_Y,
                                   ((int) t.z) % GRID_SIZE_Z);
        pmeAtomGridIndex[atom] = make_int2(atom, gridIndex.x*GRID_SIZE_Y*GRID_SIZE_Z+gridIndex.y*GRID_SIZE_Z+gridIndex.z);
    }
}

extern "C" __global__ void gridSpreadCharge(const real4* __restrict__ posq, real* __restrict__ originalPmeGrid,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real3 recipBoxVecX, real3 recipBoxVecY, real3 recipBoxVecZ, const int2* __restrict__ pmeAtomGridIndex
#ifdef CHARGE_FROM_SIGEPS
        , const float2* __restrict__ sigmaEpsilon
#else
        , const real* __restrict__ charges
#endif
        ) {
    // To improve memory efficiency, we divide indices along the z axis into
    // PME_ORDER blocks, where the data for each block is stored together.  We
    // can ensure that all threads write to the same block at the same time,
    // which leads to better coalescing of writes.
    
    __shared__ int zindexTable[GRID_SIZE_Z+PME_ORDER];
    int blockSize = (int) ceil(GRID_SIZE_Z/(real) PME_ORDER);
    for (int i = threadIdx.x; i < GRID_SIZE_Z+PME_ORDER; i += blockDim.x) {
        int zindex = i % GRID_SIZE_Z;
	int block = zindex % PME_ORDER;
        zindexTable[i] = zindex/PME_ORDER + block*GRID_SIZE_X*GRID_SIZE_Y*blockSize;
    }
    __syncthreads();
    
    // Process the atoms in spatially sorted order.  This improves efficiency when writing
    // the grid values.
    
    real3 data[PME_ORDER];
    const real scale = RECIP(PME_ORDER-1);
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_ATOMS; i += blockDim.x*gridDim.x) {
        int atom = pmeAtomGridIndex[i].x;
        real4 pos = posq[atom];
#ifdef CHARGE_FROM_SIGEPS
        const float2 sigEps = sigmaEpsilon[atom];
        const real charge = 8*sigEps.x*sigEps.x*sigEps.x*sigEps.y;
#else
        const real charge = (CHARGE)*EPSILON_FACTOR;
#endif
        if (charge == 0)
            continue;
        APPLY_PERIODIC_TO_POS(pos)
        real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                             pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                             pos.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x) % GRID_SIZE_X,
                                   ((int) t.y) % GRID_SIZE_Y,
                                   ((int) t.z) % GRID_SIZE_Z);

        // Since we need the full set of thetas, it's faster to compute them here than load them
        // from global memory.
        
        real3 dr = make_real3(t.x-(int) t.x, t.y-(int) t.y, t.z-(int) t.z);
        data[PME_ORDER-1] = make_real3(0);
        data[1] = dr;
        data[0] = make_real3(1)-dr;
        for (int j = 3; j < PME_ORDER; j++) {
            real div = RECIP(j-1);
            data[j-1] = div*dr*data[j-2];
            for (int k = 1; k < (j-1); k++)
                data[j-k-1] = div*((dr+make_real3(k))*data[j-k-2] + (make_real3(j-k)-dr)*data[j-k-1]);
            data[0] = div*(make_real3(1)-dr)*data[0];
        }
        data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
        for (int j = 1; j < (PME_ORDER-1); j++)
            data[PME_ORDER-j-1] = scale*((dr+make_real3(j))*data[PME_ORDER-j-2] + (make_real3(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
        data[0] = scale*(make_real3(1)-dr)*data[0];
        
        // Spread the charge from this atom onto each grid point.

	int izoffset = (PME_ORDER-(gridIndex.z%PME_ORDER)) % PME_ORDER;
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = gridIndex.x+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y;
            real dx = charge*data[ix].x;
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = gridIndex.y+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = (xbase+ybase)*blockSize;
                real dxdy = dx*data[iy].y;
                for (int i = 0; i < PME_ORDER; i++) {
		    int iz = (i+izoffset) % PME_ORDER;
                    int zindex = gridIndex.z+iz;
                    int index = ybase + zindexTable[zindex];
                    real add = dxdy*data[iz].z;
#if defined(USE_DOUBLE_PRECISION) || defined(USE_DETERMINISTIC_FORCES)
                    unsigned long long * ulonglong_p = (unsigned long long *) originalPmeGrid;
                    atomicAdd(&ulonglong_p[index],  static_cast<unsigned long long>((long long) (add*0x100000000)));
#else
                    atomicAdd(&originalPmeGrid[index], add);
#endif
                }
            }
        }
    }
}

extern "C" __global__ void finishSpreadCharge(
#if defined(USE_DOUBLE_PRECISION) || defined(USE_DETERMINISTIC_FORCES)
        const long long* __restrict__ grid1,
#else
        const real* __restrict__ grid1,
#endif
        real* __restrict__ grid2) {
    // During charge spreading, we shuffled the order of indices along the z
    // axis to make memory access more efficient.  We now need to unshuffle
    // them.  If the values were accumulated as fixed point, we also need to
    // convert them to floating point.

    __shared__ int zindexTable[GRID_SIZE_Z];
    int blockSize = (int) ceil(GRID_SIZE_Z/(real) PME_ORDER);
    for (int i = threadIdx.x; i < GRID_SIZE_Z; i += blockDim.x) {
	int block = i % PME_ORDER;
        zindexTable[i] = i/PME_ORDER + block*GRID_SIZE_X*GRID_SIZE_Y*blockSize;
    }
    __syncthreads();
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z;
    real scale = 1/(real) 0x100000000;
    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < gridSize; index += blockDim.x*gridDim.x) {
        int zindex = index%GRID_SIZE_Z;
        int loadIndex = zindexTable[zindex] + blockSize*(int) (index/GRID_SIZE_Z);
#if defined(USE_DOUBLE_PRECISION) || defined(USE_DETERMINISTIC_FORCES)
        grid2[index] = scale*grid1[loadIndex];
#else
        grid2[index] = grid1[loadIndex];
#endif
    }
}

// convolutes on the halfcomplex_pmeGrid, which is of size NX*NY*(NZ/2+1) as F(Q) is conjugate symmetric
extern "C" __global__ void 
reciprocalConvolution(real2* __restrict__ halfcomplex_pmeGrid, mixed* __restrict__ energyBuffer, 
                      const real* __restrict__ pmeBsplineModuliX, const real* __restrict__ pmeBsplineModuliY, const real* __restrict__ pmeBsplineModuliZ, 
                      real4 periodicBoxSize, real3 recipBoxVecX, real3 recipBoxVecY, real3 recipBoxVecZ) {
    // R2C stores into a half complex matrix where the last dimension is cut by half
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*(GRID_SIZE_Z/2+1);
#ifdef USE_LJPME
    const real recipScaleFactor = -2*M_PI*SQRT(M_PI)*RECIP(6*periodicBoxSize.x*periodicBoxSize.y*periodicBoxSize.z);
    real bfac = M_PI / EWALD_ALPHA;
    real fac1 = 2*M_PI*M_PI*M_PI*SQRT(M_PI);
    real fac2 = EWALD_ALPHA*EWALD_ALPHA*EWALD_ALPHA;
    real fac3 = -2*EWALD_ALPHA*M_PI*M_PI;
#else
    const real recipScaleFactor = RECIP(M_PI*periodicBoxSize.x*periodicBoxSize.y*periodicBoxSize.z);
#endif

    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < gridSize; index += blockDim.x*gridDim.x) {
        // real indices
        int kx = index/(GRID_SIZE_Y*(GRID_SIZE_Z/2+1));
        int remainder = index-kx*GRID_SIZE_Y*(GRID_SIZE_Z/2+1);
        int ky = remainder/(GRID_SIZE_Z/2+1);
        int kz = remainder-ky*(GRID_SIZE_Z/2+1);
        int mx = (kx < (GRID_SIZE_X+1)/2) ? kx : (kx-GRID_SIZE_X);
        int my = (ky < (GRID_SIZE_Y+1)/2) ? ky : (ky-GRID_SIZE_Y);
        int mz = (kz < (GRID_SIZE_Z+1)/2) ? kz : (kz-GRID_SIZE_Z);
        real mhx = mx*recipBoxVecX.x;
        real mhy = mx*recipBoxVecY.x+my*recipBoxVecY.y;
        real mhz = mx*recipBoxVecZ.x+my*recipBoxVecZ.y+mz*recipBoxVecZ.z;
        real bx = pmeBsplineModuliX[kx];
        real by = pmeBsplineModuliY[ky];
        real bz = pmeBsplineModuliZ[kz];
        real2 grid = halfcomplex_pmeGrid[index];
        real m2 = mhx*mhx+mhy*mhy+mhz*mhz;
#ifdef USE_LJPME
        real denom = recipScaleFactor/(bx*by*bz);
        real m = SQRT(m2);
        real m3 = m*m2;
        real b = bfac*m;
        real expfac = -b*b;
        real expterm = EXP(expfac);
        real erfcterm = ERFC(b);
        real eterm = (fac1*erfcterm*m3 + expterm*(fac2 + fac3*m2)) * denom;
        halfcomplex_pmeGrid[index] = make_real2(grid.x*eterm, grid.y*eterm);
#else
        real denom = m2*bx*by*bz;
        real eterm = recipScaleFactor*EXP(-RECIP_EXP_FACTOR*m2)/denom;
        if (kx != 0 || ky != 0 || kz != 0) {
            halfcomplex_pmeGrid[index] = make_real2(grid.x*eterm, grid.y*eterm);
        }
#endif
    }
}


extern "C" __global__ void
gridEvaluateEnergy(real2* __restrict__ halfcomplex_pmeGrid, mixed* __restrict__ energyBuffer,
                      const real* __restrict__ pmeBsplineModuliX, const real* __restrict__ pmeBsplineModuliY, const real* __restrict__ pmeBsplineModuliZ,
                      real4 periodicBoxSize, real3 recipBoxVecX, real3 recipBoxVecY, real3 recipBoxVecZ) {
    // R2C stores into a half complex matrix where the last dimension is cut by half
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z;
 #ifdef USE_LJPME
    const real recipScaleFactor = -2*M_PI*SQRT(M_PI)*RECIP(6*periodicBoxSize.x*periodicBoxSize.y*periodicBoxSize.z);
    real bfac = M_PI / EWALD_ALPHA;
    real fac1 = 2*M_PI*M_PI*M_PI*SQRT(M_PI);
    real fac2 = EWALD_ALPHA*EWALD_ALPHA*EWALD_ALPHA;
    real fac3 = -2*EWALD_ALPHA*M_PI*M_PI;
#else
    const real recipScaleFactor = RECIP(M_PI*periodicBoxSize.x*periodicBoxSize.y*periodicBoxSize.z);
#endif

    mixed energy = 0;
    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < gridSize; index += blockDim.x*gridDim.x) {
        // real indices
        int kx = index/(GRID_SIZE_Y*(GRID_SIZE_Z));
        int remainder = index-kx*GRID_SIZE_Y*(GRID_SIZE_Z);
        int ky = remainder/(GRID_SIZE_Z);
        int kz = remainder-ky*(GRID_SIZE_Z);
        int mx = (kx < (GRID_SIZE_X+1)/2) ? kx : (kx-GRID_SIZE_X);
        int my = (ky < (GRID_SIZE_Y+1)/2) ? ky : (ky-GRID_SIZE_Y);
        int mz = (kz < (GRID_SIZE_Z+1)/2) ? kz : (kz-GRID_SIZE_Z);
        real mhx = mx*recipBoxVecX.x;
        real mhy = mx*recipBoxVecY.x+my*recipBoxVecY.y;
        real mhz = mx*recipBoxVecZ.x+my*recipBoxVecZ.y+mz*recipBoxVecZ.z;
        real m2 = mhx*mhx+mhy*mhy+mhz*mhz;
        real bx = pmeBsplineModuliX[kx];
        real by = pmeBsplineModuliY[ky];
        real bz = pmeBsplineModuliZ[kz];
#ifdef USE_LJPME
        real denom = recipScaleFactor/(bx*by*bz);
        real m = SQRT(m2);
        real m3 = m*m2;
        real b = bfac*m;
        real expfac = -b*b;
        real expterm = EXP(expfac);
        real erfcterm = ERFC(b);
        real eterm = (fac1*erfcterm*m3 + expterm*(fac2 + fac3*m2)) * denom;
#else
        real denom = m2*bx*by*bz;
        real eterm = recipScaleFactor*EXP(-RECIP_EXP_FACTOR*m2)/denom;
#endif

        if (kz >= (GRID_SIZE_Z/2+1)) {
            kx = ((kx == 0) ? kx : GRID_SIZE_X-kx);
            ky = ((ky == 0) ? ky : GRID_SIZE_Y-ky);
            kz = GRID_SIZE_Z-kz;
        } 
        int indexInHalfComplexGrid = kz + ky*(GRID_SIZE_Z/2+1)+kx*(GRID_SIZE_Y*(GRID_SIZE_Z/2+1));
        real2 grid = halfcomplex_pmeGrid[indexInHalfComplexGrid];
#ifndef USE_LJPME
        if (kx != 0 || ky != 0 || kz != 0)
#endif
            energy += eterm*(grid.x*grid.x + grid.y*grid.y);
    }
#if defined(USE_PME_STREAM) && !defined(USE_LJPME)
    energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] = 0.5f*energy;
#else
    energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] += 0.5f*energy;
#endif
}

extern "C" __global__
void gridInterpolateForce(const real4* __restrict__ posq, unsigned long long* __restrict__ forceBuffers, const real* __restrict__ originalPmeGrid,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real3 recipBoxVecX, real3 recipBoxVecY, real3 recipBoxVecZ, const int2* __restrict__ pmeAtomGridIndex
#ifdef CHARGE_FROM_SIGEPS
        , const float2* __restrict__ sigmaEpsilon
#else
        , const real* __restrict__ charges
#endif
        ) {
    real3 data[PME_ORDER];
    real3 ddata[PME_ORDER];
    const real scale = RECIP(PME_ORDER-1);
    
    // Process the atoms in spatially sorted order.  This improves cache performance when loading
    // the grid values.
    
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_ATOMS; i += blockDim.x*gridDim.x) {
        int atom = pmeAtomGridIndex[i].x;
        real3 force = make_real3(0);
        real4 pos = posq[atom];
        APPLY_PERIODIC_TO_POS(pos)
        real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                             pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                             pos.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x) % GRID_SIZE_X,
                                   ((int) t.y) % GRID_SIZE_Y,
                                   ((int) t.z) % GRID_SIZE_Z);

        // Since we need the full set of thetas, it's faster to compute them here than load them
        // from global memory.
        
        real3 dr = make_real3(t.x-(int) t.x, t.y-(int) t.y, t.z-(int) t.z);
        data[PME_ORDER-1] = make_real3(0);
        data[1] = dr;
        data[0] = make_real3(1)-dr;
        for (int j = 3; j < PME_ORDER; j++) {
            real div = RECIP(j-1);
            data[j-1] = div*dr*data[j-2];
            for (int k = 1; k < (j-1); k++)
                data[j-k-1] = div*((dr+make_real3(k))*data[j-k-2] + (make_real3(j-k)-dr)*data[j-k-1]);
            data[0] = div*(make_real3(1)-dr)*data[0];
        }
        ddata[0] = -data[0];
        for (int j = 1; j < PME_ORDER; j++)
            ddata[j] = data[j-1]-data[j];
        data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
        for (int j = 1; j < (PME_ORDER-1); j++)
            data[PME_ORDER-j-1] = scale*((dr+make_real3(j))*data[PME_ORDER-j-2] + (make_real3(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
        data[0] = scale*(make_real3(1)-dr)*data[0];
        
        // Compute the force on this atom.
         
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = gridIndex.x+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y*GRID_SIZE_Z;
            real dx = data[ix].x;
            real ddx = ddata[ix].x;
            
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = gridIndex.y+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = xbase + ybase*GRID_SIZE_Z;
                real dy = data[iy].y;
                real ddy = ddata[iy].y;
                
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = gridIndex.z+iz;
                    zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    int index = ybase + zindex;
                    real gridvalue = originalPmeGrid[index];
                    force.x += ddx*dy*data[iz].z*gridvalue;
                    force.y += dx*ddy*data[iz].z*gridvalue;
                    force.z += dx*dy*ddata[iz].z*gridvalue;
                }
            }
        }
#ifdef CHARGE_FROM_SIGEPS
        const float2 sigEps = sigmaEpsilon[atom];
        real q = 8*sigEps.x*sigEps.x*sigEps.x*sigEps.y;
#else
        real q = CHARGE*EPSILON_FACTOR;
#endif
        real forceX = -q*(force.x*GRID_SIZE_X*recipBoxVecX.x);
        real forceY = -q*(force.x*GRID_SIZE_X*recipBoxVecY.x+force.y*GRID_SIZE_Y*recipBoxVecY.y);
        real forceZ = -q*(force.x*GRID_SIZE_X*recipBoxVecZ.x+force.y*GRID_SIZE_Y*recipBoxVecZ.y+force.z*GRID_SIZE_Z*recipBoxVecZ.z);
        atomicAdd(&forceBuffers[atom], static_cast<unsigned long long>((long long) (forceX*0x100000000)));
        atomicAdd(&forceBuffers[atom+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (forceY*0x100000000)));
        atomicAdd(&forceBuffers[atom+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (forceZ*0x100000000)));
    }
}

extern "C" __global__
void addForces(const real4* __restrict__ forces, unsigned long long* __restrict__ forceBuffers) {
    for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < NUM_ATOMS; atom += blockDim.x*gridDim.x) {
        real4 f = forces[atom];
        forceBuffers[atom] += static_cast<unsigned long long>((long long) (f.x*0x100000000));
        forceBuffers[atom+PADDED_NUM_ATOMS] += static_cast<unsigned long long>((long long) (f.y*0x100000000));
        forceBuffers[atom+2*PADDED_NUM_ATOMS] += static_cast<unsigned long long>((long long) (f.z*0x100000000));
    }
}

extern "C" __global__
void addEnergy(const mixed* __restrict__ pmeEnergyBuffer, mixed* __restrict__ energyBuffer, int bufferSize) {
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < bufferSize; i += blockDim.x*gridDim.x)
        energyBuffer[i] += pmeEnergyBuffer[i];
}
