#ifndef __OPENMM_HIPFFT3D_H__
#define __OPENMM_HIPFFT3D_H__

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2015 Stanford University and the Authors.      *
 * Portions copyright (C) 2020 Advanced Micro Devices, Inc. All Rights        *
 * Reserved.                                                                  *
 * Authors: Peter Eastman, Nicholas Curtis                                    *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "HipArray.h"

namespace OpenMM {

/**
 * This class performs three dimensional Fast Fourier Transforms.  It is based on the
 * mixed radix algorithm described in
 * <p>
 * Takahashi, D. and Kanada, Y., "High-Performance Radix-2, 3 and 5 Parallel 1-D Complex
 * FFT Algorithms for Distributed-Memory Parallel Computers."  Journal of Supercomputing,
 * 15, 207–228 (2000).
 * <p>
 * This class places certain restrictions on the allowed dimensions of the grid.  First,
 * the size of each dimension may have no prime factors other than 2, 3, 5, and 7.  You
 * can call findLegalDimension() to determine the smallest size that satisfies this
 * requirement and is greater than or equal to a specified minimum size.  Second, the size
 * of each dimension must be small enough to compute each 1D transform entirely in local
 * memory with one work unit per data point.  This will vary between platforms, but is
 * typically at least 512.
 * <p>
 * Note that this class performs an unnormalized transform.  That means that if you perform
 * a forward transform followed immediately by an inverse transform, the effect is to
 * multiply every value of the original data set by the total number of data points.
 */

class OPENMM_EXPORT_COMMON HipFFT3D {
public:
    /**
     * Create an HipFFT3D object for performing transforms of a particular size.
     *
     * @param context the context in which to perform calculations
     * @param xsize   the first dimension of the data sets on which FFTs will be performed
     * @param ysize   the second dimension of the data sets on which FFTs will be performed
     * @param zsize   the third dimension of the data sets on which FFTs will be performed
     * @param realToComplex  if true, a real-to-complex transform will be done.  Otherwise, it is complex-to-complex.
     */
    HipFFT3D(HipContext& context, int xsize, int ysize, int zsize, bool realToComplex=false);
    /**
     * Perform a Fourier transform.  The transform cannot be done in-place: the input and output
     * arrays must be different.  Also, the input array is used as workspace, so its contents
     * are destroyed.  This also means that both arrays must be large enough to hold complex values,
     * even when performing a real-to-complex transform.
     * <p>
     * When performing a real-to-complex transform, the output data is of size xsize*ysize*(zsize/2+1)
     * and contains only the non-redundant elements.
     *
     * @param in       the data to transform, ordered such that in[x*ysize*zsize + y*zsize + z] contains element (x, y, z)
     * @param out      on exit, this contains the transformed data
     * @param forward  true to perform a forward transform, false to perform an inverse transform
     */
    void execFFT(HipArray& in, HipArray& out, bool forward = true);
    /**
     * Get the smallest legal size for a dimension of the grid (that is, a size with no prime
     * factors other than 2, 3, 5, and 7).
     *
     * @param minimum   the minimum size the return value must be greater than or equal to
     */
    static int findLegalDimension(int minimum);
private:
    hipFunction_t createKernel(int xsize, int ysize, int zsize, int& threads, int axis, bool forward, bool inputIsReal);
    int xsize, ysize, zsize;
    int xthreads, ythreads, zthreads;
    bool packRealAsComplex;
    HipContext& context;
    hipFunction_t xkernel, ykernel, zkernel;
    hipFunction_t invxkernel, invykernel, invzkernel;
    hipFunction_t packForwardKernel, unpackForwardKernel, packBackwardKernel, unpackBackwardKernel;
};

} // namespace OpenMM

#endif // __OPENMM_HIPFFT3D_H__
