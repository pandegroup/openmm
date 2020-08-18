#ifndef __OPENMM_HIPSORT_H__
#define __OPENMM_HIPSORT_H__

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2010-2018 Stanford University and the Authors.      *
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
#include "openmm/common/windowsExportCommon.h"
#include "HipContext.h"

namespace OpenMM {

/**
 * This class sorts arrays of values.  It supports any type of values, not just scalars,
 * so long as an appropriate sorting key can be defined by which to sort them.
 *
 * The sorting behavior is specified by a "trait" class that defines the type of data to
 * sort and the key for sorting it.  Here is an example of a trait class for
 * sorting floats:
 *
 * class FloatTrait : public HipSort::SortTrait {
 *     int getDataSize() const {return 4;}
 *     int getKeySize() const {return 4;}
 *     const char* getDataType() const {return "float";}
 *     const char* getKeyType() const {return "float";}
 *     const char* getMinKey() const {return "-3.40282e+38f";}
 *     const char* getMaxKey() const {return "3.40282e+38f";}
 *     const char* getMaxValue() const {return "3.40282e+38f";}
 *     const char* getSortKey() const {return "value";}
 * };
 *
 * The algorithm used is a bucket sort, followed by a bitonic sort within each bucket
 * (in local memory when possible, in global memory otherwise).  This is similar to
 * the algorithm described in
 *
 * Shifu Chen, Jing Qin, Yongming Xie, Junping Zhao, and Pheng-Ann Heng.  "An Efficient
 * Sorting Algorithm with CUDA"  Journal of the Chinese Institute of Engineers, 32(7),
 * pp. 915-921 (2009)
 *
 * but with many modifications and simplifications.  In particular, this algorithm
 * involves much less communication between host and device, which is critical to get
 * good performance with the array sizes we typically work with (10,000 to 100,000
 * elements).
 */

class OPENMM_EXPORT_COMMON HipSort {
public:
    class SortTrait;
    /**
     * Create a HipSort object for sorting data of a particular type.
     *
     * @param context    the context in which to perform calculations
     * @param trait      a SortTrait defining the type of data to sort.  It should have been allocated
     *                   on the heap with the "new" operator.  This object takes over ownership of it,
     *                   and deletes it when the HipSort is deleted.
     * @param length     the length of the arrays this object will be used to sort
     */
    HipSort(HipContext& context, SortTrait* trait, unsigned int length);
    ~HipSort();
    /**
     * Sort an array.
     */
    void sort(HipArray& data);
private:
    HipContext& context;
    SortTrait* trait;
    HipArray dataRange;
    HipArray bucketOfElement;
    HipArray offsetInBucket;
    HipArray bucketOffset;
    HipArray buckets;
    hipFunction_t shortListKernel, shortList2Kernel, computeRangeKernel, assignElementsKernel, computeBucketPositionsKernel, copyToBucketsKernel, sortBucketsKernel;
    unsigned int dataLength, rangeKernelSize, positionsKernelSize, sortKernelSize;
    bool isShortList;
};

/**
 * A subclass of SortTrait defines the type of value to sort, and the key for sorting them.
 */
class HipSort::SortTrait {
public:
    virtual ~SortTrait() {
    }
    /**
     * Get the size of each data value in bytes.
     */
    virtual int getDataSize() const = 0;
    /**
     * Get the size of each key value in bytes.
     */
    virtual int getKeySize() const = 0;
    /**
     * Get the data type of the values to sort.
     */
    virtual const char* getDataType() const = 0;
    /**
     * Get the data type of the sorting key.
     */
    virtual const char* getKeyType() const = 0;
    /**
     * Get the minimum value a key can take.
     */
    virtual const char* getMinKey() const = 0;
    /**
     * Get the maximum value a key can take.
     */
    virtual const char* getMaxKey() const = 0;
    /**
     * Get a value whose key is guaranteed to equal getMaxKey().
     */
    virtual const char* getMaxValue() const = 0;
    /**
     * Get the CUDA code to select the key from the data value.
     */
    virtual const char* getSortKey() const = 0;
};


} // namespace OpenMM

#endif // __OPENMM_HIPSORT_H__
