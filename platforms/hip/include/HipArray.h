#ifndef OPENMM_HIPARRAY_H_
#define OPENMM_HIPARRAY_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2019 Stanford University and the Authors.      *
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

#include "openmm/OpenMMException.h"
#include "openmm/common/windowsExportCommon.h"
#include "openmm/common/ArrayInterface.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <sstream>
#include <vector>

namespace OpenMM {

class HipContext;

/**
 * This class encapsulates a block of CUDA device memory.  It provides a simplified API
 * for working with it and for copying data to and from device memory.
 */

class OPENMM_EXPORT_COMMON HipArray : public ArrayInterface {
public:
    /**
     * Create a HipArray object.  The object is allocated on the heap with the "new" operator.
     * The template argument is the data type of each array element.
     *
     * @param context           the context for which to create the array
     * @param size              the number of elements in the array
     * @param name              the name of the array
     */
    template <class T>
    static HipArray* create(HipContext& context, int size, const std::string& name) {
        return new HipArray(context, size, sizeof(T), name);
    }
    /**
     * Create an uninitialized HipArray object.  It does not point to any device memory,
     * and cannot be used until initialize() is called on it.
     */
    HipArray();
    /**
     * Create a HipArray object.
     *
     * @param context           the context for which to create the array
     * @param size              the number of elements in the array
     * @param elementSize       the size of each element in bytes
     * @param name              the name of the array
     */
    HipArray(HipContext& context, int size, int elementSize, const std::string& name);
    ~HipArray();
    /**
     * Initialize this object.
     *
     * @param context           the context for which to create the array
     * @param size              the number of elements in the array
     * @param elementSize       the size of each element in bytes
     * @param name              the name of the array
     */
    void initialize(ComputeContext& context, int size, int elementSize, const std::string& name);
    /**
     * Initialize this object.  The template argument is the data type of each array element.
     *
     * @param context           the context for which to create the array
     * @param size              the number of elements in the array
     * @param name              the name of the array
     */
    template <class T>
    void initialize(ComputeContext& context, int size, const std::string& name) {
        initialize(context, size, sizeof(T), name);
    }
    /**
     * Recreate the internal storage to have a different size.
     */
    void resize(int size);
    /**
     * Get whether this array has been initialized.
     */
    bool isInitialized() const {
        return (pointer != 0);
    }
    /**
     * Get the number of elements in the array.
     */
    int getSize() const {
        return size;
    }
    /**
     * Get the size of each element in bytes.
     */
    int getElementSize() const {
        return elementSize;
    }
    /**
     * Get the name of the array.
     */
    const std::string& getName() const {
        return name;
    }
    /**
     * Get the context this array belongs to.
     */
    ComputeContext& getContext();
    /**
     * Get a pointer to the device memory.
     */
    hipDeviceptr_t& getDevicePointer() {
        return pointer;
    }
    /**
     * Copy the values in a vector to the device memory.
     */
    template <class T>
    void upload(const std::vector<T>& data, bool convert=false) {
        ArrayInterface::upload(data, convert);
    }
    /**
     * Copy the values in the Buffer to a vector.
     */
    template <class T>
    void download(std::vector<T>& data) const {
        ArrayInterface::download(data);
    }
    /**
     * Copy the values in an array to the device memory.
     *
     * @param data     the data to copy
     * @param blocking if true, this call will block until the transfer is complete.  If false,
     *                 the source array  must be in page-locked memory.
     */
    void upload(const void* data, bool blocking=true);
    /**
     * Copy the values in the device memory to an array.
     *
     * @param data     the array to copy the memory to
     * @param blocking if true, this call will block until the transfer is complete.  If false,
     *                 the destination array must be in page-locked memory.
     */
    void download(void* data, bool blocking=true) const;
    /**
     * Copy the values in the device memory to a second array.
     *
     * @param dest     the destination array to copy to
     */
    void copyTo(ArrayInterface& dest) const;
private:
    HipContext* context;
    hipDeviceptr_t pointer;
    int size, elementSize;
    bool ownsMemory;
    std::string name;
};

} // namespace OpenMM

#endif /*OPENMM_HIPARRAY_H_*/
