#ifndef OPENMM_HIPKERNEL_H_
#define OPENMM_HIPKERNEL_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2019 Stanford University and the Authors.           *
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
#include "HipContext.h"
#include <string>
#include <vector>

namespace OpenMM {

/**
 * This is the CUDA implementation of the ComputeKernelImpl interface.
 */

class HipKernel : public ComputeKernelImpl {
public:
    /**
     * Create a new HipKernel.
     *
     * @param context      the context this kernel belongs to
     * @param kernel       the kernel to be invoked
     * @param name         the name of the kernel function
     */
    HipKernel(HipContext& context, hipFunction_t kernel, const std::string& name);
    /**
     * Get the name of this kernel.
     */
    std::string getName() const;
    /**
     * Execute this kernel.
     *
     * @param threads      the maximum number of threads that should be used.  Depending on the
     *                     computing device, it may choose to use fewer threads than this number.
     * @param blockSize    the number of threads in each thread block.  If this is omitted, a
     *                     default size that is appropriate for the computing device is used.
     */
    void execute(int threads, int blockSize=-1);
protected:
    /**
     * Add an argument to pass the kernel when it is invoked, where the value is a
     * subclass of ArrayInterface.
     *
     * @param value     the value to pass to the kernel
     */
    void addArrayArg(ArrayInterface& value);
    /**
     * Add an argument to pass the kernel when it is invoked, where the value is a primitive type.
     *
     * @param value    a pointer to the argument value
     * @param size     the size of the value in bytes
     */
    void addPrimitiveArg(const void* value, int size);
    /**
     * Add a placeholder for an argument without specifying its value.
     */
    void addEmptyArg();
    /**
     * Add an argument to pass the kernel when it is invoked, where the value is a
     * subclass of ArrayInterface.
     *
     * @param index     the index of the argument to set
     * @param value     the value to pass to the kernel
     */
    void setArrayArg(int index, ArrayInterface& value);
    /**
     * Add an argument to pass the kernel when it is invoked, where the value is a primitive type.
     *
     * @param index     the index of the argument to set
     * @param value    a pointer to the argument value
     * @param size     the size of the value in bytes
     */
    void setPrimitiveArg(int index, const void* value, int size);
private:
    HipContext& context;
    hipFunction_t kernel;
    std::string name;
    std::vector<double4> primitiveArgs;
    std::vector<HipArray*> arrayArgs;
    std::vector<void*> argPointers;
};

} // namespace OpenMM

#endif /*OPENMM_HIPKERNEL_H_*/
