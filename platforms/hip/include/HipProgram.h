#ifndef OPENMM_HIPPROGRAM_H_
#define OPENMM_HIPPROGRAM_H_

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

#include "openmm/common/ComputeProgram.h"
#include "HipContext.h"

namespace OpenMM {

/**
 * This is the CUDA implementation of the ComputeProgramImpl interface.
 */

class HipProgram : public ComputeProgramImpl {
public:
    /**
     * Create a new HipProgram.
     *
     * @param context      the context this kernel belongs to
     * @param module       the compiled module
     */
    HipProgram(HipContext& context, hipModule_t module);
    /**
     * Create a ComputeKernel for one of the kernels in this program.
     *
     * @param name    the name of the kernel to get
     */
    ComputeKernel createKernel(const std::string& name);
private:
    HipContext& context;
    hipModule_t module;
};

} // namespace OpenMM

#endif /*OPENMM_HIPPROGRAM_H_*/
