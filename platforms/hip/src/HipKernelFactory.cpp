/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2019 Stanford University and the Authors.      *
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

#include "HipKernelFactory.h"
#include "HipKernels.h"
#include "HipParallelKernels.h"
#include "HipPlatform.h"
#include "openmm/common/CommonKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;

KernelImpl* HipKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    HipPlatform::PlatformData& data = *static_cast<HipPlatform::PlatformData*>(context.getPlatformData());
    if (data.contexts.size() > 1) {
        // We are running in parallel on multiple devices, so we may want to create a parallel kernel.

        if (name == CalcForcesAndEnergyKernel::Name())
            return new HipParallelCalcForcesAndEnergyKernel(name, platform, data);
        if (name == CalcHarmonicBondForceKernel::Name())
            return new HipParallelCalcHarmonicBondForceKernel(name, platform, data, context.getSystem());
        if (name == CalcCustomBondForceKernel::Name())
            return new HipParallelCalcCustomBondForceKernel(name, platform, data, context.getSystem());
        if (name == CalcHarmonicAngleForceKernel::Name())
            return new HipParallelCalcHarmonicAngleForceKernel(name, platform, data, context.getSystem());
        if (name == CalcCustomAngleForceKernel::Name())
            return new HipParallelCalcCustomAngleForceKernel(name, platform, data, context.getSystem());
        if (name == CalcPeriodicTorsionForceKernel::Name())
            return new HipParallelCalcPeriodicTorsionForceKernel(name, platform, data, context.getSystem());
        if (name == CalcRBTorsionForceKernel::Name())
            return new HipParallelCalcRBTorsionForceKernel(name, platform, data, context.getSystem());
        if (name == CalcCMAPTorsionForceKernel::Name())
            return new HipParallelCalcCMAPTorsionForceKernel(name, platform, data, context.getSystem());
        if (name == CalcCustomTorsionForceKernel::Name())
            return new HipParallelCalcCustomTorsionForceKernel(name, platform, data, context.getSystem());
        if (name == CalcNonbondedForceKernel::Name())
            return new HipParallelCalcNonbondedForceKernel(name, platform, data, context.getSystem());
        if (name == CalcCustomNonbondedForceKernel::Name())
            return new HipParallelCalcCustomNonbondedForceKernel(name, platform, data, context.getSystem());
        if (name == CalcCustomExternalForceKernel::Name())
            return new HipParallelCalcCustomExternalForceKernel(name, platform, data, context.getSystem());
        if (name == CalcCustomHbondForceKernel::Name())
            return new HipParallelCalcCustomHbondForceKernel(name, platform, data, context.getSystem());
        if (name == CalcCustomCompoundBondForceKernel::Name())
            return new HipParallelCalcCustomCompoundBondForceKernel(name, platform, data, context.getSystem());
    }
    HipContext& cu = *data.contexts[0];
    if (name == CalcForcesAndEnergyKernel::Name())
        return new HipCalcForcesAndEnergyKernel(name, platform, cu);
    if (name == UpdateStateDataKernel::Name())
        return new HipUpdateStateDataKernel(name, platform, cu);
    if (name == ApplyConstraintsKernel::Name())
        return new HipApplyConstraintsKernel(name, platform, cu);
    if (name == VirtualSitesKernel::Name())
        return new HipVirtualSitesKernel(name, platform, cu);
    if (name == CalcHarmonicBondForceKernel::Name())
        return new CommonCalcHarmonicBondForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomBondForceKernel::Name())
        return new CommonCalcCustomBondForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcHarmonicAngleForceKernel::Name())
        return new CommonCalcHarmonicAngleForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomAngleForceKernel::Name())
        return new CommonCalcCustomAngleForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcPeriodicTorsionForceKernel::Name())
        return new CommonCalcPeriodicTorsionForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcRBTorsionForceKernel::Name())
        return new CommonCalcRBTorsionForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCMAPTorsionForceKernel::Name())
        return new CommonCalcCMAPTorsionForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomTorsionForceKernel::Name())
        return new CommonCalcCustomTorsionForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcNonbondedForceKernel::Name())
        return new HipCalcNonbondedForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomNonbondedForceKernel::Name())
        return new CommonCalcCustomNonbondedForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcGBSAOBCForceKernel::Name())
        return new CommonCalcGBSAOBCForceKernel(name, platform, cu);
    if (name == CalcCustomGBForceKernel::Name())
        return new CommonCalcCustomGBForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomExternalForceKernel::Name())
        return new CommonCalcCustomExternalForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomHbondForceKernel::Name())
        return new CommonCalcCustomHbondForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomCentroidBondForceKernel::Name())
        return new CommonCalcCustomCentroidBondForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomCompoundBondForceKernel::Name())
        return new CommonCalcCustomCompoundBondForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcCustomCVForceKernel::Name())
        return new HipCalcCustomCVForceKernel(name, platform, cu);
    if (name == CalcRMSDForceKernel::Name())
        return new CommonCalcRMSDForceKernel(name, platform, cu);
    if (name == CalcCustomManyParticleForceKernel::Name())
        return new CommonCalcCustomManyParticleForceKernel(name, platform, cu, context.getSystem());
    if (name == CalcGayBerneForceKernel::Name())
        return new CommonCalcGayBerneForceKernel(name, platform, cu);
    if (name == IntegrateVerletStepKernel::Name())
        return new CommonIntegrateVerletStepKernel(name, platform, cu);
    if (name == IntegrateLangevinStepKernel::Name())
        return new CommonIntegrateLangevinStepKernel(name, platform, cu);
    if (name == IntegrateLangevinMiddleStepKernel::Name())
        return new CommonIntegrateLangevinMiddleStepKernel(name, platform, cu);
    if (name == IntegrateBrownianStepKernel::Name())
        return new CommonIntegrateBrownianStepKernel(name, platform, cu);
    if (name == IntegrateVariableVerletStepKernel::Name())
        return new CommonIntegrateVariableVerletStepKernel(name, platform, cu);
    if (name == IntegrateVariableLangevinStepKernel::Name())
        return new CommonIntegrateVariableLangevinStepKernel(name, platform, cu);
    if (name == IntegrateCustomStepKernel::Name())
        return new CommonIntegrateCustomStepKernel(name, platform, cu);
    if (name == ApplyAndersenThermostatKernel::Name())
        return new CommonApplyAndersenThermostatKernel(name, platform, cu);
    if (name == IntegrateNoseHooverStepKernel::Name())
        return new CommonIntegrateNoseHooverStepKernel(name, platform, cu);
    if (name == ApplyMonteCarloBarostatKernel::Name())
        return new HipApplyMonteCarloBarostatKernel(name, platform, cu);
    if (name == RemoveCMMotionKernel::Name())
        return new CommonRemoveCMMotionKernel(name, platform, cu);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
