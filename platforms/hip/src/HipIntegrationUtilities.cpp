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

#include "HipIntegrationUtilities.h"
#include "HipContext.h"

using namespace OpenMM;
using namespace std;

#define CHECK_RESULT(result) CHECK_RESULT2(result, errorMessage);
#define CHECK_RESULT2(result, prefix) \
    if (result != hipSuccess) { \
        std::stringstream m; \
        m<<prefix<<": "<<dynamic_cast<HipContext&>(context).getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }

HipIntegrationUtilities::HipIntegrationUtilities(HipContext& context, const System& system) : IntegrationUtilities(context, system),
        ccmaConvergedMemory(NULL) {
        CHECK_RESULT2(hipEventCreateWithFlags(&ccmaEvent, hipEventDisableTiming), "Error creating event for CCMA");
        CHECK_RESULT2(hipHostMalloc((void**) &ccmaConvergedMemory, sizeof(int), hipHostMallocMapped), "Error allocating pinned memory");
        CHECK_RESULT2(hipHostGetDevicePointer(&ccmaConvergedDeviceMemory, ccmaConvergedMemory, 0), "Error getting device address for pinned memory");
}

HipIntegrationUtilities::~HipIntegrationUtilities() {
    context.setAsCurrent();
    if (ccmaConvergedMemory != NULL) {
        hipHostFree(ccmaConvergedMemory);
        hipEventDestroy(ccmaEvent);
    }
}

HipArray& HipIntegrationUtilities::getPosDelta() {
    return dynamic_cast<HipContext&>(context).unwrap(posDelta);
}

HipArray& HipIntegrationUtilities::getRandom() {
    return dynamic_cast<HipContext&>(context).unwrap(random);
}

HipArray& HipIntegrationUtilities::getStepSize() {
    return dynamic_cast<HipContext&>(context).unwrap(stepSize);
}

void HipIntegrationUtilities::applyConstraintsImpl(bool constrainVelocities, double tol) {
    ComputeKernel settleKernel, shakeKernel, ccmaForceKernel;
    if (constrainVelocities) {
        settleKernel = settleVelKernel;
        shakeKernel = shakeVelKernel;
        ccmaForceKernel = ccmaVelForceKernel;
    }
    else {
        settleKernel = settlePosKernel;
        shakeKernel = shakePosKernel;
        ccmaForceKernel = ccmaPosForceKernel;
    }
    if (settleAtoms.isInitialized()) {
        if (context.getUseDoublePrecision() || context.getUseMixedPrecision())
            settleKernel->setArg(1, tol);
        else
            settleKernel->setArg(1, (float) tol);
        settleKernel->execute(settleAtoms.getSize());
    }
    if (shakeAtoms.isInitialized()) {
        if (context.getUseDoublePrecision() || context.getUseMixedPrecision())
            shakeKernel->setArg(1, tol);
        else
            shakeKernel->setArg(1, (float) tol);
        shakeKernel->execute(shakeAtoms.getSize());
    }
    if (ccmaAtoms.isInitialized()) {
        ccmaForceKernel->setArg(6, ccmaConvergedDeviceMemory);
        if (context.getUseDoublePrecision() || context.getUseMixedPrecision())
            ccmaForceKernel->setArg(7, tol);
        else
            ccmaForceKernel->setArg(7, (float) tol);
        ccmaDirectionsKernel->execute(ccmaAtoms.getSize());
        const int checkInterval = 4;
        ccmaConvergedMemory[0] = 0;
        ccmaUpdateKernel->setArg(3, constrainVelocities ? context.getVelm() : posDelta);
        for (int i = 0; i < 150; i++) {
            ccmaForceKernel->setArg(8, i);
            ccmaForceKernel->execute(ccmaAtoms.getSize());
            if ((i+1)%checkInterval == 0)
                CHECK_RESULT2(hipEventRecord(ccmaEvent, 0), "Error recording event for CCMA");
            ccmaMultiplyKernel->setArg(5, i);
            ccmaMultiplyKernel->execute(ccmaAtoms.getSize());
            ccmaUpdateKernel->setArg(8, i);
            ccmaUpdateKernel->execute(context.getNumAtoms());
            if ((i+1)%checkInterval == 0) {
                CHECK_RESULT2(hipEventSynchronize(ccmaEvent), "Error synchronizing on event for CCMA");
                if (ccmaConvergedMemory[0])
                    break;
            }
        }
    }
}

void HipIntegrationUtilities::distributeForcesFromVirtualSites() {
    if (numVsites > 0) {
        vsiteForceKernel->setArg(2, context.getLongForceBuffer());
        vsiteForceKernel->execute(numVsites);
    }
}
