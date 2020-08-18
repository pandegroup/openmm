#include "hip/hip_runtime.h"
/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2011-2019 Stanford University and the Authors.      *
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

#include "HipBondedUtilities.h"
#include "HipContext.h"
#include "HipExpressionUtilities.h"
#include "HipKernelSources.h"
#include "openmm/OpenMMException.h"
#include "HipNonbondedUtilities.h"
#include <iostream>

using namespace OpenMM;
using namespace std;

HipBondedUtilities::HipBondedUtilities(HipContext& context) : context(context), numForceBuffers(0), maxBonds(0), allGroups(0), hasInitializedKernels(false) {
}

void HipBondedUtilities::addInteraction(const vector<vector<int> >& atoms, const string& source, int group) {
    if (atoms.size() > 0) {
        forceAtoms.push_back(atoms);
        forceSource.push_back(source);
        forceGroup.push_back(group);
        allGroups |= 1<<group;
    }
}

string HipBondedUtilities::addArgument(hipDeviceptr_t data, const string& type) {
    arguments.push_back(data);
    argTypes.push_back(type);
    return "customArg"+context.intToString(arguments.size());
}

string HipBondedUtilities::addArgument(ArrayInterface& data, const string& type) {
    return addArgument(context.unwrap(data).getDevicePointer(), type);
}

string HipBondedUtilities::addEnergyParameterDerivative(const string& param) {
    // See if the parameter has already been added.

    int index;
    for (index = 0; index < energyParameterDerivatives.size(); index++)
        if (param == energyParameterDerivatives[index])
            break;
    if (index == energyParameterDerivatives.size())
        energyParameterDerivatives.push_back(param);
    context.addEnergyParameterDerivative(param);
    return string("energyParamDeriv")+context.intToString(index);
}

void HipBondedUtilities::addPrefixCode(const string& source) {
    for (int i = 0; i < (int) prefixCode.size(); i++)
        if (prefixCode[i] == source)
            return;
    prefixCode.push_back(source);
}

void HipBondedUtilities::initialize(const System& system) {
    int numForces = forceAtoms.size();
    hasInteractions = (numForces > 0);
    if (!hasInteractions)
        return;

    // Build the lists of atom indices.

    atomIndices.resize(numForces);
    for (int i = 0; i < numForces; i++) {
        int numBonds = forceAtoms[i].size();
        int numAtoms = forceAtoms[i][0].size();
        int numArrays = (numAtoms+3)/4;
        int startAtom = 0;
        atomIndices[i].resize(numArrays);
        for (int j = 0; j < numArrays; j++) {
            int width = min(numAtoms-startAtom, 4);
            int paddedWidth = (width == 3 ? 4 : width);
            vector<unsigned int> indexVec(paddedWidth*numBonds);
            for (int bond = 0; bond < numBonds; bond++) {
                for (int atom = 0; atom < width; atom++)
                    indexVec[bond*paddedWidth+atom] = forceAtoms[i][bond][startAtom+atom];
            }
            atomIndices[i][j].initialize(context, numBonds, 4*paddedWidth, "bondedIndices");
            atomIndices[i][j].upload(&indexVec[0]);
            startAtom += width;
        }
    }

    // Create the kernel.

    stringstream s;
    s<<HipKernelSources::vectorOps;
    for (int i = 0; i < (int) prefixCode.size(); i++)
        s<<prefixCode[i];
    s<<"extern \"C\" __global__ void computeBondedForces(unsigned long long* __restrict__ forceBuffer, mixed* __restrict__ energyBuffer, const real4* __restrict__ posq, int groups, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ";
    for (int force = 0; force < numForces; force++) {
        for (int i = 0; i < (int) atomIndices[force].size(); i++) {
            int indexWidth = atomIndices[force][i].getElementSize()/4;
            string indexType = "uint"+context.intToString(indexWidth);
            s<<", const "<<indexType<<"* __restrict__ atomIndices"<<force<<"_"<<i;
        }
    }
    for (int i = 0; i < (int) arguments.size(); i++)
        s<<", "<<argTypes[i]<<"* customArg"<<(i+1);
    if (energyParameterDerivatives.size() > 0)
        s<<", mixed* __restrict__ energyParamDerivs";
    s<<") {\n";
    s<<"mixed energy = 0;\n";
    for (int i = 0; i < energyParameterDerivatives.size(); i++)
        s<<"mixed energyParamDeriv"<<i<<" = 0;\n";
    for (int force = 0; force < numForces; force++)
        s<<createForceSource(force, forceAtoms[force].size(), forceAtoms[force][0].size(), forceGroup[force], forceSource[force]);
    s<<"energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] += energy;\n";
    const vector<string>& allParamDerivNames = context.getEnergyParamDerivNames();
    int numDerivs = allParamDerivNames.size();
    for (int i = 0; i < energyParameterDerivatives.size(); i++)
        for (int index = 0; index < numDerivs; index++)
            if (allParamDerivNames[index] == energyParameterDerivatives[i])
                s<<"energyParamDerivs[(blockIdx.x*blockDim.x+threadIdx.x)*"<<numDerivs<<"+"<<index<<"] += energyParamDeriv"<<i<<";\n";
    s<<"}\n";
    map<string, string> defines;
    defines["PADDED_NUM_ATOMS"] = context.intToString(context.getPaddedNumAtoms());
    hipModule_t module = context.createModule(s.str(), defines);
    kernel = context.getKernel(module, "computeBondedForces");
    forceAtoms.clear();
    forceSource.clear();
}

string HipBondedUtilities::createForceSource(int forceIndex, int numBonds, int numAtoms, int group, const string& computeForce) {
    maxBonds = max(maxBonds, numBonds);
    string suffix[] = {".x", ".y", ".z", ".w"};
    stringstream s;
    s<<"if ((groups&"<<(1<<group)<<") != 0)\n";
    s<<"for (unsigned int index = blockIdx.x*blockDim.x+threadIdx.x; index < "<<numBonds<<"; index += blockDim.x*gridDim.x) {\n";
    int startAtom = 0;
    for (int i = 0; i < (int) atomIndices[forceIndex].size(); i++) {
        int indexWidth = atomIndices[forceIndex][i].getElementSize()/4;
        string indexType = "uint"+context.intToString(indexWidth);
        s<<"    "<<indexType<<" atoms"<<i<<" = atomIndices"<<forceIndex<<"_"<<i<<"[index];\n";
        int atomsToLoad = min(indexWidth, numAtoms-startAtom);
        for (int j = 0; j < atomsToLoad; j++) {
            s<<"    unsigned int atom"<<(startAtom+j+1)<<" = atoms"<<i<<suffix[j]<<";\n";
            s<<"    real4 pos"<<(startAtom+j+1)<<" = posq[atom"<<(startAtom+j+1)<<"];\n";
        }
        startAtom += indexWidth;
    }
    s<<computeForce<<"\n";
    for (int i = 0; i < numAtoms; i++) {
        s<<"    atomicAdd(&forceBuffer[atom"<<(i+1)<<"], static_cast<unsigned long long>((long long) (force"<<(i+1)<<".x*0x100000000)));\n";
        s<<"    atomicAdd(&forceBuffer[atom"<<(i+1)<<"+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force"<<(i+1)<<".y*0x100000000)));\n";
        s<<"    atomicAdd(&forceBuffer[atom"<<(i+1)<<"+PADDED_NUM_ATOMS*2], static_cast<unsigned long long>((long long) (force"<<(i+1)<<".z*0x100000000)));\n";
        s<<"    __threadfence_block();\n";
    }
    s<<"}\n";
    return s.str();
}

void HipBondedUtilities::computeInteractions(int groups) {
    if ((groups&allGroups) == 0)
        return;
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        kernelArgs.push_back(&context.getForce().getDevicePointer());
        kernelArgs.push_back(&context.getEnergyBuffer().getDevicePointer());
        kernelArgs.push_back(&context.getPosq().getDevicePointer());
        kernelArgs.push_back(NULL);
        kernelArgs.push_back(context.getPeriodicBoxSizePointer());
        kernelArgs.push_back(context.getInvPeriodicBoxSizePointer());
        kernelArgs.push_back(context.getPeriodicBoxVecXPointer());
        kernelArgs.push_back(context.getPeriodicBoxVecYPointer());
        kernelArgs.push_back(context.getPeriodicBoxVecZPointer());
        for (int i = 0; i < (int) atomIndices.size(); i++)
            for (int j = 0; j < (int) atomIndices[i].size(); j++)
                kernelArgs.push_back(&atomIndices[i][j].getDevicePointer());
        for (int i = 0; i < (int) arguments.size(); i++)
            kernelArgs.push_back(&arguments[i]);
        if (energyParameterDerivatives.size() > 0)
            kernelArgs.push_back(&context.getEnergyParamDerivBuffer().getDevicePointer());
    }
    if (!hasInteractions)
        return;
    kernelArgs[3] = &groups;
    context.executeKernel(kernel, &kernelArgs[0], maxBonds);
}
