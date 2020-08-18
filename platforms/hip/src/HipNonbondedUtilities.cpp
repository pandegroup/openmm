/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2018 Stanford University and the Authors.      *
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
#include "HipNonbondedUtilities.h"
#include "HipArray.h"
#include "HipContext.h"
#include "HipKernelSources.h"
#include "HipExpressionUtilities.h"
#include "HipSort.h"
#include <algorithm>
#include <map>
#include <set>
#include <utility>

using namespace OpenMM;
using namespace std;

#define CHECK_RESULT(result) \
    if (result != hipSuccess) { \
        std::stringstream m; \
        m<<errorMessage<<": "<<context.getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }


class HipNonbondedUtilities::BlockSortTrait : public HipSort::SortTrait {
public:
    BlockSortTrait(bool useDouble) : useDouble(useDouble) {
    }
    int getDataSize() const {return useDouble ? sizeof(double2) : sizeof(float2);}
    int getKeySize() const {return useDouble ? sizeof(double) : sizeof(float);}
    const char* getDataType() const {return "real2";}
    const char* getKeyType() const {return "real";}
    const char* getMinKey() const {return "-3.40282e+38f";}
    const char* getMaxKey() const {return "3.40282e+38f";}
    const char* getMaxValue() const {return "make_real2(3.40282e+38f, 3.40282e+38f)";}
    const char* getSortKey() const {return "value.x";}
private:
    bool useDouble;
};

HipNonbondedUtilities::HipNonbondedUtilities(HipContext& context) : context(context), useCutoff(false), usePeriodic(false), anyExclusions(false), usePadding(true),
        blockSorter(NULL), pinnedCountBuffer(NULL), forceRebuildNeighborList(true), lastCutoff(0.0), groupFlags(0), canUsePairList(true) {
    // Decide how many thread blocks to use.

    string errorMessage = "Error initializing nonbonded utilities";
    int multiprocessors;
    CHECK_RESULT(hipDeviceGetAttribute(&multiprocessors, hipDeviceAttributeMultiprocessorCount, context.getDevice()));
    CHECK_RESULT(hipEventCreateWithFlags(&downloadCountEvent, 0));
    CHECK_RESULT(hipHostMalloc((void**) &pinnedCountBuffer, 2*sizeof(int), hipHostMallocPortable));
    numForceThreadBlocks = 4*multiprocessors;
    forceThreadBlockSize = (256);
    setKernelSource(HipKernelSources::nonbonded);
}

HipNonbondedUtilities::~HipNonbondedUtilities() {
    if (blockSorter != NULL)
        delete blockSorter;
    if (pinnedCountBuffer != NULL)
        hipHostFree(pinnedCountBuffer);
    hipEventDestroy(downloadCountEvent);
}

void HipNonbondedUtilities::addInteraction(bool usesCutoff, bool usesPeriodic, bool usesExclusions, double cutoffDistance, const vector<vector<int> >& exclusionList, const string& kernel, int forceGroup) {
    addInteraction(usesCutoff, usesPeriodic, usesExclusions, cutoffDistance, exclusionList, kernel, forceGroup, false);
}

void HipNonbondedUtilities::addInteraction(bool usesCutoff, bool usesPeriodic, bool usesExclusions, double cutoffDistance, const vector<vector<int> >& exclusionList, const string& kernel, int forceGroup, bool supportsPairList) {
    if (groupCutoff.size() > 0) {
        if (usesCutoff != useCutoff)
            throw OpenMMException("All Forces must agree on whether to use a cutoff");
        if (usesPeriodic != usePeriodic)
            throw OpenMMException("All Forces must agree on whether to use periodic boundary conditions");
        if (usesCutoff && groupCutoff.find(forceGroup) != groupCutoff.end() && groupCutoff[forceGroup] != cutoffDistance)
            throw OpenMMException("All Forces in a single force group must use the same cutoff distance");
    }
    if (usesExclusions)
        requestExclusions(exclusionList);
    useCutoff = usesCutoff;
    usePeriodic = usesPeriodic;
    groupCutoff[forceGroup] = cutoffDistance;
    groupFlags |= 1<<forceGroup;
    canUsePairList &= supportsPairList;
    if (kernel.size() > 0) {
        if (groupKernelSource.find(forceGroup) == groupKernelSource.end())
            groupKernelSource[forceGroup] = "";
        map<string, string> replacements;
        replacements["CUTOFF"] = "CUTOFF_"+context.intToString(forceGroup);
        replacements["CUTOFF_SQUARED"] = "CUTOFF_"+context.intToString(forceGroup)+"_SQUARED";
        groupKernelSource[forceGroup] += context.replaceStrings(kernel, replacements)+"\n";
    }
}

void HipNonbondedUtilities::addParameter(ComputeParameterInfo parameter) {
    parameters.push_back(ParameterInfo(parameter.getName(), parameter.getComponentType(), parameter.getNumComponents(),
            parameter.getSize(), context.unwrap(parameter.getArray()).getDevicePointer()));
}

void HipNonbondedUtilities::addParameter(const ParameterInfo& parameter) {
    parameters.push_back(parameter);
}

void HipNonbondedUtilities::addArgument(ComputeParameterInfo parameter) {
    arguments.push_back(ParameterInfo(parameter.getName(), parameter.getComponentType(), parameter.getNumComponents(),
            parameter.getSize(), context.unwrap(parameter.getArray()).getDevicePointer()));
}

void HipNonbondedUtilities::addArgument(const ParameterInfo& parameter) {
    arguments.push_back(parameter);
}

string HipNonbondedUtilities::addEnergyParameterDerivative(const string& param) {
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

void HipNonbondedUtilities::requestExclusions(const vector<vector<int> >& exclusionList) {
    if (anyExclusions) {
        bool sameExclusions = (exclusionList.size() == atomExclusions.size());
        for (int i = 0; i < (int) exclusionList.size() && sameExclusions; i++) {
             if (exclusionList[i].size() != atomExclusions[i].size())
                 sameExclusions = false;
            set<int> expectedExclusions;
            expectedExclusions.insert(atomExclusions[i].begin(), atomExclusions[i].end());
            for (int j = 0; j < (int) exclusionList[i].size(); j++)
                if (expectedExclusions.find(exclusionList[i][j]) == expectedExclusions.end())
                     sameExclusions = false;
        }
        if (!sameExclusions)
            throw OpenMMException("All Forces must have identical exceptions");
    }
    else {
        atomExclusions = exclusionList;
        anyExclusions = true;
    }
}

static bool compareUshort2(ushort2 a, ushort2 b) {
    return ((a.y < b.y) || (a.y == b.y && a.x < b.x));
}

void HipNonbondedUtilities::initialize(const System& system) {
    string errorMessage = "Error initializing nonbonded utilities";
    if (atomExclusions.size() == 0) {
        // No exclusions were specifically requested, so just mark every atom as not interacting with itself.

        atomExclusions.resize(context.getNumAtoms());
        for (int i = 0; i < (int) atomExclusions.size(); i++)
            atomExclusions[i].push_back(i);
    }

    // Create the list of tiles.

    numAtoms = context.getNumAtoms();
    int numAtomBlocks = context.getNumAtomBlocks();
    int numContexts = context.getPlatformData().contexts.size();
    setAtomBlockRange(context.getContextIndex()/(double) numContexts, (context.getContextIndex()+1)/(double) numContexts);

    // Build a list of tiles that contain exclusions.

    set<pair<int, int> > tilesWithExclusions;
    for (int atom1 = 0; atom1 < (int) atomExclusions.size(); ++atom1) {
        int x = atom1/HipContext::TileSize;
        for (int j = 0; j < (int) atomExclusions[atom1].size(); ++j) {
            int atom2 = atomExclusions[atom1][j];
            int y = atom2/HipContext::TileSize;
            tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
        }
    }
    vector<ushort2> exclusionTilesVec;
    for (set<pair<int, int> >::const_iterator iter = tilesWithExclusions.begin(); iter != tilesWithExclusions.end(); ++iter)
        exclusionTilesVec.push_back(make_ushort2((unsigned short) iter->first, (unsigned short) iter->second));
    sort(exclusionTilesVec.begin(), exclusionTilesVec.end(), compareUshort2);
    exclusionTiles.initialize<ushort2>(context, exclusionTilesVec.size(), "exclusionTiles");
    exclusionTiles.upload(exclusionTilesVec);
    map<pair<int, int>, int> exclusionTileMap;
    for (int i = 0; i < (int) exclusionTilesVec.size(); i++) {
        ushort2 tile = exclusionTilesVec[i];
        exclusionTileMap[make_pair(tile.x, tile.y)] = i;
    }
    vector<vector<int> > exclusionBlocksForBlock(numAtomBlocks);
    for (set<pair<int, int> >::const_iterator iter = tilesWithExclusions.begin(); iter != tilesWithExclusions.end(); ++iter) {
        exclusionBlocksForBlock[iter->first].push_back(iter->second);
        if (iter->first != iter->second)
            exclusionBlocksForBlock[iter->second].push_back(iter->first);
    }
    vector<unsigned int> exclusionRowIndicesVec(numAtomBlocks+1, 0);
    vector<unsigned int> exclusionIndicesVec;
    for (int i = 0; i < numAtomBlocks; i++) {
        exclusionIndicesVec.insert(exclusionIndicesVec.end(), exclusionBlocksForBlock[i].begin(), exclusionBlocksForBlock[i].end());
        exclusionRowIndicesVec[i+1] = exclusionIndicesVec.size();
    }
    maxExclusions = 0;
    for (int i = 0; i < (int) exclusionBlocksForBlock.size(); i++)
        maxExclusions = (maxExclusions > exclusionBlocksForBlock[i].size() ? maxExclusions : exclusionBlocksForBlock[i].size());
    exclusionIndices.initialize<unsigned int>(context, exclusionIndicesVec.size(), "exclusionIndices");
    exclusionRowIndices.initialize<unsigned int>(context, exclusionRowIndicesVec.size(), "exclusionRowIndices");
    exclusionIndices.upload(exclusionIndicesVec);
    exclusionRowIndices.upload(exclusionRowIndicesVec);

    // Record the exclusion data.

    exclusions.initialize<tileflags>(context, tilesWithExclusions.size()*HipContext::TileSize, "exclusions");
    tileflags allFlags = static_cast<tileflags>(-1);
    vector<tileflags> exclusionVec(exclusions.getSize(), allFlags);
    for (int atom1 = 0; atom1 < (int) atomExclusions.size(); ++atom1) {
        int x = atom1/HipContext::TileSize;
        int offset1 = atom1-x*HipContext::TileSize;
        for (int j = 0; j < (int) atomExclusions[atom1].size(); ++j) {
            int atom2 = atomExclusions[atom1][j];
            int y = atom2/HipContext::TileSize;
            int offset2 = atom2-y*HipContext::TileSize;
            if (x > y) {
                int index = exclusionTileMap[make_pair(x, y)]*HipContext::TileSize;
                exclusionVec[index+offset1] &= allFlags-(static_cast<tileflags>(1)<<offset2);
            }
            else {
                int index = exclusionTileMap[make_pair(y, x)]*HipContext::TileSize;
                exclusionVec[index+offset2] &= allFlags-(static_cast<tileflags>(1)<<offset1);
            }
        }
    }
    atomExclusions.clear(); // We won't use this again, so free the memory it used
    exclusions.upload(exclusionVec);

    // Create data structures for the neighbor list.

    if (useCutoff) {
        // Select a size for the arrays that hold the neighbor list.  We have to make a fairly
        // arbitrary guess, but if this turns out to be too small we'll increase it later.

        maxTiles = 20*numAtomBlocks;
        if (maxTiles > numTiles)
            maxTiles = numTiles;
        if (maxTiles < 1)
            maxTiles = 1;
        maxSinglePairs = 5*numAtoms;
        interactingTiles.initialize<int>(context, maxTiles, "interactingTiles");
        interactingAtoms.initialize<int>(context, HipContext::TileSize*maxTiles, "interactingAtoms");
        interactionCount.initialize<unsigned int>(context, 2, "interactionCount");
        singlePairs.initialize<int2>(context, maxSinglePairs, "singlePairs");
        int elementSize = (context.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
        blockCenter.initialize(context, numAtomBlocks, 4*elementSize, "blockCenter");
        blockBoundingBox.initialize(context, numAtomBlocks, 4*elementSize, "blockBoundingBox");
        sortedBlocks.initialize(context, numAtomBlocks, 2*elementSize, "sortedBlocks");
        sortedBlockCenter.initialize(context, numAtomBlocks+1, 4*elementSize, "sortedBlockCenter");
        sortedBlockBoundingBox.initialize(context, numAtomBlocks+1, 4*elementSize, "sortedBlockBoundingBox");
        oldPositions.initialize(context, numAtoms, 4*elementSize, "oldPositions");
        rebuildNeighborList.initialize<int>(context, 1, "rebuildNeighborList");
        blockSorter = new HipSort(context, new BlockSortTrait(context.getUseDoublePrecision()), numAtomBlocks);
        vector<unsigned int> count(2, 0);
        interactionCount.upload(count);
        rebuildNeighborList.upload(&count[0]);
    }

    // Record arguments for kernels.

    forceArgs.push_back(&context.getForce().getDevicePointer());
    forceArgs.push_back(&context.getEnergyBuffer().getDevicePointer());
    forceArgs.push_back(&context.getPosq().getDevicePointer());
    forceArgs.push_back(&exclusions.getDevicePointer());
    forceArgs.push_back(&exclusionTiles.getDevicePointer());
    forceArgs.push_back(&startTileIndex);
    forceArgs.push_back(&numTiles);
    if (useCutoff) {
        forceArgs.push_back(&interactingTiles.getDevicePointer());
        forceArgs.push_back(&interactionCount.getDevicePointer());
        forceArgs.push_back(context.getPeriodicBoxSizePointer());
        forceArgs.push_back(context.getInvPeriodicBoxSizePointer());
        forceArgs.push_back(context.getPeriodicBoxVecXPointer());
        forceArgs.push_back(context.getPeriodicBoxVecYPointer());
        forceArgs.push_back(context.getPeriodicBoxVecZPointer());
        forceArgs.push_back(&maxTiles);
        forceArgs.push_back(&blockCenter.getDevicePointer());
        forceArgs.push_back(&blockBoundingBox.getDevicePointer());
        forceArgs.push_back(&interactingAtoms.getDevicePointer());
        forceArgs.push_back(&maxSinglePairs);
        forceArgs.push_back(&singlePairs.getDevicePointer());
    }
    for (int i = 0; i < (int) parameters.size(); i++)
        forceArgs.push_back(&parameters[i].getMemory());
    for (int i = 0; i < (int) arguments.size(); i++)
        forceArgs.push_back(&arguments[i].getMemory());
    if (energyParameterDerivatives.size() > 0)
        forceArgs.push_back(&context.getEnergyParamDerivBuffer().getDevicePointer());
    if (useCutoff) {
        findBlockBoundsArgs.push_back(&numAtoms);
        findBlockBoundsArgs.push_back(context.getPeriodicBoxSizePointer());
        findBlockBoundsArgs.push_back(context.getInvPeriodicBoxSizePointer());
        findBlockBoundsArgs.push_back(context.getPeriodicBoxVecXPointer());
        findBlockBoundsArgs.push_back(context.getPeriodicBoxVecYPointer());
        findBlockBoundsArgs.push_back(context.getPeriodicBoxVecZPointer());
        findBlockBoundsArgs.push_back(&context.getPosq().getDevicePointer());
        findBlockBoundsArgs.push_back(&blockCenter.getDevicePointer());
        findBlockBoundsArgs.push_back(&blockBoundingBox.getDevicePointer());
        findBlockBoundsArgs.push_back(&rebuildNeighborList.getDevicePointer());
        findBlockBoundsArgs.push_back(&sortedBlocks.getDevicePointer());
        sortBoxDataArgs.push_back(&sortedBlocks.getDevicePointer());
        sortBoxDataArgs.push_back(&blockCenter.getDevicePointer());
        sortBoxDataArgs.push_back(&blockBoundingBox.getDevicePointer());
        sortBoxDataArgs.push_back(&sortedBlockCenter.getDevicePointer());
        sortBoxDataArgs.push_back(&sortedBlockBoundingBox.getDevicePointer());
        sortBoxDataArgs.push_back(&context.getPosq().getDevicePointer());
        sortBoxDataArgs.push_back(&oldPositions.getDevicePointer());
        sortBoxDataArgs.push_back(&interactionCount.getDevicePointer());
        sortBoxDataArgs.push_back(&rebuildNeighborList.getDevicePointer());
        sortBoxDataArgs.push_back(&forceRebuildNeighborList);
        findInteractingBlocksArgs.push_back(context.getPeriodicBoxSizePointer());
        findInteractingBlocksArgs.push_back(context.getInvPeriodicBoxSizePointer());
        findInteractingBlocksArgs.push_back(context.getPeriodicBoxVecXPointer());
        findInteractingBlocksArgs.push_back(context.getPeriodicBoxVecYPointer());
        findInteractingBlocksArgs.push_back(context.getPeriodicBoxVecZPointer());
        findInteractingBlocksArgs.push_back(&interactionCount.getDevicePointer());
        findInteractingBlocksArgs.push_back(&interactingTiles.getDevicePointer());
        findInteractingBlocksArgs.push_back(&interactingAtoms.getDevicePointer());
        findInteractingBlocksArgs.push_back(&singlePairs.getDevicePointer());
        findInteractingBlocksArgs.push_back(&context.getPosq().getDevicePointer());
        findInteractingBlocksArgs.push_back(&maxTiles);
        findInteractingBlocksArgs.push_back(&maxSinglePairs);
        findInteractingBlocksArgs.push_back(&startBlockIndex);
        findInteractingBlocksArgs.push_back(&numBlocks);
        findInteractingBlocksArgs.push_back(&sortedBlocks.getDevicePointer());
        findInteractingBlocksArgs.push_back(&sortedBlockCenter.getDevicePointer());
        findInteractingBlocksArgs.push_back(&sortedBlockBoundingBox.getDevicePointer());
        findInteractingBlocksArgs.push_back(&exclusionIndices.getDevicePointer());
        findInteractingBlocksArgs.push_back(&exclusionRowIndices.getDevicePointer());
        findInteractingBlocksArgs.push_back(&oldPositions.getDevicePointer());
        findInteractingBlocksArgs.push_back(&rebuildNeighborList.getDevicePointer());
    }
}

double HipNonbondedUtilities::getMaxCutoffDistance() {
    double cutoff = 0.0;
    for (map<int, double>::const_iterator iter = groupCutoff.begin(); iter != groupCutoff.end(); ++iter)
        cutoff = max(cutoff, iter->second);
    return cutoff;
}

double HipNonbondedUtilities::padCutoff(double cutoff) {
    double padding = (usePadding ? 0.1*cutoff : 0.0);
    return cutoff+padding;
}

void HipNonbondedUtilities::prepareInteractions(int forceGroups) {
    if ((forceGroups&groupFlags) == 0)
        return;
    if (groupKernels.find(forceGroups) == groupKernels.end())
        createKernelsForGroups(forceGroups);
    if (!useCutoff)
        return;
    if (numTiles == 0)
        return;
    KernelSet& kernels = groupKernels[forceGroups];
    if (usePeriodic) {
        double4 box = context.getPeriodicBoxSize();
        double minAllowedSize = 1.999999*kernels.cutoffDistance;
        if (box.x < minAllowedSize || box.y < minAllowedSize || box.z < minAllowedSize)
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
    }

    // Compute the neighbor list.

    if (lastCutoff != kernels.cutoffDistance)
        forceRebuildNeighborList = true;
    context.executeKernel(kernels.findBlockBoundsKernel, &findBlockBoundsArgs[0], context.getNumAtoms());
    blockSorter->sort(sortedBlocks);
    context.executeKernel(kernels.sortBoxDataKernel, &sortBoxDataArgs[0], context.getNumAtoms());
    context.executeKernel(kernels.findInteractingBlocksKernel, &findInteractingBlocksArgs[0], context.getNumAtoms(), 256);
    forceRebuildNeighborList = false;
    lastCutoff = kernels.cutoffDistance;
    interactionCount.download(pinnedCountBuffer, false);
    hipEventRecord(downloadCountEvent, context.getCurrentStream());
}

void HipNonbondedUtilities::computeInteractions(int forceGroups, bool includeForces, bool includeEnergy) {
    if ((forceGroups&groupFlags) == 0)
        return;
    KernelSet& kernels = groupKernels[forceGroups];
    if (kernels.hasForces) {
        hipFunction_t& kernel = (includeForces ? (includeEnergy ? kernels.forceEnergyKernel : kernels.forceKernel) : kernels.energyKernel);
        if (kernel == NULL)
            kernel = createInteractionKernel(kernels.source, parameters, arguments, true, true, forceGroups, includeForces, includeEnergy);
        context.executeKernel(kernel, &forceArgs[0], numForceThreadBlocks*forceThreadBlockSize, forceThreadBlockSize);
    }
    if (useCutoff && numTiles > 0) {
        hipEventSynchronize(downloadCountEvent);
        updateNeighborListSize();
    }
}

bool HipNonbondedUtilities::updateNeighborListSize() {
    if (!useCutoff)
        return false;
    if (pinnedCountBuffer[0] <= maxTiles && pinnedCountBuffer[1] <= maxSinglePairs)
        return false;

    // The most recent timestep had too many interactions to fit in the arrays.  Make the arrays bigger to prevent
    // this from happening in the future.

    if (pinnedCountBuffer[0] > maxTiles) {
        maxTiles = (int) (1.2*pinnedCountBuffer[0]);
        int totalTiles = context.getNumAtomBlocks()*(context.getNumAtomBlocks()+1)/2;
        if (maxTiles > totalTiles)
            maxTiles = totalTiles;
        interactingTiles.resize(maxTiles);
        interactingAtoms.resize(HipContext::TileSize*maxTiles);
        if (forceArgs.size() > 0)
            forceArgs[7] = &interactingTiles.getDevicePointer();
        findInteractingBlocksArgs[6] = &interactingTiles.getDevicePointer();
        if (forceArgs.size() > 0)
            forceArgs[17] = &interactingAtoms.getDevicePointer();
        findInteractingBlocksArgs[7] = &interactingAtoms.getDevicePointer();
    }
    if (pinnedCountBuffer[1] > maxSinglePairs) {
        maxSinglePairs = (int) (1.2*pinnedCountBuffer[1]);
        singlePairs.resize(maxSinglePairs);
        if (forceArgs.size() > 0)
            forceArgs[19] = &singlePairs.getDevicePointer();
        findInteractingBlocksArgs[8] = &singlePairs.getDevicePointer();
    }
    forceRebuildNeighborList = true;
    context.setForcesValid(false);
    return true;
}

void HipNonbondedUtilities::setUsePadding(bool padding) {
    usePadding = padding;
}

void HipNonbondedUtilities::setAtomBlockRange(double startFraction, double endFraction) {
    int numAtomBlocks = context.getNumAtomBlocks();
    startBlockIndex = (int) (startFraction*numAtomBlocks);
    numBlocks = (int) (endFraction*numAtomBlocks)-startBlockIndex;
    int totalTiles = context.getNumAtomBlocks()*(context.getNumAtomBlocks()+1)/2;
    startTileIndex = (int) (startFraction*totalTiles);
    numTiles = (int) (endFraction*totalTiles)-startTileIndex;
    forceRebuildNeighborList = true;
}

void HipNonbondedUtilities::createKernelsForGroups(int groups) {
    KernelSet kernels;
    double cutoff = 0.0;
    string source;
    for (int i = 0; i < 32; i++) {
        if ((groups&(1<<i)) != 0) {
            cutoff = max(cutoff, groupCutoff[i]);
            source += groupKernelSource[i];
        }
    }
    kernels.hasForces = (source.size() > 0);
    kernels.cutoffDistance = cutoff;
    kernels.source = source;
    kernels.forceKernel = kernels.energyKernel = kernels.forceEnergyKernel = NULL;
    if (useCutoff) {
        double paddedCutoff = padCutoff(cutoff);
        map<string, string> defines;
        defines["TILE_SIZE"] = context.intToString(HipContext::TileSize);
        defines["NUM_BLOCKS"] = context.intToString(context.getNumAtomBlocks());
        defines["NUM_ATOMS"] = context.intToString(context.getNumAtoms());
        defines["PADDING"] = context.doubleToString(paddedCutoff-cutoff);
        defines["PADDED_CUTOFF"] = context.doubleToString(paddedCutoff);
        defines["PADDED_CUTOFF_SQUARED"] = context.doubleToString(paddedCutoff*paddedCutoff);
        defines["NUM_TILES_WITH_EXCLUSIONS"] = context.intToString(exclusionTiles.getSize());
        if (usePeriodic)
            defines["USE_PERIODIC"] = "1";
        if (context.getBoxIsTriclinic())
            defines["TRICLINIC"] = "1";
        defines["MAX_EXCLUSIONS"] = context.intToString(maxExclusions);
        // Temporarily disable the pair list until we figure out why it's failing on some GPUs.
        defines["MAX_BITS_FOR_PAIRS"] = "0";//(canUsePairList ? "2" : "0");
        hipModule_t interactingBlocksProgram = context.createModule(HipKernelSources::vectorOps+HipKernelSources::findInteractingBlocks, defines);
        kernels.findBlockBoundsKernel = context.getKernel(interactingBlocksProgram, "findBlockBounds");
        kernels.sortBoxDataKernel = context.getKernel(interactingBlocksProgram, "sortBoxData");
        kernels.findInteractingBlocksKernel = context.getKernel(interactingBlocksProgram, "findBlocksWithInteractions");
    }
    groupKernels[groups] = kernels;
}

hipFunction_t HipNonbondedUtilities::createInteractionKernel(const string& source, vector<ParameterInfo>& params, vector<ParameterInfo>& arguments, bool useExclusions, bool isSymmetric, int groups, bool includeForces, bool includeEnergy) {
    map<string, string> replacements;
    replacements["COMPUTE_INTERACTION"] = source;
    const string suffixes[] = {"x", "y", "z", "w"};
    stringstream localData;
    int localDataSize = 0;
    for (int i = 0; i < (int) params.size(); i++) {
        if (params[i].getNumComponents() == 1)
            localData<<params[i].getType()<<" "<<params[i].getName()<<";\n";
        else {
            for (int j = 0; j < params[i].getNumComponents(); ++j)
                localData<<params[i].getComponentType()<<" "<<params[i].getName()<<"_"<<suffixes[j]<<";\n";
        }
        localDataSize += params[i].getSize();
    }
    replacements["ATOM_PARAMETER_DATA"] = localData.str();
    stringstream args;
    for (int i = 0; i < (int) params.size(); i++) {
        args << ", ";
        if (params[i].isConstant())
            args << "const ";
        args << params[i].getType();
        args << "* __restrict__ global_";
        args << params[i].getName();
    }
    for (int i = 0; i < (int) arguments.size(); i++) {
        args << ", ";
        if (arguments[i].isConstant())
            args << "const ";
        args << arguments[i].getType();
        args << "* __restrict__ ";
        args << arguments[i].getName();
    }
    if (energyParameterDerivatives.size() > 0)
        args << ", mixed* __restrict__ energyParamDerivs";
    replacements["PARAMETER_ARGUMENTS"] = args.str();

    stringstream load1;
    for (int i = 0; i < (int) params.size(); i++) {
        load1 << params[i].getType();
        load1 << " ";
        load1 << params[i].getName();
        load1 << "1 = global_";
        load1 << params[i].getName();
        load1 << "[atom1];\n";
    }
    replacements["LOAD_ATOM1_PARAMETERS"] = load1.str();


    bool useShuffle = context.getSupportsWarpShuffle();

    // Part 1. Defines for on diagonal exclusion tiles
    stringstream loadLocal1;
    if(useShuffle) {
        // not needed if using shuffles as we can directly fetch from register
    } else {
        for (int i = 0; i < (int) params.size(); i++) {
            if (params[i].getNumComponents() == 1) {
                loadLocal1<<"localData[threadIdx.x]."<<params[i].getName()<<" = "<<params[i].getName()<<"1;\n";
            }
            else {
                for (int j = 0; j < params[i].getNumComponents(); ++j)
                    loadLocal1<<"localData[threadIdx.x]."<<params[i].getName()<<"_"<<suffixes[j]<<" = "<<params[i].getName()<<"1."<<suffixes[j]<<";\n";
            }
        }
    }
    replacements["LOAD_LOCAL_PARAMETERS_FROM_1"] = loadLocal1.str();

    stringstream broadcastWarpData;
    if(useShuffle) {
        broadcastWarpData << "posq2.x = real_shfl(shflPosq.x, j);\n";
        broadcastWarpData << "posq2.y = real_shfl(shflPosq.y, j);\n";
        broadcastWarpData << "posq2.z = real_shfl(shflPosq.z, j);\n";
        broadcastWarpData << "posq2.w = real_shfl(shflPosq.w, j);\n";
        for(int i=0; i< (int) params.size();i++) {
            broadcastWarpData << params[i].getType() << " shfl" << params[i].getName() << ";\n";
            for(int j=0; j < params[i].getNumComponents(); j++) {
                string name;
                if (params[i].getNumComponents() == 1) {
                    broadcastWarpData << "shfl" << params[i].getName() << "=real_shfl(" << params[i].getName() <<"1,j);\n";

                } else {
                    broadcastWarpData << "shfl" << params[i].getName()+"."+suffixes[j] << "=real_shfl(" << params[i].getName()+"1."+suffixes[j] <<",j);\n";
                }
            }
        }
    } else {
        // not used if not shuffling
    }
    replacements["BROADCAST_WARP_DATA"] = broadcastWarpData.str();

    // Part 2. Defines for off-diagonal exclusions, and neighborlist tiles.
    stringstream declareLocal2;
    if(useShuffle) {
        for(int i=0; i< (int) params.size(); i++) {
            declareLocal2<<params[i].getType()<<" shfl"<<params[i].getName()<<";\n";
        }
    } else {
        // not used if using shared memory
    }
    replacements["DECLARE_LOCAL_PARAMETERS"] = declareLocal2.str();

    stringstream loadLocal2;
    if(useShuffle) {
        for(int i=0; i< (int) params.size(); i++) {
            loadLocal2<<"shfl"<<params[i].getName()<<" = global_"<<params[i].getName()<<"[j];\n";
        }
    } else {
        for (int i = 0; i < (int) params.size(); i++) {
            if (params[i].getNumComponents() == 1) {
                loadLocal2<<"localData[threadIdx.x]."<<params[i].getName()<<" = global_"<<params[i].getName()<<"[j];\n";
            }
            else {
                loadLocal2<<params[i].getType()<<" temp_"<<params[i].getName()<<" = global_"<<params[i].getName()<<"[j];\n";
                for (int j = 0; j < params[i].getNumComponents(); ++j)
                    loadLocal2<<"localData[threadIdx.x]."<<params[i].getName()<<"_"<<suffixes[j]<<" = temp_"<<params[i].getName()<<"."<<suffixes[j]<<";\n";
            }
        }
    }
    replacements["LOAD_LOCAL_PARAMETERS_FROM_GLOBAL"] = loadLocal2.str();

    stringstream load2j;
    if(useShuffle) {
        for(int i = 0; i < (int) params.size(); i++)
            load2j<<params[i].getType()<<" "<<params[i].getName()<<"2 = shfl"<<params[i].getName()<<";\n";
    } else {
        for (int i = 0; i < (int) params.size(); i++) {
            if (params[i].getNumComponents() == 1) {
                load2j<<params[i].getType()<<" "<<params[i].getName()<<"2 = localData[atom2]."<<params[i].getName()<<";\n";
            }
            else {
                load2j<<params[i].getType()<<" "<<params[i].getName()<<"2 = make_"<<params[i].getType()<<"(";
                for (int j = 0; j < params[i].getNumComponents(); ++j) {
                    if (j > 0)
                        load2j<<", ";
                    load2j<<"localData[atom2]."<<params[i].getName()<<"_"<<suffixes[j];
                }
                load2j<<");\n";
            }
        }
    }
    replacements["LOAD_ATOM2_PARAMETERS"] = load2j.str();
    stringstream initDerivs;
    for (int i = 0; i < energyParameterDerivatives.size(); i++)
        initDerivs<<"mixed energyParamDeriv"<<i<<" = 0;\n";
    replacements["INIT_DERIVATIVES"] = initDerivs.str();
    stringstream saveDerivs;
    const vector<string>& allParamDerivNames = context.getEnergyParamDerivNames();
    int numDerivs = allParamDerivNames.size();
    for (int i = 0; i < energyParameterDerivatives.size(); i++)
        for (int index = 0; index < numDerivs; index++)
            if (allParamDerivNames[index] == energyParameterDerivatives[i])
                saveDerivs<<"energyParamDerivs[(blockIdx.x*blockDim.x+threadIdx.x)*"<<numDerivs<<"+"<<index<<"] += energyParamDeriv"<<i<<";\n";
    replacements["SAVE_DERIVATIVES"] = saveDerivs.str();

    stringstream shuffleWarpData;
    if(useShuffle) {
        shuffleWarpData << "shflPosq.x = real_shfl(shflPosq.x, tgx+1);\n";
        shuffleWarpData << "shflPosq.y = real_shfl(shflPosq.y, tgx+1);\n";
        shuffleWarpData << "shflPosq.z = real_shfl(shflPosq.z, tgx+1);\n";
        shuffleWarpData << "shflPosq.w = real_shfl(shflPosq.w, tgx+1);\n";
        shuffleWarpData << "shflForce.x = real_shfl(shflForce.x, tgx+1);\n";
        shuffleWarpData << "shflForce.y = real_shfl(shflForce.y, tgx+1);\n";
        shuffleWarpData << "shflForce.z = real_shfl(shflForce.z, tgx+1);\n";
        for(int i=0; i < (int) params.size(); i++) {
            if(params[i].getNumComponents() == 1) {
                shuffleWarpData<<"shfl"<<params[i].getName()<<"=real_shfl(shfl"<<params[i].getName()<<", tgx+1);\n";
            } else {
                for(int j=0;j<params[i].getNumComponents();j++) {
                    // looks something like shflsigmaEpsilon.x = real_shfl(shflsigmaEpsilon.x,tgx+1);
                    shuffleWarpData<<"shfl"<<params[i].getName()
                        <<"."<<suffixes[j]<<"=real_shfl(shfl"
                        <<params[i].getName()<<"."<<suffixes[j]
                        <<", tgx+1);\n";
                }
            }
        }
    } else {
        // not used otherwise
    }
    replacements["SHUFFLE_WARP_DATA"] = shuffleWarpData.str();

    map<string, string> defines;
    if (useCutoff)
        defines["USE_CUTOFF"] = "1";
    if (usePeriodic)
        defines["USE_PERIODIC"] = "1";
    if (useExclusions)
        defines["USE_EXCLUSIONS"] = "1";
    if (isSymmetric)
        defines["USE_SYMMETRIC"] = "1";
    if (useShuffle)
        defines["ENABLE_SHUFFLE"] = "1";
    if (includeForces)
        defines["INCLUDE_FORCES"] = "1";
    if (includeEnergy)
        defines["INCLUDE_ENERGY"] = "1";
    defines["THREAD_BLOCK_SIZE"] = context.intToString(forceThreadBlockSize);
    double maxCutoff = 0.0;
    for (int i = 0; i < 32; i++) {
        if ((groups&(1<<i)) != 0) {
            double cutoff = groupCutoff[i];
            maxCutoff = max(maxCutoff, cutoff);
            defines["CUTOFF_"+context.intToString(i)+"_SQUARED"] = context.doubleToString(cutoff*cutoff);
            defines["CUTOFF_"+context.intToString(i)] = context.doubleToString(cutoff);
        }
    }
    defines["MAX_CUTOFF"] = context.doubleToString(maxCutoff);
    defines["NUM_ATOMS"] = context.intToString(context.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = context.intToString(context.getPaddedNumAtoms());
    defines["NUM_BLOCKS"] = context.intToString(context.getNumAtomBlocks());
    defines["TILE_SIZE"] = context.intToString(HipContext::TileSize);
    int numExclusionTiles = exclusionTiles.getSize();
    defines["NUM_TILES_WITH_EXCLUSIONS"] = context.intToString(numExclusionTiles);
    int numContexts = context.getPlatformData().contexts.size();
    int startExclusionIndex = context.getContextIndex()*numExclusionTiles/numContexts;
    int endExclusionIndex = (context.getContextIndex()+1)*numExclusionTiles/numContexts;
    defines["FIRST_EXCLUSION_TILE"] = context.intToString(startExclusionIndex);
    defines["LAST_EXCLUSION_TILE"] = context.intToString(endExclusionIndex);
    if ((localDataSize/4)%2 == 0 && !context.getUseDoublePrecision())
        defines["PARAMETER_SIZE_IS_EVEN"] = "1";
    hipModule_t program = context.createModule(HipKernelSources::vectorOps+context.replaceStrings(kernelSource, replacements), defines);
    hipFunction_t kernel = context.getKernel(program, "computeNonbonded");
    return kernel;
}

void HipNonbondedUtilities::setKernelSource(const string& source) {
    kernelSource = source;
}
