#ifndef OPENMM_HIPNONBONDEDUTILITIES_H_
#define OPENMM_HIPNONBONDEDUTILITIES_H_

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

#include "openmm/System.h"
#include "HipArray.h"
#include "HipExpressionUtilities.h"
#include "openmm/common/NonbondedUtilities.h"
#include <hip/hip_runtime.h>
#include <sstream>
#include <string>
#include <vector>

namespace OpenMM {

class HipContext;
class HipSort;

/**
 * This class provides a generic interface for calculating nonbonded interactions.  It does this in two
 * ways.  First, it can be used to create kernels that evaluate nonbonded interactions.  Clients
 * only need to provide the code for evaluating a single interaction and the list of parameters it depends on.
 * A complete kernel is then synthesized using an appropriate algorithm to evaluate all interactions on all
 * atoms.
 *
 * Second, this class itself creates and invokes a single "default" interaction kernel, allowing several
 * different forces to be evaluated at once for greater efficiency.  Call addInteraction() and addParameter()
 * to add interactions to this default kernel.
 *
 * During each force or energy evaluation, the following sequence of steps takes place:
 *
 * 1. Data structures (e.g. neighbor lists) are calculated to allow nonbonded interactions to be evaluated
 * quickly.
 *
 * 2. calcForcesAndEnergy() is called on each ForceImpl in the System.
 *
 * 3. Finally, the default interaction kernel is invoked to calculate all interactions that were added
 * to it.
 *
 * This sequence means that the default interaction kernel may depend on quantities that were calculated
 * by ForceImpls during calcForcesAndEnergy().
 */

class OPENMM_EXPORT_COMMON HipNonbondedUtilities : public NonbondedUtilities  {
public:
    class ParameterInfo;
    HipNonbondedUtilities(HipContext& context);
    ~HipNonbondedUtilities();
    /**
     * Add a nonbonded interaction to be evaluated by the default interaction kernel.
     *
     * @param usesCutoff       specifies whether a cutoff should be applied to this interaction
     * @param usesPeriodic     specifies whether periodic boundary conditions should be applied to this interaction
     * @param usesExclusions   specifies whether this interaction uses exclusions.  If this is true, it must have identical exclusions to every other interaction.
     * @param cutoffDistance   the cutoff distance for this interaction (ignored if usesCutoff is false)
     * @param exclusionList    for each atom, specifies the list of other atoms whose interactions should be excluded
     * @param kernel           the code to evaluate the interaction
     * @param forceGroup       the force group in which the interaction should be calculated
     */
    void addInteraction(bool usesCutoff, bool usesPeriodic, bool usesExclusions, double cutoffDistance, const std::vector<std::vector<int> >& exclusionList, const std::string& kernel, int forceGroup);
    /**
     * Add a nonbonded interaction to be evaluated by the default interaction kernel.
     *
     * @param usesCutoff       specifies whether a cutoff should be applied to this interaction
     * @param usesPeriodic     specifies whether periodic boundary conditions should be applied to this interaction
     * @param usesExclusions   specifies whether this interaction uses exclusions.  If this is true, it must have identical exclusions to every other interaction.
     * @param cutoffDistance   the cutoff distance for this interaction (ignored if usesCutoff is false)
     * @param exclusionList    for each atom, specifies the list of other atoms whose interactions should be excluded
     * @param kernel           the code to evaluate the interaction
     * @param forceGroup       the force group in which the interaction should be calculated
     * @param supportsPairList specifies whether this interaction can work with a neighbor list that uses a separate pair list
     */
    void addInteraction(bool usesCutoff, bool usesPeriodic, bool usesExclusions, double cutoffDistance, const std::vector<std::vector<int> >& exclusionList, const std::string& kernel, int forceGroup, bool supportsPairList);
    /**
     * Add a per-atom parameter that the default interaction kernel may depend on.
     */
    void addParameter(ComputeParameterInfo parameter);
    /**
     * Add a per-atom parameter that the default interaction kernel may depend on.
     *
     * @deprecated Use the version that takes a ComputeParameterInfo instead.
     */
    void addParameter(const ParameterInfo& parameter);
    /**
     * Add an array (other than a per-atom parameter) that should be passed as an argument to the default interaction kernel.
     */
    void addArgument(ComputeParameterInfo parameter);
    /**
     * Add an array (other than a per-atom parameter) that should be passed as an argument to the default interaction kernel.
     *
     * @deprecated Use the version that takes a ComputeParameterInfo instead.
     */
    void addArgument(const ParameterInfo& parameter);
    /**
     * Register that the interaction kernel will be computing the derivative of the potential energy
     * with respect to a parameter.
     *
     * @param param   the name of the parameter
     * @return the variable that will be used to accumulate the derivative.  Any code you pass to addInteraction() should
     * add its contributions to this variable.
     */
    std::string addEnergyParameterDerivative(const std::string& param);
    /**
     * Specify the list of exclusions that an interaction outside the default kernel will depend on.
     *
     * @param exclusionList  for each atom, specifies the list of other atoms whose interactions should be excluded
     */
    void requestExclusions(const std::vector<std::vector<int> >& exclusionList);
    /**
     * Initialize this object in preparation for a simulation.
     */
    void initialize(const System& system);
    /**
     * Get the number of force buffers required for nonbonded forces.
     */
    int getNumForceBuffers() const {
        return 0;
    }
    /**
     * Get the number of energy buffers required for nonbonded forces.
     */
    int getNumEnergyBuffers() {
        return numForceThreadBlocks*forceThreadBlockSize;
    }
    /**
     * Get whether a cutoff is being used.
     */
    bool getUseCutoff() {
        return useCutoff;
    }
    /**
     * Get whether periodic boundary conditions are being used.
     */
    bool getUsePeriodic() {
        return usePeriodic;
    }
    /**
     * Get the number of work groups used for computing nonbonded forces.
     */
    int getNumForceThreadBlocks() {
        return numForceThreadBlocks;
    }
    /**
     * Get the size of each work group used for computing nonbonded forces.
     */
    int getForceThreadBlockSize() {
        return forceThreadBlockSize;
    }
    /**
     * Get the maximum cutoff distance used by any force group.
     */
    double getMaxCutoffDistance();
    /**
     * Given a nonbonded cutoff, get the padded cutoff distance used in computing
     * the neighbor list.
     */
    double padCutoff(double cutoff);
    /**
     * Prepare to compute interactions.  This updates the neighbor list.
     */
    void prepareInteractions(int forceGroups);
    /**
     * Compute the nonbonded interactions.
     *
     * @param forceGroups    the flags specifying which force groups to include
     * @param includeForces  whether to compute forces
     * @param includeEnergy  whether to compute the potential energy
     */
    void computeInteractions(int forceGroups, bool includeForces, bool includeEnergy);
    /**
     * Check to see if the neighbor list arrays are large enough, and make them bigger if necessary.
     *
     * @return true if the neighbor list needed to be enlarged.
     */
    bool updateNeighborListSize();
    /**
     * Get the array containing the center of each atom block.
     */
    HipArray& getBlockCenters() {
        return blockCenter;
    }
    /**
     * Get the array containing the dimensions of each atom block.
     */
    HipArray& getBlockBoundingBoxes() {
        return blockBoundingBox;
    }
    /**
     * Get the array whose first element contains the number of tiles with interactions.
     */
    HipArray& getInteractionCount() {
        return interactionCount;
    }
    /**
     * Get the array containing tiles with interactions.
     */
    HipArray& getInteractingTiles() {
        return interactingTiles;
    }
    /**
     * Get the array containing the atoms in each tile with interactions.
     */
    HipArray& getInteractingAtoms() {
        return interactingAtoms;
    }
    /**
     * Get the array containing single pairs in the neighbor list.
     */
    HipArray& getSinglePairs() {
        return singlePairs;
    }
    /**
     * Get the array containing exclusion flags.
     */
    HipArray& getExclusions() {
        return exclusions;
    }
    /**
     * Get the array containing tiles with exclusions.
     */
    HipArray& getExclusionTiles() {
        return exclusionTiles;
    }
    /**
     * Get the array containing the index into the exclusion array for each tile.
     */
    HipArray& getExclusionIndices() {
        return exclusionIndices;
    }
    /**
     * Get the array listing where the exclusion data starts for each row.
     */
    HipArray& getExclusionRowIndices() {
        return exclusionRowIndices;
    }
    /**
     * Get the array containing a flag for whether the neighbor list was rebuilt
     * on the most recent call to prepareInteractions().
     */
    HipArray& getRebuildNeighborList() {
        return rebuildNeighborList;
    }
    /**
     * Get the index of the first tile this context is responsible for processing.
     */
    int getStartTileIndex() const {
        return startTileIndex;
    }
    /**
     * Get the total number of tiles this context is responsible for processing.
     */
    int getNumTiles() const {
        return numTiles;
    }
    /**
     * Set whether to add padding to the cutoff distance when building the neighbor list.
     * This increases the size of the neighbor list (and thus the cost of computing interactions),
     * but also means we don't need to rebuild it every time step.  The default value is true,
     * since usually this improves performance.  For very expensive interactions, however,
     * it may be better to set this to false.
     */
    void setUsePadding(bool padding);
    /**
     * Set the range of atom blocks and tiles that should be processed by this context.
     */
    void setAtomBlockRange(double startFraction, double endFraction);
    /**
     * Create a Kernel for evaluating a nonbonded interaction.  Cutoffs and periodic boundary conditions
     * are assumed to be the same as those for the default interaction Kernel, since this kernel will use
     * the same neighbor list.
     *
     * @param source        the source code for evaluating the force and energy
     * @param params        the per-atom parameters this kernel may depend on
     * @param arguments     arrays (other than per-atom parameters) that should be passed as arguments to the kernel
     * @param useExclusions specifies whether exclusions are applied to this interaction
     * @param isSymmetric   specifies whether the interaction is symmetric
     * @param groups        the set of force groups this kernel is for
     * @param includeForces whether this kernel should compute forces
     * @param includeEnergy whether this kernel should compute potential energy
     */
    hipFunction_t createInteractionKernel(const std::string& source, std::vector<ParameterInfo>& params, std::vector<ParameterInfo>& arguments, bool useExclusions, bool isSymmetric, int groups, bool includeForces, bool includeEnergy);
    /**
     * Create the set of kernels that will be needed for a particular combination of force groups.
     *
     * @param groups    the set of force groups
     */
    void createKernelsForGroups(int groups);
    /**
     * Set the source code for the main kernel.  This defaults to the content of nonbonded.hip.  It only needs to be
     * changed in very unusual circumstances.
     */
    void setKernelSource(const std::string& source);
private:
    class KernelSet;
    class BlockSortTrait;
    HipContext& context;
    std::map<int, KernelSet> groupKernels;
    HipArray exclusionTiles;
    HipArray exclusions;
    HipArray exclusionIndices;
    HipArray exclusionRowIndices;
    HipArray interactingTiles;
    HipArray interactingAtoms;
    HipArray interactionCount;
    HipArray singlePairs;
    HipArray singlePairCount;
    HipArray blockCenter;
    HipArray blockBoundingBox;
    HipArray sortedBlocks;
    HipArray sortedBlockCenter;
    HipArray sortedBlockBoundingBox;
    HipArray oldPositions;
    HipArray rebuildNeighborList;
    HipSort* blockSorter;
    hipEvent_t downloadCountEvent;
    int* pinnedCountBuffer;
    std::vector<void*> forceArgs, findBlockBoundsArgs, sortBoxDataArgs, findInteractingBlocksArgs;
    std::vector<std::vector<int> > atomExclusions;
    std::vector<ParameterInfo> parameters;
    std::vector<ParameterInfo> arguments;
    std::vector<std::string> energyParameterDerivatives;
    std::map<int, double> groupCutoff;
    std::map<int, std::string> groupKernelSource;
    double lastCutoff;
    bool useCutoff, usePeriodic, anyExclusions, usePadding, forceRebuildNeighborList, canUsePairList;
    int startTileIndex, numTiles, startBlockIndex, numBlocks, maxTiles, maxSinglePairs, maxExclusions, numForceThreadBlocks, forceThreadBlockSize, numAtoms, groupFlags;
    std::string kernelSource;
};

/**
 * This class stores the kernels to execute for a set of force groups.
 */

class HipNonbondedUtilities::KernelSet {
public:
    bool hasForces;
    double cutoffDistance;
    std::string source;
    hipFunction_t forceKernel, energyKernel, forceEnergyKernel;
    hipFunction_t findBlockBoundsKernel;
    hipFunction_t sortBoxDataKernel;
    hipFunction_t findInteractingBlocksKernel;
    hipFunction_t findInteractionsWithinBlocksKernel;
};

/**
 * This class stores information about a per-atom parameter that may be used in a nonbonded kernel.
 */

class HipNonbondedUtilities::ParameterInfo {
public:
    /**
     * Create a ParameterInfo object.
     *
     * @param name           the name of the parameter
     * @param type           the data type of the parameter's components
     * @param numComponents  the number of components in the parameter
     * @param size           the size of the parameter in bytes
     * @param memory         the memory containing the parameter values
     * @param constant       whether the memory should be marked as constant
     */
    ParameterInfo(const std::string& name, const std::string& componentType, int numComponents, int size, hipDeviceptr_t memory, bool constant=true) :
            name(name), componentType(componentType), numComponents(numComponents), size(size), memory(memory), constant(constant) {
        if (numComponents == 1)
            type = componentType;
        else {
            std::stringstream s;
            s << componentType << numComponents;
            type = s.str();
        }
    }
    const std::string& getName() const {
        return name;
    }
    const std::string& getComponentType() const {
        return componentType;
    }
    const std::string& getType() const {
        return type;
    }
    int getNumComponents() const {
        return numComponents;
    }
    int getSize() const {
        return size;
    }
    hipDeviceptr_t& getMemory() {
        return memory;
    }
    bool isConstant() const {
        return constant;
    }
private:
    std::string name;
    std::string componentType;
    std::string type;
    int size, numComponents;
    hipDeviceptr_t memory;
    bool constant;
};

} // namespace OpenMM

#endif /*OPENMM_HIPNONBONDEDUTILITIES_H_*/
