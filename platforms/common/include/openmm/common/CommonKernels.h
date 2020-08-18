#ifndef OPENMM_COMMONKERNELS_H_
#define OPENMM_COMMONKERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2020 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
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

#include "openmm/common/ComputeArray.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeParameterSet.h"
#include "openmm/Platform.h"
#include "openmm/kernels.h"
#include "openmm/internal/CompiledExpressionSet.h"
#include "openmm/internal/CustomIntegratorUtilities.h"
#include "lepton/CompiledExpression.h"

namespace OpenMM {


/**
 * This kernel is invoked by HarmonicBondForce to calculate the forces acting on the system and the energy of the system.
 */
class CommonCalcHarmonicBondForceKernel : public CalcHarmonicBondForceKernel {
public:
    CommonCalcHarmonicBondForceKernel(std::string name, const Platform& platform, ComputeContext& cc, const System& system) : CalcHarmonicBondForceKernel(name, platform),
            hasInitializedKernel(false), cc(cc), system(system) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the HarmonicBondForce this kernel will be used for
     */
    void initialize(const System& system, const HarmonicBondForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the HarmonicBondForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const HarmonicBondForce& force);
private:
    class ForceInfo;
    int numBonds;
    bool hasInitializedKernel;
    ComputeContext& cc;
    ForceInfo* info;
    const System& system;
    ComputeArray params;
};

/**
 * This kernel is invoked by CustomBondForce to calculate the forces acting on the system and the energy of the system.
 */
class CommonCalcCustomBondForceKernel : public CalcCustomBondForceKernel {
public:
    CommonCalcCustomBondForceKernel(std::string name, const Platform& platform, ComputeContext& cc, const System& system) : CalcCustomBondForceKernel(name, platform),
            hasInitializedKernel(false), cc(cc), system(system), params(NULL) {
    }
    ~CommonCalcCustomBondForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the CustomBondForce this kernel will be used for
     */
    void initialize(const System& system, const CustomBondForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the CustomBondForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const CustomBondForce& force);
private:
    class ForceInfo;
    int numBonds;
    bool hasInitializedKernel;
    ComputeContext& cc;
    ForceInfo* info;
    const System& system;
    ComputeParameterSet* params;
    ComputeArray globals;
    std::vector<std::string> globalParamNames;
    std::vector<float> globalParamValues;
};

/**
 * This kernel is invoked by HarmonicAngleForce to calculate the forces acting on the system and the energy of the system.
 */
class CommonCalcHarmonicAngleForceKernel : public CalcHarmonicAngleForceKernel {
public:
    CommonCalcHarmonicAngleForceKernel(std::string name, const Platform& platform, ComputeContext& cc, const System& system) : CalcHarmonicAngleForceKernel(name, platform),
            hasInitializedKernel(false), cc(cc), system(system) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the HarmonicAngleForce this kernel will be used for
     */
    void initialize(const System& system, const HarmonicAngleForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the HarmonicAngleForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const HarmonicAngleForce& force);
private:
    class ForceInfo;
    int numAngles;
    bool hasInitializedKernel;
    ComputeContext& cc;
    ForceInfo* info;
    const System& system;
    ComputeArray params;
};

/**
 * This kernel is invoked by CustomAngleForce to calculate the forces acting on the system and the energy of the system.
 */
class CommonCalcCustomAngleForceKernel : public CalcCustomAngleForceKernel {
public:
    CommonCalcCustomAngleForceKernel(std::string name, const Platform& platform, ComputeContext& cc, const System& system) : CalcCustomAngleForceKernel(name, platform),
            hasInitializedKernel(false), cc(cc), system(system), params(NULL) {
    }
    ~CommonCalcCustomAngleForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the CustomAngleForce this kernel will be used for
     */
    void initialize(const System& system, const CustomAngleForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the CustomAngleForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const CustomAngleForce& force);
private:
    class ForceInfo;
    int numAngles;
    bool hasInitializedKernel;
    ComputeContext& cc;
    ForceInfo* info;
    const System& system;
    ComputeParameterSet* params;
    ComputeArray globals;
    std::vector<std::string> globalParamNames;
    std::vector<float> globalParamValues;
};

/**
 * This kernel is invoked by PeriodicTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class CommonCalcPeriodicTorsionForceKernel : public CalcPeriodicTorsionForceKernel {
public:
    CommonCalcPeriodicTorsionForceKernel(std::string name, const Platform& platform, ComputeContext& cc, const System& system) : CalcPeriodicTorsionForceKernel(name, platform),
            hasInitializedKernel(false), cc(cc), system(system) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the PeriodicTorsionForce this kernel will be used for
     */
    void initialize(const System& system, const PeriodicTorsionForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the PeriodicTorsionForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const PeriodicTorsionForce& force);
private:
    class ForceInfo;
    int numTorsions;
    bool hasInitializedKernel;
    ComputeContext& cc;
    ForceInfo* info;
    const System& system;
    ComputeArray params;
};

/**
 * This kernel is invoked by RBTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class CommonCalcRBTorsionForceKernel : public CalcRBTorsionForceKernel {
public:
    CommonCalcRBTorsionForceKernel(std::string name, const Platform& platform, ComputeContext& cc, const System& system) : CalcRBTorsionForceKernel(name, platform),
            hasInitializedKernel(false), cc(cc), system(system) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the RBTorsionForce this kernel will be used for
     */
    void initialize(const System& system, const RBTorsionForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the RBTorsionForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const RBTorsionForce& force);
private:
    class ForceInfo;
    int numTorsions;
    bool hasInitializedKernel;
    ComputeContext& cc;
    ForceInfo* info;
    const System& system;
    ComputeArray params1;
    ComputeArray params2;
};

/**
 * This kernel is invoked by CustomTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class CommonCalcCustomTorsionForceKernel : public CalcCustomTorsionForceKernel {
public:
    CommonCalcCustomTorsionForceKernel(std::string name, const Platform& platform, ComputeContext& cc, const System& system) : CalcCustomTorsionForceKernel(name, platform),
            hasInitializedKernel(false), cc(cc), system(system), params(NULL) {
    }
    ~CommonCalcCustomTorsionForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the CustomTorsionForce this kernel will be used for
     */
    void initialize(const System& system, const CustomTorsionForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the CustomTorsionForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const CustomTorsionForce& force);
private:
    class ForceInfo;
    int numTorsions;
    bool hasInitializedKernel;
    ComputeContext& cc;
    ForceInfo* info;
    const System& system;
    ComputeParameterSet* params;
    ComputeArray globals;
    std::vector<std::string> globalParamNames;
    std::vector<float> globalParamValues;
};

/**
 * This kernel is invoked by CMAPTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class CommonCalcCMAPTorsionForceKernel : public CalcCMAPTorsionForceKernel {
public:
    CommonCalcCMAPTorsionForceKernel(std::string name, const Platform& platform, ComputeContext& cc, const System& system) : CalcCMAPTorsionForceKernel(name, platform),
            hasInitializedKernel(false), cc(cc), system(system) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the CMAPTorsionForce this kernel will be used for
     */
    void initialize(const System& system, const CMAPTorsionForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the CMAPTorsionForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const CMAPTorsionForce& force);
private:
    class ForceInfo;
    int numTorsions;
    bool hasInitializedKernel;
    ComputeContext& cc;
    ForceInfo* info;
    const System& system;
    std::vector<mm_int2> mapPositionsVec;
    ComputeArray coefficients;
    ComputeArray mapPositions;
    ComputeArray torsionMaps;
};

/**
 * This kernel is invoked by CustomExternalForce to calculate the forces acting on the system and the energy of the system.
 */
class CommonCalcCustomExternalForceKernel : public CalcCustomExternalForceKernel {
public:
    CommonCalcCustomExternalForceKernel(std::string name, const Platform& platform, ComputeContext& cc, const System& system) : CalcCustomExternalForceKernel(name, platform),
            hasInitializedKernel(false), cc(cc), system(system), params(NULL) {
    }
    ~CommonCalcCustomExternalForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the CustomExternalForce this kernel will be used for
     */
    void initialize(const System& system, const CustomExternalForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the CustomExternalForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const CustomExternalForce& force);
private:
    class ForceInfo;
    int numParticles;
    bool hasInitializedKernel;
    ComputeContext& cc;
    ForceInfo* info;
    const System& system;
    ComputeParameterSet* params;
    ComputeArray globals;
    std::vector<std::string> globalParamNames;
    std::vector<float> globalParamValues;
};

/**
 * This kernel is invoked by CustomCompoundBondForce to calculate the forces acting on the system.
 */
class CommonCalcCustomCompoundBondForceKernel : public CalcCustomCompoundBondForceKernel {
public:
    CommonCalcCustomCompoundBondForceKernel(std::string name, const Platform& platform, ComputeContext& cc, const System& system) : CalcCustomCompoundBondForceKernel(name, platform),
            cc(cc), params(NULL), system(system) {
    }
    ~CommonCalcCustomCompoundBondForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the CustomCompoundBondForce this kernel will be used for
     */
    void initialize(const System& system, const CustomCompoundBondForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the CustomCompoundBondForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const CustomCompoundBondForce& force);

private:
    class ForceInfo;
    int numBonds;
    ComputeContext& cc;
    ForceInfo* info;
    ComputeParameterSet* params;
    ComputeArray globals;
    std::vector<std::string> globalParamNames;
    std::vector<float> globalParamValues;
    std::vector<ComputeArray> tabulatedFunctions;
    const System& system;
};

/**
 * This kernel is invoked by CustomCentroidBondForce to calculate the forces acting on the system.
 */
class CommonCalcCustomCentroidBondForceKernel : public CalcCustomCentroidBondForceKernel {
public:
    CommonCalcCustomCentroidBondForceKernel(std::string name, const Platform& platform, ComputeContext& cc, const System& system) : CalcCustomCentroidBondForceKernel(name, platform),
            cc(cc), params(NULL), system(system), blockSize(64) {
    }
    ~CommonCalcCustomCentroidBondForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the CustomCentroidBondForce this kernel will be used for
     */
    void initialize(const System& system, const CustomCentroidBondForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the CustomCentroidBondForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const CustomCentroidBondForce& force);

private:
    class ForceInfo;
    int numGroups, numBonds;
    // note -- we assume a blocksize of 64 here, i.e., 2 warps on NV, and a single wavefront on AMD
    //         this seems to match the behaviour in the CUDA version.
    const int blockSize;
    bool needEnergyParamDerivs;
    ComputeContext& cc;
    ForceInfo* info;
    ComputeParameterSet* params;
    ComputeArray globals, groupParticles, groupWeights, groupOffsets;
    ComputeArray groupForces, bondGroups, centerPositions;
    std::vector<std::string> globalParamNames;
    std::vector<float> globalParamValues;
    std::vector<ComputeArray> tabulatedFunctions;
    std::vector<void*> groupForcesArgs;
    ComputeKernel computeCentersKernel, groupForcesKernel, applyForcesKernel;
    const System& system;
};

/**
 * This kernel is invoked by CustomNonbondedForce to calculate the forces acting on the system.
 */
class CommonCalcCustomNonbondedForceKernel : public CalcCustomNonbondedForceKernel {
public:
    CommonCalcCustomNonbondedForceKernel(std::string name, const Platform& platform, ComputeContext& cc, const System& system) : CalcCustomNonbondedForceKernel(name, platform),
            cc(cc), params(NULL), forceCopy(NULL), system(system), hasInitializedKernel(false) {
    }
    ~CommonCalcCustomNonbondedForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the CustomNonbondedForce this kernel will be used for
     */
    void initialize(const System& system, const CustomNonbondedForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the CustomNonbondedForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const CustomNonbondedForce& force);
private:
    class ForceInfo;
    void launchInitInteractionGroups(const CustomNonbondedForce& force, const std::string& interactionSource, const std::vector<std::string>& tableTypes);
    template<typename tileflags, typename mm_intv>
    void initInteractionGroups(const CustomNonbondedForce& force, const std::string& interactionSource, const std::vector<std::string>& tableTypes);
    ComputeContext& cc;
    ForceInfo* info;
    ComputeParameterSet* params;
    ComputeArray globals, interactionGroupData, filteredGroupData, numGroupTiles;
    ComputeKernel interactionGroupKernel, prepareNeighborListKernel, buildNeighborListKernel;
    std::vector<void*> interactionGroupArgs;
    std::vector<std::string> globalParamNames;
    std::vector<float> globalParamValues;
    std::vector<ComputeArray> tabulatedFunctions;
    double longRangeCoefficient;
    std::vector<double> longRangeCoefficientDerivs;
    bool hasInitializedLongRangeCorrection, hasInitializedKernel, hasParamDerivs, useNeighborList;
    int numGroupThreadBlocks;
    CustomNonbondedForce* forceCopy;
    const System& system;
};

/**
 * This kernel is invoked by GBSAOBCForce to calculate the forces acting on the system.
 */
class CommonCalcGBSAOBCForceKernel : public CalcGBSAOBCForceKernel {
public:
    CommonCalcGBSAOBCForceKernel(std::string name, const Platform& platform, ComputeContext& cc) : CalcGBSAOBCForceKernel(name, platform), cc(cc),
            hasCreatedKernels(false) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the GBSAOBCForce this kernel will be used for
     */
    void initialize(const System& system, const GBSAOBCForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the GBSAOBCForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const GBSAOBCForce& force);
private:
    class ForceInfo;
    double prefactor, surfaceAreaFactor, cutoff;
    bool hasCreatedKernels;
    int maxTiles;
    ComputeContext& cc;
    ForceInfo* info;
    ComputeArray params, charges, bornSum, bornRadii, bornForce, obcChain;
    ComputeKernel computeBornSumKernel, reduceBornSumKernel, force1Kernel, reduceBornForceKernel;
};

/**
 * This kernel is invoked by CustomGBForce to calculate the forces acting on the system.
 */
class CommonCalcCustomGBForceKernel : public CalcCustomGBForceKernel {
public:
    CommonCalcCustomGBForceKernel(std::string name, const Platform& platform, ComputeContext& cc, const System& system) : CalcCustomGBForceKernel(name, platform),
            hasInitializedKernels(false), cc(cc), params(NULL), computedValues(NULL), energyDerivs(NULL), energyDerivChain(NULL), system(system) {
    }
    ~CommonCalcCustomGBForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the CustomGBForce this kernel will be used for
     */
    void initialize(const System& system, const CustomGBForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the CustomGBForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const CustomGBForce& force);
private:
    class ForceInfo;
    double cutoff;
    bool hasInitializedKernels, needParameterGradient, needEnergyParamDerivs;
    int maxTiles, numComputedValues;
    ComputeContext& cc;
    ForceInfo* info;
    ComputeParameterSet* params;
    ComputeParameterSet* computedValues;
    ComputeParameterSet* energyDerivs;
    ComputeParameterSet* energyDerivChain;
    std::vector<ComputeParameterSet*> dValuedParam;
    std::vector<ComputeArray> dValue0dParam;
    ComputeArray longEnergyDerivs, globals, valueBuffers;
    std::vector<std::string> globalParamNames;
    std::vector<float> globalParamValues;
    std::vector<ComputeArray> tabulatedFunctions;
    std::vector<bool> pairValueUsesParam, pairEnergyUsesParam, pairEnergyUsesValue;
    const System& system;
    ComputeKernel pairValueKernel, perParticleValueKernel, pairEnergyKernel, perParticleEnergyKernel, gradientChainRuleKernel;
    std::string pairValueSrc, pairEnergySrc;
    std::map<std::string, std::string> pairValueDefines, pairEnergyDefines;
};

/**
 * This kernel is invoked by CustomHbondForce to calculate the forces acting on the system.
 */
class CommonCalcCustomHbondForceKernel : public CalcCustomHbondForceKernel {
public:
    CommonCalcCustomHbondForceKernel(std::string name, const Platform& platform, ComputeContext& cc, const System& system) : CalcCustomHbondForceKernel(name, platform),
            hasInitializedKernel(false), cc(cc), donorParams(NULL), acceptorParams(NULL), system(system) {
    }
    ~CommonCalcCustomHbondForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the CustomHbondForce this kernel will be used for
     */
    void initialize(const System& system, const CustomHbondForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the CustomHbondForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const CustomHbondForce& force);
private:
    class ForceInfo;
    int numDonors, numAcceptors;
    bool hasInitializedKernel;
    ComputeContext& cc;
    ForceInfo* info;
    ComputeParameterSet* donorParams;
    ComputeParameterSet* acceptorParams;
    ComputeArray globals;
    ComputeArray donors;
    ComputeArray acceptors;
    ComputeArray donorBufferIndices;
    ComputeArray acceptorBufferIndices;
    ComputeArray donorExclusions;
    ComputeArray acceptorExclusions;
    std::vector<std::string> globalParamNames;
    std::vector<float> globalParamValues;
    std::vector<ComputeArray> tabulatedFunctions;
    const System& system;
    ComputeKernel donorKernel, acceptorKernel;
};

/**
 * This kernel is invoked by CustomManyParticleForce to calculate the forces acting on the system.
 */
class CommonCalcCustomManyParticleForceKernel : public CalcCustomManyParticleForceKernel {
public:
    CommonCalcCustomManyParticleForceKernel(std::string name, const Platform& platform, ComputeContext& cc, const System& system) : CalcCustomManyParticleForceKernel(name, platform),
            hasInitializedKernel(false), cc(cc), params(NULL), system(system) {
    }
    ~CommonCalcCustomManyParticleForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the CustomManyParticleForce this kernel will be used for
     */
    void initialize(const System& system, const CustomManyParticleForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the CustomManyParticleForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const CustomManyParticleForce& force);

private:
    class ForceInfo;
    ComputeContext& cc;
    ForceInfo* info;
    bool hasInitializedKernel;
    NonbondedMethod nonbondedMethod;
    int maxNeighborPairs, forceWorkgroupSize, findNeighborsWorkgroupSize, computeNeighborsWorkgroupSize;
    ComputeParameterSet* params;
    ComputeArray globals, particleTypes,  orderIndex, particleOrder;
    ComputeArray exclusions, exclusionStartIndex, blockCenter, blockBoundingBox;
    ComputeArray neighborPairs, numNeighborPairs, neighborStartIndex, numNeighborsForAtom, neighbors;
    std::vector<std::string> globalParamNames;
    std::vector<float> globalParamValues;
    std::vector<ComputeArray> tabulatedFunctions;
    const System& system;
    ComputeKernel forceKernel, blockBoundsKernel, neighborsKernel, startIndicesKernel, copyPairsKernel;
    ComputeEvent event;
};

/**
 * This kernel is invoked by GayBerneForce to calculate the forces acting on the system.
 */
class CommonCalcGayBerneForceKernel : public CalcGayBerneForceKernel {
public:
    CommonCalcGayBerneForceKernel(std::string name, const Platform& platform, ComputeContext& cc) : CalcGayBerneForceKernel(name, platform), cc(cc),
            hasInitializedKernels(false) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the GayBerneForce this kernel will be used for
     */
    void initialize(const System& system, const GayBerneForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the GayBerneForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const GayBerneForce& force);
private:
    class ForceInfo;
    class ReorderListener;
    void sortAtoms();
    ComputeContext& cc;
    ForceInfo* info;
    bool hasInitializedKernels;
    int numRealParticles, maxNeighborBlocks;
    GayBerneForce::NonbondedMethod nonbondedMethod;
    ComputeArray sortedParticles, axisParticleIndices, sigParams, epsParams;
    ComputeArray scale, exceptionParticles, exceptionParams;
    ComputeArray aMatrix, bMatrix, gMatrix;
    ComputeArray exclusions, exclusionStartIndex, blockCenter, blockBoundingBox;
    ComputeArray neighbors, neighborIndex, neighborBlockCount;
    ComputeArray sortedPos, torque;
    std::vector<bool> isRealParticle;
    std::vector<std::pair<int, int> > exceptionAtoms;
    std::vector<std::pair<int, int> > excludedPairs;
    ComputeKernel framesKernel, blockBoundsKernel, neighborsKernel, forceKernel, torqueKernel;
    ComputeEvent event;
};

/**
 * This kernel is invoked by VerletIntegrator to take one time step.
 */
class CommonIntegrateVerletStepKernel : public IntegrateVerletStepKernel {
public:
    CommonIntegrateVerletStepKernel(std::string name, const Platform& platform, ComputeContext& cc) : IntegrateVerletStepKernel(name, platform), cc(cc),
            hasInitializedKernels(false) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the VerletIntegrator this kernel will be used for
     */
    void initialize(const System& system, const VerletIntegrator& integrator);
    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the VerletIntegrator this kernel is being used for
     */
    void execute(ContextImpl& context, const VerletIntegrator& integrator);
    /**
     * Compute the kinetic energy.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the VerletIntegrator this kernel is being used for
     */
    double computeKineticEnergy(ContextImpl& context, const VerletIntegrator& integrator);
private:
    ComputeContext& cc;
    bool hasInitializedKernels;
    ComputeKernel kernel1, kernel2;
};

/**
 * This kernel is invoked by LangevinIntegrator to take one time step.
 */
class CommonIntegrateLangevinStepKernel : public IntegrateLangevinStepKernel {
public:
    CommonIntegrateLangevinStepKernel(std::string name, const Platform& platform, ComputeContext& cc) : IntegrateLangevinStepKernel(name, platform), cc(cc),
            hasInitializedKernels(false) {
    }
    /**
     * Initialize the kernel, setting up the particle masses.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the LangevinIntegrator this kernel will be used for
     */
    void initialize(const System& system, const LangevinIntegrator& integrator);
    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the LangevinIntegrator this kernel is being used for
     */
    void execute(ContextImpl& context, const LangevinIntegrator& integrator);
    /**
     * Compute the kinetic energy.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the LangevinIntegrator this kernel is being used for
     */
    double computeKineticEnergy(ContextImpl& context, const LangevinIntegrator& integrator);
private:
    ComputeContext& cc;
    double prevTemp, prevFriction, prevStepSize;
    bool hasInitializedKernels;
    ComputeArray params;
    ComputeKernel kernel1, kernel2;
};

/**
 * This kernel is invoked by LangevinMiddleIntegrator to take one time step.
 */
class CommonIntegrateLangevinMiddleStepKernel : public IntegrateLangevinMiddleStepKernel {
public:
    CommonIntegrateLangevinMiddleStepKernel(std::string name, const Platform& platform, ComputeContext& cc) : IntegrateLangevinMiddleStepKernel(name, platform), cc(cc),
            hasInitializedKernels(false) {
    }
    /**
     * Initialize the kernel, setting up the particle masses.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the LangevinMiddleIntegrator this kernel will be used for
     */
    void initialize(const System& system, const LangevinMiddleIntegrator& integrator);
    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the LangevinMiddleIntegrator this kernel is being used for
     */
    void execute(ContextImpl& context, const LangevinMiddleIntegrator& integrator);
    /**
     * Compute the kinetic energy.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the LangevinMiddleIntegrator this kernel is being used for
     */
    double computeKineticEnergy(ContextImpl& context, const LangevinMiddleIntegrator& integrator);
private:
    ComputeContext& cc;
    double prevTemp, prevFriction, prevStepSize;
    bool hasInitializedKernels;
    ComputeArray params, oldDelta;
    ComputeKernel kernel1, kernel2, kernel3;
};

/*
 * This kernel is invoked by NoseHooverIntegrator to take one time step.
 */
class CommonIntegrateNoseHooverStepKernel : public IntegrateNoseHooverStepKernel {
public:
    CommonIntegrateNoseHooverStepKernel(std::string name, const Platform& platform, ComputeContext& cc) :
                                  IntegrateNoseHooverStepKernel(name, platform), cc(cc), hasInitializedKernels(false),
                                  hasInitializedKineticEnergyKernel(false), hasInitializedHeatBathEnergyKernel(false),
                                  hasInitializedScaleVelocitiesKernel(false), hasInitializedPropagateKernel(false) {}
    ~CommonIntegrateNoseHooverStepKernel() {}
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the NoseHooverIntegrator this kernel will be used for
     */
    void initialize(const System& system, const NoseHooverIntegrator& integrator);
    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the VerletIntegrator this kernel is being used for
     * @param forcesAreValid a reference to the parent integrator's boolean for keeping
     *                       track of the validity of the current forces.
     */
    void execute(ContextImpl& context, const NoseHooverIntegrator& integrator, bool &forcesAreValid);
    /**
     * Compute the kinetic energy.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the NoseHooverIntegrator this kernel is being used for
     */
    double computeKineticEnergy(ContextImpl& context, const NoseHooverIntegrator& integrator);
    /**
     * Execute the kernel that propagates the Nose Hoover chain and determines the velocity scale factor.
     * 
     * @param context  the context in which to execute this kernel
     * @param noseHooverChain the object describing the chain to be propagated.
     * @param kineticEnergy the {center of mass, relative} kineticEnergies of the particles being thermostated by this chain.
     * @param timeStep the time step used by the integrator.
     * @return the velocity scale factor to apply to the particles associated with this heat bath.
     */
    std::pair<double, double> propagateChain(ContextImpl& context, const NoseHooverChain &noseHooverChain, std::pair<double, double> kineticEnergy, double timeStep);
    /**
     * Execute the kernal that computes the total (kinetic + potential) heat bath energy.
     *
     * @param context the context in which to execute this kernel
     * @param noseHooverChain the chain whose energy is to be determined.
     * @return the total heat bath energy.
     */
    double computeHeatBathEnergy(ContextImpl& context, const NoseHooverChain &noseHooverChain);
    /**
     * Execute the kernel that computes the kinetic energy for a subset of atoms,
     * or the relative kinetic energy of Drude particles with respect to their parent atoms
     *
     * @param context the context in which to execute this kernel
     * @param noseHooverChain the chain whose energy is to be determined.
     * @param downloadValue whether the computed value should be downloaded and returned.
     */
    std::pair<double, double> computeMaskedKineticEnergy(ContextImpl& context, const NoseHooverChain &noseHooverChain, bool downloadValue);
    /**
     * Execute the kernel that scales the velocities of particles associated with a nose hoover chain
     *
     * @param context the context in which to execute this kernel
     * @param noseHooverChain the chain whose energy is to be determined.
     * @param scaleFactor the multiplicative factor by which {absolute, relative} velocities are scaled.
     */
    void scaleVelocities(ContextImpl& context, const NoseHooverChain &noseHooverChain, std::pair<double, double> scaleFactor);
    /**
     * Write the chain states to a checkpoint.
     */
    void createCheckpoint(ContextImpl& context, std::ostream& stream) const;
    /**
     * Load the chain states from a checkpoint.
     */
    void loadCheckpoint(ContextImpl& context, std::istream& stream);
    /**
     * Get the internal states of all chains.
     * 
     * @param context       the context for which to get the states
     * @param positions     element [i][j] contains the position of bead j for chain i
     * @param velocities    element [i][j] contains the velocity of bead j for chain i
     */
    void getChainStates(ContextImpl& context, std::vector<std::vector<double> >& positions, std::vector<std::vector<double> >& velocities) const;
    /**
     * Set the internal states of all chains.
     * 
     * @param context       the context for which to get the states
     * @param positions     element [i][j] contains the position of bead j for chain i
     * @param velocities    element [i][j] contains the velocity of bead j for chain i
     */
    void setChainStates(ContextImpl& context, const std::vector<std::vector<double> >& positions, const std::vector<std::vector<double> >& velocities);
private:
    ComputeContext& cc;
    float prevMaxPairDistance;
    ComputeArray maxPairDistanceBuffer, pairListBuffer, atomListBuffer, pairTemperatureBuffer, oldDelta;
    std::map<int, ComputeArray> chainState;
    ComputeKernel kernel1, kernel2, kernel3, kernel4, kernelHardWall;
    bool hasInitializedKernels;
    ComputeKernel reduceEnergyKernel;
    ComputeKernel computeHeatBathEnergyKernel;
    ComputeKernel computeAtomsKineticEnergyKernel;
    ComputeKernel computePairsKineticEnergyKernel;
    ComputeKernel scaleAtomsVelocitiesKernel;
    ComputeKernel scalePairsVelocitiesKernel;
    ComputeArray energyBuffer, scaleFactorBuffer, kineticEnergyBuffer, chainMasses, chainForces, heatBathEnergy;
    std::map<int, ComputeArray> atomlists, pairlists;
    std::map<int, ComputeKernel> propagateKernels;
    bool hasInitializedPropagateKernel;
    bool hasInitializedKineticEnergyKernel;
    bool hasInitializedHeatBathEnergyKernel;
    bool hasInitializedScaleVelocitiesKernel;
};

/**
 * This kernel is invoked by BrownianIntegrator to take one time step.
 */
class CommonIntegrateBrownianStepKernel : public IntegrateBrownianStepKernel {
public:
    CommonIntegrateBrownianStepKernel(std::string name, const Platform& platform, ComputeContext& cc) : IntegrateBrownianStepKernel(name, platform), cc(cc),
            hasInitializedKernels(false), prevTemp(-1), prevFriction(-1), prevStepSize(-1) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the BrownianIntegrator this kernel will be used for
     */
    void initialize(const System& system, const BrownianIntegrator& integrator);
    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the BrownianIntegrator this kernel is being used for
     */
    void execute(ContextImpl& context, const BrownianIntegrator& integrator);
    /**
     * Compute the kinetic energy.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the BrownianIntegrator this kernel is being used for
     */
    double computeKineticEnergy(ContextImpl& context, const BrownianIntegrator& integrator);
private:
    ComputeContext& cc;
    double prevTemp, prevFriction, prevStepSize;
    bool hasInitializedKernels;
    ComputeKernel kernel1, kernel2;
};

/**
 * This kernel is invoked by VerletIntegrator to take one time step.
 */
class CommonIntegrateVariableVerletStepKernel : public IntegrateVariableVerletStepKernel {
public:
    CommonIntegrateVariableVerletStepKernel(std::string name, const Platform& platform, ComputeContext& cc) : IntegrateVariableVerletStepKernel(name, platform), cc(cc),
            hasInitializedKernels(false) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the VariableVerletIntegrator this kernel will be used for
     */
    void initialize(const System& system, const VariableVerletIntegrator& integrator);
    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the VariableVerletIntegrator this kernel is being used for
     * @param maxTime    the maximum time beyond which the simulation should not be advanced
     * @return the size of the step that was taken
     */
    double execute(ContextImpl& context, const VariableVerletIntegrator& integrator, double maxTime);
    /**
     * Compute the kinetic energy.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the VariableVerletIntegrator this kernel is being used for
     */
    double computeKineticEnergy(ContextImpl& context, const VariableVerletIntegrator& integrator);
private:
    ComputeContext& cc;
    bool hasInitializedKernels;
    int blockSize;
    ComputeKernel kernel1, kernel2, selectSizeKernel;
};

/**
 * This kernel is invoked by VariableLangevinIntegrator to take one time step.
 */
class CommonIntegrateVariableLangevinStepKernel : public IntegrateVariableLangevinStepKernel {
public:
    CommonIntegrateVariableLangevinStepKernel(std::string name, const Platform& platform, ComputeContext& cc) : IntegrateVariableLangevinStepKernel(name, platform), cc(cc),
            hasInitializedKernels(false) {
    }
    /**
     * Initialize the kernel, setting up the particle masses.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the VariableLangevinIntegrator this kernel will be used for
     */
    void initialize(const System& system, const VariableLangevinIntegrator& integrator);
    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the VariableLangevinIntegrator this kernel is being used for
     * @param maxTime    the maximum time beyond which the simulation should not be advanced
     * @return the size of the step that was taken
     */
    double execute(ContextImpl& context, const VariableLangevinIntegrator& integrator, double maxTime);
    /**
     * Compute the kinetic energy.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the VariableLangevinIntegrator this kernel is being used for
     */
    double computeKineticEnergy(ContextImpl& context, const VariableLangevinIntegrator& integrator);
private:
    ComputeContext& cc;
    bool hasInitializedKernels;
    int blockSize;
    ComputeArray params;
    ComputeKernel kernel1, kernel2, selectSizeKernel;
    double prevTemp, prevFriction, prevErrorTol;
};

/**
 * This kernel is invoked by CustomIntegrator to take one time step.
 */
class CommonIntegrateCustomStepKernel : public IntegrateCustomStepKernel {
public:
    enum GlobalTargetType {DT, VARIABLE, PARAMETER};
    CommonIntegrateCustomStepKernel(std::string name, const Platform& platform, ComputeContext& cc) : IntegrateCustomStepKernel(name, platform), cc(cc),
            hasInitializedKernels(false), needsEnergyParamDerivs(false) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the CustomIntegrator this kernel will be used for
     */
    void initialize(const System& system, const CustomIntegrator& integrator);
    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the CustomIntegrator this kernel is being used for
     * @param forcesAreValid if the context has been modified since the last time step, this will be
     *                       false to show that cached forces are invalid and must be recalculated.
     *                       On exit, this should specify whether the cached forces are valid at the
     *                       end of the step.
     */
    void execute(ContextImpl& context, CustomIntegrator& integrator, bool& forcesAreValid);
    /**
     * Compute the kinetic energy.
     *
     * @param context    the context in which to execute this kernel
     * @param integrator the CustomIntegrator this kernel is being used for
     * @param forcesAreValid if the context has been modified since the last time step, this will be
     *                       false to show that cached forces are invalid and must be recalculated.
     *                       On exit, this should specify whether the cached forces are valid at the
     *                       end of the step.
     */
    double computeKineticEnergy(ContextImpl& context, CustomIntegrator& integrator, bool& forcesAreValid);
    /**
     * Get the values of all global variables.
     *
     * @param context   the context in which to execute this kernel
     * @param values    on exit, this contains the values
     */
    void getGlobalVariables(ContextImpl& context, std::vector<double>& values) const;
    /**
     * Set the values of all global variables.
     *
     * @param context   the context in which to execute this kernel
     * @param values    a vector containing the values
     */
    void setGlobalVariables(ContextImpl& context, const std::vector<double>& values);
    /**
     * Get the values of a per-DOF variable.
     *
     * @param context   the context in which to execute this kernel
     * @param variable  the index of the variable to get
     * @param values    on exit, this contains the values
     */
    void getPerDofVariable(ContextImpl& context, int variable, std::vector<Vec3>& values) const;
    /**
     * Set the values of a per-DOF variable.
     *
     * @param context   the context in which to execute this kernel
     * @param variable  the index of the variable to get
     * @param values    a vector containing the values
     */
    void setPerDofVariable(ContextImpl& context, int variable, const std::vector<Vec3>& values);
private:
    class ReorderListener;
    class GlobalTarget;
    class DerivFunction;
    std::string createPerDofComputation(const std::string& variable, const Lepton::ParsedExpression& expr, CustomIntegrator& integrator,
        const std::string& forceName, const std::string& energyName, std::vector<const TabulatedFunction*>& functions,
        std::vector<std::pair<std::string, std::string> >& functionNames);
    void prepareForComputation(ContextImpl& context, CustomIntegrator& integrator, bool& forcesAreValid);
    Lepton::ExpressionTreeNode replaceDerivFunctions(const Lepton::ExpressionTreeNode& node, OpenMM::ContextImpl& context);
    void findExpressionsForDerivs(const Lepton::ExpressionTreeNode& node, std::vector<std::pair<Lepton::ExpressionTreeNode, std::string> >& variableNodes);
    void recordGlobalValue(double value, GlobalTarget target, CustomIntegrator& integrator);
    void recordChangedParameters(ContextImpl& context);
    bool evaluateCondition(int step);
    ComputeContext& cc;
    double energy;
    float energyFloat;
    int numGlobalVariables, sumWorkGroupSize;
    bool hasInitializedKernels, deviceGlobalsAreCurrent, modifiesParameters, hasAnyConstraints, needsEnergyParamDerivs;
    std::vector<bool> deviceValuesAreCurrent;
    mutable std::vector<bool> localValuesAreCurrent;
    ComputeArray globalValues, sumBuffer, summedValue;
    ComputeArray uniformRandoms, randomSeed, perDofEnergyParamDerivs;
    std::vector<ComputeArray> tabulatedFunctions, perDofValues;
    std::map<int, double> savedEnergy;
    std::map<int, ComputeArray> savedForces;
    std::set<int> validSavedForces;
    mutable std::vector<std::vector<mm_float4> > localPerDofValuesFloat;
    mutable std::vector<std::vector<mm_double4> > localPerDofValuesDouble;
    std::map<std::string, double> energyParamDerivs;
    std::vector<std::string> perDofEnergyParamDerivNames;
    std::vector<double> localPerDofEnergyParamDerivs;
    std::vector<double> localGlobalValues;
    std::vector<double> initialGlobalVariables;
    std::vector<std::vector<ComputeKernel> > kernels;
    ComputeKernel randomKernel, kineticEnergyKernel, sumKineticEnergyKernel;
    std::vector<CustomIntegrator::ComputationType> stepType;
    std::vector<CustomIntegratorUtilities::Comparison> comparisons;
    std::vector<std::vector<Lepton::CompiledExpression> > globalExpressions;
    CompiledExpressionSet expressionSet;
    std::vector<bool> needsGlobals, needsForces, needsEnergy;
    std::vector<bool> computeBothForceAndEnergy, invalidatesForces, merged;
    std::vector<int> forceGroupFlags, blockEnd, requiredGaussian, requiredUniform;
    std::vector<int> stepEnergyVariableIndex, globalVariableIndex, parameterVariableIndex;
    int gaussianVariableIndex, uniformVariableIndex, dtVariableIndex;
    std::vector<std::string> parameterNames;
    std::vector<GlobalTarget> stepTarget;
};

class CommonIntegrateCustomStepKernel::GlobalTarget {
public:
    CommonIntegrateCustomStepKernel::GlobalTargetType type;
    int variableIndex;
    GlobalTarget() {
    }
    GlobalTarget(CommonIntegrateCustomStepKernel::GlobalTargetType type, int variableIndex) : type(type), variableIndex(variableIndex) {
    }
};

/**
 * This kernel is invoked to remove center of mass motion from the system.
 */
class CommonRemoveCMMotionKernel : public RemoveCMMotionKernel {
public:
    CommonRemoveCMMotionKernel(std::string name, const Platform& platform, ComputeContext& cc) : RemoveCMMotionKernel(name, platform), cc(cc) {
    }
    /**
     * Initialize the kernel, setting up the particle masses.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the CMMotionRemover this kernel will be used for
     */
    void initialize(const System& system, const CMMotionRemover& force);
    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     */
    void execute(ContextImpl& context);
private:
    ComputeContext& cc;
    int frequency;
    ComputeArray cmMomentum;
    ComputeKernel kernel1, kernel2;
};

/**
 * This kernel is invoked by RMSDForce to calculate the forces acting on the system and the energy of the system.
 */
class CommonCalcRMSDForceKernel : public CalcRMSDForceKernel {
public:
    CommonCalcRMSDForceKernel(std::string name, const Platform& platform, ComputeContext& cc) : CalcRMSDForceKernel(name, platform), cc(cc) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the RMSDForce this kernel will be used for
     */
    void initialize(const System& system, const RMSDForce& force);
    /**
     * Record the reference positions and particle indices.
     */
    void recordParameters(const RMSDForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * This is the internal implementation of execute(), templatized on whether we're
     * using single or double precision.
     */
    template <class REAL>
    double executeImpl(ContextImpl& context);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the RMSDForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const RMSDForce& force);
private:
    class ForceInfo;
    ComputeContext& cc;
    ForceInfo* info;
    int blockSize;
    double sumNormRef;
    ComputeArray referencePos, particles, buffer;
    ComputeKernel kernel1, kernel2;
};

/**
 * This kernel is invoked by AndersenThermostat at the start of each time step to adjust the particle velocities.
 */
class CommonApplyAndersenThermostatKernel : public ApplyAndersenThermostatKernel {
public:
    CommonApplyAndersenThermostatKernel(std::string name, const Platform& platform, ComputeContext& cc) : ApplyAndersenThermostatKernel(name, platform), cc(cc) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param thermostat the AndersenThermostat this kernel will be used for
     */
    void initialize(const System& system, const AndersenThermostat& thermostat);
    /**
     * Execute the kernel.
     *
     * @param context    the context in which to execute this kernel
     */
    void execute(ContextImpl& context);
private:
    ComputeContext& cc;
    int randomSeed;
    ComputeArray atomGroups;
    ComputeKernel kernel;
};

} // namespace OpenMM

#endif /*OPENMM_COMMONKERNELS_H_*/
