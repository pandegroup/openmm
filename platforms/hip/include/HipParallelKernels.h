#ifndef OPENMM_HIPPARALLELKERNELS_H_
#define OPENMM_HIPPARALLELKERNELS_H_

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

#include "HipPlatform.h"
#include "HipContext.h"
#include "HipKernels.h"
#include "openmm/common/CommonKernels.h"

namespace OpenMM {

/**
 * This kernel is invoked at the beginning and end of force and energy computations.  It gives the
 * Platform a chance to clear buffers and do other initialization at the beginning, and to do any
 * necessary work at the end to determine the final results.
 */
class HipParallelCalcForcesAndEnergyKernel : public CalcForcesAndEnergyKernel {
public:
    HipParallelCalcForcesAndEnergyKernel(std::string name, const Platform& platform, HipPlatform::PlatformData& data);
    ~HipParallelCalcForcesAndEnergyKernel();
    HipCalcForcesAndEnergyKernel& getKernel(int index) {
        return dynamic_cast<HipCalcForcesAndEnergyKernel&>(kernels[index].getImpl());
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     */
    void initialize(const System& system);
    /**
     * This is called at the beginning of each force/energy computation, before calcForcesAndEnergy() has been called on
     * any ForceImpl.
     *
     * @param context       the context in which to execute this kernel
     * @param includeForce  true if forces should be computed
     * @param includeEnergy true if potential energy should be computed
     * @param groups        a set of bit flags for which force groups to include
     */
    void beginComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups);
    /**
     * This is called at the end of each force/energy computation, after calcForcesAndEnergy() has been called on
     * every ForceImpl.
     *
     * @param context       the context in which to execute this kernel
     * @param includeForce  true if forces should be computed
     * @param includeEnergy true if potential energy should be computed
     * @param groups        a set of bit flags for which force groups to include
     * @param valid         the method may set this to false to indicate the results are invalid and the force/energy
     *                      calculation should be repeated
     * @return the potential energy of the system.  This value is added to all values returned by ForceImpls'
     * calcForcesAndEnergy() methods.  That is, each force kernel may <i>either</i> return its contribution to the
     * energy directly, <i>or</i> add it to an internal buffer so that it will be included here.
     */
    double finishComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups, bool& valid);
private:
    class BeginComputationTask;
    class FinishComputationTask;
    HipPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
    std::vector<long long> completionTimes;
    std::vector<double> contextNonbondedFractions;
    int2* interactionCounts;
    HipArray contextForces;
    void* pinnedPositionBuffer;
    long long* pinnedForceBuffer;
    hipFunction_t sumKernel;
    hipEvent_t event;
    hipStream_t peerCopyStream;
};

/**
 * This kernel is invoked by HarmonicBondForce to calculate the forces acting on the system and the energy of the system.
 */
class HipParallelCalcHarmonicBondForceKernel : public CalcHarmonicBondForceKernel {
public:
    HipParallelCalcHarmonicBondForceKernel(std::string name, const Platform& platform, HipPlatform::PlatformData& data, const System& system);
    CommonCalcHarmonicBondForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcHarmonicBondForceKernel&>(kernels[index].getImpl());
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
    class Task;
    HipPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

/**
 * This kernel is invoked by CustomBondForce to calculate the forces acting on the system and the energy of the system.
 */
class HipParallelCalcCustomBondForceKernel : public CalcCustomBondForceKernel {
public:
    HipParallelCalcCustomBondForceKernel(std::string name, const Platform& platform, HipPlatform::PlatformData& data, const System& system);
    CommonCalcCustomBondForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcCustomBondForceKernel&>(kernels[index].getImpl());
    }
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
    class Task;
    HipPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

/**
 * This kernel is invoked by HarmonicAngleForce to calculate the forces acting on the system and the energy of the system.
 */
class HipParallelCalcHarmonicAngleForceKernel : public CalcHarmonicAngleForceKernel {
public:
    HipParallelCalcHarmonicAngleForceKernel(std::string name, const Platform& platform, HipPlatform::PlatformData& data, const System& system);
    CommonCalcHarmonicAngleForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcHarmonicAngleForceKernel&>(kernels[index].getImpl());
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
    class Task;
    HipPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

/**
 * This kernel is invoked by CustomAngleForce to calculate the forces acting on the system and the energy of the system.
 */
class HipParallelCalcCustomAngleForceKernel : public CalcCustomAngleForceKernel {
public:
    HipParallelCalcCustomAngleForceKernel(std::string name, const Platform& platform, HipPlatform::PlatformData& data, const System& system);
    CommonCalcCustomAngleForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcCustomAngleForceKernel&>(kernels[index].getImpl());
    }
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
    class Task;
    HipPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

/**
 * This kernel is invoked by PeriodicTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class HipParallelCalcPeriodicTorsionForceKernel : public CalcPeriodicTorsionForceKernel {
public:
    HipParallelCalcPeriodicTorsionForceKernel(std::string name, const Platform& platform, HipPlatform::PlatformData& data, const System& system);
    CommonCalcPeriodicTorsionForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcPeriodicTorsionForceKernel&>(kernels[index].getImpl());
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
    class Task;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the PeriodicTorsionForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const PeriodicTorsionForce& force);
private:
    HipPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

/**
 * This kernel is invoked by RBTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class HipParallelCalcRBTorsionForceKernel : public CalcRBTorsionForceKernel {
public:
    HipParallelCalcRBTorsionForceKernel(std::string name, const Platform& platform, HipPlatform::PlatformData& data, const System& system);
    CommonCalcRBTorsionForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcRBTorsionForceKernel&>(kernels[index].getImpl());
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
    class Task;
    HipPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

/**
 * This kernel is invoked by CMAPTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class HipParallelCalcCMAPTorsionForceKernel : public CalcCMAPTorsionForceKernel {
public:
    HipParallelCalcCMAPTorsionForceKernel(std::string name, const Platform& platform, HipPlatform::PlatformData& data, const System& system);
    CommonCalcCMAPTorsionForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcCMAPTorsionForceKernel&>(kernels[index].getImpl());
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
    class Task;
    HipPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

/**
 * This kernel is invoked by CustomTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class HipParallelCalcCustomTorsionForceKernel : public CalcCustomTorsionForceKernel {
public:
    HipParallelCalcCustomTorsionForceKernel(std::string name, const Platform& platform, HipPlatform::PlatformData& data, const System& system);
    CommonCalcCustomTorsionForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcCustomTorsionForceKernel&>(kernels[index].getImpl());
    }
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
    class Task;
    HipPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

/**
 * This kernel is invoked by NonbondedForce to calculate the forces acting on the system.
 */
class HipParallelCalcNonbondedForceKernel : public CalcNonbondedForceKernel {
public:
    HipParallelCalcNonbondedForceKernel(std::string name, const Platform& platform, HipPlatform::PlatformData& data, const System& system);
    HipCalcNonbondedForceKernel& getKernel(int index) {
        return dynamic_cast<HipCalcNonbondedForceKernel&>(kernels[index].getImpl());
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the NonbondedForce this kernel will be used for
     */
    void initialize(const System& system, const NonbondedForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @param includeReciprocal  true if reciprocal space interactions should be included
     * @param includeReciprocal  true if reciprocal space interactions should be included
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the NonbondedForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const NonbondedForce& force);
    /**
     * Get the parameters being used for PME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Get the dispersion parameters being used for the dispersion term in LJPME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
private:
    class Task;
    HipPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

/**
 * This kernel is invoked by CustomNonbondedForce to calculate the forces acting on the system.
 */
class HipParallelCalcCustomNonbondedForceKernel : public CalcCustomNonbondedForceKernel {
public:
    HipParallelCalcCustomNonbondedForceKernel(std::string name, const Platform& platform, HipPlatform::PlatformData& data, const System& system);
    CommonCalcCustomNonbondedForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcCustomNonbondedForceKernel&>(kernels[index].getImpl());
    }
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
    class Task;
    HipPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

/**
 * This kernel is invoked by CustomExternalForce to calculate the forces acting on the system and the energy of the system.
 */
class HipParallelCalcCustomExternalForceKernel : public CalcCustomExternalForceKernel {
public:
    HipParallelCalcCustomExternalForceKernel(std::string name, const Platform& platform, HipPlatform::PlatformData& data, const System& system);
    CommonCalcCustomExternalForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcCustomExternalForceKernel&>(kernels[index].getImpl());
    }
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
    class Task;
    HipPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

/**
 * This kernel is invoked by CustomHbondForce to calculate the forces acting on the system.
 */
class HipParallelCalcCustomHbondForceKernel : public CalcCustomHbondForceKernel {
public:
    HipParallelCalcCustomHbondForceKernel(std::string name, const Platform& platform, HipPlatform::PlatformData& data, const System& system);
    CommonCalcCustomHbondForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcCustomHbondForceKernel&>(kernels[index].getImpl());
    }
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
    class Task;
    HipPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

/**
 * This kernel is invoked by CustomCompoundBondForce to calculate the forces acting on the system.
 */
class HipParallelCalcCustomCompoundBondForceKernel : public CalcCustomCompoundBondForceKernel {
public:
    HipParallelCalcCustomCompoundBondForceKernel(std::string name, const Platform& platform, HipPlatform::PlatformData& data, const System& system);
    CommonCalcCustomCompoundBondForceKernel& getKernel(int index) {
        return dynamic_cast<CommonCalcCustomCompoundBondForceKernel&>(kernels[index].getImpl());
    }
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
    class Task;
    HipPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

} // namespace OpenMM

#endif /*OPENMM_HIPPARALLELKERNELS_H_*/
