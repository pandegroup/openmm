#ifndef OPENMM_HIPKERNELS_H_
#define OPENMM_HIPKERNELS_H_

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

#include "HipPlatform.h"
#include "HipArray.h"
#include "HipContext.h"
#include "HipFFT3D.h"
#include "HipParameterSet.h"
#include "HipSort.h"
#include "openmm/kernels.h"
#include "openmm/System.h"
#include "openmm/internal/CompiledExpressionSet.h"
#include "openmm/internal/CustomIntegratorUtilities.h"
#include "lepton/CompiledExpression.h"
#include "lepton/ExpressionProgram.h"
#include <hipfft.h>

namespace OpenMM {

/**
 * This abstract class defines an interface for code that can compile CUDA kernels.  This allows a plugin to take advantage of runtime compilation
 * when running on recent versions of CUDA.
 */
class HipCompilerKernel : public KernelImpl {
public:
    static std::string Name() {
        return "HipCompilerKernel";
    }
    HipCompilerKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Compile a kernel to PTX.
     *
     * @param source     the source code for the kernel
     * @param options    the flags to be passed to the compiler
     * @param cu         the HipContext for which the kernel is being compiled
     */
    virtual std::string createModule(const std::string& source, const std::string& flags, HipContext& cu) = 0;
};

/**
 * This kernel is invoked at the beginning and end of force and energy computations.  It gives the
 * Platform a chance to clear buffers and do other initialization at the beginning, and to do any
 * necessary work at the end to determine the final results.
 */
class HipCalcForcesAndEnergyKernel : public CalcForcesAndEnergyKernel {
public:
    HipCalcForcesAndEnergyKernel(std::string name, const Platform& platform, HipContext& cu) : CalcForcesAndEnergyKernel(name, platform), cu(cu) {
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
   HipContext& cu;
};

/**
 * This kernel provides methods for setting and retrieving various state data: time, positions,
 * velocities, and forces.
 */
class HipUpdateStateDataKernel : public UpdateStateDataKernel {
public:
    HipUpdateStateDataKernel(std::string name, const Platform& platform, HipContext& cu) : UpdateStateDataKernel(name, platform), cu(cu) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     */
    void initialize(const System& system);
    /**
     * Get the current time (in picoseconds).
     *
     * @param context    the context in which to execute this kernel
     */
    double getTime(const ContextImpl& context) const;
    /**
     * Set the current time (in picoseconds).
     *
     * @param context    the context in which to execute this kernel
     */
    void setTime(ContextImpl& context, double time);
    /**
     * Get the positions of all particles.
     *
     * @param positions  on exit, this contains the particle positions
     */
    void getPositions(ContextImpl& context, std::vector<Vec3>& positions);
    /**
     * Set the positions of all particles.
     *
     * @param positions  a vector containg the particle positions
     */
    void setPositions(ContextImpl& context, const std::vector<Vec3>& positions);
    /**
     * Get the velocities of all particles.
     *
     * @param velocities  on exit, this contains the particle velocities
     */
    void getVelocities(ContextImpl& context, std::vector<Vec3>& velocities);
    /**
     * Set the velocities of all particles.
     *
     * @param velocities  a vector containg the particle velocities
     */
    void setVelocities(ContextImpl& context, const std::vector<Vec3>& velocities);
    /**
     * Get the current forces on all particles.
     *
     * @param forces  on exit, this contains the forces
     */
    void getForces(ContextImpl& context, std::vector<Vec3>& forces);
    /**
     * Get the current derivatives of the energy with respect to context parameters.
     *
     * @param derivs  on exit, this contains the derivatives
     */
    void getEnergyParameterDerivatives(ContextImpl& context, std::map<std::string, double>& derivs);
    /**
     * Get the current periodic box vectors.
     *
     * @param a      on exit, this contains the vector defining the first edge of the periodic box
     * @param b      on exit, this contains the vector defining the second edge of the periodic box
     * @param c      on exit, this contains the vector defining the third edge of the periodic box
     */
    void getPeriodicBoxVectors(ContextImpl& context, Vec3& a, Vec3& b, Vec3& c) const;
    /**
     * Set the current periodic box vectors.
     *
     * @param a      the vector defining the first edge of the periodic box
     * @param b      the vector defining the second edge of the periodic box
     * @param c      the vector defining the third edge of the periodic box
     */
    void setPeriodicBoxVectors(ContextImpl& context, const Vec3& a, const Vec3& b, const Vec3& c);
    /**
     * Create a checkpoint recording the current state of the Context.
     *
     * @param stream    an output stream the checkpoint data should be written to
     */
    void createCheckpoint(ContextImpl& context, std::ostream& stream);
    /**
     * Load a checkpoint that was written by createCheckpoint().
     *
     * @param stream    an input stream the checkpoint data should be read from
     */
    void loadCheckpoint(ContextImpl& context, std::istream& stream);
private:
    HipContext& cu;
};

/**
 * This kernel modifies the positions of particles to enforce distance constraints.
 */
class HipApplyConstraintsKernel : public ApplyConstraintsKernel {
public:
    HipApplyConstraintsKernel(std::string name, const Platform& platform, HipContext& cu) : ApplyConstraintsKernel(name, platform),
            cu(cu), hasInitializedKernel(false) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     */
    void initialize(const System& system);
    /**
     * Update particle positions to enforce constraints.
     *
     * @param context    the context in which to execute this kernel
     * @param tol        the distance tolerance within which constraints must be satisfied.
     */
    void apply(ContextImpl& context, double tol);
    /**
     * Update particle velocities to enforce constraints.
     *
     * @param context    the context in which to execute this kernel
     * @param tol        the velocity tolerance within which constraints must be satisfied.
     */
    void applyToVelocities(ContextImpl& context, double tol);
private:
    HipContext& cu;
    bool hasInitializedKernel;
    hipFunction_t applyDeltasKernel;
};

/**
 * This kernel recomputes the positions of virtual sites.
 */
class HipVirtualSitesKernel : public VirtualSitesKernel {
public:
    HipVirtualSitesKernel(std::string name, const Platform& platform, HipContext& cu) : VirtualSitesKernel(name, platform), cu(cu) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     */
    void initialize(const System& system);
    /**
     * Compute the virtual site locations.
     *
     * @param context    the context in which to execute this kernel
     */
    void computePositions(ContextImpl& context);
private:
    HipContext& cu;
};

/**
 * This kernel is invoked by NonbondedForce to calculate the forces acting on the system.
 */
class HipCalcNonbondedForceKernel : public CalcNonbondedForceKernel {
public:
    HipCalcNonbondedForceKernel(std::string name, const Platform& platform, HipContext& cu, const System& system) : CalcNonbondedForceKernel(name, platform),
            cu(cu), hasInitializedFFT(false), sort(NULL), dispersionFft(NULL), fft(NULL), pmeio(NULL), usePmeStream(false) {
    }
    ~HipCalcNonbondedForceKernel();
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
     * @param includeDirect  true if direct space interactions should be included
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
    class SortTrait : public HipSort::SortTrait {
        int getDataSize() const {return 8;}
        int getKeySize() const {return 4;}
        const char* getDataType() const {return "int2";}
        const char* getKeyType() const {return "int";}
        const char* getMinKey() const {return "(-2147483647-1)";}
        const char* getMaxKey() const {return "2147483647";}
        const char* getMaxValue() const {return "make_int2(2147483647, 2147483647)";}
        const char* getSortKey() const {return "value.y";}
    };
    class ForceInfo;
    class PmeIO;
    class PmePreComputation;
    class PmePostComputation;
    class SyncStreamPreComputation;
    class SyncStreamPostComputation;
    HipContext& cu;
    ForceInfo* info;
    bool hasInitializedFFT;
    HipArray charges;
    HipArray sigmaEpsilon;
    HipArray exceptionParams;
    HipArray exclusionAtoms;
    HipArray exclusionParams;
    HipArray baseParticleParams;
    HipArray baseExceptionParams;
    HipArray particleParamOffsets;
    HipArray exceptionParamOffsets;
    HipArray particleOffsetIndices;
    HipArray exceptionOffsetIndices;
    HipArray globalParams;
    HipArray cosSinSums;
    HipArray pmeGrid1;
    HipArray pmeGrid2;
    HipArray pmeBsplineModuliX;
    HipArray pmeBsplineModuliY;
    HipArray pmeBsplineModuliZ;
    HipArray pmeDispersionBsplineModuliX;
    HipArray pmeDispersionBsplineModuliY;
    HipArray pmeDispersionBsplineModuliZ;
    HipArray pmeAtomGridIndex;
    HipArray pmeEnergyBuffer;
    HipSort* sort;
    Kernel cpuPme;
    PmeIO* pmeio;
    hipStream_t pmeStream;
    hipEvent_t pmeSyncEvent, paramsSyncEvent;
    HipFFT3D* fft;
    hipfftHandle fftForward;
    hipfftHandle fftBackward;
    HipFFT3D* dispersionFft;
    hipfftHandle dispersionFftForward;
    hipfftHandle dispersionFftBackward;
    hipFunction_t computeParamsKernel, computeExclusionParamsKernel;
    hipFunction_t ewaldSumsKernel;
    hipFunction_t ewaldForcesKernel;
    hipFunction_t pmeGridIndexKernel;
    hipFunction_t pmeDispersionGridIndexKernel;
    hipFunction_t pmeSpreadChargeKernel;
    hipFunction_t pmeDispersionSpreadChargeKernel;
    hipFunction_t pmeFinishSpreadChargeKernel;
    hipFunction_t pmeDispersionFinishSpreadChargeKernel;
    hipFunction_t pmeEvalEnergyKernel;
    hipFunction_t pmeEvalDispersionEnergyKernel;
    hipFunction_t pmeConvolutionKernel;
    hipFunction_t pmeDispersionConvolutionKernel;
    hipFunction_t pmeInterpolateForceKernel;
    hipFunction_t pmeInterpolateDispersionForceKernel;
    std::vector<std::pair<int, int> > exceptionAtoms;
    std::vector<std::string> paramNames;
    std::vector<double> paramValues;
    double ewaldSelfEnergy, dispersionCoefficient, alpha, dispersionAlpha;
    int interpolateForceThreads;
    int gridSizeX, gridSizeY, gridSizeZ;
    int dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ;
    bool hasCoulomb, hasLJ, usePmeStream, useHipFFT, doLJPME, usePosqCharges, recomputeParams, hasOffsets;
    NonbondedMethod nonbondedMethod;
    static const int PmeOrder = 5;
};

/**
 * This kernel is invoked by CustomCVForce to calculate the forces acting on the system and the energy of the system.
 */
class HipCalcCustomCVForceKernel : public CalcCustomCVForceKernel {
public:
    HipCalcCustomCVForceKernel(std::string name, const Platform& platform, HipContext& cu) : CalcCustomCVForceKernel(name, platform),
            cu(cu), hasInitializedListeners(false) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the CustomCVForce this kernel will be used for
     * @param innerContext   the context created by the CustomCVForce for computing collective variables
     */
    void initialize(const System& system, const CustomCVForce& force, ContextImpl& innerContext);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param innerContext   the context created by the CustomCVForce for computing collective variables
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, ContextImpl& innerContext, bool includeForces, bool includeEnergy);
    /**
     * Copy state information to the inner context.
     *
     * @param context        the context in which to execute this kernel
     * @param innerContext   the context created by the CustomCVForce for computing collective variables
     */
    void copyState(ContextImpl& context, ContextImpl& innerContext);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the CustomCVForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const CustomCVForce& force);
private:
    class ForceInfo;
    class ReorderListener;
    HipContext& cu;
    bool hasInitializedListeners;
    Lepton::ExpressionProgram energyExpression;
    std::vector<std::string> variableNames, paramDerivNames, globalParameterNames;
    std::vector<Lepton::ExpressionProgram> variableDerivExpressions;
    std::vector<Lepton::ExpressionProgram> paramDerivExpressions;
    std::vector<HipArray> cvForces;
    HipArray invAtomOrder;
    HipArray innerInvAtomOrder;
    hipFunction_t copyStateKernel, copyForcesKernel, addForcesKernel;
};

/**
 * This kernel is invoked by MonteCarloBarostat to adjust the periodic box volume
 */
class HipApplyMonteCarloBarostatKernel : public ApplyMonteCarloBarostatKernel {
public:
    HipApplyMonteCarloBarostatKernel(std::string name, const Platform& platform, HipContext& cu) : ApplyMonteCarloBarostatKernel(name, platform), cu(cu),
            hasInitializedKernels(false) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param barostat   the MonteCarloBarostat this kernel will be used for
     */
    void initialize(const System& system, const Force& barostat);
    /**
     * Attempt a Monte Carlo step, scaling particle positions (or cluster centers) by a specified value.
     * This version scales the x, y, and z positions independently.
     * This is called BEFORE the periodic box size is modified.  It should begin by translating each particle
     * or cluster into the first periodic box, so that coordinates will still be correct after the box size
     * is changed.
     *
     * @param context    the context in which to execute this kernel
     * @param scaleX     the scale factor by which to multiply particle x-coordinate
     * @param scaleY     the scale factor by which to multiply particle y-coordinate
     * @param scaleZ     the scale factor by which to multiply particle z-coordinate
     */
    void scaleCoordinates(ContextImpl& context, double scaleX, double scaleY, double scaleZ);
    /**
     * Reject the most recent Monte Carlo step, restoring the particle positions to where they were before
     * scaleCoordinates() was last called.
     *
     * @param context    the context in which to execute this kernel
     */
    void restoreCoordinates(ContextImpl& context);
private:
    HipContext& cu;
    bool hasInitializedKernels;
    int numMolecules;
    HipArray savedPositions;
    HipArray savedForces;
    HipArray moleculeAtoms;
    HipArray moleculeStartIndex;
    hipFunction_t kernel;
    std::vector<int> lastAtomOrder;
};

} // namespace OpenMM

#endif /*OPENMM_HIPKERNELS_H_*/
