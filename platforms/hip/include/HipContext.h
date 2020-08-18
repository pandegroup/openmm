#ifndef OPENMM_HIP_CONTEXT_H_
#define OPENMM_HIP_CONTEXT_H_

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


/*
 * Porting notes:
  - Hip only marginally supports the CUDA context API, and will remove
    support eventually.  To my knowledge, contexts don't really buy you anything
    that streams / hipSetDevice don't.  Hence, for this implementation, we are doing
    away entirely with the context usage.
 */


#include <map>
#include <string>
#include <utility>
#define __CL_ENABLE_EXCEPTIONS
#ifdef _MSC_VER
    #error "Windows unsupported for HIP platform"
    // Prevent Windows from defining macros that interfere with other code.
    #define NOMINMAX
#endif
#include <pthread.h>
#include <hip/hip_runtime.h>
#ifdef __HIP_PLATFORM_NVCC__
#include <builtin_types.h>
#include <vector_functions.h>
#endif
#include "openmm/common/windowsExportCommon.h"
#include "HipArray.h"
#include "HipBondedUtilities.h"
#include "HipExpressionUtilities.h"
#include "HipIntegrationUtilities.h"
#include "HipNonbondedUtilities.h"
#include "HipPlatform.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/Kernel.h"

typedef unsigned long tileflags;

namespace OpenMM {

/**
 * This class contains the information associated with a Context by the HIP Platform.  Each HipContext is
 * specific to a particular device, and manages data structures and kernels for that device.  When running a simulation
 * in parallel on multiple devices, there is a separate HipContext for each one.  The list of all contexts is
 * stored in the HipPlatform::PlatformData.
 * <p>
 * In addition, a worker thread is created for each HipContext.  This is used for parallel computations, so that
 * blocking calls to one device will not block other devices.  When only a single device is being used, the worker
 * thread is not used and calculations are performed on the main application thread.
 */

class OPENMM_EXPORT_COMMON HipContext : public ComputeContext {
public:
    class WorkTask;
    class WorkThread;
    class ReorderListener;
    class ForcePreComputation;
    class ForcePostComputation;
    static const int ThreadBlockSize;
    static const int TileSize;
    HipContext(const System& system, int deviceIndex, bool useBlockingSync, const std::string& precision,
            const std::string& compiler, const std::string& tempDir, const std::string& hostCompiler, HipPlatform::PlatformData& platformData,
            HipContext* originalContext);
    ~HipContext();
    /**
     * This is called to initialize internal data structures after all Forces in the system
     * have been initialized.
     */
    void initialize();
    /**
     * Get whether the context associated with this object is valid.
     */
    bool getContextIsValid() const {
        return contextIsValid;
    }
    /**
     * Set the hipCtx_t associated with this object to be the current context.  If the context is not
     * valid, this returns without doing anything.
     */
    void setAsCurrent();
    /**
     * Get the hipDevice_t associated with this object.
     */
    hipDevice_t getDevice() {
        return device;
    }
    /**
     * Get the compute capability of the device associated with this object.
     */
    double getComputeCapability() const {
        return computeCapability;
    }
    /**
     * Get the index of the hipDevice_t associated with this object.
     */
    int getDeviceIndex() const {
        return deviceIndex;
    }
    /**
     * Get the PlatformData object this context is part of.
     */
    HipPlatform::PlatformData& getPlatformData() {
        return platformData;
    }
    /**
     * Get the number of contexts being used for the current simulation.
     * This is relevant when a simulation is parallelized across multiple devices.  In that case,
     * one HipContext is created for each device.
     */
    int getNumContexts() const {
        return platformData.contexts.size();
    }
    /**
     * Get the index of this context in the list stored in the PlatformData.
     */
    int getContextIndex() const {
        return contextIndex;
    }
    /**
     * Get the stream currently being used for execution.
     */
    hipStream_t getCurrentStream();
    /**
     * Set the stream to use for execution.
     */
    void setCurrentStream(hipStream_t stream);
    /**
     * Reset the context to using the default stream for execution.
     */
    void restoreDefaultStream();
    /**
     * Construct an uninitialized array of the appropriate class for this platform.  The returned
     * value should be created on the heap with the "new" operator.
     */
    HipArray* createArray();
    /**
     * Construct a ComputeEvent object of the appropriate class for this platform.
     */
    ComputeEvent createEvent();
    /**
     * Compile source code to create a ComputeProgram.
     *
     * @param source             the source code of the program
     * @param defines            a set of preprocessor definitions (name, value) to define when compiling the program
     */
    ComputeProgram compileProgram(const std::string source, const std::map<std::string, std::string>& defines=std::map<std::string, std::string>());
    /**
     * Convert an array to an HipArray.  If the argument is already an HipArray, this simply casts it.
     * If the argument is a ComputeArray that wraps a HipArray, this returns the wrapped array.  For any
     * other argument, this throws an exception.
     */
    HipArray& unwrap(ArrayInterface& array) const;
    /**
     * Get the array which contains the position (the xyz components) and charge (the w component) of each atom.
     */
    HipArray& getPosq() {
        return posq;
    }
    /**
     * Get the array which contains a correction to the position of each atom.  This only exists if getUseMixedPrecision() returns true.
     */
    HipArray& getPosqCorrection() {
        return posqCorrection;
    }
    /**
     * Get the array which contains the velocity (the xyz components) and inverse mass (the w component) of each atom.
     */
    HipArray& getVelm() {
        return velm;
    }
    /**
     * Get the array which contains the force on each atom (represented as three long longs in 64 bit fixed point).
     */
    HipArray& getForce() {
        return force;
    }
    /**
     * Get the array which contains a contribution to each force represented as 64 bit fixed point.
     * This is a synonym for getForce().  It exists to satisfy the ComputeContext interface.
     */
    HipArray& getLongForceBuffer() {
        return force;
    }
    /**
     * Not all HIP devices support 64 bit atomics, so this throws an exception.
     * @return
     */
    ArrayInterface& getForceBuffers() {
        throw OpenMMException("HIP platform does not use floating point force buffers");
    }
    /**
     * Get the array which contains the buffer in which energy is computed.
     */
    HipArray& getEnergyBuffer() {
        return energyBuffer;
    }
    /**
     * Get the array which contains the buffer in which derivatives of the energy with respect to parameters are computed.
     */
    HipArray& getEnergyParamDerivBuffer() {
        return energyParamDerivBuffer;
    }
    /**
     * Get a pointer to a block of pinned memory that can be used for efficient transfers between host and device.
     * This is guaranteed to be at least as large as any of the arrays returned by methods of this class.
     */
    void* getPinnedBuffer() {
        return pinnedBuffer;
    }
    /**
     * Get a shared ThreadPool that code can use to parallelize operations.
     *
     * Because this object is freely available to all code, care is needed to avoid conflicts.  Only use it
     * from the main thread, and make sure all operations are complete before you invoke any other code that
     * might make use of it
     */
    ThreadPool& getThreadPool() {
        return getPlatformData().threads;
    }
    /**
     * Get the array which contains the index of each atom.
     */
    HipArray& getAtomIndexArray() {
        return atomIndexDevice;
    }
    /**
     * Create a HIP_ module from source code.
     *
     * @param source             the source code of the module
     * @param optimizationFlags  the optimization flags to pass to the HIP_ compiler.  If this is
     *                           omitted, a default set of options will be used
     */
    hipModule_t createModule(const std::string source, const char* optimizationFlags = NULL);
    /**
     * Create a HIP_ module from source code.
     *
     * @param source             the source code of the module
     * @param defines            a set of preprocessor definitions (name, value) to define when compiling the program
     * @param optimizationFlags  the optimization flags to pass to the HIP_ compiler.  If this is
     *                           omitted, a default set of options will be used
     */
    hipModule_t createModule(const std::string source, const std::map<std::string, std::string>& defines, const char* optimizationFlags = NULL);
    /**
     * Get a kernel from a HIP_ module.
     *
     * @param module    the module to get the kernel from
     * @param name      the name of the kernel to get
     */
    hipFunction_t getKernel(hipModule_t& module, const std::string& name);
    /**
     * Execute a kernel.
     *
     * @param kernel       the kernel to execute
     * @param arguments    an array of pointers to the kernel arguments
     * @param threads      the maximum number of threads that should be used
     * @param blockSize    the size of each thread block to use
     * @param sharedSize   the amount of dynamic shared memory to allocated for the kernel, in bytes
     */
    void executeKernel(hipFunction_t kernel, void** arguments, int workUnits, int blockSize = -1, unsigned int sharedSize = 0);
    /**
     * Compute the largest thread block size that can be used for a kernel that requires a particular amount of
     * shared memory per thread.
     *
     * @param memory        the number of bytes of shared memory per thread
     * @param preferShared  whether the kernel is set to prefer shared memory over cache
     */
    int computeThreadBlockSize(double memory, bool preferShared=true) const;
    /**
     * Set all elements of an array to 0.
     */
    void clearBuffer(ArrayInterface& array);
    /**
     * Set all elements of an array to 0.
     *
     * @param memory     the memory to clear
     * @param size       the size of the buffer in bytes
     */
    void clearBuffer(hipDeviceptr_t memory, int size);
    /**
     * Register a buffer that should be automatically cleared (all elements set to 0) at the start of each force or energy computation.
     */
    void addAutoclearBuffer(ArrayInterface& array);
    /**
     * Register a buffer that should be automatically cleared (all elements set to 0) at the start of each force or energy computation.
     *
     * @param memory     the memory to clear
     * @param size       the size of the buffer in bytes
     */
    void addAutoclearBuffer(hipDeviceptr_t memory, int size);
    /**
     * Clear all buffers that have been registered with addAutoclearBuffer().
     */
    void clearAutoclearBuffers();
    /**
     * Sum the buffer containing energy.
     */
    double reduceEnergy();
    /**
     * Get the number of blocks of TileSize atoms.
     */
    int getNumAtomBlocks() const {
        return numAtomBlocks;
    }
    /**
     * Get the standard number of thread blocks to use when executing kernels.
     */
    int getNumThreadBlocks() const {
        return numThreadBlocks;
    }
    /**
     * Get the maximum number of threads in a thread block supported by this device.
     */
    int getMaxThreadBlockSize() const {
        return 1024;
    }
    /**
     * Get whether the device being used is a CPU.  In some cases, different algorithms
     * may be more efficient on CPUs and GPUs.
     */
    bool getIsCPU() const {
        return false;
    }
    /**
     * Get the SIMD width of the device being used.
     */
    int getSIMDWidth() const {
        return simdWidth;
    }
    /**
     * Get whether the device being used warp shuffles.
     */
    bool getSupportsWarpShuffle() const {
        return hasWarpShuffle;
    }
    /**
     * Get whether the device being used supports 64 bit atomic operations on global memory.
     */
    bool getSupports64BitGlobalAtomics() const {
        return hasGlobalInt64Atomics;
    }
    /**
     * Get whether the device being used supports double precision math.
     */
    bool getSupportsDoublePrecision() const {
        return hasDoubles;
    }
    /**
     * Get whether double precision is being used.
     */
    bool getUseDoublePrecision() const {
        return useDoublePrecision;
    }
    /**
     * Get whether mixed precision is being used.
     */
    bool getUseMixedPrecision() const {
        return useMixedPrecision;
    }
    /**
     * Get whether the periodic box is triclinic.
     */
    bool getBoxIsTriclinic() const {
        return boxIsTriclinic;
    }
    /**
     * Convert a HIP_ result code to the corresponding string description.
     */
    static std::string getErrorString(hipError_t result);
    /**
     * Get the vectors defining the periodic box.
     */
    void getPeriodicBoxVectors(Vec3& a, Vec3& b, Vec3& c) const {
        a = Vec3(periodicBoxVecX.x, periodicBoxVecX.y, periodicBoxVecX.z);
        b = Vec3(periodicBoxVecY.x, periodicBoxVecY.y, periodicBoxVecY.z);
        c = Vec3(periodicBoxVecZ.x, periodicBoxVecZ.y, periodicBoxVecZ.z);
    }
    /**
     * Set the vectors defining the periodic box.
     */
    void setPeriodicBoxVectors(const Vec3& a, const Vec3& b, const Vec3& c) {
        periodicBoxVecX = make_double4(a[0], a[1], a[2], 0.0);
        periodicBoxVecY = make_double4(b[0], b[1], b[2], 0.0);
        periodicBoxVecZ = make_double4(c[0], c[1], c[2], 0.0);
        periodicBoxVecXFloat = make_float4((float) a[0], (float) a[1], (float) a[2], 0.0f);
        periodicBoxVecYFloat = make_float4((float) b[0], (float) b[1], (float) b[2], 0.0f);
        periodicBoxVecZFloat = make_float4((float) c[0], (float) c[1], (float) c[2], 0.0f);
        periodicBoxSize = make_double4(a[0], b[1], c[2], 0.0);
        invPeriodicBoxSize = make_double4(1.0/a[0], 1.0/b[1], 1.0/c[2], 0.0);
        periodicBoxSizeFloat = make_float4((float) a[0], (float) b[1], (float) c[2], 0.0f);
        invPeriodicBoxSizeFloat = make_float4(1.0f/(float) a[0], 1.0f/(float) b[1], 1.0f/(float) c[2], 0.0f);
    }
    /**
     * Get the size of the periodic box.
     */
    double4 getPeriodicBoxSize() const {
        return periodicBoxSize;
    }
    /**
     * Get the inverse of the size of the periodic box.
     */
    double4 getInvPeriodicBoxSize() const {
        return invPeriodicBoxSize;
    }
    /**
     * Get a pointer to the size of the periodic box, represented as either a float4 or double4 depending on
     * this context's precision.  This value is suitable for passing to kernels as an argument.
     */
    void* getPeriodicBoxSizePointer() {
        return (useDoublePrecision ? reinterpret_cast<void*>(&periodicBoxSize) : reinterpret_cast<void*>(&periodicBoxSizeFloat));
    }
    /**
     * Get a pointer to the inverse of the size of the periodic box, represented as either a float4 or double4 depending on
     * this context's precision.  This value is suitable for passing to kernels as an argument.
     */
    void* getInvPeriodicBoxSizePointer() {
        return (useDoublePrecision ? reinterpret_cast<void*>(&invPeriodicBoxSize) : reinterpret_cast<void*>(&invPeriodicBoxSizeFloat));
    }
    /**
     * Get a pointer to the first periodic box vector, represented as either a float4 or double4 depending on
     * this context's precision.  This value is suitable for passing to kernels as an argument.
     */
    void* getPeriodicBoxVecXPointer() {
        return (useDoublePrecision ? reinterpret_cast<void*>(&periodicBoxVecX) : reinterpret_cast<void*>(&periodicBoxVecXFloat));
    }
    /**
     * Get a pointer to the second periodic box vector, represented as either a float4 or double4 depending on
     * this context's precision.  This value is suitable for passing to kernels as an argument.
     */
    void* getPeriodicBoxVecYPointer() {
        return (useDoublePrecision ? reinterpret_cast<void*>(&periodicBoxVecY) : reinterpret_cast<void*>(&periodicBoxVecYFloat));
    }
    /**
     * Get a pointer to the third periodic box vector, represented as either a float4 or double4 depending on
     * this context's precision.  This value is suitable for passing to kernels as an argument.
     */
    void* getPeriodicBoxVecZPointer() {
        return (useDoublePrecision ? reinterpret_cast<void*>(&periodicBoxVecZ) : reinterpret_cast<void*>(&periodicBoxVecZFloat));
    }
    /**
     * Get the HipIntegrationUtilities for this context.
     */
    HipIntegrationUtilities& getIntegrationUtilities() {
        return *integration;
    }
    /**
     * Get the HipExpressionUtilities for this context.
     */
    HipExpressionUtilities& getExpressionUtilities() {
        return *expression;
    }
    /**
     * Get the HipBondedUtilities for this context.
     */
    HipBondedUtilities& getBondedUtilities() {
        return *bonded;
    }
    /**
     * Get the HipNonbondedUtilities for this context.
     */
    HipNonbondedUtilities& getNonbondedUtilities() {
        return *nonbonded;
    }
    /**
     * This should be called by the Integrator from its own initialize() method.
     * It ensures all contexts are fully initialized.
     */
    void initializeContexts();
    /**
     * Set the particle charges.  These are packed into the fourth element of the posq array.
     */
    void setCharges(const std::vector<double>& charges);
    /**
     * Request to use the fourth element of the posq array for storing charges.  Since only one force can
     * do that, this returns true the first time it is called, and false on all subsequent calls.
     */
    bool requestPosqCharges();
    /**
     * Get the names of all parameters with respect to which energy derivatives are computed.
     */
    const std::vector<std::string>& getEnergyParamDerivNames() const {
        return energyParamDerivNames;
    }
    /**
     * Get a workspace data structure used for accumulating the values of derivatives of the energy
     * with respect to parameters.
     */
    std::map<std::string, double>& getEnergyParamDerivWorkspace() {
        return energyParamDerivWorkspace;
    }
    /**
     * Register that the derivative of potential energy with respect to a context parameter
     * will need to be calculated.  If this is called multiple times for a single parameter,
     * it is only added to the list once.
     *
     * @param param    the name of the parameter to add
     */
    void addEnergyParameterDerivative(const std::string& param);
    /**
     * Wait until all work that has been queued (kernel executions, asynchronous data transfers, etc.)
     * has been submitted to the device.  This does not mean it has necessarily been completed.
     * Calling this periodically may improve the responsiveness of the computer's GUI, but at the
     * expense of reduced simulation performance.
     */
    void flushQueue();
private:
    /**
     * Compute a sorted list of device indices in decreasing order of desirability
     */
    std::vector<int> getDevicePrecedence();
    static bool hasInitializedHip;
    double computeCapability;
    HipPlatform::PlatformData& platformData;
    int deviceIndex;
    int contextIndex;
    int numAtomBlocks;
    int numThreadBlocks;
    int simdWidth;
    int sharedMemPerBlock;
    bool hasDoubles;
    bool hasGlobalInt64Atomics;
    bool hasWarpShuffle;
    bool useBlockingSync, useDoublePrecision, useMixedPrecision, contextIsValid, boxIsTriclinic, hasCompilerKernel, isHipccAvailable, hasAssignedPosqCharges;
    bool isLinkedContext;
    std::string compiler, tempDir, cacheDir, gpuArchitecture;
    float4 periodicBoxVecXFloat, periodicBoxVecYFloat, periodicBoxVecZFloat, periodicBoxSizeFloat, invPeriodicBoxSizeFloat;
    double4 periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ, periodicBoxSize, invPeriodicBoxSize;
    std::string defaultOptimizationOptions;
    std::map<std::string, std::string> compilationDefines;
    hipDevice_t device;
    hipStream_t currentStream;
    hipFunction_t clearBufferKernel;
    hipFunction_t clearTwoBuffersKernel;
    hipFunction_t clearThreeBuffersKernel;
    hipFunction_t clearFourBuffersKernel;
    hipFunction_t clearFiveBuffersKernel;
    hipFunction_t clearSixBuffersKernel;
    hipFunction_t reduceEnergyKernel;
    hipFunction_t setChargesKernel;
    void* pinnedBuffer;
    HipArray posq;
    HipArray posqCorrection;
    HipArray velm;
    HipArray force;
    HipArray energyBuffer;
    HipArray energySum;
    HipArray energyParamDerivBuffer;
    HipArray atomIndexDevice;
    HipArray chargeBuffer;
    std::vector<std::string> energyParamDerivNames;
    std::map<std::string, double> energyParamDerivWorkspace;
    std::vector<hipDeviceptr_t> autoclearBuffers;
    std::vector<int> autoclearBufferSizes;
    HipIntegrationUtilities* integration;
    HipExpressionUtilities* expression;
    HipBondedUtilities* bonded;
    HipNonbondedUtilities* nonbonded;
    Kernel compilerKernel;
};

/**
 * This class exists only for backward compatibility.  Use ComputeContext::WorkTask instead.
 */
class OPENMM_EXPORT_COMMON HipContext::WorkTask : public ComputeContext::WorkTask {
};

/**
 * This class exists only for backward compatibility.  Use ComputeContext::ReorderListener instead.
 */
class OPENMM_EXPORT_COMMON HipContext::ReorderListener : public ComputeContext::ReorderListener {
};

/**
 * This class exists only for backward compatibility.  Use ComputeContext::ForcePreComputation instead.
 */
class OPENMM_EXPORT_COMMON HipContext::ForcePreComputation : public ComputeContext::ForcePreComputation {
};

/**
 * This class exists only for backward compatibility.  Use ComputeContext::ForcePostComputation instead.
 */
class OPENMM_EXPORT_COMMON HipContext::ForcePostComputation : public ComputeContext::ForcePostComputation {
};

} // namespace OpenMM

#endif /*OPENMM_HIP_CONTEXT_H_*/
