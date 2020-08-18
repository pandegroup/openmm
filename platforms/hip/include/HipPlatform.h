#ifndef OPENMM_HIPPLATFORM_H_
#define OPENMM_HIPPLATFORM_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
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

#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/internal/ThreadPool.h"
#include "openmm/common/windowsExportCommon.h"

namespace OpenMM {

class HipContext;

/**
 * This Platform subclass uses HIP implementations of the OpenMM kernels.
 */

class OPENMM_EXPORT_COMMON HipPlatform : public Platform {
public:
    class PlatformData;
    HipPlatform();
    const std::string& getName() const {
        static const std::string name = "HIP";
        return name;
    }
    double getSpeed() const;
    bool supportsDoublePrecision() const;
    const std::string& getPropertyValue(const Context& context, const std::string& property) const;
    void setPropertyValue(Context& context, const std::string& property, const std::string& value) const;
    void contextCreated(ContextImpl& context, const std::map<std::string, std::string>& properties) const;
    void linkedContextCreated(ContextImpl& context, ContextImpl& originalContext) const;
    void contextDestroyed(ContextImpl& context) const;
    /**
     * This is the name of the parameter for selecting which HIP device or devices to use.
     */
    static const std::string& HipDeviceIndex() {
        static const std::string key = "DeviceIndex";
        return key;
    }
    /**
     * This is the name of the parameter that reports the HIP device or devices being used.
     */
    static const std::string& HipDeviceName() {
        static const std::string key = "DeviceName";
        return key;
    }
    /**
     * This is the name of the parameter for selecting whether HIP should sync or spin loop while waiting for results.
     */
    static const std::string& HipUseBlockingSync() {
        static const std::string key = "UseBlockingSync";
        return key;
    }
    /**
     * This is the name of the parameter for selecting what numerical precision to use.
     */
    static const std::string& HipPrecision() {
        static const std::string key = "Precision";
        return key;
    }
    /**
     * This is the name of the parameter for selecting whether to use the CPU based PME calculation.
     */
    static const std::string& HipUseCpuPme() {
        static const std::string key = "UseCpuPme";
        return key;
    }
    /**
     * This is the name of the parameter for specifying the path to the HIP compiler.
     */
    static const std::string& HipCompiler() {
        static const std::string key = "HipCompiler";
        return key;
    }
    /**
     * This is the name of the parameter for specifying the host compiler for the HIP compiler to use.
     */
    static const std::string& HipHostCompiler() {
        static const std::string key = "HipHostCompiler";
        return key;
    }
    /**
     * This is the name of the parameter for specifying the path to the directory for creating temporary files.
     */
    static const std::string& HipTempDirectory() {
        static const std::string key = "TempDirectory";
        return key;
    }
    /**
     * This is the name of the parameter for selecting whether to disable use of a separate stream for PME.
     */
    static const std::string& HipDisablePmeStream() {
        static const std::string key = "DisablePmeStream";
        return key;
    }
    /**
     * This is the name of the parameter for requesting that force computations be fully deterministic.
     */
    static const std::string& HipDeterministicForces() {
        static const std::string key = "DeterministicForces";
        return key;
    }
};

class OPENMM_EXPORT_COMMON HipPlatform::PlatformData {
public:
    PlatformData(ContextImpl* context, const System& system, const std::string& deviceIndexProperty, const std::string& blockingProperty, const std::string& precisionProperty,
            const std::string& cpuPmeProperty, const std::string& compilerProperty, const std::string& tempProperty, const std::string& hostCompilerProperty,
            const std::string& pmeStreamProperty, const std::string& deterministicForcesProperty, int numThreads, ContextImpl* originalContext);
    ~PlatformData();
    void initializeContexts(const System& system);
    void syncContexts();
    ContextImpl* context;
    std::vector<HipContext*> contexts;
    std::vector<double> contextEnergy;
    bool hasInitializedContexts, removeCM, peerAccessSupported, useCpuPme, disablePmeStream, deterministicForces;
    int cmMotionFrequency;
    int stepCount, computeForceCount;
    double time;
    std::map<std::string, std::string> propertyValues;
    ThreadPool threads;
};

} // namespace OpenMM

#endif /*OPENMM_HIPPLATFORM_H_*/
