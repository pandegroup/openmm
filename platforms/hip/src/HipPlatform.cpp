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

#include "HipContext.h"
#include "HipExpressionUtilities.h"
#include "HipPlatform.h"
#include "HipKernelFactory.h"
#include "HipKernels.h"
#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/hardware.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <cstdio>
#ifdef _MSC_VER
    #error "Windows unsupported for HIP platform"
#endif
using namespace OpenMM;
using namespace std;

#define CHECK_RESULT(result, prefix) \
    if (result != hipSuccess) { \
        std::stringstream m; \
        m<<prefix<<": "<<HipContext::getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }


#ifdef OPENMM_COMMON_BUILDING_STATIC_LIBRARY
extern "C" void registerHipPlatform() {
    Platform::registerPlatform(new HipPlatform());
}
#else
extern "C" OPENMM_EXPORT_COMMON void registerPlatforms() {
    Platform::registerPlatform(new HipPlatform());
}
#endif

HipPlatform::HipPlatform() {
    deprecatedPropertyReplacements["HipDeviceIndex"] = HipDeviceIndex();
    deprecatedPropertyReplacements["HipDeviceName"] = HipDeviceName();
    deprecatedPropertyReplacements["HipUseBlockingSync"] = HipUseBlockingSync();
    deprecatedPropertyReplacements["HipPrecision"] = HipPrecision();
    deprecatedPropertyReplacements["HipUseCpuPme"] = HipUseCpuPme();
    deprecatedPropertyReplacements["HipTempDirectory"] = HipTempDirectory();
    deprecatedPropertyReplacements["HipDisablePmeStream"] = HipDisablePmeStream();
    deprecatedPropertyReplacements["HipDeterministicForces"] = HipDeterministicForces();
    HipKernelFactory* factory = new HipKernelFactory();
    registerKernelFactory(CalcForcesAndEnergyKernel::Name(), factory);
    registerKernelFactory(UpdateStateDataKernel::Name(), factory);
    registerKernelFactory(ApplyConstraintsKernel::Name(), factory);
    registerKernelFactory(VirtualSitesKernel::Name(), factory);
    registerKernelFactory(CalcHarmonicBondForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomBondForceKernel::Name(), factory);
    registerKernelFactory(CalcHarmonicAngleForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomAngleForceKernel::Name(), factory);
    registerKernelFactory(CalcPeriodicTorsionForceKernel::Name(), factory);
    registerKernelFactory(CalcRBTorsionForceKernel::Name(), factory);
    registerKernelFactory(CalcCMAPTorsionForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomTorsionForceKernel::Name(), factory);
    registerKernelFactory(CalcNonbondedForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomNonbondedForceKernel::Name(), factory);
    registerKernelFactory(CalcGBSAOBCForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomGBForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomExternalForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomHbondForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomCentroidBondForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomCompoundBondForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomCVForceKernel::Name(), factory);
    registerKernelFactory(CalcRMSDForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomManyParticleForceKernel::Name(), factory);
    registerKernelFactory(CalcGayBerneForceKernel::Name(), factory);
    registerKernelFactory(IntegrateVerletStepKernel::Name(), factory);
    registerKernelFactory(IntegrateNoseHooverStepKernel::Name(), factory);
    registerKernelFactory(IntegrateLangevinStepKernel::Name(), factory);
    registerKernelFactory(IntegrateLangevinMiddleStepKernel::Name(), factory);
    registerKernelFactory(IntegrateBrownianStepKernel::Name(), factory);
    registerKernelFactory(IntegrateVariableVerletStepKernel::Name(), factory);
    registerKernelFactory(IntegrateVariableLangevinStepKernel::Name(), factory);
    registerKernelFactory(IntegrateCustomStepKernel::Name(), factory);
    registerKernelFactory(ApplyAndersenThermostatKernel::Name(), factory);
    registerKernelFactory(ApplyMonteCarloBarostatKernel::Name(), factory);
    registerKernelFactory(RemoveCMMotionKernel::Name(), factory);
    platformProperties.push_back(HipDeviceIndex());
    platformProperties.push_back(HipDeviceName());
    platformProperties.push_back(HipUseBlockingSync());
    platformProperties.push_back(HipPrecision());
    platformProperties.push_back(HipUseCpuPme());
    platformProperties.push_back(HipCompiler());
    platformProperties.push_back(HipTempDirectory());
    platformProperties.push_back(HipHostCompiler());
    platformProperties.push_back(HipDisablePmeStream());
    platformProperties.push_back(HipDeterministicForces());
    setPropertyDefaultValue(HipDeviceIndex(), "");
    setPropertyDefaultValue(HipDeviceName(), "");
    setPropertyDefaultValue(HipUseBlockingSync(), "true");
    setPropertyDefaultValue(HipPrecision(), "single");
    setPropertyDefaultValue(HipUseCpuPme(), "false");
    setPropertyDefaultValue(HipDisablePmeStream(), "false");
    setPropertyDefaultValue(HipDeterministicForces(), "false");
    char* compiler = getenv("OPENMM_HIP_COMPILER");
    string hipcc = (compiler == NULL ? "/opt/rocm/bin/hipcc" : string(compiler));
    setPropertyDefaultValue(HipCompiler(), hipcc);
    char* tmpdir = getenv("TMPDIR");
    string tmp = (tmpdir == NULL ? string(P_tmpdir) : string(tmpdir));
    setPropertyDefaultValue(HipTempDirectory(), tmp);
    char* hostCompiler = getenv("HIP_HOST_COMPILER");
    setPropertyDefaultValue(HipHostCompiler(), (hostCompiler == NULL ? "" : string(hostCompiler)));
}

double HipPlatform::getSpeed() const {
    return 100;
}

bool HipPlatform::supportsDoublePrecision() const {
    return true;
}

const string& HipPlatform::getPropertyValue(const Context& context, const string& property) const {
    const ContextImpl& impl = getContextImpl(context);
    const PlatformData* data = reinterpret_cast<const PlatformData*>(impl.getPlatformData());
    string propertyName = property;
    if (deprecatedPropertyReplacements.find(property) != deprecatedPropertyReplacements.end())
        propertyName = deprecatedPropertyReplacements.find(property)->second;
    map<string, string>::const_iterator value = data->propertyValues.find(propertyName);
    if (value != data->propertyValues.end())
        return value->second;
    return Platform::getPropertyValue(context, property);
}

void HipPlatform::setPropertyValue(Context& context, const string& property, const string& value) const {
}

void HipPlatform::contextCreated(ContextImpl& context, const map<string, string>& properties) const {
    const string& devicePropValue = (properties.find(HipDeviceIndex()) == properties.end() ?
            getPropertyDefaultValue(HipDeviceIndex()) : properties.find(HipDeviceIndex())->second);
    string blockingPropValue = (properties.find(HipUseBlockingSync()) == properties.end() ?
            getPropertyDefaultValue(HipUseBlockingSync()) : properties.find(HipUseBlockingSync())->second);
    string precisionPropValue = (properties.find(HipPrecision()) == properties.end() ?
            getPropertyDefaultValue(HipPrecision()) : properties.find(HipPrecision())->second);
    string cpuPmePropValue = (properties.find(HipUseCpuPme()) == properties.end() ?
            getPropertyDefaultValue(HipUseCpuPme()) : properties.find(HipUseCpuPme())->second);
    const string& compilerPropValue = (properties.find(HipCompiler()) == properties.end() ?
            getPropertyDefaultValue(HipCompiler()) : properties.find(HipCompiler())->second);
    const string& tempPropValue = (properties.find(HipTempDirectory()) == properties.end() ?
            getPropertyDefaultValue(HipTempDirectory()) : properties.find(HipTempDirectory())->second);
    const string& hostCompilerPropValue = (properties.find(HipHostCompiler()) == properties.end() ?
            getPropertyDefaultValue(HipHostCompiler()) : properties.find(HipHostCompiler())->second);
    string pmeStreamPropValue = (properties.find(HipDisablePmeStream()) == properties.end() ?
            getPropertyDefaultValue(HipDisablePmeStream()) : properties.find(HipDisablePmeStream())->second);
    string deterministicForcesValue = (properties.find(HipDeterministicForces()) == properties.end() ?
            getPropertyDefaultValue(HipDeterministicForces()) : properties.find(HipDeterministicForces())->second);
    transform(blockingPropValue.begin(), blockingPropValue.end(), blockingPropValue.begin(), ::tolower);
    transform(precisionPropValue.begin(), precisionPropValue.end(), precisionPropValue.begin(), ::tolower);
    transform(cpuPmePropValue.begin(), cpuPmePropValue.end(), cpuPmePropValue.begin(), ::tolower);
    transform(pmeStreamPropValue.begin(), pmeStreamPropValue.end(), pmeStreamPropValue.begin(), ::tolower);
    transform(deterministicForcesValue.begin(), deterministicForcesValue.end(), deterministicForcesValue.begin(), ::tolower);
    vector<string> pmeKernelName;
    pmeKernelName.push_back(CalcPmeReciprocalForceKernel::Name());
    if (!supportsKernels(pmeKernelName))
        cpuPmePropValue = "false";
    int threads = getNumProcessors();
    char* threadsEnv = getenv("OPENMM_CPU_THREADS");
    if (threadsEnv != NULL)
        stringstream(threadsEnv) >> threads;
    context.setPlatformData(new PlatformData(&context, context.getSystem(), devicePropValue, blockingPropValue, precisionPropValue, cpuPmePropValue, compilerPropValue, tempPropValue,
            hostCompilerPropValue, pmeStreamPropValue, deterministicForcesValue, threads, NULL));
}

void HipPlatform::linkedContextCreated(ContextImpl& context, ContextImpl& originalContext) const {
    Platform& platform = originalContext.getPlatform();
    string devicePropValue = platform.getPropertyValue(originalContext.getOwner(), HipDeviceIndex());
    string blockingPropValue = platform.getPropertyValue(originalContext.getOwner(), HipUseBlockingSync());
    string precisionPropValue = platform.getPropertyValue(originalContext.getOwner(), HipPrecision());
    string cpuPmePropValue = platform.getPropertyValue(originalContext.getOwner(), HipUseCpuPme());
    string compilerPropValue = platform.getPropertyValue(originalContext.getOwner(), HipCompiler());
    string tempPropValue = platform.getPropertyValue(originalContext.getOwner(), HipTempDirectory());
    string hostCompilerPropValue = platform.getPropertyValue(originalContext.getOwner(), HipHostCompiler());
    string pmeStreamPropValue = platform.getPropertyValue(originalContext.getOwner(), HipDisablePmeStream());
    string deterministicForcesValue = platform.getPropertyValue(originalContext.getOwner(), HipDeterministicForces());
    int threads = reinterpret_cast<PlatformData*>(originalContext.getPlatformData())->threads.getNumThreads();
    context.setPlatformData(new PlatformData(&context, context.getSystem(), devicePropValue, blockingPropValue, precisionPropValue, cpuPmePropValue, compilerPropValue, tempPropValue,
            hostCompilerPropValue, pmeStreamPropValue, deterministicForcesValue, threads, &originalContext));
}

void HipPlatform::contextDestroyed(ContextImpl& context) const {
    PlatformData* data = reinterpret_cast<PlatformData*>(context.getPlatformData());
    delete data;
}

HipPlatform::PlatformData::PlatformData(ContextImpl* context, const System& system, const string& deviceIndexProperty, const string& blockingProperty, const string& precisionProperty,
            const string& cpuPmeProperty, const string& compilerProperty, const string& tempProperty, const string& hostCompilerProperty, const string& pmeStreamProperty,
            const string& deterministicForcesProperty, int numThreads, ContextImpl* originalContext) :
                context(context), removeCM(false), stepCount(0), computeForceCount(0), time(0.0), hasInitializedContexts(false), threads(numThreads) {
    bool blocking = (blockingProperty == "true");
    vector<string> devices;
    size_t searchPos = 0, nextPos;
    while ((nextPos = deviceIndexProperty.find_first_of(", ", searchPos)) != string::npos) {
        devices.push_back(deviceIndexProperty.substr(searchPos, nextPos-searchPos));
        searchPos = nextPos+1;
    }
    devices.push_back(deviceIndexProperty.substr(searchPos));
    PlatformData* originalData = NULL;
    if (originalContext != NULL)
        originalData = reinterpret_cast<PlatformData*>(originalContext->getPlatformData());
    try {
        for (int i = 0; i < (int) devices.size(); i++) {
            if (devices[i].length() > 0) {
                int deviceIndex;
                stringstream(devices[i]) >> deviceIndex;
                contexts.push_back(new HipContext(system, deviceIndex, blocking, precisionProperty, compilerProperty, tempProperty, hostCompilerProperty, *this, (originalData == NULL ? NULL : originalData->contexts[i])));
            }
        }
        if (contexts.size() == 0)
            contexts.push_back(new HipContext(system, -1, blocking, precisionProperty, compilerProperty, tempProperty, hostCompilerProperty, *this, (originalData == NULL ? NULL : originalData->contexts[0])));
    }
    catch (...) {
        // If an exception was thrown, do our best to clean up memory.

        for (int i = 0; i < (int) contexts.size(); i++)
            delete contexts[i];
        throw;
    }
    stringstream deviceIndex, deviceName;
    for (int i = 0; i < (int) contexts.size(); i++) {
        if (i > 0) {
            deviceIndex << ',';
            deviceName << ',';
        }
        deviceIndex << contexts[i]->getDeviceIndex();
        char name[1000];
        CHECK_RESULT(hipDeviceGetName(name, 1000, contexts[i]->getDevice()), "Error querying device name");
        deviceName << name;
    }
#ifdef __HIP_PLATFORM_NVCC__
    // hipDeviceSetLimit is not implemented in HIP
    size_t printfsize;
    hipDeviceGetLimit(&printfsize, CU_LIMIT_PRINTF_FIFO_SIZE);
    cudaDeviceSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, 10*printfsize);
#endif

    useCpuPme = (cpuPmeProperty == "true" && !contexts[0]->getUseDoublePrecision());
    disablePmeStream = (pmeStreamProperty == "true");
    deterministicForces = (deterministicForcesProperty == "true");
    propertyValues[HipPlatform::HipDeviceIndex()] = deviceIndex.str();
    propertyValues[HipPlatform::HipDeviceName()] = deviceName.str();
    propertyValues[HipPlatform::HipUseBlockingSync()] = blocking ? "true" : "false";
    propertyValues[HipPlatform::HipPrecision()] = precisionProperty;
    propertyValues[HipPlatform::HipUseCpuPme()] = useCpuPme ? "true" : "false";
    propertyValues[HipPlatform::HipCompiler()] = compilerProperty;
    propertyValues[HipPlatform::HipTempDirectory()] = tempProperty;
    propertyValues[HipPlatform::HipHostCompiler()] = hostCompilerProperty;
    propertyValues[HipPlatform::HipDisablePmeStream()] = disablePmeStream ? "true" : "false";
    propertyValues[HipPlatform::HipDeterministicForces()] = deterministicForces ? "true" : "false";
    contextEnergy.resize(contexts.size());

    // Determine whether peer-to-peer copying is supported, and enable it if so.

    peerAccessSupported = true;
    for (int i = 1; i < contexts.size(); i++) {
        int canAccess;
        hipDeviceCanAccessPeer(&canAccess, contexts[i]->getDevice(), contexts[0]->getDevice());
        if (!canAccess) {
            peerAccessSupported = false;
            break;
        }
    }
}

HipPlatform::PlatformData::~PlatformData() {
    for (int i = 0; i < (int) contexts.size(); i++)
        delete contexts[i];
}

void HipPlatform::PlatformData::initializeContexts(const System& system) {
    if (hasInitializedContexts)
        return;
    for (int i = 0; i < (int) contexts.size(); i++)
        contexts[i]->initialize();
    hasInitializedContexts = true;
}

void HipPlatform::PlatformData::syncContexts() {
    for (int i = 0; i < (int) contexts.size(); i++)
        contexts[i]->getWorkThread().flush();
}
