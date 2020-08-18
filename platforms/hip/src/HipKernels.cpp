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

#include "HipKernels.h"
#include "HipForceInfo.h"
#include "openmm/Context.h"
#include "openmm/internal/AndersenThermostatImpl.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/CustomCompoundBondForceImpl.h"
#include "openmm/internal/CustomHbondForceImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "openmm/internal/OSRngSeed.h"
#include "HipBondedUtilities.h"
#include "HipExpressionUtilities.h"
#include "HipIntegrationUtilities.h"
#include "HipNonbondedUtilities.h"
#include "HipKernelSources.h"
#include "lepton/CustomFunction.h"
#include "lepton/ExpressionTreeNode.h"
#include "lepton/Operation.h"
#include "lepton/Parser.h"
#include "lepton/ParsedExpression.h"
#include "ReferenceTabulatedFunction.h"
#include "SimTKOpenMMRealType.h"
#include "SimTKOpenMMUtilities.h"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>
#include <assert.h>

using namespace OpenMM;
using namespace std;
using namespace Lepton;

#define CHECK_RESULT(result, prefix) \
    if (result != hipSuccess) { \
        std::stringstream m; \
        m<<prefix<<": "<<HipContext::getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }

static bool isZeroExpression(const Lepton::ParsedExpression& expression) {
    const Lepton::Operation& op = expression.getRootNode().getOperation();
    if (op.getId() != Lepton::Operation::CONSTANT)
        return false;
    return (dynamic_cast<const Lepton::Operation::Constant&>(op).getValue() == 0.0);
}

static bool usesVariable(const Lepton::ExpressionTreeNode& node, const string& variable) {
    const Lepton::Operation& op = node.getOperation();
    if (op.getId() == Lepton::Operation::VARIABLE && op.getName() == variable)
        return true;
    for (auto& child : node.getChildren())
        if (usesVariable(child, variable))
            return true;
    return false;
}

static bool usesVariable(const Lepton::ParsedExpression& expression, const string& variable) {
    return usesVariable(expression.getRootNode(), variable);
}

static pair<ExpressionTreeNode, string> makeVariable(const string& name, const string& value) {
    return make_pair(ExpressionTreeNode(new Operation::Variable(name)), value);
}

static void replaceFunctionsInExpression(map<string, CustomFunction*>& functions, ExpressionProgram& expression) {
    for (int i = 0; i < expression.getNumOperations(); i++) {
        if (expression.getOperation(i).getId() == Operation::CUSTOM) {
            const Operation::Custom& op = dynamic_cast<const Operation::Custom&>(expression.getOperation(i));
            expression.setOperation(i, new Operation::Custom(op.getName(), functions[op.getName()]->clone(), op.getDerivOrder()));
        }
    }
}

void HipCalcForcesAndEnergyKernel::initialize(const System& system) {
}

void HipCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    cu.setForcesValid(true);
    cu.setAsCurrent();
    cu.clearAutoclearBuffers();
    for (auto computation : cu.getPreComputations())
        computation->computeForceAndEnergy(includeForces, includeEnergy, groups);
    HipNonbondedUtilities& nb = cu.getNonbondedUtilities();
    cu.setComputeForceCount(cu.getComputeForceCount()+1);
    nb.prepareInteractions(groups);
    map<string, double>& derivs = cu.getEnergyParamDerivWorkspace();
    for (auto& param : context.getParameters())
        derivs[param.first] = 0;
}

double HipCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups, bool& valid) {
    cu.setAsCurrent();
    cu.getBondedUtilities().computeInteractions(groups);
    cu.getNonbondedUtilities().computeInteractions(groups, includeForces, includeEnergy);
    double sum = 0.0;
    for (auto computation : cu.getPostComputations())
        sum += computation->computeForceAndEnergy(includeForces, includeEnergy, groups);
    cu.getIntegrationUtilities().distributeForcesFromVirtualSites();
    if (includeEnergy)
        sum += cu.reduceEnergy();
    if (!cu.getForcesValid())
        valid = false;
    return sum;
}

void HipUpdateStateDataKernel::initialize(const System& system) {
}

double HipUpdateStateDataKernel::getTime(const ContextImpl& context) const {
    return cu.getTime();
}

void HipUpdateStateDataKernel::setTime(ContextImpl& context, double time) {
    vector<HipContext*>& contexts = cu.getPlatformData().contexts;
    for (auto ctx : contexts)
        ctx->setTime(time);
}

void HipUpdateStateDataKernel::getPositions(ContextImpl& context, vector<Vec3>& positions) {
    cu.setAsCurrent();
    int numParticles = context.getSystem().getNumParticles();
    positions.resize(numParticles);
    vector<float4> posCorrection;
    if (cu.getUseDoublePrecision()) {
        double4* posq = (double4*) cu.getPinnedBuffer();
        cu.getPosq().download(posq);
    }
    else if (cu.getUseMixedPrecision()) {
        float4* posq = (float4*) cu.getPinnedBuffer();
        cu.getPosq().download(posq, false);
        posCorrection.resize(numParticles);
        cu.getPosqCorrection().download(posCorrection);
    }
    else {
        float4* posq = (float4*) cu.getPinnedBuffer();
        cu.getPosq().download(posq);
    }

    // Filling in the output array is done in parallel for speed.

    cu.getPlatformData().threads.execute([&] (ThreadPool& threads, int threadIndex) {
        // Compute the position of each particle to return to the user.  This is done in parallel for speed.

        const vector<int>& order = cu.getAtomIndex();
        int numParticles = cu.getNumAtoms();
        Vec3 boxVectors[3];
        cu.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        int numThreads = threads.getNumThreads();
        int start = threadIndex*numParticles/numThreads;
        int end = (threadIndex+1)*numParticles/numThreads;
        if (cu.getUseDoublePrecision()) {
            double4* posq = (double4*) cu.getPinnedBuffer();
            for (int i = start; i < end; ++i) {
                double4 pos = posq[i];
                mm_int4 offset = cu.getPosCellOffsets()[i];
                positions[order[i]] = Vec3(pos.x, pos.y, pos.z)-boxVectors[0]*offset.x-boxVectors[1]*offset.y-boxVectors[2]*offset.z;
            }
        }
        else if (cu.getUseMixedPrecision()) {
            float4* posq = (float4*) cu.getPinnedBuffer();
            for (int i = start; i < end; ++i) {
                float4 pos1 = posq[i];
                float4 pos2 = posCorrection[i];
                mm_int4 offset = cu.getPosCellOffsets()[i];
                positions[order[i]] = Vec3((double)pos1.x+(double)pos2.x, (double)pos1.y+(double)pos2.y, (double)pos1.z+(double)pos2.z)-boxVectors[0]*offset.x-boxVectors[1]*offset.y-boxVectors[2]*offset.z;
            }
        }
        else {
            float4* posq = (float4*) cu.getPinnedBuffer();
            for (int i = start; i < end; ++i) {
                float4 pos = posq[i];
                mm_int4 offset = cu.getPosCellOffsets()[i];
                positions[order[i]] = Vec3(pos.x, pos.y, pos.z)-boxVectors[0]*offset.x-boxVectors[1]*offset.y-boxVectors[2]*offset.z;
            }
        }
    });
    cu.getPlatformData().threads.waitForThreads();
}

void HipUpdateStateDataKernel::setPositions(ContextImpl& context, const vector<Vec3>& positions) {
    cu.setAsCurrent();
    const vector<int>& order = cu.getAtomIndex();
    int numParticles = context.getSystem().getNumParticles();
    if (cu.getUseDoublePrecision()) {
        double4* posq = (double4*) cu.getPinnedBuffer();
        cu.getPosq().download(posq);
        for (int i = 0; i < numParticles; ++i) {
            double4& pos = posq[i];
            const Vec3& p = positions[order[i]];
            pos.x = p[0];
            pos.y = p[1];
            pos.z = p[2];
        }
        for (int i = numParticles; i < cu.getPaddedNumAtoms(); i++)
            posq[i] = make_double4(0.0, 0.0, 0.0, 0.0);
        cu.getPosq().upload(posq);
    }
    else {
        float4* posq = (float4*) cu.getPinnedBuffer();
        cu.getPosq().download(posq);
        for (int i = 0; i < numParticles; ++i) {
            float4& pos = posq[i];
            const Vec3& p = positions[order[i]];
            pos.x = (float) p[0];
            pos.y = (float) p[1];
            pos.z = (float) p[2];
        }
        for (int i = numParticles; i < cu.getPaddedNumAtoms(); i++)
            posq[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        cu.getPosq().upload(posq);
    }
    if (cu.getUseMixedPrecision()) {
        float4* posCorrection = (float4*) cu.getPinnedBuffer();
        for (int i = 0; i < numParticles; ++i) {
            float4& c = posCorrection[i];
            const Vec3& p = positions[order[i]];
            c.x = (float) (p[0]-(float)p[0]);
            c.y = (float) (p[1]-(float)p[1]);
            c.z = (float) (p[2]-(float)p[2]);
            c.w = 0;
        }
        for (int i = numParticles; i < cu.getPaddedNumAtoms(); i++)
            posCorrection[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        cu.getPosqCorrection().upload(posCorrection);
    }
    for (auto& offset : cu.getPosCellOffsets())
        offset = mm_int4(0, 0, 0, 0);
    cu.reorderAtoms();
}

void HipUpdateStateDataKernel::getVelocities(ContextImpl& context, vector<Vec3>& velocities) {
    cu.setAsCurrent();
    const vector<int>& order = cu.getAtomIndex();
    int numParticles = context.getSystem().getNumParticles();
    velocities.resize(numParticles);
    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        double4* velm = (double4*) cu.getPinnedBuffer();
        cu.getVelm().download(velm);
        for (int i = 0; i < numParticles; ++i) {
            double4 vel = velm[i];
            mm_int4 offset = cu.getPosCellOffsets()[i];
            velocities[order[i]] = Vec3(vel.x, vel.y, vel.z);
        }
    }
    else {
        float4* velm = (float4*) cu.getPinnedBuffer();
        cu.getVelm().download(velm);
        for (int i = 0; i < numParticles; ++i) {
            float4 vel = velm[i];
            mm_int4 offset = cu.getPosCellOffsets()[i];
            velocities[order[i]] = Vec3(vel.x, vel.y, vel.z);
        }
    }
}

void HipUpdateStateDataKernel::setVelocities(ContextImpl& context, const vector<Vec3>& velocities) {
    cu.setAsCurrent();
    const vector<int>& order = cu.getAtomIndex();
    int numParticles = context.getSystem().getNumParticles();
    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        double4* velm = (double4*) cu.getPinnedBuffer();
        cu.getVelm().download(velm);
        for (int i = 0; i < numParticles; ++i) {
            double4& vel = velm[i];
            const Vec3& p = velocities[order[i]];
            vel.x = p[0];
            vel.y = p[1];
            vel.z = p[2];
        }
        for (int i = numParticles; i < cu.getPaddedNumAtoms(); i++)
            velm[i] = make_double4(0.0, 0.0, 0.0, 0.0);
        cu.getVelm().upload(velm);
    }
    else {
        float4* velm = (float4*) cu.getPinnedBuffer();
        cu.getVelm().download(velm);
        for (int i = 0; i < numParticles; ++i) {
            float4& vel = velm[i];
            const Vec3& p = velocities[order[i]];
            vel.x = p[0];
            vel.y = p[1];
            vel.z = p[2];
        }
        for (int i = numParticles; i < cu.getPaddedNumAtoms(); i++)
            velm[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        cu.getVelm().upload(velm);
    }
}

void HipUpdateStateDataKernel::getForces(ContextImpl& context, vector<Vec3>& forces) {
    cu.setAsCurrent();
    long long* force = (long long*) cu.getPinnedBuffer();
    cu.getForce().download(force);
    const vector<int>& order = cu.getAtomIndex();
    int numParticles = context.getSystem().getNumParticles();
    int paddedNumParticles = cu.getPaddedNumAtoms();
    forces.resize(numParticles);
    double scale = 1.0/(double) 0x100000000LL;
    for (int i = 0; i < numParticles; ++i)
        forces[order[i]] = Vec3(scale*force[i], scale*force[i+paddedNumParticles], scale*force[i+paddedNumParticles*2]);
}

void HipUpdateStateDataKernel::getEnergyParameterDerivatives(ContextImpl& context, map<string, double>& derivs) {
    const vector<string>& paramDerivNames = cu.getEnergyParamDerivNames();
    int numDerivs = paramDerivNames.size();
    if (numDerivs == 0)
        return;
    derivs = cu.getEnergyParamDerivWorkspace();
    HipArray& derivArray = cu.getEnergyParamDerivBuffer();
    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        vector<double> derivBuffers;
        derivArray.download(derivBuffers);
        for (int i = numDerivs; i < derivArray.getSize(); i += numDerivs)
            for (int j = 0; j < numDerivs; j++)
                derivBuffers[j] += derivBuffers[i+j];
        for (int i = 0; i < numDerivs; i++)
            derivs[paramDerivNames[i]] += derivBuffers[i];
    }
    else {
        vector<float> derivBuffers;
        derivArray.download(derivBuffers);
        for (int i = numDerivs; i < derivArray.getSize(); i += numDerivs)
            for (int j = 0; j < numDerivs; j++)
                derivBuffers[j] += derivBuffers[i+j];
        for (int i = 0; i < numDerivs; i++)
            derivs[paramDerivNames[i]] += derivBuffers[i];
    }
}

void HipUpdateStateDataKernel::getPeriodicBoxVectors(ContextImpl& context, Vec3& a, Vec3& b, Vec3& c) const {
    cu.getPeriodicBoxVectors(a, b, c);
}

void HipUpdateStateDataKernel::setPeriodicBoxVectors(ContextImpl& context, const Vec3& a, const Vec3& b, const Vec3& c) {
    vector<HipContext*>& contexts = cu.getPlatformData().contexts;

    // If any particles have been wrapped to the first periodic box, we need to unwrap them
    // to avoid changing their positions.

    vector<Vec3> positions;
    for (auto& offset : cu.getPosCellOffsets()) {
        if (offset.x != 0 || offset.y != 0 || offset.z != 0) {
            getPositions(context, positions);
            break;
        }
    }

    // Update the vectors.

    for (auto ctx : contexts)
        ctx->setPeriodicBoxVectors(a, b, c);
    if (positions.size() > 0)
        setPositions(context, positions);
}

void HipUpdateStateDataKernel::createCheckpoint(ContextImpl& context, ostream& stream) {
    cu.setAsCurrent();
    int version = 3;
    stream.write((char*) &version, sizeof(int));
    int precision = (cu.getUseDoublePrecision() ? 2 : cu.getUseMixedPrecision() ? 1 : 0);
    stream.write((char*) &precision, sizeof(int));
    double time = cu.getTime();
    stream.write((char*) &time, sizeof(double));
    int stepCount = cu.getStepCount();
    stream.write((char*) &stepCount, sizeof(int));
    int stepsSinceReorder = cu.getStepsSinceReorder();
    stream.write((char*) &stepsSinceReorder, sizeof(int));
    char* buffer = (char*) cu.getPinnedBuffer();
    cu.getPosq().download(buffer);
    stream.write(buffer, cu.getPosq().getSize()*cu.getPosq().getElementSize());
    if (cu.getUseMixedPrecision()) {
        cu.getPosqCorrection().download(buffer);
        stream.write(buffer, cu.getPosqCorrection().getSize()*cu.getPosqCorrection().getElementSize());
    }
    cu.getVelm().download(buffer);
    stream.write(buffer, cu.getVelm().getSize()*cu.getVelm().getElementSize());
    stream.write((char*) &cu.getAtomIndex()[0], sizeof(int)*cu.getAtomIndex().size());
    stream.write((char*) &cu.getPosCellOffsets()[0], sizeof(int4)*cu.getPosCellOffsets().size());
    Vec3 boxVectors[3];
    cu.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    stream.write((char*) boxVectors, 3*sizeof(Vec3));
    cu.getIntegrationUtilities().createCheckpoint(stream);
    SimTKOpenMMUtilities::createCheckpoint(stream);
}

void HipUpdateStateDataKernel::loadCheckpoint(ContextImpl& context, istream& stream) {
    cu.setAsCurrent();
    int version;
    stream.read((char*) &version, sizeof(int));
    if (version != 3)
        throw OpenMMException("Checkpoint was created with a different version of OpenMM");
    int precision;
    stream.read((char*) &precision, sizeof(int));
    int expectedPrecision = (cu.getUseDoublePrecision() ? 2 : cu.getUseMixedPrecision() ? 1 : 0);
    if (precision != expectedPrecision)
        throw OpenMMException("Checkpoint was created with a different numeric precision");
    double time;
    stream.read((char*) &time, sizeof(double));
    int stepCount, stepsSinceReorder;
    stream.read((char*) &stepCount, sizeof(int));
    stream.read((char*) &stepsSinceReorder, sizeof(int));
    vector<HipContext*>& contexts = cu.getPlatformData().contexts;
    for (auto ctx : contexts) {
        ctx->setTime(time);
        ctx->setStepCount(stepCount);
        ctx->setStepsSinceReorder(stepsSinceReorder);
    }
    char* buffer = (char*) cu.getPinnedBuffer();
    stream.read(buffer, cu.getPosq().getSize()*cu.getPosq().getElementSize());
    cu.getPosq().upload(buffer);
    if (cu.getUseMixedPrecision()) {
        stream.read(buffer, cu.getPosqCorrection().getSize()*cu.getPosqCorrection().getElementSize());
        cu.getPosqCorrection().upload(buffer);
    }
    stream.read(buffer, cu.getVelm().getSize()*cu.getVelm().getElementSize());
    cu.getVelm().upload(buffer);
    stream.read((char*) &cu.getAtomIndex()[0], sizeof(int)*cu.getAtomIndex().size());
    cu.getAtomIndexArray().upload(cu.getAtomIndex());
    stream.read((char*) &cu.getPosCellOffsets()[0], sizeof(int4)*cu.getPosCellOffsets().size());
    Vec3 boxVectors[3];
    stream.read((char*) &boxVectors, 3*sizeof(Vec3));
    for (auto ctx : contexts)
        ctx->setPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    cu.getIntegrationUtilities().loadCheckpoint(stream);
    SimTKOpenMMUtilities::loadCheckpoint(stream);
    for (auto listener : cu.getReorderListeners())
        listener->execute();
}

void HipApplyConstraintsKernel::initialize(const System& system) {
}

void HipApplyConstraintsKernel::apply(ContextImpl& context, double tol) {
    cu.setAsCurrent();
    if (!hasInitializedKernel) {
        hasInitializedKernel = true;
        map<string, string> defines;
        hipModule_t module = cu.createModule(HipKernelSources::constraints, defines);
        applyDeltasKernel = cu.getKernel(module, "applyPositionDeltas");
    }
    HipIntegrationUtilities& integration = cu.getIntegrationUtilities();
    cu.clearBuffer(integration.getPosDelta());
    integration.applyConstraints(tol);
    hipDeviceptr_t posCorrection = (cu.getUseMixedPrecision() ? cu.getPosqCorrection().getDevicePointer() : 0);
    int numAtoms = cu.getNumAtoms();
    void* args[] = {&numAtoms, &cu.getPosq().getDevicePointer(), &posCorrection, &cu.getIntegrationUtilities().getPosDelta().getDevicePointer()};
    cu.executeKernel(applyDeltasKernel, args, cu.getNumAtoms());
    integration.computeVirtualSites();
}

void HipApplyConstraintsKernel::applyToVelocities(ContextImpl& context, double tol) {
    cu.getIntegrationUtilities().applyVelocityConstraints(tol);
}

void HipVirtualSitesKernel::initialize(const System& system) {
}

void HipVirtualSitesKernel::computePositions(ContextImpl& context) {
    cu.getIntegrationUtilities().computeVirtualSites();
}

class HipCalcNonbondedForceKernel::ForceInfo : public HipForceInfo {
public:
    ForceInfo(const NonbondedForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double charge1, charge2, sigma1, sigma2, epsilon1, epsilon2;
        force.getParticleParameters(particle1, charge1, sigma1, epsilon1);
        force.getParticleParameters(particle2, charge2, sigma2, epsilon2);
        return (charge1 == charge2 && sigma1 == sigma2 && epsilon1 == epsilon2);
    }
    int getNumParticleGroups() {
        return force.getNumExceptions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(index, particle1, particle2, chargeProd, sigma, epsilon);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2;
        double chargeProd1, chargeProd2, sigma1, sigma2, epsilon1, epsilon2;
        force.getExceptionParameters(group1, particle1, particle2, chargeProd1, sigma1, epsilon1);
        force.getExceptionParameters(group2, particle1, particle2, chargeProd2, sigma2, epsilon2);
        return (chargeProd1 == chargeProd2 && sigma1 == sigma2 && epsilon1 == epsilon2);
    }
private:
    const NonbondedForce& force;
};

class HipCalcNonbondedForceKernel::PmeIO : public CalcPmeReciprocalForceKernel::IO {
public:
    PmeIO(HipContext& cu, hipFunction_t addForcesKernel) : cu(cu), addForcesKernel(addForcesKernel) {
        forceTemp.initialize<float4>(cu, cu.getNumAtoms(), "PmeForce");
    }
    float* getPosq() {
        cu.setAsCurrent();
        cu.getPosq().download(posq);
        return (float*) &posq[0];
    }
    void setForce(float* force) {
        forceTemp.upload(force);
        void* args[] = {&forceTemp.getDevicePointer(), &cu.getForce().getDevicePointer()};
        cu.executeKernel(addForcesKernel, args, cu.getNumAtoms());
    }
private:
    HipContext& cu;
    vector<float4> posq;
    HipArray forceTemp;
    hipFunction_t addForcesKernel;
};

class HipCalcNonbondedForceKernel::PmePreComputation : public HipContext::ForcePreComputation {
public:
    PmePreComputation(HipContext& cu, Kernel& pme, CalcPmeReciprocalForceKernel::IO& io) : cu(cu), pme(pme), io(io) {
    }
    void computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        Vec3 boxVectors[3] = {Vec3(cu.getPeriodicBoxSize().x, 0, 0), Vec3(0, cu.getPeriodicBoxSize().y, 0), Vec3(0, 0, cu.getPeriodicBoxSize().z)};
        pme.getAs<CalcPmeReciprocalForceKernel>().beginComputation(io, boxVectors, includeEnergy);
    }
private:
    HipContext& cu;
    Kernel pme;
    CalcPmeReciprocalForceKernel::IO& io;
};

class HipCalcNonbondedForceKernel::PmePostComputation : public HipContext::ForcePostComputation {
public:
    PmePostComputation(Kernel& pme, CalcPmeReciprocalForceKernel::IO& io) : pme(pme), io(io) {
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        return pme.getAs<CalcPmeReciprocalForceKernel>().finishComputation(io);
    }
private:
    Kernel pme;
    CalcPmeReciprocalForceKernel::IO& io;
};

class HipCalcNonbondedForceKernel::SyncStreamPreComputation : public HipContext::ForcePreComputation {
public:
    SyncStreamPreComputation(HipContext& cu, hipStream_t stream, hipEvent_t event, int forceGroup) : cu(cu), stream(stream), event(event), forceGroup(forceGroup) {
    }
    void computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            hipEventRecord(event, cu.getCurrentStream());
            hipStreamWaitEvent(stream, event, 0);
        }
    }
private:
    HipContext& cu;
    hipStream_t stream;
    hipEvent_t event;
    int forceGroup;
};

class HipCalcNonbondedForceKernel::SyncStreamPostComputation : public HipContext::ForcePostComputation {
public:
    SyncStreamPostComputation(HipContext& cu, hipEvent_t event, hipFunction_t addEnergyKernel, HipArray& pmeEnergyBuffer, int forceGroup) : cu(cu), event(event),
            addEnergyKernel(addEnergyKernel), pmeEnergyBuffer(pmeEnergyBuffer), forceGroup(forceGroup) {
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            hipStreamWaitEvent(cu.getCurrentStream(), event, 0);
            if (includeEnergy) {
                int bufferSize = pmeEnergyBuffer.getSize();
                void* args[] = {&pmeEnergyBuffer.getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(), &bufferSize};
                cu.executeKernel(addEnergyKernel, args, bufferSize);
            }
        }
        return 0.0;
    }
private:
    HipContext& cu;
    hipEvent_t event;
    hipFunction_t addEnergyKernel;
    HipArray& pmeEnergyBuffer;
    int forceGroup;
};

HipCalcNonbondedForceKernel::~HipCalcNonbondedForceKernel() {
    cu.setAsCurrent();
    if (sort != NULL)
        delete sort;
    if (fft != NULL)
        delete fft;
    if (dispersionFft != NULL)
        delete dispersionFft;
    if (pmeio != NULL)
        delete pmeio;
    if (hasInitializedFFT) {
        if (useHipFFT) {
            hipfftDestroy(fftForward);
            hipfftDestroy(fftBackward);
            if (doLJPME) {
                hipfftDestroy(dispersionFftForward);
                hipfftDestroy(dispersionFftBackward);
            }
        }
        if (usePmeStream) {
            hipStreamDestroy(pmeStream);
            hipEventDestroy(pmeSyncEvent);
            hipEventDestroy(paramsSyncEvent);
        }
    }
}

void HipCalcNonbondedForceKernel::initialize(const System& system, const NonbondedForce& force) {
    cu.setAsCurrent();
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() && &system.getForce(forceIndex) != &force; ++forceIndex)
        ;
    string prefix = "nonbonded"+cu.intToString(forceIndex)+"_";

    // Identify which exceptions are 1-4 interactions.

    set<int> exceptionsWithOffsets;
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        exceptionsWithOffsets.insert(exception);
    }
    vector<pair<int, int> > exclusions;
    vector<int> exceptions;
    map<int, int> exceptionIndex;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        exclusions.push_back(pair<int, int>(particle1, particle2));
        if (chargeProd != 0.0 || epsilon != 0.0 || exceptionsWithOffsets.find(i) != exceptionsWithOffsets.end()) {
            exceptionIndex[i] = exceptions.size();
            exceptions.push_back(i);
        }
    }

    // Initialize nonbonded interactions.

    int numParticles = force.getNumParticles();
    vector<float4> baseParticleParamVec(cu.getPaddedNumAtoms(), make_float4(0, 0, 0, 0));
    vector<vector<int> > exclusionList(numParticles);
    hasCoulomb = false;
    hasLJ = false;
    for (int i = 0; i < numParticles; i++) {
        double charge, sigma, epsilon;
        force.getParticleParameters(i, charge, sigma, epsilon);
        baseParticleParamVec[i] = make_float4(charge, sigma, epsilon, 0);
        exclusionList[i].push_back(i);
        if (charge != 0.0)
            hasCoulomb = true;
        if (epsilon != 0.0)
            hasLJ = true;
    }
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string param;
        int particle;
        double charge, sigma, epsilon;
        force.getParticleParameterOffset(i, param, particle, charge, sigma, epsilon);
        if (charge != 0.0)
            hasCoulomb = true;
        if (epsilon != 0.0)
            hasLJ = true;
    }
    for (auto exclusion : exclusions) {
        exclusionList[exclusion.first].push_back(exclusion.second);
        exclusionList[exclusion.second].push_back(exclusion.first);
    }
    nonbondedMethod = CalcNonbondedForceKernel::NonbondedMethod(force.getNonbondedMethod());
    bool useCutoff = (nonbondedMethod != NoCutoff);
    bool usePeriodic = (nonbondedMethod != NoCutoff && nonbondedMethod != CutoffNonPeriodic);
    doLJPME = (nonbondedMethod == LJPME && hasLJ);
    usePosqCharges = hasCoulomb ? cu.requestPosqCharges() : false;

    map<string, string> defines;
    defines["HAS_COULOMB"] = (hasCoulomb ? "1" : "0");
    defines["HAS_LENNARD_JONES"] = (hasLJ ? "1" : "0");
    defines["USE_LJ_SWITCH"] = (useCutoff && force.getUseSwitchingFunction() ? "1" : "0");
    if (useCutoff) {
        // Compute the reaction field constants.

        double reactionFieldK = pow(force.getCutoffDistance(), -3.0)*(force.getReactionFieldDielectric()-1.0)/(2.0*force.getReactionFieldDielectric()+1.0);
        double reactionFieldC = (1.0 / force.getCutoffDistance())*(3.0*force.getReactionFieldDielectric())/(2.0*force.getReactionFieldDielectric()+1.0);
        defines["REACTION_FIELD_K"] = cu.doubleToString(reactionFieldK);
        defines["REACTION_FIELD_C"] = cu.doubleToString(reactionFieldC);

        // Compute the switching coefficients.

        if (force.getUseSwitchingFunction()) {
            defines["LJ_SWITCH_CUTOFF"] = cu.doubleToString(force.getSwitchingDistance());
            defines["LJ_SWITCH_C3"] = cu.doubleToString(10/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 3.0));
            defines["LJ_SWITCH_C4"] = cu.doubleToString(15/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 4.0));
            defines["LJ_SWITCH_C5"] = cu.doubleToString(6/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 5.0));
        }
    }
    if (force.getUseDispersionCorrection() && cu.getContextIndex() == 0 && !doLJPME)
        dispersionCoefficient = NonbondedForceImpl::calcDispersionCorrection(system, force);
    else
        dispersionCoefficient = 0.0;
    alpha = 0;
    ewaldSelfEnergy = 0.0;
    map<string, string> paramsDefines;
    hasOffsets = (force.getNumParticleParameterOffsets() > 0 || force.getNumExceptionParameterOffsets() > 0);
    if (hasOffsets)
        paramsDefines["HAS_OFFSETS"] = "1";
    if (usePosqCharges)
        paramsDefines["USE_POSQ_CHARGES"] = "1";
    if (nonbondedMethod == Ewald) {
        // Compute the Ewald parameters.

        int kmaxx, kmaxy, kmaxz;
        NonbondedForceImpl::calcEwaldParameters(system, force, alpha, kmaxx, kmaxy, kmaxz);
        defines["EWALD_ALPHA"] = cu.doubleToString(alpha);
        defines["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
        defines["USE_EWALD"] = "1";
        if (cu.getContextIndex() == 0) {
            paramsDefines["INCLUDE_EWALD"] = "1";
            paramsDefines["EWALD_SELF_ENERGY_SCALE"] = cu.doubleToString(ONE_4PI_EPS0*alpha/sqrt(M_PI));
            for (int i = 0; i < numParticles; i++)
                ewaldSelfEnergy -= baseParticleParamVec[i].x*baseParticleParamVec[i].x*ONE_4PI_EPS0*alpha/sqrt(M_PI);

            // Create the reciprocal space kernels.

            map<string, string> replacements;
            replacements["NUM_ATOMS"] = cu.intToString(numParticles);
            replacements["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
            replacements["KMAX_X"] = cu.intToString(kmaxx);
            replacements["KMAX_Y"] = cu.intToString(kmaxy);
            replacements["KMAX_Z"] = cu.intToString(kmaxz);
            replacements["EXP_COEFFICIENT"] = cu.doubleToString(-1.0/(4.0*alpha*alpha));
            replacements["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0);
            replacements["M_PI"] = cu.doubleToString(M_PI);
            hipModule_t module = cu.createModule(HipKernelSources::vectorOps+HipKernelSources::ewald, replacements);
            ewaldSumsKernel = cu.getKernel(module, "calculateEwaldCosSinSums");
            ewaldForcesKernel = cu.getKernel(module, "calculateEwaldForces");
            int elementSize = (cu.getUseDoublePrecision() ? sizeof(double2) : sizeof(float2));
            cosSinSums.initialize(cu, (2*kmaxx-1)*(2*kmaxy-1)*(2*kmaxz-1), elementSize, "cosSinSums");
        }
    }
    else if (((nonbondedMethod == PME || nonbondedMethod == LJPME) && hasCoulomb) || doLJPME) {
        // Compute the PME parameters.

        NonbondedForceImpl::calcPMEParameters(system, force, alpha, gridSizeX, gridSizeY, gridSizeZ, false);
        gridSizeX = HipFFT3D::findLegalDimension(gridSizeX);
        gridSizeY = HipFFT3D::findLegalDimension(gridSizeY);
        gridSizeZ = HipFFT3D::findLegalDimension(gridSizeZ);
        if (doLJPME) {
            NonbondedForceImpl::calcPMEParameters(system, force, dispersionAlpha, dispersionGridSizeX,
                                                  dispersionGridSizeY, dispersionGridSizeZ, true);
            dispersionGridSizeX = HipFFT3D::findLegalDimension(dispersionGridSizeX);
            dispersionGridSizeY = HipFFT3D::findLegalDimension(dispersionGridSizeY);
            dispersionGridSizeZ = HipFFT3D::findLegalDimension(dispersionGridSizeZ);
        }

        defines["EWALD_ALPHA"] = cu.doubleToString(alpha);
        defines["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
        defines["USE_EWALD"] = "1";
        defines["DO_LJPME"] = doLJPME ? "1" : "0";
        if (doLJPME)
            defines["EWALD_DISPERSION_ALPHA"] = cu.doubleToString(dispersionAlpha);
        if (cu.getContextIndex() == 0) {
            paramsDefines["INCLUDE_EWALD"] = "1";
            paramsDefines["EWALD_SELF_ENERGY_SCALE"] = cu.doubleToString(ONE_4PI_EPS0*alpha/sqrt(M_PI));
            for (int i = 0; i < numParticles; i++)
                ewaldSelfEnergy -= baseParticleParamVec[i].x*baseParticleParamVec[i].x*ONE_4PI_EPS0*alpha/sqrt(M_PI);
            if (doLJPME) {
                paramsDefines["INCLUDE_LJPME"] = "1";
                paramsDefines["LJPME_SELF_ENERGY_SCALE"] = cu.doubleToString(pow(dispersionAlpha, 6)/3.0);
                for (int i = 0; i < numParticles; i++)
                    ewaldSelfEnergy += baseParticleParamVec[i].z*pow(baseParticleParamVec[i].y*dispersionAlpha, 6)/3.0;
            }
            char deviceName[100];
            hipDeviceGetName(deviceName, 100, cu.getDevice());
            usePmeStream = (!cu.getPlatformData().disablePmeStream && string(deviceName) != "GeForce GTX 980"); // Using a separate stream is slower on GTX 980
            map<string, string> pmeDefines;
            pmeDefines["PME_ORDER"] = cu.intToString(PmeOrder);
            pmeDefines["NUM_ATOMS"] = cu.intToString(numParticles);
            pmeDefines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
            pmeDefines["RECIP_EXP_FACTOR"] = cu.doubleToString(M_PI*M_PI/(alpha*alpha));
            pmeDefines["GRID_SIZE_X"] = cu.intToString(gridSizeX);
            pmeDefines["GRID_SIZE_Y"] = cu.intToString(gridSizeY);
            pmeDefines["GRID_SIZE_Z"] = cu.intToString(gridSizeZ);
            pmeDefines["EPSILON_FACTOR"] = cu.doubleToString(sqrt(ONE_4PI_EPS0));
            pmeDefines["M_PI"] = cu.doubleToString(M_PI);
            if (cu.getUseDoublePrecision())
                pmeDefines["USE_DOUBLE_PRECISION"] = "1";
            if (usePmeStream)
                pmeDefines["USE_PME_STREAM"] = "1";
            if (cu.getPlatformData().deterministicForces)
                pmeDefines["USE_DETERMINISTIC_FORCES"] = "1";
            map<string, string> replacements;
            replacements["CHARGE"] = (usePosqCharges ? "pos.w" : "charges[atom]");
            hipModule_t module = cu.createModule(HipKernelSources::vectorOps+cu.replaceStrings(HipKernelSources::pme, replacements), pmeDefines);
            if (cu.getPlatformData().useCpuPme && !doLJPME && usePosqCharges) {
                // Create the CPU PME kernel.

                try {
                    cpuPme = getPlatform().createKernel(CalcPmeReciprocalForceKernel::Name(), *cu.getPlatformData().context);
                    cpuPme.getAs<CalcPmeReciprocalForceKernel>().initialize(gridSizeX, gridSizeY, gridSizeZ, numParticles, alpha, cu.getPlatformData().deterministicForces);
                    hipFunction_t addForcesKernel = cu.getKernel(module, "addForces");
                    pmeio = new PmeIO(cu, addForcesKernel);
                    cu.addPreComputation(new PmePreComputation(cu, cpuPme, *pmeio));
                    cu.addPostComputation(new PmePostComputation(cpuPme, *pmeio));
                }
                catch (OpenMMException& ex) {
                    // The CPU PME plugin isn't available.
                }
            }
            if (pmeio == NULL) {
                pmeGridIndexKernel = cu.getKernel(module, "findAtomGridIndex");
                pmeSpreadChargeKernel = cu.getKernel(module, "gridSpreadCharge");
                pmeConvolutionKernel = cu.getKernel(module, "reciprocalConvolution");
                pmeInterpolateForceKernel = cu.getKernel(module, "gridInterpolateForce");
                pmeEvalEnergyKernel = cu.getKernel(module, "gridEvaluateEnergy");
                pmeFinishSpreadChargeKernel = cu.getKernel(module, "finishSpreadCharge");
                hipFuncSetCacheConfig(pmeSpreadChargeKernel, hipFuncCachePreferL1);
                hipFuncSetCacheConfig(pmeInterpolateForceKernel, hipFuncCachePreferL1);
                if (doLJPME) {
                    pmeDefines["EWALD_ALPHA"] = cu.doubleToString(dispersionAlpha);
                    pmeDefines["GRID_SIZE_X"] = cu.intToString(dispersionGridSizeX);
                    pmeDefines["GRID_SIZE_Y"] = cu.intToString(dispersionGridSizeY);
                    pmeDefines["GRID_SIZE_Z"] = cu.intToString(dispersionGridSizeZ);
                    pmeDefines["RECIP_EXP_FACTOR"] = cu.doubleToString(M_PI*M_PI/(dispersionAlpha*dispersionAlpha));
                    pmeDefines["USE_LJPME"] = "1";
                    pmeDefines["CHARGE_FROM_SIGEPS"] = "1";
                    double invRCut6 = pow(force.getCutoffDistance(), -6);
                    double dalphaR = dispersionAlpha * force.getCutoffDistance();
                    double dar2 = dalphaR*dalphaR;
                    double dar4 = dar2*dar2;
                    double multShift6 = -invRCut6*(1.0 - exp(-dar2) * (1.0 + dar2 + 0.5*dar4));
                    defines["INVCUT6"] = cu.doubleToString(invRCut6);
                    defines["MULTSHIFT6"] = cu.doubleToString(multShift6);
                    module = cu.createModule(HipKernelSources::vectorOps+HipKernelSources::pme, pmeDefines);
                    pmeDispersionFinishSpreadChargeKernel = cu.getKernel(module, "finishSpreadCharge");
                    pmeDispersionGridIndexKernel = cu.getKernel(module, "findAtomGridIndex");
                    pmeDispersionSpreadChargeKernel = cu.getKernel(module, "gridSpreadCharge");
                    pmeDispersionConvolutionKernel = cu.getKernel(module, "reciprocalConvolution");
                    pmeEvalDispersionEnergyKernel = cu.getKernel(module, "gridEvaluateEnergy");
                    pmeInterpolateDispersionForceKernel = cu.getKernel(module, "gridInterpolateForce");
                    hipFuncSetCacheConfig(pmeDispersionSpreadChargeKernel, hipFuncCachePreferL1);
                }

                // Create required data structures.

                int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
                int roundedZSize = PmeOrder*(int) ceil(gridSizeZ/(double) PmeOrder);
                int gridElements = gridSizeX*gridSizeY*roundedZSize;
                if (doLJPME) {
                    roundedZSize = PmeOrder*(int) ceil(dispersionGridSizeZ/(double) PmeOrder);
                    gridElements = max(gridElements, dispersionGridSizeX*dispersionGridSizeY*roundedZSize);
                }
                pmeGrid1.initialize(cu, gridElements, 2*elementSize, "pmeGrid1");
                pmeGrid2.initialize(cu, gridElements, 2*elementSize, "pmeGrid2");
                cu.addAutoclearBuffer(pmeGrid2);
                pmeBsplineModuliX.initialize(cu, gridSizeX, elementSize, "pmeBsplineModuliX");
                pmeBsplineModuliY.initialize(cu, gridSizeY, elementSize, "pmeBsplineModuliY");
                pmeBsplineModuliZ.initialize(cu, gridSizeZ, elementSize, "pmeBsplineModuliZ");
                if (doLJPME) {
                    pmeDispersionBsplineModuliX.initialize(cu, dispersionGridSizeX, elementSize, "pmeDispersionBsplineModuliX");
                    pmeDispersionBsplineModuliY.initialize(cu, dispersionGridSizeY, elementSize, "pmeDispersionBsplineModuliY");
                    pmeDispersionBsplineModuliZ.initialize(cu, dispersionGridSizeZ, elementSize, "pmeDispersionBsplineModuliZ");
                }
                pmeAtomGridIndex.initialize<int2>(cu, numParticles, "pmeAtomGridIndex");
                int energyElementSize = (cu.getUseDoublePrecision() || cu.getUseMixedPrecision() ? sizeof(double) : sizeof(float));
                pmeEnergyBuffer.initialize(cu, cu.getNumThreadBlocks()*HipContext::ThreadBlockSize, energyElementSize, "pmeEnergyBuffer");
                cu.clearBuffer(pmeEnergyBuffer);
                sort = new HipSort(cu, new SortTrait(), cu.getNumAtoms());
                int cufftVersion;
                hipfftGetVersion(&cufftVersion);
                useHipFFT = (cufftVersion >= 7050); // There was a critical bug in version 7.0
                if (useHipFFT) {
                    hipfftResult result = hipfftPlan3d(&fftForward, gridSizeX, gridSizeY, gridSizeZ, cu.getUseDoublePrecision() ? HIPFFT_D2Z : HIPFFT_R2C);
                    if (result != HIPFFT_SUCCESS)
                        throw OpenMMException("Error initializing FFT: "+cu.intToString(result));
                    result = hipfftPlan3d(&fftBackward, gridSizeX, gridSizeY, gridSizeZ, cu.getUseDoublePrecision() ? HIPFFT_Z2D : HIPFFT_C2R);
                    if (result != HIPFFT_SUCCESS)
                        throw OpenMMException("Error initializing FFT: "+cu.intToString(result));
                    if (doLJPME) {
                        result = hipfftPlan3d(&dispersionFftForward, dispersionGridSizeX, dispersionGridSizeY,
                                                dispersionGridSizeZ, cu.getUseDoublePrecision() ? HIPFFT_D2Z : HIPFFT_R2C);
                        if (result != HIPFFT_SUCCESS)
                            throw OpenMMException("Error initializing disperison FFT: "+cu.intToString(result));
                        result = hipfftPlan3d(&dispersionFftBackward, dispersionGridSizeX, dispersionGridSizeY,
                                             dispersionGridSizeZ, cu.getUseDoublePrecision() ? HIPFFT_Z2D : HIPFFT_C2R);
                        if (result != HIPFFT_SUCCESS)
                            throw OpenMMException("Error initializing disperison FFT: "+cu.intToString(result));
                    }
                }
                else {
                    fft = new HipFFT3D(cu, gridSizeX, gridSizeY, gridSizeZ, true);
                    if (doLJPME)
                        dispersionFft = new HipFFT3D(cu, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, true);
                }

                // Prepare for doing PME on its own stream.

                if (usePmeStream) {
                    hipStreamCreateWithFlags(&pmeStream, hipStreamNonBlocking);
                    if (useHipFFT) {
                        hipfftSetStream(fftForward, pmeStream);
                        hipfftSetStream(fftBackward, pmeStream);
                        if (doLJPME) {
                            hipfftSetStream(dispersionFftForward, pmeStream);
                            hipfftSetStream(dispersionFftBackward, pmeStream);
                        }
                    }
                    CHECK_RESULT(hipEventCreateWithFlags(&pmeSyncEvent, hipEventDisableTiming), "Error creating event for NonbondedForce");
                    CHECK_RESULT(hipEventCreateWithFlags(&paramsSyncEvent, hipEventDisableTiming), "Error creating event for NonbondedForce");
                    int recipForceGroup = force.getReciprocalSpaceForceGroup();
                    if (recipForceGroup < 0)
                        recipForceGroup = force.getForceGroup();
                    cu.addPreComputation(new SyncStreamPreComputation(cu, pmeStream, pmeSyncEvent, recipForceGroup));
                    cu.addPostComputation(new SyncStreamPostComputation(cu, pmeSyncEvent, cu.getKernel(module, "addEnergy"), pmeEnergyBuffer, recipForceGroup));
                }
                hasInitializedFFT = true;

                // Initialize the b-spline moduli.

                for (int grid = 0; grid < 2; grid++) {
                    int xsize, ysize, zsize;
                    HipArray *xmoduli, *ymoduli, *zmoduli;
                    if (grid == 0) {
                        xsize = gridSizeX;
                        ysize = gridSizeY;
                        zsize = gridSizeZ;
                        xmoduli = &pmeBsplineModuliX;
                        ymoduli = &pmeBsplineModuliY;
                        zmoduli = &pmeBsplineModuliZ;
                    }
                    else {
                        if (!doLJPME)
                            continue;
                        xsize = dispersionGridSizeX;
                        ysize = dispersionGridSizeY;
                        zsize = dispersionGridSizeZ;
                        xmoduli = &pmeDispersionBsplineModuliX;
                        ymoduli = &pmeDispersionBsplineModuliY;
                        zmoduli = &pmeDispersionBsplineModuliZ;
                    }
                    int maxSize = max(max(xsize, ysize), zsize);
                    vector<double> data(PmeOrder);
                    vector<double> ddata(PmeOrder);
                    vector<double> bsplines_data(maxSize);
                    data[PmeOrder-1] = 0.0;
                    data[1] = 0.0;
                    data[0] = 1.0;
                    for (int i = 3; i < PmeOrder; i++) {
                        double div = 1.0/(i-1.0);
                        data[i-1] = 0.0;
                        for (int j = 1; j < (i-1); j++)
                            data[i-j-1] = div*(j*data[i-j-2]+(i-j)*data[i-j-1]);
                        data[0] = div*data[0];
                    }

                    // Differentiate.

                    ddata[0] = -data[0];
                    for (int i = 1; i < PmeOrder; i++)
                        ddata[i] = data[i-1]-data[i];
                    double div = 1.0/(PmeOrder-1);
                    data[PmeOrder-1] = 0.0;
                    for (int i = 1; i < (PmeOrder-1); i++)
                        data[PmeOrder-i-1] = div*(i*data[PmeOrder-i-2]+(PmeOrder-i)*data[PmeOrder-i-1]);
                    data[0] = div*data[0];
                    for (int i = 0; i < maxSize; i++)
                        bsplines_data[i] = 0.0;
                    for (int i = 1; i <= PmeOrder; i++)
                        bsplines_data[i] = data[i-1];

                    // Evaluate the actual bspline moduli for X/Y/Z.

                    for (int dim = 0; dim < 3; dim++) {
                        int ndata = (dim == 0 ? xsize : dim == 1 ? ysize : zsize);
                        vector<double> moduli(ndata);
                        for (int i = 0; i < ndata; i++) {
                            double sc = 0.0;
                            double ss = 0.0;
                            for (int j = 0; j < ndata; j++) {
                                double arg = (2.0*M_PI*i*j)/ndata;
                                sc += bsplines_data[j]*cos(arg);
                                ss += bsplines_data[j]*sin(arg);
                            }
                            moduli[i] = sc*sc+ss*ss;
                        }
                        for (int i = 0; i < ndata; i++)
                            if (moduli[i] < 1.0e-7)
                                moduli[i] = (moduli[i-1]+moduli[i+1])*0.5;
                        if (dim == 0)
                            xmoduli->upload(moduli, true);
                        else if (dim == 1)
                            ymoduli->upload(moduli, true);
                        else
                            zmoduli->upload(moduli, true);
                    }
                }
            }
        }
    }

    // Add code to subtract off the reciprocal part of excluded interactions.

    if ((nonbondedMethod == Ewald || nonbondedMethod == PME || nonbondedMethod == LJPME) && pmeio == NULL) {
        int numContexts = cu.getPlatformData().contexts.size();
        int startIndex = cu.getContextIndex()*force.getNumExceptions()/numContexts;
        int endIndex = (cu.getContextIndex()+1)*force.getNumExceptions()/numContexts;
        int numExclusions = endIndex-startIndex;
        if (numExclusions > 0) {
            paramsDefines["HAS_EXCLUSIONS"] = "1";
            vector<vector<int> > atoms(numExclusions, vector<int>(2));
            exclusionAtoms.initialize<int2>(cu, numExclusions, "exclusionAtoms");
            exclusionParams.initialize<float4>(cu, numExclusions, "exclusionParams");
            vector<int2> exclusionAtomsVec(numExclusions);
            for (int i = 0; i < numExclusions; i++) {
                int j = i+startIndex;
                exclusionAtomsVec[i] = make_int2(exclusions[j].first, exclusions[j].second);
                atoms[i][0] = exclusions[j].first;
                atoms[i][1] = exclusions[j].second;
            }
            exclusionAtoms.upload(exclusionAtomsVec);
            map<string, string> replacements;
            replacements["PARAMS"] = cu.getBondedUtilities().addArgument(exclusionParams.getDevicePointer(), "float4");
            replacements["EWALD_ALPHA"] = cu.doubleToString(alpha);
            replacements["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
            replacements["DO_LJPME"] = doLJPME ? "1" : "0";
            if (doLJPME)
                replacements["EWALD_DISPERSION_ALPHA"] = cu.doubleToString(dispersionAlpha);
            cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(HipKernelSources::pmeExclusions, replacements), force.getForceGroup());
        }
    }

    // Add the interaction to the default nonbonded kernel.

    string source = cu.replaceStrings(HipKernelSources::coulombLennardJones, defines);
    charges.initialize(cu, cu.getPaddedNumAtoms(), cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "charges");
    baseParticleParams.initialize<float4>(cu, cu.getPaddedNumAtoms(), "baseParticleParams");
    baseParticleParams.upload(baseParticleParamVec);
    map<string, string> replacements;
    if (usePosqCharges) {
        replacements["CHARGE1"] = "posq1.w";
        replacements["CHARGE2"] = "posq2.w";
    }
    else {
        replacements["CHARGE1"] = prefix+"charge1";
        replacements["CHARGE2"] = prefix+"charge2";
    }
    if (hasCoulomb)
        cu.getNonbondedUtilities().addParameter(HipNonbondedUtilities::ParameterInfo(prefix+"charge", "real", 1, charges.getElementSize(), charges.getDevicePointer()));
    sigmaEpsilon.initialize<float2>(cu, cu.getPaddedNumAtoms(), "sigmaEpsilon");
    if (hasLJ) {
        replacements["SIGMA_EPSILON1"] = prefix+"sigmaEpsilon1";
        replacements["SIGMA_EPSILON2"] = prefix+"sigmaEpsilon2";
        cu.getNonbondedUtilities().addParameter(HipNonbondedUtilities::ParameterInfo(prefix+"sigmaEpsilon", "float", 2, sizeof(float2), sigmaEpsilon.getDevicePointer()));
    }
    source = cu.replaceStrings(source, replacements);
    cu.getNonbondedUtilities().addInteraction(useCutoff, usePeriodic, true, force.getCutoffDistance(), exclusionList, source, force.getForceGroup(), true);

    // Initialize the exceptions.

    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions > 0) {
        paramsDefines["HAS_EXCEPTIONS"] = "1";
        exceptionAtoms.resize(numExceptions);
        vector<vector<int> > atoms(numExceptions, vector<int>(2));
        exceptionParams.initialize<float4>(cu, numExceptions, "exceptionParams");
        baseExceptionParams.initialize<float4>(cu, numExceptions, "baseExceptionParams");
        vector<float4> baseExceptionParamsVec(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            double chargeProd, sigma, epsilon;
            force.getExceptionParameters(exceptions[startIndex+i], atoms[i][0], atoms[i][1], chargeProd, sigma, epsilon);
            baseExceptionParamsVec[i] = make_float4(chargeProd, sigma, epsilon, 0);
            exceptionAtoms[i] = make_pair(atoms[i][0], atoms[i][1]);
        }
        baseExceptionParams.upload(baseExceptionParamsVec);
        map<string, string> replacements;
        replacements["APPLY_PERIODIC"] = (usePeriodic && force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0");
        replacements["PARAMS"] = cu.getBondedUtilities().addArgument(exceptionParams.getDevicePointer(), "float4");
        cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(HipKernelSources::nonbondedExceptions, replacements), force.getForceGroup());
    }

    // Initialize parameter offsets.

    vector<vector<float4> > particleOffsetVec(force.getNumParticles());
    vector<vector<float4> > exceptionOffsetVec(force.getNumExceptions());
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string param;
        int particle;
        double charge, sigma, epsilon;
        force.getParticleParameterOffset(i, param, particle, charge, sigma, epsilon);
        auto paramPos = find(paramNames.begin(), paramNames.end(), param);
        int paramIndex;
        if (paramPos == paramNames.end()) {
            paramIndex = paramNames.size();
            paramNames.push_back(param);
        }
        else
            paramIndex = paramPos-paramNames.begin();
        particleOffsetVec[particle].push_back(make_float4(charge, sigma, epsilon, paramIndex));
    }
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        auto paramPos = find(paramNames.begin(), paramNames.end(), param);
        int paramIndex;
        if (paramPos == paramNames.end()) {
            paramIndex = paramNames.size();
            paramNames.push_back(param);
        }
        else
            paramIndex = paramPos-paramNames.begin();
        exceptionOffsetVec[exceptionIndex[exception]].push_back(make_float4(charge, sigma, epsilon, paramIndex));
    }
    paramValues.resize(paramNames.size(), 0.0);
    particleParamOffsets.initialize<float4>(cu, max(force.getNumParticleParameterOffsets(), 1), "particleParamOffsets");
    exceptionParamOffsets.initialize<float4>(cu, max(force.getNumExceptionParameterOffsets(), 1), "exceptionParamOffsets");
    particleOffsetIndices.initialize<int>(cu, cu.getPaddedNumAtoms()+1, "particleOffsetIndices");
    exceptionOffsetIndices.initialize<int>(cu, force.getNumExceptions()+1, "exceptionOffsetIndices");
    vector<int> particleOffsetIndicesVec, exceptionOffsetIndicesVec;
    vector<float4> p, e;
    for (int i = 0; i < particleOffsetVec.size(); i++) {
        particleOffsetIndicesVec.push_back(p.size());
        for (int j = 0; j < particleOffsetVec[i].size(); j++)
            p.push_back(particleOffsetVec[i][j]);
    }
    while (particleOffsetIndicesVec.size() < particleOffsetIndices.getSize())
        particleOffsetIndicesVec.push_back(p.size());
    for (int i = 0; i < exceptionOffsetVec.size(); i++) {
        exceptionOffsetIndicesVec.push_back(e.size());
        for (int j = 0; j < exceptionOffsetVec[i].size(); j++)
            e.push_back(exceptionOffsetVec[i][j]);
    }
    exceptionOffsetIndicesVec.push_back(e.size());
    if (force.getNumParticleParameterOffsets() > 0) {
        particleParamOffsets.upload(p);
        particleOffsetIndices.upload(particleOffsetIndicesVec);
    }
    if (force.getNumExceptionParameterOffsets() > 0) {
        exceptionParamOffsets.upload(e);
        exceptionOffsetIndices.upload(exceptionOffsetIndicesVec);
    }
    globalParams.initialize(cu, max((int) paramValues.size(), 1), cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "globalParams");
    recomputeParams = true;

    // Initialize the kernel for updating parameters.

    hipModule_t module = cu.createModule(HipKernelSources::nonbondedParameters, paramsDefines);
    computeParamsKernel = cu.getKernel(module, "computeParameters");
    computeExclusionParamsKernel = cu.getKernel(module, "computeExclusionParameters");
    info = new ForceInfo(force);
    cu.addForce(info);
}

double HipCalcNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    // Update particle and exception parameters.

    bool paramChanged = false;
    for (int i = 0; i < paramNames.size(); i++) {
        double value = context.getParameter(paramNames[i]);
        if (value != paramValues[i]) {
            paramValues[i] = value;;
            paramChanged = true;
        }
    }
    if (paramChanged) {
        recomputeParams = true;
        globalParams.upload(paramValues, true);
    }
    double energy = (includeReciprocal ? ewaldSelfEnergy : 0.0);
    if (recomputeParams || hasOffsets) {
        bool computeSelfEnergy = (includeEnergy && includeReciprocal);
        int numAtoms = cu.getPaddedNumAtoms();
        vector<void*> paramsArgs = {&cu.getEnergyBuffer().getDevicePointer(), &computeSelfEnergy, &globalParams.getDevicePointer(), &numAtoms,
                &baseParticleParams.getDevicePointer(), &cu.getPosq().getDevicePointer(), &charges.getDevicePointer(), &sigmaEpsilon.getDevicePointer(),
                &particleParamOffsets.getDevicePointer(), &particleOffsetIndices.getDevicePointer()};
        int numExceptions;
        if (exceptionParams.isInitialized()) {
            numExceptions = exceptionParams.getSize();
            paramsArgs.push_back(&numExceptions);
            paramsArgs.push_back(&baseExceptionParams.getDevicePointer());
            paramsArgs.push_back(&exceptionParams.getDevicePointer());
            paramsArgs.push_back(&exceptionParamOffsets.getDevicePointer());
            paramsArgs.push_back(&exceptionOffsetIndices.getDevicePointer());
        }
        cu.executeKernel(computeParamsKernel, &paramsArgs[0], cu.getPaddedNumAtoms());
        if (exclusionParams.isInitialized()) {
            int numExclusions = exclusionParams.getSize();
            vector<void*> exclusionParamsArgs = {&cu.getPosq().getDevicePointer(), &charges.getDevicePointer(), &sigmaEpsilon.getDevicePointer(),
                    &numExclusions, &exclusionAtoms.getDevicePointer(), &exclusionParams.getDevicePointer()};
            cu.executeKernel(computeExclusionParamsKernel, &exclusionParamsArgs[0], numExclusions);
        }
        if (usePmeStream) {
            hipEventRecord(paramsSyncEvent, cu.getCurrentStream());
            hipStreamWaitEvent(pmeStream, paramsSyncEvent, 0);
        }
        if (hasOffsets)
            energy = 0.0; // The Ewald self energy was computed in the kernel.
        recomputeParams = false;
    }

    // Do reciprocal space calculations.

    if (cosSinSums.isInitialized() && includeReciprocal) {
        void* sumsArgs[] = {&cu.getEnergyBuffer().getDevicePointer(), &cu.getPosq().getDevicePointer(), &cosSinSums.getDevicePointer(), cu.getPeriodicBoxSizePointer()};
        cu.executeKernel(ewaldSumsKernel, sumsArgs, cosSinSums.getSize());
        void* forcesArgs[] = {&cu.getForce().getDevicePointer(), &cu.getPosq().getDevicePointer(), &cosSinSums.getDevicePointer(), cu.getPeriodicBoxSizePointer()};
        cu.executeKernel(ewaldForcesKernel, forcesArgs, cu.getNumAtoms());
    }
    if (pmeGrid1.isInitialized() && includeReciprocal) {
        if (usePmeStream)
            cu.setCurrentStream(pmeStream);

        // Invert the periodic box vectors.

        Vec3 boxVectors[3];
        cu.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double determinant = boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2];
        double scale = 1.0/determinant;
        double3 recipBoxVectors[3];
        recipBoxVectors[0] = make_double3(boxVectors[1][1]*boxVectors[2][2]*scale, 0, 0);
        recipBoxVectors[1] = make_double3(-boxVectors[1][0]*boxVectors[2][2]*scale, boxVectors[0][0]*boxVectors[2][2]*scale, 0);
        recipBoxVectors[2] = make_double3((boxVectors[1][0]*boxVectors[2][1]-boxVectors[1][1]*boxVectors[2][0])*scale, -boxVectors[0][0]*boxVectors[2][1]*scale, boxVectors[0][0]*boxVectors[1][1]*scale);
        float3 recipBoxVectorsFloat[3];
        void* recipBoxVectorPointer[3];
        if (cu.getUseDoublePrecision()) {
            recipBoxVectorPointer[0] = &recipBoxVectors[0];
            recipBoxVectorPointer[1] = &recipBoxVectors[1];
            recipBoxVectorPointer[2] = &recipBoxVectors[2];
        }
        else {
            recipBoxVectorsFloat[0] = make_float3((float) recipBoxVectors[0].x, 0, 0);
            recipBoxVectorsFloat[1] = make_float3((float) recipBoxVectors[1].x, (float) recipBoxVectors[1].y, 0);
            recipBoxVectorsFloat[2] = make_float3((float) recipBoxVectors[2].x, (float) recipBoxVectors[2].y, (float) recipBoxVectors[2].z);
            recipBoxVectorPointer[0] = &recipBoxVectorsFloat[0];
            recipBoxVectorPointer[1] = &recipBoxVectorsFloat[1];
            recipBoxVectorPointer[2] = &recipBoxVectorsFloat[2];
        }

        // Execute the reciprocal space kernels.

        if (hasCoulomb) {
            void* gridIndexArgs[] = {&cu.getPosq().getDevicePointer(), &pmeAtomGridIndex.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                    cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
            cu.executeKernel(pmeGridIndexKernel, gridIndexArgs, cu.getNumAtoms());

            sort->sort(pmeAtomGridIndex);

            void* spreadArgs[] = {&cu.getPosq().getDevicePointer(), &pmeGrid2.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                    cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &pmeAtomGridIndex.getDevicePointer(),
                    &charges.getDevicePointer()};
            cu.executeKernel(pmeSpreadChargeKernel, spreadArgs, cu.getNumAtoms(), 128);

            void* finishSpreadArgs[] = {&pmeGrid2.getDevicePointer(), &pmeGrid1.getDevicePointer()};
            cu.executeKernel(pmeFinishSpreadChargeKernel, finishSpreadArgs, gridSizeX*gridSizeY*gridSizeZ, 256);

            if (useHipFFT) {
                if (cu.getUseDoublePrecision())
                    hipfftExecD2Z(fftForward, (double*) pmeGrid1.getDevicePointer(), (double2*) pmeGrid2.getDevicePointer());
                else
                    hipfftExecR2C(fftForward, (float*) pmeGrid1.getDevicePointer(), (float2*) pmeGrid2.getDevicePointer());
            }
            else {
                fft->execFFT(pmeGrid1, pmeGrid2, true);
            }

            if (includeEnergy) {
                void* computeEnergyArgs[] = {&pmeGrid2.getDevicePointer(), usePmeStream ? &pmeEnergyBuffer.getDevicePointer() : &cu.getEnergyBuffer().getDevicePointer(),
                        &pmeBsplineModuliX.getDevicePointer(), &pmeBsplineModuliY.getDevicePointer(), &pmeBsplineModuliZ.getDevicePointer(),
                        cu.getPeriodicBoxSizePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
                cu.executeKernel(pmeEvalEnergyKernel, computeEnergyArgs, gridSizeX*gridSizeY*gridSizeZ);
            }

            void* convolutionArgs[] = {&pmeGrid2.getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(),
                    &pmeBsplineModuliX.getDevicePointer(), &pmeBsplineModuliY.getDevicePointer(), &pmeBsplineModuliZ.getDevicePointer(),
                    cu.getPeriodicBoxSizePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
            cu.executeKernel(pmeConvolutionKernel, convolutionArgs, gridSizeX*gridSizeY*gridSizeZ, 256);

            if (useHipFFT) {
                if (cu.getUseDoublePrecision())
                    hipfftExecZ2D(fftBackward, (double2*) pmeGrid2.getDevicePointer(), (double*) pmeGrid1.getDevicePointer());
                else
                    hipfftExecC2R(fftBackward, (float2*) pmeGrid2.getDevicePointer(), (float*)  pmeGrid1.getDevicePointer());
            }
            else {
                fft->execFFT(pmeGrid2, pmeGrid1, false);
            }

            void* interpolateArgs[] = {&cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &pmeGrid1.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                    cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &pmeAtomGridIndex.getDevicePointer(),
                    &charges.getDevicePointer()};
            cu.executeKernel(pmeInterpolateForceKernel, interpolateArgs, cu.getNumAtoms(), 128);
        }

        if (doLJPME && hasLJ) {
            if (!hasCoulomb) {
                void* gridIndexArgs[] = {&cu.getPosq().getDevicePointer(), &pmeAtomGridIndex.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                        cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                        recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
                cu.executeKernel(pmeDispersionGridIndexKernel, gridIndexArgs, cu.getNumAtoms());

                sort->sort(pmeAtomGridIndex);
                cu.clearBuffer(pmeEnergyBuffer);
            }

            cu.clearBuffer(pmeGrid2);
            void* spreadArgs[] = {&cu.getPosq().getDevicePointer(), &pmeGrid2.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                    cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &pmeAtomGridIndex.getDevicePointer(),
                    &sigmaEpsilon.getDevicePointer()};
            cu.executeKernel(pmeDispersionSpreadChargeKernel, spreadArgs, cu.getNumAtoms(), 128);

            void* finishSpreadArgs[] = {&pmeGrid2.getDevicePointer(), &pmeGrid1.getDevicePointer()};
            cu.executeKernel(pmeDispersionFinishSpreadChargeKernel, finishSpreadArgs, dispersionGridSizeX*dispersionGridSizeY*dispersionGridSizeZ, 256);

            if (useHipFFT) {
                if (cu.getUseDoublePrecision())
                    hipfftExecD2Z(dispersionFftForward, (double*) pmeGrid1.getDevicePointer(), (double2*) pmeGrid2.getDevicePointer());
                else
                    hipfftExecR2C(dispersionFftForward, (float*) pmeGrid1.getDevicePointer(), (float2*) pmeGrid2.getDevicePointer());
            }
            else {
                dispersionFft->execFFT(pmeGrid1, pmeGrid2, true);
            }

            if (includeEnergy) {
                void* computeEnergyArgs[] = {&pmeGrid2.getDevicePointer(), usePmeStream ? &pmeEnergyBuffer.getDevicePointer() : &cu.getEnergyBuffer().getDevicePointer(),
                        &pmeDispersionBsplineModuliX.getDevicePointer(), &pmeDispersionBsplineModuliY.getDevicePointer(), &pmeDispersionBsplineModuliZ.getDevicePointer(),
                        cu.getPeriodicBoxSizePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
                cu.executeKernel(pmeEvalDispersionEnergyKernel, computeEnergyArgs, dispersionGridSizeX*dispersionGridSizeY*dispersionGridSizeZ);
            }

            void* convolutionArgs[] = {&pmeGrid2.getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(),
                    &pmeDispersionBsplineModuliX.getDevicePointer(), &pmeDispersionBsplineModuliY.getDevicePointer(), &pmeDispersionBsplineModuliZ.getDevicePointer(),
                    cu.getPeriodicBoxSizePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
            cu.executeKernel(pmeDispersionConvolutionKernel, convolutionArgs, dispersionGridSizeX*dispersionGridSizeY*dispersionGridSizeZ, 256);

            if (useHipFFT) {
                if (cu.getUseDoublePrecision())
                    hipfftExecZ2D(dispersionFftBackward, (double2*) pmeGrid2.getDevicePointer(), (double*) pmeGrid1.getDevicePointer());
                else
                    hipfftExecC2R(dispersionFftBackward, (float2*) pmeGrid2.getDevicePointer(), (float*)  pmeGrid1.getDevicePointer());
            }
            else {
                dispersionFft->execFFT(pmeGrid2, pmeGrid1, false);
            }

            void* interpolateArgs[] = {&cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &pmeGrid1.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                    cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &pmeAtomGridIndex.getDevicePointer(),
                    &sigmaEpsilon.getDevicePointer()};
            cu.executeKernel(pmeInterpolateDispersionForceKernel, interpolateArgs, cu.getNumAtoms(), 128);
        }
        if (usePmeStream) {
            hipEventRecord(pmeSyncEvent, pmeStream);
            cu.restoreDefaultStream();
        }
    }

    if (dispersionCoefficient != 0.0 && includeDirect) {
        double4 boxSize = cu.getPeriodicBoxSize();
        energy += dispersionCoefficient/(boxSize.x*boxSize.y*boxSize.z);
    }
    return energy;
}

void HipCalcNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const NonbondedForce& force) {
    // Make sure the new parameters are acceptable.

    cu.setAsCurrent();
    if (force.getNumParticles() != cu.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    if (!hasCoulomb || !hasLJ) {
        for (int i = 0; i < force.getNumParticles(); i++) {
            double charge, sigma, epsilon;
            force.getParticleParameters(i, charge, sigma, epsilon);
            if (!hasCoulomb && charge != 0.0)
                throw OpenMMException("updateParametersInContext: The nonbonded force kernel does not include Coulomb interactions, because all charges were originally 0");
            if (!hasLJ && epsilon != 0.0)
                throw OpenMMException("updateParametersInContext: The nonbonded force kernel does not include Lennard-Jones interactions, because all epsilons were originally 0");
        }
    }
    vector<int> exceptions;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        if (exceptionAtoms.size() > exceptions.size() && make_pair(particle1, particle2) == exceptionAtoms[exceptions.size()])
            exceptions.push_back(i);
        else if (chargeProd != 0.0 || epsilon != 0.0)
            throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");
    }
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;

    // Record the per-particle parameters.

    vector<float4> baseParticleParamVec(cu.getPaddedNumAtoms(), make_float4(0, 0, 0, 0));
    const vector<int>& order = cu.getAtomIndex();
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, sigma, epsilon;
        force.getParticleParameters(i, charge, sigma, epsilon);
        baseParticleParamVec[i] = make_float4(charge, sigma, epsilon, 0);
    }
    baseParticleParams.upload(baseParticleParamVec);

    // Record the exceptions.

    if (numExceptions > 0) {
        vector<vector<int> > atoms(numExceptions, vector<int>(2));
        vector<float4> baseExceptionParamsVec(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            double chargeProd, sigma, epsilon;
            force.getExceptionParameters(exceptions[startIndex+i], atoms[i][0], atoms[i][1], chargeProd, sigma, epsilon);
            baseExceptionParamsVec[i] = make_float4(chargeProd, sigma, epsilon, 0);
        }
        baseExceptionParams.upload(baseExceptionParamsVec);
    }

    // Compute other values.

    ewaldSelfEnergy = 0.0;
    if (nonbondedMethod == Ewald || nonbondedMethod == PME || nonbondedMethod == LJPME) {
        if (cu.getContextIndex() == 0) {
            for (int i = 0; i < force.getNumParticles(); i++) {
                ewaldSelfEnergy -= baseParticleParamVec[i].x*baseParticleParamVec[i].x*ONE_4PI_EPS0*alpha/sqrt(M_PI);
                if (doLJPME)
                    ewaldSelfEnergy += baseParticleParamVec[i].z*pow(baseParticleParamVec[i].y*dispersionAlpha, 6)/3.0;
            }
        }
    }
    if (force.getUseDispersionCorrection() && cu.getContextIndex() == 0 && (nonbondedMethod == CutoffPeriodic || nonbondedMethod == Ewald || nonbondedMethod == PME))
        dispersionCoefficient = NonbondedForceImpl::calcDispersionCorrection(context.getSystem(), force);
    cu.invalidateMolecules();
    recomputeParams = true;
}

void HipCalcNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != PME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    if (cu.getPlatformData().useCpuPme)
        cpuPme.getAs<CalcPmeReciprocalForceKernel>().getPMEParameters(alpha, nx, ny, nz);
    else {
        alpha = this->alpha;
        nx = gridSizeX;
        ny = gridSizeY;
        nz = gridSizeZ;
    }
}

void HipCalcNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (!doLJPME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    if (cu.getPlatformData().useCpuPme)
        //cpuPme.getAs<CalcPmeReciprocalForceKernel>().getLJPMEParameters(alpha, nx, ny, nz);
        throw OpenMMException("getPMEParametersInContext: CPUPME has not been implemented for LJPME yet.");
    else {
        alpha = this->dispersionAlpha;
        nx = dispersionGridSizeX;
        ny = dispersionGridSizeY;
        nz = dispersionGridSizeZ;
    }
}

class HipCalcCustomCVForceKernel::ForceInfo : public HipForceInfo {
public:
    ForceInfo(ComputeForceInfo& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        return force.areParticlesIdentical(particle1, particle2);
    }
    int getNumParticleGroups() {
        return force.getNumParticleGroups();
    }
    void getParticlesInGroup(int index, std::vector<int>& particles) {
        force.getParticlesInGroup(index, particles);
    }
    bool areGroupsIdentical(int group1, int group2) {
        return force.areGroupsIdentical(group1, group2);
    }
private:
    ComputeForceInfo& force;
};

class HipCalcCustomCVForceKernel::ReorderListener : public HipContext::ReorderListener {
public:
    ReorderListener(HipContext& cu, HipArray& invAtomOrder) : cu(cu), invAtomOrder(invAtomOrder) {
    }
    void execute() {
        vector<int> invOrder(cu.getPaddedNumAtoms());
        const vector<int>& order = cu.getAtomIndex();
        for (int i = 0; i < order.size(); i++)
            invOrder[order[i]] = i;
        invAtomOrder.upload(invOrder);
    }
private:
    HipContext& cu;
    HipArray& invAtomOrder;
};

void HipCalcCustomCVForceKernel::initialize(const System& system, const CustomCVForce& force, ContextImpl& innerContext) {
    int numCVs = force.getNumCollectiveVariables();
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        globalParameterNames.push_back(force.getGlobalParameterName(i));
    for (int i = 0; i < numCVs; i++)
        variableNames.push_back(force.getCollectiveVariableName(i));
    for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
        string name = force.getEnergyParameterDerivativeName(i);
        paramDerivNames.push_back(name);
        cu.addEnergyParameterDerivative(name);
    }

    // Create custom functions for the tabulated functions.

    map<string, Lepton::CustomFunction*> functions;
    for (int i = 0; i < (int) force.getNumTabulatedFunctions(); i++)
        functions[force.getTabulatedFunctionName(i)] = createReferenceTabulatedFunction(force.getTabulatedFunction(i));

    // Create the expressions.

    Lepton::ParsedExpression energyExpr = Lepton::Parser::parse(force.getEnergyFunction(), functions);
    energyExpression = energyExpr.createProgram();
    variableDerivExpressions.clear();
    for (auto& name : variableNames)
        variableDerivExpressions.push_back(energyExpr.differentiate(name).optimize().createProgram());
    paramDerivExpressions.clear();
    for (auto& name : paramDerivNames)
        paramDerivExpressions.push_back(energyExpr.differentiate(name).optimize().createProgram());

    // Delete the custom functions.

    for (auto& function : functions)
        delete function.second;

    // Copy parameter derivatives from the inner context.

    HipContext& cu2 = *reinterpret_cast<HipPlatform::PlatformData*>(innerContext.getPlatformData())->contexts[0];
    for (auto& param : cu2.getEnergyParamDerivNames())
        cu.addEnergyParameterDerivative(param);

    // Create arrays for storing information.

    cvForces.resize(numCVs);
    for (int i = 0; i < numCVs; i++)
        cvForces[i].initialize<long long>(cu, 3*cu.getPaddedNumAtoms(), "cvForce");
    invAtomOrder.initialize<int>(cu, cu.getPaddedNumAtoms(), "invAtomOrder");
    innerInvAtomOrder.initialize<int>(cu, cu.getPaddedNumAtoms(), "innerInvAtomOrder");

    // Create the kernels.

    stringstream args, add;
    for (int i = 0; i < numCVs; i++) {
        args << ", long long* __restrict__ force" << i << ", real dEdV" << i;
        add << "forces[i] += (long long) (force" << i << "[i]*dEdV" << i << ");\n";
    }
    map<string, string> replacements;
    replacements["PARAMETER_ARGUMENTS"] = args.str();
    replacements["ADD_FORCES"] = add.str();
    hipModule_t module = cu.createModule(cu.replaceStrings(HipKernelSources::vectorOps+HipKernelSources::customCVForce, replacements));
    copyStateKernel = cu.getKernel(module, "copyState");
    copyForcesKernel = cu.getKernel(module, "copyForces");
    addForcesKernel = cu.getKernel(module, "addForces");

    // This context needs to respect all forces in the inner context when reordering atoms.

    for (auto* info : cu2.getForceInfos())
        cu.addForce(new ForceInfo(*info));
}

double HipCalcCustomCVForceKernel::execute(ContextImpl& context, ContextImpl& innerContext, bool includeForces, bool includeEnergy) {
    copyState(context, innerContext);
    int numCVs = variableNames.size();
    int numAtoms = cu.getNumAtoms();
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    HipContext& cu2 = *reinterpret_cast<HipPlatform::PlatformData*>(innerContext.getPlatformData())->contexts[0];
    vector<double> cvValues;
    vector<map<string, double> > cvDerivs(numCVs);
    void* copyForcesArgs[] = {NULL, &invAtomOrder.getDevicePointer(), &cu2.getForce().getDevicePointer(), &cu2.getAtomIndexArray().getDevicePointer(), &numAtoms, &paddedNumAtoms};
    for (int i = 0; i < numCVs; i++) {
        cvValues.push_back(innerContext.calcForcesAndEnergy(true, true, 1<<i));
        copyForcesArgs[0] = &cvForces[i].getDevicePointer();
        cu.executeKernel(copyForcesKernel, copyForcesArgs, numAtoms);
        innerContext.getEnergyParameterDerivatives(cvDerivs[i]);
    }

    // Compute the energy and forces.

    map<string, double> variables;
    for (auto& name : globalParameterNames)
        variables[name] = context.getParameter(name);
    for (int i = 0; i < numCVs; i++)
        variables[variableNames[i]] = cvValues[i];
    double energy = energyExpression.evaluate(variables);
    int bufferSize = cu.getForce().getSize();
    vector<void*> addForcesArgs;
    addForcesArgs.push_back(&cu.getForce().getDevicePointer());
    addForcesArgs.push_back(&bufferSize);
    vector<double> dEdV(numCVs);
    vector<float> dEdVFloat(numCVs);
    for (int i = 0; i < numCVs; i++) {
        dEdV[i] = variableDerivExpressions[i].evaluate(variables);
        dEdVFloat[i] = (float) dEdV[i];
        addForcesArgs.push_back(&cvForces[i].getDevicePointer());
        if (cu.getUseDoublePrecision())
            addForcesArgs.push_back(&dEdV[i]);
        else
            addForcesArgs.push_back(&dEdVFloat[i]);
    }
    cu.executeKernel(addForcesKernel, &addForcesArgs[0], numAtoms);

    // Compute the energy parameter derivatives.

    map<string, double>& energyParamDerivs = cu.getEnergyParamDerivWorkspace();
    for (int i = 0; i < paramDerivExpressions.size(); i++)
        energyParamDerivs[paramDerivNames[i]] += paramDerivExpressions[i].evaluate(variables);
    for (int i = 0; i < numCVs; i++) {
        double dEdV = variableDerivExpressions[i].evaluate(variables);
        for (auto& deriv : cvDerivs[i])
            energyParamDerivs[deriv.first] += dEdV*deriv.second;
    }
    return energy;
}

void HipCalcCustomCVForceKernel::copyState(ContextImpl& context, ContextImpl& innerContext) {
    int numAtoms = cu.getNumAtoms();
    HipContext& cu2 = *reinterpret_cast<HipPlatform::PlatformData*>(innerContext.getPlatformData())->contexts[0];
    if (!hasInitializedListeners) {
        hasInitializedListeners = true;

        // Initialize the listeners.

        ReorderListener* listener1 = new ReorderListener(cu, invAtomOrder);
        ReorderListener* listener2 = new ReorderListener(cu2, innerInvAtomOrder);
        cu.addReorderListener(listener1);
        cu2.addReorderListener(listener2);
        listener1->execute();
        listener2->execute();
    }
    hipDeviceptr_t posCorrection = (cu.getUseMixedPrecision() ? cu.getPosqCorrection().getDevicePointer() : 0);
    hipDeviceptr_t posCorrection2 = (cu2.getUseMixedPrecision() ? cu2.getPosqCorrection().getDevicePointer() : 0);
    void* copyStateArgs[] = {&cu.getPosq().getDevicePointer(), &posCorrection, &cu.getVelm().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(),
        &cu2.getPosq().getDevicePointer(), &posCorrection2,& cu2.getVelm().getDevicePointer(), &innerInvAtomOrder.getDevicePointer(), &numAtoms};
    cu.executeKernel(copyStateKernel, copyStateArgs, numAtoms);
    Vec3 a, b, c;
    context.getPeriodicBoxVectors(a, b, c);
    innerContext.setPeriodicBoxVectors(a, b, c);
    innerContext.setTime(context.getTime());
    map<string, double> innerParameters = innerContext.getParameters();
    for (auto& param : innerParameters)
        innerContext.setParameter(param.first, context.getParameter(param.first));
}

void HipCalcCustomCVForceKernel::copyParametersToContext(ContextImpl& context, const CustomCVForce& force) {
    // Create custom functions for the tabulated functions.

    map<string, CustomFunction*> functions;
    for (int i = 0; i < (int) force.getNumTabulatedFunctions(); i++)
        functions[force.getTabulatedFunctionName(i)] = createReferenceTabulatedFunction(force.getTabulatedFunction(i));

    // Replace tabulated functions in the expressions.

    replaceFunctionsInExpression(functions, energyExpression);
    for (auto& expression : variableDerivExpressions)
        replaceFunctionsInExpression(functions, expression);
    for (auto& expression : paramDerivExpressions)
        replaceFunctionsInExpression(functions, expression);

    // Delete the custom functions.

    for (auto& function : functions)
        delete function.second;
}

void HipApplyMonteCarloBarostatKernel::initialize(const System& system, const Force& thermostat) {
    cu.setAsCurrent();
    savedPositions.initialize(cu, cu.getPaddedNumAtoms(), cu.getUseDoublePrecision() ? sizeof(double4) : sizeof(float4), "savedPositions");
    savedForces.initialize<long long>(cu, cu.getPaddedNumAtoms()*3, "savedForces");
    hipModule_t module = cu.createModule(HipKernelSources::monteCarloBarostat);
    kernel = cu.getKernel(module, "scalePositions");
}

void HipApplyMonteCarloBarostatKernel::scaleCoordinates(ContextImpl& context, double scaleX, double scaleY, double scaleZ) {
    cu.setAsCurrent();
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;

        // Create the arrays with the molecule definitions.

        vector<vector<int> > molecules = context.getMolecules();
        numMolecules = molecules.size();
        moleculeAtoms.initialize<int>(cu, cu.getNumAtoms(), "moleculeAtoms");
        moleculeStartIndex.initialize<int>(cu, numMolecules+1, "moleculeStartIndex");
        vector<int> atoms(moleculeAtoms.getSize());
        vector<int> startIndex(moleculeStartIndex.getSize());
        int index = 0;
        for (int i = 0; i < numMolecules; i++) {
            startIndex[i] = index;
            for (int molecule : molecules[i])
                atoms[index++] = molecule;
        }
        startIndex[numMolecules] = index;
        moleculeAtoms.upload(atoms);
        moleculeStartIndex.upload(startIndex);

        // Initialize the kernel arguments.

    }
    int bytesToCopy = cu.getPosq().getSize()*(cu.getUseDoublePrecision() ? sizeof(double4) : sizeof(float4));
    hipError_t result = hipMemcpyDtoD(savedPositions.getDevicePointer(), cu.getPosq().getDevicePointer(), bytesToCopy);
    if (result != hipSuccess) {
        std::stringstream m;
        m<<"Error saving positions for MC barostat: "<<cu.getErrorString(result)<<" ("<<result<<")";
        throw OpenMMException(m.str());
    }
    result = hipMemcpyDtoD(savedForces.getDevicePointer(), cu.getForce().getDevicePointer(), savedForces.getSize()*savedForces.getElementSize());
    if (result != hipSuccess) {
        std::stringstream m;
        m<<"Error saving forces for MC barostat: "<<cu.getErrorString(result)<<" ("<<result<<")";
        throw OpenMMException(m.str());
    }
    float scalefX = (float) scaleX;
    float scalefY = (float) scaleY;
    float scalefZ = (float) scaleZ;
    void* args[] = {&scalefX, &scalefY, &scalefZ, &numMolecules, cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
                    cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
		    &cu.getPosq().getDevicePointer(), &moleculeAtoms.getDevicePointer(), &moleculeStartIndex.getDevicePointer()};
    cu.executeKernel(kernel, args, cu.getNumAtoms());
    for (auto& offset : cu.getPosCellOffsets())
        offset = mm_int4(0, 0, 0, 0);
    lastAtomOrder = cu.getAtomIndex();
}

void HipApplyMonteCarloBarostatKernel::restoreCoordinates(ContextImpl& context) {
    cu.setAsCurrent();
    int bytesToCopy = cu.getPosq().getSize()*(cu.getUseDoublePrecision() ? sizeof(double4) : sizeof(float4));
    hipError_t result = hipMemcpyDtoD(cu.getPosq().getDevicePointer(), savedPositions.getDevicePointer(), bytesToCopy);
    if (result != hipSuccess) {
        std::stringstream m;
        m<<"Error restoring positions for MC barostat: "<<cu.getErrorString(result)<<" ("<<result<<")";
        throw OpenMMException(m.str());
    }
    result = hipMemcpyDtoD(cu.getForce().getDevicePointer(), savedForces.getDevicePointer(), savedForces.getSize()*savedForces.getElementSize());
    if (result != hipSuccess) {
        std::stringstream m;
        m<<"Error restoring forces for MC barostat: "<<cu.getErrorString(result)<<" ("<<result<<")";
        throw OpenMMException(m.str());
    }
}
