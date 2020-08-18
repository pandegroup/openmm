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

#include "openmm/common/CommonKernels.h"
#include "openmm/common/ExpressionUtilities.h"
#include "openmm/Context.h"
#include "openmm/internal/AndersenThermostatImpl.h"
#include "openmm/internal/CMAPTorsionForceImpl.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/CustomCentroidBondForceImpl.h"
#include "openmm/internal/CustomCompoundBondForceImpl.h"
#include "openmm/internal/CustomHbondForceImpl.h"
#include "openmm/internal/CustomManyParticleForceImpl.h"
#include "openmm/internal/CustomNonbondedForceImpl.h"
#include "CommonKernelSources.h"
#include "lepton/CustomFunction.h"
#include "lepton/ExpressionTreeNode.h"
#include "lepton/Operation.h"
#include "lepton/Parser.h"
#include "lepton/ParsedExpression.h"
#include "ReferenceTabulatedFunction.h"
#include "SimTKOpenMMRealType.h"
#include "SimTKOpenMMUtilities.h"
#include "jama_eig.h"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>
#include <string>

using namespace OpenMM;
using namespace std;
using namespace Lepton;

static void setPeriodicBoxArgs(ComputeContext& cc, ComputeKernel kernel, int index) {
    Vec3 a, b, c;
    cc.getPeriodicBoxVectors(a, b, c);
    if (cc.getUseDoublePrecision()) {
        kernel->setArg(index++, mm_double4(a[0], b[1], c[2], 0.0));
        kernel->setArg(index++, mm_double4(1.0/a[0], 1.0/b[1], 1.0/c[2], 0.0));
        kernel->setArg(index++, mm_double4(a[0], a[1], a[2], 0.0));
        kernel->setArg(index++, mm_double4(b[0], b[1], b[2], 0.0));
        kernel->setArg(index, mm_double4(c[0], c[1], c[2], 0.0));
    }
    else {
        kernel->setArg(index++, mm_float4((float) a[0], (float) b[1], (float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4(1.0f/(float) a[0], 1.0f/(float) b[1], 1.0f/(float) c[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) a[0], (float) a[1], (float) a[2], 0.0f));
        kernel->setArg(index++, mm_float4((float) b[0], (float) b[1], (float) b[2], 0.0f));
        kernel->setArg(index, mm_float4((float) c[0], (float) c[1], (float) c[2], 0.0f));
    }
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

class CommonCalcHarmonicBondForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const HarmonicBondForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumBonds();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        double length, k;
        force.getBondParameters(index, particle1, particle2, length, k);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2;
        double length1, length2, k1, k2;
        force.getBondParameters(group1, particle1, particle2, length1, k1);
        force.getBondParameters(group2, particle1, particle2, length2, k2);
        return (length1 == length2 && k1 == k2);
    }
private:
    const HarmonicBondForce& force;
};

void CommonCalcHarmonicBondForceKernel::initialize(const System& system, const HarmonicBondForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumBonds()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumBonds()/numContexts;
    numBonds = endIndex-startIndex;
    if (numBonds == 0)
        return;
    vector<vector<int> > atoms(numBonds, vector<int>(2));
    params.initialize<mm_float2>(cc, numBonds, "bondParams");
    vector<mm_float2> paramVector(numBonds);
    for (int i = 0; i < numBonds; i++) {
        double length, k;
        force.getBondParameters(startIndex+i, atoms[i][0], atoms[i][1], length, k);
        paramVector[i] = mm_float2((float) length, (float) k);
    }
    params.upload(paramVector);
    map<string, string> replacements;
    replacements["APPLY_PERIODIC"] = (force.usesPeriodicBoundaryConditions() ? "1" : "0");
    replacements["COMPUTE_FORCE"] = CommonKernelSources::harmonicBondForce;
    replacements["PARAMS"] = cc.getBondedUtilities().addArgument(params, "float2");
    cc.getBondedUtilities().addInteraction(atoms, cc.replaceStrings(CommonKernelSources::bondForce, replacements), force.getForceGroup());
    info = new ForceInfo(force);
    cc.addForce(info);
}

double CommonCalcHarmonicBondForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    return 0.0;
}

void CommonCalcHarmonicBondForceKernel::copyParametersToContext(ContextImpl& context, const HarmonicBondForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumBonds()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumBonds()/numContexts;
    if (numBonds != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of bonds has changed");
    if (numBonds == 0)
        return;

    // Record the per-bond parameters.

    vector<mm_float2> paramVector(numBonds);
    for (int i = 0; i < numBonds; i++) {
        int atom1, atom2;
        double length, k;
        force.getBondParameters(startIndex+i, atom1, atom2, length, k);
        paramVector[i] = mm_float2((float) length, (float) k);
    }
    params.upload(paramVector);

    // Mark that the current reordering may be invalid.

    cc.invalidateMolecules(info);
}
class CommonCalcCustomBondForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const CustomBondForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumBonds();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        vector<double> parameters;
        force.getBondParameters(index, particle1, particle2, parameters);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2;
        vector<double> parameters1, parameters2;
        force.getBondParameters(group1, particle1, particle2, parameters1);
        force.getBondParameters(group2, particle1, particle2, parameters2);
        for (int i = 0; i < (int) parameters1.size(); i++)
            if (parameters1[i] != parameters2[i])
                return false;
        return true;
    }
private:
    const CustomBondForce& force;
};

CommonCalcCustomBondForceKernel::~CommonCalcCustomBondForceKernel() {
    cc.setAsCurrent();
    if (params != NULL)
        delete params;
}

void CommonCalcCustomBondForceKernel::initialize(const System& system, const CustomBondForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumBonds()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumBonds()/numContexts;
    numBonds = endIndex-startIndex;
    if (numBonds == 0)
        return;
    vector<vector<int> > atoms(numBonds, vector<int>(2));
    params = new ComputeParameterSet(cc, force.getNumPerBondParameters(), numBonds, "customBondParams");
    vector<vector<float> > paramVector(numBonds);
    for (int i = 0; i < numBonds; i++) {
        vector<double> parameters;
        force.getBondParameters(startIndex+i, atoms[i][0], atoms[i][1], parameters);
        paramVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);
    info = new ForceInfo(force);
    cc.addForce(info);

    // Record information for the expressions.

    globalParamNames.resize(force.getNumGlobalParameters());
    globalParamValues.resize(force.getNumGlobalParameters());
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        globalParamNames[i] = force.getGlobalParameterName(i);
        globalParamValues[i] = (float) force.getGlobalParameterDefaultValue(i);
    }
    Lepton::ParsedExpression energyExpression = Lepton::Parser::parse(force.getEnergyFunction()).optimize();
    Lepton::ParsedExpression forceExpression = energyExpression.differentiate("r").optimize();
    map<string, Lepton::ParsedExpression> expressions;
    expressions["energy += "] = energyExpression;
    expressions["real dEdR = "] = forceExpression;

    // Create the kernels.

    map<string, string> variables;
    variables["r"] = "r";
    for (int i = 0; i < force.getNumPerBondParameters(); i++) {
        const string& name = force.getPerBondParameterName(i);
        variables[name] = "bondParams"+params->getParameterSuffix(i);
    }
    if (force.getNumGlobalParameters() > 0) {
        globals.initialize<float>(cc, force.getNumGlobalParameters(), "customBondGlobals");
        globals.upload(globalParamValues);
        string argName = cc.getBondedUtilities().addArgument(globals, "float");
        for (int i = 0; i < force.getNumGlobalParameters(); i++) {
            const string& name = force.getGlobalParameterName(i);
            string value = argName+"["+cc.intToString(i)+"]";
            variables[name] = value;
        }
    }
    for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
        string paramName = force.getEnergyParameterDerivativeName(i);
        string derivVariable = cc.getBondedUtilities().addEnergyParameterDerivative(paramName);
        Lepton::ParsedExpression derivExpression = energyExpression.differentiate(paramName).optimize();
        expressions[derivVariable+" += "] = derivExpression;
    }
    stringstream compute;
    for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
        ComputeParameterInfo& parameter = params->getParameterInfos()[i];
        string argName = cc.getBondedUtilities().addArgument(parameter.getArray(), parameter.getType());
        compute<<parameter.getType()<<" bondParams"<<(i+1)<<" = "<<argName<<"[index];\n";
    }
    vector<const TabulatedFunction*> functions;
    vector<pair<string, string> > functionNames;
    compute << cc.getExpressionUtilities().createExpressions(expressions, variables, functions, functionNames, "temp");
    map<string, string> replacements;
    replacements["APPLY_PERIODIC"] = (force.usesPeriodicBoundaryConditions() ? "1" : "0");
    replacements["COMPUTE_FORCE"] = compute.str();
    cc.getBondedUtilities().addInteraction(atoms, cc.replaceStrings(CommonKernelSources::bondForce, replacements), force.getForceGroup());
}

double CommonCalcCustomBondForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (globals.isInitialized()) {
        bool changed = false;
        for (int i = 0; i < (int) globalParamNames.size(); i++) {
            float value = (float) context.getParameter(globalParamNames[i]);
            if (value != globalParamValues[i])
                changed = true;
            globalParamValues[i] = value;
        }
        if (changed)
            globals.upload(globalParamValues);
    }
    return 0.0;
}

void CommonCalcCustomBondForceKernel::copyParametersToContext(ContextImpl& context, const CustomBondForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumBonds()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumBonds()/numContexts;
    if (numBonds != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of bonds has changed");
    if (numBonds == 0)
        return;

    // Record the per-bond parameters.

    vector<vector<float> > paramVector(numBonds);
    vector<double> parameters;
    for (int i = 0; i < numBonds; i++) {
        int atom1, atom2;
        force.getBondParameters(startIndex+i, atom1, atom2, parameters);
        paramVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);

    // Mark that the current reordering may be invalid.

    cc.invalidateMolecules(info);
}

class CommonCalcHarmonicAngleForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const HarmonicAngleForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumAngles();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2, particle3;
        double angle, k;
        force.getAngleParameters(index, particle1, particle2, particle3, angle, k);
        particles.resize(3);
        particles[0] = particle1;
        particles[1] = particle2;
        particles[2] = particle3;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2, particle3;
        double angle1, angle2, k1, k2;
        force.getAngleParameters(group1, particle1, particle2, particle3, angle1, k1);
        force.getAngleParameters(group2, particle1, particle2, particle3, angle2, k2);
        return (angle1 == angle2 && k1 == k2);
    }
private:
    const HarmonicAngleForce& force;
};

void CommonCalcHarmonicAngleForceKernel::initialize(const System& system, const HarmonicAngleForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumAngles()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumAngles()/numContexts;
    numAngles = endIndex-startIndex;
    if (numAngles == 0)
        return;
    vector<vector<int> > atoms(numAngles, vector<int>(3));
    params.initialize<mm_float2>(cc, numAngles, "angleParams");
    vector<mm_float2> paramVector(numAngles);
    for (int i = 0; i < numAngles; i++) {
        double angle, k;
        force.getAngleParameters(startIndex+i, atoms[i][0], atoms[i][1], atoms[i][2], angle, k);
        paramVector[i] = mm_float2((float) angle, (float) k);

    }
    params.upload(paramVector);
    map<string, string> replacements;
    replacements["APPLY_PERIODIC"] = (force.usesPeriodicBoundaryConditions() ? "1" : "0");
    replacements["COMPUTE_FORCE"] = CommonKernelSources::harmonicAngleForce;
    replacements["PARAMS"] = cc.getBondedUtilities().addArgument(params, "float2");
    cc.getBondedUtilities().addInteraction(atoms, cc.replaceStrings(CommonKernelSources::angleForce, replacements), force.getForceGroup());
    info = new ForceInfo(force);
    cc.addForce(info);
}

double CommonCalcHarmonicAngleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    return 0.0;
}

void CommonCalcHarmonicAngleForceKernel::copyParametersToContext(ContextImpl& context, const HarmonicAngleForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumAngles()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumAngles()/numContexts;
    if (numAngles != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of angles has changed");
    if (numAngles == 0)
        return;

    // Record the per-angle parameters.

    vector<mm_float2> paramVector(numAngles);
    for (int i = 0; i < numAngles; i++) {
        int atom1, atom2, atom3;
        double angle, k;
        force.getAngleParameters(startIndex+i, atom1, atom2, atom3, angle, k);
        paramVector[i] = mm_float2((float) angle, (float) k);
    }
    params.upload(paramVector);

    // Mark that the current reordering may be invalid.

    cc.invalidateMolecules();
}

class CommonCalcCustomAngleForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const CustomAngleForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumAngles();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2, particle3;
        vector<double> parameters;
        force.getAngleParameters(index, particle1, particle2, particle3, parameters);
        particles.resize(3);
        particles[0] = particle1;
        particles[1] = particle2;
        particles[2] = particle3;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2, particle3;
        vector<double> parameters1, parameters2;
        force.getAngleParameters(group1, particle1, particle2, particle3, parameters1);
        force.getAngleParameters(group2, particle1, particle2, particle3, parameters2);
        for (int i = 0; i < (int) parameters1.size(); i++)
            if (parameters1[i] != parameters2[i])
                return false;
        return true;
    }
private:
    const CustomAngleForce& force;
};

CommonCalcCustomAngleForceKernel::~CommonCalcCustomAngleForceKernel() {
    cc.setAsCurrent();
    if (params != NULL)
        delete params;
}

void CommonCalcCustomAngleForceKernel::initialize(const System& system, const CustomAngleForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumAngles()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumAngles()/numContexts;
    numAngles = endIndex-startIndex;
    if (numAngles == 0)
        return;
    vector<vector<int> > atoms(numAngles, vector<int>(3));
    params = new ComputeParameterSet(cc, force.getNumPerAngleParameters(), numAngles, "customAngleParams");
    vector<vector<float> > paramVector(numAngles);
    for (int i = 0; i < numAngles; i++) {
        vector<double> parameters;
        force.getAngleParameters(startIndex+i, atoms[i][0], atoms[i][1], atoms[i][2], parameters);
        paramVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);
    info = new ForceInfo(force);
    cc.addForce(info);

    // Record information for the expressions.

    globalParamNames.resize(force.getNumGlobalParameters());
    globalParamValues.resize(force.getNumGlobalParameters());
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        globalParamNames[i] = force.getGlobalParameterName(i);
        globalParamValues[i] = (float) force.getGlobalParameterDefaultValue(i);
    }
    Lepton::ParsedExpression energyExpression = Lepton::Parser::parse(force.getEnergyFunction()).optimize();
    Lepton::ParsedExpression forceExpression = energyExpression.differentiate("theta").optimize();
    map<string, Lepton::ParsedExpression> expressions;
    expressions["energy += "] = energyExpression;
    expressions["real dEdAngle = "] = forceExpression;

    // Create the kernels.

    map<string, string> variables;
    variables["theta"] = "theta";
    for (int i = 0; i < force.getNumPerAngleParameters(); i++) {
        const string& name = force.getPerAngleParameterName(i);
        variables[name] = "angleParams"+params->getParameterSuffix(i);
    }
    if (force.getNumGlobalParameters() > 0) {
        globals.initialize<float>(cc, force.getNumGlobalParameters(), "customAngleGlobals");
        globals.upload(globalParamValues);
        string argName = cc.getBondedUtilities().addArgument(globals, "float");
        for (int i = 0; i < force.getNumGlobalParameters(); i++) {
            const string& name = force.getGlobalParameterName(i);
            string value = argName+"["+cc.intToString(i)+"]";
            variables[name] = value;
        }
    }
    for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
        string paramName = force.getEnergyParameterDerivativeName(i);
        string derivVariable = cc.getBondedUtilities().addEnergyParameterDerivative(paramName);
        Lepton::ParsedExpression derivExpression = energyExpression.differentiate(paramName).optimize();
        expressions[derivVariable+" += "] = derivExpression;
    }
    stringstream compute;
    for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
        ComputeParameterInfo& parameter = params->getParameterInfos()[i];
        string argName = cc.getBondedUtilities().addArgument(parameter.getArray(), parameter.getType());
        compute<<parameter.getType()<<" angleParams"<<(i+1)<<" = "<<argName<<"[index];\n";
    }
    vector<const TabulatedFunction*> functions;
    vector<pair<string, string> > functionNames;
    compute << cc.getExpressionUtilities().createExpressions(expressions, variables, functions, functionNames, "temp");
    map<string, string> replacements;
    replacements["APPLY_PERIODIC"] = (force.usesPeriodicBoundaryConditions() ? "1" : "0");
    replacements["COMPUTE_FORCE"] = compute.str();
    cc.getBondedUtilities().addInteraction(atoms, cc.replaceStrings(CommonKernelSources::angleForce, replacements), force.getForceGroup());
}

double CommonCalcCustomAngleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (globals.isInitialized()) {
        bool changed = false;
        for (int i = 0; i < (int) globalParamNames.size(); i++) {
            float value = (float) context.getParameter(globalParamNames[i]);
            if (value != globalParamValues[i])
                changed = true;
            globalParamValues[i] = value;
        }
        if (changed)
            globals.upload(globalParamValues);
    }
    return 0.0;
}

void CommonCalcCustomAngleForceKernel::copyParametersToContext(ContextImpl& context, const CustomAngleForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumAngles()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumAngles()/numContexts;
    if (numAngles != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of angles has changed");
    if (numAngles == 0)
        return;

    // Record the per-angle parameters.

    vector<vector<float> > paramVector(numAngles);
    vector<double> parameters;
    for (int i = 0; i < numAngles; i++) {
        int atom1, atom2, atom3;
        force.getAngleParameters(startIndex+i, atom1, atom2, atom3, parameters);
        paramVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);

    // Mark that the current reordering may be invalid.

    cc.invalidateMolecules(info);
}

class CommonCalcPeriodicTorsionForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const PeriodicTorsionForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumTorsions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2, particle3, particle4, periodicity;
        double phase, k;
        force.getTorsionParameters(index, particle1, particle2, particle3, particle4, periodicity, phase, k);
        particles.resize(4);
        particles[0] = particle1;
        particles[1] = particle2;
        particles[2] = particle3;
        particles[3] = particle4;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2, particle3, particle4, periodicity1, periodicity2;
        double phase1, phase2, k1, k2;
        force.getTorsionParameters(group1, particle1, particle2, particle3, particle4, periodicity1, phase1, k1);
        force.getTorsionParameters(group2, particle1, particle2, particle3, particle4, periodicity2, phase2, k2);
        return (periodicity1 == periodicity2 && phase1 == phase2 && k1 == k2);
    }
private:
    const PeriodicTorsionForce& force;
};

void CommonCalcPeriodicTorsionForceKernel::initialize(const System& system, const PeriodicTorsionForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumTorsions()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumTorsions()/numContexts;
    numTorsions = endIndex-startIndex;
    if (numTorsions == 0)
        return;
    vector<vector<int> > atoms(numTorsions, vector<int>(4));
    params.initialize<mm_float4>(cc, numTorsions, "periodicTorsionParams");
    vector<mm_float4> paramVector(numTorsions);
    for (int i = 0; i < numTorsions; i++) {
        int periodicity;
        double phase, k;
        force.getTorsionParameters(startIndex+i, atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3], periodicity, phase, k);
        paramVector[i] = mm_float4((float) k, (float) phase, (float) periodicity, 0.0f);
    }
    params.upload(paramVector);
    map<string, string> replacements;
    replacements["APPLY_PERIODIC"] = (force.usesPeriodicBoundaryConditions() ? "1" : "0");
    replacements["COMPUTE_FORCE"] = CommonKernelSources::periodicTorsionForce;
    replacements["PARAMS"] = cc.getBondedUtilities().addArgument(params, "float4");
    cc.getBondedUtilities().addInteraction(atoms, cc.replaceStrings(CommonKernelSources::torsionForce, replacements), force.getForceGroup());
    info = new ForceInfo(force);
    cc.addForce(info);
}

double CommonCalcPeriodicTorsionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    return 0.0;
}

void CommonCalcPeriodicTorsionForceKernel::copyParametersToContext(ContextImpl& context, const PeriodicTorsionForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumTorsions()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumTorsions()/numContexts;
    if (numTorsions != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of torsions has changed");
    if (numTorsions == 0)
        return;

    // Record the per-torsion parameters.

    vector<mm_float4> paramVector(numTorsions);
    for (int i = 0; i < numTorsions; i++) {
        int atom1, atom2, atom3, atom4, periodicity;
        double phase, k;
        force.getTorsionParameters(startIndex+i, atom1, atom2, atom3, atom4, periodicity, phase, k);
        paramVector[i] = mm_float4((float) k, (float) phase, (float) periodicity, 0.0f);
    }
    params.upload(paramVector);

    // Mark that the current reordering may be invalid.

    cc.invalidateMolecules();
}

class CommonCalcRBTorsionForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const RBTorsionForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumTorsions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2, particle3, particle4;
        double c0, c1, c2, c3, c4, c5;
        force.getTorsionParameters(index, particle1, particle2, particle3, particle4, c0, c1, c2, c3, c4, c5);
        particles.resize(4);
        particles[0] = particle1;
        particles[1] = particle2;
        particles[2] = particle3;
        particles[3] = particle4;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2, particle3, particle4;
        double c0a, c0b, c1a, c1b, c2a, c2b, c3a, c3b, c4a, c4b, c5a, c5b;
        force.getTorsionParameters(group1, particle1, particle2, particle3, particle4, c0a, c1a, c2a, c3a, c4a, c5a);
        force.getTorsionParameters(group2, particle1, particle2, particle3, particle4, c0b, c1b, c2b, c3b, c4b, c5b);
        return (c0a == c0b && c1a == c1b && c2a == c2b && c3a == c3b && c4a == c4b && c5a == c5b);
    }
private:
    const RBTorsionForce& force;
};

void CommonCalcRBTorsionForceKernel::initialize(const System& system, const RBTorsionForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumTorsions()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumTorsions()/numContexts;
    numTorsions = endIndex-startIndex;
    if (numTorsions == 0)
        return;
    vector<vector<int> > atoms(numTorsions, vector<int>(4));
    params1.initialize<mm_float4>(cc, numTorsions, "rbTorsionParams1");
    params2.initialize<mm_float2>(cc, numTorsions, "rbTorsionParams2");
    vector<mm_float4> paramVector1(numTorsions);
    vector<mm_float2> paramVector2(numTorsions);
    for (int i = 0; i < numTorsions; i++) {
        double c0, c1, c2, c3, c4, c5;
        force.getTorsionParameters(startIndex+i, atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3], c0, c1, c2, c3, c4, c5);
        paramVector1[i] = mm_float4((float) c0, (float) c1, (float) c2, (float) c3);
        paramVector2[i] = mm_float2((float) c4, (float) c5);

    }
    params1.upload(paramVector1);
    params2.upload(paramVector2);
    map<string, string> replacements;
    replacements["APPLY_PERIODIC"] = (force.usesPeriodicBoundaryConditions() ? "1" : "0");
    replacements["COMPUTE_FORCE"] = CommonKernelSources::rbTorsionForce;
    replacements["PARAMS1"] = cc.getBondedUtilities().addArgument(params1, "float4");
    replacements["PARAMS2"] = cc.getBondedUtilities().addArgument(params2, "float2");
    cc.getBondedUtilities().addInteraction(atoms, cc.replaceStrings(CommonKernelSources::torsionForce, replacements), force.getForceGroup());
    info = new ForceInfo(force);
    cc.addForce(info);
}

double CommonCalcRBTorsionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    return 0.0;
}

void CommonCalcRBTorsionForceKernel::copyParametersToContext(ContextImpl& context, const RBTorsionForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumTorsions()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumTorsions()/numContexts;
    if (numTorsions != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of torsions has changed");
    if (numTorsions == 0)
        return;

    // Record the per-torsion parameters.

    vector<mm_float4> paramVector1(numTorsions);
    vector<mm_float2> paramVector2(numTorsions);
    for (int i = 0; i < numTorsions; i++) {
        int atom1, atom2, atom3, atom4;
        double c0, c1, c2, c3, c4, c5;
        force.getTorsionParameters(startIndex+i, atom1, atom2, atom3, atom4, c0, c1, c2, c3, c4, c5);
        paramVector1[i] = mm_float4((float) c0, (float) c1, (float) c2, (float) c3);
        paramVector2[i] = mm_float2((float) c4, (float) c5);
    }
    params1.upload(paramVector1);
    params2.upload(paramVector2);

    // Mark that the current reordering may be invalid.

    cc.invalidateMolecules();
}

class CommonCalcCustomTorsionForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const CustomTorsionForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumTorsions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2, particle3, particle4;
        vector<double> parameters;
        force.getTorsionParameters(index, particle1, particle2, particle3, particle4, parameters);
        particles.resize(4);
        particles[0] = particle1;
        particles[1] = particle2;
        particles[2] = particle3;
        particles[3] = particle4;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2, particle3, particle4;
        vector<double> parameters1, parameters2;
        force.getTorsionParameters(group1, particle1, particle2, particle3, particle4, parameters1);
        force.getTorsionParameters(group2, particle1, particle2, particle3, particle4, parameters2);
        for (int i = 0; i < (int) parameters1.size(); i++)
            if (parameters1[i] != parameters2[i])
                return false;
        return true;
    }
private:
    const CustomTorsionForce& force;
};

CommonCalcCustomTorsionForceKernel::~CommonCalcCustomTorsionForceKernel() {
    if (params != NULL)
        delete params;
}

void CommonCalcCustomTorsionForceKernel::initialize(const System& system, const CustomTorsionForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumTorsions()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumTorsions()/numContexts;
    numTorsions = endIndex-startIndex;
    if (numTorsions == 0)
        return;
    vector<vector<int> > atoms(numTorsions, vector<int>(4));
    params = new ComputeParameterSet(cc, force.getNumPerTorsionParameters(), numTorsions, "customTorsionParams");
    vector<vector<float> > paramVector(numTorsions);
    for (int i = 0; i < numTorsions; i++) {
        vector<double> parameters;
        force.getTorsionParameters(startIndex+i, atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3], parameters);
        paramVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);
    info = new ForceInfo(force);
    cc.addForce(info);

    // Record information for the expressions.

    globalParamNames.resize(force.getNumGlobalParameters());
    globalParamValues.resize(force.getNumGlobalParameters());
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        globalParamNames[i] = force.getGlobalParameterName(i);
        globalParamValues[i] = (float) force.getGlobalParameterDefaultValue(i);
    }
    Lepton::ParsedExpression energyExpression = Lepton::Parser::parse(force.getEnergyFunction()).optimize();
    Lepton::ParsedExpression forceExpression = energyExpression.differentiate("theta").optimize();
    map<string, Lepton::ParsedExpression> expressions;
    expressions["energy += "] = energyExpression;
    expressions["real dEdAngle = "] = forceExpression;

    // Create the kernels.

    map<string, string> variables;
    variables["theta"] = "theta";
    for (int i = 0; i < force.getNumPerTorsionParameters(); i++) {
        const string& name = force.getPerTorsionParameterName(i);
        variables[name] = "torsionParams"+params->getParameterSuffix(i);
    }
    if (force.getNumGlobalParameters() > 0) {
        globals.initialize<float>(cc, force.getNumGlobalParameters(), "customTorsionGlobals");
        globals.upload(globalParamValues);
        string argName = cc.getBondedUtilities().addArgument(globals, "float");
        for (int i = 0; i < force.getNumGlobalParameters(); i++) {
            const string& name = force.getGlobalParameterName(i);
            string value = argName+"["+cc.intToString(i)+"]";
            variables[name] = value;
        }
    }
    for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
        string paramName = force.getEnergyParameterDerivativeName(i);
        string derivVariable = cc.getBondedUtilities().addEnergyParameterDerivative(paramName);
        Lepton::ParsedExpression derivExpression = energyExpression.differentiate(paramName).optimize();
        expressions[derivVariable+" += "] = derivExpression;
    }
    stringstream compute;
    for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
        ComputeParameterInfo& parameter = params->getParameterInfos()[i];
        string argName = cc.getBondedUtilities().addArgument(parameter.getArray(), parameter.getType());
        compute<<parameter.getType()<<" torsionParams"<<(i+1)<<" = "<<argName<<"[index];\n";
    }
    vector<const TabulatedFunction*> functions;
    vector<pair<string, string> > functionNames;
    compute << cc.getExpressionUtilities().createExpressions(expressions, variables, functions, functionNames, "temp");
    map<string, string> replacements;
    replacements["APPLY_PERIODIC"] = (force.usesPeriodicBoundaryConditions() ? "1" : "0");
    replacements["COMPUTE_FORCE"] = compute.str();
    cc.getBondedUtilities().addInteraction(atoms, cc.replaceStrings(CommonKernelSources::torsionForce, replacements), force.getForceGroup());
}

double CommonCalcCustomTorsionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (globals.isInitialized()) {
        bool changed = false;
        for (int i = 0; i < (int) globalParamNames.size(); i++) {
            float value = (float) context.getParameter(globalParamNames[i]);
            if (value != globalParamValues[i])
                changed = true;
            globalParamValues[i] = value;
        }
        if (changed)
            globals.upload(globalParamValues);
    }
    return 0.0;
}

void CommonCalcCustomTorsionForceKernel::copyParametersToContext(ContextImpl& context, const CustomTorsionForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumTorsions()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumTorsions()/numContexts;
    if (numTorsions != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of torsions has changed");
    if (numTorsions == 0)
        return;

    // Record the per-torsion parameters.

    vector<vector<float> > paramVector(numTorsions);
    vector<double> parameters;
    for (int i = 0; i < numTorsions; i++) {
        int atom1, atom2, atom3, atom4;
        force.getTorsionParameters(startIndex+i, atom1, atom2, atom3, atom4, parameters);
        paramVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);

    // Mark that the current reordering may be invalid.

    cc.invalidateMolecules(info);
}

class CommonCalcCMAPTorsionForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const CMAPTorsionForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumTorsions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int map, a1, a2, a3, a4, b1, b2, b3, b4;
        force.getTorsionParameters(index, map, a1, a2, a3, a4, b1, b2, b3, b4);
        particles.resize(8);
        particles[0] = a1;
        particles[1] = a2;
        particles[2] = a3;
        particles[3] = a4;
        particles[4] = b1;
        particles[5] = b2;
        particles[6] = b3;
        particles[7] = b4;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int map1, map2, a1, a2, a3, a4, b1, b2, b3, b4;
        force.getTorsionParameters(group1, map1, a1, a2, a3, a4, b1, b2, b3, b4);
        force.getTorsionParameters(group2, map2, a1, a2, a3, a4, b1, b2, b3, b4);
        return (map1 == map2);
    }
private:
    const CMAPTorsionForce& force;
};

void CommonCalcCMAPTorsionForceKernel::initialize(const System& system, const CMAPTorsionForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumTorsions()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumTorsions()/numContexts;
    numTorsions = endIndex-startIndex;
    if (numTorsions == 0)
        return;
    int numMaps = force.getNumMaps();
    vector<mm_float4> coeffVec;
    mapPositionsVec.resize(numMaps);
    vector<double> energy;
    vector<vector<double> > c;
    int currentPosition = 0;
    for (int i = 0; i < numMaps; i++) {
        int size;
        force.getMapParameters(i, size, energy);
        CMAPTorsionForceImpl::calcMapDerivatives(size, energy, c);
        mapPositionsVec[i] = mm_int2(currentPosition, size);
        currentPosition += 4*size*size;
        for (int j = 0; j < size*size; j++) {
            coeffVec.push_back(mm_float4((float) c[j][0], (float) c[j][1], (float) c[j][2], (float) c[j][3]));
            coeffVec.push_back(mm_float4((float) c[j][4], (float) c[j][5], (float) c[j][6], (float) c[j][7]));
            coeffVec.push_back(mm_float4((float) c[j][8], (float) c[j][9], (float) c[j][10], (float) c[j][11]));
            coeffVec.push_back(mm_float4((float) c[j][12], (float) c[j][13], (float) c[j][14], (float) c[j][15]));
        }
    }
    vector<vector<int> > atoms(numTorsions, vector<int>(8));
    vector<int> torsionMapsVec(numTorsions);
    for (int i = 0; i < numTorsions; i++)
        force.getTorsionParameters(startIndex+i, torsionMapsVec[i], atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3], atoms[i][4], atoms[i][5], atoms[i][6], atoms[i][7]);
    coefficients.initialize<mm_float4>(cc, coeffVec.size(), "cmapTorsionCoefficients");
    mapPositions.initialize<mm_int2>(cc, numMaps, "cmapTorsionMapPositions");
    torsionMaps.initialize<int>(cc, numTorsions, "cmapTorsionMaps");
    coefficients.upload(coeffVec);
    mapPositions.upload(mapPositionsVec);
    torsionMaps.upload(torsionMapsVec);
    map<string, string> replacements;
    replacements["APPLY_PERIODIC"] = (force.usesPeriodicBoundaryConditions() ? "1" : "0");
    replacements["COEFF"] = cc.getBondedUtilities().addArgument(coefficients, "float4");
    replacements["MAP_POS"] = cc.getBondedUtilities().addArgument(mapPositions, "int2");
    replacements["MAPS"] = cc.getBondedUtilities().addArgument(torsionMaps, "int");
    cc.getBondedUtilities().addInteraction(atoms, cc.replaceStrings(CommonKernelSources::cmapTorsionForce, replacements), force.getForceGroup());
    info = new ForceInfo(force);
    cc.addForce(info);
}

double CommonCalcCMAPTorsionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    return 0.0;
}

void CommonCalcCMAPTorsionForceKernel::copyParametersToContext(ContextImpl& context, const CMAPTorsionForce& force) {
    int numMaps = force.getNumMaps();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumTorsions()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumTorsions()/numContexts;
    numTorsions = endIndex-startIndex;
    if (mapPositions.getSize() != numMaps)
        throw OpenMMException("updateParametersInContext: The number of maps has changed");
    if (torsionMaps.getSize() != numTorsions)
        throw OpenMMException("updateParametersInContext: The number of CMAP torsions has changed");

    // Update the maps.

    vector<mm_float4> coeffVec;
    vector<double> energy;
    vector<vector<double> > c;
    int currentPosition = 0;
    for (int i = 0; i < numMaps; i++) {
        int size;
        force.getMapParameters(i, size, energy);
        if (size != mapPositionsVec[i].y)
            throw OpenMMException("updateParametersInContext: The size of a map has changed");
        CMAPTorsionForceImpl::calcMapDerivatives(size, energy, c);
        currentPosition += 4*size*size;
        for (int j = 0; j < size*size; j++) {
            coeffVec.push_back(mm_float4((float) c[j][0], (float) c[j][1], (float) c[j][2], (float) c[j][3]));
            coeffVec.push_back(mm_float4((float) c[j][4], (float) c[j][5], (float) c[j][6], (float) c[j][7]));
            coeffVec.push_back(mm_float4((float) c[j][8], (float) c[j][9], (float) c[j][10], (float) c[j][11]));
            coeffVec.push_back(mm_float4((float) c[j][12], (float) c[j][13], (float) c[j][14], (float) c[j][15]));
        }
    }
    coefficients.upload(coeffVec);

    // Update the indices.

    vector<int> torsionMapsVec(numTorsions);
    for (int i = 0; i < numTorsions; i++) {
        int index[8];
        force.getTorsionParameters(i, torsionMapsVec[i], index[0], index[1], index[2], index[3], index[4], index[5], index[6], index[7]);
    }
    torsionMaps.upload(torsionMapsVec);
}
class CommonCalcCustomExternalForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const CustomExternalForce& force, int numParticles) : force(force), indices(numParticles, -1) {
        vector<double> params;
        for (int i = 0; i < force.getNumParticles(); i++) {
            int particle;
            force.getParticleParameters(i, particle, params);
            indices[particle] = i;
        }
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        particle1 = indices[particle1];
        particle2 = indices[particle2];
        if (particle1 == -1 && particle2 == -1)
            return true;
        if (particle1 == -1 || particle2 == -1)
            return false;
        int temp;
        vector<double> params1;
        vector<double> params2;
        force.getParticleParameters(particle1, temp, params1);
        force.getParticleParameters(particle2, temp, params2);
        for (int i = 0; i < (int) params1.size(); i++)
            if (params1[i] != params2[i])
                return false;
        return true;
    }
private:
    const CustomExternalForce& force;
    vector<int> indices;
};

CommonCalcCustomExternalForceKernel::~CommonCalcCustomExternalForceKernel() {
    cc.setAsCurrent();
    if (params != NULL)
        delete params;
}

void CommonCalcCustomExternalForceKernel::initialize(const System& system, const CustomExternalForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumParticles()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumParticles()/numContexts;
    numParticles = endIndex-startIndex;
    if (numParticles == 0)
        return;
    vector<vector<int> > atoms(numParticles, vector<int>(1));
    params = new ComputeParameterSet(cc, force.getNumPerParticleParameters(), numParticles, "customExternalParams");
    vector<vector<float> > paramVector(numParticles);
    for (int i = 0; i < numParticles; i++) {
        vector<double> parameters;
        force.getParticleParameters(startIndex+i, atoms[i][0], parameters);
        paramVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);
    info = new ForceInfo(force, system.getNumParticles());
    cc.addForce(info);

    // Record information for the expressions.

    globalParamNames.resize(force.getNumGlobalParameters());
    globalParamValues.resize(force.getNumGlobalParameters());
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        globalParamNames[i] = force.getGlobalParameterName(i);
        globalParamValues[i] = (float) force.getGlobalParameterDefaultValue(i);
    }
    map<string, Lepton::CustomFunction*> customFunctions;
    customFunctions["periodicdistance"] = cc.getExpressionUtilities().getPeriodicDistancePlaceholder();
    Lepton::ParsedExpression energyExpression = Lepton::Parser::parse(force.getEnergyFunction(), customFunctions).optimize();
    Lepton::ParsedExpression forceExpressionX = energyExpression.differentiate("x").optimize();
    Lepton::ParsedExpression forceExpressionY = energyExpression.differentiate("y").optimize();
    Lepton::ParsedExpression forceExpressionZ = energyExpression.differentiate("z").optimize();
    map<string, Lepton::ParsedExpression> expressions;
    expressions["energy += "] = energyExpression;
    expressions["real dEdX = "] = forceExpressionX;
    expressions["real dEdY = "] = forceExpressionY;
    expressions["real dEdZ = "] = forceExpressionZ;

    // Create the kernels.

    map<string, string> variables;
    variables["x"] = "pos1.x";
    variables["y"] = "pos1.y";
    variables["z"] = "pos1.z";
    for (int i = 0; i < force.getNumPerParticleParameters(); i++) {
        const string& name = force.getPerParticleParameterName(i);
        variables[name] = "particleParams"+params->getParameterSuffix(i);
    }
    if (force.getNumGlobalParameters() > 0) {
        globals.initialize<float>(cc, force.getNumGlobalParameters(), "customExternalGlobals");
        globals.upload(globalParamValues);
        string argName = cc.getBondedUtilities().addArgument(globals, "float");
        for (int i = 0; i < force.getNumGlobalParameters(); i++) {
            const string& name = force.getGlobalParameterName(i);
            string value = argName+"["+cc.intToString(i)+"]";
            variables[name] = value;
        }
    }
    stringstream compute;
    for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
        ComputeParameterInfo& parameter = params->getParameterInfos()[i];
        string argName = cc.getBondedUtilities().addArgument(parameter.getArray(), parameter.getType());
        compute<<parameter.getType()<<" particleParams"<<(i+1)<<" = "<<argName<<"[index];\n";
    }
    vector<const TabulatedFunction*> functions;
    vector<pair<string, string> > functionNames;
    compute << cc.getExpressionUtilities().createExpressions(expressions, variables, functions, functionNames, "temp");
    map<string, string> replacements;
    replacements["COMPUTE_FORCE"] = compute.str();
    cc.getBondedUtilities().addInteraction(atoms, cc.replaceStrings(CommonKernelSources::customExternalForce, replacements), force.getForceGroup());
}

double CommonCalcCustomExternalForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (globals.isInitialized()) {
        bool changed = false;
        for (int i = 0; i < (int) globalParamNames.size(); i++) {
            float value = (float) context.getParameter(globalParamNames[i]);
            if (value != globalParamValues[i])
                changed = true;
            globalParamValues[i] = value;
        }
        if (changed)
            globals.upload(globalParamValues);
    }
    return 0.0;
}

void CommonCalcCustomExternalForceKernel::copyParametersToContext(ContextImpl& context, const CustomExternalForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumParticles()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumParticles()/numContexts;
    if (numParticles != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    if (numParticles == 0)
        return;

    // Record the per-particle parameters.

    vector<vector<float> > paramVector(numParticles);
    vector<double> parameters;
    for (int i = 0; i < numParticles; i++) {
        int particle;
        force.getParticleParameters(startIndex+i, particle, parameters);
        paramVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);

    // Mark that the current reordering may be invalid.

    cc.invalidateMolecules(info);
}

class CommonCalcCustomCompoundBondForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const CustomCompoundBondForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumBonds();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        vector<double> parameters;
        force.getBondParameters(index, particles, parameters);
    }
    bool areGroupsIdentical(int group1, int group2) {
        vector<int> particles;
        vector<double> parameters1, parameters2;
        force.getBondParameters(group1, particles, parameters1);
        force.getBondParameters(group2, particles, parameters2);
        for (int i = 0; i < (int) parameters1.size(); i++)
            if (parameters1[i] != parameters2[i])
                return false;
        return true;
    }
private:
    const CustomCompoundBondForce& force;
};

CommonCalcCustomCompoundBondForceKernel::~CommonCalcCustomCompoundBondForceKernel() {
    cc.setAsCurrent();
    if (params != NULL)
        delete params;
}

void CommonCalcCustomCompoundBondForceKernel::initialize(const System& system, const CustomCompoundBondForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumBonds()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumBonds()/numContexts;
    numBonds = endIndex-startIndex;
    if (numBonds == 0)
        return;
    int particlesPerBond = force.getNumParticlesPerBond();
    vector<vector<int> > atoms(numBonds, vector<int>(particlesPerBond));
    params = new ComputeParameterSet(cc, force.getNumPerBondParameters(), numBonds, "customCompoundBondParams");
    vector<vector<float> > paramVector(numBonds);
    for (int i = 0; i < numBonds; i++) {
        vector<double> parameters;
        force.getBondParameters(startIndex+i, atoms[i], parameters);
        paramVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);
    info = new ForceInfo(force);
    cc.addForce(info);

    // Record the tabulated functions.

    map<string, Lepton::CustomFunction*> functions;
    vector<pair<string, string> > functionDefinitions;
    vector<const TabulatedFunction*> functionList;
    tabulatedFunctions.resize(force.getNumTabulatedFunctions());
    for (int i = 0; i < force.getNumTabulatedFunctions(); i++) {
        functionList.push_back(&force.getTabulatedFunction(i));
        string name = force.getTabulatedFunctionName(i);
        functions[name] = cc.getExpressionUtilities().getFunctionPlaceholder(force.getTabulatedFunction(i));
        int width;
        vector<float> f = cc.getExpressionUtilities().computeFunctionCoefficients(force.getTabulatedFunction(i), width);
        tabulatedFunctions[i].initialize<float>(cc, f.size(), "TabulatedFunction");
        tabulatedFunctions[i].upload(f);
        string arrayName = cc.getBondedUtilities().addArgument(tabulatedFunctions[i], width == 1 ? "float" : "float"+cc.intToString(width));
        functionDefinitions.push_back(make_pair(name, arrayName));
    }

    // Record information about parameters.

    globalParamNames.resize(force.getNumGlobalParameters());
    globalParamValues.resize(force.getNumGlobalParameters());
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        globalParamNames[i] = force.getGlobalParameterName(i);
        globalParamValues[i] = (float) force.getGlobalParameterDefaultValue(i);
    }
    map<string, string> variables;
    for (int i = 0; i < particlesPerBond; i++) {
        string index = cc.intToString(i+1);
        variables["x"+index] = "pos"+index+".x";
        variables["y"+index] = "pos"+index+".y";
        variables["z"+index] = "pos"+index+".z";
    }
    for (int i = 0; i < force.getNumPerBondParameters(); i++) {
        const string& name = force.getPerBondParameterName(i);
        variables[name] = "bondParams"+params->getParameterSuffix(i);
    }
    if (force.getNumGlobalParameters() > 0) {
        globals.initialize<float>(cc, force.getNumGlobalParameters(), "customCompoundBondGlobals");
        globals.upload(globalParamValues);
        string argName = cc.getBondedUtilities().addArgument(globals, "float");
        for (int i = 0; i < force.getNumGlobalParameters(); i++) {
            const string& name = force.getGlobalParameterName(i);
            string value = argName+"["+cc.intToString(i)+"]";
            variables[name] = value;
        }
    }

    // Now to generate the kernel.  First, it needs to calculate all distances, angles,
    // and dihedrals the expression depends on.

    map<string, vector<int> > distances;
    map<string, vector<int> > angles;
    map<string, vector<int> > dihedrals;
    Lepton::ParsedExpression energyExpression = CustomCompoundBondForceImpl::prepareExpression(force, functions, distances, angles, dihedrals);
    map<string, Lepton::ParsedExpression> forceExpressions;
    set<string> computedDeltas;
    vector<string> atomNames, posNames;
    for (int i = 0; i < particlesPerBond; i++) {
        string index = cc.intToString(i+1);
        atomNames.push_back("P"+index);
        posNames.push_back("pos"+index);
    }
    stringstream compute;
    int index = 0;
    for (auto& distance : distances) {
        const vector<int>& atoms = distance.second;
        string deltaName = atomNames[atoms[0]]+atomNames[atoms[1]];
        if (computedDeltas.count(deltaName) == 0) {
            compute<<"real4 delta"<<deltaName<<" = ccb_delta("<<posNames[atoms[0]]<<", "<<posNames[atoms[1]]<<", "<<force.usesPeriodicBoundaryConditions()<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName);
        }
        compute<<"real r_"<<deltaName<<" = SQRT(delta"<<deltaName<<".w);\n";
        variables[distance.first] = "r_"+deltaName;
        forceExpressions["real dEdDistance"+cc.intToString(index)+" = "] = energyExpression.differentiate(distance.first).optimize();
        index++;
    }
    index = 0;
    for (auto& angle : angles) {
        const vector<int>& atoms = angle.second;
        string deltaName1 = atomNames[atoms[1]]+atomNames[atoms[0]];
        string deltaName2 = atomNames[atoms[1]]+atomNames[atoms[2]];
        string angleName = "angle_"+atomNames[atoms[0]]+atomNames[atoms[1]]+atomNames[atoms[2]];
        if (computedDeltas.count(deltaName1) == 0) {
            compute<<"real4 delta"<<deltaName1<<" = ccb_delta("<<posNames[atoms[1]]<<", "<<posNames[atoms[0]]<<", "<<force.usesPeriodicBoundaryConditions()<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName1);
        }
        if (computedDeltas.count(deltaName2) == 0) {
            compute<<"real4 delta"<<deltaName2<<" = ccb_delta("<<posNames[atoms[1]]<<", "<<posNames[atoms[2]]<<", "<<force.usesPeriodicBoundaryConditions()<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName2);
        }
        compute<<"real "<<angleName<<" = ccb_computeAngle(delta"<<deltaName1<<", delta"<<deltaName2<<");\n";
        variables[angle.first] = angleName;
        forceExpressions["real dEdAngle"+cc.intToString(index)+" = "] = energyExpression.differentiate(angle.first).optimize();
        index++;
    }
    index = 0;
    for (auto& dihedral : dihedrals) {
        const vector<int>& atoms = dihedral.second;
        string deltaName1 = atomNames[atoms[0]]+atomNames[atoms[1]];
        string deltaName2 = atomNames[atoms[2]]+atomNames[atoms[1]];
        string deltaName3 = atomNames[atoms[2]]+atomNames[atoms[3]];
        string crossName1 = "cross_"+deltaName1+"_"+deltaName2;
        string crossName2 = "cross_"+deltaName2+"_"+deltaName3;
        string dihedralName = "dihedral_"+atomNames[atoms[0]]+atomNames[atoms[1]]+atomNames[atoms[2]]+atomNames[atoms[3]];
        if (computedDeltas.count(deltaName1) == 0) {
            compute<<"real4 delta"<<deltaName1<<" = ccb_delta("<<posNames[atoms[0]]<<", "<<posNames[atoms[1]]<<", "<<force.usesPeriodicBoundaryConditions()<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName1);
        }
        if (computedDeltas.count(deltaName2) == 0) {
            compute<<"real4 delta"<<deltaName2<<" = ccb_delta("<<posNames[atoms[2]]<<", "<<posNames[atoms[1]]<<", "<<force.usesPeriodicBoundaryConditions()<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName2);
        }
        if (computedDeltas.count(deltaName3) == 0) {
            compute<<"real4 delta"<<deltaName3<<" = ccb_delta("<<posNames[atoms[2]]<<", "<<posNames[atoms[3]]<<", "<<force.usesPeriodicBoundaryConditions()<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName3);
        }
        compute<<"real4 "<<crossName1<<" = ccb_computeCross(delta"<<deltaName1<<", delta"<<deltaName2<<");\n";
        compute<<"real4 "<<crossName2<<" = ccb_computeCross(delta"<<deltaName2<<", delta"<<deltaName3<<");\n";
        compute<<"real "<<dihedralName<<" = ccb_computeAngle("<<crossName1<<", "<<crossName2<<");\n";
        compute<<dihedralName<<" *= (delta"<<deltaName1<<".x*"<<crossName2<<".x + delta"<<deltaName1<<".y*"<<crossName2<<".y + delta"<<deltaName1<<".z*"<<crossName2<<".z < 0 ? -1 : 1);\n";
        variables[dihedral.first] = dihedralName;
        forceExpressions["real dEdDihedral"+cc.intToString(index)+" = "] = energyExpression.differentiate(dihedral.first).optimize();
        index++;
    }

    // Now evaluate the expressions.

    for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
        ComputeParameterInfo& parameter = params->getParameterInfos()[i];
        string argName = cc.getBondedUtilities().addArgument(parameter.getArray(), parameter.getType());
        compute<<parameter.getType()<<" bondParams"<<(i+1)<<" = "<<argName<<"[index];\n";
    }
    forceExpressions["energy += "] = energyExpression;
    for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
        string paramName = force.getEnergyParameterDerivativeName(i);
        string derivVariable = cc.getBondedUtilities().addEnergyParameterDerivative(paramName);
        Lepton::ParsedExpression derivExpression = energyExpression.differentiate(paramName).optimize();
        forceExpressions[derivVariable+" += "] = derivExpression;
    }
    compute << cc.getExpressionUtilities().createExpressions(forceExpressions, variables, functionList, functionDefinitions, "temp");

    // Finally, apply forces to atoms.

    vector<string> forceNames;
    for (int i = 0; i < particlesPerBond; i++) {
        string istr = cc.intToString(i+1);
        string forceName = "force"+istr;
        forceNames.push_back(forceName);
        compute<<"real3 "<<forceName<<" = make_real3(0);\n";
        compute<<"{\n";
        Lepton::ParsedExpression forceExpressionX = energyExpression.differentiate("x"+istr).optimize();
        Lepton::ParsedExpression forceExpressionY = energyExpression.differentiate("y"+istr).optimize();
        Lepton::ParsedExpression forceExpressionZ = energyExpression.differentiate("z"+istr).optimize();
        map<string, Lepton::ParsedExpression> expressions;
        if (!isZeroExpression(forceExpressionX))
            expressions[forceName+".x -= "] = forceExpressionX;
        if (!isZeroExpression(forceExpressionY))
            expressions[forceName+".y -= "] = forceExpressionY;
        if (!isZeroExpression(forceExpressionZ))
            expressions[forceName+".z -= "] = forceExpressionZ;
        if (expressions.size() > 0)
            compute<<cc.getExpressionUtilities().createExpressions(expressions, variables, functionList, functionDefinitions, "coordtemp");
        compute<<"}\n";
    }
    index = 0;
    for (auto& distance : distances) {
        const vector<int>& atoms = distance.second;
        string deltaName = atomNames[atoms[0]]+atomNames[atoms[1]];
        string value = "(dEdDistance"+cc.intToString(index)+"/r_"+deltaName+")*trimTo3(delta"+deltaName+")";
        compute<<forceNames[atoms[0]]<<" += "<<"-"<<value<<";\n";
        compute<<forceNames[atoms[1]]<<" += "<<value<<";\n";
        index++;
    }
    index = 0;
    for (auto& angle : angles) {
        const vector<int>& atoms = angle.second;
        string deltaName1 = atomNames[atoms[1]]+atomNames[atoms[0]];
        string deltaName2 = atomNames[atoms[1]]+atomNames[atoms[2]];
        compute<<"{\n";
        compute<<"real3 crossProd = trimTo3(cross(delta"<<deltaName2<<", delta"<<deltaName1<<"));\n";
        compute<<"real lengthCross = max(SQRT(dot(crossProd, crossProd)), (real) 1e-6f);\n";
        compute<<"real3 deltaCross0 = -cross(trimTo3(delta"<<deltaName1<<"), crossProd)*dEdAngle"<<cc.intToString(index)<<"/(delta"<<deltaName1<<".w*lengthCross);\n";
        compute<<"real3 deltaCross2 = cross(trimTo3(delta"<<deltaName2<<"), crossProd)*dEdAngle"<<cc.intToString(index)<<"/(delta"<<deltaName2<<".w*lengthCross);\n";
        compute<<"real3 deltaCross1 = -(deltaCross0+deltaCross2);\n";
        compute<<forceNames[atoms[0]]<<" += deltaCross0;\n";
        compute<<forceNames[atoms[1]]<<" += deltaCross1;\n";
        compute<<forceNames[atoms[2]]<<" += deltaCross2;\n";
        compute<<"}\n";
        index++;
    }
    index = 0;
    for (auto& dihedral : dihedrals) {
        const vector<int>& atoms = dihedral.second;
        string deltaName1 = atomNames[atoms[0]]+atomNames[atoms[1]];
        string deltaName2 = atomNames[atoms[2]]+atomNames[atoms[1]];
        string deltaName3 = atomNames[atoms[2]]+atomNames[atoms[3]];
        string crossName1 = "cross_"+deltaName1+"_"+deltaName2;
        string crossName2 = "cross_"+deltaName2+"_"+deltaName3;
        compute<<"{\n";
        compute<<"real r = SQRT(delta"<<deltaName2<<".w);\n";
        compute<<"real4 ff;\n";
        compute<<"ff.x = (-dEdDihedral"<<cc.intToString(index)<<"*r)/"<<crossName1<<".w;\n";
        compute<<"ff.y = (delta"<<deltaName1<<".x*delta"<<deltaName2<<".x + delta"<<deltaName1<<".y*delta"<<deltaName2<<".y + delta"<<deltaName1<<".z*delta"<<deltaName2<<".z)/delta"<<deltaName2<<".w;\n";
        compute<<"ff.z = (delta"<<deltaName3<<".x*delta"<<deltaName2<<".x + delta"<<deltaName3<<".y*delta"<<deltaName2<<".y + delta"<<deltaName3<<".z*delta"<<deltaName2<<".z)/delta"<<deltaName2<<".w;\n";
        compute<<"ff.w = (dEdDihedral"<<cc.intToString(index)<<"*r)/"<<crossName2<<".w;\n";
        compute<<"real3 internalF0 = ff.x*trimTo3("<<crossName1<<");\n";
        compute<<"real3 internalF3 = ff.w*trimTo3("<<crossName2<<");\n";
        compute<<"real3 s = ff.y*internalF0 - ff.z*internalF3;\n";
        compute<<forceNames[atoms[0]]<<" += internalF0;\n";
        compute<<forceNames[atoms[1]]<<" += s-internalF0;\n";
        compute<<forceNames[atoms[2]]<<" += -s-internalF3;\n";
        compute<<forceNames[atoms[3]]<<" += internalF3;\n";
        compute<<"}\n";
        index++;
    }
    cc.getBondedUtilities().addInteraction(atoms, compute.str(), force.getForceGroup());
    map<string, string> replacements;
    replacements["M_PI"] = cc.doubleToString(M_PI);
    cc.getBondedUtilities().addPrefixCode(cc.replaceStrings(CommonKernelSources::customCompoundBond, replacements));
}

double CommonCalcCustomCompoundBondForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (globals.isInitialized()) {
        bool changed = false;
        for (int i = 0; i < (int) globalParamNames.size(); i++) {
            float value = (float) context.getParameter(globalParamNames[i]);
            if (value != globalParamValues[i])
                changed = true;
            globalParamValues[i] = value;
        }
        if (changed)
            globals.upload(globalParamValues);
    }
    return 0.0;
}

void CommonCalcCustomCompoundBondForceKernel::copyParametersToContext(ContextImpl& context, const CustomCompoundBondForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumBonds()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumBonds()/numContexts;
    if (numBonds != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of bonds has changed");
    if (numBonds == 0)
        return;

    // Record the per-bond parameters.

    vector<vector<float> > paramVector(numBonds);
    vector<int> particles;
    vector<double> parameters;
    for (int i = 0; i < numBonds; i++) {
        force.getBondParameters(startIndex+i, particles, parameters);
        paramVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);

    // Mark that the current reordering may be invalid.

    cc.invalidateMolecules(info);
}

class CommonCalcCustomCentroidBondForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const CustomCentroidBondForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumBonds();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        vector<double> parameters;
        vector<int> groups;
        force.getBondParameters(index, groups, parameters);
        for (int group : groups) {
            vector<int> groupParticles;
            vector<double> weights;
            force.getGroupParameters(group, groupParticles, weights);
            particles.insert(particles.end(), groupParticles.begin(), groupParticles.end());
        }
    }
    bool areGroupsIdentical(int group1, int group2) {
        vector<int> groups1, groups2;
        vector<double> parameters1, parameters2;
        force.getBondParameters(group1, groups1, parameters1);
        force.getBondParameters(group2, groups2, parameters2);
        for (int i = 0; i < (int) parameters1.size(); i++)
            if (parameters1[i] != parameters2[i])
                return false;
        for (int i = 0; i < groups1.size(); i++) {
            vector<int> groupParticles;
            vector<double> weights1, weights2;
            force.getGroupParameters(groups1[i], groupParticles, weights1);
            force.getGroupParameters(groups2[i], groupParticles, weights2);
            if (weights1.size() != weights2.size())
                return false;
            for (int j = 0; j < weights1.size(); j++)
                if (weights1[j] != weights2[j])
                    return false;
        }
        return true;
    }
private:
    const CustomCentroidBondForce& force;
};

CommonCalcCustomCentroidBondForceKernel::~CommonCalcCustomCentroidBondForceKernel() {
    cc.setAsCurrent();
    if (params != NULL)
        delete params;
}

void CommonCalcCustomCentroidBondForceKernel::initialize(const System& system, const CustomCentroidBondForce& force) {
    cc.setAsCurrent();
    numBonds = force.getNumBonds();
    if (numBonds == 0)
        return;
    if (!cc.getSupports64BitGlobalAtomics())
        throw OpenMMException("CustomCentroidBondForce requires a device that supports 64 bit atomic operations");
    info = new ForceInfo(force);
    cc.addForce(info);

    // Record the groups.

    numGroups = force.getNumGroups();
    vector<int> groupParticleVec;
    vector<double> groupWeightVec;
    vector<int> groupOffsetVec;
    groupOffsetVec.push_back(0);
    for (int i = 0; i < numGroups; i++) {
        vector<int> particles;
        vector<double> weights;
        force.getGroupParameters(i, particles, weights);
        groupParticleVec.insert(groupParticleVec.end(), particles.begin(), particles.end());
        groupOffsetVec.push_back(groupParticleVec.size());
    }
    vector<vector<double> > normalizedWeights;
    CustomCentroidBondForceImpl::computeNormalizedWeights(force, system, normalizedWeights);
    for (int i = 0; i < numGroups; i++)
        groupWeightVec.insert(groupWeightVec.end(), normalizedWeights[i].begin(), normalizedWeights[i].end());
    groupParticles.initialize<int>(cc, groupParticleVec.size(), "groupParticles");
    groupParticles.upload(groupParticleVec);
    if (cc.getUseDoublePrecision()) {
        groupWeights.initialize<double>(cc, groupParticleVec.size(), "groupWeights");
        centerPositions.initialize<mm_double4>(cc, numGroups, "centerPositions");
    }
    else {
        groupWeights.initialize<float>(cc, groupParticleVec.size(), "groupWeights");
        centerPositions.initialize<mm_float4>(cc, numGroups, "centerPositions");
    }
    groupWeights.upload(groupWeightVec, true);
    groupOffsets.initialize<int>(cc, groupOffsetVec.size(), "groupOffsets");
    groupOffsets.upload(groupOffsetVec);
    groupForces.initialize<long long>(cc, numGroups*3, "groupForces");
    cc.addAutoclearBuffer(groupForces);

    // Record the bonds.

    int groupsPerBond = force.getNumGroupsPerBond();
    vector<int> bondGroupVec(numBonds*groupsPerBond);
    params = new ComputeParameterSet(cc, force.getNumPerBondParameters(), numBonds, "customCentroidBondParams");
    vector<vector<float> > paramVector(numBonds);
    for (int i = 0; i < numBonds; i++) {
        vector<int> groups;
        vector<double> parameters;
        force.getBondParameters(i, groups, parameters);
        for (int j = 0; j < groups.size(); j++)
            bondGroupVec[i+j*numBonds] = groups[j];
        paramVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);
    bondGroups.initialize<int>(cc, bondGroupVec.size(), "bondGroups");
    bondGroups.upload(bondGroupVec);

    // Record the tabulated functions.

    map<string, Lepton::CustomFunction*> functions;
    vector<pair<string, string> > functionDefinitions;
    vector<const TabulatedFunction*> functionList;
    stringstream extraArgs;
    tabulatedFunctions.resize(force.getNumTabulatedFunctions());
    for (int i = 0; i < force.getNumTabulatedFunctions(); i++) {
        functionList.push_back(&force.getTabulatedFunction(i));
        string name = force.getTabulatedFunctionName(i);
        string arrayName = "table"+cc.intToString(i);
        functionDefinitions.push_back(make_pair(name, arrayName));
        functions[name] = cc.getExpressionUtilities().getFunctionPlaceholder(force.getTabulatedFunction(i));
        int width;
        vector<float> f = cc.getExpressionUtilities().computeFunctionCoefficients(force.getTabulatedFunction(i), width);
        tabulatedFunctions[i].initialize<float>(cc, f.size(), "TabulatedFunction");
        tabulatedFunctions[i].upload(f);
        extraArgs << ", GLOBAL const float";
        if (width > 1)
            extraArgs << width;
        extraArgs << "* RESTRICT " << arrayName;
    }

    // Record information about parameters.

    globalParamNames.resize(force.getNumGlobalParameters());
    globalParamValues.resize(force.getNumGlobalParameters());
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        globalParamNames[i] = force.getGlobalParameterName(i);
        globalParamValues[i] = (float) force.getGlobalParameterDefaultValue(i);
    }
    map<string, string> variables;
    for (int i = 0; i < groupsPerBond; i++) {
        string index = cc.intToString(i+1);
        variables["x"+index] = "pos"+index+".x";
        variables["y"+index] = "pos"+index+".y";
        variables["z"+index] = "pos"+index+".z";
    }
    for (int i = 0; i < force.getNumPerBondParameters(); i++) {
        const string& name = force.getPerBondParameterName(i);
        variables[name] = "bondParams"+params->getParameterSuffix(i);
    }
    needEnergyParamDerivs = (force.getNumEnergyParameterDerivatives() > 0);
    if (needEnergyParamDerivs)
        extraArgs << ", GLOBAL mixed* RESTRICT energyParamDerivs";
    if (force.getNumGlobalParameters() > 0) {
        globals.initialize<float>(cc, force.getNumGlobalParameters(), "customCentroidBondGlobals");
        globals.upload(globalParamValues);
        extraArgs << ", GLOBAL const float* RESTRICT globals";
        for (int i = 0; i < force.getNumGlobalParameters(); i++) {
            const string& name = force.getGlobalParameterName(i);
            string value = "globals["+cc.intToString(i)+"]";
            variables[name] = value;
        }
    }

    // Now to generate the kernel.  First, it needs to calculate all distances, angles,
    // and dihedrals the expression depends on.

    map<string, vector<int> > distances;
    map<string, vector<int> > angles;
    map<string, vector<int> > dihedrals;
    Lepton::ParsedExpression energyExpression = CustomCentroidBondForceImpl::prepareExpression(force, functions, distances, angles, dihedrals);
    map<string, Lepton::ParsedExpression> forceExpressions;
    set<string> computedDeltas;
    vector<string> atomNames, posNames;
    for (int i = 0; i < groupsPerBond; i++) {
        string index = cc.intToString(i+1);
        atomNames.push_back("P"+index);
        posNames.push_back("pos"+index);
    }
    stringstream compute, initParamDerivs, saveParamDerivs;
    for (int i = 0; i < groupsPerBond; i++) {
        compute<<"int group"<<(i+1)<<" = bondGroups[index+"<<(i*numBonds)<<"];\n";
        compute<<"real4 pos"<<(i+1)<<" = centerPositions[group"<<(i+1)<<"];\n";
    }
    int index = 0;
    for (auto& distance : distances) {
        const vector<int>& groups = distance.second;
        string deltaName = atomNames[groups[0]]+atomNames[groups[1]];
        if (computedDeltas.count(deltaName) == 0) {
            compute<<"real4 delta"<<deltaName<<" = delta("<<posNames[groups[0]]<<", "<<posNames[groups[1]]<<", "<<force.usesPeriodicBoundaryConditions()<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName);
        }
        compute<<"real r_"<<deltaName<<" = sqrt(delta"<<deltaName<<".w);\n";
        variables[distance.first] = "r_"+deltaName;
        forceExpressions["real dEdDistance"+cc.intToString(index)+" = "] = energyExpression.differentiate(distance.first).optimize();
        index++;
    }
    index = 0;
    for (auto& angle : angles) {
        const vector<int>& groups = angle.second;
        string deltaName1 = atomNames[groups[1]]+atomNames[groups[0]];
        string deltaName2 = atomNames[groups[1]]+atomNames[groups[2]];
        string angleName = "angle_"+atomNames[groups[0]]+atomNames[groups[1]]+atomNames[groups[2]];
        if (computedDeltas.count(deltaName1) == 0) {
            compute<<"real4 delta"<<deltaName1<<" = delta("<<posNames[groups[1]]<<", "<<posNames[groups[0]]<<", "<<force.usesPeriodicBoundaryConditions()<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName1);
        }
        if (computedDeltas.count(deltaName2) == 0) {
            compute<<"real4 delta"<<deltaName2<<" = delta("<<posNames[groups[1]]<<", "<<posNames[groups[2]]<<", "<<force.usesPeriodicBoundaryConditions()<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName2);
        }
        compute<<"real "<<angleName<<" = computeAngle(delta"<<deltaName1<<", delta"<<deltaName2<<");\n";
        variables[angle.first] = angleName;
        forceExpressions["real dEdAngle"+cc.intToString(index)+" = "] = energyExpression.differentiate(angle.first).optimize();
        index++;
    }
    index = 0;
    for (auto& dihedral : dihedrals) {
        const vector<int>& groups = dihedral.second;
        string deltaName1 = atomNames[groups[0]]+atomNames[groups[1]];
        string deltaName2 = atomNames[groups[2]]+atomNames[groups[1]];
        string deltaName3 = atomNames[groups[2]]+atomNames[groups[3]];
        string crossName1 = "cross_"+deltaName1+"_"+deltaName2;
        string crossName2 = "cross_"+deltaName2+"_"+deltaName3;
        string dihedralName = "dihedral_"+atomNames[groups[0]]+atomNames[groups[1]]+atomNames[groups[2]]+atomNames[groups[3]];
        if (computedDeltas.count(deltaName1) == 0) {
            compute<<"real4 delta"<<deltaName1<<" = delta("<<posNames[groups[0]]<<", "<<posNames[groups[1]]<<", "<<force.usesPeriodicBoundaryConditions()<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName1);
        }
        if (computedDeltas.count(deltaName2) == 0) {
            compute<<"real4 delta"<<deltaName2<<" = delta("<<posNames[groups[2]]<<", "<<posNames[groups[1]]<<", "<<force.usesPeriodicBoundaryConditions()<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName2);
        }
        if (computedDeltas.count(deltaName3) == 0) {
            compute<<"real4 delta"<<deltaName3<<" = delta("<<posNames[groups[2]]<<", "<<posNames[groups[3]]<<", "<<force.usesPeriodicBoundaryConditions()<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName3);
        }
        compute<<"real4 "<<crossName1<<" = computeCross(delta"<<deltaName1<<", delta"<<deltaName2<<");\n";
        compute<<"real4 "<<crossName2<<" = computeCross(delta"<<deltaName2<<", delta"<<deltaName3<<");\n";
        compute<<"real "<<dihedralName<<" = computeAngle("<<crossName1<<", "<<crossName2<<");\n";
        compute<<dihedralName<<" *= (delta"<<deltaName1<<".x*"<<crossName2<<".x + delta"<<deltaName1<<".y*"<<crossName2<<".y + delta"<<deltaName1<<".z*"<<crossName2<<".z < 0 ? -1 : 1);\n";
        variables[dihedral.first] = dihedralName;
        forceExpressions["real dEdDihedral"+cc.intToString(index)+" = "] = energyExpression.differentiate(dihedral.first).optimize();
        index++;
    }

    // Now evaluate the expressions.

    for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
        ComputeParameterInfo& parameter = params->getParameterInfos()[i];
        extraArgs<<", GLOBAL const "<<parameter.getType()<<"* RESTRICT globalParams"<<i;
        compute<<parameter.getType()<<" bondParams"<<(i+1)<<" = globalParams"<<i<<"[index];\n";
    }
    forceExpressions["energy += "] = energyExpression;
    if (needEnergyParamDerivs) {
        for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
            string paramName = force.getEnergyParameterDerivativeName(i);
            cc.addEnergyParameterDerivative(paramName);
            Lepton::ParsedExpression derivExpression = energyExpression.differentiate(paramName).optimize();
            forceExpressions[string("energyParamDeriv")+cc.intToString(i)+" += "] = derivExpression;
            initParamDerivs << "mixed energyParamDeriv" << i << " = 0;\n";
        }
        const vector<string>& allParamDerivNames = cc.getEnergyParamDerivNames();
        int numDerivs = allParamDerivNames.size();
        for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++)
            for (int index = 0; index < numDerivs; index++)
                if (allParamDerivNames[index] == force.getEnergyParameterDerivativeName(i))
                    saveParamDerivs << "energyParamDerivs[GLOBAL_ID*" << numDerivs << "+" << index << "] += energyParamDeriv" << i << ";\n";
    }
    compute << cc.getExpressionUtilities().createExpressions(forceExpressions, variables, functionList, functionDefinitions, "temp");

    // Finally, apply forces to groups.

    vector<string> forceNames;
    for (int i = 0; i < groupsPerBond; i++) {
        string istr = cc.intToString(i+1);
        string forceName = "force"+istr;
        forceNames.push_back(forceName);
        compute<<"real3 "<<forceName<<" = make_real3(0);\n";
        compute<<"{\n";
        Lepton::ParsedExpression forceExpressionX = energyExpression.differentiate("x"+istr).optimize();
        Lepton::ParsedExpression forceExpressionY = energyExpression.differentiate("y"+istr).optimize();
        Lepton::ParsedExpression forceExpressionZ = energyExpression.differentiate("z"+istr).optimize();
        map<string, Lepton::ParsedExpression> expressions;
        if (!isZeroExpression(forceExpressionX))
            expressions[forceName+".x -= "] = forceExpressionX;
        if (!isZeroExpression(forceExpressionY))
            expressions[forceName+".y -= "] = forceExpressionY;
        if (!isZeroExpression(forceExpressionZ))
            expressions[forceName+".z -= "] = forceExpressionZ;
        if (expressions.size() > 0)
            compute<<cc.getExpressionUtilities().createExpressions(expressions, variables, functionList, functionDefinitions, "coordtemp");
        compute<<"}\n";
    }
    index = 0;
    for (auto& distance : distances) {
        const vector<int>& groups = distance.second;
        string deltaName = atomNames[groups[0]]+atomNames[groups[1]];
        string value = "(dEdDistance"+cc.intToString(index)+"/r_"+deltaName+")*trimTo3(delta"+deltaName+")";
        compute<<forceNames[groups[0]]<<" += "<<"-"<<value<<";\n";
        compute<<forceNames[groups[1]]<<" += "<<value<<";\n";
        index++;
    }
    index = 0;
    for (auto& angle : angles) {
        const vector<int>& groups = angle.second;
        string deltaName1 = atomNames[groups[1]]+atomNames[groups[0]];
        string deltaName2 = atomNames[groups[1]]+atomNames[groups[2]];
        compute<<"{\n";
        compute<<"real3 crossProd = trimTo3(cross(delta"<<deltaName2<<", delta"<<deltaName1<<"));\n";
        compute<<"real lengthCross = max(SQRT(dot(crossProd, crossProd)), (real) 1e-6f);\n";
        compute<<"real3 deltaCross0 = -cross(trimTo3(delta"<<deltaName1<<"), crossProd)*dEdAngle"<<cc.intToString(index)<<"/(delta"<<deltaName1<<".w*lengthCross);\n";
        compute<<"real3 deltaCross2 = cross(trimTo3(delta"<<deltaName2<<"), crossProd)*dEdAngle"<<cc.intToString(index)<<"/(delta"<<deltaName2<<".w*lengthCross);\n";
        compute<<"real3 deltaCross1 = -(deltaCross0+deltaCross2);\n";
        compute<<forceNames[groups[0]]<<" += deltaCross0;\n";
        compute<<forceNames[groups[1]]<<" += deltaCross1;\n";
        compute<<forceNames[groups[2]]<<" += deltaCross2;\n";
        compute<<"}\n";
        index++;
    }
    index = 0;
    for (auto& dihedral : dihedrals) {
        const vector<int>& groups = dihedral.second;
        string deltaName1 = atomNames[groups[0]]+atomNames[groups[1]];
        string deltaName2 = atomNames[groups[2]]+atomNames[groups[1]];
        string deltaName3 = atomNames[groups[2]]+atomNames[groups[3]];
        string crossName1 = "cross_"+deltaName1+"_"+deltaName2;
        string crossName2 = "cross_"+deltaName2+"_"+deltaName3;
        compute<<"{\n";
        compute<<"real r = sqrt(delta"<<deltaName2<<".w);\n";
        compute<<"real4 ff;\n";
        compute<<"ff.x = (-dEdDihedral"<<cc.intToString(index)<<"*r)/"<<crossName1<<".w;\n";
        compute<<"ff.y = (delta"<<deltaName1<<".x*delta"<<deltaName2<<".x + delta"<<deltaName1<<".y*delta"<<deltaName2<<".y + delta"<<deltaName1<<".z*delta"<<deltaName2<<".z)/delta"<<deltaName2<<".w;\n";
        compute<<"ff.z = (delta"<<deltaName3<<".x*delta"<<deltaName2<<".x + delta"<<deltaName3<<".y*delta"<<deltaName2<<".y + delta"<<deltaName3<<".z*delta"<<deltaName2<<".z)/delta"<<deltaName2<<".w;\n";
        compute<<"ff.w = (dEdDihedral"<<cc.intToString(index)<<"*r)/"<<crossName2<<".w;\n";
        compute<<"real3 internalF0 = ff.x*trimTo3("<<crossName1<<");\n";
        compute<<"real3 internalF3 = ff.w*trimTo3("<<crossName2<<");\n";
        compute<<"real3 s = ff.y*internalF0 - ff.z*internalF3;\n";
        compute<<forceNames[groups[0]]<<" += internalF0;\n";
        compute<<forceNames[groups[1]]<<" += s-internalF0;\n";
        compute<<forceNames[groups[2]]<<" += -s-internalF3;\n";
        compute<<forceNames[groups[3]]<<" += internalF3;\n";
        compute<<"}\n";
        index++;
    }

    // Save the forces to global memory.

    for (int i = 0; i < groupsPerBond; i++) {
        compute<<"ATOMIC_ADD(&groupForce[group"<<(i+1)<<"], (mm_ulong) ((mm_long) (force"<<(i+1)<<".x*0x100000000)));\n";
        compute<<"ATOMIC_ADD(&groupForce[group"<<(i+1)<<"+numParticleGroups], (mm_ulong) ((mm_long) (force"<<(i+1)<<".y*0x100000000)));\n";
        compute<<"ATOMIC_ADD(&groupForce[group"<<(i+1)<<"+numParticleGroups*2], (mm_ulong) ((mm_long) (force"<<(i+1)<<".z*0x100000000)));\n";
        compute<<"MEM_FENCE;\n";
    }
    map<string, string> replacements;
    replacements["M_PI"] = cc.doubleToString(M_PI);
    replacements["NUM_BONDS"] = cc.intToString(numBonds);
    replacements["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
    replacements["EXTRA_ARGS"] = extraArgs.str();
    replacements["COMPUTE_FORCE"] = compute.str();
    replacements["INIT_PARAM_DERIVS"] = initParamDerivs.str();
    replacements["SAVE_PARAM_DERIVS"] = saveParamDerivs.str();
    replacements["BLOCK_SIZE"] = to_string(this->blockSize);
    replacements["WARP_SIZE"] = to_string(cc.getSIMDWidth());
    ComputeProgram program = cc.compileProgram(cc.replaceStrings(CommonKernelSources::customCentroidBond, replacements));
    computeCentersKernel = program->createKernel("computeGroupCenters");
    computeCentersKernel->addArg(numGroups);
    computeCentersKernel->addArg(cc.getPosq());
    computeCentersKernel->addArg(groupParticles);
    computeCentersKernel->addArg(groupWeights);
    computeCentersKernel->addArg(groupOffsets);
    computeCentersKernel->addArg(centerPositions);
    groupForcesKernel = program->createKernel("computeGroupForces");
    groupForcesKernel->addArg(numGroups);
    groupForcesKernel->addArg(groupForces);
    groupForcesKernel->addArg(); // Energy buffer hasn't been created yet
    groupForcesKernel->addArg(centerPositions);
    groupForcesKernel->addArg(bondGroups);
    for (int i = 0; i < 5; i++)
        groupForcesKernel->addArg(); // Periodic box information will be set just before it is executed.
    if (needEnergyParamDerivs)
        groupForcesKernel->addArg(); // Deriv buffer hasn't been created yet.
    for (auto& function : tabulatedFunctions)
        groupForcesKernel->addArg(function);
    if (globals.isInitialized())
        groupForcesKernel->addArg(globals);
    for (auto& parameter : params->getParameterInfos())
        groupForcesKernel->addArg(parameter.getArray());
    applyForcesKernel = program->createKernel("applyForcesToAtoms");
    applyForcesKernel->addArg(numGroups);
    applyForcesKernel->addArg(groupParticles);
    applyForcesKernel->addArg(groupWeights);
    applyForcesKernel->addArg(groupOffsets);
    applyForcesKernel->addArg(groupForces);
    applyForcesKernel->addArg();
}

double CommonCalcCustomCentroidBondForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (numBonds == 0)
        return 0.0;
    if (globals.isInitialized()) {
        bool changed = false;
        for (int i = 0; i < (int) globalParamNames.size(); i++) {
            float value = (float) context.getParameter(globalParamNames[i]);
            if (value != globalParamValues[i])
                changed = true;
            globalParamValues[i] = value;
        }
        if (changed)
            globals.upload(globalParamValues);
    }
    computeCentersKernel->execute(this->blockSize*numGroups);
    groupForcesKernel->setArg(2, cc.getEnergyBuffer());
    setPeriodicBoxArgs(cc, groupForcesKernel, 5);
    if (needEnergyParamDerivs)
        groupForcesKernel->setArg(10, cc.getEnergyParamDerivBuffer());
    groupForcesKernel->execute(numBonds);
    applyForcesKernel->setArg(5, cc.getLongForceBuffer());
    applyForcesKernel->execute(this->blockSize*numGroups);
    return 0.0;
}

void CommonCalcCustomCentroidBondForceKernel::copyParametersToContext(ContextImpl& context, const CustomCentroidBondForce& force) {
    cc.setAsCurrent();
    if (numBonds != force.getNumBonds())
        throw OpenMMException("updateParametersInContext: The number of bonds has changed");
    if (numBonds == 0)
        return;

    // Record the per-bond parameters.

    vector<vector<float> > paramVector(numBonds);
    vector<int> particles;
    vector<double> parameters;
    for (int i = 0; i < numBonds; i++) {
        force.getBondParameters(i, particles, parameters);
        paramVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);

    // Mark that the current reordering may be invalid.

    cc.invalidateMolecules(info);
}

class CommonCalcCustomNonbondedForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const CustomNonbondedForce& force) : force(force) {
        if (force.getNumInteractionGroups() > 0) {
            groupsForParticle.resize(force.getNumParticles());
            for (int i = 0; i < force.getNumInteractionGroups(); i++) {
                set<int> set1, set2;
                force.getInteractionGroupParameters(i, set1, set2);
                for (int p : set1)
                    groupsForParticle[p].insert(2*i);
                for (int p : set2)
                    groupsForParticle[p].insert(2*i+1);
            }
        }
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        vector<double> params1;
        vector<double> params2;
        force.getParticleParameters(particle1, params1);
        force.getParticleParameters(particle2, params2);
        for (int i = 0; i < (int) params1.size(); i++)
            if (params1[i] != params2[i])
                return false;
        if (groupsForParticle.size() > 0 && groupsForParticle[particle1] != groupsForParticle[particle2])
            return false;
        return true;
    }
    int getNumParticleGroups() {
        return force.getNumExclusions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        force.getExclusionParticles(index, particle1, particle2);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        return true;
    }
private:
    const CustomNonbondedForce& force;
    vector<set<int> > groupsForParticle;
};

CommonCalcCustomNonbondedForceKernel::~CommonCalcCustomNonbondedForceKernel() {
    cc.setAsCurrent();
    if (params != NULL)
        delete params;
    if (forceCopy != NULL)
        delete forceCopy;
}

void CommonCalcCustomNonbondedForceKernel::initialize(const System& system, const CustomNonbondedForce& force) {
    cc.setAsCurrent();
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() && &system.getForce(forceIndex) != &force; ++forceIndex)
        ;
    string prefix = (force.getNumInteractionGroups() == 0 ? "custom"+cc.intToString(forceIndex)+"_" : "");

    // Record parameters and exclusions.

    int numParticles = force.getNumParticles();
    int paddedNumParticles = cc.getPaddedNumAtoms();
    int numParams = force.getNumPerParticleParameters();
    params = new ComputeParameterSet(cc, numParams, paddedNumParticles, "customNonbondedParameters");
    if (force.getNumGlobalParameters() > 0)
        globals.initialize<float>(cc, force.getNumGlobalParameters(), "customNonbondedGlobals");
    vector<vector<float> > paramVector(paddedNumParticles, vector<float>(numParams, 0));
    vector<vector<int> > exclusionList(numParticles);
    for (int i = 0; i < numParticles; i++) {
        vector<double> parameters;
        force.getParticleParameters(i, parameters);
        paramVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
        exclusionList[i].push_back(i);
    }
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int particle1, particle2;
        force.getExclusionParticles(i, particle1, particle2);
        exclusionList[particle1].push_back(particle2);
        exclusionList[particle2].push_back(particle1);
    }
    params->setParameterValues(paramVector);

    // Record the tabulated functions.

    map<string, Lepton::CustomFunction*> functions;
    vector<pair<string, string> > functionDefinitions;
    vector<const TabulatedFunction*> functionList;
    vector<string> tableTypes;
    tabulatedFunctions.resize(force.getNumTabulatedFunctions());
    for (int i = 0; i < force.getNumTabulatedFunctions(); i++) {
        functionList.push_back(&force.getTabulatedFunction(i));
        string name = force.getTabulatedFunctionName(i);
        string arrayName = prefix+"table"+cc.intToString(i);
        functionDefinitions.push_back(make_pair(name, arrayName));
        functions[name] = cc.getExpressionUtilities().getFunctionPlaceholder(force.getTabulatedFunction(i));
        int width;
        vector<float> f = cc.getExpressionUtilities().computeFunctionCoefficients(force.getTabulatedFunction(i), width);
        tabulatedFunctions[i].initialize<float>(cc, f.size(), "TabulatedFunction");
        tabulatedFunctions[i].upload(f);
        cc.getNonbondedUtilities().addArgument(ComputeParameterInfo(tabulatedFunctions[i], arrayName, "float", width));
        if (width == 1)
            tableTypes.push_back("float");
        else
            tableTypes.push_back("float"+cc.intToString(width));
    }

    // Record information for the expressions.

    globalParamNames.resize(force.getNumGlobalParameters());
    globalParamValues.resize(force.getNumGlobalParameters());
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        globalParamNames[i] = force.getGlobalParameterName(i);
        globalParamValues[i] = (float) force.getGlobalParameterDefaultValue(i);
    }
    if (globals.isInitialized())
        globals.upload(globalParamValues);
    bool useCutoff = (force.getNonbondedMethod() != CustomNonbondedForce::NoCutoff);
    bool usePeriodic = (force.getNonbondedMethod() != CustomNonbondedForce::NoCutoff && force.getNonbondedMethod() != CustomNonbondedForce::CutoffNonPeriodic);
    Lepton::ParsedExpression energyExpression = Lepton::Parser::parse(force.getEnergyFunction(), functions).optimize();
    Lepton::ParsedExpression forceExpression = energyExpression.differentiate("r").optimize();
    map<string, Lepton::ParsedExpression> forceExpressions;
    forceExpressions["real customEnergy = "] = energyExpression;
    forceExpressions["tempForce -= "] = forceExpression;

    // Create the kernels.

    vector<pair<ExpressionTreeNode, string> > variables;
    ExpressionTreeNode rnode(new Operation::Variable("r"));
    variables.push_back(make_pair(rnode, "r"));
    variables.push_back(make_pair(ExpressionTreeNode(new Operation::Square(), rnode), "r2"));
    variables.push_back(make_pair(ExpressionTreeNode(new Operation::Reciprocal(), rnode), "invR"));
    for (int i = 0; i < force.getNumPerParticleParameters(); i++) {
        const string& name = force.getPerParticleParameterName(i);
        variables.push_back(makeVariable(name+"1", prefix+"params"+params->getParameterSuffix(i, "1")));
        variables.push_back(makeVariable(name+"2", prefix+"params"+params->getParameterSuffix(i, "2")));
    }
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        const string& name = force.getGlobalParameterName(i);
        string value = "globals["+cc.intToString(i)+"]";
        variables.push_back(makeVariable(name, prefix+value));
    }
    for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
        string paramName = force.getEnergyParameterDerivativeName(i);
        string derivVariable = cc.getNonbondedUtilities().addEnergyParameterDerivative(paramName);
        Lepton::ParsedExpression derivExpression = energyExpression.differentiate(paramName).optimize();
        forceExpressions[derivVariable+" += interactionScale*switchValue*"] = derivExpression;
    }
    stringstream compute;
    compute << cc.getExpressionUtilities().createExpressions(forceExpressions, variables, functionList, functionDefinitions, prefix+"temp");
    map<string, string> replacements;
    replacements["COMPUTE_FORCE"] = compute.str();
    replacements["USE_SWITCH"] = (useCutoff && force.getUseSwitchingFunction() ? "1" : "0");
    if (force.getUseSwitchingFunction()) {
        // Compute the switching coefficients.

        replacements["SWITCH_CUTOFF"] = cc.doubleToString(force.getSwitchingDistance());
        replacements["SWITCH_C3"] = cc.doubleToString(10/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 3.0));
        replacements["SWITCH_C4"] = cc.doubleToString(15/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 4.0));
        replacements["SWITCH_C5"] = cc.doubleToString(6/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 5.0));
    }
    string source = cc.replaceStrings(CommonKernelSources::customNonbonded, replacements);
    if (force.getNumInteractionGroups() > 0)
        launchInitInteractionGroups(force, source, tableTypes);
    else {
        cc.getNonbondedUtilities().addInteraction(useCutoff, usePeriodic, true, force.getCutoffDistance(), exclusionList, source, force.getForceGroup());
        for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
            ComputeParameterInfo& parameter = params->getParameterInfos()[i];
            cc.getNonbondedUtilities().addParameter(ComputeParameterInfo(parameter.getArray(), prefix+"params"+cc.intToString(i+1),
                    parameter.getComponentType(), parameter.getNumComponents()));
        }
        if (globals.isInitialized()) {
            globals.upload(globalParamValues);
            cc.getNonbondedUtilities().addArgument(ComputeParameterInfo(globals, prefix+"globals", "float", 1));
        }
    }
    info = new ForceInfo(force);
    cc.addForce(info);

    // Record information for the long range correction.

    if (force.getNonbondedMethod() == CustomNonbondedForce::CutoffPeriodic && force.getUseLongRangeCorrection() && cc.getContextIndex() == 0) {
        forceCopy = new CustomNonbondedForce(force);
        hasInitializedLongRangeCorrection = false;
    }
    else {
        longRangeCoefficient = 0.0;
        hasInitializedLongRangeCorrection = true;
    }
}

// the shifting method previously used in initInteractionGroups
// is undefined for signed types, hence we switched
// to unsigned types for repeatability, and a special case switch operator
// see https://en.cppreference.com/w/c/language/operator_arithmetic
template<typename tileflags>
tileflags inline full_shift(tileflags operand, int operant)
{
    if (operant == sizeof(tileflags) * 8) {
        // shifting by the size of the operand is undefined
        // the intent here appears to be to return a -0
        return tileflags(0);
    }
    return (operand<<operant);
}


template <typename tileflags, typename mm_intv>
void CommonCalcCustomNonbondedForceKernel::initInteractionGroups(const CustomNonbondedForce& force, const string& interactionSource, const vector<string>& tableTypes) {
    // Process groups to form tiles.

    const int simdWidth = cc.getSIMDWidth();
    const int simdHalfWidth = simdWidth / 2;

    vector<vector<int> > atomLists;
    vector<pair<int, int> > tiles;
    vector<int> tileGroup;
    vector<vector<int> > duplicateAtomsForGroup;
    for (int group = 0; group < force.getNumInteractionGroups(); group++) {
        // Get the list of atoms in this group and sort them.

        set<int> set1, set2;
        force.getInteractionGroupParameters(group, set1, set2);
        vector<int> atoms1, atoms2;
        atoms1.insert(atoms1.begin(), set1.begin(), set1.end());
        atoms2.insert(atoms2.begin(), set2.begin(), set2.end());
        sort(atoms1.begin(), atoms1.end());
        sort(atoms2.begin(), atoms2.end());
        duplicateAtomsForGroup.push_back(vector<int>());
        set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
                inserter(duplicateAtomsForGroup[group], duplicateAtomsForGroup[group].begin()));
        sort(duplicateAtomsForGroup[group].begin(), duplicateAtomsForGroup[group].end());

        // Find how many tiles we will create for this group.

        int tileWidth = min(min(cc.getSIMDWidth(), (int) atoms1.size()), (int) atoms2.size());
        if (tileWidth == 0)
            continue;
        int numBlocks1 = (atoms1.size()+tileWidth-1)/tileWidth;
        int numBlocks2 = (atoms2.size()+tileWidth-1)/tileWidth;

        // Add the tiles.

        int firstTile = tiles.size();
        for (int i = 0; i < numBlocks1; i++)
            for (int j = 0; j < numBlocks2; j++) {
                tiles.push_back(make_pair(atomLists.size()+i, atomLists.size()+numBlocks1+j));
                tileGroup.push_back(group);
            }

        // Add the atom lists.

        for (int i = 0; i < numBlocks1; i++) {
            vector<int> atoms;
            int first = i*tileWidth;
            int last = min((i+1)*tileWidth, (int) atoms1.size());
            for (int j = first; j < last; j++)
                atoms.push_back(atoms1[j]);
            atomLists.push_back(atoms);
        }
        for (int i = 0; i < numBlocks2; i++) {
            vector<int> atoms;
            int first = i*tileWidth;
            int last = min((i+1)*tileWidth, (int) atoms2.size());
            for (int j = first; j < last; j++)
                atoms.push_back(atoms2[j]);
            atomLists.push_back(atoms);
        }
    }

    // Build a lookup table for quickly identifying excluded interactions.

    vector<set<int> > exclusions(force.getNumParticles());
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int p1, p2;
        force.getExclusionParticles(i, p1, p2);
        exclusions[p1].insert(p2);
        exclusions[p2].insert(p1);
    }

    // Build the exclusion flags for each tile.  While we're at it, filter out tiles
    // where all interactions are excluded, and sort the tiles by size.

    vector<vector<tileflags> > exclusionFlags(tiles.size());
    vector<pair<int, int> > tileOrder;
    for (int tile = 0; tile < tiles.size(); tile++) {
        bool swapped = false;
        if (atomLists[tiles[tile].first].size() < atomLists[tiles[tile].second].size()) {
            // For efficiency, we want the first axis to be the larger one.

            int swap = tiles[tile].first;
            tiles[tile].first = tiles[tile].second;
            tiles[tile].second = swap;
            swapped = true;
        }
        vector<int>& atoms1 = atomLists[tiles[tile].first];
        vector<int>& atoms2 = atomLists[tiles[tile].second];
        vector<int>& duplicateAtoms = duplicateAtomsForGroup[tileGroup[tile]];
        vector<tileflags>& flags = exclusionFlags[tile];
        flags.resize(atoms1.size(), full_shift(static_cast<tileflags>(1), atoms2.size())-1);
        int numExcluded = 0;
        for (int i = 0; i < (int) atoms1.size(); i++) {
            int a1 = atoms1[i];
            bool a1IsDuplicate = binary_search(duplicateAtoms.begin(), duplicateAtoms.end(), a1);
            for (int j = 0; j < (int) atoms2.size(); j++) {
                int a2 = atoms2[j];
                bool isExcluded = false;
                if (a1 == a2 || exclusions[a1].find(a2) != exclusions[a1].end())
                    isExcluded = true; // This is an excluded interaction.
                else if ((a1 > a2) == swapped && a1IsDuplicate && binary_search(duplicateAtoms.begin(), duplicateAtoms.end(), a2))
                    isExcluded = true; // Both atoms are in both sets, so skip duplicate interactions.
                if (isExcluded) {
                    flags[i] &= static_cast<tileflags>(-1) - full_shift(static_cast<tileflags>(1), j);
                    numExcluded++;
                }
            }
        }
        if (numExcluded == atoms1.size()*atoms2.size())
            continue; // All interactions are excluded.
        tileOrder.push_back(make_pair(-((int)atoms2.size()), tile));
    }
    sort(tileOrder.begin(), tileOrder.end());

    // Merge tiles to get as close as possible to simdWIDTH along the first axis of each one.

    vector<int> tileSetStart;
    tileSetStart.push_back(0);
    int tileSetSize = 0;
    for (int i = 0; i < tileOrder.size(); i++) {
        int tile = tileOrder[i].second;
        int size = atomLists[tiles[tile].first].size();
        if (tileSetSize+size > cc.getSIMDWidth()) {
            tileSetStart.push_back(i);
            tileSetSize = 0;
        }
        tileSetSize += size;
    }
    tileSetStart.push_back(tileOrder.size());

    // Build the data structures.

    int numTileSets = tileSetStart.size()-1;
    vector<mm_intv> groupData;
    for (int tileSet = 0; tileSet < numTileSets; tileSet++) {
        int indexInTileSet = 0;
        int minSize = 0;
        if (cc.getSIMDWidth() < 32) {
            // We need to include a barrier inside the inner loop, so ensure that all
            // threads will loop the same number of times.

            for (int i = tileSetStart[tileSet]; i < tileSetStart[tileSet+1]; i++)
                minSize = max(minSize, (int) atomLists[tiles[tileOrder[i].second].first].size());
        }
        for (int i = tileSetStart[tileSet]; i < tileSetStart[tileSet+1]; i++) {
            int tile = tileOrder[i].second;
            vector<int>& atoms1 = atomLists[tiles[tile].first];
            vector<int>& atoms2 = atomLists[tiles[tile].second];
            tileflags range = indexInTileSet + (static_cast<tileflags>(indexInTileSet+max(minSize, (int) atoms1.size()))<<simdHalfWidth);
            tileflags allFlags = full_shift(static_cast<tileflags>(1), atoms2.size())-1;
            for (int j = 0; j < (int) atoms1.size(); j++) {
                int a1 = atoms1[j];
                int a2 = (j < atoms2.size() ? atoms2[j] : 0);
                tileflags flags = (exclusionFlags[tile].size() > 0 ? exclusionFlags[tile][j] : allFlags);
                groupData.push_back(mm_intv(a1, a2, range, static_cast<tileflags>(flags)<<indexInTileSet));
            }
            indexInTileSet += atoms1.size();
        }
        for (; indexInTileSet < simdWidth; indexInTileSet++)
            groupData.push_back(mm_intv(0, 0, static_cast<tileflags>(minSize)<<simdHalfWidth, 0));
    }

    interactionGroupData.initialize<mm_intv>(cc, groupData.size(), "interactionGroupData");
    interactionGroupData.upload(groupData);
    numGroupTiles.initialize<int>(cc, 1, "numGroupTiles");

    // Allocate space for a neighbor list, if necessary.

    if (force.getNonbondedMethod() != CustomNonbondedForce::NoCutoff && groupData.size() > cc.getNumThreadBlocks()) {
        filteredGroupData.initialize<mm_intv>(cc, groupData.size(), "filteredGroupData");
        interactionGroupData.copyTo(filteredGroupData);
        int numTiles = groupData.size()/simdWidth;
        numGroupTiles.upload(&numTiles);
    }

    // Create the kernel.

    hasParamDerivs = (force.getNumEnergyParameterDerivatives() > 0);
    map<string, string> replacements;
    replacements["COMPUTE_INTERACTION"] = interactionSource;
    const string suffixes[] = {"x", "y", "z", "w"};
    stringstream localData;
    int localDataSize = 0;
    vector<ComputeParameterInfo>& buffers = params->getParameterInfos();
    for (int i = 0; i < (int) buffers.size(); i++) {
        if (buffers[i].getNumComponents() == 1)
            localData<<buffers[i].getComponentType()<<" params"<<(i+1)<<";\n";
        else {
            for (int j = 0; j < buffers[i].getNumComponents(); ++j)
                localData<<buffers[i].getComponentType()<<" params"<<(i+1)<<"_"<<suffixes[j]<<";\n";
        }
        localDataSize += buffers[i].getSize();
    }
    replacements["ATOM_PARAMETER_DATA"] = localData.str();
    stringstream args;
    for (int i = 0; i < (int) buffers.size(); i++)
        args<<", GLOBAL const "<<buffers[i].getType()<<"* RESTRICT global_params"<<(i+1);
    for (int i = 0; i < (int) tabulatedFunctions.size(); i++)
        args << ", GLOBAL const " << tableTypes[i]<< "* RESTRICT table" << i;
    if (globals.isInitialized())
        args<<", GLOBAL const float* RESTRICT globals";
    if (hasParamDerivs)
        args << ", GLOBAL mixed* RESTRICT energyParamDerivs";
    replacements["PARAMETER_ARGUMENTS"] = args.str();
    stringstream load1;
    for (int i = 0; i < (int) buffers.size(); i++)
        load1<<buffers[i].getType()<<" params"<<(i+1)<<"1 = global_params"<<(i+1)<<"[atom1];\n";
    replacements["LOAD_ATOM1_PARAMETERS"] = load1.str();
    stringstream loadLocal2;
    for (int i = 0; i < (int) buffers.size(); i++) {
        if (buffers[i].getNumComponents() == 1)
            loadLocal2<<"localData[LOCAL_ID].params"<<(i+1)<<" = global_params"<<(i+1)<<"[atom2];\n";
        else {
            loadLocal2<<buffers[i].getType()<<" temp_params"<<(i+1)<<" = global_params"<<(i+1)<<"[atom2];\n";
            for (int j = 0; j < buffers[i].getNumComponents(); ++j)
                loadLocal2<<"localData[LOCAL_ID].params"<<(i+1)<<"_"<<suffixes[j]<<" = temp_params"<<(i+1)<<"."<<suffixes[j]<<";\n";
        }
    }
    replacements["LOAD_LOCAL_PARAMETERS"] = loadLocal2.str();
    stringstream load2;
    for (int i = 0; i < (int) buffers.size(); i++) {
        if (buffers[i].getNumComponents() == 1)
            load2<<buffers[i].getType()<<" params"<<(i+1)<<"2 = localData[localIndex].params"<<(i+1)<<";\n";
        else {
            load2<<buffers[i].getType()<<" params"<<(i+1)<<"2 = make_"<<buffers[i].getType()<<"(";
            for (int j = 0; j < buffers[i].getNumComponents(); ++j) {
                if (j > 0)
                    load2<<", ";
                load2<<"localData[localIndex].params"<<(i+1)<<"_"<<suffixes[j];
            }
            load2<<");\n";
        }
    }
    replacements["LOAD_ATOM2_PARAMETERS"] = load2.str();
    stringstream initDerivs, saveDerivs;
    const vector<string>& allParamDerivNames = cc.getEnergyParamDerivNames();
    int numDerivs = allParamDerivNames.size();
    for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
        string paramName = force.getEnergyParameterDerivativeName(i);
        string derivVariable = cc.getNonbondedUtilities().addEnergyParameterDerivative(paramName);
        initDerivs<<"mixed "<<derivVariable<<" = 0;\n";
        for (int index = 0; index < numDerivs; index++)
            if (allParamDerivNames[index] == paramName)
                saveDerivs<<"energyParamDerivs[GLOBAL_ID*"<<numDerivs<<"+"<<index<<"] += "<<derivVariable<<";\n";
    }
    replacements["INIT_DERIVATIVES"] = initDerivs.str();
    replacements["SAVE_DERIVATIVES"] = saveDerivs.str();
    map<string, string> defines;
    if (force.getNonbondedMethod() != CustomNonbondedForce::NoCutoff)
        defines["USE_CUTOFF"] = "1";
    if (force.getNonbondedMethod() == CustomNonbondedForce::CutoffPeriodic)
        defines["USE_PERIODIC"] = "1";
    int localMemorySize = max(simdWidth, cc.getNonbondedUtilities().getForceThreadBlockSize());
    defines["LOCAL_MEMORY_SIZE"] = cc.intToString(localMemorySize);
    defines["WARPS_IN_BLOCK"] = cc.intToString(localMemorySize/simdWidth);
    double cutoff = force.getCutoffDistance();
    defines["CUTOFF_SQUARED"] = cc.doubleToString(cutoff*cutoff);
    double paddedCutoff = cc.getNonbondedUtilities().padCutoff(cutoff);
    defines["PADDED_CUTOFF_SQUARED"] = cc.doubleToString(paddedCutoff*paddedCutoff);
    defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
    defines["TILE_SIZE"] = to_string(simdWidth);
    defines["NUM_TILES"] = cc.intToString(numTileSets);
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*numTileSets/numContexts;
    int endIndex = (cc.getContextIndex()+1)*numTileSets/numContexts;
    defines["FIRST_TILE"] = cc.intToString(startIndex);
    defines["LAST_TILE"] = cc.intToString(endIndex);
    defines["tileflags_t"] = (simdWidth > 32 ? "long" : "int");
    defines["excl4"] = (simdWidth > 32 ? "long4" : "int4");
    defines["SIMD_HALF_WIDTH"] = cc.intToString(simdHalfWidth);
    defines["SIMD_HALF_MASK"] = (simdWidth > 32 ? "0xFFFFFFFF" : "0xFFFF");
    if ((localDataSize/4)%2 == 0 && !cc.getUseDoublePrecision())
        defines["PARAMETER_SIZE_IS_EVEN"] = "1";
    ComputeProgram program = cc.compileProgram(cc.replaceStrings(CommonKernelSources::customNonbondedGroups, replacements), defines);
    interactionGroupKernel = program->createKernel("computeInteractionGroups");
    prepareNeighborListKernel = program->createKernel("prepareToBuildNeighborList");
    buildNeighborListKernel = program->createKernel("buildNeighborList");
    numGroupThreadBlocks = cc.getNonbondedUtilities().getNumForceThreadBlocks();
}

void CommonCalcCustomNonbondedForceKernel::launchInitInteractionGroups(const CustomNonbondedForce& force, const string& interactionSource, const vector<string>& tableTypes) {
    // find which type to launch
    const int simdWidth = cc.getSIMDWidth();
    if (simdWidth > 32)
        initInteractionGroups<long int, mm_long4>(force, interactionSource, tableTypes);
    else
        initInteractionGroups<int, mm_int4>(force, interactionSource, tableTypes);
}

double CommonCalcCustomNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    const int tileWidth = cc.getSIMDWidth();
    useNeighborList = (filteredGroupData.isInitialized() && cc.getNonbondedUtilities().getUseCutoff());
    if (useNeighborList && cc.getContextIndex() > 0) {
        // When using a neighbor list, run the whole calculation on a single device.
        return 0.0;
    }
    if (globals.isInitialized()) {
        bool changed = false;
        for (int i = 0; i < (int) globalParamNames.size(); i++) {
            float value = (float) context.getParameter(globalParamNames[i]);
            if (value != globalParamValues[i])
                changed = true;
            globalParamValues[i] = value;
        }
        if (changed) {
            globals.upload(globalParamValues);
            if (forceCopy != NULL) {
                CustomNonbondedForceImpl::calcLongRangeCorrection(*forceCopy, context.getOwner(), longRangeCoefficient, longRangeCoefficientDerivs);
                hasInitializedLongRangeCorrection = true;
            }
        }
    }
    if (!hasInitializedLongRangeCorrection) {
        CustomNonbondedForceImpl::calcLongRangeCorrection(*forceCopy, context.getOwner(), longRangeCoefficient, longRangeCoefficientDerivs);
        hasInitializedLongRangeCorrection = true;
    }
    if (interactionGroupData.isInitialized()) {
        if (!hasInitializedKernel) {
            hasInitializedKernel = true;
            bool useLong = cc.getSupports64BitGlobalAtomics();
            interactionGroupKernel->addArg((useLong ? cc.getLongForceBuffer() : cc.getForceBuffers()));
            interactionGroupKernel->addArg(cc.getEnergyBuffer());
            interactionGroupKernel->addArg(cc.getPosq());
            interactionGroupKernel->addArg((useNeighborList ? filteredGroupData : interactionGroupData));
            interactionGroupKernel->addArg(numGroupTiles);
            interactionGroupKernel->addArg((int) useNeighborList);
            for (int i = 0; i < 5; i++)
                interactionGroupKernel->addArg(); // Periodic box information will be set just before it is executed.
            for (auto& parameter : params->getParameterInfos())
                interactionGroupKernel->addArg(parameter.getArray());
            for (auto& function : tabulatedFunctions)
                interactionGroupKernel->addArg(function);
            if (globals.isInitialized())
                interactionGroupKernel->addArg(globals);
            if (hasParamDerivs)
                interactionGroupKernel->addArg(cc.getEnergyParamDerivBuffer());
            if (useNeighborList) {
                // Initialize kernels for building the interaction group neighbor list.

                prepareNeighborListKernel->addArg(cc.getNonbondedUtilities().getRebuildNeighborList());
                prepareNeighborListKernel->addArg(numGroupTiles);
                buildNeighborListKernel->addArg(cc.getNonbondedUtilities().getRebuildNeighborList());
                buildNeighborListKernel->addArg(numGroupTiles);
                buildNeighborListKernel->addArg(cc.getPosq());
                buildNeighborListKernel->addArg(interactionGroupData);
                buildNeighborListKernel->addArg(filteredGroupData);
                for (int i = 0; i < 5; i++)
                    buildNeighborListKernel->addArg(); // Periodic box information will be set just before it is executed.
            }
        }
        int forceThreadBlockSize = max(tileWidth, cc.getNonbondedUtilities().getForceThreadBlockSize());
        if (useNeighborList) {
            // Rebuild the neighbor list, if necessary.

            setPeriodicBoxArgs(cc, buildNeighborListKernel, 5);
            prepareNeighborListKernel->execute(1, 1);
            buildNeighborListKernel->execute(numGroupThreadBlocks*forceThreadBlockSize, forceThreadBlockSize);
        }
        setPeriodicBoxArgs(cc, interactionGroupKernel, 6);
        interactionGroupKernel->execute(numGroupThreadBlocks*forceThreadBlockSize, forceThreadBlockSize);
    }
    Vec3 a, b, c;
    cc.getPeriodicBoxVectors(a, b, c);
    double volume = a[0]*b[1]*c[2];
    map<string, double>& derivs = cc.getEnergyParamDerivWorkspace();
    for (int i = 0; i < longRangeCoefficientDerivs.size(); i++)
        derivs[forceCopy->getEnergyParameterDerivativeName(i)] += longRangeCoefficientDerivs[i]/volume;
    return longRangeCoefficient/volume;
}

void CommonCalcCustomNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const CustomNonbondedForce& force) {
    cc.setAsCurrent();
    int numParticles = force.getNumParticles();
    if (numParticles != cc.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Record the per-particle parameters.

    int paddedNumParticles = cc.getPaddedNumAtoms();
    int numParams = force.getNumPerParticleParameters();
    vector<vector<float> > paramVector(paddedNumParticles, vector<float>(numParams, 0));
    vector<double> parameters;
    for (int i = 0; i < numParticles; i++) {
        force.getParticleParameters(i, parameters);
        paramVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);

    // If necessary, recompute the long range correction.

    if (forceCopy != NULL) {
        CustomNonbondedForceImpl::calcLongRangeCorrection(force, context.getOwner(), longRangeCoefficient, longRangeCoefficientDerivs);
        hasInitializedLongRangeCorrection = true;
        *forceCopy = force;
    }

    // Mark that the current reordering may be invalid.

    cc.invalidateMolecules(info);
}

class CommonCalcGBSAOBCForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const GBSAOBCForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double charge1, charge2, radius1, radius2, scale1, scale2;
        force.getParticleParameters(particle1, charge1, radius1, scale1);
        force.getParticleParameters(particle2, charge2, radius2, scale2);
        return (charge1 == charge2 && radius1 == radius2 && scale1 == scale2);
    }
private:
    const GBSAOBCForce& force;
};

void CommonCalcGBSAOBCForceKernel::initialize(const System& system, const GBSAOBCForce& force) {
    cc.setAsCurrent();
    if (cc.getNumContexts() > 1)
        throw OpenMMException("GBSAOBCForce does not support using multiple devices");
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() && &system.getForce(forceIndex) != &force; ++forceIndex)
        ;
    string prefix = "obc"+cc.intToString(forceIndex)+"_";
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    params.initialize<mm_float2>(cc, cc.getPaddedNumAtoms(), "gbsaObcParams");
    int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    charges.initialize(cc, cc.getPaddedNumAtoms(), elementSize, "gbsaObcCharges");
    bornRadii.initialize(cc, cc.getPaddedNumAtoms(), elementSize, "bornRadii");
    obcChain.initialize(cc, cc.getPaddedNumAtoms(), elementSize, "obcChain");
    if (cc.getSupports64BitGlobalAtomics()) {
        bornSum.initialize<long long>(cc, cc.getPaddedNumAtoms(), "bornSum");
        bornForce.initialize<long long>(cc, cc.getPaddedNumAtoms(), "bornForce");
    }
    else {
        int bufferSize = cc.getPaddedNumAtoms()*nb.getNumForceBuffers();
        bornSum.initialize(cc, bufferSize, elementSize, "bornSum");
        bornForce.initialize(cc, bufferSize, elementSize, "bornForce");
    }
    cc.addAutoclearBuffer(bornSum);
    cc.addAutoclearBuffer(bornForce);
    vector<double> chargeVec(cc.getPaddedNumAtoms());
    vector<mm_float2> paramsVector(cc.getPaddedNumAtoms(), mm_float2(1,1));
    const double dielectricOffset = 0.009;
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, radius, scalingFactor;
        force.getParticleParameters(i, charge, radius, scalingFactor);
        radius -= dielectricOffset;
        chargeVec[i] = charge;
        paramsVector[i] = mm_float2((float) radius, (float) (scalingFactor*radius));
    }
    charges.upload(chargeVec, true);
    params.upload(paramsVector);
    prefactor = -ONE_4PI_EPS0*((1.0/force.getSoluteDielectric())-(1.0/force.getSolventDielectric()));
    surfaceAreaFactor = -6.0*4*M_PI*force.getSurfaceAreaEnergy();
    bool useCutoff = (force.getNonbondedMethod() != GBSAOBCForce::NoCutoff);
    bool usePeriodic = (force.getNonbondedMethod() != GBSAOBCForce::NoCutoff && force.getNonbondedMethod() != GBSAOBCForce::CutoffNonPeriodic);
    cutoff = force.getCutoffDistance();
    string source = CommonKernelSources::gbsaObc2;
    map<string, string> replacements;
    replacements["CHARGE1"] = prefix+"charge1";
    replacements["CHARGE2"] = prefix+"charge2";
    replacements["OBC_PARAMS1"] = prefix+"obcParams1";
    replacements["OBC_PARAMS2"] = prefix+"obcParams2";
    replacements["BORN_FORCE1"] = prefix+"bornForce1";
    replacements["BORN_FORCE2"] = prefix+"bornForce2";
    source = cc.replaceStrings(source, replacements);
    nb.addInteraction(useCutoff, usePeriodic, false, cutoff, vector<vector<int> >(), source, force.getForceGroup());
    nb.addParameter(ComputeParameterInfo(charges, prefix+"charge", "float", 1));
    nb.addParameter(ComputeParameterInfo(params, prefix+"obcParams", "float", 2));
    nb.addParameter(ComputeParameterInfo(bornForce, prefix+"bornForce", cc.getSupports64BitGlobalAtomics() ? "mm_long" : "real", 1));
    info = new ForceInfo(force);
    cc.addForce(info);
}

double CommonCalcGBSAOBCForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    const int tileWidth = cc.getSIMDWidth();
    bool deviceIsCpu = cc.getIsCPU();
    if (!hasCreatedKernels) {
        // These Kernels cannot be created in initialize(), because the NonbondedUtilities has not been initialized yet then.

        hasCreatedKernels = true;
        maxTiles = (nb.getUseCutoff() ? nb.getInteractingTiles().getSize() : 0);
        int numAtomBlocks = cc.getPaddedNumAtoms()/tileWidth;
        map<string, string> defines;
        if (nb.getUseCutoff())
            defines["USE_CUTOFF"] = "1";
        if (nb.getUsePeriodic())
            defines["USE_PERIODIC"] = "1";
        defines["CUTOFF_SQUARED"] = cc.doubleToString(cutoff*cutoff);
        defines["CUTOFF"] = cc.doubleToString(cutoff);
        defines["PREFACTOR"] = cc.doubleToString(prefactor);
        defines["SURFACE_AREA_FACTOR"] = cc.doubleToString(surfaceAreaFactor);
        defines["NUM_ATOMS"] = cc.intToString(cc.getNumAtoms());
        defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
        defines["NUM_BLOCKS"] = cc.intToString(numAtomBlocks);
        defines["FORCE_WORK_GROUP_SIZE"] = cc.intToString(nb.getForceThreadBlockSize());
        defines["TILE_SIZE"] = to_string(tileWidth);
        int numExclusionTiles = nb.getExclusionTiles().getSize();
        defines["NUM_TILES_WITH_EXCLUSIONS"] = cc.intToString(numExclusionTiles);
        defines["FIRST_EXCLUSION_TILE"] = "0";
        defines["LAST_EXCLUSION_TILE"] = cc.intToString(numExclusionTiles);
        string file;
        if (deviceIsCpu)
            file = CommonKernelSources::gbsaObc_cpu;
        else
            file = CommonKernelSources::gbsaObc;
        ComputeProgram program = cc.compileProgram(file, defines);
        bool useLong = cc.getSupports64BitGlobalAtomics();
        computeBornSumKernel = program->createKernel("computeBornSum");
        computeBornSumKernel->addArg(bornSum);
        computeBornSumKernel->addArg(cc.getPosq());
        computeBornSumKernel->addArg(charges);
        computeBornSumKernel->addArg(params);
        if (nb.getUseCutoff()) {
            computeBornSumKernel->addArg(nb.getInteractingTiles());
            computeBornSumKernel->addArg(nb.getInteractionCount());
            for (int i = 0; i < 5; i++)
                computeBornSumKernel->addArg(); // The periodic box size arguments are set when the kernel is executed.
            computeBornSumKernel->addArg(maxTiles);
            computeBornSumKernel->addArg(nb.getBlockCenters());
            computeBornSumKernel->addArg(nb.getBlockBoundingBoxes());
            computeBornSumKernel->addArg(nb.getInteractingAtoms());
        }
        else
            computeBornSumKernel->addArg(numAtomBlocks*(numAtomBlocks+1)/2);
        computeBornSumKernel->addArg(nb.getExclusionTiles());
        force1Kernel = program->createKernel("computeGBSAForce1");
        force1Kernel->addArg(useLong ? cc.getLongForceBuffer() : cc.getForceBuffers());
        force1Kernel->addArg(bornForce);
        force1Kernel->addArg(cc.getEnergyBuffer());
        force1Kernel->addArg(cc.getPosq());
        force1Kernel->addArg(charges);
        force1Kernel->addArg(bornRadii);
        force1Kernel->addArg(); // Whether to include energy.
        if (nb.getUseCutoff()) {
            force1Kernel->addArg(nb.getInteractingTiles());
            force1Kernel->addArg(nb.getInteractionCount());
            for (int i = 0; i < 5; i++)
                force1Kernel->addArg(); // The periodic box size arguments are set when the kernel is executed.
            force1Kernel->addArg(maxTiles);
            force1Kernel->addArg(nb.getBlockCenters());
            force1Kernel->addArg(nb.getBlockBoundingBoxes());
            force1Kernel->addArg(nb.getInteractingAtoms());
        }
        else
            force1Kernel->addArg(numAtomBlocks*(numAtomBlocks+1)/2);
        force1Kernel->addArg(nb.getExclusionTiles());
        program = cc.compileProgram(CommonKernelSources::gbsaObcReductions, defines);
        reduceBornSumKernel = program->createKernel("reduceBornSum");
        reduceBornSumKernel->addArg(1.0f);
        reduceBornSumKernel->addArg(0.8f);
        reduceBornSumKernel->addArg(4.85f);
        reduceBornSumKernel->addArg(bornSum);
        if (!useLong) {
            reduceBornSumKernel->addArg(cc.getPaddedNumAtoms());
            reduceBornSumKernel->addArg(cc.getForceBuffers().getSize()/cc.getPaddedNumAtoms());
        }
        reduceBornSumKernel->addArg(params);
        reduceBornSumKernel->addArg(bornRadii);
        reduceBornSumKernel->addArg(obcChain);
        reduceBornForceKernel = program->createKernel("reduceBornForce");
        reduceBornForceKernel->addArg(bornForce);
        if (!useLong) {
            reduceBornForceKernel->addArg(cc.getPaddedNumAtoms());
            reduceBornForceKernel->addArg(cc.getForceBuffers().getSize()/cc.getPaddedNumAtoms());
        }
        reduceBornForceKernel->addArg(cc.getEnergyBuffer());
        reduceBornForceKernel->addArg(params);
        reduceBornForceKernel->addArg(bornRadii);
        reduceBornForceKernel->addArg(obcChain);
    }
    force1Kernel->setArg(6, (int) includeEnergy);
    if (nb.getUseCutoff()) {
        setPeriodicBoxArgs(cc, computeBornSumKernel, 6);
        setPeriodicBoxArgs(cc, force1Kernel, 9);
        if (maxTiles < nb.getInteractingTiles().getSize()) {
            maxTiles = nb.getInteractingTiles().getSize();
            computeBornSumKernel->setArg(11, maxTiles);
            force1Kernel->setArg(14, maxTiles);
        }
    }
    computeBornSumKernel->execute(nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize(), nb.getForceThreadBlockSize());
    reduceBornSumKernel->execute(cc.getPaddedNumAtoms());
    force1Kernel->execute(nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize(), nb.getForceThreadBlockSize());
    reduceBornForceKernel->execute(cc.getPaddedNumAtoms());
    return 0.0;
}

void CommonCalcGBSAOBCForceKernel::copyParametersToContext(ContextImpl& context, const GBSAOBCForce& force) {
    // Make sure the new parameters are acceptable.

    cc.setAsCurrent();
    int numParticles = force.getNumParticles();
    if (numParticles != cc.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Record the per-particle parameters.

    vector<double> chargeVector(cc.getPaddedNumAtoms(), 0.0);
    vector<mm_float2> paramsVector(cc.getPaddedNumAtoms());
    const double dielectricOffset = 0.009;
    for (int i = 0; i < numParticles; i++) {
        double charge, radius, scalingFactor;
        force.getParticleParameters(i, charge, radius, scalingFactor);
        chargeVector[i] = charge;
        radius -= dielectricOffset;
        paramsVector[i] = mm_float2((float) radius, (float) (scalingFactor*radius));
    }
    for (int i = numParticles; i < cc.getPaddedNumAtoms(); i++)
        paramsVector[i] = mm_float2(1,1);
    charges.upload(chargeVector, true);
    params.upload(paramsVector);

    // Mark that the current reordering may be invalid.

    cc.invalidateMolecules(info);
}

class CommonCalcCustomGBForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const CustomGBForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        vector<double> params1;
        vector<double> params2;
        force.getParticleParameters(particle1, params1);
        force.getParticleParameters(particle2, params2);
        for (int i = 0; i < (int) params1.size(); i++)
            if (params1[i] != params2[i])
                return false;
        return true;
    }
    int getNumParticleGroups() {
        return force.getNumExclusions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        force.getExclusionParticles(index, particle1, particle2);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        return true;
    }
private:
    const CustomGBForce& force;
};

CommonCalcCustomGBForceKernel::~CommonCalcCustomGBForceKernel() {
    cc.setAsCurrent();
    if (params != NULL)
        delete params;
    if (computedValues != NULL)
        delete computedValues;
    if (energyDerivs != NULL)
        delete energyDerivs;
    if (energyDerivChain != NULL)
        delete energyDerivChain;
    for (auto d : dValuedParam)
        delete d;
}

void CommonCalcCustomGBForceKernel::initialize(const System& system, const CustomGBForce& force) {
    cc.setAsCurrent();
    const int tileWidth = cc.getSIMDWidth();
    if (cc.getNumContexts() > 1)
        throw OpenMMException("CustomGBForce does not support using multiple devices");
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    cutoff = force.getCutoffDistance();
    bool useExclusionsForValue = false;
    numComputedValues = force.getNumComputedValues();
    vector<string> computedValueNames(numComputedValues);
    vector<string> computedValueExpressions(numComputedValues);
    if (numComputedValues > 0) {
        CustomGBForce::ComputationType type;
        force.getComputedValueParameters(0, computedValueNames[0], computedValueExpressions[0], type);
        if (type == CustomGBForce::SingleParticle)
            throw OpenMMException("The first computed value for a CustomGBForce must be of type ParticlePair or ParticlePairNoExclusions.");
        useExclusionsForValue = (type == CustomGBForce::ParticlePair);
        for (int i = 1; i < numComputedValues; i++) {
            force.getComputedValueParameters(i, computedValueNames[i], computedValueExpressions[i], type);
            if (type != CustomGBForce::SingleParticle)
                throw OpenMMException("A CustomGBForce may only have one computed value of type ParticlePair or ParticlePairNoExclusions.");
        }
    }
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() && &system.getForce(forceIndex) != &force; ++forceIndex)
        ;
    string prefix = "custom"+cc.intToString(forceIndex)+"_";

    // Record parameters and exclusions.

    int numParticles = force.getNumParticles();
    int paddedNumParticles = cc.getPaddedNumAtoms();
    int numParams = force.getNumPerParticleParameters();
    params = new ComputeParameterSet(cc, force.getNumPerParticleParameters(), paddedNumParticles, "customGBParameters", true);
    computedValues = new ComputeParameterSet(cc, numComputedValues, paddedNumParticles, "customGBComputedValues", true, cc.getUseDoublePrecision());
    if (force.getNumGlobalParameters() > 0)
        globals.initialize<float>(cc, force.getNumGlobalParameters(), "customGBGlobals");
    vector<vector<float> > paramVector(paddedNumParticles, vector<float>(numParams, 0));
    vector<vector<int> > exclusionList(numParticles);
    for (int i = 0; i < numParticles; i++) {
        vector<double> parameters;
        force.getParticleParameters(i, parameters);
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
        exclusionList[i].push_back(i);
    }
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int particle1, particle2;
        force.getExclusionParticles(i, particle1, particle2);
        exclusionList[particle1].push_back(particle2);
        exclusionList[particle2].push_back(particle1);
    }
    params->setParameterValues(paramVector);

    // Record the tabulated functions.

    map<string, Lepton::CustomFunction*> functions;
    vector<pair<string, string> > functionDefinitions;
    vector<const TabulatedFunction*> functionList;
    stringstream tableArgs;
    tabulatedFunctions.resize(force.getNumTabulatedFunctions());
    for (int i = 0; i < force.getNumTabulatedFunctions(); i++) {
        functionList.push_back(&force.getTabulatedFunction(i));
        string name = force.getTabulatedFunctionName(i);
        string arrayName = prefix+"table"+cc.intToString(i);
        functionDefinitions.push_back(make_pair(name, arrayName));
        functions[name] = cc.getExpressionUtilities().getFunctionPlaceholder(force.getTabulatedFunction(i));
        int width;
        vector<float> f = cc.getExpressionUtilities().computeFunctionCoefficients(force.getTabulatedFunction(i), width);
        tabulatedFunctions[i].initialize<float>(cc, f.size(), "TabulatedFunction");
        tabulatedFunctions[i].upload(f);
        nb.addArgument(ComputeParameterInfo(tabulatedFunctions[i], arrayName, "float", width));
        tableArgs << ", GLOBAL const float";
        if (width > 1)
            tableArgs << width;
        tableArgs << "* RESTRICT " << arrayName;
    }

    // Record the global parameters.

    globalParamNames.resize(force.getNumGlobalParameters());
    globalParamValues.resize(force.getNumGlobalParameters());
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        globalParamNames[i] = force.getGlobalParameterName(i);
        globalParamValues[i] = (float) force.getGlobalParameterDefaultValue(i);
    }
    if (globals.isInitialized())
        globals.upload(globalParamValues);

    // Record derivatives of expressions needed for the chain rule terms.

    vector<vector<Lepton::ParsedExpression> > valueGradientExpressions(numComputedValues);
    vector<vector<Lepton::ParsedExpression> > valueDerivExpressions(numComputedValues);
    vector<vector<Lepton::ParsedExpression> > valueParamDerivExpressions(numComputedValues);
    needParameterGradient = false;
    for (int i = 0; i < numComputedValues; i++) {
        Lepton::ParsedExpression ex = Lepton::Parser::parse(computedValueExpressions[i], functions).optimize();
        if (i > 0) {
            valueGradientExpressions[i].push_back(ex.differentiate("x").optimize());
            valueGradientExpressions[i].push_back(ex.differentiate("y").optimize());
            valueGradientExpressions[i].push_back(ex.differentiate("z").optimize());
            if (!isZeroExpression(valueGradientExpressions[i][0]) || !isZeroExpression(valueGradientExpressions[i][1]) || !isZeroExpression(valueGradientExpressions[i][2]))
                needParameterGradient = true;
            for (int j = 0; j < i; j++)
                valueDerivExpressions[i].push_back(ex.differentiate(computedValueNames[j]).optimize());
        }
        for (int j = 0; j < force.getNumEnergyParameterDerivatives(); j++)
            valueParamDerivExpressions[i].push_back(ex.differentiate(force.getEnergyParameterDerivativeName(j)).optimize());
    }
    vector<vector<Lepton::ParsedExpression> > energyDerivExpressions(force.getNumEnergyTerms());
    vector<vector<Lepton::ParsedExpression> > energyParamDerivExpressions(force.getNumEnergyTerms());
    vector<bool> needChainForValue(numComputedValues, false);
    for (int i = 0; i < force.getNumEnergyTerms(); i++) {
        string expression;
        CustomGBForce::ComputationType type;
        force.getEnergyTermParameters(i, expression, type);
        Lepton::ParsedExpression ex = Lepton::Parser::parse(expression, functions).optimize();
        for (int j = 0; j < numComputedValues; j++) {
            if (type == CustomGBForce::SingleParticle) {
                energyDerivExpressions[i].push_back(ex.differentiate(computedValueNames[j]).optimize());
                if (!isZeroExpression(energyDerivExpressions[i].back()))
                    needChainForValue[j] = true;
            }
            else {
                energyDerivExpressions[i].push_back(ex.differentiate(computedValueNames[j]+"1").optimize());
                if (!isZeroExpression(energyDerivExpressions[i].back()))
                    needChainForValue[j] = true;
                energyDerivExpressions[i].push_back(ex.differentiate(computedValueNames[j]+"2").optimize());
                if (!isZeroExpression(energyDerivExpressions[i].back()))
                    needChainForValue[j] = true;
            }
        }
        for (int j = 0; j < force.getNumEnergyParameterDerivatives(); j++)
            energyParamDerivExpressions[i].push_back(ex.differentiate(force.getEnergyParameterDerivativeName(j)).optimize());
    }
    bool deviceIsCpu = cc.getIsCPU();
    bool useLong = cc.getSupports64BitGlobalAtomics();
    int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    if (useLong) {
        valueBuffers.initialize<long long>(cc, cc.getPaddedNumAtoms(), "customGBValueBuffers");
        longEnergyDerivs.initialize<long long>(cc, numComputedValues*cc.getPaddedNumAtoms(), "customGBLongEnergyDerivatives");
        energyDerivs = new ComputeParameterSet(cc, numComputedValues, cc.getPaddedNumAtoms(), "customGBEnergyDerivatives", true);
    }
    else {
        int bufferSize = cc.getPaddedNumAtoms()*nb.getNumForceBuffers();
        valueBuffers.initialize(cc, bufferSize, elementSize, "customGBValueBuffers");
        energyDerivs = new ComputeParameterSet(cc, numComputedValues, bufferSize, "customGBEnergyDerivatives", true);
    }
    cc.addAutoclearBuffer(valueBuffers);
    energyDerivChain = new ComputeParameterSet(cc, numComputedValues, cc.getPaddedNumAtoms(), "customGBEnergyDerivativeChain", true);
    needEnergyParamDerivs = (force.getNumEnergyParameterDerivatives() > 0);
    dValue0dParam.resize(force.getNumEnergyParameterDerivatives());
    for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
        dValuedParam.push_back(new ComputeParameterSet(cc, numComputedValues, cc.getPaddedNumAtoms(), "dValuedParam", true, cc.getUseDoublePrecision()));
        if (useLong)
            dValue0dParam[i].initialize<long long>(cc, cc.getPaddedNumAtoms(), "dValue0dParam");
        else
            dValue0dParam[i].initialize(cc, cc.getPaddedNumAtoms()*nb.getNumForceBuffers(), elementSize, "dValue0dParam");
        cc.addAutoclearBuffer(dValue0dParam[i]);
        string name = force.getEnergyParameterDerivativeName(i);
        cc.addEnergyParameterDerivative(name);
    }

    // Create the kernels.

    bool useCutoff = (force.getNonbondedMethod() != CustomGBForce::NoCutoff);
    bool usePeriodic = (force.getNonbondedMethod() != CustomGBForce::NoCutoff && force.getNonbondedMethod() != CustomGBForce::CutoffNonPeriodic);
    int numAtomBlocks = cc.getPaddedNumAtoms()/tileWidth;
    {
        // Create the N2 value kernel.

        vector<pair<ExpressionTreeNode, string> > variables;
        map<string, string> rename;
        ExpressionTreeNode rnode(new Operation::Variable("r"));
        variables.push_back(make_pair(rnode, "r"));
        variables.push_back(make_pair(ExpressionTreeNode(new Operation::Square(), rnode), "r2"));
        variables.push_back(make_pair(ExpressionTreeNode(new Operation::Reciprocal(), rnode), "invR"));
        for (int i = 0; i < force.getNumPerParticleParameters(); i++) {
            const string& name = force.getPerParticleParameterName(i);
            variables.push_back(makeVariable(name+"1", "params"+params->getParameterSuffix(i, "1")));
            variables.push_back(makeVariable(name+"2", "params"+params->getParameterSuffix(i, "2")));
            rename[name+"1"] = name+"2";
            rename[name+"2"] = name+"1";
        }
        for (int i = 0; i < force.getNumGlobalParameters(); i++) {
            const string& name = force.getGlobalParameterName(i);
            string value = "globals["+cc.intToString(i)+"]";
            variables.push_back(makeVariable(name, value));
        }
        map<string, Lepton::ParsedExpression> n2ValueExpressions;
        stringstream n2ValueSource;
        Lepton::ParsedExpression ex = Lepton::Parser::parse(computedValueExpressions[0], functions).optimize();
        n2ValueExpressions["tempValue1 = "] = ex;
        n2ValueExpressions["tempValue2 = "] = ex.renameVariables(rename);
        for (int i = 0; i < valueParamDerivExpressions[0].size(); i++) {
            string variableBase = "temp_dValue0dParam"+cc.intToString(i+1);
            if (!isZeroExpression(valueParamDerivExpressions[0][i])) {
                n2ValueExpressions[variableBase+"_1 = "] = valueParamDerivExpressions[0][i];
                n2ValueExpressions[variableBase+"_2 = "] = valueParamDerivExpressions[0][i].renameVariables(rename);
            }
        }
        n2ValueSource << cc.getExpressionUtilities().createExpressions(n2ValueExpressions, variables, functionList, functionDefinitions, "temp");
        map<string, string> replacements;
        string n2ValueStr = n2ValueSource.str();
        replacements["COMPUTE_VALUE"] = n2ValueStr;
        stringstream extraArgs, atomParams, loadLocal1, loadLocal2, load1, load2, tempDerivs1, tempDerivs2, storeDeriv1, storeDeriv2;
        if (force.getNumGlobalParameters() > 0)
            extraArgs << ", GLOBAL const float* globals";
        pairValueUsesParam.resize(params->getParameterInfos().size(), false);
        for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
            ComputeParameterInfo& buffer = params->getParameterInfos()[i];
            string paramName = "params"+cc.intToString(i+1);
            if (n2ValueStr.find(paramName+"1") != n2ValueStr.npos || n2ValueStr.find(paramName+"2") != n2ValueStr.npos) {
                extraArgs << ", GLOBAL const " << buffer.getType() << "* RESTRICT global_" << paramName;
                atomParams << "LOCAL " << buffer.getType() << " local_" << paramName << "[LOCAL_BUFFER_SIZE];\n";
                loadLocal1 << "local_" << paramName << "[localAtomIndex] = " << paramName << "1;\n";
                loadLocal2 << "local_" << paramName << "[localAtomIndex] = global_" << paramName << "[j];\n";
                load1 << buffer.getType() << " " << paramName << "1 = global_" << paramName << "[atom1];\n";
                load2 << buffer.getType() << " " << paramName << "2 = local_" << paramName << "[atom2];\n";
                pairValueUsesParam[i] = true;
            }
        }
        for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
            string derivName = "dValue0dParam"+cc.intToString(i+1);
            if (useLong)
                extraArgs << ", GLOBAL mm_ulong* RESTRICT global_" << derivName;
            else
                extraArgs << ", GLOBAL real* RESTRICT global_" << derivName;
            atomParams << "LOCAL real local_" << derivName << "[LOCAL_BUFFER_SIZE];\n";
            loadLocal2 << "local_" << derivName << "[localAtomIndex] = 0;\n";
            load1 << "real " << derivName << " = 0;\n";
            if (!isZeroExpression(valueParamDerivExpressions[0][i])) {
                load2 << "real temp_" << derivName << "_1 = 0;\n";
                load2 << "real temp_" << derivName << "_2 = 0;\n";
                tempDerivs1 << derivName << " += temp_" << derivName << "_1;\n";
                if (deviceIsCpu)
                    tempDerivs2 << "local_" << derivName << "[j] += temp_" << derivName << "_2;\n";
                else
                    tempDerivs2 << "local_" << derivName << "[tbx+tj] += temp_" << derivName << "_2;\n";
                if (useLong) {
                    storeDeriv1 << "ATOMIC_ADD(&global_" << derivName << "[offset1], (mm_ulong) ((mm_long) (" << derivName << "*0x100000000)));\n";
                    if (deviceIsCpu)
                        storeDeriv2 << "ATOMIC_ADD(&global_" << derivName << "[offset2], (mm_ulong) ((mm_long) (local_" << derivName << "[tgx]*0x100000000)));\n";
                    else
                        storeDeriv2 << "ATOMIC_ADD(&global_" << derivName << "[offset2], (mm_ulong) ((mm_long) (local_" << derivName << "[LOCAL_ID]*0x100000000)));\n";
                }
                else {
                    storeDeriv1 << "global_" << derivName << "[offset1] += " << derivName << ";\n";
                    if (deviceIsCpu)
                        storeDeriv2 << "global_" << derivName << "[offset2] += local_" << derivName << "[tgx];\n";
                    else
                        storeDeriv2 << "global_" << derivName << "[offset2] += local_" << derivName << "[LOCAL_ID];\n";
                }
            }
        }
        replacements["PARAMETER_ARGUMENTS"] = extraArgs.str()+tableArgs.str();
        replacements["ATOM_PARAMETER_DATA"] = atomParams.str();
        replacements["LOAD_LOCAL_PARAMETERS_FROM_1"] = loadLocal1.str();
        replacements["LOAD_LOCAL_PARAMETERS_FROM_GLOBAL"] = loadLocal2.str();
        replacements["LOAD_ATOM1_PARAMETERS"] = load1.str();
        replacements["LOAD_ATOM2_PARAMETERS"] = load2.str();
        replacements["ADD_TEMP_DERIVS1"] = tempDerivs1.str();
        replacements["ADD_TEMP_DERIVS2"] = tempDerivs2.str();
        replacements["STORE_PARAM_DERIVS1"] = storeDeriv1.str();
        replacements["STORE_PARAM_DERIVS2"] = storeDeriv2.str();
        if (useCutoff)
            pairValueDefines["USE_CUTOFF"] = "1";
        if (usePeriodic)
            pairValueDefines["USE_PERIODIC"] = "1";
        if (useExclusionsForValue)
            pairValueDefines["USE_EXCLUSIONS"] = "1";
        pairValueDefines["LOCAL_BUFFER_SIZE"] = cc.intToString(deviceIsCpu ? tileWidth : nb.getForceThreadBlockSize());
        pairValueDefines["CUTOFF_SQUARED"] = cc.doubleToString(cutoff*cutoff);
        pairValueDefines["NUM_ATOMS"] = cc.intToString(cc.getNumAtoms());
        pairValueDefines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
        pairValueDefines["NUM_BLOCKS"] = cc.intToString(numAtomBlocks);
        pairValueDefines["TILE_SIZE"] = to_string(tileWidth);
        pairValueDefines["tileflags_t"] = (tileWidth > 32 ? "unsigned long" : "unsigned int");
        string file;
        if (deviceIsCpu)
            file = CommonKernelSources::customGBValueN2_cpu;
        else
            file = CommonKernelSources::customGBValueN2;
        pairValueSrc = cc.replaceStrings(file, replacements);
    }
    {
        // Create the kernel to reduce the N2 value and calculate other values.

        stringstream reductionSource, extraArgs, deriv0;
        if (force.getNumGlobalParameters() > 0)
            extraArgs << ", GLOBAL const float* globals";
        for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
            ComputeParameterInfo& buffer = params->getParameterInfos()[i];
            string paramName = "params"+cc.intToString(i+1);
            extraArgs << ", GLOBAL const " << buffer.getType() << "* RESTRICT " << paramName;
        }
        for (int i = 0; i < (int) computedValues->getParameterInfos().size(); i++) {
            ComputeParameterInfo& buffer = computedValues->getParameterInfos()[i];
            string valueName = "values"+cc.intToString(i+1);
            extraArgs << ", GLOBAL " << buffer.getType() << "* RESTRICT global_" << valueName;
            reductionSource << buffer.getType() << " local_" << valueName << ";\n";
        }
        for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
            string variableName = "dValuedParam_0_"+cc.intToString(i);
            if (useLong) {
                extraArgs << ", GLOBAL const mm_long* RESTRICT dValue0dParam" << i;
                deriv0 << "real " << variableName << " = RECIP((real) 0x100000000)*dValue0dParam" << i << "[index];\n";
            }
            else {
                extraArgs << ", GLOBAL const real* RESTRICT dValue0dParam" << i;
                deriv0 << "real " << variableName << " = dValue0dParam" << i << "[index];\n";
                deriv0 << "for (int i = index+bufferSize; i < totalSize; i += bufferSize)\n";
                deriv0 << "    " << variableName << " += dValue0dParam" << i << "[i];\n";
            }
            for (int j = 0; j < dValuedParam[i]->getParameterInfos().size(); j++)
                extraArgs << ", GLOBAL real* RESTRICT global_dValuedParam_" << j << "_" << i;
            deriv0 << "global_dValuedParam_0_" << i << "[index] = dValuedParam_0_" << i << ";\n";
        }
        reductionSource << "local_values" << computedValues->getParameterSuffix(0) << " = sum;\n";
        map<string, string> variables;
        variables["x"] = "pos.x";
        variables["y"] = "pos.y";
        variables["z"] = "pos.z";
        for (int i = 0; i < force.getNumPerParticleParameters(); i++)
            variables[force.getPerParticleParameterName(i)] = "params"+params->getParameterSuffix(i, "[index]");
        for (int i = 0; i < force.getNumGlobalParameters(); i++)
            variables[force.getGlobalParameterName(i)] = "globals["+cc.intToString(i)+"]";
        for (int i = 1; i < numComputedValues; i++) {
            variables[computedValueNames[i-1]] = "local_values"+computedValues->getParameterSuffix(i-1);
            map<string, Lepton::ParsedExpression> valueExpressions;
            valueExpressions["local_values"+computedValues->getParameterSuffix(i)+" = "] = Lepton::Parser::parse(computedValueExpressions[i], functions).optimize();
            reductionSource << cc.getExpressionUtilities().createExpressions(valueExpressions, variables, functionList, functionDefinitions, "value"+cc.intToString(i)+"_temp");
        }
        for (int i = 0; i < (int) computedValues->getParameterInfos().size(); i++) {
            string valueName = "values"+cc.intToString(i+1);
            reductionSource << "global_" << valueName << "[index] = local_" << valueName << ";\n";
        }
        if (needEnergyParamDerivs) {
            map<string, Lepton::ParsedExpression> derivExpressions;
            for (int i = 1; i < numComputedValues; i++) {
                for (int j = 0; j < valueParamDerivExpressions[i].size(); j++)
                    derivExpressions["real dValuedParam_"+cc.intToString(i)+"_"+cc.intToString(j)+" = "] = valueParamDerivExpressions[i][j];
                for (int j = 0; j < i; j++)
                    derivExpressions["real dVdV_"+cc.intToString(i)+"_"+cc.intToString(j)+" = "] = valueDerivExpressions[i][j];
            }
            reductionSource << cc.getExpressionUtilities().createExpressions(derivExpressions, variables, functionList, functionDefinitions, "derivChain_temp");
            for (int i = 1; i < numComputedValues; i++) {
                for (int j = 0; j < i; j++)
                    for (int k = 0; k < valueParamDerivExpressions[i].size(); k++)
                        reductionSource << "dValuedParam_" << i << "_" << k << " += dVdV_" << i << "_" << j << "*dValuedParam_" << j <<"_" << k << ";\n";
                for (int j = 0; j < valueParamDerivExpressions[i].size(); j++)
                    reductionSource << "global_dValuedParam_" << i << "_" << j << "[index] = dValuedParam_" << i << "_" << j << ";\n";
            }
        }
        map<string, string> replacements;
        replacements["PARAMETER_ARGUMENTS"] = extraArgs.str()+tableArgs.str();
        replacements["REDUCE_PARAM0_DERIV"] = deriv0.str();
        replacements["COMPUTE_VALUES"] = reductionSource.str();
        map<string, string> defines;
        defines["NUM_ATOMS"] = cc.intToString(cc.getNumAtoms());
        ComputeProgram program = cc.compileProgram(cc.replaceStrings(CommonKernelSources::customGBValuePerParticle, replacements), defines);
        perParticleValueKernel = program->createKernel("computePerParticleValues");
    }
    {
        // Create the N2 energy kernel.

        vector<pair<ExpressionTreeNode, string> > variables;
        ExpressionTreeNode rnode(new Operation::Variable("r"));
        variables.push_back(make_pair(rnode, "r"));
        variables.push_back(make_pair(ExpressionTreeNode(new Operation::Square(), rnode), "r2"));
        variables.push_back(make_pair(ExpressionTreeNode(new Operation::Reciprocal(), rnode), "invR"));
        for (int i = 0; i < force.getNumPerParticleParameters(); i++) {
            const string& name = force.getPerParticleParameterName(i);
            variables.push_back(makeVariable(name+"1", "params"+params->getParameterSuffix(i, "1")));
            variables.push_back(makeVariable(name+"2", "params"+params->getParameterSuffix(i, "2")));
        }
        for (int i = 0; i < numComputedValues; i++) {
            variables.push_back(makeVariable(computedValueNames[i]+"1", "values"+computedValues->getParameterSuffix(i, "1")));
            variables.push_back(makeVariable(computedValueNames[i]+"2", "values"+computedValues->getParameterSuffix(i, "2")));
        }
        for (int i = 0; i < force.getNumGlobalParameters(); i++)
            variables.push_back(makeVariable(force.getGlobalParameterName(i), "globals["+cc.intToString(i)+"]"));
        stringstream n2EnergySource;
        bool anyExclusions = (force.getNumExclusions() > 0);
        for (int i = 0; i < force.getNumEnergyTerms(); i++) {
            string expression;
            CustomGBForce::ComputationType type;
            force.getEnergyTermParameters(i, expression, type);
            if (type == CustomGBForce::SingleParticle)
                continue;
            bool exclude = (anyExclusions && type == CustomGBForce::ParticlePair);
            map<string, Lepton::ParsedExpression> n2EnergyExpressions;
            n2EnergyExpressions["tempEnergy += "] = Lepton::Parser::parse(expression, functions).optimize();
            n2EnergyExpressions["dEdR += "] = Lepton::Parser::parse(expression, functions).differentiate("r").optimize();
            if (useLong) {
                for (int j = 0; j < numComputedValues; j++) {
                    if (needChainForValue[j]) {
                        string index = cc.intToString(j+1);
                        n2EnergyExpressions["/*"+cc.intToString(i+1)+"*/ deriv"+index+"_1 += "] = energyDerivExpressions[i][2*j];
                        n2EnergyExpressions["/*"+cc.intToString(i+1)+"*/ deriv"+index+"_2 += "] = energyDerivExpressions[i][2*j+1];
                    }
                }
            }
            else {
                for (int j = 0; j < numComputedValues; j++) {
                    if (needChainForValue[j]) {
                        n2EnergyExpressions["/*"+cc.intToString(i+1)+"*/ deriv"+energyDerivs->getParameterSuffix(j, "_1")+" += "] = energyDerivExpressions[i][2*j];
                        n2EnergyExpressions["/*"+cc.intToString(i+1)+"*/ deriv"+energyDerivs->getParameterSuffix(j, "_2")+" += "] = energyDerivExpressions[i][2*j+1];
                    }
                }
            }
            for (int j = 0; j < force.getNumEnergyParameterDerivatives(); j++)
                n2EnergyExpressions["energyParamDeriv"+cc.intToString(j)+" += interactionScale*"] = energyParamDerivExpressions[i][j];
            if (exclude)
                n2EnergySource << "if (!isExcluded) {\n";
            n2EnergySource << cc.getExpressionUtilities().createExpressions(n2EnergyExpressions, variables, functionList, functionDefinitions, "temp");
            if (exclude)
                n2EnergySource << "}\n";
        }
        map<string, string> replacements;
        string n2EnergyStr = n2EnergySource.str();
        replacements["COMPUTE_INTERACTION"] = n2EnergyStr;
        stringstream extraArgs, atomParams, loadLocal1, loadLocal2, clearLocal, load1, load2, declare1, recordDeriv, storeDerivs1, storeDerivs2, initParamDerivs, saveParamDerivs;
        if (force.getNumGlobalParameters() > 0)
            extraArgs << ", GLOBAL const float* globals";
        pairEnergyUsesParam.resize(params->getParameterInfos().size(), false);
        for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
            ComputeParameterInfo& buffer = params->getParameterInfos()[i];
            string paramName = "params"+cc.intToString(i+1);
            if (n2EnergyStr.find(paramName+"1") != n2EnergyStr.npos || n2EnergyStr.find(paramName+"2") != n2EnergyStr.npos) {
                extraArgs << ", GLOBAL const " << buffer.getType() << "* RESTRICT global_" << paramName;
                atomParams << "LOCAL " << buffer.getType() << " local_" << paramName << "[LOCAL_BUFFER_SIZE];\n";
                loadLocal1 << "local_" << paramName << "[localAtomIndex] = " << paramName << "1;\n";
                loadLocal2 << "local_" << paramName << "[localAtomIndex] = global_" << paramName << "[j];\n";
                load1 << buffer.getType() << " " << paramName << "1 = global_" << paramName << "[atom1];\n";
                load2 << buffer.getType() << " " << paramName << "2 = local_" << paramName << "[atom2];\n";
                pairEnergyUsesParam[i] = true;
            }
        }
        pairEnergyUsesValue.resize(computedValues->getParameterInfos().size(), false);
        for (int i = 0; i < (int) computedValues->getParameterInfos().size(); i++) {
            ComputeParameterInfo& buffer = computedValues->getParameterInfos()[i];
            string valueName = "values"+cc.intToString(i+1);
            if (n2EnergyStr.find(valueName+"1") != n2EnergyStr.npos || n2EnergyStr.find(valueName+"2") != n2EnergyStr.npos) {
                extraArgs << ", GLOBAL const " << buffer.getType() << "* RESTRICT global_" << valueName;
                atomParams << "LOCAL " << buffer.getType() << " local_" << valueName << "[LOCAL_BUFFER_SIZE];\n";
                loadLocal1 << "local_" << valueName << "[localAtomIndex] = " << valueName << "1;\n";
                loadLocal2 << "local_" << valueName << "[localAtomIndex] = global_" << valueName << "[j];\n";
                load1 << buffer.getType() << " " << valueName << "1 = global_" << valueName << "[atom1];\n";
                load2 << buffer.getType() << " " << valueName << "2 = local_" << valueName << "[atom2];\n";
                pairEnergyUsesValue[i] = true;
            }
        }
        if (useLong) {
            extraArgs << ", GLOBAL mm_ulong* RESTRICT derivBuffers";
            for (int i = 0; i < numComputedValues; i++) {
                string index = cc.intToString(i+1);
                atomParams << "LOCAL real local_deriv" << index << "[LOCAL_BUFFER_SIZE];\n";
                clearLocal << "local_deriv" << index << "[localAtomIndex] = 0.0f;\n";
                declare1 << "real deriv" << index << "_1 = 0;\n";
                load2 << "real deriv" << index << "_2 = 0;\n";
                recordDeriv << "local_deriv" << index << "[atom2] += deriv" << index << "_2;\n";
                storeDerivs1 << "STORE_DERIVATIVE_1(" << index << ")\n";
                storeDerivs2 << "STORE_DERIVATIVE_2(" << index << ")\n";
            }
        }
        else {
            for (int i = 0; i < (int) energyDerivs->getParameterInfos().size(); i++) {
                const ComputeParameterInfo& buffer = energyDerivs->getParameterInfos()[i];
                string index = cc.intToString(i+1);
                extraArgs << ", GLOBAL " << buffer.getType() << "* RESTRICT derivBuffers" << index;
                atomParams << "LOCAL " << buffer.getType() << " local_deriv" << index << "[LOCAL_BUFFER_SIZE];\n";
                clearLocal << "local_deriv" << index << "[localAtomIndex] = 0.0f;\n";
                declare1 << buffer.getType() << " deriv" << index << "_1 = 0.0f;\n";
                load2 << buffer.getType() << " deriv" << index << "_2 = 0.0f;\n";
                recordDeriv << "local_deriv" << index << "[atom2] += deriv" << index << "_2;\n";
                storeDerivs1 << "STORE_DERIVATIVE_1(" << index << ")\n";
                storeDerivs2 << "STORE_DERIVATIVE_2(" << index << ")\n";
            }
        }
        if (needEnergyParamDerivs) {
            extraArgs << ", GLOBAL mixed* RESTRICT energyParamDerivs";
            const vector<string>& allParamDerivNames = cc.getEnergyParamDerivNames();
            int numDerivs = allParamDerivNames.size();
            for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
                initParamDerivs << "mixed energyParamDeriv" << i << " = 0;\n";
                for (int index = 0; index < numDerivs; index++)
                    if (allParamDerivNames[index] == force.getEnergyParameterDerivativeName(i))
                        saveParamDerivs << "energyParamDerivs[GLOBAL_ID*" << numDerivs << "+" << index << "] += energyParamDeriv" << i << ";\n";
            }
        }
        replacements["PARAMETER_ARGUMENTS"] = extraArgs.str()+tableArgs.str();
        replacements["ATOM_PARAMETER_DATA"] = atomParams.str();
        replacements["LOAD_LOCAL_PARAMETERS_FROM_1"] = loadLocal1.str();
        replacements["LOAD_LOCAL_PARAMETERS_FROM_GLOBAL"] = loadLocal2.str();
        replacements["CLEAR_LOCAL_DERIVATIVES"] = clearLocal.str();
        replacements["LOAD_ATOM1_PARAMETERS"] = load1.str();
        replacements["LOAD_ATOM2_PARAMETERS"] = load2.str();
        replacements["DECLARE_ATOM1_DERIVATIVES"] = declare1.str();
        replacements["RECORD_DERIVATIVE_2"] = recordDeriv.str();
        replacements["STORE_DERIVATIVES_1"] = storeDerivs1.str();
        replacements["STORE_DERIVATIVES_2"] = storeDerivs2.str();
        replacements["INIT_PARAM_DERIVS"] = initParamDerivs.str();
        replacements["SAVE_PARAM_DERIVS"] = saveParamDerivs.str();
        if (useCutoff)
            pairEnergyDefines["USE_CUTOFF"] = "1";
        if (usePeriodic)
            pairEnergyDefines["USE_PERIODIC"] = "1";
        if (anyExclusions)
            pairEnergyDefines["USE_EXCLUSIONS"] = "1";
        pairEnergyDefines["LOCAL_BUFFER_SIZE"] = cc.intToString(deviceIsCpu ? tileWidth : nb.getForceThreadBlockSize());
        pairEnergyDefines["CUTOFF_SQUARED"] = cc.doubleToString(cutoff*cutoff);
        pairEnergyDefines["NUM_ATOMS"] = cc.intToString(cc.getNumAtoms());
        pairEnergyDefines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
        pairEnergyDefines["NUM_BLOCKS"] = cc.intToString(numAtomBlocks);
        pairEnergyDefines["TILE_SIZE"] = to_string(tileWidth);
        pairValueDefines["tileflags_t"] = (tileWidth > 32 ? "unsigned long" : "unsigned int") ;
        string file;
        if (deviceIsCpu)
            file = CommonKernelSources::customGBEnergyN2_cpu;
        else
            file = CommonKernelSources::customGBEnergyN2;
        pairEnergySrc = cc.replaceStrings(file, replacements);
    }
    {
        // Create the kernel to reduce the derivatives and calculate per-particle energy terms.

        stringstream compute, extraArgs, reduce, initParamDerivs, saveParamDerivs;
        if (force.getNumGlobalParameters() > 0)
            extraArgs << ", GLOBAL const float* globals";
        for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
            ComputeParameterInfo& buffer = params->getParameterInfos()[i];
            string paramName = "params"+cc.intToString(i+1);
            extraArgs << ", GLOBAL const " << buffer.getType() << "* RESTRICT " << paramName;
        }
        for (int i = 0; i < (int) computedValues->getParameterInfos().size(); i++) {
            ComputeParameterInfo& buffer = computedValues->getParameterInfos()[i];
            string valueName = "values"+cc.intToString(i+1);
            extraArgs << ", GLOBAL const " << buffer.getType() << "* RESTRICT " << valueName;
        }
        for (int i = 0; i < (int) energyDerivs->getParameterInfos().size(); i++) {
            ComputeParameterInfo& buffer = energyDerivs->getParameterInfos()[i];
            string index = cc.intToString(i+1);
            extraArgs << ", GLOBAL " << buffer.getType() << "* RESTRICT derivBuffers" << index;
            compute << buffer.getType() << " deriv" << index << " = derivBuffers" << index << "[index];\n";
        }
        for (int i = 0; i < (int) energyDerivChain->getParameterInfos().size(); i++) {
            ComputeParameterInfo& buffer = energyDerivChain->getParameterInfos()[i];
            string index = cc.intToString(i+1);
            extraArgs << ", GLOBAL " << buffer.getType() << "* RESTRICT derivChain" << index;
        }
        if (useLong) {
            extraArgs << ", GLOBAL const mm_long* RESTRICT derivBuffersIn";
            for (int i = 0; i < energyDerivs->getNumParameters(); ++i)
                reduce << "derivBuffers" << energyDerivs->getParameterSuffix(i, "[index]") <<
                        " = RECIP((real) 0x100000000)*derivBuffersIn[index+PADDED_NUM_ATOMS*" << cc.intToString(i) << "];\n";
        }
        else {
            for (int i = 0; i < (int) energyDerivs->getParameterInfos().size(); i++)
                reduce << "REDUCE_VALUE(derivBuffers" << cc.intToString(i+1) << ", " << energyDerivs->getParameterInfos()[i].getType() << ")\n";
        }
        if (needEnergyParamDerivs) {
            extraArgs << ", GLOBAL mixed* RESTRICT energyParamDerivs";
            const vector<string>& allParamDerivNames = cc.getEnergyParamDerivNames();
            int numDerivs = allParamDerivNames.size();
            for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
                initParamDerivs << "mixed energyParamDeriv" << i << " = 0;\n";
                for (int index = 0; index < numDerivs; index++)
                    if (allParamDerivNames[index] == force.getEnergyParameterDerivativeName(i))
                        saveParamDerivs << "energyParamDerivs[GLOBAL_ID*" << numDerivs << "+" << index << "] += energyParamDeriv" << i << ";\n";
            }
        }

        // Compute the various expressions.

        map<string, string> variables;
        variables["x"] = "pos.x";
        variables["y"] = "pos.y";
        variables["z"] = "pos.z";
        for (int i = 0; i < force.getNumPerParticleParameters(); i++)
            variables[force.getPerParticleParameterName(i)] = "params"+params->getParameterSuffix(i, "[index]");
        for (int i = 0; i < force.getNumGlobalParameters(); i++)
            variables[force.getGlobalParameterName(i)] = "globals["+cc.intToString(i)+"]";
        for (int i = 0; i < numComputedValues; i++)
            variables[computedValueNames[i]] = "values"+computedValues->getParameterSuffix(i, "[index]");
        map<string, Lepton::ParsedExpression> expressions;
        for (int i = 0; i < force.getNumEnergyTerms(); i++) {
            string expression;
            CustomGBForce::ComputationType type;
            force.getEnergyTermParameters(i, expression, type);
            if (type != CustomGBForce::SingleParticle)
                continue;
            Lepton::ParsedExpression parsed = Lepton::Parser::parse(expression, functions).optimize();
            expressions["/*"+cc.intToString(i+1)+"*/ energy += "] = parsed;
            for (int j = 0; j < numComputedValues; j++)
                expressions["/*"+cc.intToString(i+1)+"*/ deriv"+energyDerivs->getParameterSuffix(j)+" += "] = energyDerivExpressions[i][j];
            Lepton::ParsedExpression gradx = parsed.differentiate("x").optimize();
            Lepton::ParsedExpression grady = parsed.differentiate("y").optimize();
            Lepton::ParsedExpression gradz = parsed.differentiate("z").optimize();
            if (!isZeroExpression(gradx))
                expressions["/*"+cc.intToString(i+1)+"*/ force.x -= "] = gradx;
            if (!isZeroExpression(grady))
                expressions["/*"+cc.intToString(i+1)+"*/ force.y -= "] = grady;
            if (!isZeroExpression(gradz))
                expressions["/*"+cc.intToString(i+1)+"*/ force.z -= "] = gradz;
            for (int j = 0; j < force.getNumEnergyParameterDerivatives(); j++)
                expressions["/*"+cc.intToString(i+1)+"*/ energyParamDeriv"+cc.intToString(j)+" += "] = energyParamDerivExpressions[i][j];
        }
        for (int i = 1; i < numComputedValues; i++)
            for (int j = 0; j < i; j++)
                expressions["real dV"+cc.intToString(i)+"dV"+cc.intToString(j)+" = "] = valueDerivExpressions[i][j];
        compute << cc.getExpressionUtilities().createExpressions(expressions, variables, functionList, functionDefinitions, "temp");

        // Record values.

        for (int i = 0; i < (int) energyDerivs->getParameterInfos().size(); i++) {
            string index = cc.intToString(i+1);
            compute << "derivBuffers" << index << "[index] = deriv" << index << ";\n";
        }
        if (useLong) {
            compute << "forceBuffers[index] += (mm_long) (force.x*0x100000000);\n";
            compute << "forceBuffers[index+PADDED_NUM_ATOMS] += (mm_long) (force.y*0x100000000);\n";
            compute << "forceBuffers[index+PADDED_NUM_ATOMS*2] += (mm_long) (force.z*0x100000000);\n";
        }
        else
            compute << "forceBuffers[index] = forceBuffers[index]+make_real4(force.x, force.y, force.z, 0);\n";
        for (int i = 1; i < numComputedValues; i++) {
            compute << "real totalDeriv"<<i<<" = dV"<<i<<"dV0";
            for (int j = 1; j < i; j++)
                compute << " + totalDeriv"<<j<<"*dV"<<i<<"dV"<<j;
            compute << ";\n";
            compute << "deriv"<<(i+1)<<" *= totalDeriv"<<i<<";\n";
        }
        for (int i = 0; i < (int) energyDerivs->getParameterInfos().size(); i++) {
            string index = cc.intToString(i+1);
            compute << "derivChain" << index << "[index] = deriv" << index << ";\n";
        }
        map<string, string> replacements;
        replacements["PARAMETER_ARGUMENTS"] = extraArgs.str()+tableArgs.str();
        replacements["REDUCE_DERIVATIVES"] = reduce.str();
        replacements["COMPUTE_ENERGY"] = compute.str();
        replacements["INIT_PARAM_DERIVS"] = initParamDerivs.str();
        replacements["SAVE_PARAM_DERIVS"] = saveParamDerivs.str();
        map<string, string> defines;
        defines["NUM_ATOMS"] = cc.intToString(cc.getNumAtoms());
        defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
        ComputeProgram program = cc.compileProgram(cc.replaceStrings(CommonKernelSources::customGBEnergyPerParticle, replacements), defines);
        perParticleEnergyKernel = program->createKernel("computePerParticleEnergy");
    }
    if (needParameterGradient || needEnergyParamDerivs) {
        // Create the kernel to compute chain rule terms for computed values that depend explicitly on particle coordinates, and for
        // derivatives with respect to global parameters.

        stringstream compute, extraArgs, initParamDerivs, saveParamDerivs;
        if (force.getNumGlobalParameters() > 0)
            extraArgs << ", GLOBAL const float* globals";
        for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
            ComputeParameterInfo& buffer = params->getParameterInfos()[i];
            string paramName = "params"+cc.intToString(i+1);
            extraArgs << ", GLOBAL const " << buffer.getType() << "* RESTRICT " << paramName;
        }
        for (int i = 0; i < (int) computedValues->getParameterInfos().size(); i++) {
            ComputeParameterInfo& buffer = computedValues->getParameterInfos()[i];
            string valueName = "values"+cc.intToString(i+1);
            extraArgs << ", GLOBAL const " << buffer.getType() << "* RESTRICT " << valueName;
        }
        for (int i = 0; i < (int) energyDerivs->getParameterInfos().size(); i++) {
            ComputeParameterInfo& buffer = energyDerivs->getParameterInfos()[i];
            string index = cc.intToString(i+1);
            extraArgs << ", GLOBAL " << buffer.getType() << "* RESTRICT derivBuffers" << index;
            compute << buffer.getType() << " deriv" << index << " = derivBuffers" << index << "[index];\n";
        }
        if (needEnergyParamDerivs) {
            extraArgs << ", GLOBAL mixed* RESTRICT energyParamDerivs";
            const vector<string>& allParamDerivNames = cc.getEnergyParamDerivNames();
            int numDerivs = allParamDerivNames.size();
            for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
                for (int j = 0; j < dValuedParam[i]->getParameterInfos().size(); j++)
                    extraArgs << ", GLOBAL real* RESTRICT dValuedParam_" << j << "_" << i;
                initParamDerivs << "mixed energyParamDeriv" << i << " = 0;\n";
                for (int index = 0; index < numDerivs; index++)
                    if (allParamDerivNames[index] == force.getEnergyParameterDerivativeName(i))
                        saveParamDerivs << "energyParamDerivs[GLOBAL_ID*" << numDerivs << "+" << index << "] += energyParamDeriv" << i << ";\n";
            }
        }
        map<string, string> variables;
        variables["x"] = "pos.x";
        variables["y"] = "pos.y";
        variables["z"] = "pos.z";
        for (int i = 0; i < force.getNumPerParticleParameters(); i++)
            variables[force.getPerParticleParameterName(i)] = "params"+params->getParameterSuffix(i, "[index]");
        for (int i = 0; i < force.getNumGlobalParameters(); i++)
            variables[force.getGlobalParameterName(i)] = "globals["+cc.intToString(i)+"]";
        for (int i = 0; i < numComputedValues; i++)
            variables[computedValueNames[i]] = "values"+computedValues->getParameterSuffix(i, "[index]");
        if (needParameterGradient) {
            for (int i = 1; i < numComputedValues; i++) {
                string is = cc.intToString(i);
                compute << "real3 dV"<<is<<"dR = make_real3(0);\n";
                for (int j = 1; j < i; j++) {
                    if (!isZeroExpression(valueDerivExpressions[i][j])) {
                        map<string, Lepton::ParsedExpression> derivExpressions;
                        string js = cc.intToString(j);
                        derivExpressions["real dV"+is+"dV"+js+" = "] = valueDerivExpressions[i][j];
                        compute << cc.getExpressionUtilities().createExpressions(derivExpressions, variables, functionList, functionDefinitions, "temp_"+is+"_"+js);
                        compute << "dV"<<is<<"dR += dV"<<is<<"dV"<<js<<"*dV"<<js<<"dR;\n";
                    }
                }
                map<string, Lepton::ParsedExpression> gradientExpressions;
                if (!isZeroExpression(valueGradientExpressions[i][0]))
                    gradientExpressions["dV"+is+"dR.x += "] = valueGradientExpressions[i][0];
                if (!isZeroExpression(valueGradientExpressions[i][1]))
                    gradientExpressions["dV"+is+"dR.y += "] = valueGradientExpressions[i][1];
                if (!isZeroExpression(valueGradientExpressions[i][2]))
                    gradientExpressions["dV"+is+"dR.z += "] = valueGradientExpressions[i][2];
                compute << cc.getExpressionUtilities().createExpressions(gradientExpressions, variables, functionList, functionDefinitions, "temp");
            }
            for (int i = 1; i < numComputedValues; i++)
                compute << "force -= deriv"<<energyDerivs->getParameterSuffix(i)<<"*dV"<<i<<"dR;\n";
        }
        if (needEnergyParamDerivs)
            for (int i = 0; i < numComputedValues; i++)
                for (int j = 0; j < dValuedParam.size(); j++)
                    compute << "energyParamDeriv"<<j<<" += deriv"<<energyDerivs->getParameterSuffix(i)<<"*dValuedParam_"<<i<<"_"<<j<<"[index];\n";
        map<string, string> replacements;
        replacements["PARAMETER_ARGUMENTS"] = extraArgs.str()+tableArgs.str();
        replacements["COMPUTE_FORCES"] = compute.str();
        replacements["INIT_PARAM_DERIVS"] = initParamDerivs.str();
        replacements["SAVE_PARAM_DERIVS"] = saveParamDerivs.str();
        map<string, string> defines;
        defines["NUM_ATOMS"] = cc.intToString(cc.getNumAtoms());
        defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
        ComputeProgram program = cc.compileProgram(cc.replaceStrings(CommonKernelSources::customGBGradientChainRule, replacements), defines);
        gradientChainRuleKernel = program->createKernel("computeGradientChainRuleTerms");
    }
    {
        // Create the code to calculate chain rule terms as part of the default nonbonded kernel.

        vector<pair<ExpressionTreeNode, string> > globalVariables;
        for (int i = 0; i < force.getNumGlobalParameters(); i++) {
            const string& name = force.getGlobalParameterName(i);
            string value = "globals["+cc.intToString(i)+"]";
            globalVariables.push_back(makeVariable(name, prefix+value));
        }
        vector<pair<ExpressionTreeNode, string> > variables = globalVariables;
        map<string, string> rename;
        ExpressionTreeNode rnode(new Operation::Variable("r"));
        variables.push_back(make_pair(rnode, "r"));
        variables.push_back(make_pair(ExpressionTreeNode(new Operation::Square(), rnode), "r2"));
        variables.push_back(make_pair(ExpressionTreeNode(new Operation::Reciprocal(), rnode), "invR"));
        for (int i = 0; i < force.getNumPerParticleParameters(); i++) {
            const string& name = force.getPerParticleParameterName(i);
            variables.push_back(makeVariable(name+"1", prefix+"params"+params->getParameterSuffix(i, "1")));
            variables.push_back(makeVariable(name+"2", prefix+"params"+params->getParameterSuffix(i, "2")));
            rename[name+"1"] = name+"2";
            rename[name+"2"] = name+"1";
        }
        map<string, Lepton::ParsedExpression> derivExpressions;
        stringstream chainSource;
        Lepton::ParsedExpression dVdR = Lepton::Parser::parse(computedValueExpressions[0], functions).differentiate("r").optimize();
        derivExpressions["real dV0dR1 = "] = dVdR;
        derivExpressions["real dV0dR2 = "] = dVdR.renameVariables(rename);
        chainSource << cc.getExpressionUtilities().createExpressions(derivExpressions, variables, functionList, functionDefinitions, prefix+"temp0_");
        if (needChainForValue[0]) {
            if (useExclusionsForValue)
                chainSource << "if (!isExcluded) {\n";
            chainSource << "tempForce -= dV0dR1*" << prefix << "dEdV" << energyDerivs->getParameterSuffix(0, "1") << ";\n";
            chainSource << "tempForce -= dV0dR2*" << prefix << "dEdV" << energyDerivs->getParameterSuffix(0, "2") << ";\n";
            if (useExclusionsForValue)
                chainSource << "}\n";
        }
        for (int i = 1; i < numComputedValues; i++) {
            if (needChainForValue[i]) {
                chainSource << "tempForce -= dV0dR1*" << prefix << "dEdV" << energyDerivs->getParameterSuffix(i, "1") << ";\n";
                chainSource << "tempForce -= dV0dR2*" << prefix << "dEdV" << energyDerivs->getParameterSuffix(i, "2") << ";\n";
            }
        }
        map<string, string> replacements;
        string chainStr = chainSource.str();
        replacements["COMPUTE_FORCE"] = chainStr;
        string source = cc.replaceStrings(CommonKernelSources::customGBChainRule, replacements);
        vector<ComputeParameterInfo> parameters;
        vector<ComputeParameterInfo> arguments;
        for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
            ComputeParameterInfo& buffer = params->getParameterInfos()[i];
            string paramName = prefix+"params"+cc.intToString(i+1);
            if (chainStr.find(paramName+"1") != chainStr.npos || chainStr.find(paramName+"2") != chainStr.npos)
                parameters.push_back(ComputeParameterInfo(buffer.getArray(), paramName, buffer.getComponentType(), buffer.getNumComponents()));
        }
        for (int i = 0; i < (int) computedValues->getParameterInfos().size(); i++) {
            ComputeParameterInfo& buffer = computedValues->getParameterInfos()[i];
            string paramName = prefix+"values"+cc.intToString(i+1);
            if (chainStr.find(paramName+"1") != chainStr.npos || chainStr.find(paramName+"2") != chainStr.npos)
                parameters.push_back(ComputeParameterInfo(buffer.getArray(), paramName, buffer.getComponentType(), buffer.getNumComponents()));
        }
        for (int i = 0; i < (int) energyDerivChain->getParameterInfos().size(); i++) {
            if (needChainForValue[i]) {
                ComputeParameterInfo& buffer = energyDerivChain->getParameterInfos()[i];
                string paramName = prefix+"dEdV"+cc.intToString(i+1);
                parameters.push_back(ComputeParameterInfo(buffer.getArray(), paramName, buffer.getComponentType(), buffer.getNumComponents()));
            }
        }
        if (globals.isInitialized()) {
            globals.upload(globalParamValues);
            arguments.push_back(ComputeParameterInfo(globals, prefix+"globals", "float", 1));
        }
        nb.addInteraction(useCutoff, usePeriodic, force.getNumExclusions() > 0, cutoff, exclusionList, source, force.getForceGroup());
        for (auto param : parameters)
            nb.addParameter(param);
        for (auto arg : arguments)
            nb.addArgument(arg);
    }
    info = new ForceInfo(force);
    cc.addForce(info);
    if (useLong)
        cc.addAutoclearBuffer(longEnergyDerivs);
    else {
        for (auto& buffer : energyDerivs->getParameterInfos())
            cc.addAutoclearBuffer(buffer.getArray());
    }
}

double CommonCalcCustomGBForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    bool deviceIsCpu = cc.getIsCPU();
    NonbondedUtilities& nb = cc.getNonbondedUtilities();
    int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;

        // These two kernels can't be compiled in initialize(), because the nonbonded utilities object
        // has not yet been initialized then.

        {
            int numExclusionTiles = nb.getExclusionTiles().getSize();
            pairValueDefines["NUM_TILES_WITH_EXCLUSIONS"] = cc.intToString(numExclusionTiles);
            int numContexts = cc.getNumContexts();
            int startExclusionIndex = cc.getContextIndex()*numExclusionTiles/numContexts;
            int endExclusionIndex = (cc.getContextIndex()+1)*numExclusionTiles/numContexts;
            pairValueDefines["FIRST_EXCLUSION_TILE"] = cc.intToString(startExclusionIndex);
            pairValueDefines["LAST_EXCLUSION_TILE"] = cc.intToString(endExclusionIndex);
            pairValueDefines["CUTOFF"] = cc.doubleToString(cutoff);
            ComputeProgram program = cc.compileProgram(pairValueSrc, pairValueDefines);
            pairValueKernel = program->createKernel("computeN2Value");
            pairValueSrc = "";
            pairValueDefines.clear();
        }
        {
            int numExclusionTiles = nb.getExclusionTiles().getSize();
            pairEnergyDefines["NUM_TILES_WITH_EXCLUSIONS"] = cc.intToString(numExclusionTiles);
            int numContexts = cc.getNumContexts();
            int startExclusionIndex = cc.getContextIndex()*numExclusionTiles/numContexts;
            int endExclusionIndex = (cc.getContextIndex()+1)*numExclusionTiles/numContexts;
            pairEnergyDefines["FIRST_EXCLUSION_TILE"] = cc.intToString(startExclusionIndex);
            pairEnergyDefines["LAST_EXCLUSION_TILE"] = cc.intToString(endExclusionIndex);
            pairEnergyDefines["CUTOFF"] = cc.doubleToString(cutoff);
            ComputeProgram program = cc.compileProgram(pairEnergySrc, pairEnergyDefines);
            pairEnergyKernel = program->createKernel("computeN2Energy");
            pairEnergySrc = "";
            pairEnergyDefines.clear();
        }

        // Set arguments for kernels.

        const int tileWidth = cc.getSIMDWidth();
        maxTiles = (nb.getUseCutoff() ? nb.getInteractingTiles().getSize() : 0);
        int numAtomBlocks = cc.getPaddedNumAtoms()/tileWidth;
        bool useLong = cc.getSupports64BitGlobalAtomics();
        pairValueKernel->addArg(cc.getPosq());
        pairValueKernel->addArg(cc.getNonbondedUtilities().getExclusions());
        pairValueKernel->addArg(cc.getNonbondedUtilities().getExclusionTiles());
        pairValueKernel->addArg(valueBuffers);
        if (nb.getUseCutoff()) {
            pairValueKernel->addArg(nb.getInteractingTiles());
            pairValueKernel->addArg(nb.getInteractionCount());
            for (int i = 0; i < 5; i++)
                pairValueKernel->addArg(); // Periodic box size arguments are set when the kernel is executed.
            pairValueKernel->addArg(maxTiles);
            pairValueKernel->addArg(nb.getBlockCenters());
            pairValueKernel->addArg(nb.getBlockBoundingBoxes());
            pairValueKernel->addArg(nb.getInteractingAtoms());
        }
        else
            pairValueKernel->addArg(numAtomBlocks*(numAtomBlocks+1)/2);
        if (globals.isInitialized())
            pairValueKernel->addArg(globals);
        for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
            if (pairValueUsesParam[i]) {
                ComputeParameterInfo& buffer = params->getParameterInfos()[i];
                pairValueKernel->addArg(buffer.getArray());
            }
        }
        for (auto& d : dValue0dParam)
            pairValueKernel->addArg(d);
        for (auto& function : tabulatedFunctions)
            pairValueKernel->addArg(function);
        perParticleValueKernel->addArg(cc.getPosq());
        perParticleValueKernel->addArg(valueBuffers);
        if (!useLong) {
            perParticleValueKernel->addArg(cc.getPaddedNumAtoms());
            perParticleValueKernel->addArg(cc.getForceBuffers().getSize()/cc.getPaddedNumAtoms());
        }
        if (globals.isInitialized())
            perParticleValueKernel->addArg(globals);
        for (auto& buffer : params->getParameterInfos())
            perParticleValueKernel->addArg(buffer.getArray());
        for (auto& buffer : computedValues->getParameterInfos())
            perParticleValueKernel->addArg(buffer.getArray());
        for (int i = 0; i < dValuedParam.size(); i++) {
            perParticleValueKernel->addArg(dValue0dParam[i]);
            for (int j = 0; j < dValuedParam[i]->getParameterInfos().size(); j++)
                perParticleValueKernel->addArg(dValuedParam[i]->getParameterInfos()[j].getArray());
        }
        for (auto& function : tabulatedFunctions)
            perParticleValueKernel->addArg(function);
        pairEnergyKernel->addArg(useLong ? cc.getLongForceBuffer() : cc.getForceBuffers());
        pairEnergyKernel->addArg(cc.getEnergyBuffer());
        pairEnergyKernel->addArg(cc.getPosq());
        pairEnergyKernel->addArg(cc.getNonbondedUtilities().getExclusions());
        pairEnergyKernel->addArg(cc.getNonbondedUtilities().getExclusionTiles());
        pairEnergyKernel->addArg(); // Whether to include energy.
        if (nb.getUseCutoff()) {
            pairEnergyKernel->addArg(nb.getInteractingTiles());
            pairEnergyKernel->addArg(nb.getInteractionCount());
            for (int i = 0; i < 5; i++)
                pairEnergyKernel->addArg(); // Periodic box size arguments are set when the kernel is executed.
            pairEnergyKernel->addArg(maxTiles);
            pairEnergyKernel->addArg(nb.getBlockCenters());
            pairEnergyKernel->addArg(nb.getBlockBoundingBoxes());
            pairEnergyKernel->addArg(nb.getInteractingAtoms());
        }
        else
            pairEnergyKernel->addArg(numAtomBlocks*(numAtomBlocks+1)/2);
        if (globals.isInitialized())
            pairEnergyKernel->addArg(globals);
        for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
            if (pairEnergyUsesParam[i]) {
                ComputeParameterInfo& buffer = params->getParameterInfos()[i];
                pairEnergyKernel->addArg(buffer.getArray());
            }
        }
        for (int i = 0; i < (int) computedValues->getParameterInfos().size(); i++) {
            if (pairEnergyUsesValue[i]) {
                ComputeParameterInfo& buffer = computedValues->getParameterInfos()[i];
                pairEnergyKernel->addArg(buffer.getArray());
            }
        }
        if (useLong)
            pairEnergyKernel->addArg(longEnergyDerivs);
        else
            for (auto& buffer : energyDerivs->getParameterInfos())
                pairEnergyKernel->addArg(buffer.getArray());
        if (needEnergyParamDerivs)
            pairEnergyKernel->addArg(cc.getEnergyParamDerivBuffer());
        for (auto& function : tabulatedFunctions)
            pairEnergyKernel->addArg(function);
        perParticleEnergyKernel->addArg(cc.getEnergyBuffer());
        perParticleEnergyKernel->addArg(cc.getPosq());
        if (cc.getSupports64BitGlobalAtomics())
            perParticleEnergyKernel->addArg(cc.getLongForceBuffer());
        else {
            perParticleEnergyKernel->addArg(cc.getForceBuffers());
            perParticleEnergyKernel->addArg(cc.getPaddedNumAtoms());
            perParticleEnergyKernel->addArg(cc.getForceBuffers().getSize()/cc.getPaddedNumAtoms());
        }
        if (globals.isInitialized())
            perParticleEnergyKernel->addArg(globals);
        for (auto& buffer : params->getParameterInfos())
            perParticleEnergyKernel->addArg(buffer.getArray());
        for (auto& buffer : computedValues->getParameterInfos())
            perParticleEnergyKernel->addArg(buffer.getArray());
        for (auto& buffer : energyDerivs->getParameterInfos())
            perParticleEnergyKernel->addArg(buffer.getArray());
        for (auto& buffer : energyDerivChain->getParameterInfos())
            perParticleEnergyKernel->addArg(buffer.getArray());
        if (useLong)
            perParticleEnergyKernel->addArg(longEnergyDerivs);
        if (needEnergyParamDerivs)
            perParticleEnergyKernel->addArg(cc.getEnergyParamDerivBuffer());
        for (auto& function : tabulatedFunctions)
            perParticleEnergyKernel->addArg(function);
        if (needParameterGradient || needEnergyParamDerivs) {
            gradientChainRuleKernel->addArg(cc.getPosq());
            gradientChainRuleKernel->addArg(useLong ? cc.getLongForceBuffer() : cc.getForceBuffers());
            if (globals.isInitialized())
                gradientChainRuleKernel->addArg(globals);
            for (auto& buffer : params->getParameterInfos())
                gradientChainRuleKernel->addArg(buffer.getArray());
            for (auto& buffer : computedValues->getParameterInfos())
                gradientChainRuleKernel->addArg(buffer.getArray());
            for (auto& buffer : energyDerivs->getParameterInfos())
                gradientChainRuleKernel->addArg(buffer.getArray());
            if (needEnergyParamDerivs) {
                gradientChainRuleKernel->addArg(cc.getEnergyParamDerivBuffer());
                for (auto d : dValuedParam)
                    for (auto& buffer : d->getParameterInfos())
                        gradientChainRuleKernel->addArg(buffer.getArray());
            }
            for (auto& function : tabulatedFunctions)
                gradientChainRuleKernel->addArg(function);
        }
    }
    if (globals.isInitialized()) {
        bool changed = false;
        for (int i = 0; i < (int) globalParamNames.size(); i++) {
            float value = (float) context.getParameter(globalParamNames[i]);
            if (value != globalParamValues[i])
                changed = true;
            globalParamValues[i] = value;
        }
        if (changed)
            globals.upload(globalParamValues);
    }
    pairEnergyKernel->setArg(5, (int) includeEnergy);
    if (nb.getUseCutoff()) {
        setPeriodicBoxArgs(cc, pairValueKernel, 6);
        setPeriodicBoxArgs(cc, pairEnergyKernel, 8);
        if (maxTiles < nb.getInteractingTiles().getSize()) {
            maxTiles = nb.getInteractingTiles().getSize();
            pairValueKernel->setArg(11, maxTiles);
            pairEnergyKernel->setArg(13, maxTiles);
        }
    }
    pairValueKernel->execute(nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize(), nb.getForceThreadBlockSize());
    perParticleValueKernel->execute(cc.getPaddedNumAtoms());
    pairEnergyKernel->execute(nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize(), nb.getForceThreadBlockSize());
    perParticleEnergyKernel->execute(cc.getPaddedNumAtoms());
    if (needParameterGradient || needEnergyParamDerivs)
        gradientChainRuleKernel->execute(cc.getPaddedNumAtoms());
    return 0.0;
}

void CommonCalcCustomGBForceKernel::copyParametersToContext(ContextImpl& context, const CustomGBForce& force) {
    cc.setAsCurrent();
    int numParticles = force.getNumParticles();
    if (numParticles != cc.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Record the per-particle parameters.

    vector<vector<float> > paramVector(cc.getPaddedNumAtoms(), vector<float>(force.getNumPerParticleParameters(), 0));
    vector<double> parameters;
    for (int i = 0; i < numParticles; i++) {
        force.getParticleParameters(i, parameters);
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);

    // Mark that the current reordering may be invalid.

    cc.invalidateMolecules(info);
}

class CommonCalcCustomHbondForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const CustomHbondForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        return true;
    }
    int getNumParticleGroups() {
        return force.getNumDonors()+force.getNumAcceptors()+force.getNumExclusions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int p1, p2, p3;
        vector<double> parameters;
        if (index < force.getNumDonors()) {
            force.getDonorParameters(index, p1, p2, p3, parameters);
            particles.clear();
            particles.push_back(p1);
            if (p2 > -1)
                particles.push_back(p2);
            if (p3 > -1)
                particles.push_back(p3);
            return;
        }
        index -= force.getNumDonors();
        if (index < force.getNumAcceptors()) {
            force.getAcceptorParameters(index, p1, p2, p3, parameters);
            particles.clear();
            particles.push_back(p1);
            if (p2 > -1)
                particles.push_back(p2);
            if (p3 > -1)
                particles.push_back(p3);
            return;
        }
        index -= force.getNumAcceptors();
        int donor, acceptor;
        force.getExclusionParticles(index, donor, acceptor);
        particles.clear();
        force.getDonorParameters(donor, p1, p2, p3, parameters);
        particles.push_back(p1);
        if (p2 > -1)
            particles.push_back(p2);
        if (p3 > -1)
            particles.push_back(p3);
        force.getAcceptorParameters(acceptor, p1, p2, p3, parameters);
        particles.push_back(p1);
        if (p2 > -1)
            particles.push_back(p2);
        if (p3 > -1)
            particles.push_back(p3);
    }
    bool areGroupsIdentical(int group1, int group2) {
        int p1, p2, p3;
        vector<double> params1, params2;
        if (group1 < force.getNumDonors() && group2 < force.getNumDonors()) {
            force.getDonorParameters(group1, p1, p2, p3, params1);
            force.getDonorParameters(group2, p1, p2, p3, params2);
            return (params1 == params2 && params1 == params2);
        }
        if (group1 < force.getNumDonors() || group2 < force.getNumDonors())
            return false;
        group1 -= force.getNumDonors();
        group2 -= force.getNumDonors();
        if (group1 < force.getNumAcceptors() && group2 < force.getNumAcceptors()) {
            force.getAcceptorParameters(group1, p1, p2, p3, params1);
            force.getAcceptorParameters(group2, p1, p2, p3, params2);
            return (params1 == params2 && params1 == params2);
        }
        if (group1 < force.getNumAcceptors() || group2 < force.getNumAcceptors())
            return false;
        return true;
    }
private:
    const CustomHbondForce& force;
};

CommonCalcCustomHbondForceKernel::~CommonCalcCustomHbondForceKernel() {
    cc.setAsCurrent();
    if (donorParams != NULL)
        delete donorParams;
    if (acceptorParams != NULL)
        delete acceptorParams;
}

static void addDonorAndAcceptorCode(stringstream& computeDonor, stringstream& computeAcceptor, const string& value) {
    computeDonor << value;
    computeAcceptor << value;
}

static void applyDonorAndAcceptorForces(stringstream& applyToDonor, stringstream& applyToAcceptor, int atom, const string& value, bool trim=true) {
    string forceNames[] = {"f1", "f2", "f3"};
    string toAdd = (trim ? "trimTo3("+value+")" : value);
    if (atom < 3)
        applyToAcceptor << forceNames[atom]<<" += "<<toAdd<<";\n";
    else
        applyToDonor << forceNames[atom-3]<<" += "<<toAdd<<";\n";
}

void CommonCalcCustomHbondForceKernel::initialize(const System& system, const CustomHbondForce& force) {
    // Record the lists of donors and acceptors, and the parameters for each one.

    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumDonors()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumDonors()/numContexts;
    numDonors = endIndex-startIndex;
    numAcceptors = force.getNumAcceptors();
    if (numDonors == 0 || numAcceptors == 0)
        return;
    int numParticles = system.getNumParticles();
    donors.initialize<mm_int4>(cc, numDonors, "customHbondDonors");
    acceptors.initialize<mm_int4>(cc, numAcceptors, "customHbondAcceptors");
    donorParams = new ComputeParameterSet(cc, force.getNumPerDonorParameters(), numDonors, "customHbondDonorParameters");
    acceptorParams = new ComputeParameterSet(cc, force.getNumPerAcceptorParameters(), numAcceptors, "customHbondAcceptorParameters");
    if (force.getNumGlobalParameters() > 0)
        globals.initialize<float>(cc, force.getNumGlobalParameters(), "customHbondGlobals");
    vector<vector<float> > donorParamVector(numDonors);
    vector<mm_int4> donorVector(numDonors);
    for (int i = 0; i < numDonors; i++) {
        vector<double> parameters;
        force.getDonorParameters(startIndex+i, donorVector[i].x, donorVector[i].y, donorVector[i].z, parameters);
        donorParamVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            donorParamVector[i][j] = (float) parameters[j];
    }
    donors.upload(donorVector);
    donorParams->setParameterValues(donorParamVector);
    vector<vector<float> > acceptorParamVector(numAcceptors);
    vector<mm_int4> acceptorVector(numAcceptors);
    for (int i = 0; i < numAcceptors; i++) {
        vector<double> parameters;
        force.getAcceptorParameters(i, acceptorVector[i].x, acceptorVector[i].y, acceptorVector[i].z, parameters);
        acceptorParamVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            acceptorParamVector[i][j] = (float) parameters[j];
    }
    acceptors.upload(acceptorVector);
    acceptorParams->setParameterValues(acceptorParamVector);

    // Select an output buffer index for each donor and acceptor.

    if (!cc.getSupports64BitGlobalAtomics()) {
        donorBufferIndices.initialize<mm_int4>(cc, numDonors, "customHbondDonorBuffers");
        acceptorBufferIndices.initialize<mm_int4>(cc, numAcceptors, "customHbondAcceptorBuffers");
        vector<mm_int4> donorBufferVector(numDonors);
        vector<mm_int4> acceptorBufferVector(numAcceptors);
        vector<int> donorBufferCounter(numParticles, 0);
        for (int i = 0; i < numDonors; i++)
            donorBufferVector[i] = mm_int4(donorVector[i].x > -1 ? donorBufferCounter[donorVector[i].x]++ : 0,
                                           donorVector[i].y > -1 ? donorBufferCounter[donorVector[i].y]++ : 0,
                                           donorVector[i].z > -1 ? donorBufferCounter[donorVector[i].z]++ : 0, 0);
        vector<int> acceptorBufferCounter(numParticles, 0);
        for (int i = 0; i < numAcceptors; i++)
            acceptorBufferVector[i] = mm_int4(acceptorVector[i].x > -1 ? acceptorBufferCounter[acceptorVector[i].x]++ : 0,
                                           acceptorVector[i].y > -1 ? acceptorBufferCounter[acceptorVector[i].y]++ : 0,
                                           acceptorVector[i].z > -1 ? acceptorBufferCounter[acceptorVector[i].z]++ : 0, 0);
        donorBufferIndices.upload(donorBufferVector);
        acceptorBufferIndices.upload(acceptorBufferVector);
        int maxBuffers = 1;
        for (int i : donorBufferCounter)
            maxBuffers = max(maxBuffers, i);
        for (int i : acceptorBufferCounter)
            maxBuffers = max(maxBuffers, i);
        cc.requestForceBuffers(maxBuffers);
    }
    info = new ForceInfo(force);
    cc.addForce(info);

    // Record exclusions.

    vector<mm_int4> donorExclusionVector(numDonors, mm_int4(-1, -1, -1, -1));
    vector<mm_int4> acceptorExclusionVector(numAcceptors, mm_int4(-1, -1, -1, -1));
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int donor, acceptor;
        force.getExclusionParticles(i, donor, acceptor);
        if (donor < startIndex || donor >= endIndex)
            continue;
        donor -= startIndex;
        if (donorExclusionVector[donor].x == -1)
            donorExclusionVector[donor].x = acceptor;
        else if (donorExclusionVector[donor].y == -1)
            donorExclusionVector[donor].y = acceptor;
        else if (donorExclusionVector[donor].z == -1)
            donorExclusionVector[donor].z = acceptor;
        else if (donorExclusionVector[donor].w == -1)
            donorExclusionVector[donor].w = acceptor;
        else
            throw OpenMMException("CustomHbondForce: this platform does not support more than four exclusions per donor");
        if (acceptorExclusionVector[acceptor].x == -1)
            acceptorExclusionVector[acceptor].x = donor;
        else if (acceptorExclusionVector[acceptor].y == -1)
            acceptorExclusionVector[acceptor].y = donor;
        else if (acceptorExclusionVector[acceptor].z == -1)
            acceptorExclusionVector[acceptor].z = donor;
        else if (acceptorExclusionVector[acceptor].w == -1)
            acceptorExclusionVector[acceptor].w = donor;
        else
            throw OpenMMException("CustomHbondForce: this platform does not support more than four exclusions per acceptor");
    }
    donorExclusions.initialize<mm_int4>(cc, numDonors, "customHbondDonorExclusions");
    acceptorExclusions.initialize<mm_int4>(cc, numAcceptors, "customHbondAcceptorExclusions");
    donorExclusions.upload(donorExclusionVector);
    acceptorExclusions.upload(acceptorExclusionVector);

    // Record the tabulated functions.

    map<string, Lepton::CustomFunction*> functions;
    vector<pair<string, string> > functionDefinitions;
    vector<const TabulatedFunction*> functionList;
    stringstream tableArgs;
    tabulatedFunctions.resize(force.getNumTabulatedFunctions());
    for (int i = 0; i < force.getNumTabulatedFunctions(); i++) {
        functionList.push_back(&force.getTabulatedFunction(i));
        string name = force.getTabulatedFunctionName(i);
        string arrayName = "table"+cc.intToString(i);
        functionDefinitions.push_back(make_pair(name, arrayName));
        functions[name] = cc.getExpressionUtilities().getFunctionPlaceholder(force.getTabulatedFunction(i));
        int width;
        vector<float> f = cc.getExpressionUtilities().computeFunctionCoefficients(force.getTabulatedFunction(i), width);
        tabulatedFunctions[i].initialize<float>(cc, f.size(), "TabulatedFunction");
        tabulatedFunctions[i].upload(f);
        tableArgs << ", GLOBAL const float";
        if (width > 1)
            tableArgs << width;
        tableArgs << "* RESTRICT " << arrayName;
    }

    // Record information about parameters.

    globalParamNames.resize(force.getNumGlobalParameters());
    globalParamValues.resize(force.getNumGlobalParameters());
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        globalParamNames[i] = force.getGlobalParameterName(i);
        globalParamValues[i] = (float) force.getGlobalParameterDefaultValue(i);
    }
    if (globals.isInitialized())
        globals.upload(globalParamValues);
    map<string, string> variables;
    for (int i = 0; i < force.getNumPerDonorParameters(); i++) {
        const string& name = force.getPerDonorParameterName(i);
        variables[name] = "donorParams"+donorParams->getParameterSuffix(i);
    }
    for (int i = 0; i < force.getNumPerAcceptorParameters(); i++) {
        const string& name = force.getPerAcceptorParameterName(i);
        variables[name] = "acceptorParams"+acceptorParams->getParameterSuffix(i);
    }
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        const string& name = force.getGlobalParameterName(i);
        variables[name] = "globals["+cc.intToString(i)+"]";
    }

    // Now to generate the kernel.  First, it needs to calculate all distances, angles,
    // and dihedrals the expression depends on.

    map<string, vector<int> > distances;
    map<string, vector<int> > angles;
    map<string, vector<int> > dihedrals;
    Lepton::ParsedExpression energyExpression = CustomHbondForceImpl::prepareExpression(force, functions, distances, angles, dihedrals);
    map<string, Lepton::ParsedExpression> forceExpressions;
    set<string> computedDeltas;
    computedDeltas.insert("D1A1");
    string atomNames[] = {"A1", "A2", "A3", "D1", "D2", "D3"};
    string atomNamesLower[] = {"a1", "a2", "a3", "d1", "d2", "d3"};
    stringstream computeDonor, computeAcceptor, extraArgs;
    int index = 0;
    for (auto& distance : distances) {
        const vector<int>& atoms = distance.second;
        string deltaName = atomNames[atoms[0]]+atomNames[atoms[1]];
        if (computedDeltas.count(deltaName) == 0) {
            addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real4 delta"+deltaName+" = delta("+atomNamesLower[atoms[0]]+", "+atomNamesLower[atoms[1]]+", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n");
            computedDeltas.insert(deltaName);
        }
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real r_"+deltaName+" = SQRT(delta"+deltaName+".w);\n");
        variables[distance.first] = "r_"+deltaName;
        forceExpressions["real dEdDistance"+cc.intToString(index)+" = "] = energyExpression.differentiate(distance.first).optimize();
        index++;
    }
    index = 0;
    for (auto& angle : angles) {
        const vector<int>& atoms = angle.second;
        string deltaName1 = atomNames[atoms[1]]+atomNames[atoms[0]];
        string deltaName2 = atomNames[atoms[1]]+atomNames[atoms[2]];
        string angleName = "angle_"+atomNames[atoms[0]]+atomNames[atoms[1]]+atomNames[atoms[2]];
        if (computedDeltas.count(deltaName1) == 0) {
            addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real4 delta"+deltaName1+" = delta("+atomNamesLower[atoms[1]]+", "+atomNamesLower[atoms[0]]+", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n");
            computedDeltas.insert(deltaName1);
        }
        if (computedDeltas.count(deltaName2) == 0) {
            addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real4 delta"+deltaName2+" = delta("+atomNamesLower[atoms[1]]+", "+atomNamesLower[atoms[2]]+", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n");
            computedDeltas.insert(deltaName2);
        }
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real "+angleName+" = computeAngle(delta"+deltaName1+", delta"+deltaName2+");\n");
        variables[angle.first] = angleName;
        forceExpressions["real dEdAngle"+cc.intToString(index)+" = "] = energyExpression.differentiate(angle.first).optimize();
        index++;
    }
    index = 0;
    for (auto& dihedral : dihedrals) {
        const vector<int>& atoms = dihedral.second;
        string deltaName1 = atomNames[atoms[0]]+atomNames[atoms[1]];
        string deltaName2 = atomNames[atoms[2]]+atomNames[atoms[1]];
        string deltaName3 = atomNames[atoms[2]]+atomNames[atoms[3]];
        string crossName1 = "cross_"+deltaName1+"_"+deltaName2;
        string crossName2 = "cross_"+deltaName2+"_"+deltaName3;
        string dihedralName = "dihedral_"+atomNames[atoms[0]]+atomNames[atoms[1]]+atomNames[atoms[2]]+atomNames[atoms[3]];
        if (computedDeltas.count(deltaName1) == 0) {
            addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real4 delta"+deltaName1+" = delta("+atomNamesLower[atoms[0]]+", "+atomNamesLower[atoms[1]]+", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n");
            computedDeltas.insert(deltaName1);
        }
        if (computedDeltas.count(deltaName2) == 0) {
            addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real4 delta"+deltaName2+" = delta("+atomNamesLower[atoms[2]]+", "+atomNamesLower[atoms[1]]+", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n");
            computedDeltas.insert(deltaName2);
        }
        if (computedDeltas.count(deltaName3) == 0) {
            addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real4 delta"+deltaName3+" = delta("+atomNamesLower[atoms[2]]+", "+atomNamesLower[atoms[3]]+", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n");
            computedDeltas.insert(deltaName3);
        }
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real4 "+crossName1+" = computeCross(delta"+deltaName1+", delta"+deltaName2+");\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real4 "+crossName2+" = computeCross(delta"+deltaName2+", delta"+deltaName3+");\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real "+dihedralName+" = computeAngle("+crossName1+", "+crossName2+");\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, dihedralName+" *= (delta"+deltaName1+".x*"+crossName2+".x + delta"+deltaName1+".y*"+crossName2+".y + delta"+deltaName1+".z*"+crossName2+".z < 0 ? -1 : 1);\n");
        variables[dihedral.first] = dihedralName;
        forceExpressions["real dEdDihedral"+cc.intToString(index)+" = "] = energyExpression.differentiate(dihedral.first).optimize();
        index++;
    }

    // Next it needs to load parameters from global memory.

    if (force.getNumGlobalParameters() > 0)
        extraArgs << ", GLOBAL const float* RESTRICT globals";
    for (int i = 0; i < (int) donorParams->getParameterInfos().size(); i++) {
        ComputeParameterInfo& parameter = donorParams->getParameterInfos()[i];
        extraArgs << ", GLOBAL const "+parameter.getType()+"* RESTRICT donor"+parameter.getName();
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, parameter.getType()+" donorParams"+cc.intToString(i+1)+" = donor"+parameter.getName()+"[donorIndex];\n");
    }
    for (int i = 0; i < (int) acceptorParams->getParameterInfos().size(); i++) {
        ComputeParameterInfo& parameter = acceptorParams->getParameterInfos()[i];
        extraArgs << ", GLOBAL const "+parameter.getType()+"* RESTRICT acceptor"+parameter.getName();
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, parameter.getType()+" acceptorParams"+cc.intToString(i+1)+" = acceptor"+parameter.getName()+"[acceptorIndex];\n");
    }

    // Now evaluate the expressions.

    computeAcceptor << cc.getExpressionUtilities().createExpressions(forceExpressions, variables, functionList, functionDefinitions, "temp");
    forceExpressions["energy += "] = energyExpression;
    computeDonor << cc.getExpressionUtilities().createExpressions(forceExpressions, variables, functionList, functionDefinitions, "temp");

    // Finally, apply forces to atoms.

    index = 0;
    for (auto& distance : distances) {
        const vector<int>& atoms = distance.second;
        string deltaName = atomNames[atoms[0]]+atomNames[atoms[1]];
        string value = "(dEdDistance"+cc.intToString(index)+"/r_"+deltaName+")*delta"+deltaName;
        applyDonorAndAcceptorForces(computeDonor, computeAcceptor, atoms[0], "-"+value);
        applyDonorAndAcceptorForces(computeDonor, computeAcceptor, atoms[1], value);
        index++;
    }
    index = 0;
    for (auto& angle : angles) {
        const vector<int>& atoms = angle.second;
        string deltaName1 = atomNames[atoms[1]]+atomNames[atoms[0]];
        string deltaName2 = atomNames[atoms[1]]+atomNames[atoms[2]];
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "{\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real3 crossProd = trimTo3(cross(delta"+deltaName2+", delta"+deltaName1+"));\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real lengthCross = max(SQRT(dot(crossProd,crossProd)), (real) 1e-6f);\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real3 deltaCross0 = -cross(trimTo3(delta"+deltaName1+"), crossProd)*dEdAngle"+cc.intToString(index)+"/(delta"+deltaName1+".w*lengthCross);\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real3 deltaCross2 = cross(trimTo3(delta"+deltaName2+"), crossProd)*dEdAngle"+cc.intToString(index)+"/(delta"+deltaName2+".w*lengthCross);\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real3 deltaCross1 = -(deltaCross0+deltaCross2);\n");
        applyDonorAndAcceptorForces(computeDonor, computeAcceptor, atoms[0], "deltaCross0", false);
        applyDonorAndAcceptorForces(computeDonor, computeAcceptor, atoms[1], "deltaCross1", false);
        applyDonorAndAcceptorForces(computeDonor, computeAcceptor, atoms[2], "deltaCross2", false);
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "}\n");
        index++;
    }
    index = 0;
    for (auto& dihedral : dihedrals) {
        const vector<int>& atoms = dihedral.second;
        string deltaName1 = atomNames[atoms[0]]+atomNames[atoms[1]];
        string deltaName2 = atomNames[atoms[2]]+atomNames[atoms[1]];
        string deltaName3 = atomNames[atoms[2]]+atomNames[atoms[3]];
        string crossName1 = "cross_"+deltaName1+"_"+deltaName2;
        string crossName2 = "cross_"+deltaName2+"_"+deltaName3;
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "{\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real r = SQRT(delta"+deltaName2+".w);\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real4 ff;\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "ff.x = (-dEdDihedral"+cc.intToString(index)+"*r)/"+crossName1+".w;\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "ff.y = (delta"+deltaName1+".x*delta"+deltaName2+".x + delta"+deltaName1+".y*delta"+deltaName2+".y + delta"+deltaName1+".z*delta"+deltaName2+".z)/delta"+deltaName2+".w;\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "ff.z = (delta"+deltaName3+".x*delta"+deltaName2+".x + delta"+deltaName3+".y*delta"+deltaName2+".y + delta"+deltaName3+".z*delta"+deltaName2+".z)/delta"+deltaName2+".w;\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "ff.w = (dEdDihedral"+cc.intToString(index)+"*r)/"+crossName2+".w;\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real4 internalF0 = ff.x*"+crossName1+";\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real4 internalF3 = ff.w*"+crossName2+";\n");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "real4 s = ff.y*internalF0 - ff.z*internalF3;\n");
        applyDonorAndAcceptorForces(computeDonor, computeAcceptor, atoms[0], "internalF0");
        applyDonorAndAcceptorForces(computeDonor, computeAcceptor, atoms[1], "s-internalF0");
        applyDonorAndAcceptorForces(computeDonor, computeAcceptor, atoms[2], "-s-internalF3");
        applyDonorAndAcceptorForces(computeDonor, computeAcceptor, atoms[3], "internalF3");
        addDonorAndAcceptorCode(computeDonor, computeAcceptor, "}\n");
        index++;
    }

    // Generate the kernels.

    map<string, string> replacements;
    replacements["COMPUTE_DONOR_FORCE"] = computeDonor.str();
    replacements["COMPUTE_ACCEPTOR_FORCE"] = computeAcceptor.str();
    replacements["PARAMETER_ARGUMENTS"] = extraArgs.str()+tableArgs.str();
    map<string, string> defines;
    defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
    defines["NUM_DONORS"] = cc.intToString(numDonors);
    defines["NUM_ACCEPTORS"] = cc.intToString(numAcceptors);
    defines["M_PI"] = cc.doubleToString(M_PI);
    defines["THREAD_BLOCK_SIZE"] = "64";
    if (force.getNonbondedMethod() != CustomHbondForce::NoCutoff) {
        defines["USE_CUTOFF"] = "1";
        defines["CUTOFF_SQUARED"] = cc.doubleToString(force.getCutoffDistance()*force.getCutoffDistance());
    }
    if (force.getNonbondedMethod() != CustomHbondForce::NoCutoff && force.getNonbondedMethod() != CustomHbondForce::CutoffNonPeriodic)
        defines["USE_PERIODIC"] = "1";
    if (force.getNumExclusions() > 0)
        defines["USE_EXCLUSIONS"] = "1";
    ComputeProgram program = cc.compileProgram(cc.replaceStrings(CommonKernelSources::customHbondForce, replacements), defines);
    donorKernel = program->createKernel("computeDonorForces");
    acceptorKernel = program->createKernel("computeAcceptorForces");
}

double CommonCalcCustomHbondForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (numDonors == 0 || numAcceptors == 0)
        return 0.0;
    if (globals.isInitialized()) {
        bool changed = false;
        for (int i = 0; i < (int) globalParamNames.size(); i++) {
            float value = (float) context.getParameter(globalParamNames[i]);
            if (value != globalParamValues[i])
                changed = true;
            globalParamValues[i] = value;
        }
        if (changed)
            globals.upload(globalParamValues);
    }
    if (!hasInitializedKernel) {
        hasInitializedKernel = true;
        if (cc.getSupports64BitGlobalAtomics())
            donorKernel->addArg(cc.getLongForceBuffer());
        else {
            donorKernel->addArg(cc.getForceBuffers());
            donorKernel->addArg(donorBufferIndices);
        }
        donorKernel->addArg(cc.getEnergyBuffer());
        donorKernel->addArg(cc.getPosq());
        donorKernel->addArg(donorExclusions);
        donorKernel->addArg(donors);
        donorKernel->addArg(acceptors);
        for (int i = 0; i < 5; i++)
            donorKernel->addArg(); // Periodic box size arguments are set when the kernel is executed.
        if (globals.isInitialized())
            donorKernel->addArg(globals);
        for (auto& parameter : donorParams->getParameterInfos())
            donorKernel->addArg(parameter.getArray());
        for (auto& parameter : acceptorParams->getParameterInfos())
            donorKernel->addArg(parameter.getArray());
        for (auto& function : tabulatedFunctions)
            donorKernel->addArg(function);
        if (cc.getSupports64BitGlobalAtomics())
            acceptorKernel->addArg(cc.getLongForceBuffer());
        else {
            acceptorKernel->addArg(cc.getForceBuffers());
            acceptorKernel->addArg(acceptorBufferIndices);
        }
        acceptorKernel->addArg(cc.getEnergyBuffer());
        acceptorKernel->addArg(cc.getPosq());
        acceptorKernel->addArg(acceptorExclusions);
        acceptorKernel->addArg(donors);
        acceptorKernel->addArg(acceptors);
        for (int i = 0; i < 5; i++)
            acceptorKernel->addArg(); // Periodic box size arguments are set when the kernel is executed.
        if (globals.isInitialized())
            acceptorKernel->addArg(globals);
        for (auto& parameter : donorParams->getParameterInfos())
            acceptorKernel->addArg(parameter.getArray());
        for (auto& parameter : acceptorParams->getParameterInfos())
            acceptorKernel->addArg(parameter.getArray());
        for (auto& function : tabulatedFunctions)
            acceptorKernel->addArg(function);
    }
    setPeriodicBoxArgs(cc, donorKernel, cc.getSupports64BitGlobalAtomics() ? 6 : 7);
    donorKernel->execute(max(numDonors, numAcceptors), 2 * cc.getSIMDWidth());
    setPeriodicBoxArgs(cc, acceptorKernel, cc.getSupports64BitGlobalAtomics() ? 6 : 7);
    acceptorKernel->execute(max(numDonors, numAcceptors), 2 * cc.getSIMDWidth());
    return 0.0;
}

void CommonCalcCustomHbondForceKernel::copyParametersToContext(ContextImpl& context, const CustomHbondForce& force) {
    cc.setAsCurrent();
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*force.getNumDonors()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*force.getNumDonors()/numContexts;
    if (numDonors != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of donors has changed");
    if (numAcceptors != force.getNumAcceptors())
        throw OpenMMException("updateParametersInContext: The number of acceptors has changed");

    // Record the per-donor parameters.

    if (numDonors > 0) {
        vector<vector<float> > donorParamVector(numDonors);
        vector<double> parameters;
        for (int i = 0; i < numDonors; i++) {
            int d1, d2, d3;
            force.getDonorParameters(startIndex+i, d1, d2, d3, parameters);
            donorParamVector[i].resize(parameters.size());
            for (int j = 0; j < (int) parameters.size(); j++)
                donorParamVector[i][j] = (float) parameters[j];
        }
        donorParams->setParameterValues(donorParamVector);
    }

    // Record the per-acceptor parameters.

    if (numAcceptors > 0) {
        vector<vector<float> > acceptorParamVector(numAcceptors);
        vector<double> parameters;
        for (int i = 0; i < numAcceptors; i++) {
            int a1, a2, a3;
            force.getAcceptorParameters(i, a1, a2, a3, parameters);
            acceptorParamVector[i].resize(parameters.size());
            for (int j = 0; j < (int) parameters.size(); j++)
                acceptorParamVector[i][j] = (float) parameters[j];
        }
        acceptorParams->setParameterValues(acceptorParamVector);
    }

    // Mark that the current reordering may be invalid.

    cc.invalidateMolecules(info);
}

class CommonCalcCustomManyParticleForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const CustomManyParticleForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        vector<double> params1, params2;
        int type1, type2;
        force.getParticleParameters(particle1, params1, type1);
        force.getParticleParameters(particle2, params2, type2);
        if (type1 != type2)
            return false;
        for (int i = 0; i < (int) params1.size(); i++)
            if (params1[i] != params2[i])
                return false;
        return true;
    }
    int getNumParticleGroups() {
        return force.getNumExclusions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        force.getExclusionParticles(index, particle1, particle2);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        return true;
    }
private:
    const CustomManyParticleForce& force;
};

CommonCalcCustomManyParticleForceKernel::~CommonCalcCustomManyParticleForceKernel() {
    cc.setAsCurrent();
    if (params != NULL)
        delete params;
}

void CommonCalcCustomManyParticleForceKernel::initialize(const System& system, const CustomManyParticleForce& force) {
    cc.setAsCurrent();
    const int tileWidth = cc.getSIMDWidth();
    if (!cc.getSupports64BitGlobalAtomics())
        throw OpenMMException("CustomManyParticleForce requires a device that supports 64 bit atomic operations");
    int numParticles = force.getNumParticles();
    int particlesPerSet = force.getNumParticlesPerSet();
    bool centralParticleMode = (force.getPermutationMode() == CustomManyParticleForce::UniqueCentralParticle);
    nonbondedMethod = CalcCustomManyParticleForceKernel::NonbondedMethod(force.getNonbondedMethod());
    forceWorkgroupSize = tileWidth * 4;
    findNeighborsWorkgroupSize = (cc.getSIMDWidth() >= tileWidth ? forceWorkgroupSize : tileWidth);
    computeNeighborsWorkgroupSize = 256;

    // Record parameter values.

    params = new ComputeParameterSet(cc, force.getNumPerParticleParameters(), numParticles, "customManyParticleParameters");
    vector<vector<float> > paramVector(numParticles);
    for (int i = 0; i < numParticles; i++) {
        vector<double> parameters;
        int type;
        force.getParticleParameters(i, parameters, type);
        paramVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);
    info = new ForceInfo(force);
    cc.addForce(info);

    // Record the tabulated functions.

    map<string, Lepton::CustomFunction*> functions;
    vector<pair<string, string> > functionDefinitions;
    vector<const TabulatedFunction*> functionList;
    stringstream tableArgs;
    tabulatedFunctions.resize(force.getNumTabulatedFunctions());
    for (int i = 0; i < force.getNumTabulatedFunctions(); i++) {
        functionList.push_back(&force.getTabulatedFunction(i));
        string name = force.getTabulatedFunctionName(i);
        string arrayName = "table"+cc.intToString(i);
        functionDefinitions.push_back(make_pair(name, arrayName));
        functions[name] = cc.getExpressionUtilities().getFunctionPlaceholder(force.getTabulatedFunction(i));
        int width;
        vector<float> f = cc.getExpressionUtilities().computeFunctionCoefficients(force.getTabulatedFunction(i), width);
        tabulatedFunctions[i].initialize<float>(cc, f.size(), "TabulatedFunction");
        tabulatedFunctions[i].upload(f);
        tableArgs << ", GLOBAL const float";
        if (width > 1)
            tableArgs << width;
        tableArgs << "* RESTRICT " << arrayName;
    }

    // Record information about parameters.

    globalParamNames.resize(force.getNumGlobalParameters());
    globalParamValues.resize(force.getNumGlobalParameters());
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        globalParamNames[i] = force.getGlobalParameterName(i);
        globalParamValues[i] = (float) force.getGlobalParameterDefaultValue(i);
    }
    vector<pair<ExpressionTreeNode, string> > variables;
    for (int i = 0; i < particlesPerSet; i++) {
        string index = cc.intToString(i+1);
        variables.push_back(makeVariable("x"+index, "pos"+index+".x"));
        variables.push_back(makeVariable("y"+index, "pos"+index+".y"));
        variables.push_back(makeVariable("z"+index, "pos"+index+".z"));
    }
    for (int i = 0; i < force.getNumPerParticleParameters(); i++) {
        const string& name = force.getPerParticleParameterName(i);
        for (int j = 0; j < particlesPerSet; j++) {
            string index = cc.intToString(j+1);
            variables.push_back(makeVariable(name+index, "params"+params->getParameterSuffix(i, index)));
        }
    }
    if (force.getNumGlobalParameters() > 0) {
        globals.initialize<float>(cc, force.getNumGlobalParameters(), "customManyParticleGlobals");
        globals.upload(globalParamValues);
        for (int i = 0; i < force.getNumGlobalParameters(); i++) {
            const string& name = force.getGlobalParameterName(i);
            string value = "globals["+cc.intToString(i)+"]";
            variables.push_back(makeVariable(name, value));
        }
    }

    // Build data structures for type filters.

    vector<int> particleTypesVec;
    vector<int> orderIndexVec;
    vector<std::vector<int> > particleOrderVec;
    int numTypes;
    CustomManyParticleForceImpl::buildFilterArrays(force, numTypes, particleTypesVec, orderIndexVec, particleOrderVec);
    bool hasTypeFilters = (particleOrderVec.size() > 1);
    if (hasTypeFilters) {
        particleTypes.initialize<int>(cc, particleTypesVec.size(), "customManyParticleTypes");
        orderIndex.initialize<int>(cc, orderIndexVec.size(), "customManyParticleOrderIndex");
        particleOrder.initialize<int>(cc, particleOrderVec.size()*particlesPerSet, "customManyParticleOrder");
        particleTypes.upload(particleTypesVec);
        orderIndex.upload(orderIndexVec);
        vector<int> flattenedOrder(particleOrder.getSize());
        for (int i = 0; i < (int) particleOrderVec.size(); i++)
            for (int j = 0; j < particlesPerSet; j++)
                flattenedOrder[i*particlesPerSet+j] = particleOrderVec[i][j];
        particleOrder.upload(flattenedOrder);
    }

    // Build data structures for exclusions.

    if (force.getNumExclusions() > 0) {
        vector<vector<int> > particleExclusions(numParticles);
        for (int i = 0; i < force.getNumExclusions(); i++) {
            int p1, p2;
            force.getExclusionParticles(i, p1, p2);
            particleExclusions[p1].push_back(p2);
            particleExclusions[p2].push_back(p1);
        }
        vector<int> exclusionsVec;
        vector<int> exclusionStartIndexVec(numParticles+1);
        exclusionStartIndexVec[0] = 0;
        for (int i = 0; i < numParticles; i++) {
            sort(particleExclusions[i].begin(), particleExclusions[i].end());
            exclusionsVec.insert(exclusionsVec.end(), particleExclusions[i].begin(), particleExclusions[i].end());
            exclusionStartIndexVec[i+1] = exclusionsVec.size();
        }
        exclusions.initialize<int>(cc, exclusionsVec.size(), "customManyParticleExclusions");
        exclusionStartIndex.initialize<int>(cc, exclusionStartIndexVec.size(), "customManyParticleExclusionStart");
        exclusions.upload(exclusionsVec);
        exclusionStartIndex.upload(exclusionStartIndexVec);
    }

    // Build data structures for the neighbor list.

    int numAtomBlocks = cc.getPaddedNumAtoms()/tileWidth;
    if (nonbondedMethod != NoCutoff) {
        int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
        blockCenter.initialize(cc, numAtomBlocks, 4*elementSize, "blockCenter");
        blockBoundingBox.initialize(cc, numAtomBlocks, 4*elementSize, "blockBoundingBox");
        numNeighborPairs.initialize<int>(cc, 1, "customManyParticleNumNeighborPairs");
        neighborStartIndex.initialize<int>(cc, numParticles+1, "customManyParticleNeighborStartIndex");
        numNeighborsForAtom.initialize<int>(cc, numParticles, "customManyParticleNumNeighborsForAtom");

        // Select a size for the array that holds the neighbor list.  We have to make a fairly
        // arbitrary guess, but if this turns out to be too small we'll increase it later.

        maxNeighborPairs = 150*numParticles;
        neighborPairs.initialize<mm_int2>(cc, maxNeighborPairs, "customManyParticleNeighborPairs");
        neighbors.initialize<int>(cc, maxNeighborPairs, "customManyParticleNeighbors");
    }

    // Now to generate the kernel.  First, it needs to calculate all distances, angles,
    // and dihedrals the expression depends on.

    map<string, vector<int> > distances;
    map<string, vector<int> > angles;
    map<string, vector<int> > dihedrals;
    Lepton::ParsedExpression energyExpression = CustomManyParticleForceImpl::prepareExpression(force, functions, distances, angles, dihedrals);
    map<string, Lepton::ParsedExpression> forceExpressions;
    set<string> computedDeltas;
    vector<string> atomNames, posNames;
    for (int i = 0; i < particlesPerSet; i++) {
        string index = cc.intToString(i+1);
        atomNames.push_back("P"+index);
        posNames.push_back("pos"+index);
    }
    stringstream compute;
    int index = 0;
    for (auto& distance : distances) {
        const vector<int>& atoms = distance.second;
        string deltaName = atomNames[atoms[0]]+atomNames[atoms[1]];
        if (computedDeltas.count(deltaName) == 0) {
            compute<<"real4 delta"<<deltaName<<" = delta("<<posNames[atoms[0]]<<", "<<posNames[atoms[1]]<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName);
        }
        compute<<"real r_"<<deltaName<<" = sqrt(delta"<<deltaName<<".w);\n";
        variables.push_back(makeVariable(distance.first, "r_"+deltaName));
        forceExpressions["real dEdDistance"+cc.intToString(index)+" = "] = energyExpression.differentiate(distance.first).optimize();
        index++;
    }
    index = 0;
    for (auto& angle : angles) {
        const vector<int>& atoms = angle.second;
        string deltaName1 = atomNames[atoms[1]]+atomNames[atoms[0]];
        string deltaName2 = atomNames[atoms[1]]+atomNames[atoms[2]];
        string angleName = "angle_"+atomNames[atoms[0]]+atomNames[atoms[1]]+atomNames[atoms[2]];
        if (computedDeltas.count(deltaName1) == 0) {
            compute<<"real4 delta"<<deltaName1<<" = delta("<<posNames[atoms[1]]<<", "<<posNames[atoms[0]]<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName1);
        }
        if (computedDeltas.count(deltaName2) == 0) {
            compute<<"real4 delta"<<deltaName2<<" = delta("<<posNames[atoms[1]]<<", "<<posNames[atoms[2]]<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName2);
        }
        compute<<"real "<<angleName<<" = computeAngle(delta"<<deltaName1<<", delta"<<deltaName2<<");\n";
        variables.push_back(makeVariable(angle.first, angleName));
        forceExpressions["real dEdAngle"+cc.intToString(index)+" = "] = energyExpression.differentiate(angle.first).optimize();
        index++;
    }
    index = 0;
    for (auto& dihedral : dihedrals) {
        const vector<int>& atoms = dihedral.second;
        string deltaName1 = atomNames[atoms[0]]+atomNames[atoms[1]];
        string deltaName2 = atomNames[atoms[2]]+atomNames[atoms[1]];
        string deltaName3 = atomNames[atoms[2]]+atomNames[atoms[3]];
        string crossName1 = "cross_"+deltaName1+"_"+deltaName2;
        string crossName2 = "cross_"+deltaName2+"_"+deltaName3;
        string dihedralName = "dihedral_"+atomNames[atoms[0]]+atomNames[atoms[1]]+atomNames[atoms[2]]+atomNames[atoms[3]];
        if (computedDeltas.count(deltaName1) == 0) {
            compute<<"real4 delta"<<deltaName1<<" = delta("<<posNames[atoms[0]]<<", "<<posNames[atoms[1]]<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName1);
        }
        if (computedDeltas.count(deltaName2) == 0) {
            compute<<"real4 delta"<<deltaName2<<" = delta("<<posNames[atoms[2]]<<", "<<posNames[atoms[1]]<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName2);
        }
        if (computedDeltas.count(deltaName3) == 0) {
            compute<<"real4 delta"<<deltaName3<<" = delta("<<posNames[atoms[2]]<<", "<<posNames[atoms[3]]<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);\n";
            computedDeltas.insert(deltaName3);
        }
        compute<<"real4 "<<crossName1<<" = computeCross(delta"<<deltaName1<<", delta"<<deltaName2<<");\n";
        compute<<"real4 "<<crossName2<<" = computeCross(delta"<<deltaName2<<", delta"<<deltaName3<<");\n";
        compute<<"real "<<dihedralName<<" = computeAngle("<<crossName1<<", "<<crossName2<<");\n";
        compute<<dihedralName<<" *= (delta"<<deltaName1<<".x*"<<crossName2<<".x + delta"<<deltaName1<<".y*"<<crossName2<<".y + delta"<<deltaName1<<".z*"<<crossName2<<".z < 0 ? -1 : 1);\n";
        variables.push_back(makeVariable(dihedral.first, dihedralName));
        forceExpressions["real dEdDihedral"+cc.intToString(index)+" = "] = energyExpression.differentiate(dihedral.first).optimize();
        index++;
    }

    // Now evaluate the expressions.

    for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
        ComputeParameterInfo& parameter = params->getParameterInfos()[i];
        compute<<parameter.getType()<<" params"<<(i+1)<<" = global_params"<<(i+1)<<"[index];\n";
    }
    forceExpressions["energy += "] = energyExpression;
    compute << cc.getExpressionUtilities().createExpressions(forceExpressions, variables, functionList, functionDefinitions, "temp");

    // Apply forces to atoms.

    vector<string> forceNames;
    for (int i = 0; i < particlesPerSet; i++) {
        string istr = cc.intToString(i+1);
        string forceName = "force"+istr;
        forceNames.push_back(forceName);
        compute<<"real3 "<<forceName<<" = make_real3(0);\n";
        compute<<"{\n";
        Lepton::ParsedExpression forceExpressionX = energyExpression.differentiate("x"+istr).optimize();
        Lepton::ParsedExpression forceExpressionY = energyExpression.differentiate("y"+istr).optimize();
        Lepton::ParsedExpression forceExpressionZ = energyExpression.differentiate("z"+istr).optimize();
        map<string, Lepton::ParsedExpression> expressions;
        if (!isZeroExpression(forceExpressionX))
            expressions[forceName+".x -= "] = forceExpressionX;
        if (!isZeroExpression(forceExpressionY))
            expressions[forceName+".y -= "] = forceExpressionY;
        if (!isZeroExpression(forceExpressionZ))
            expressions[forceName+".z -= "] = forceExpressionZ;
        if (expressions.size() > 0)
            compute<<cc.getExpressionUtilities().createExpressions(expressions, variables, functionList, functionDefinitions, "coordtemp");
        compute<<"}\n";
    }
    index = 0;
    for (auto& distance : distances) {
        const vector<int>& atoms = distance.second;
        string deltaName = atomNames[atoms[0]]+atomNames[atoms[1]];
        string value = "(dEdDistance"+cc.intToString(index)+"/r_"+deltaName+")*trimTo3(delta"+deltaName+")";
        compute<<forceNames[atoms[0]]<<" += "<<"-"<<value<<";\n";
        compute<<forceNames[atoms[1]]<<" += "<<value<<";\n";
        index++;
    }
    index = 0;
    for (auto& angle : angles) {
        const vector<int>& atoms = angle.second;
        string deltaName1 = atomNames[atoms[1]]+atomNames[atoms[0]];
        string deltaName2 = atomNames[atoms[1]]+atomNames[atoms[2]];
        compute<<"{\n";
        compute<<"real3 crossProd = trimTo3(cross(delta"<<deltaName2<<", delta"<<deltaName1<<"));\n";
        compute<<"real lengthCross = max(SQRT(dot(crossProd, crossProd)), (real) 1e-6f);\n";
        compute<<"real3 deltaCross0 = -cross(trimTo3(delta"<<deltaName1<<"), crossProd)*dEdAngle"<<cc.intToString(index)<<"/(delta"<<deltaName1<<".w*lengthCross);\n";
        compute<<"real3 deltaCross2 = cross(trimTo3(delta"<<deltaName2<<"), crossProd)*dEdAngle"<<cc.intToString(index)<<"/(delta"<<deltaName2<<".w*lengthCross);\n";
        compute<<"real3 deltaCross1 = -(deltaCross0+deltaCross2);\n";
        compute<<forceNames[atoms[0]]<<" += deltaCross0;\n";
        compute<<forceNames[atoms[1]]<<" += deltaCross1;\n";
        compute<<forceNames[atoms[2]]<<" += deltaCross2;\n";
        compute<<"}\n";
        index++;
    }
    index = 0;
    for (auto& dihedral : dihedrals) {
        const vector<int>& atoms = dihedral.second;
        string deltaName1 = atomNames[atoms[0]]+atomNames[atoms[1]];
        string deltaName2 = atomNames[atoms[2]]+atomNames[atoms[1]];
        string deltaName3 = atomNames[atoms[2]]+atomNames[atoms[3]];
        string crossName1 = "cross_"+deltaName1+"_"+deltaName2;
        string crossName2 = "cross_"+deltaName2+"_"+deltaName3;
        compute<<"{\n";
        compute<<"real r = sqrt(delta"<<deltaName2<<".w);\n";
        compute<<"real4 ff;\n";
        compute<<"ff.x = (-dEdDihedral"<<cc.intToString(index)<<"*r)/"<<crossName1<<".w;\n";
        compute<<"ff.y = (delta"<<deltaName1<<".x*delta"<<deltaName2<<".x + delta"<<deltaName1<<".y*delta"<<deltaName2<<".y + delta"<<deltaName1<<".z*delta"<<deltaName2<<".z)/delta"<<deltaName2<<".w;\n";
        compute<<"ff.z = (delta"<<deltaName3<<".x*delta"<<deltaName2<<".x + delta"<<deltaName3<<".y*delta"<<deltaName2<<".y + delta"<<deltaName3<<".z*delta"<<deltaName2<<".z)/delta"<<deltaName2<<".w;\n";
        compute<<"ff.w = (dEdDihedral"<<cc.intToString(index)<<"*r)/"<<crossName2<<".w;\n";
        compute<<"real3 internalF0 = ff.x*trimTo3("<<crossName1<<");\n";
        compute<<"real3 internalF3 = ff.w*trimTo3("<<crossName2<<");\n";
        compute<<"real3 s = ff.y*internalF0 - ff.z*internalF3;\n";
        compute<<forceNames[atoms[0]]<<" += internalF0;\n";
        compute<<forceNames[atoms[1]]<<" += s-internalF0;\n";
        compute<<forceNames[atoms[2]]<<" += -s-internalF3;\n";
        compute<<forceNames[atoms[3]]<<" += internalF3;\n";
        compute<<"}\n";
        index++;
    }

    // Store forces to global memory.

    for (int i = 0; i < particlesPerSet; i++)
        compute<<"storeForce(atom"<<(i+1)<<", "<<forceNames[i]<<", forceBuffers);\n";

    // Create other replacements that depend on the number of particles per set.

    stringstream numCombinations, atomsForCombination, isValidCombination, permute, loadData, verifyCutoff, verifyExclusions;
    if (hasTypeFilters) {
        permute<<"int particleSet[] = {";
        for (int i = 0; i < particlesPerSet; i++) {
            permute<<"p"<<(i+1);
            if (i < particlesPerSet-1)
                permute<<", ";
        }
        permute<<"};\n";
    }
    for (int i = 0; i < particlesPerSet; i++) {
        if (hasTypeFilters)
            permute<<"int atom"<<(i+1)<<" = particleSet[particleOrder["<<particlesPerSet<<"*order+"<<i<<"]];\n";
        else
            permute<<"int atom"<<(i+1)<<" = p"<<(i+1)<<";\n";
        loadData<<"real3 pos"<<(i+1)<<" = trimTo3(posq[atom"<<(i+1)<<"]);\n";
        for (int j = 0; j < (int) params->getParameterInfos().size(); j++)
            loadData<<params->getParameterInfos()[j].getType()<<" params"<<(j+1)<<(i+1)<<" = global_params"<<(j+1)<<"[atom"<<(i+1)<<"];\n";
    }
    if (centralParticleMode) {
        for (int i = 1; i < particlesPerSet; i++) {
            if (i > 1)
                isValidCombination<<" && p"<<(i+1)<<">p"<<i<<" && ";
            isValidCombination<<"p"<<(i+1)<<"!=p1";
        }
    }
    else {
        for (int i = 2; i < particlesPerSet; i++) {
            if (i > 2)
                isValidCombination<<" && ";
            isValidCombination<<"a"<<(i+1)<<">a"<<i;
        }
    }
    atomsForCombination<<"int tempIndex = index;\n";
    for (int i = 1; i < particlesPerSet; i++) {
        if (i > 1)
            numCombinations<<"*";
        numCombinations<<"numNeighbors";
        if (centralParticleMode)
            atomsForCombination<<"int a"<<(i+1)<<" = tempIndex%numNeighbors;\n";
        else
            atomsForCombination<<"int a"<<(i+1)<<" = 1+tempIndex%numNeighbors;\n";
        if (i < particlesPerSet-1)
            atomsForCombination<<"tempIndex /= numNeighbors;\n";
    }
    if (particlesPerSet > 2) {
        if (centralParticleMode)
            atomsForCombination<<"a2 = (a3%2 == 0 ? a2 : numNeighbors-a2-1);\n";
        else
            atomsForCombination<<"a2 = (a3%2 == 0 ? a2 : numNeighbors-a2+1);\n";
    }
    for (int i = 1; i < particlesPerSet; i++) {
        if (nonbondedMethod == NoCutoff) {
            if (centralParticleMode)
                atomsForCombination<<"int p"<<(i+1)<<" = a"<<(i+1)<<";\n";
            else
                atomsForCombination<<"int p"<<(i+1)<<" = p1+a"<<(i+1)<<";\n";
        }
        else {
            if (centralParticleMode)
                atomsForCombination<<"int p"<<(i+1)<<" = neighbors[firstNeighbor+a"<<(i+1)<<"];\n";
            else
                atomsForCombination<<"int p"<<(i+1)<<" = neighbors[firstNeighbor-1+a"<<(i+1)<<"];\n";
        }
    }
    if (nonbondedMethod != NoCutoff) {
        for (int i = 1; i < particlesPerSet; i++)
            verifyCutoff<<"real3 pos"<<(i+1)<<" = trimTo3(posq[p"<<(i+1)<<"]);\n";
        if (!centralParticleMode) {
            for (int i = 1; i < particlesPerSet; i++) {
                for (int j = i+1; j < particlesPerSet; j++)
                    verifyCutoff<<"includeInteraction &= (delta(pos"<<(i+1)<<", pos"<<(j+1)<<", periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ).w < CUTOFF_SQUARED);\n";
            }
        }
    }
    if (force.getNumExclusions() > 0) {
        int startCheckFrom = (nonbondedMethod == NoCutoff ? 0 : 1);
        for (int i = startCheckFrom; i < particlesPerSet; i++)
            for (int j = i+1; j < particlesPerSet; j++)
                verifyExclusions<<"includeInteraction &= !isInteractionExcluded(p"<<(i+1)<<", p"<<(j+1)<<", exclusions, exclusionStartIndex);\n";
    }
    string computeTypeIndex = "particleTypes[p"+cc.intToString(particlesPerSet)+"]";
    for (int i = particlesPerSet-2; i >= 0; i--)
        computeTypeIndex = "particleTypes[p"+cc.intToString(i+1)+"]+"+cc.intToString(numTypes)+"*("+computeTypeIndex+")";

    // Create replacements for extra arguments.

    stringstream extraArgs;
    if (force.getNumGlobalParameters() > 0)
        extraArgs << ", GLOBAL const float* globals";
    for (int i = 0; i < (int) params->getParameterInfos().size(); i++) {
        ComputeParameterInfo& parameter = params->getParameterInfos()[i];
        extraArgs<<", GLOBAL const "<<parameter.getType()<<"* RESTRICT global_params"<<(i+1);
    }

    // Create the kernels.

    map<string, string> replacements;
    replacements["COMPUTE_INTERACTION"] = compute.str();
    replacements["NUM_CANDIDATE_COMBINATIONS"] = numCombinations.str();
    replacements["FIND_ATOMS_FOR_COMBINATION_INDEX"] = atomsForCombination.str();
    replacements["IS_VALID_COMBINATION"] = isValidCombination.str();
    replacements["VERIFY_CUTOFF"] = verifyCutoff.str();
    replacements["VERIFY_EXCLUSIONS"] = verifyExclusions.str();
    replacements["PERMUTE_ATOMS"] = permute.str();
    replacements["LOAD_PARTICLE_DATA"] = loadData.str();
    replacements["COMPUTE_TYPE_INDEX"] = computeTypeIndex;
    replacements["PARAMETER_ARGUMENTS"] = extraArgs.str()+tableArgs.str();
    map<string, string> defines;
    if (nonbondedMethod != NoCutoff)
        defines["USE_CUTOFF"] = "1";
    if (nonbondedMethod == CutoffPeriodic)
        defines["USE_PERIODIC"] = "1";
    if (centralParticleMode)
        defines["USE_CENTRAL_PARTICLE"] = "1";
    if (hasTypeFilters)
        defines["USE_FILTERS"] = "1";
    if (force.getNumExclusions() > 0)
        defines["USE_EXCLUSIONS"] = "1";
    defines["NUM_ATOMS"] = cc.intToString(cc.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
    defines["M_PI"] = cc.doubleToString(M_PI);
    defines["CUTOFF_SQUARED"] = cc.doubleToString(force.getCutoffDistance()*force.getCutoffDistance());
    defines["TILE_SIZE"] = cc.intToString(tileWidth);
    defines["NUM_BLOCKS"] = cc.intToString(numAtomBlocks);
    defines["FIND_NEIGHBORS_WORKGROUP_SIZE"] = cc.intToString(findNeighborsWorkgroupSize);
    defines["COMPUTE_NEIGHBORS_WORKGROUP_SIZE"] = cc.intToString(computeNeighborsWorkgroupSize);
    defines["tileflags_t"] = (tileWidth > 32 ? "unsigned long" : "unsigned int") ;
    ComputeProgram program = cc.compileProgram(cc.replaceStrings(CommonKernelSources::customManyParticle, replacements), defines);
    forceKernel = program->createKernel("computeInteraction");
    blockBoundsKernel = program->createKernel("findBlockBounds");
    neighborsKernel = program->createKernel("findNeighbors");
    startIndicesKernel = program->createKernel("computeNeighborStartIndices");
    copyPairsKernel = program->createKernel("copyPairsToNeighborList");
    event = cc.createEvent();
}

double CommonCalcCustomManyParticleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    const int tileWidth = cc.getSIMDWidth();
    if (!hasInitializedKernel) {
        hasInitializedKernel = true;

        // Set arguments for the force kernel.

        forceKernel->addArg(cc.getLongForceBuffer());
        forceKernel->addArg(cc.getEnergyBuffer());
        forceKernel->addArg(cc.getPosq());
        for (int i = 0; i < 5; i++)
            forceKernel->addArg();
        setPeriodicBoxArgs(cc, forceKernel, 3);
        if (nonbondedMethod != NoCutoff) {
            forceKernel->addArg(neighbors);
            forceKernel->addArg(neighborStartIndex);
        }
        if (particleTypes.isInitialized()) {
            forceKernel->addArg(particleTypes);
            forceKernel->addArg(orderIndex);
            forceKernel->addArg(particleOrder);
        }
        if (exclusions.isInitialized()) {
            forceKernel->addArg(exclusions);
            forceKernel->addArg(exclusionStartIndex);
        }
        if (globals.isInitialized())
            forceKernel->addArg(globals);
        for (auto& parameter : params->getParameterInfos())
            forceKernel->addArg(parameter.getArray());
        for (auto& function : tabulatedFunctions)
            forceKernel->addArg(function);

        if (nonbondedMethod != NoCutoff) {
            // Set arguments for the block bounds kernel.

            for (int i = 0; i < 5; i++)
                blockBoundsKernel->addArg(); // Periodic box information will be set just before it is executed.
            blockBoundsKernel->addArg(cc.getPosq());
            blockBoundsKernel->addArg(blockCenter);
            blockBoundsKernel->addArg(blockBoundingBox);
            blockBoundsKernel->addArg(numNeighborPairs);

            // Set arguments for the neighbor list kernel.

            for (int i = 0; i < 5; i++)
                neighborsKernel->addArg(); // Periodic box information will be set just before it is executed.
            neighborsKernel->addArg(cc.getPosq());
            neighborsKernel->addArg(blockCenter);
            neighborsKernel->addArg(blockBoundingBox);
            neighborsKernel->addArg(neighborPairs);
            neighborsKernel->addArg(numNeighborPairs);
            neighborsKernel->addArg(numNeighborsForAtom);
            neighborsKernel->addArg(maxNeighborPairs);
            if (exclusions.isInitialized()) {
                neighborsKernel->addArg(exclusions);
                neighborsKernel->addArg(exclusionStartIndex);
            }

            // Set arguments for the kernel to find neighbor list start indices.

            startIndicesKernel->addArg(numNeighborsForAtom);
            startIndicesKernel->addArg(neighborStartIndex);
            startIndicesKernel->addArg(numNeighborPairs);
            startIndicesKernel->addArg(maxNeighborPairs);

            // Set arguments for the kernel to assemble the final neighbor list.

            copyPairsKernel->addArg(neighborPairs);
            copyPairsKernel->addArg(neighbors);
            copyPairsKernel->addArg(numNeighborPairs);
            copyPairsKernel->addArg(maxNeighborPairs);
            copyPairsKernel->addArg(numNeighborsForAtom);
            copyPairsKernel->addArg(neighborStartIndex);
       }
    }
    if (globals.isInitialized()) {
        bool changed = false;
        for (int i = 0; i < (int) globalParamNames.size(); i++) {
            float value = (float) context.getParameter(globalParamNames[i]);
            if (value != globalParamValues[i])
                changed = true;
            globalParamValues[i] = value;
        }
        if (changed)
            globals.upload(globalParamValues);
    }
    while (true) {
        int* numPairs = (int*) cc.getPinnedBuffer();
        if (nonbondedMethod != NoCutoff) {
            setPeriodicBoxArgs(cc, forceKernel, 3);
            setPeriodicBoxArgs(cc, blockBoundsKernel, 0);
            setPeriodicBoxArgs(cc, neighborsKernel, 0);
            blockBoundsKernel->execute(cc.getPaddedNumAtoms()/tileWidth);
            neighborsKernel->execute(cc.getNumAtoms(), findNeighborsWorkgroupSize);

            // We need to make sure there was enough memory for the neighbor list.  Download the
            // information asynchronously so kernels can be running at the same time.

            numNeighborPairs.download(numPairs, false);
            event->enqueue();
            startIndicesKernel->execute(computeNeighborsWorkgroupSize, computeNeighborsWorkgroupSize);
            copyPairsKernel->execute(maxNeighborPairs);
        }
        int maxThreads = min(cc.getNumAtoms()*forceWorkgroupSize, cc.getEnergyBuffer().getSize());
        forceKernel->execute(maxThreads, forceWorkgroupSize);
        if (nonbondedMethod != NoCutoff) {
            // Make sure there was enough memory for the neighbor list.

            event->wait();
            if (*numPairs > maxNeighborPairs) {
                // Resize the arrays and run the calculation again.

                maxNeighborPairs = (int) (1.1*(*numPairs));
                neighborPairs.resize(maxNeighborPairs);
                neighbors.resize(maxNeighborPairs);
                neighborsKernel->setArg(11, maxNeighborPairs);
                startIndicesKernel->setArg(3, maxNeighborPairs);
                copyPairsKernel->setArg(3, maxNeighborPairs);
                continue;
            }
        }
        break;
    }
    return 0.0;
}

void CommonCalcCustomManyParticleForceKernel::copyParametersToContext(ContextImpl& context, const CustomManyParticleForce& force) {
    cc.setAsCurrent();
    int numParticles = force.getNumParticles();
    if (numParticles != cc.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Record the per-particle parameters.

    vector<vector<float> > paramVector(numParticles);
    vector<double> parameters;
    int type;
    for (int i = 0; i < numParticles; i++) {
        force.getParticleParameters(i, parameters, type);
        paramVector[i].resize(parameters.size());
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);

    // Mark that the current reordering may be invalid.

    cc.invalidateMolecules(info);
}

class CommonCalcGayBerneForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const GayBerneForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        int xparticle1, yparticle1;
        double sigma1, epsilon1, sx1, sy1, sz1, ex1, ey1, ez1;
        int xparticle2, yparticle2;
        double sigma2, epsilon2, sx2, sy2, sz2, ex2, ey2, ez2;
        force.getParticleParameters(particle1, sigma1, epsilon1, xparticle1, yparticle1, sx1, sy1, sz1, ex1, ey1, ez1);
        force.getParticleParameters(particle2, sigma2, epsilon2, xparticle2, yparticle2, sx2, sy2, sz2, ex2, ey2, ez2);
        return (sigma1 == sigma2 && epsilon1 == epsilon2 && sx1 == sx2 && sy1 == sy2 && sz1 == sz2 && ex1 == ex2 && ey1 == ey2 && ez1 == ez2);
    }
    int getNumParticleGroups() {
        return force.getNumExceptions()+force.getNumParticles();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        if (index < force.getNumExceptions()) {
            int particle1, particle2;
            double sigma, epsilon;
            force.getExceptionParameters(index, particle1, particle2, sigma, epsilon);
            particles.resize(2);
            particles[0] = particle1;
            particles[1] = particle2;
        }
        else {
            int particle = index-force.getNumExceptions();
            int xparticle, yparticle;
            double sigma, epsilon, sx, sy, sz, ex, ey, ez;
            force.getParticleParameters(particle, sigma, epsilon, xparticle, yparticle, sx, sy, sz, ex, ey, ez);
            particles.clear();
            particles.push_back(particle);
            if (xparticle > -1)
                particles.push_back(xparticle);
            if (yparticle > -1)
                particles.push_back(yparticle);
        }
    }
    bool areGroupsIdentical(int group1, int group2) {
        if (group1 < force.getNumExceptions() && group2 < force.getNumExceptions()) {
            int particle1, particle2;
            double sigma1, sigma2, epsilon1, epsilon2;
            force.getExceptionParameters(group1, particle1, particle2, sigma1, epsilon1);
            force.getExceptionParameters(group2, particle1, particle2, sigma2, epsilon2);
            return (sigma1 == sigma2 && epsilon1 == epsilon2);
        }
        return true;
    }
private:
    const GayBerneForce& force;
};

class CommonCalcGayBerneForceKernel::ReorderListener : public ComputeContext::ReorderListener {
public:
    ReorderListener(CommonCalcGayBerneForceKernel& owner) : owner(owner) {
    }
    void execute() {
        owner.sortAtoms();
    }
private:
    CommonCalcGayBerneForceKernel& owner;
};

void CommonCalcGayBerneForceKernel::initialize(const System& system, const GayBerneForce& force) {
    if (!cc.getSupports64BitGlobalAtomics())
        throw OpenMMException("GayBerneForce requires a device that supports 64 bit atomic operations");

    // Initialize interactions.

    int numParticles = force.getNumParticles();
    sigParams.initialize<mm_float4>(cc, cc.getPaddedNumAtoms(), "sigParams");
    epsParams.initialize<mm_float2>(cc, cc.getPaddedNumAtoms(), "epsParams");
    scale.initialize<mm_float4>(cc, cc.getPaddedNumAtoms(), "scale");
    axisParticleIndices.initialize<mm_int2>(cc, cc.getPaddedNumAtoms(), "axisParticleIndices");
    sortedParticles.initialize<int>(cc, cc.getPaddedNumAtoms(), "sortedParticles");
    aMatrix.initialize<float>(cc, 9*cc.getPaddedNumAtoms(), "aMatrix");
    bMatrix.initialize<float>(cc, 9*cc.getPaddedNumAtoms(), "bMatrix");
    gMatrix.initialize<float>(cc, 9*cc.getPaddedNumAtoms(), "gMatrix");
    vector<mm_float4> sigParamsVector(cc.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
    vector<mm_float2> epsParamsVector(cc.getPaddedNumAtoms(), mm_float2(0, 0));
    vector<mm_float4> scaleVector(cc.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
    vector<mm_int2> axisParticleVector(cc.getPaddedNumAtoms(), mm_int2(0, 0));
    isRealParticle.resize(cc.getPaddedNumAtoms());
    for (int i = 0; i < numParticles; i++) {
        int xparticle, yparticle;
        double sigma, epsilon, sx, sy, sz, ex, ey, ez;
        force.getParticleParameters(i, sigma, epsilon, xparticle, yparticle, sx, sy, sz, ex, ey, ez);
        axisParticleVector[i] = mm_int2(xparticle, yparticle);
        sigParamsVector[i] = mm_float4((float) (0.5*sigma), (float) (0.25*sx*sx), (float) (0.25*sy*sy), (float) (0.25*sz*sz));
        epsParamsVector[i] = mm_float2((float) sqrt(epsilon), (float) (0.125*(sx*sy + sz*sz)*sqrt(sx*sy)));
        scaleVector[i] = mm_float4((float) (1/sqrt(ex)), (float) (1/sqrt(ey)), (float) (1/sqrt(ez)), 0);
        isRealParticle[i] = (epsilon != 0.0);
    }
    sigParams.upload(sigParamsVector);
    epsParams.upload(epsParamsVector);
    scale.upload(scaleVector);
    axisParticleIndices.upload(axisParticleVector);

    // Record exceptions and exclusions.

    vector<mm_float2> exceptionParamsVec;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, sigma, epsilon);
        if (epsilon != 0.0) {
            exceptionParamsVec.push_back(mm_float2((float) sigma, (float) epsilon));
            exceptionAtoms.push_back(make_pair(particle1, particle2));
            isRealParticle[particle1] = true;
            isRealParticle[particle2] = true;
        }
        if (isRealParticle[particle1] && isRealParticle[particle2])
            excludedPairs.push_back(pair<int, int>(particle1, particle2));
    }
    numRealParticles = 0;
    for (int i = 0; i < isRealParticle.size(); i++)
        if (isRealParticle[i])
            numRealParticles++;
    int numExceptions = exceptionParamsVec.size();
    exclusions.initialize<int>(cc, max(1, (int) excludedPairs.size()), "exclusions");
    exclusionStartIndex.initialize<int>(cc, numRealParticles+1, "exclusionStartIndex");
    exceptionParticles.initialize<mm_int4>(cc, max(1, numExceptions), "exceptionParticles");
    exceptionParams.initialize<mm_float2>(cc, max(1, numExceptions), "exceptionParams");
    if (numExceptions > 0)
        exceptionParams.upload(exceptionParamsVec);

    // Create data structures used for the neighbor list.

    const int tileWidth = cc.getSIMDWidth();
    int numAtomBlocks = (numRealParticles+(tileWidth-1))/tileWidth;
    int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    blockCenter.initialize(cc, numAtomBlocks, 4*elementSize, "blockCenter");
    blockBoundingBox.initialize(cc, numAtomBlocks, 4*elementSize, "blockBoundingBox");
    sortedPos.initialize(cc, numRealParticles, 4*elementSize, "sortedPos");
    maxNeighborBlocks = numRealParticles*2;
    neighbors.initialize<int>(cc, maxNeighborBlocks*tileWidth, "neighbors");
    neighborIndex.initialize<int>(cc, maxNeighborBlocks, "neighborIndex");
    neighborBlockCount.initialize<int>(cc, 1, "neighborBlockCount");
    event = cc.createEvent();

    // Create array for accumulating torques.

    torque.initialize<long long>(cc, 3*cc.getPaddedNumAtoms(), "torque");
    cc.addAutoclearBuffer(torque);

    // Create the kernels.

    nonbondedMethod = force.getNonbondedMethod();
    bool useCutoff = (nonbondedMethod != GayBerneForce::NoCutoff);
    bool usePeriodic = (nonbondedMethod == GayBerneForce::CutoffPeriodic);
    map<string, string> defines;
    defines["USE_SWITCH"] = (useCutoff && force.getUseSwitchingFunction() ? "1" : "0");
    double cutoff = force.getCutoffDistance();
    defines["CUTOFF_SQUARED"] = cc.doubleToString(cutoff*cutoff);
    if (useCutoff) {
        defines["USE_CUTOFF"] = 1;
        if (usePeriodic)
            defines["USE_PERIODIC"] = "1";

        // Compute the switching coefficients.

        if (force.getUseSwitchingFunction()) {
            defines["SWITCH_CUTOFF"] = cc.doubleToString(force.getSwitchingDistance());
            defines["SWITCH_C3"] = cc.doubleToString(10/pow(force.getSwitchingDistance()-cutoff, 3.0));
            defines["SWITCH_C4"] = cc.doubleToString(15/pow(force.getSwitchingDistance()-cutoff, 4.0));
            defines["SWITCH_C5"] = cc.doubleToString(6/pow(force.getSwitchingDistance()-cutoff, 5.0));
        }
    }
    defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
    // note: the gayBerne exclusions are pair-wise, otherwise we'd have to do the tileflags_t define
    defines["TILE_SIZE"] = cc.intToString(cc.getSIMDWidth());
    ComputeProgram program = cc.compileProgram(CommonKernelSources::gayBerne, defines);
    framesKernel = program->createKernel("computeEllipsoidFrames");
    blockBoundsKernel = program->createKernel("findBlockBounds");
    neighborsKernel = program->createKernel("findNeighbors");
    forceKernel = program->createKernel("computeForce");
    torqueKernel = program->createKernel("applyTorques");
    info = new ForceInfo(force);
    cc.addForce(info);
    cc.addReorderListener(new ReorderListener(*this));
}

double CommonCalcGayBerneForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        sortAtoms();
        framesKernel->addArg(numRealParticles);
        framesKernel->addArg(cc.getPosq());
        framesKernel->addArg(axisParticleIndices);
        framesKernel->addArg(sigParams);
        framesKernel->addArg(scale);
        framesKernel->addArg(aMatrix);
        framesKernel->addArg(bMatrix);
        framesKernel->addArg(gMatrix);
        framesKernel->addArg(sortedParticles);
        blockBoundsKernel->addArg(numRealParticles);
        for (int i = 0; i < 5; i++)
            blockBoundsKernel->addArg(); // Periodic box information will be set just before it is executed.
        blockBoundsKernel->addArg(sortedParticles);
        blockBoundsKernel->addArg(cc.getPosq());
        blockBoundsKernel->addArg(sortedPos);
        blockBoundsKernel->addArg(blockCenter);
        blockBoundsKernel->addArg(blockBoundingBox);
        blockBoundsKernel->addArg(neighborBlockCount);
        neighborsKernel->addArg(numRealParticles);
        neighborsKernel->addArg(maxNeighborBlocks);
        for (int i = 0; i < 5; i++)
            neighborsKernel->addArg(); // Periodic box information will be set just before it is executed.
        neighborsKernel->addArg(sortedPos);
        neighborsKernel->addArg(blockCenter);
        neighborsKernel->addArg(blockBoundingBox);
        neighborsKernel->addArg(neighbors);
        neighborsKernel->addArg(neighborIndex);
        neighborsKernel->addArg(neighborBlockCount);
        neighborsKernel->addArg(exclusions);
        neighborsKernel->addArg(exclusionStartIndex);
        forceKernel->addArg(cc.getLongForceBuffer());
        forceKernel->addArg(torque);
        forceKernel->addArg(numRealParticles);
        forceKernel->addArg((int) exceptionAtoms.size());
        forceKernel->addArg(cc.getEnergyBuffer());
        forceKernel->addArg(sortedPos);
        forceKernel->addArg(sigParams);
        forceKernel->addArg(epsParams);
        forceKernel->addArg(sortedParticles);
        forceKernel->addArg(aMatrix);
        forceKernel->addArg(bMatrix);
        forceKernel->addArg(gMatrix);
        forceKernel->addArg(exclusions);
        forceKernel->addArg(exclusionStartIndex);
        forceKernel->addArg(exceptionParticles);
        forceKernel->addArg(exceptionParams);
        if (nonbondedMethod != GayBerneForce::NoCutoff) {
            forceKernel->addArg(maxNeighborBlocks);
            forceKernel->addArg(neighbors);
            forceKernel->addArg(neighborIndex);
            forceKernel->addArg(neighborBlockCount);
            for (int i = 0; i < 5; i++)
                forceKernel->addArg(); // Periodic box information will be set just before it is executed.
        }
        torqueKernel->addArg(cc.getLongForceBuffer());
        torqueKernel->addArg(torque);
        torqueKernel->addArg(numRealParticles);
        torqueKernel->addArg(cc.getPosq());
        torqueKernel->addArg(axisParticleIndices);
        torqueKernel->addArg(sortedParticles);
    }
    framesKernel->execute(numRealParticles);
    setPeriodicBoxArgs(cc, blockBoundsKernel, 1);
    const int tileWidth = cc.getSIMDWidth();
    blockBoundsKernel->execute((numRealParticles+(tileWidth-1))/tileWidth);
    if (nonbondedMethod == GayBerneForce::NoCutoff)
        forceKernel->execute(cc.getNonbondedUtilities().getNumForceThreadBlocks()*cc.getNonbondedUtilities().getForceThreadBlockSize());
    else {
        while (true) {
            setPeriodicBoxArgs(cc, neighborsKernel, 2);
            neighborsKernel->execute(numRealParticles);
            int* count = (int*) cc.getPinnedBuffer();
            neighborBlockCount.download(count, false);
            event->enqueue();
            setPeriodicBoxArgs(cc, forceKernel, 20);
            forceKernel->execute(cc.getNonbondedUtilities().getNumForceThreadBlocks()*cc.getNonbondedUtilities().getForceThreadBlockSize());
            event->wait();
            if (*count <= maxNeighborBlocks)
                break;

            // There wasn't enough room for the neighbor list, so we need to recreate it.

            maxNeighborBlocks = (int) ceil((*count)*1.1);
            neighbors.resize(maxNeighborBlocks*tileWidth);
            neighborIndex.resize(maxNeighborBlocks);
            neighborsKernel->setArg(10, neighbors);
            neighborsKernel->setArg(11, neighborIndex);
            forceKernel->setArg(17, neighbors);
            forceKernel->setArg(18, neighborIndex);
        }
    }
    torqueKernel->execute(numRealParticles);
    return 0.0;
}

void CommonCalcGayBerneForceKernel::copyParametersToContext(ContextImpl& context, const GayBerneForce& force) {
    // Make sure the new parameters are acceptable.

    if (force.getNumParticles() != cc.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    vector<int> exceptions;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, sigma, epsilon);
        if (exceptionAtoms.size() > exceptions.size() && make_pair(particle1, particle2) == exceptionAtoms[exceptions.size()])
            exceptions.push_back(i);
        else if (epsilon != 0.0)
            throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");
    }
    int numExceptions = exceptionAtoms.size();

    // Record the per-particle parameters.

    vector<mm_float4> sigParamsVector(cc.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
    vector<mm_float2> epsParamsVector(cc.getPaddedNumAtoms(), mm_float2(0, 0));
    vector<mm_float4> scaleVector(cc.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
    for (int i = 0; i < force.getNumParticles(); i++) {
        int xparticle, yparticle;
        double sigma, epsilon, sx, sy, sz, ex, ey, ez;
        force.getParticleParameters(i, sigma, epsilon, xparticle, yparticle, sx, sy, sz, ex, ey, ez);
        sigParamsVector[i] = mm_float4((float) (0.5*sigma), (float) (0.25*sx*sx), (float) (0.25*sy*sy), (float) (0.25*sz*sz));
        epsParamsVector[i] = mm_float2((float) sqrt(epsilon), (float) (0.125*(sx*sy + sz*sz)*sqrt(sx*sy)));
        scaleVector[i] = mm_float4((float) (1/sqrt(ex)), (float) (1/sqrt(ey)), (float) (1/sqrt(ez)), 0);
        if (epsilon != 0.0 && !isRealParticle[i])
            throw OpenMMException("updateParametersInContext: The set of ignored particles (ones with epsilon=0) has changed");
    }
    sigParams.upload(sigParamsVector);
    epsParams.upload(epsParamsVector);
    scale.upload(scaleVector);

    // Record the exceptions.

    if (numExceptions > 0) {
        vector<mm_float2> exceptionParamsVec(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            int atom1, atom2;
            double sigma, epsilon;
            force.getExceptionParameters(exceptions[i], atom1, atom2, sigma, epsilon);
            exceptionParamsVec[i] = mm_float2((float) sigma, (float) epsilon);
        }
        exceptionParams.upload(exceptionParamsVec);
    }
    cc.invalidateMolecules(info);
    sortAtoms();
}

void CommonCalcGayBerneForceKernel::sortAtoms() {
    // Sort the list of atoms by type to avoid thread divergence.  This is executed every time
    // the atoms are reordered.

    int nextIndex = 0;
    vector<int> particles(cc.getPaddedNumAtoms(), 0);
    const vector<int>& order = cc.getAtomIndex();
    vector<int> inverseOrder(order.size(), -1);
    for (int i = 0; i < cc.getNumAtoms(); i++) {
        int atom = order[i];
        if (isRealParticle[atom]) {
            inverseOrder[atom] = nextIndex;
            particles[nextIndex++] = atom;
        }
    }
    sortedParticles.upload(particles);

    // Update the list of exception particles.

    int numExceptions = exceptionAtoms.size();
    if (numExceptions > 0) {
        vector<mm_int4> exceptionParticlesVec(numExceptions);
        for (int i = 0; i < numExceptions; i++)
            exceptionParticlesVec[i] = mm_int4(exceptionAtoms[i].first, exceptionAtoms[i].second, inverseOrder[exceptionAtoms[i].first], inverseOrder[exceptionAtoms[i].second]);
        exceptionParticles.upload(exceptionParticlesVec);
    }

    // Rebuild the list of exclusions.

    vector<vector<int> > excludedAtoms(numRealParticles);
    for (int i = 0; i < excludedPairs.size(); i++) {
        int first = inverseOrder[min(excludedPairs[i].first, excludedPairs[i].second)];
        int second = inverseOrder[max(excludedPairs[i].first, excludedPairs[i].second)];
        excludedAtoms[first].push_back(second);
    }
    int index = 0;
    vector<int> exclusionVec(exclusions.getSize());
    vector<int> startIndexVec(exclusionStartIndex.getSize());
    for (int i = 0; i < numRealParticles; i++) {
        startIndexVec[i] = index;
        for (int j = 0; j < excludedAtoms[i].size(); j++)
            exclusionVec[index++] = excludedAtoms[i][j];
    }
    startIndexVec[numRealParticles] = index;
    exclusions.upload(exclusionVec);
    exclusionStartIndex.upload(startIndexVec);
}

void CommonIntegrateVerletStepKernel::initialize(const System& system, const VerletIntegrator& integrator) {
    cc.initializeContexts();
    cc.setAsCurrent();
    ComputeProgram program = cc.compileProgram(CommonKernelSources::verlet);
    kernel1 = program->createKernel("integrateVerletPart1");
    kernel2 = program->createKernel("integrateVerletPart2");
}

void CommonIntegrateVerletStepKernel::execute(ContextImpl& context, const VerletIntegrator& integrator) {
    cc.setAsCurrent();
    IntegrationUtilities& integration = cc.getIntegrationUtilities();
    int numAtoms = cc.getNumAtoms();
    int paddedNumAtoms = cc.getPaddedNumAtoms();
    double dt = integrator.getStepSize();
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        kernel1->addArg(numAtoms);
        kernel1->addArg(paddedNumAtoms);
        kernel1->addArg(integration.getStepSize());
        kernel1->addArg(cc.getPosq());
        kernel1->addArg(cc.getVelm());
        kernel1->addArg(cc.getLongForceBuffer());
        kernel1->addArg(integration.getPosDelta());
        if (cc.getUseMixedPrecision())
            kernel1->addArg(cc.getPosqCorrection());
        kernel2->addArg(numAtoms);
        kernel2->addArg(integration.getStepSize());
        kernel2->addArg(cc.getPosq());
        kernel2->addArg(cc.getVelm());
        kernel2->addArg(integration.getPosDelta());
        if (cc.getUseMixedPrecision())
            kernel2->addArg(cc.getPosqCorrection());
    }
    integration.setNextStepSize(dt);

    // Call the first integration kernel.

    kernel1->execute(numAtoms);

    // Apply constraints.

    integration.applyConstraints(integrator.getConstraintTolerance());

    // Call the second integration kernel.

    kernel2->execute(numAtoms);
    integration.computeVirtualSites();

    // Update the time and step count.

    cc.setTime(cc.getTime()+dt);
    cc.setStepCount(cc.getStepCount()+1);
    cc.reorderAtoms();

    // Reduce UI lag.

#ifdef WIN32
    cc.flushQueue();
#endif
}

double CommonIntegrateVerletStepKernel::computeKineticEnergy(ContextImpl& context, const VerletIntegrator& integrator) {
    return cc.getIntegrationUtilities().computeKineticEnergy(0.5*integrator.getStepSize());
}

void CommonIntegrateLangevinStepKernel::initialize(const System& system, const LangevinIntegrator& integrator) {
    cc.initializeContexts();
    cc.setAsCurrent();
    cc.getIntegrationUtilities().initRandomNumberGenerator(integrator.getRandomNumberSeed());
    ComputeProgram program = cc.compileProgram(CommonKernelSources::langevin);
    kernel1 = program->createKernel("integrateLangevinPart1");
    kernel2 = program->createKernel("integrateLangevinPart2");
    params.initialize(cc, 3, cc.getUseDoublePrecision() || cc.getUseMixedPrecision() ? sizeof(double) : sizeof(float), "langevinParams");
    prevStepSize = -1.0;
}

void CommonIntegrateLangevinStepKernel::execute(ContextImpl& context, const LangevinIntegrator& integrator) {
    cc.setAsCurrent();
    IntegrationUtilities& integration = cc.getIntegrationUtilities();
    int numAtoms = cc.getNumAtoms();
    int paddedNumAtoms = cc.getPaddedNumAtoms();
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        kernel1->addArg(numAtoms);
        kernel1->addArg(paddedNumAtoms);
        kernel1->addArg(cc.getVelm());
        kernel1->addArg(cc.getLongForceBuffer());
        kernel1->addArg(integration.getPosDelta());
        kernel1->addArg(params);
        kernel1->addArg(integration.getStepSize());
        kernel1->addArg(integration.getRandom());
        kernel1->addArg();
        kernel2->addArg(numAtoms);
        kernel2->addArg(cc.getPosq());
        kernel2->addArg(integration.getPosDelta());
        kernel2->addArg(cc.getVelm());
        kernel2->addArg(integration.getStepSize());
        if (cc.getUseMixedPrecision())
            kernel2->addArg(cc.getPosqCorrection());
    }
    double temperature = integrator.getTemperature();
    double friction = integrator.getFriction();
    double stepSize = integrator.getStepSize();
    cc.getIntegrationUtilities().setNextStepSize(stepSize);
    if (temperature != prevTemp || friction != prevFriction || stepSize != prevStepSize) {
        // Calculate the integration parameters.

        double kT = BOLTZ*temperature;
        double vscale = exp(-stepSize*friction);
        double fscale = (friction == 0 ? stepSize : (1-vscale)/friction);
        double noisescale = sqrt(kT*(1-vscale*vscale));
        vector<double> p(params.getSize());
        p[0] = vscale;
        p[1] = fscale;
        p[2] = noisescale;
        params.upload(p, true);
        prevTemp = temperature;
        prevFriction = friction;
        prevStepSize = stepSize;
    }

    // Call the first integration kernel.

    kernel1->setArg(8, integration.prepareRandomNumbers(cc.getPaddedNumAtoms()));
    kernel1->execute(numAtoms);

    // Apply constraints.

    integration.applyConstraints(integrator.getConstraintTolerance());

    // Call the second integration kernel.

    kernel2->execute(numAtoms);
    integration.computeVirtualSites();

    // Update the time and step count.

    cc.setTime(cc.getTime()+stepSize);
    cc.setStepCount(cc.getStepCount()+1);
    cc.reorderAtoms();

    // Reduce UI lag.

#ifdef WIN32
    cc.flushQueue();
#endif
}

double CommonIntegrateLangevinStepKernel::computeKineticEnergy(ContextImpl& context, const LangevinIntegrator& integrator) {
    return cc.getIntegrationUtilities().computeKineticEnergy(0.5*integrator.getStepSize());
}

void CommonIntegrateLangevinMiddleStepKernel::initialize(const System& system, const LangevinMiddleIntegrator& integrator) {
    cc.initializeContexts();
    cc.setAsCurrent();
    cc.getIntegrationUtilities().initRandomNumberGenerator(integrator.getRandomNumberSeed());
    ComputeProgram program = cc.compileProgram(CommonKernelSources::langevinMiddle);
    kernel1 = program->createKernel("integrateLangevinMiddlePart1");
    kernel2 = program->createKernel("integrateLangevinMiddlePart2");
    kernel3 = program->createKernel("integrateLangevinMiddlePart3");
    if (cc.getUseDoublePrecision() || cc.getUseMixedPrecision()) {
        params.initialize<double>(cc, 2, "langevinMiddleParams");
        oldDelta.initialize<mm_double4>(cc, cc.getPaddedNumAtoms(), "oldDelta");
    }
    else {
        params.initialize<float>(cc, 2, "langevinMiddleParams");
        oldDelta.initialize<mm_float4>(cc, cc.getPaddedNumAtoms(), "oldDelta");
    }
    prevStepSize = -1.0;
}

void CommonIntegrateLangevinMiddleStepKernel::execute(ContextImpl& context, const LangevinMiddleIntegrator& integrator) {
    IntegrationUtilities& integration = cc.getIntegrationUtilities();
    int numAtoms = cc.getNumAtoms();
    int paddedNumAtoms = cc.getPaddedNumAtoms();
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        kernel1->addArg(numAtoms);
        kernel1->addArg(paddedNumAtoms);
        kernel1->addArg(cc.getVelm());
        kernel1->addArg(cc.getLongForceBuffer());
        kernel1->addArg(integration.getStepSize());
        kernel2->addArg(numAtoms);
        kernel2->addArg(cc.getVelm());
        kernel2->addArg(integration.getPosDelta());
        kernel2->addArg(oldDelta);
        kernel2->addArg(params);
        kernel2->addArg(integration.getStepSize());
        kernel2->addArg(integration.getRandom());
        kernel2->addArg(); // Random index will be set just before it is executed.
        kernel3->addArg(numAtoms);
        kernel3->addArg(cc.getPosq());
        kernel3->addArg(cc.getVelm());
        kernel3->addArg(integration.getPosDelta());
        kernel3->addArg(oldDelta);
        kernel3->addArg(integration.getStepSize());
        if (cc.getUseMixedPrecision())
            kernel3->addArg(cc.getPosqCorrection());
    }
    double temperature = integrator.getTemperature();
    double friction = integrator.getFriction();
    double stepSize = integrator.getStepSize();
    cc.getIntegrationUtilities().setNextStepSize(stepSize);
    if (temperature != prevTemp || friction != prevFriction || stepSize != prevStepSize) {
        // Calculate the integration parameters.

        double kT = BOLTZ*temperature;
        double vscale = exp(-stepSize*friction);
        double noisescale = sqrt(kT*(1-vscale*vscale));
        vector<double> p(params.getSize());
        p[0] = vscale;
        p[1] = noisescale;
        params.upload(p, true);
        prevTemp = temperature;
        prevFriction = friction;
        prevStepSize = stepSize;
    }

    // Perform the integration.

    kernel2->setArg(7, integration.prepareRandomNumbers(cc.getPaddedNumAtoms()));
    kernel1->execute(numAtoms);
    integration.applyVelocityConstraints(integrator.getConstraintTolerance());
    kernel2->execute(numAtoms);
    integration.applyConstraints(integrator.getConstraintTolerance());
    kernel3->execute(numAtoms);
    integration.computeVirtualSites();

    // Update the time and step count.

    cc.setTime(cc.getTime()+stepSize);
    cc.setStepCount(cc.getStepCount()+1);
    cc.reorderAtoms();

    // Reduce UI lag.

#ifdef WIN32
    cc.flushQueue();
#endif
}

double CommonIntegrateLangevinMiddleStepKernel::computeKineticEnergy(ContextImpl& context, const LangevinMiddleIntegrator& integrator) {
    return cc.getIntegrationUtilities().computeKineticEnergy(0.0);
}

void CommonIntegrateNoseHooverStepKernel::initialize(const System& system, const NoseHooverIntegrator& integrator) {
    cc.initializeContexts();
    bool useDouble = cc.getUseDoublePrecision() || cc.getUseMixedPrecision();
    map<string, string> defines;
    defines["BOLTZ"] = cc.doubleToString(BOLTZ);
    ComputeProgram program = cc.compileProgram(CommonKernelSources::noseHooverIntegrator, defines);
    kernel1 = program->createKernel("integrateNoseHooverMiddlePart1");
    kernel2 = program->createKernel("integrateNoseHooverMiddlePart2");
    kernel3 = program->createKernel("integrateNoseHooverMiddlePart3");
    kernel4 = program->createKernel("integrateNoseHooverMiddlePart4");
    if (useDouble) {
        oldDelta.initialize<mm_double4>(cc, cc.getPaddedNumAtoms(), "oldDelta");
    } else {
        oldDelta.initialize<mm_float4>(cc, cc.getPaddedNumAtoms(), "oldDelta");
    }
    kernelHardWall = program->createKernel("integrateNoseHooverHardWall");
    prevMaxPairDistance = -1.0f;
    maxPairDistanceBuffer.initialize<float>(cc, 1, "maxPairDistanceBuffer");

    int workGroupSize = std::min(cc.getMaxThreadBlockSize(), 512);
    defines["WORK_GROUP_SIZE"] = std::to_string(workGroupSize);

    defines["BEGIN_YS_LOOP"] = "const real arr[1] = {1.0};"
                               "for(int i=0;i<1;++i) {"
                               "const real ys = arr[i];";
    defines["END_YS_LOOP"] = "}";
    program = cc.compileProgram(CommonKernelSources::noseHooverChain, defines);
    propagateKernels[1] = program->createKernel("propagateNoseHooverChain");

    defines["BEGIN_YS_LOOP"] = "const real arr[3] = {0.828981543588751, -0.657963087177502, 0.828981543588751};"
                               "for(int i=0;i<3;++i) {"
                               "const real ys = arr[i];";
    program = cc.compileProgram(CommonKernelSources::noseHooverChain, defines);
    propagateKernels[3] = program->createKernel("propagateNoseHooverChain");

    defines["BEGIN_YS_LOOP"] = "const real arr[5] = {0.2967324292201065, 0.2967324292201065, -0.186929716880426, 0.2967324292201065, 0.2967324292201065};"
                               "for(int i=0;i<5;++i) {"
                               "const real ys = arr[i];";
    program = cc.compileProgram(CommonKernelSources::noseHooverChain, defines);
    propagateKernels[5] = program->createKernel("propagateNoseHooverChain");

    defines["BEGIN_YS_LOOP"] = "const real arr[7] = {0.784513610477560, 0.235573213359357, -1.17767998417887, 1.31518632068391,-1.17767998417887, 0.235573213359357, 0.784513610477560};"
                               "for(int i=0;i<7;++i) {"
                               "const real ys = arr[i];";
    program = cc.compileProgram(CommonKernelSources::noseHooverChain, defines);
    propagateKernels[7] = program->createKernel("propagateNoseHooverChain");
    program = cc.compileProgram(CommonKernelSources::noseHooverChain, defines);
    reduceEnergyKernel = program->createKernel("reduceEnergyPair");

    computeHeatBathEnergyKernel = program->createKernel("computeHeatBathEnergy");
    computeAtomsKineticEnergyKernel = program->createKernel("computeAtomsKineticEnergy");
    computePairsKineticEnergyKernel = program->createKernel("computePairsKineticEnergy");
    scaleAtomsVelocitiesKernel = program->createKernel("scaleAtomsVelocities");
    scalePairsVelocitiesKernel = program->createKernel("scalePairsVelocities");
    int energyBufferSize = cc.getEnergyBuffer().getSize();
    if (cc.getUseDoublePrecision() || cc.getUseMixedPrecision())
        energyBuffer.initialize<mm_double2>(cc, energyBufferSize, "energyBuffer");
    else
        energyBuffer.initialize<mm_float2>(cc, energyBufferSize, "energyBuffer");
}

void CommonIntegrateNoseHooverStepKernel::execute(ContextImpl& context, const NoseHooverIntegrator& integrator, bool &forcesAreValid) {
    IntegrationUtilities& integration = cc.getIntegrationUtilities();
    int paddedNumAtoms = cc.getPaddedNumAtoms();
    double dt = integrator.getStepSize();
    cc.getIntegrationUtilities().setNextStepSize(dt);

    // If the atom reordering has occured, the forces from the previous step are permuted and thus invalid.
    // They need to be either sorted or recomputed; here we choose the latter.
    if (!forcesAreValid || cc.getAtomsWereReordered()) context.calcForcesAndEnergy(true, false, integrator.getIntegrationForceGroups());

    const auto& atomList = integrator.getAllThermostatedIndividualParticles();
    const auto& pairList = integrator.getAllThermostatedPairs();
    int numAtoms = atomList.size();
    int numPairs = pairList.size();
    int numParticles = numAtoms + 2*numPairs;
    float maxPairDistance = integrator.getMaximumPairDistance();
    // Make sure atom and pair metadata is uploaded and has the correct dimensions
    if (prevMaxPairDistance != maxPairDistance) {
        std::vector<float> tmp(1, maxPairDistance);
        maxPairDistanceBuffer.upload(tmp);
        prevMaxPairDistance = maxPairDistance;
    }
    if (numAtoms !=0 && (!atomListBuffer.isInitialized() || atomListBuffer.getSize() != numAtoms)) {
        if (atomListBuffer.isInitialized())
            atomListBuffer.resize(atomList.size());
        else
            atomListBuffer.initialize<int>(cc, atomList.size(), "atomListBuffer");
        atomListBuffer.upload(atomList);
    }
    if (numPairs !=0 && (!pairListBuffer.isInitialized() || pairListBuffer.getSize() != numPairs)) {
        if (pairListBuffer.isInitialized()) {
            pairListBuffer.resize(pairList.size());
            pairTemperatureBuffer.resize(pairList.size());
        }
        else {
            pairListBuffer.initialize<mm_int2>(cc, pairList.size(), "pairListBuffer");
            pairTemperatureBuffer.initialize<float>(cc, pairList.size(), "pairTemperatureBuffer");
        }
        std::vector<mm_int2> tmp;
        std::vector<float> tmp2;
        for(const auto &pair : pairList) {
            tmp.push_back(mm_int2(std::get<0>(pair), std::get<1>(pair)));
            tmp2.push_back(std::get<2>(pair));
        }
        pairListBuffer.upload(tmp);
        pairTemperatureBuffer.upload(tmp2);
    }
    int totalAtoms = cc.getNumAtoms();
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        kernel1->addArg(numAtoms);
        kernel1->addArg(numPairs);
        kernel1->addArg(paddedNumAtoms);
        kernel1->addArg(cc.getVelm());
        kernel1->addArg(cc.getLongForceBuffer());
        kernel1->addArg(integration.getStepSize());
        kernel1->addArg(numAtoms > 0 ? atomListBuffer : cc.getEnergyBuffer()); // The array is not used if num == 0
        kernel1->addArg(numPairs > 0 ? pairListBuffer : cc.getEnergyBuffer()); // The array is not used if num == 0
        kernel2->addArg(totalAtoms);
        kernel2->addArg(cc.getVelm());
        kernel2->addArg(integration.getPosDelta());
        kernel2->addArg(oldDelta);
        kernel2->addArg(integration.getStepSize());
        kernel3->addArg(totalAtoms);
        kernel3->addArg(cc.getVelm());
        kernel3->addArg(integration.getPosDelta());
        kernel3->addArg(oldDelta);
        kernel3->addArg(integration.getStepSize());
        kernel4->addArg(totalAtoms);
        kernel4->addArg(cc.getPosq());
        kernel4->addArg(cc.getVelm());
        kernel4->addArg(integration.getPosDelta());
        kernel4->addArg(oldDelta);
        kernel4->addArg(integration.getStepSize());
        if (cc.getUseMixedPrecision())
            kernel4->addArg(cc.getPosqCorrection());
        if (numPairs > 0) {
            kernelHardWall->addArg(numPairs);
            kernelHardWall->addArg(maxPairDistanceBuffer);
            kernelHardWall->addArg(integration.getStepSize());
            kernelHardWall->addArg(cc.getPosq());
            kernelHardWall->addArg(cc.getVelm());
            kernelHardWall->addArg(pairListBuffer);
            kernelHardWall->addArg(pairTemperatureBuffer);
            if (cc.getUseMixedPrecision())
                kernelHardWall->addArg(cc.getPosqCorrection());
        }
    }

    /*
     * Carry out the LF-middle integration (c.f. J. Phys. Chem. A 2019, 123, 6056−6079)
     */
    // Velocity update
    kernel1->execute(std::max(numAtoms, numPairs));
    integration.applyVelocityConstraints(integrator.getConstraintTolerance());
    // Position update
    kernel2->execute(numParticles);
    // Apply the thermostat
    int numChains = integrator.getNumThermostats();
    for(int chain = 0; chain < numChains; ++chain) {
        const auto &thermostatChain = integrator.getThermostat(chain);
        auto KEs = computeMaskedKineticEnergy(context, thermostatChain, false);
        auto scaleFactors = propagateChain(context, thermostatChain, KEs, dt);
        scaleVelocities(context, thermostatChain, scaleFactors);
    }
    // Position update
    kernel3->execute(numParticles);
    integration.applyConstraints(integrator.getConstraintTolerance());
    // Apply constraint forces
    kernel4->execute(numAtoms);
    // Make sure any Drude-like particles have not wandered too far from home
    if (numPairs > 0) kernelHardWall->execute(numPairs);
    integration.computeVirtualSites();

    // Update the time and step count.
    cc.setTime(cc.getTime()+dt);
    cc.setStepCount(cc.getStepCount()+1);
    cc.reorderAtoms();

    // Reduce UI lag.
#ifdef WIN32
    cc.flushQueue();
#endif
}

double CommonIntegrateNoseHooverStepKernel::computeKineticEnergy(ContextImpl& context, const NoseHooverIntegrator& integrator) {
    return cc.getIntegrationUtilities().computeKineticEnergy(0);
}


std::pair<double, double> CommonIntegrateNoseHooverStepKernel::propagateChain(ContextImpl& context, const NoseHooverChain &nhc, std::pair<double, double> kineticEnergies, double timeStep) {
    bool useDouble = cc.getUseDoublePrecision() || cc.getUseMixedPrecision();
    int chainID = nhc.getChainID();
    int nAtoms = nhc.getThermostatedAtoms().size();
    int nPairs = nhc.getThermostatedPairs().size();
    int chainLength = nhc.getChainLength();
    int numYS = nhc.getNumYoshidaSuzukiTimeSteps();
    int numMTS = nhc.getNumMultiTimeSteps();

    if (numYS != 1 && numYS != 3 && numYS != 5 && numYS != 7) {
        throw OpenMMException("Number of Yoshida Suzuki time steps has to be 1, 3, 5, or 7.");
    }

    if (!scaleFactorBuffer.isInitialized() || scaleFactorBuffer.getSize() == 0) {
        if (useDouble) {
            std::vector<mm_double2> zeros{{0,0}};
            if (scaleFactorBuffer.isInitialized())
                scaleFactorBuffer.resize(1);
            else
                scaleFactorBuffer.initialize<mm_double2>(cc, 1, "scaleFactorBuffer");
            scaleFactorBuffer.upload(zeros);
        }
        else {
            std::vector<mm_float2> zeros{{0,0}};
            if (scaleFactorBuffer.isInitialized())
                scaleFactorBuffer.resize(1);
            else
                scaleFactorBuffer.initialize<mm_float2>(cc, 1, "scaleFactorBuffer");
            scaleFactorBuffer.upload(zeros);
        }
    }
    if (!chainForces.isInitialized() || !chainMasses.isInitialized()) {
        if (useDouble) {
            std::vector<double> zeros(chainLength,0);
            if (chainForces.isInitialized()) {
                chainMasses.resize(chainLength);
                chainForces.resize(chainLength);
            }
            else {
                chainMasses.initialize<double>(cc, chainLength, "chainMasses");
                chainForces.initialize<double>(cc, chainLength, "chainForces");
            }
            chainMasses.upload(zeros);
            chainForces.upload(zeros);
        }
        else {
            std::vector<float> zeros(chainLength,0);
            if (chainForces.isInitialized()) {
                chainMasses.resize(chainLength);
                chainForces.resize(chainLength);
            }
            else {
                chainMasses.initialize<float>(cc, chainLength, "chainMasses");
                chainForces.initialize<float>(cc, chainLength, "chainForces");
            }
            chainMasses.upload(zeros);
            chainForces.upload(zeros);
        }
    }
    if (chainForces.getSize() < chainLength)
        chainForces.resize(chainLength);
    if (chainMasses.getSize() < chainLength)
        chainMasses.resize(chainLength);


    // N.B. We ignore the incoming kineticEnergy and grab it from the device buffer instead
    if (nAtoms) {
        if (!chainState.count(2*chainID))
            chainState[2*chainID] = ComputeArray();
        if (!chainState.at(2*chainID).isInitialized() || chainState.at(2*chainID).getSize() != chainLength) {
            // We need to upload the Common array
            if (useDouble) {
                if (chainState.at(2*chainID).isInitialized())
                    chainState.at(2*chainID).resize(chainLength);
                else
                    chainState.at(2*chainID).initialize<mm_double2>(cc, chainLength, "chainState" + std::to_string(2*chainID));
                std::vector<mm_double2> zeros(chainLength, mm_double2(0.0, 0.0));
                chainState.at(2*chainID).upload(zeros.data());
            }
            else {
                if (chainState.at(2*chainID).isInitialized())
                    chainState.at(2*chainID).resize(chainLength);
                else
                    chainState.at(2*chainID).initialize<mm_float2>(cc, chainLength, "chainState" + std::to_string(2*chainID));
                std::vector<mm_float2> zeros(chainLength, mm_float2(0.0f, 0.0f));
                chainState.at(2*chainID).upload(zeros.data());
            }
        }
    }

    if (nPairs) {
        if (!chainState.count(2*chainID+1))
            chainState[2*chainID+1] = ComputeArray();
        if (!chainState.at(2*chainID+1).isInitialized() || chainState.at(2*chainID+1).getSize() != chainLength) {
            // We need to upload the Common array
            if (useDouble) {
                if (chainState.at(2*chainID+1).isInitialized())
                    chainState.at(2*chainID+1).resize(chainLength);
                else
                    chainState.at(2*chainID+1).initialize<mm_double2>(cc, chainLength, "chainState" + std::to_string(2*chainID+1));
                std::vector<mm_double2> zeros(chainLength, mm_double2(0.0, 0.0));
                chainState.at(2*chainID+1).upload(zeros.data());
            }
            else {
                if (chainState.at(2*chainID+1).isInitialized())
                    chainState.at(2*chainID+1).resize(chainLength);
                else
                    chainState.at(2*chainID+1).initialize<mm_float2>(cc, chainLength, "chainState" + std::to_string(2*chainID+1));
                std::vector<mm_float2> zeros(chainLength, mm_float2(0.0f, 0.0f));
                chainState.at(2*chainID+1).upload(zeros.data());
            }
        }
    }

    if (!hasInitializedPropagateKernel) {
        hasInitializedPropagateKernel = true;
        propagateKernels[numYS]->addArg(); // ChainState
        propagateKernels[numYS]->addArg(kineticEnergyBuffer);
        propagateKernels[numYS]->addArg(scaleFactorBuffer);
        propagateKernels[numYS]->addArg(chainMasses);
        propagateKernels[numYS]->addArg(chainForces);
        propagateKernels[numYS]->addArg(); // ChainType
        propagateKernels[numYS]->addArg(chainLength);
        propagateKernels[numYS]->addArg(numMTS);
        propagateKernels[numYS]->addArg(); // numDoFs
        propagateKernels[numYS]->addArg((float)timeStep);
        propagateKernels[numYS]->addArg(); // kT
        propagateKernels[numYS]->addArg(); // frequency
    }

    if (nAtoms) {
        int chainType = 0;
        double temperature = nhc.getTemperature();
        float frequency = nhc.getCollisionFrequency();
        double kT = BOLTZ * temperature;
        int numDOFs = nhc.getNumDegreesOfFreedom();
        propagateKernels[numYS]->setArg(0, chainState[2*chainID]);
        propagateKernels[numYS]->setArg(5, chainType);
        propagateKernels[numYS]->setArg(8, numDOFs);
        if (useDouble) {
            propagateKernels[numYS]->setArg(10, kT);
        } else {
            propagateKernels[numYS]->setArg(10, (float)kT);
        }
        propagateKernels[numYS]->setArg(11, frequency);
        propagateKernels[numYS]->execute(1, 1);
    }
    if (nPairs) {
        int chainType = 1;
        double relativeTemperature = nhc.getRelativeTemperature();
        float relativeFrequency = nhc.getRelativeCollisionFrequency();
        double kT = BOLTZ * relativeTemperature;
        int ndf = 3*nPairs;
        propagateKernels[numYS]->setArg(0, chainState[2*chainID+1]);
        propagateKernels[numYS]->setArg(5, chainType);
        propagateKernels[numYS]->setArg(8, ndf);
        if (useDouble) {
            propagateKernels[numYS]->setArg(10, kT);
        } else {
            propagateKernels[numYS]->setArg(10, (float)kT);
        }
        propagateKernels[numYS]->setArg(11, relativeFrequency);
        propagateKernels[numYS]->execute(1, 1);
    }
    return {0, 0};
}

double CommonIntegrateNoseHooverStepKernel::computeHeatBathEnergy(ContextImpl& context, const NoseHooverChain &nhc) {

    bool useDouble = cc.getUseDoublePrecision() || cc.getUseMixedPrecision();

    int chainID = nhc.getChainID();
    int chainLength = nhc.getChainLength();

    bool absChainIsValid = chainState.count(2*chainID) != 0 &&
                           chainState[2*chainID].isInitialized() &&
                           chainState[2*chainID].getSize() == chainLength;
    bool relChainIsValid = chainState.count(2*chainID+1) != 0 &&
                           chainState[2*chainID+1].isInitialized() &&
                           chainState[2*chainID+1].getSize() == chainLength;

    if (!absChainIsValid && !relChainIsValid) return 0.0;

    if (!heatBathEnergy.isInitialized() || heatBathEnergy.getSize() == 0) {
        if (useDouble) {
            std::vector<double> one(1);
            heatBathEnergy.initialize<double>(cc, 1, "heatBathEnergy");
            heatBathEnergy.upload(one);
        }
        else {
            std::vector<float> one(1);
            heatBathEnergy.initialize<float>(cc, 1, "heatBathEnergy");
            heatBathEnergy.upload(one);
        }
    }

    cc.clearBuffer(heatBathEnergy);

    if(!hasInitializedHeatBathEnergyKernel) {
        hasInitializedHeatBathEnergyKernel = true;
        computeHeatBathEnergyKernel->addArg(heatBathEnergy);
        computeHeatBathEnergyKernel->addArg(chainLength);
        computeHeatBathEnergyKernel->addArg(); // numDOFs
        computeHeatBathEnergyKernel->addArg(); // kT
        computeHeatBathEnergyKernel->addArg(); // frequency
        computeHeatBathEnergyKernel->addArg(); // chainstate
    }

    if (absChainIsValid) {
        int numDOFs = nhc.getNumDegreesOfFreedom();
        double temperature = nhc.getTemperature();
        float frequency = nhc.getCollisionFrequency();
        double kT = BOLTZ * temperature;

        computeHeatBathEnergyKernel->setArg(2, numDOFs);
        if (useDouble) {
            computeHeatBathEnergyKernel->setArg(3, kT);
        } else {
            computeHeatBathEnergyKernel->setArg(3, (float)kT);
        }
        computeHeatBathEnergyKernel->setArg(4, frequency);
        computeHeatBathEnergyKernel->setArg(5, chainState[2*chainID]);
        computeHeatBathEnergyKernel->execute(1, 1);
    }
    if (relChainIsValid) {
        int numDOFs = 3 * nhc.getThermostatedPairs().size();
        double temperature = nhc.getRelativeTemperature();
        float frequency = nhc.getRelativeCollisionFrequency();
        double kT = BOLTZ * temperature;

        computeHeatBathEnergyKernel->setArg(2, numDOFs);
        if (useDouble) {
            computeHeatBathEnergyKernel->setArg(3, kT);
        } else {
            computeHeatBathEnergyKernel->setArg(3, (float)kT);
        }
        computeHeatBathEnergyKernel->setArg(4, frequency);
        computeHeatBathEnergyKernel->setArg(5, chainState[2*chainID+1]);
        computeHeatBathEnergyKernel->execute(1, 1);
    }


    void * pinnedBuffer = cc.getPinnedBuffer();
    heatBathEnergy.download(pinnedBuffer);
    if (useDouble)
        return *((double*) pinnedBuffer);
    else
        return *((float*) pinnedBuffer);
}

std::pair<double, double> CommonIntegrateNoseHooverStepKernel::computeMaskedKineticEnergy(ContextImpl& context, const NoseHooverChain &nhc, bool downloadValue) {

    bool useDouble = cc.getUseDoublePrecision() || cc.getUseMixedPrecision();

    int chainID = nhc.getChainID();
    const auto & nhcAtoms = nhc.getThermostatedAtoms();
    const auto & nhcPairs = nhc.getThermostatedPairs();
    int nAtoms = nhcAtoms.size();
    int nPairs = nhcPairs.size();
    if (nAtoms) {
        if (!atomlists.count(chainID)) {
            // We need to upload the Common array
            atomlists[chainID] = ComputeArray();
            atomlists[chainID].initialize<int>(cc, nAtoms, "atomlist" + std::to_string(chainID));
            atomlists[chainID].upload(nhcAtoms);
        }
        if (atomlists[chainID].getSize() != nAtoms) {
            throw OpenMMException("Number of atoms changed. Cannot be handled by the same Nose-Hoover thermostat.");
        }
    }
    if (nPairs) {
        if (!pairlists.count(chainID)) {
            // We need to upload the Common array
            pairlists[chainID] = ComputeArray();
            pairlists[chainID].initialize<mm_int2>(cc, nPairs, "pairlist" + std::to_string(chainID));
            std::vector<mm_int2> int2vec;
            for(const auto &p : nhcPairs) int2vec.push_back(mm_int2(p.first, p.second));
            pairlists[chainID].upload(int2vec);
        }
        if (pairlists[chainID].getSize() != nPairs) {
            throw OpenMMException("Number of thermostated pairs changed. Cannot be handled by the same Nose-Hoover thermostat.");
        }
    }
    if (!kineticEnergyBuffer.isInitialized() || kineticEnergyBuffer.getSize() == 0) {
        if (useDouble) {
            std::vector<mm_double2> zeros{{0,0}};
            kineticEnergyBuffer.initialize<mm_double2>(cc, 1, "kineticEnergyBuffer");
            kineticEnergyBuffer.upload(zeros);
        }
        else {
            std::vector<mm_float2> zeros{{0,0}};
            kineticEnergyBuffer.initialize<mm_float2>(cc, 1, "kineticEnergyBuffer");
            kineticEnergyBuffer.upload(zeros);
        }
    }

    int workGroupSize = std::min(cc.getMaxThreadBlockSize(), 512);
    if (!hasInitializedKineticEnergyKernel) {
        hasInitializedKineticEnergyKernel = true;
        computeAtomsKineticEnergyKernel->addArg(energyBuffer);
        computeAtomsKineticEnergyKernel->addArg(); // nAtoms
        computeAtomsKineticEnergyKernel->addArg(cc.getVelm());
        computeAtomsKineticEnergyKernel->addArg(); // atom list

        computePairsKineticEnergyKernel->addArg(energyBuffer);
        computePairsKineticEnergyKernel->addArg(); // nPairs
        computePairsKineticEnergyKernel->addArg(cc.getVelm());
        computePairsKineticEnergyKernel->addArg(); // pair list

        reduceEnergyKernel->addArg(energyBuffer);
        reduceEnergyKernel->addArg(kineticEnergyBuffer);
        reduceEnergyKernel->addArg(energyBuffer.getSize());
    }

    cc.clearBuffer(energyBuffer);
    if (nAtoms) {
        computeAtomsKineticEnergyKernel->setArg(1, nAtoms);
        computeAtomsKineticEnergyKernel->setArg(3, atomlists[chainID]);
        computeAtomsKineticEnergyKernel->execute(nAtoms);
    }
    if (nPairs) {
        computePairsKineticEnergyKernel->setArg(1, nPairs);
        computePairsKineticEnergyKernel->setArg(3, pairlists[chainID]);
        computePairsKineticEnergyKernel->execute(nPairs);
    }
    reduceEnergyKernel->execute(workGroupSize, workGroupSize);

    std::pair<double, double> KEs = {0, 0};
    if (downloadValue) {
        if (useDouble) {
            mm_double2 tmp;
            kineticEnergyBuffer.download(&tmp);
            KEs.first = tmp.x;
            KEs.second = tmp.y;
        }
        else {
            mm_float2 tmp;
            kineticEnergyBuffer.download(&tmp);
            KEs.first = tmp.x;
            KEs.second = tmp.y;
        }
    }
    return KEs;
}

void CommonIntegrateNoseHooverStepKernel::scaleVelocities(ContextImpl& context, const NoseHooverChain &nhc, std::pair<double, double> scaleFactor) {
    // For now we assume that the atoms and pairs info is valid, because compute{Atoms|Pairs}KineticEnergy must have been
    // called before this kernel.  If that ever ceases to be true, some sanity checks are needed here.

    int chainID = nhc.getChainID();
    int nAtoms = nhc.getThermostatedAtoms().size();
    int nPairs = nhc.getThermostatedPairs().size();
    if (!hasInitializedScaleVelocitiesKernel) {
        hasInitializedScaleVelocitiesKernel = true;
        scaleAtomsVelocitiesKernel->addArg(scaleFactorBuffer);
        scaleAtomsVelocitiesKernel->addArg(); // nAtoms
        scaleAtomsVelocitiesKernel->addArg(cc.getVelm());
        scaleAtomsVelocitiesKernel->addArg(); // atom list

        scalePairsVelocitiesKernel->addArg(scaleFactorBuffer);
        scalePairsVelocitiesKernel->addArg(); // nPairs
        scalePairsVelocitiesKernel->addArg(cc.getVelm());
        scalePairsVelocitiesKernel->addArg(); // pair list
    }
    if (nAtoms) {
        scaleAtomsVelocitiesKernel->setArg(1, nAtoms);
        scaleAtomsVelocitiesKernel->setArg(3, atomlists[chainID]);
        scaleAtomsVelocitiesKernel->execute(nAtoms);
    }
    if (nPairs) {
        scalePairsVelocitiesKernel->setArg(1, nPairs);
        scalePairsVelocitiesKernel->setArg(3, pairlists[chainID]);
        scalePairsVelocitiesKernel->execute(nPairs);
    }
}

void CommonIntegrateNoseHooverStepKernel::createCheckpoint(ContextImpl& context, ostream& stream) const {
    int numChains = chainState.size();
    bool useDouble = cc.getUseDoublePrecision() || cc.getUseMixedPrecision();
    stream.write((char*) &numChains, sizeof(int));
    for (auto& state : chainState){
        int chainID = state.first;
        int chainLength = state.second.getSize();
        stream.write((char*) &chainID, sizeof(int));
        stream.write((char*) &chainLength, sizeof(int));
        if (useDouble) {
            vector<mm_double2> stateVec;
            state.second.download(stateVec);
            stream.write((char*) stateVec.data(), sizeof(mm_double2)*chainLength);
        }
        else {
            vector<mm_float2> stateVec;
            state.second.download(stateVec);
            stream.write((char*) stateVec.data(), sizeof(mm_float2)*chainLength);
        }
    }
}

void CommonIntegrateNoseHooverStepKernel::loadCheckpoint(ContextImpl& context, istream& stream) {
    int numChains;
    bool useDouble = cc.getUseDoublePrecision() || cc.getUseMixedPrecision();
    stream.read((char*) &numChains, sizeof(int));
    chainState.clear();
    for (int i = 0; i < numChains; i++) {
        int chainID, chainLength;
        stream.read((char*) &chainID, sizeof(int));
        stream.read((char*) &chainLength, sizeof(int));
        if (useDouble) {
            chainState[chainID] = ComputeArray();
            chainState[chainID].initialize<mm_double2>(cc, chainLength, "chainState" + to_string(chainID));
            vector<mm_double2> stateVec(chainLength);
            stream.read((char*) &stateVec[0], sizeof(mm_double2)*chainLength);
            chainState[chainID].upload(stateVec);
        }
        else {
            chainState[chainID] = ComputeArray();
            chainState[chainID].initialize<mm_float2>(cc, chainLength, "chainState" + to_string(chainID));
            vector<mm_float2> stateVec(chainLength);
            stream.read((char*) &stateVec[0], sizeof(mm_float2)*chainLength);
            chainState[chainID].upload(stateVec);
        }
    }
}

void CommonIntegrateNoseHooverStepKernel::getChainStates(ContextImpl& context, vector<vector<double> >& positions, vector<vector<double> >& velocities) const {
    int numChains = chainState.size();
    bool useDouble = cc.getUseDoublePrecision() || cc.getUseMixedPrecision();
    positions.clear();
    velocities.clear();
    positions.resize(numChains);
    velocities.resize(numChains);
    for (int i = 0; i < numChains; i++) {
        const ComputeArray& state = chainState.at(i);
        if (useDouble) {
            vector<mm_double2> stateVec;
            state.download(stateVec);
            for (int j = 0; j < stateVec.size(); j++) {
                positions[i].push_back(stateVec[j].x);
                velocities[i].push_back(stateVec[j].y);
            }
        }
        else {
            vector<mm_float2> stateVec;
            state.download(stateVec);
            for (int j = 0; j < stateVec.size(); j++) {
                positions[i].push_back((float) stateVec[j].x);
                velocities[i].push_back((float) stateVec[j].y);
            }
        }
    }
}

void CommonIntegrateNoseHooverStepKernel::setChainStates(ContextImpl& context, const vector<vector<double> >& positions, const vector<vector<double> >& velocities) {
    int numChains = positions.size();
    bool useDouble = cc.getUseDoublePrecision() || cc.getUseMixedPrecision();
    chainState.clear();
    for (int i = 0; i < numChains; i++) {
        int chainLength = positions[i].size();
        chainState[i] = ComputeArray();
        if (useDouble) {
            chainState[i].initialize<mm_double2>(cc, chainLength, "chainState"+cc.intToString(i));
            vector<mm_double2> stateVec;
            for (int j = 0; j < chainLength; j++)
                stateVec.push_back(mm_double2(positions[i][j], velocities[i][j]));
            chainState[i].upload(stateVec);
        }
        else {
            chainState[i].initialize<mm_float2>(cc, chainLength, "chainState"+cc.intToString(i));
            vector<mm_float2> stateVec;
            for (int j = 0; j < chainLength; j++)
                stateVec.push_back(mm_float2((float) positions[i][j], (float) velocities[i][j]));
            chainState[i].upload(stateVec);
        }
    }
}

void CommonIntegrateBrownianStepKernel::initialize(const System& system, const BrownianIntegrator& integrator) {
    cc.initializeContexts();
    cc.setAsCurrent();
    cc.getIntegrationUtilities().initRandomNumberGenerator(integrator.getRandomNumberSeed());
    ComputeProgram program = cc.compileProgram(CommonKernelSources::brownian);
    kernel1 = program->createKernel("integrateBrownianPart1");
    kernel2 = program->createKernel("integrateBrownianPart2");
    prevStepSize = -1.0;
}

void CommonIntegrateBrownianStepKernel::execute(ContextImpl& context, const BrownianIntegrator& integrator) {
    cc.setAsCurrent();
    IntegrationUtilities& integration = cc.getIntegrationUtilities();
    int numAtoms = cc.getNumAtoms();
    int paddedNumAtoms = cc.getPaddedNumAtoms();
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        kernel1->addArg(numAtoms);
        kernel1->addArg(paddedNumAtoms);
        kernel1->addArg(); // tauDeltaT
        kernel1->addArg(); // noiseAmplitude
        kernel1->addArg(cc.getLongForceBuffer());
        kernel1->addArg(integration.getPosDelta());
        kernel1->addArg(cc.getVelm());
        kernel1->addArg(integration.getRandom());
        kernel1->addArg(); // Random index will be set just before it is executed.
        kernel2->addArg(numAtoms);
        kernel2->addArg(); // oneOverDeltaT
        kernel2->addArg(cc.getPosq());
        kernel2->addArg(cc.getVelm());
        kernel2->addArg(integration.getPosDelta());
        if (cc.getUseMixedPrecision())
            kernel2->addArg(cc.getPosqCorrection());
    }
    double temperature = integrator.getTemperature();
    double friction = integrator.getFriction();
    double stepSize = integrator.getStepSize();
    if (temperature != prevTemp || friction != prevFriction || stepSize != prevStepSize) {
        double tau = (friction == 0.0 ? 0.0 : 1.0/friction);
        if (cc.getUseDoublePrecision() || cc.getUseMixedPrecision()) {
            kernel1->setArg(2, tau*stepSize);
            kernel1->setArg(3, sqrt(2.0f*BOLTZ*temperature*stepSize*tau));
            kernel2->setArg(1, 1.0/stepSize);
        }
        else {
            kernel1->setArg(2, (float) (tau*stepSize));
            kernel1->setArg(3, (float) (sqrt(2.0f*BOLTZ*temperature*stepSize*tau)));
            kernel2->setArg(1, (float) (1.0/stepSize));
        }
        prevTemp = temperature;
        prevFriction = friction;
        prevStepSize = stepSize;
    }

    // Call the first integration kernel.

    kernel1->setArg(8, integration.prepareRandomNumbers(cc.getPaddedNumAtoms()));
    kernel1->execute(numAtoms);

    // Apply constraints.

    integration.applyConstraints(integrator.getConstraintTolerance());

    // Call the second integration kernel.

    kernel2->execute(numAtoms);
    integration.computeVirtualSites();

    // Update the time and step count.

    cc.setTime(cc.getTime()+stepSize);
    cc.setStepCount(cc.getStepCount()+1);
    cc.reorderAtoms();

    // Reduce UI lag.

#ifdef WIN32
    cc.flushQueue();
#endif
}

double CommonIntegrateBrownianStepKernel::computeKineticEnergy(ContextImpl& context, const BrownianIntegrator& integrator) {
    return cc.getIntegrationUtilities().computeKineticEnergy(0);
}

void CommonIntegrateVariableVerletStepKernel::initialize(const System& system, const VariableVerletIntegrator& integrator) {
    cc.initializeContexts();
    cc.setAsCurrent();
    ComputeProgram program = cc.compileProgram(CommonKernelSources::verlet);
    kernel1 = program->createKernel("integrateVerletPart1");
    kernel2 = program->createKernel("integrateVerletPart2");
    selectSizeKernel = program->createKernel("selectVerletStepSize");
    blockSize = min(256, system.getNumParticles());
}

double CommonIntegrateVariableVerletStepKernel::execute(ContextImpl& context, const VariableVerletIntegrator& integrator, double maxTime) {
    cc.setAsCurrent();
    IntegrationUtilities& integration = cc.getIntegrationUtilities();
    int numAtoms = cc.getNumAtoms();
    int paddedNumAtoms = cc.getPaddedNumAtoms();
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        kernel1->addArg(numAtoms);
        kernel1->addArg(paddedNumAtoms);
        kernel1->addArg(integration.getStepSize());
        kernel1->addArg(cc.getPosq());
        kernel1->addArg(cc.getVelm());
        kernel1->addArg(cc.getLongForceBuffer());
        kernel1->addArg(integration.getPosDelta());
        if (cc.getUseMixedPrecision())
            kernel1->addArg(cc.getPosqCorrection());
        kernel2->addArg(numAtoms);
        kernel2->addArg(integration.getStepSize());
        kernel2->addArg(cc.getPosq());
        kernel2->addArg(cc.getVelm());
        kernel2->addArg(integration.getPosDelta());
        if (cc.getUseMixedPrecision())
            kernel2->addArg(cc.getPosqCorrection());
        selectSizeKernel->addArg(numAtoms);
        selectSizeKernel->addArg(paddedNumAtoms);
        selectSizeKernel->addArg();
        selectSizeKernel->addArg();
        selectSizeKernel->addArg(cc.getIntegrationUtilities().getStepSize());
        selectSizeKernel->addArg(cc.getVelm());
        selectSizeKernel->addArg(cc.getLongForceBuffer());
    }

    // Select the step size to use.

    bool useDouble = cc.getUseDoublePrecision() || cc.getUseMixedPrecision();
    double maxStepSize = maxTime-cc.getTime();
    if (integrator.getMaximumStepSize() > 0)
        maxStepSize = min(integrator.getMaximumStepSize(), maxStepSize);
    float maxStepSizeFloat = (float) maxStepSize;
    if (useDouble) {
        selectSizeKernel->setArg(2, maxStepSize);
        selectSizeKernel->setArg(3, integrator.getErrorTolerance());
    }
    else {
        selectSizeKernel->setArg(2, maxStepSizeFloat);
        selectSizeKernel->setArg(3, (float) integrator.getErrorTolerance());
    }
    selectSizeKernel->execute(blockSize, blockSize);

    // Call the first integration kernel.

    kernel1->execute(numAtoms);

    // Apply constraints.

    integration.applyConstraints(integrator.getConstraintTolerance());

    // Call the second integration kernel.

    kernel2->execute(numAtoms);
    integration.computeVirtualSites();

    // Reduce UI lag.

#ifdef WIN32
    cc.flushQueue();
#endif

    // Update the time and step count.

    double dt = cc.getIntegrationUtilities().getLastStepSize();
    double time = cc.getTime()+dt;
    if (useDouble) {
        if (dt == maxStepSize)
            time = maxTime; // Avoid round-off error
    }
    else {
        if (dt == maxStepSizeFloat)
            time = maxTime; // Avoid round-off error
    }
    cc.setTime(time);
    cc.setStepCount(cc.getStepCount()+1);
    cc.reorderAtoms();
    return dt;
}

double CommonIntegrateVariableVerletStepKernel::computeKineticEnergy(ContextImpl& context, const VariableVerletIntegrator& integrator) {
    return cc.getIntegrationUtilities().computeKineticEnergy(0.5*integrator.getStepSize());
}

void CommonIntegrateVariableLangevinStepKernel::initialize(const System& system, const VariableLangevinIntegrator& integrator) {
    cc.initializeContexts();
    cc.setAsCurrent();
    cc.getIntegrationUtilities().initRandomNumberGenerator(integrator.getRandomNumberSeed());
    ComputeProgram program = cc.compileProgram(CommonKernelSources::langevin);
    kernel1 = program->createKernel("integrateLangevinPart1");
    kernel2 = program->createKernel("integrateLangevinPart2");
    selectSizeKernel = program->createKernel("selectLangevinStepSize");
    params.initialize(cc, 3, cc.getUseDoublePrecision() || cc.getUseMixedPrecision() ? sizeof(double) : sizeof(float), "langevinParams");
    blockSize = min(256, system.getNumParticles());
    blockSize = max(blockSize, params.getSize());
}

double CommonIntegrateVariableLangevinStepKernel::execute(ContextImpl& context, const VariableLangevinIntegrator& integrator, double maxTime) {
    cc.setAsCurrent();
    IntegrationUtilities& integration = cc.getIntegrationUtilities();
    int numAtoms = cc.getNumAtoms();
    int paddedNumAtoms = cc.getPaddedNumAtoms();
    bool useDouble = cc.getUseDoublePrecision() || cc.getUseMixedPrecision();
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        kernel1->addArg(numAtoms);
        kernel1->addArg(paddedNumAtoms);
        kernel1->addArg(cc.getVelm());
        kernel1->addArg(cc.getLongForceBuffer());
        kernel1->addArg(integration.getPosDelta());
        kernel1->addArg(params);
        kernel1->addArg(integration.getStepSize());
        kernel1->addArg(integration.getRandom());
        kernel1->addArg();
        kernel2->addArg(numAtoms);
        kernel2->addArg(cc.getPosq());
        kernel2->addArg(integration.getPosDelta());
        kernel2->addArg(cc.getVelm());
        kernel2->addArg(integration.getStepSize());
        if (cc.getUseMixedPrecision())
            kernel2->addArg(cc.getPosqCorrection());
        selectSizeKernel->addArg(numAtoms);
        selectSizeKernel->addArg(paddedNumAtoms);
        for (int i = 0; i < 4; i++)
            selectSizeKernel->addArg();
        selectSizeKernel->addArg(integration.getStepSize());
        selectSizeKernel->addArg(cc.getVelm());
        selectSizeKernel->addArg(cc.getLongForceBuffer());
        selectSizeKernel->addArg(params);
    }

    // Select the step size to use.

    double maxStepSize = maxTime-cc.getTime();
    if (integrator.getMaximumStepSize() > 0)
        maxStepSize = min(integrator.getMaximumStepSize(), maxStepSize);
    float maxStepSizeFloat = (float) maxStepSize;
    if (useDouble) {
        selectSizeKernel->setArg(2, maxStepSize);
        selectSizeKernel->setArg(3, integrator.getErrorTolerance());
        selectSizeKernel->setArg(4, integrator.getFriction());
        selectSizeKernel->setArg(5, BOLTZ*integrator.getTemperature());
    }
    else {
        selectSizeKernel->setArg(2, maxStepSizeFloat);
        selectSizeKernel->setArg(3, (float) integrator.getErrorTolerance());
        selectSizeKernel->setArg(4, (float) integrator.getFriction());
        selectSizeKernel->setArg(5, (float) (BOLTZ*integrator.getTemperature()));
    }
    selectSizeKernel->execute(blockSize, blockSize);

    // Call the first integration kernel.

    kernel1->setArg(8, integration.prepareRandomNumbers(cc.getPaddedNumAtoms()));
    kernel1->execute(numAtoms);

    // Apply constraints.

    integration.applyConstraints(integrator.getConstraintTolerance());

    // Call the second integration kernel.

    kernel2->execute(numAtoms);
    integration.computeVirtualSites();

    // Reduce UI lag.

#ifdef WIN32
    cc.flushQueue();
#endif

    // Update the time and step count.

    double dt = cc.getIntegrationUtilities().getLastStepSize();
    double time = cc.getTime()+dt;
    if (useDouble) {
        if (dt == maxStepSize)
            time = maxTime; // Avoid round-off error
    }
    else {
        if (dt == maxStepSizeFloat)
            time = maxTime; // Avoid round-off error
    }
    cc.setTime(time);
    cc.setStepCount(cc.getStepCount()+1);
    cc.reorderAtoms();
    return dt;
}

double CommonIntegrateVariableLangevinStepKernel::computeKineticEnergy(ContextImpl& context, const VariableLangevinIntegrator& integrator) {
    return cc.getIntegrationUtilities().computeKineticEnergy(0.5*integrator.getStepSize());
}

class CommonIntegrateCustomStepKernel::ReorderListener : public ComputeContext::ReorderListener {
public:
    ReorderListener(ComputeContext& cc, vector<ComputeArray>& perDofValues, vector<vector<mm_float4> >& localPerDofValuesFloat, vector<vector<mm_double4> >& localPerDofValuesDouble, vector<bool>& deviceValuesAreCurrent) :
            cc(cc), perDofValues(perDofValues), localPerDofValuesFloat(localPerDofValuesFloat), localPerDofValuesDouble(localPerDofValuesDouble), deviceValuesAreCurrent(deviceValuesAreCurrent) {
        int numAtoms = cc.getNumAtoms();
        lastAtomOrder.resize(numAtoms);
        for (int i = 0; i < numAtoms; i++)
            lastAtomOrder[i] = cc.getAtomIndex()[i];
    }
    void execute() {
        // Reorder the per-DOF variables to reflect the new atom order.

        if (perDofValues.size() == 0)
            return;
        int numAtoms = cc.getNumAtoms();
        const vector<int>& order = cc.getAtomIndex();
        for (int index = 0; index < perDofValues.size(); index++) {
            if (cc.getUseDoublePrecision() || cc.getUseMixedPrecision()) {
                if (deviceValuesAreCurrent[index])
                    perDofValues[index].download(localPerDofValuesDouble[index]);
                vector<mm_double4> swap(numAtoms);
                for (int i = 0; i < numAtoms; i++)
                    swap[lastAtomOrder[i]] = localPerDofValuesDouble[index][i];
                for (int i = 0; i < numAtoms; i++)
                    localPerDofValuesDouble[index][i] = swap[order[i]];
                perDofValues[index].upload(localPerDofValuesDouble[index]);
            }
            else {
                if (deviceValuesAreCurrent[index])
                    perDofValues[index].download(localPerDofValuesFloat[index]);
                vector<mm_float4> swap(numAtoms);
                for (int i = 0; i < numAtoms; i++)
                    swap[lastAtomOrder[i]] = localPerDofValuesFloat[index][i];
                for (int i = 0; i < numAtoms; i++)
                    localPerDofValuesFloat[index][i] = swap[order[i]];
                perDofValues[index].upload(localPerDofValuesFloat[index]);
            }
            deviceValuesAreCurrent[index] = true;
        }
        for (int i = 0; i < numAtoms; i++)
            lastAtomOrder[i] = order[i];
    }
private:
    ComputeContext& cc;
    vector<ComputeArray>& perDofValues;
    vector<vector<mm_float4> >& localPerDofValuesFloat;
    vector<vector<mm_double4> >& localPerDofValuesDouble;
    vector<bool>& deviceValuesAreCurrent;
    vector<int> lastAtomOrder;
};

class CommonIntegrateCustomStepKernel::DerivFunction : public CustomFunction {
public:
    DerivFunction(map<string, double>& energyParamDerivs, const string& param) : energyParamDerivs(energyParamDerivs), param(param) {
    }
    int getNumArguments() const {
        return 0;
    }
    double evaluate(const double* arguments) const {
        return energyParamDerivs[param];
    }
    double evaluateDerivative(const double* arguments, const int* derivOrder) const {
        return 0;
    }
    CustomFunction* clone() const {
        return new DerivFunction(energyParamDerivs, param);
    }
private:
    map<string, double>& energyParamDerivs;
    string param;
};

void CommonIntegrateCustomStepKernel::initialize(const System& system, const CustomIntegrator& integrator) {
    cc.initializeContexts();
    cc.setAsCurrent();
    cc.getIntegrationUtilities().initRandomNumberGenerator(integrator.getRandomNumberSeed());
    numGlobalVariables = integrator.getNumGlobalVariables();
    int elementSize = (cc.getUseDoublePrecision() || cc.getUseMixedPrecision() ? sizeof(double) : sizeof(float));
    sumBuffer.initialize(cc, system.getNumParticles(), elementSize, "sumBuffer");
    summedValue.initialize(cc, 1, elementSize, "summedValue");
    perDofValues.resize(integrator.getNumPerDofVariables());
    localPerDofValuesFloat.resize(perDofValues.size());
    localPerDofValuesDouble.resize(perDofValues.size());
    for (int i = 0; i < perDofValues.size(); i++)
        perDofValues[i].initialize(cc, system.getNumParticles(), 4*elementSize, "perDofVariables");
    localValuesAreCurrent.resize(integrator.getNumPerDofVariables(), false);
    deviceValuesAreCurrent.resize(integrator.getNumPerDofVariables(), false);
    cc.addReorderListener(new ReorderListener(cc, perDofValues, localPerDofValuesFloat, localPerDofValuesDouble, deviceValuesAreCurrent));
    SimTKOpenMMUtilities::setRandomNumberSeed(integrator.getRandomNumberSeed());
}

string CommonIntegrateCustomStepKernel::createPerDofComputation(const string& variable, const Lepton::ParsedExpression& expr, CustomIntegrator& integrator,
        const string& forceName, const string& energyName, vector<const TabulatedFunction*>& functions, vector<pair<string, string> >& functionNames) {
    string tempType = (cc.getSupportsDoublePrecision() ? "double3" : "float3");
    map<string, Lepton::ParsedExpression> expressions;
    expressions[tempType+" tempResult = "] = expr;
    map<string, string> variables;
    variables["x"] = "make_"+tempType+"(position.x, position.y, position.z)";
    variables["v"] = "make_"+tempType+"(velocity.x, velocity.y, velocity.z)";
    variables[forceName] = "make_"+tempType+"(f.x, f.y, f.z)";
    variables["gaussian"] = "make_"+tempType+"(gaussian.x, gaussian.y, gaussian.z)";
    variables["uniform"] = "make_"+tempType+"(uniform.x, uniform.y, uniform.z)";
    variables["m"] = "mass";
    variables["dt"] = "stepSize";
    if (energyName != "")
        variables[energyName] = "make_"+tempType+"(energy)";
    for (int i = 0; i < integrator.getNumGlobalVariables(); i++)
        variables[integrator.getGlobalVariableName(i)] = "make_"+tempType+"(globals["+cc.intToString(globalVariableIndex[i])+"])";
    for (int i = 0; i < integrator.getNumPerDofVariables(); i++)
        variables[integrator.getPerDofVariableName(i)] = "convertToTempType3(perDof"+cc.intToString(i)+")";
    for (int i = 0; i < (int) parameterNames.size(); i++)
        variables[parameterNames[i]] = "make_"+tempType+"(globals["+cc.intToString(parameterVariableIndex[i])+"])";
    vector<pair<ExpressionTreeNode, string> > variableNodes;
    findExpressionsForDerivs(expr.getRootNode(), variableNodes);
    for (auto& var : variables)
        variableNodes.push_back(make_pair(ExpressionTreeNode(new Operation::Variable(var.first)), var.second));
    string result = cc.getExpressionUtilities().createExpressions(expressions, variableNodes, functions, functionNames, "temp", tempType);
    if (variable == "x")
        result += "position.x = tempResult.x; position.y = tempResult.y; position.z = tempResult.z;\n";
    else if (variable == "v")
        result += "velocity.x = tempResult.x; velocity.y = tempResult.y; velocity.z = tempResult.z;\n";
    else if (variable == "")
        result += "sum[index] = tempResult.x+tempResult.y+tempResult.z;\n";
    else {
        for (int i = 0; i < integrator.getNumPerDofVariables(); i++)
            if (variable == integrator.getPerDofVariableName(i)) {
                string varName = "perDof"+cc.intToString(i);
                result += varName+".x = tempResult.x; "+varName+".y = tempResult.y; "+varName+".z = tempResult.z;\n";
            }
    }
    return result;
}

void CommonIntegrateCustomStepKernel::prepareForComputation(ContextImpl& context, CustomIntegrator& integrator, bool& forcesAreValid) {
    cc.setAsCurrent();
    IntegrationUtilities& integration = cc.getIntegrationUtilities();
    int numAtoms = cc.getNumAtoms();
    int numSteps = integrator.getNumComputations();
    bool useDouble = cc.getUseDoublePrecision() || cc.getUseMixedPrecision();
    string tempType = (cc.getSupportsDoublePrecision() ? "double3" : "float3");
    string perDofType = (useDouble ? "double4" : "float4");
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;

        // Initialize various data structures.

        const map<string, double>& params = context.getParameters();
        for (auto& param : params)
            parameterNames.push_back(param.first);
        kernels.resize(integrator.getNumComputations());
        requiredGaussian.resize(integrator.getNumComputations(), 0);
        requiredUniform.resize(integrator.getNumComputations(), 0);
        needsGlobals.resize(numSteps, false);
        globalExpressions.resize(numSteps);
        stepType.resize(numSteps);
        stepTarget.resize(numSteps);
        merged.resize(numSteps, false);
        modifiesParameters = false;
        sumWorkGroupSize = cc.getMaxThreadBlockSize();
        if (sumWorkGroupSize > 512)
            sumWorkGroupSize = 512;
        map<string, string> defines;
        defines["NUM_ATOMS"] = cc.intToString(cc.getNumAtoms());
        defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
        defines["WORK_GROUP_SIZE"] = cc.intToString(sumWorkGroupSize);

        // Record the tabulated functions.

        map<string, Lepton::CustomFunction*> functions;
        vector<pair<string, string> > functionNames;
        vector<const TabulatedFunction*> functionList;
        vector<string> tableTypes;
        tabulatedFunctions.resize(integrator.getNumTabulatedFunctions());
        for (int i = 0; i < integrator.getNumTabulatedFunctions(); i++) {
            functionList.push_back(&integrator.getTabulatedFunction(i));
            string name = integrator.getTabulatedFunctionName(i);
            string arrayName = "table"+cc.intToString(i);
            functionNames.push_back(make_pair(name, arrayName));
            functions[name] = createReferenceTabulatedFunction(integrator.getTabulatedFunction(i));
            int width;
            vector<float> f = cc.getExpressionUtilities().computeFunctionCoefficients(integrator.getTabulatedFunction(i), width);
            tabulatedFunctions[i].initialize<float>(cc, f.size(), "TabulatedFunction");
            tabulatedFunctions[i].upload(f);
            if (width == 1)
                tableTypes.push_back("float");
            else
                tableTypes.push_back("float"+cc.intToString(width));
        }

        // Record information about all the computation steps.

        vector<string> variable(numSteps);
        vector<int> forceGroup;
        vector<vector<Lepton::ParsedExpression> > expression;
        CustomIntegratorUtilities::analyzeComputations(context, integrator, expression, comparisons, blockEnd, invalidatesForces, needsForces, needsEnergy, computeBothForceAndEnergy, forceGroup, functions);
        for (int step = 0; step < numSteps; step++) {
            string expr;
            integrator.getComputationStep(step, stepType[step], variable[step], expr);
            if (stepType[step] == CustomIntegrator::WhileBlockStart)
                blockEnd[blockEnd[step]] = step; // Record where to branch back to.
            if (stepType[step] == CustomIntegrator::ComputeGlobal || stepType[step] == CustomIntegrator::IfBlockStart || stepType[step] == CustomIntegrator::WhileBlockStart)
                for (auto& expr : expression[step])
                    globalExpressions[step].push_back(ParsedExpression(replaceDerivFunctions(expr.getRootNode(), context)).createCompiledExpression());
        }
        for (int step = 0; step < numSteps; step++) {
            for (auto& expr : globalExpressions[step])
                expressionSet.registerExpression(expr);
        }

        // Record the indices for variables in the CompiledExpressionSet.

        gaussianVariableIndex = expressionSet.getVariableIndex("gaussian");
        uniformVariableIndex = expressionSet.getVariableIndex("uniform");
        dtVariableIndex = expressionSet.getVariableIndex("dt");
        for (int i = 0; i < integrator.getNumGlobalVariables(); i++)
            globalVariableIndex.push_back(expressionSet.getVariableIndex(integrator.getGlobalVariableName(i)));
        for (auto& name : parameterNames)
            parameterVariableIndex.push_back(expressionSet.getVariableIndex(name));

        // Record the variable names and flags for the force and energy in each step.

        forceGroupFlags.resize(numSteps, integrator.getIntegrationForceGroups());
        vector<string> forceGroupName;
        vector<string> energyGroupName;
        for (int i = 0; i < 32; i++) {
            stringstream fname;
            fname << "f" << i;
            forceGroupName.push_back(fname.str());
            stringstream ename;
            ename << "energy" << i;
            energyGroupName.push_back(ename.str());
        }
        vector<string> forceName(numSteps, "f");
        vector<string> energyName(numSteps, "energy");
        stepEnergyVariableIndex.resize(numSteps, expressionSet.getVariableIndex("energy"));
        for (int step = 0; step < numSteps; step++) {
            if (needsForces[step] && forceGroup[step] > -1)
                forceName[step] = forceGroupName[forceGroup[step]];
            if (needsEnergy[step] && forceGroup[step] > -1) {
                energyName[step] = energyGroupName[forceGroup[step]];
                stepEnergyVariableIndex[step] = expressionSet.getVariableIndex(energyName[step]);
            }
            if (forceGroup[step] > -1)
                forceGroupFlags[step] = 1<<forceGroup[step];
            if (forceGroupFlags[step] == -2 && step > 0)
                forceGroupFlags[step] = forceGroupFlags[step-1];
            if (forceGroupFlags[step] != -2 && savedForces.find(forceGroupFlags[step]) == savedForces.end()) {
                savedForces[forceGroupFlags[step]] = ComputeArray();
                savedForces[forceGroupFlags[step]].initialize(cc, cc.getLongForceBuffer().getSize(), cc.getLongForceBuffer().getElementSize(), "savedForces");
            }
        }

        // Allocate space for storing global values, both on the host and the device.

        localGlobalValues.resize(expressionSet.getNumVariables());
        int elementSize = (cc.getUseDoublePrecision() || cc.getUseMixedPrecision() ? sizeof(double) : sizeof(float));
        globalValues.initialize(cc, expressionSet.getNumVariables(), elementSize, "globalValues");
        for (int i = 0; i < integrator.getNumGlobalVariables(); i++) {
            localGlobalValues[globalVariableIndex[i]] = initialGlobalVariables[i];
            expressionSet.setVariable(globalVariableIndex[i], initialGlobalVariables[i]);
        }
        for (int i = 0; i < (int) parameterVariableIndex.size(); i++) {
            double value = context.getParameter(parameterNames[i]);
            localGlobalValues[parameterVariableIndex[i]] = value;
            expressionSet.setVariable(parameterVariableIndex[i], value);
        }
        int numContextParams = context.getParameters().size();
        localPerDofEnergyParamDerivs.resize(numContextParams);
        perDofEnergyParamDerivs.initialize(cc, max(1, numContextParams), elementSize, "perDofEnergyParamDerivs");

        // Record information about the targets of steps that will be stored in global variables.

        for (int step = 0; step < numSteps; step++) {
            if (stepType[step] == CustomIntegrator::ComputeGlobal || stepType[step] == CustomIntegrator::ComputeSum) {
                if (variable[step] == "dt")
                    stepTarget[step].type = DT;
                for (int i = 0; i < integrator.getNumGlobalVariables(); i++)
                    if (variable[step] == integrator.getGlobalVariableName(i))
                        stepTarget[step].type = VARIABLE;
                for (auto& name : parameterNames)
                    if (variable[step] == name) {
                        stepTarget[step].type = PARAMETER;
                        modifiesParameters = true;
                    }
                stepTarget[step].variableIndex = expressionSet.getVariableIndex(variable[step]);
            }
        }

        // Identify which per-DOF steps are going to require global variables or context parameters.

        for (int step = 0; step < numSteps; step++) {
            if (stepType[step] == CustomIntegrator::ComputePerDof || stepType[step] == CustomIntegrator::ComputeSum) {
                for (int i = 0; i < integrator.getNumGlobalVariables(); i++)
                    if (usesVariable(expression[step][0], integrator.getGlobalVariableName(i)))
                        needsGlobals[step] = true;
                for (auto& name : parameterNames)
                    if (usesVariable(expression[step][0], name))
                        needsGlobals[step] = true;
            }
        }

        // Determine how each step will represent the position (as just a value, or a value plus a delta).

        hasAnyConstraints = (context.getSystem().getNumConstraints() > 0);
        vector<bool> storePosAsDelta(numSteps, false);
        vector<bool> loadPosAsDelta(numSteps, false);
        if (hasAnyConstraints) {
            bool beforeConstrain = false;
            for (int step = numSteps-1; step >= 0; step--) {
                if (stepType[step] == CustomIntegrator::ConstrainPositions)
                    beforeConstrain = true;
                else if (stepType[step] == CustomIntegrator::ComputePerDof && variable[step] == "x" && beforeConstrain)
                    storePosAsDelta[step] = true;
            }
            bool storedAsDelta = false;
            for (int step = 0; step < numSteps; step++) {
                loadPosAsDelta[step] = storedAsDelta;
                if (storePosAsDelta[step] == true)
                    storedAsDelta = true;
                if (stepType[step] == CustomIntegrator::ConstrainPositions)
                    storedAsDelta = false;
            }
        }

        // Identify steps that can be merged into a single kernel.

        for (int step = 1; step < numSteps; step++) {
            if (invalidatesForces[step-1] || forceGroupFlags[step] != forceGroupFlags[step-1])
                continue;
            if (stepType[step-1] == CustomIntegrator::ComputePerDof && stepType[step] == CustomIntegrator::ComputePerDof)
                merged[step] = true;
        }
        for (int step = numSteps-1; step > 0; step--)
            if (merged[step]) {
                needsForces[step-1] = (needsForces[step] || needsForces[step-1]);
                needsEnergy[step-1] = (needsEnergy[step] || needsEnergy[step-1]);
                needsGlobals[step-1] = (needsGlobals[step] || needsGlobals[step-1]);
                computeBothForceAndEnergy[step-1] = (computeBothForceAndEnergy[step] || computeBothForceAndEnergy[step-1]);
            }

        // Loop over all steps and create the kernels for them.

        for (int step = 0; step < numSteps; step++) {
            if ((stepType[step] == CustomIntegrator::ComputePerDof || stepType[step] == CustomIntegrator::ComputeSum) && !merged[step]) {
                // Compute a per-DOF value.

                stringstream compute;
                for (int i = 0; i < perDofValues.size(); i++)
                    compute << tempType<<" perDof"<<cc.intToString(i)<<" = convertToTempType3(perDofValues"<<cc.intToString(i)<<"[index]);\n";
                int numGaussian = 0, numUniform = 0;
                for (int j = step; j < numSteps && (j == step || merged[j]); j++) {
                    numGaussian += numAtoms*usesVariable(expression[j][0], "gaussian");
                    numUniform += numAtoms*usesVariable(expression[j][0], "uniform");
                    compute << "{\n";
                    if (numGaussian > 0)
                        compute << "float4 gaussian = gaussianValues[gaussianIndex+index];\n";
                    if (numUniform > 0)
                        compute << "float4 uniform = uniformValues[uniformIndex+index];\n";
                    compute << createPerDofComputation(stepType[j] == CustomIntegrator::ComputePerDof ? variable[j] : "", expression[j][0], integrator, forceName[j], energyName[j], functionList, functionNames);
                    if (variable[j] == "x") {
                        if (storePosAsDelta[j]) {
                            if (cc.getSupportsDoublePrecision())
                                compute << "posDelta[index] = convertFromDouble4(position-loadPos(posq, posqCorrection, index));\n";
                            else
                                compute << "posDelta[index] = position-posq[index];\n";
                        }
                        else
                            compute << "storePos(posq, posqCorrection, index, position);\n";
                    }
                    else if (variable[j] == "v") {
                        if (cc.getSupportsDoublePrecision())
                            compute << "velm[index] = convertFromDouble4(velocity);\n";
                        else
                            compute << "velm[index] = velocity;\n";
                    }
                    else {
                        for (int i = 0; i < perDofValues.size(); i++)
                            compute << "perDofValues"<<cc.intToString(i)<<"[index] = make_"<<perDofType<<"(perDof"<<cc.intToString(i)<<".x, perDof"<<cc.intToString(i)<<".y, perDof"<<cc.intToString(i)<<".z, 0);\n";
                    }
                    if (numGaussian > 0)
                        compute << "gaussianIndex += NUM_ATOMS;\n";
                    if (numUniform > 0)
                        compute << "uniformIndex += NUM_ATOMS;\n";
                    compute << "}\n";
                }
                map<string, string> replacements;
                replacements["COMPUTE_STEP"] = compute.str();
                stringstream args;
                for (int i = 0; i < perDofValues.size(); i++) {
                    string valueName = "perDofValues"+cc.intToString(i);
                    args << ", GLOBAL " << perDofType << "* RESTRICT " << valueName;
                }
                for (int i = 0; i < (int) tableTypes.size(); i++)
                    args << ", GLOBAL const " << tableTypes[i]<< "* RESTRICT table" << i;
                replacements["PARAMETER_ARGUMENTS"] = args.str();
                if (loadPosAsDelta[step])
                    defines["LOAD_POS_AS_DELTA"] = "1";
                else if (defines.find("LOAD_POS_AS_DELTA") != defines.end())
                    defines.erase("LOAD_POS_AS_DELTA");
                ComputeProgram program = cc.compileProgram(cc.replaceStrings(CommonKernelSources::customIntegratorPerDof, replacements), defines);
                ComputeKernel kernel = program->createKernel("computePerDof");
                kernels[step].push_back(kernel);
                requiredGaussian[step] = numGaussian;
                requiredUniform[step] = numUniform;
                kernel->addArg(cc.getPosq());
                if (cc.getUseMixedPrecision())
                    kernel->addArg(cc.getPosqCorrection());
                else
                    kernel->addArg(nullptr);
                kernel->addArg(integration.getPosDelta());
                kernel->addArg(cc.getVelm());
                kernel->addArg(cc.getLongForceBuffer());
                kernel->addArg(integration.getStepSize());
                kernel->addArg(globalValues);
                kernel->addArg(sumBuffer);
                for (int i = 0; i < 4; i++)
                    kernel->addArg();
                kernel->addArg(perDofEnergyParamDerivs);
                for (auto& array : perDofValues)
                    kernel->addArg(array);
                for (auto& array : tabulatedFunctions)
                    kernel->addArg(array);
                if (stepType[step] == CustomIntegrator::ComputeSum) {
                    // Create a second kernel for this step that sums the values.

                    program = cc.compileProgram(CommonKernelSources::customIntegrator, defines);
                    kernel = program->createKernel(useDouble ? "computeDoubleSum" : "computeFloatSum");
                    kernels[step].push_back(kernel);
                    kernel->addArg(sumBuffer);
                    kernel->addArg(summedValue);
                    kernel->addArg(numAtoms);
                }
            }
            else if (stepType[step] == CustomIntegrator::ConstrainPositions) {
                // Apply position constraints.

                ComputeProgram program = cc.compileProgram(CommonKernelSources::customIntegrator, defines);
                ComputeKernel kernel = program->createKernel("applyPositionDeltas");
                kernels[step].push_back(kernel);
                kernel->addArg(cc.getPosq());
                if (cc.getUseMixedPrecision())
                    kernel->addArg(cc.getPosqCorrection());
                else
                    kernel->addArg(nullptr);
                kernel->addArg(integration.getPosDelta());
            }
        }

        // Initialize the random number generator.

        int maxUniformRandoms = 1;
        for (int required : requiredUniform)
            maxUniformRandoms = max(maxUniformRandoms, required);
        uniformRandoms.initialize<mm_float4>(cc, maxUniformRandoms, "uniformRandoms");
        randomSeed.initialize<mm_int4>(cc, cc.getNumThreadBlocks()*cc.getSIMDWidth(), "randomSeed");
        vector<mm_int4> seed(randomSeed.getSize());
        int rseed = integrator.getRandomNumberSeed();
        // A random seed of 0 means use a unique one
        if (rseed == 0)
            rseed = osrngseed();
        unsigned int r = (unsigned int) (rseed+1);
        for (auto& s : seed) {
            s.x = r = (1664525*r + 1013904223) & 0xFFFFFFFF;
            s.y = r = (1664525*r + 1013904223) & 0xFFFFFFFF;
            s.z = r = (1664525*r + 1013904223) & 0xFFFFFFFF;
            s.w = r = (1664525*r + 1013904223) & 0xFFFFFFFF;
        }
        randomSeed.upload(seed);
        ComputeProgram randomProgram = cc.compileProgram(CommonKernelSources::customIntegrator, defines);
        randomKernel = randomProgram->createKernel("generateRandomNumbers");
        randomKernel->addArg(maxUniformRandoms);
        randomKernel->addArg(uniformRandoms);
        randomKernel->addArg(randomSeed);

        // Create the kernel for computing kinetic energy.

        stringstream computeKE;
        for (int i = 0; i < perDofValues.size(); i++)
            computeKE << tempType<<" perDof"<<cc.intToString(i)<<" = convertToTempType3(perDofValues"<<cc.intToString(i)<<"[index]);\n";
        Lepton::ParsedExpression keExpression = Lepton::Parser::parse(integrator.getKineticEnergyExpression()).optimize();
        computeKE << createPerDofComputation("", keExpression, integrator, "f", "", functionList, functionNames);
        map<string, string> replacements;
        replacements["COMPUTE_STEP"] = computeKE.str();
        stringstream args;
        for (int i = 0; i < perDofValues.size(); i++) {
            string valueName = "perDofValues"+cc.intToString(i);
            args << ", GLOBAL " << perDofType << "* RESTRICT " << valueName;
        }
        for (int i = 0; i < (int) tableTypes.size(); i++)
            args << ", GLOBAL const " << tableTypes[i]<< "* RESTRICT table" << i;
        replacements["PARAMETER_ARGUMENTS"] = args.str();
        if (defines.find("LOAD_POS_AS_DELTA") != defines.end())
            defines.erase("LOAD_POS_AS_DELTA");
        ComputeProgram program = cc.compileProgram(cc.replaceStrings(CommonKernelSources::customIntegratorPerDof, replacements), defines);
        kineticEnergyKernel = program->createKernel("computePerDof");
        kineticEnergyKernel->addArg(cc.getPosq());
        if (cc.getUseMixedPrecision())
            kineticEnergyKernel->addArg(cc.getPosqCorrection());
        else
            kineticEnergyKernel->addArg(nullptr);
        kineticEnergyKernel->addArg(integration.getPosDelta());
        kineticEnergyKernel->addArg(cc.getVelm());
        kineticEnergyKernel->addArg(cc.getLongForceBuffer());
        kineticEnergyKernel->addArg(integration.getStepSize());
        kineticEnergyKernel->addArg(globalValues);
        kineticEnergyKernel->addArg(sumBuffer);
        kineticEnergyKernel->addArg();
        kineticEnergyKernel->addArg();
        kineticEnergyKernel->addArg(uniformRandoms);
        if (cc.getUseDoublePrecision() || cc.getUseMixedPrecision())
            kineticEnergyKernel->addArg(0.0);
        else
            kineticEnergyKernel->addArg(0.0f);
        kineticEnergyKernel->addArg(perDofEnergyParamDerivs);
        for (auto& array : perDofValues)
            kineticEnergyKernel->addArg(array);
        for (auto& array : tabulatedFunctions)
            kineticEnergyKernel->addArg(array);

        // Create a second kernel to sum the values.

        program = cc.compileProgram(CommonKernelSources::customIntegrator, defines);
        sumKineticEnergyKernel = program->createKernel(useDouble ? "computeDoubleSum" : "computeFloatSum");
        sumKineticEnergyKernel->addArg(sumBuffer);
        sumKineticEnergyKernel->addArg(summedValue);
        sumKineticEnergyKernel->addArg(numAtoms);

        // Delete the custom functions.

        for (auto& function : functions)
            delete function.second;
    }

    // Make sure all values (variables, parameters, etc.) are up to date.

    for (int i = 0; i < perDofValues.size(); i++) {
        if (!deviceValuesAreCurrent[i]) {
            if (useDouble)
                perDofValues[i].upload(localPerDofValuesDouble[i]);
            else
                perDofValues[i].upload(localPerDofValuesFloat[i]);
            deviceValuesAreCurrent[i] = true;
        }
        localValuesAreCurrent[i] = false;
    }
    double stepSize = integrator.getStepSize();
    recordGlobalValue(stepSize, GlobalTarget(DT, dtVariableIndex), integrator);
    for (int i = 0; i < (int) parameterNames.size(); i++) {
        double value = context.getParameter(parameterNames[i]);
        if (value != localGlobalValues[parameterVariableIndex[i]]) {
            localGlobalValues[parameterVariableIndex[i]] = value;
            deviceGlobalsAreCurrent = false;
        }
    }
}

ExpressionTreeNode CommonIntegrateCustomStepKernel::replaceDerivFunctions(const ExpressionTreeNode& node, ContextImpl& context) {
    // This is called recursively to identify calls to the deriv() function inside global expressions,
    // and replace them with a custom function that returns the correct value.

    const Operation& op = node.getOperation();
    if (op.getId() == Operation::CUSTOM && op.getName() == "deriv") {
        string param = node.getChildren()[1].getOperation().getName();
        if (context.getParameters().find(param) == context.getParameters().end())
            throw OpenMMException("The second argument to deriv() must be a context parameter");
        needsEnergyParamDerivs = true;
        return ExpressionTreeNode(new Operation::Custom("deriv", new DerivFunction(energyParamDerivs, param)));
    }
    else {
        vector<ExpressionTreeNode> children;
        for (auto& child : node.getChildren())
            children.push_back(replaceDerivFunctions(child, context));
        return ExpressionTreeNode(op.clone(), children);
    }
}

void CommonIntegrateCustomStepKernel::findExpressionsForDerivs(const ExpressionTreeNode& node, vector<pair<ExpressionTreeNode, string> >& variableNodes) {
    // This is called recursively to identify calls to the deriv() function inside per-DOF expressions,
    // and record the code to replace them with.

    const Operation& op = node.getOperation();
    if (op.getId() == Operation::CUSTOM && op.getName() == "deriv") {
        string param = node.getChildren()[1].getOperation().getName();
        int index;
        for (index = 0; index < perDofEnergyParamDerivNames.size() && param != perDofEnergyParamDerivNames[index]; index++)
            ;
        if (index == perDofEnergyParamDerivNames.size())
            perDofEnergyParamDerivNames.push_back(param);
        string tempType = (cc.getSupportsDoublePrecision() ? "double3" : "float3");
        variableNodes.push_back(make_pair(node, "make_"+tempType+"(energyParamDerivs["+cc.intToString(index)+"])"));
        needsEnergyParamDerivs = true;
    }
    else {
        for (auto& child : node.getChildren())
            findExpressionsForDerivs(child, variableNodes);
    }
}

void CommonIntegrateCustomStepKernel::execute(ContextImpl& context, CustomIntegrator& integrator, bool& forcesAreValid) {
    prepareForComputation(context, integrator, forcesAreValid);
    IntegrationUtilities& integration = cc.getIntegrationUtilities();
    int numAtoms = cc.getNumAtoms();
    int numSteps = integrator.getNumComputations();
    if (!forcesAreValid)
        savedEnergy.clear();

    // Loop over computation steps in the integrator and execute them.

    for (int step = 0; step < numSteps; ) {
        int nextStep = step+1;
        int forceGroups = forceGroupFlags[step];
        int lastForceGroups = context.getLastForceGroups();
        bool haveForces = (!needsForces[step] || (forcesAreValid && lastForceGroups == forceGroups));
        bool haveEnergy = (!needsEnergy[step] || savedEnergy.find(forceGroups) != savedEnergy.end());
        if (!haveForces || !haveEnergy) {
            if (forcesAreValid) {
                if (savedForces.find(lastForceGroups) != savedForces.end() && validSavedForces.find(lastForceGroups) == validSavedForces.end()) {
                    // The forces are still valid.  We just need a different force group right now.  Save the old
                    // forces in case we need them again.

                    cc.getLongForceBuffer().copyTo(savedForces[lastForceGroups]);
                    validSavedForces.insert(lastForceGroups);
                }
            }
            else
                validSavedForces.clear();

            // Recompute forces and/or energy.  Figure out what is actually needed
            // between now and the next time they get invalidated again.

            bool computeForce = (needsForces[step] || computeBothForceAndEnergy[step]);
            bool computeEnergy = (needsEnergy[step] || computeBothForceAndEnergy[step]);
            if (!computeEnergy && validSavedForces.find(forceGroups) != validSavedForces.end()) {
                // We can just restore the forces we saved earlier.

                savedForces[forceGroups].copyTo(cc.getLongForceBuffer());
                context.getLastForceGroups() = forceGroups;
            }
            else {
                recordChangedParameters(context);
                energy = context.calcForcesAndEnergy(computeForce, computeEnergy, forceGroups);
                savedEnergy[forceGroups] = energy;
                if (needsEnergyParamDerivs) {
                    context.getEnergyParameterDerivatives(energyParamDerivs);
                    if (perDofEnergyParamDerivNames.size() > 0) {
                        for (int i = 0; i < perDofEnergyParamDerivNames.size(); i++)
                            localPerDofEnergyParamDerivs[i] = energyParamDerivs[perDofEnergyParamDerivNames[i]];
                        perDofEnergyParamDerivs.upload(localPerDofEnergyParamDerivs, true);
                    }
                }
            }
            forcesAreValid = true;
        }
        if (needsEnergy[step])
            energy = savedEnergy[forceGroups];
        if (needsGlobals[step] && !deviceGlobalsAreCurrent) {
            // Upload the global values to the device.

            globalValues.upload(localGlobalValues, true);
            deviceGlobalsAreCurrent = true;
        }
        bool stepInvalidatesForces = invalidatesForces[step];
        if (stepType[step] == CustomIntegrator::ComputePerDof && !merged[step]) {
            kernels[step][0]->setArg(9, integration.prepareRandomNumbers(requiredGaussian[step]));
            kernels[step][0]->setArg(8, integration.getRandom());
            kernels[step][0]->setArg(10, uniformRandoms);
            if (cc.getUseDoublePrecision() || cc.getUseMixedPrecision())
                kernels[step][0]->setArg(11, energy);
            else
                kernels[step][0]->setArg(11, (float) energy);
            if (requiredUniform[step] > 0)
                randomKernel->execute(numAtoms, 2 * cc.getSIMDWidth());
            kernels[step][0]->execute(numAtoms, 4 * cc.getSIMDWidth());
        }
        else if (stepType[step] == CustomIntegrator::ComputeGlobal) {
            expressionSet.setVariable(uniformVariableIndex, SimTKOpenMMUtilities::getUniformlyDistributedRandomNumber());
            expressionSet.setVariable(gaussianVariableIndex, SimTKOpenMMUtilities::getNormallyDistributedRandomNumber());
            expressionSet.setVariable(stepEnergyVariableIndex[step], energy);
            recordGlobalValue(globalExpressions[step][0].evaluate(), stepTarget[step], integrator);
        }
        else if (stepType[step] == CustomIntegrator::ComputeSum) {
            kernels[step][0]->setArg(9, integration.prepareRandomNumbers(requiredGaussian[step]));
            kernels[step][0]->setArg(8, integration.getRandom());
            kernels[step][0]->setArg(10, uniformRandoms);
            if (cc.getUseDoublePrecision() || cc.getUseMixedPrecision())
                kernels[step][0]->setArg(11, energy);
            else
                kernels[step][0]->setArg(11, (float) energy);
            if (requiredUniform[step] > 0)
                randomKernel->execute(numAtoms, 2 * cc.getSIMDWidth());
            cc.clearBuffer(sumBuffer);
            kernels[step][0]->execute(numAtoms, 4 * cc.getSIMDWidth());
            kernels[step][1]->execute(sumWorkGroupSize, sumWorkGroupSize);
            if (cc.getUseDoublePrecision() || cc.getUseMixedPrecision()) {
                double value;
                summedValue.download(&value);
                recordGlobalValue(value, stepTarget[step], integrator);
            }
            else {
                float value;
                summedValue.download(&value);
                recordGlobalValue(value, stepTarget[step], integrator);
            }
        }
        else if (stepType[step] == CustomIntegrator::UpdateContextState) {
            recordChangedParameters(context);
            stepInvalidatesForces = context.updateContextState();
        }
        else if (stepType[step] == CustomIntegrator::ConstrainPositions) {
            if (hasAnyConstraints) {
                cc.getIntegrationUtilities().applyConstraints(integrator.getConstraintTolerance());
                kernels[step][0]->execute(numAtoms);
            }
            cc.getIntegrationUtilities().computeVirtualSites();
        }
        else if (stepType[step] == CustomIntegrator::ConstrainVelocities) {
            cc.getIntegrationUtilities().applyVelocityConstraints(integrator.getConstraintTolerance());
        }
        else if (stepType[step] == CustomIntegrator::IfBlockStart) {
            if (!evaluateCondition(step))
                nextStep = blockEnd[step]+1;
        }
        else if (stepType[step] == CustomIntegrator::WhileBlockStart) {
            if (!evaluateCondition(step))
                nextStep = blockEnd[step]+1;
        }
        else if (stepType[step] == CustomIntegrator::BlockEnd) {
            if (blockEnd[step] != -1)
                nextStep = blockEnd[step]; // Return to the start of a while block.
        }
        if (stepInvalidatesForces) {
            forcesAreValid = false;
            savedEnergy.clear();
        }
        step = nextStep;
    }
    recordChangedParameters(context);

    // Update the time and step count.

    cc.setTime(cc.getTime()+integrator.getStepSize());
    cc.setStepCount(cc.getStepCount()+1);
    cc.reorderAtoms();
    if (cc.getAtomsWereReordered()) {
        forcesAreValid = false;
        validSavedForces.clear();
    }

    // Reduce UI lag.

#ifdef WIN32
    cc.flushQueue();
#endif
}

bool CommonIntegrateCustomStepKernel::evaluateCondition(int step) {
    expressionSet.setVariable(uniformVariableIndex, SimTKOpenMMUtilities::getUniformlyDistributedRandomNumber());
    expressionSet.setVariable(gaussianVariableIndex, SimTKOpenMMUtilities::getNormallyDistributedRandomNumber());
    expressionSet.setVariable(stepEnergyVariableIndex[step], energy);
    double lhs = globalExpressions[step][0].evaluate();
    double rhs = globalExpressions[step][1].evaluate();
    switch (comparisons[step]) {
        case CustomIntegratorUtilities::EQUAL:
            return (lhs == rhs);
        case CustomIntegratorUtilities::LESS_THAN:
            return (lhs < rhs);
        case CustomIntegratorUtilities::GREATER_THAN:
            return (lhs > rhs);
        case CustomIntegratorUtilities::NOT_EQUAL:
            return (lhs != rhs);
        case CustomIntegratorUtilities::LESS_THAN_OR_EQUAL:
            return (lhs <= rhs);
        case CustomIntegratorUtilities::GREATER_THAN_OR_EQUAL:
            return (lhs >= rhs);
    }
    throw OpenMMException("Invalid comparison operator");
}

double CommonIntegrateCustomStepKernel::computeKineticEnergy(ContextImpl& context, CustomIntegrator& integrator, bool& forcesAreValid) {
    prepareForComputation(context, integrator, forcesAreValid);
    cc.clearBuffer(sumBuffer);
    kineticEnergyKernel->setArg(8, cc.getIntegrationUtilities().getRandom());
    kineticEnergyKernel->setArg(9, 0);
    kineticEnergyKernel->execute(cc.getNumAtoms());
    sumKineticEnergyKernel->execute(sumWorkGroupSize, sumWorkGroupSize);
    if (cc.getUseDoublePrecision() || cc.getUseMixedPrecision()) {
        double ke;
        summedValue.download(&ke);
        return ke;
    }
    else {
        float ke;
        summedValue.download(&ke);
        return ke;
    }
}

void CommonIntegrateCustomStepKernel::recordGlobalValue(double value, GlobalTarget target, CustomIntegrator& integrator) {
    switch (target.type) {
        case DT:
            if (value != localGlobalValues[dtVariableIndex])
                deviceGlobalsAreCurrent = false;
            expressionSet.setVariable(dtVariableIndex, value);
            localGlobalValues[dtVariableIndex] = value;
            cc.getIntegrationUtilities().setNextStepSize(value);
            integrator.setStepSize(value);
            break;
        case VARIABLE:
        case PARAMETER:
            expressionSet.setVariable(target.variableIndex, value);
            localGlobalValues[target.variableIndex] = value;
            deviceGlobalsAreCurrent = false;
            break;
    }
}

void CommonIntegrateCustomStepKernel::recordChangedParameters(ContextImpl& context) {
    if (!modifiesParameters)
        return;
    for (int i = 0; i < (int) parameterNames.size(); i++) {
        double value = context.getParameter(parameterNames[i]);
        if (value != localGlobalValues[parameterVariableIndex[i]])
            context.setParameter(parameterNames[i], localGlobalValues[parameterVariableIndex[i]]);
    }
}

void CommonIntegrateCustomStepKernel::getGlobalVariables(ContextImpl& context, vector<double>& values) const {
    if (!globalValues.isInitialized()) {
        // The data structures haven't been created yet, so just return the list of values that was given earlier.

        values = initialGlobalVariables;
        return;
    }
    values.resize(numGlobalVariables);
    for (int i = 0; i < numGlobalVariables; i++)
        values[i] = localGlobalValues[globalVariableIndex[i]];
}

void CommonIntegrateCustomStepKernel::setGlobalVariables(ContextImpl& context, const vector<double>& values) {
    if (numGlobalVariables == 0)
        return;
    if (!globalValues.isInitialized()) {
        // The data structures haven't been created yet, so just store the list of values.

        initialGlobalVariables = values;
        return;
    }
    for (int i = 0; i < numGlobalVariables; i++) {
        localGlobalValues[globalVariableIndex[i]] = values[i];
        expressionSet.setVariable(globalVariableIndex[i], values[i]);
    }
    deviceGlobalsAreCurrent = false;
}

void CommonIntegrateCustomStepKernel::getPerDofVariable(ContextImpl& context, int variable, vector<Vec3>& values) const {
    values.resize(perDofValues[variable].getSize());
    const vector<int>& order = cc.getAtomIndex();
    if (cc.getUseDoublePrecision() || cc.getUseMixedPrecision()) {
        if (!localValuesAreCurrent[variable]) {
            perDofValues[variable].download(localPerDofValuesDouble[variable]);
            localValuesAreCurrent[variable] = true;
        }
        for (int i = 0; i < (int) values.size(); i++) {
            values[order[i]][0] = localPerDofValuesDouble[variable][i].x;
            values[order[i]][1] = localPerDofValuesDouble[variable][i].y;
            values[order[i]][2] = localPerDofValuesDouble[variable][i].z;
        }
    }
    else {
        if (!localValuesAreCurrent[variable]) {
            perDofValues[variable].download(localPerDofValuesFloat[variable]);
            localValuesAreCurrent[variable] = true;
        }
        for (int i = 0; i < (int) values.size(); i++) {
            values[order[i]][0] = localPerDofValuesFloat[variable][i].x;
            values[order[i]][1] = localPerDofValuesFloat[variable][i].y;
            values[order[i]][2] = localPerDofValuesFloat[variable][i].z;
        }
    }
}

void CommonIntegrateCustomStepKernel::setPerDofVariable(ContextImpl& context, int variable, const vector<Vec3>& values) {
    const vector<int>& order = cc.getAtomIndex();
    localValuesAreCurrent[variable] = true;
    deviceValuesAreCurrent[variable] = false;
    if (cc.getUseDoublePrecision() || cc.getUseMixedPrecision()) {
        localPerDofValuesDouble[variable].resize(values.size());
        for (int i = 0; i < (int) values.size(); i++)
            localPerDofValuesDouble[variable][i] = mm_double4(values[order[i]][0], values[order[i]][1], values[order[i]][2], 0);
    }
    else {
        localPerDofValuesFloat[variable].resize(values.size());
        for (int i = 0; i < (int) values.size(); i++)
            localPerDofValuesFloat[variable][i] = mm_float4(values[order[i]][0], values[order[i]][1], values[order[i]][2], 0);
    }
}

void CommonRemoveCMMotionKernel::initialize(const System& system, const CMMotionRemover& force) {
    cc.setAsCurrent();
    frequency = force.getFrequency();
    int numAtoms = cc.getNumAtoms();
    cmMomentum.initialize<mm_float3>(cc, cc.getPaddedNumAtoms(), "cmMomentum");
    double totalMass = 0.0;
    for (int i = 0; i < numAtoms; i++)
        totalMass += system.getParticleMass(i);
    map<string, string> defines;
    defines["INVERSE_TOTAL_MASS"] = cc.doubleToString(totalMass == 0 ? 0.0 : 1.0/totalMass);
    defines["BLOCK_SIZE"] = to_string(64); // single-wavefront blocks work pretty well for AMD
    defines["WARP_SIZE"] = to_string(cc.getSIMDWidth());
    ComputeProgram program = cc.compileProgram(CommonKernelSources::removeCM, defines);
    kernel1 = program->createKernel("calcCenterOfMassMomentum");
    kernel1->addArg(numAtoms);
    kernel1->addArg(cc.getVelm());
    kernel1->addArg(cmMomentum);
    kernel2 = program->createKernel("removeCenterOfMassMomentum");
    kernel2->addArg(numAtoms);
    kernel2->addArg(cc.getVelm());
    kernel2->addArg(cmMomentum);
}

void CommonRemoveCMMotionKernel::execute(ContextImpl& context) {
    cc.setAsCurrent();
    // 64 is a single wavefront on AMD (i.e., no thread-syncs), or two on NVIDIA
    kernel1->execute(cc.getNumAtoms(), 64);
    kernel2->execute(cc.getNumAtoms(), 64);
}

class CommonCalcRMSDForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const RMSDForce& force) : force(force) {
        updateParticles();
    }
    void updateParticles() {
        particles.clear();
        for (int i : force.getParticles())
            particles.insert(i);
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        bool include1 = (particles.find(particle1) != particles.end());
        bool include2 = (particles.find(particle2) != particles.end());
        return (include1 == include2);
    }
private:
    const RMSDForce& force;
    set<int> particles;
};

void CommonCalcRMSDForceKernel::initialize(const System& system, const RMSDForce& force) {
    // Create data structures.

    bool useDouble = cc.getUseDoublePrecision();
    int elementSize = (useDouble ? sizeof(double) : sizeof(float));
    int numParticles = force.getParticles().size();
    if (numParticles == 0)
        numParticles = system.getNumParticles();
    referencePos.initialize(cc, system.getNumParticles(), 4*elementSize, "referencePos");
    particles.initialize<int>(cc, numParticles, "particles");
    buffer.initialize(cc, 13, elementSize, "buffer");
    recordParameters(force);
    info = new ForceInfo(force);
    cc.addForce(info);

    // Create the kernels.

    blockSize = min(256, cc.getMaxThreadBlockSize());
    map<string, string> defines;
    defines["THREAD_BLOCK_SIZE"] = cc.intToString(blockSize);
    defines["WARP_SIZE"] = cc.intToString(cc.getSIMDWidth());
    ComputeProgram program = cc.compileProgram(CommonKernelSources::rmsd, defines);
    kernel1 = program->createKernel("computeRMSDPart1");
    kernel2 = program->createKernel("computeRMSDForces");
    kernel1->addArg();
    kernel1->addArg(cc.getPosq());
    kernel1->addArg(referencePos);
    kernel1->addArg(particles);
    kernel1->addArg(buffer);
    kernel2->addArg();
    kernel2->addArg(cc.getPaddedNumAtoms());
    kernel2->addArg(cc.getPosq());
    kernel2->addArg(referencePos);
    kernel2->addArg(particles);
    kernel2->addArg(buffer);
    kernel2->addArg(cc.getLongForceBuffer());
}

void CommonCalcRMSDForceKernel::recordParameters(const RMSDForce& force) {
    // Record the parameters and center the reference positions.

    vector<int> particleVec = force.getParticles();
    if (particleVec.size() == 0)
        for (int i = 0; i < cc.getNumAtoms(); i++)
            particleVec.push_back(i);
    vector<Vec3> centeredPositions = force.getReferencePositions();
    Vec3 center;
    for (int i : particleVec)
        center += centeredPositions[i];
    center /= particleVec.size();
    for (Vec3& p : centeredPositions)
        p -= center;

    // Upload them to the device.

    particles.upload(particleVec);
    vector<mm_double4> pos;
    for (Vec3 p : centeredPositions)
        pos.push_back(mm_double4(p[0], p[1], p[2], 0));
    referencePos.upload(pos, true);

    // Record the sum of the norms of the reference positions.

    sumNormRef = 0.0;
    for (int i : particleVec) {
        Vec3 p = centeredPositions[i];
        sumNormRef += p.dot(p);
    }
}

double CommonCalcRMSDForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (cc.getUseDoublePrecision())
        return executeImpl<double>(context);
    return executeImpl<float>(context);
}

template <class REAL>
double CommonCalcRMSDForceKernel::executeImpl(ContextImpl& context) {
    // Execute the first kernel.

    int numParticles = particles.getSize();
    kernel1->setArg(0, numParticles);
    kernel1->execute(blockSize, blockSize);

    // Download the results, build the F matrix, and find the maximum eigenvalue
    // and eigenvector.

    vector<REAL> b;
    buffer.download(b);
    Array2D<double> F(4, 4);
    F[0][0] =  b[0*3+0] + b[1*3+1] + b[2*3+2];
    F[1][0] =  b[1*3+2] - b[2*3+1];
    F[2][0] =  b[2*3+0] - b[0*3+2];
    F[3][0] =  b[0*3+1] - b[1*3+0];
    F[0][1] =  b[1*3+2] - b[2*3+1];
    F[1][1] =  b[0*3+0] - b[1*3+1] - b[2*3+2];
    F[2][1] =  b[0*3+1] + b[1*3+0];
    F[3][1] =  b[0*3+2] + b[2*3+0];
    F[0][2] =  b[2*3+0] - b[0*3+2];
    F[1][2] =  b[0*3+1] + b[1*3+0];
    F[2][2] = -b[0*3+0] + b[1*3+1] - b[2*3+2];
    F[3][2] =  b[1*3+2] + b[2*3+1];
    F[0][3] =  b[0*3+1] - b[1*3+0];
    F[1][3] =  b[0*3+2] + b[2*3+0];
    F[2][3] =  b[1*3+2] + b[2*3+1];
    F[3][3] = -b[0*3+0] - b[1*3+1] + b[2*3+2];
    JAMA::Eigenvalue<double> eigen(F);
    Array1D<double> values;
    eigen.getRealEigenvalues(values);
    Array2D<double> vectors;
    eigen.getV(vectors);

    // Compute the RMSD.

    double msd = (sumNormRef+b[9]-2*values[3])/numParticles;
    if (msd < 1e-20) {
        // The particles are perfectly aligned, so all the forces should be zero.
        // Numerical error can lead to NaNs, so just return 0 now.
        return 0.0;
    }
    double rmsd = sqrt(msd);
    b[9] = rmsd;

    // Compute the rotation matrix.

    double q[] = {vectors[0][3], vectors[1][3], vectors[2][3], vectors[3][3]};
    double q00 = q[0]*q[0], q01 = q[0]*q[1], q02 = q[0]*q[2], q03 = q[0]*q[3];
    double q11 = q[1]*q[1], q12 = q[1]*q[2], q13 = q[1]*q[3];
    double q22 = q[2]*q[2], q23 = q[2]*q[3];
    double q33 = q[3]*q[3];
    b[0] = q00+q11-q22-q33;
    b[1] = 2*(q12-q03);
    b[2] = 2*(q13+q02);
    b[3] = 2*(q12+q03);
    b[4] = q00-q11+q22-q33;
    b[5] = 2*(q23-q01);
    b[6] = 2*(q13-q02);
    b[7] = 2*(q23+q01);
    b[8] = q00-q11-q22+q33;

    // Upload it to the device and invoke the kernel to apply forces.

    buffer.upload(b);
    kernel2->setArg(0, numParticles);
    kernel2->execute(numParticles);
    return rmsd;
}

void CommonCalcRMSDForceKernel::copyParametersToContext(ContextImpl& context, const RMSDForce& force) {
    if (referencePos.getSize() != force.getReferencePositions().size())
        throw OpenMMException("updateParametersInContext: The number of reference positions has changed");
    int numParticles = force.getParticles().size();
    if (numParticles == 0)
        numParticles = context.getSystem().getNumParticles();
    if (numParticles != particles.getSize())
        particles.resize(numParticles);
    recordParameters(force);

    // Mark that the current reordering may be invalid.

    info->updateParticles();
    cc.invalidateMolecules(info);
}

void CommonApplyAndersenThermostatKernel::initialize(const System& system, const AndersenThermostat& thermostat) {
    cc.setAsCurrent();
    randomSeed = thermostat.getRandomNumberSeed();
    ComputeProgram program = cc.compileProgram(CommonKernelSources::andersenThermostat);
    kernel = program->createKernel("applyAndersenThermostat");
    cc.getIntegrationUtilities().initRandomNumberGenerator(randomSeed);

    // Create the arrays with the group definitions.

    vector<vector<int> > groups = AndersenThermostatImpl::calcParticleGroups(system);
    atomGroups.initialize<int>(cc, cc.getNumAtoms(), "atomGroups");
    vector<int> atoms(atomGroups.getSize());
    for (int i = 0; i < (int) groups.size(); i++) {
        for (int j = 0; j < (int) groups[i].size(); j++)
            atoms[groups[i][j]] = i;
    }
    atomGroups.upload(atoms);
    kernel->addArg(system.getNumParticles());
    kernel->addArg();
    kernel->addArg();
    kernel->addArg(cc.getVelm());
    kernel->addArg();
    kernel->addArg(cc.getIntegrationUtilities().getRandom());
    kernel->addArg();
    kernel->addArg(atomGroups);
}

void CommonApplyAndersenThermostatKernel::execute(ContextImpl& context) {
    cc.setAsCurrent();
    kernel->setArg(1, (float) context.getParameter(AndersenThermostat::CollisionFrequency()));
    kernel->setArg(2, (float) (BOLTZ*context.getParameter(AndersenThermostat::Temperature())));
    double stepSize = context.getIntegrator().getStepSize();
    if (cc.getUseDoublePrecision())
        kernel->setArg(4, stepSize);
    else
        kernel->setArg(4, (float) stepSize);
    kernel->setArg(6, cc.getIntegrationUtilities().prepareRandomNumbers(cc.getPaddedNumAtoms()));
    kernel->execute(cc.getNumAtoms());
}
