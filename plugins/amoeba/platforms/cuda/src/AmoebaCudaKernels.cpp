/* -------------------------------------------------------------------------- *
 *                               OpenMMAmoeba                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2020 Stanford University and the Authors.      *
 * Authors: Peter Eastman, Mark Friedrichs                                    *
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

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "AmoebaCudaKernels.h"
#include "CudaAmoebaKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/AmoebaGeneralizedKirkwoodForceImpl.h"
#include "openmm/internal/AmoebaMultipoleForceImpl.h"
#include "openmm/internal/AmoebaWcaDispersionForceImpl.h"
#include "openmm/internal/AmoebaTorsionTorsionForceImpl.h"
#include "openmm/internal/AmoebaVdwForceImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "CudaBondedUtilities.h"
#include "CudaFFT3D.h"
#include "CudaForceInfo.h"
#include "CudaKernelSources.h"
#include "SimTKOpenMMRealType.h"
#include "jama_lu.h"

#include <algorithm>
#include <cmath>
#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace OpenMM;
using namespace std;

#define CHECK_RESULT(result, prefix) \
    if (result != CUDA_SUCCESS) { \
        std::stringstream m; \
        m<<prefix<<": "<<cu.getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }

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

/* -------------------------------------------------------------------------- *
 *                             AmoebaMultipole                                *
 * -------------------------------------------------------------------------- */

class CudaCalcAmoebaMultipoleForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const AmoebaMultipoleForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double charge1, charge2, thole1, thole2, damping1, damping2, polarity1, polarity2;
        int axis1, axis2, multipole11, multipole12, multipole21, multipole22, multipole31, multipole32;
        vector<double> dipole1, dipole2, quadrupole1, quadrupole2;
        force.getMultipoleParameters(particle1, charge1, dipole1, quadrupole1, axis1, multipole11, multipole21, multipole31, thole1, damping1, polarity1);
        force.getMultipoleParameters(particle2, charge2, dipole2, quadrupole2, axis2, multipole12, multipole22, multipole32, thole2, damping2, polarity2);
        if (charge1 != charge2 || thole1 != thole2 || damping1 != damping2 || polarity1 != polarity2 || axis1 != axis2) {
            return false;
        }
        for (int i = 0; i < (int) dipole1.size(); ++i) {
            if (dipole1[i] != dipole2[i]) {
                return false;
            }
        }
        for (int i = 0; i < (int) quadrupole1.size(); ++i) {
            if (quadrupole1[i] != quadrupole2[i]) {
                return false;
            }
        }
        return true;
    }
    int getNumParticleGroups() {
        return 7*force.getNumMultipoles();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle = index/7;
        int type = index-7*particle;
        force.getCovalentMap(particle, AmoebaMultipoleForce::CovalentType(type), particles);
    }
    bool areGroupsIdentical(int group1, int group2) {
        return ((group1%7) == (group2%7));
    }
private:
    const AmoebaMultipoleForce& force;
};

CudaCalcAmoebaMultipoleForceKernel::CudaCalcAmoebaMultipoleForceKernel(const std::string& name, const Platform& platform, CudaContext& cu, const System& system) :
        CalcAmoebaMultipoleForceKernel(name, platform), cu(cu), system(system), hasInitializedScaleFactors(false), hasInitializedFFT(false), multipolesAreValid(false), hasCreatedEvent(false),
        gkKernel(NULL) {
}

CudaCalcAmoebaMultipoleForceKernel::~CudaCalcAmoebaMultipoleForceKernel() {
    cu.setAsCurrent();
    if (hasInitializedFFT)
        cufftDestroy(fft);
    if (hasCreatedEvent)
        cuEventDestroy(syncEvent);
}

void CudaCalcAmoebaMultipoleForceKernel::initialize(const System& system, const AmoebaMultipoleForce& force) {
    cu.setAsCurrent();

    // Initialize multipole parameters.

    numMultipoles = force.getNumMultipoles();
    CudaArray& posq = cu.getPosq();
    vector<double4> temp(posq.getSize());
    float4* posqf = (float4*) &temp[0];
    double4* posqd = (double4*) &temp[0];
    vector<float2> dampingAndTholeVec;
    vector<float> polarizabilityVec;
    vector<float> localDipolesVec;
    vector<float> localQuadrupolesVec;
    vector<int4> multipoleParticlesVec;
    for (int i = 0; i < numMultipoles; i++) {
        double charge, thole, damping, polarity;
        int axisType, atomX, atomY, atomZ;
        vector<double> dipole, quadrupole;
        force.getMultipoleParameters(i, charge, dipole, quadrupole, axisType, atomZ, atomX, atomY, thole, damping, polarity);
        if (cu.getUseDoublePrecision())
            posqd[i] = make_double4(0, 0, 0, charge);
        else
            posqf[i] = make_float4(0, 0, 0, (float) charge);
        dampingAndTholeVec.push_back(make_float2((float) damping, (float) thole));
        polarizabilityVec.push_back((float) polarity);
        multipoleParticlesVec.push_back(make_int4(atomX, atomY, atomZ, axisType));
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back((float) dipole[j]);
        localQuadrupolesVec.push_back((float) quadrupole[0]);
        localQuadrupolesVec.push_back((float) quadrupole[1]);
        localQuadrupolesVec.push_back((float) quadrupole[2]);
        localQuadrupolesVec.push_back((float) quadrupole[4]);
        localQuadrupolesVec.push_back((float) quadrupole[5]);
    }
    hasQuadrupoles = false;
    for (auto q : localQuadrupolesVec)
        if (q != 0.0)
            hasQuadrupoles = true;
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    for (int i = numMultipoles; i < paddedNumAtoms; i++) {
        dampingAndTholeVec.push_back(make_float2(0, 0));
        polarizabilityVec.push_back(0);
        multipoleParticlesVec.push_back(make_int4(0, 0, 0, 0));
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back(0);
        for (int j = 0; j < 5; j++)
            localQuadrupolesVec.push_back(0);
    }
    dampingAndThole.initialize<float2>(cu, paddedNumAtoms, "dampingAndThole");
    polarizability.initialize<float>(cu, paddedNumAtoms, "polarizability");
    multipoleParticles.initialize<int4>(cu, paddedNumAtoms, "multipoleParticles");
    localDipoles.initialize<float>(cu, 3*paddedNumAtoms, "localDipoles");
    localQuadrupoles.initialize<float>(cu, 5*paddedNumAtoms, "localQuadrupoles");
    lastPositions.initialize(cu, cu.getPosq().getSize(), cu.getPosq().getElementSize(), "lastPositions");
    dampingAndThole.upload(dampingAndTholeVec);
    polarizability.upload(polarizabilityVec);
    multipoleParticles.upload(multipoleParticlesVec);
    localDipoles.upload(localDipolesVec);
    localQuadrupoles.upload(localQuadrupolesVec);
    posq.upload(&temp[0]);
    
    // Create workspace arrays.
    
    polarizationType = force.getPolarizationType();
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    labDipoles.initialize(cu, paddedNumAtoms, 3*elementSize, "labDipoles");
    labQuadrupoles.initialize(cu, 5*paddedNumAtoms, elementSize, "labQuadrupoles");
    sphericalDipoles.initialize(cu, 3*paddedNumAtoms, elementSize, "sphericalDipoles");
    sphericalQuadrupoles.initialize(cu, 5*paddedNumAtoms, elementSize, "sphericalQuadrupoles");
    fracDipoles.initialize(cu, paddedNumAtoms, 3*elementSize, "fracDipoles");
    fracQuadrupoles.initialize(cu, 6*paddedNumAtoms, elementSize, "fracQuadrupoles");
    field.initialize(cu, 3*paddedNumAtoms, sizeof(long long), "field");
    fieldPolar.initialize(cu, 3*paddedNumAtoms, sizeof(long long), "fieldPolar");
    torque.initialize(cu, 3*paddedNumAtoms, sizeof(long long), "torque");
    inducedDipole.initialize(cu, paddedNumAtoms, 3*elementSize, "inducedDipole");
    inducedDipolePolar.initialize(cu, paddedNumAtoms, 3*elementSize, "inducedDipolePolar");
    if (polarizationType == AmoebaMultipoleForce::Mutual) {
        inducedDipoleErrors.initialize(cu, cu.getNumThreadBlocks(), sizeof(float2), "inducedDipoleErrors");
        prevDipoles.initialize(cu, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevDipoles");
        prevDipolesPolar.initialize(cu, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevDipolesPolar");
        prevErrors.initialize(cu, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevErrors");
        diisMatrix.initialize(cu, MaxPrevDIISDipoles*MaxPrevDIISDipoles, elementSize, "diisMatrix");
        diisCoefficients.initialize(cu, MaxPrevDIISDipoles+1, sizeof(float), "diisMatrix");
        CHECK_RESULT(cuEventCreate(&syncEvent, CU_EVENT_DISABLE_TIMING), "Error creating event for AmoebaMultipoleForce");
        hasCreatedEvent = true;
    }
    else if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
        int numOrders = force.getExtrapolationCoefficients().size();
        extrapolatedDipole.initialize(cu, 3*numMultipoles*numOrders, elementSize, "extrapolatedDipole");
        extrapolatedDipolePolar.initialize(cu, 3*numMultipoles*numOrders, elementSize, "extrapolatedDipolePolar");
        inducedDipoleFieldGradient.initialize(cu, 6*paddedNumAtoms, sizeof(long long), "inducedDipoleFieldGradient");
        inducedDipoleFieldGradientPolar.initialize(cu, 6*paddedNumAtoms, sizeof(long long), "inducedDipoleFieldGradientPolar");
        extrapolatedDipoleFieldGradient.initialize(cu, 6*paddedNumAtoms*(numOrders-1), elementSize, "extrapolatedDipoleFieldGradient");
        extrapolatedDipoleFieldGradientPolar.initialize(cu, 6*paddedNumAtoms*(numOrders-1), elementSize, "extrapolatedDipoleFieldGradientPolar");
    }
    cu.addAutoclearBuffer(field);
    cu.addAutoclearBuffer(fieldPolar);
    cu.addAutoclearBuffer(torque);
    
    // Record which atoms should be flagged as exclusions based on covalent groups, and determine
    // the values for the covalent group flags.
    
    vector<vector<int> > exclusions(numMultipoles);
    for (int i = 0; i < numMultipoles; i++) {
        vector<int> atoms;
        set<int> allAtoms;
        allAtoms.insert(i);
        force.getCovalentMap(i, AmoebaMultipoleForce::Covalent12, atoms);
        allAtoms.insert(atoms.begin(), atoms.end());
        force.getCovalentMap(i, AmoebaMultipoleForce::Covalent13, atoms);
        allAtoms.insert(atoms.begin(), atoms.end());
        for (int atom : allAtoms)
            covalentFlagValues.push_back(make_int3(i, atom, 0));
        force.getCovalentMap(i, AmoebaMultipoleForce::Covalent14, atoms);
        allAtoms.insert(atoms.begin(), atoms.end());
        for (int atom : atoms)
            covalentFlagValues.push_back(make_int3(i, atom, 1));
        force.getCovalentMap(i, AmoebaMultipoleForce::Covalent15, atoms);
        for (int atom : atoms)
            covalentFlagValues.push_back(make_int3(i, atom, 2));
        allAtoms.insert(atoms.begin(), atoms.end());
        force.getCovalentMap(i, AmoebaMultipoleForce::PolarizationCovalent11, atoms);
        allAtoms.insert(atoms.begin(), atoms.end());
        exclusions[i].insert(exclusions[i].end(), allAtoms.begin(), allAtoms.end());

        // Workaround for bug in TINKER: if an atom is listed in both the PolarizationCovalent11
        // and PolarizationCovalent12 maps, the latter takes precedence.

        vector<int> atoms12;
        force.getCovalentMap(i, AmoebaMultipoleForce::PolarizationCovalent12, atoms12);
        for (int atom : atoms)
            if (find(atoms12.begin(), atoms12.end(), atom) == atoms12.end())
                polarizationFlagValues.push_back(make_int2(i, atom));
    }
    set<pair<int, int> > tilesWithExclusions;
    for (int atom1 = 0; atom1 < (int) exclusions.size(); ++atom1) {
        int x = atom1/CudaContext::TileSize;
        for (int atom2 : exclusions[atom1]) {
            int y = atom2/CudaContext::TileSize;
            tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
        }
    }
    
    // Record other options.
    
    if (polarizationType == AmoebaMultipoleForce::Mutual) {
        maxInducedIterations = force.getMutualInducedMaxIterations();
        inducedEpsilon = force.getMutualInducedTargetEpsilon();
    }
    else
        maxInducedIterations = 0;
    if (polarizationType != AmoebaMultipoleForce::Direct) {
        inducedField.initialize(cu, 3*paddedNumAtoms, sizeof(long long), "inducedField");
        inducedFieldPolar.initialize(cu, 3*paddedNumAtoms, sizeof(long long), "inducedFieldPolar");
    }
    usePME = (force.getNonbondedMethod() == AmoebaMultipoleForce::PME);
    
    // See whether there's an AmoebaGeneralizedKirkwoodForce in the System.

    const AmoebaGeneralizedKirkwoodForce* gk = NULL;
    for (int i = 0; i < system.getNumForces() && gk == NULL; i++)
        gk = dynamic_cast<const AmoebaGeneralizedKirkwoodForce*>(&system.getForce(i));
    double innerDielectric = (gk == NULL ? 1.0 : gk->getSoluteDielectric());
    
    // Create the kernels.

    bool useShuffle = (cu.getComputeCapability() >= 3.0 && !cu.getUseDoublePrecision());
    double fixedThreadMemory = 19*elementSize+2*sizeof(float)+3*sizeof(int)/(double) cu.TileSize;
    double inducedThreadMemory = 15*elementSize+2*sizeof(float);
    if (polarizationType == AmoebaMultipoleForce::Extrapolated)
        inducedThreadMemory += 12*elementSize;
    double electrostaticsThreadMemory = 0;
    if (!useShuffle)
        fixedThreadMemory += 3*elementSize;
    map<string, string> defines;
    defines["NUM_ATOMS"] = cu.intToString(numMultipoles);
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    defines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
    defines["ENERGY_SCALE_FACTOR"] = cu.doubleToString(ONE_4PI_EPS0/innerDielectric);
    if (polarizationType == AmoebaMultipoleForce::Direct)
        defines["DIRECT_POLARIZATION"] = "";
    else if (polarizationType == AmoebaMultipoleForce::Mutual)
        defines["MUTUAL_POLARIZATION"] = "";
    else if (polarizationType == AmoebaMultipoleForce::Extrapolated)
        defines["EXTRAPOLATED_POLARIZATION"] = "";
    if (useShuffle)
        defines["USE_SHUFFLE"] = "";
    if (hasQuadrupoles)
        defines["INCLUDE_QUADRUPOLES"] = "";
    defines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
    int numExclusionTiles = tilesWithExclusions.size();
    defines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(numExclusionTiles);
    int numContexts = cu.getPlatformData().contexts.size();
    int startExclusionIndex = cu.getContextIndex()*numExclusionTiles/numContexts;
    int endExclusionIndex = (cu.getContextIndex()+1)*numExclusionTiles/numContexts;
    defines["FIRST_EXCLUSION_TILE"] = cu.intToString(startExclusionIndex);
    defines["LAST_EXCLUSION_TILE"] = cu.intToString(endExclusionIndex);
    maxExtrapolationOrder = force.getExtrapolationCoefficients().size();
    defines["MAX_EXTRAPOLATION_ORDER"] = cu.intToString(maxExtrapolationOrder);
    stringstream coefficients;
    for (int i = 0; i < maxExtrapolationOrder; i++) {
        if (i > 0)
            coefficients << ",";
        double sum = 0;
        for (int j = i; j < maxExtrapolationOrder; j++)
            sum += force.getExtrapolationCoefficients()[j];
        coefficients << cu.doubleToString(sum);
    }
    defines["EXTRAPOLATION_COEFFICIENTS_SUM"] = coefficients.str();
    if (usePME) {
        int nx, ny, nz;
        force.getPMEParameters(pmeAlpha, nx, ny, nz);
        if (nx == 0 || pmeAlpha == 0) {
            NonbondedForce nb;
            nb.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
            nb.setCutoffDistance(force.getCutoffDistance());
            NonbondedForceImpl::calcPMEParameters(system, nb, pmeAlpha, gridSizeX, gridSizeY, gridSizeZ, false);
            gridSizeX = CudaFFT3D::findLegalDimension(gridSizeX);
            gridSizeY = CudaFFT3D::findLegalDimension(gridSizeY);
            gridSizeZ = CudaFFT3D::findLegalDimension(gridSizeZ);
        } else {
            gridSizeX = CudaFFT3D::findLegalDimension(nx);
            gridSizeY = CudaFFT3D::findLegalDimension(ny);
            gridSizeZ = CudaFFT3D::findLegalDimension(nz);
        }
        defines["EWALD_ALPHA"] = cu.doubleToString(pmeAlpha);
        defines["SQRT_PI"] = cu.doubleToString(sqrt(M_PI));
        defines["USE_EWALD"] = "";
        defines["USE_CUTOFF"] = "";
        defines["USE_PERIODIC"] = "";
        defines["CUTOFF_SQUARED"] = cu.doubleToString(force.getCutoffDistance()*force.getCutoffDistance());
    }
    if (gk != NULL) {
        defines["USE_GK"] = "";
        defines["GK_C"] = cu.doubleToString(2.455);
        double solventDielectric = gk->getSolventDielectric();
        defines["GK_FC"] = cu.doubleToString(1*(1-solventDielectric)/(0+1*solventDielectric));
        defines["GK_FD"] = cu.doubleToString(2*(1-solventDielectric)/(1+2*solventDielectric));
        defines["GK_FQ"] = cu.doubleToString(3*(1-solventDielectric)/(2+3*solventDielectric));
        fixedThreadMemory += 4*elementSize;
        inducedThreadMemory += 13*elementSize;
        if (polarizationType == AmoebaMultipoleForce::Mutual) {
            prevDipolesGk.initialize(cu, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevDipolesGk");
            prevDipolesGkPolar.initialize(cu, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevDipolesGkPolar");
        }
        else if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
            inducedThreadMemory += 12*elementSize;
            int numOrders = force.getExtrapolationCoefficients().size();
            extrapolatedDipoleGk.initialize(cu, 3*numMultipoles*numOrders, elementSize, "extrapolatedDipoleGk");
            extrapolatedDipoleGkPolar.initialize(cu, 3*numMultipoles*numOrders, elementSize, "extrapolatedDipoleGkPolar");
            inducedDipoleFieldGradientGk.initialize(cu, 6*paddedNumAtoms, elementSize, "inducedDipoleFieldGradientGk");
            inducedDipoleFieldGradientGkPolar.initialize(cu, 6*paddedNumAtoms, elementSize, "inducedDipoleFieldGradientGkPolar");
            extrapolatedDipoleFieldGradientGk.initialize(cu, 6*paddedNumAtoms*(numOrders-1), elementSize, "extrapolatedDipoleFieldGradientGk");
            extrapolatedDipoleFieldGradientGkPolar.initialize(cu, 6*paddedNumAtoms*(numOrders-1), elementSize, "extrapolatedDipoleFieldGradientGkPolar");
        }
    }
    NonbondedUtilities& nb = cu.getNonbondedUtilities();
    int maxThreads = nb.getForceThreadBlockSize();
    fixedFieldThreads = min(maxThreads, cu.computeThreadBlockSize(fixedThreadMemory));
    inducedFieldThreads = min(maxThreads, cu.computeThreadBlockSize(inducedThreadMemory));
    ComputeProgram program = cu.compileProgram(CudaAmoebaKernelSources::multipoles, defines);
    computeMomentsKernel = program->createKernel("computeLabFrameMoments");
    computeMomentsKernel->addArg(cu.getPosq());
    computeMomentsKernel->addArg(multipoleParticles);
    computeMomentsKernel->addArg(localDipoles);
    computeMomentsKernel->addArg(localQuadrupoles);
    computeMomentsKernel->addArg(labDipoles);
    computeMomentsKernel->addArg(labQuadrupoles);
    computeMomentsKernel->addArg(sphericalDipoles);
    computeMomentsKernel->addArg(sphericalQuadrupoles);
    recordInducedDipolesKernel = program->createKernel("recordInducedDipoles");
    recordInducedDipolesKernel->addArg(field);
    recordInducedDipolesKernel->addArg(fieldPolar);
    if (gk != NULL)
        for (int i = 0; i < 3; i++)
            recordInducedDipolesKernel->addArg();
    recordInducedDipolesKernel->addArg(inducedDipole);
    recordInducedDipolesKernel->addArg(inducedDipolePolar);
    recordInducedDipolesKernel->addArg(polarizability);
    mapTorqueKernel = program->createKernel("mapTorqueToForce");
    mapTorqueKernel->addArg(cu.getLongForceBuffer());
    mapTorqueKernel->addArg(torque);
    mapTorqueKernel->addArg(cu.getPosq());
    mapTorqueKernel->addArg(multipoleParticles);
    computePotentialKernel = program->createKernel("computePotentialAtPoints");
    computePotentialKernel->addArg(cu.getPosq());
    computePotentialKernel->addArg(labDipoles);
    computePotentialKernel->addArg(labQuadrupoles);
    computePotentialKernel->addArg(inducedDipole);
    for (int i = 0; i < 8; i++)
        computePotentialKernel->addArg();
    defines["THREAD_BLOCK_SIZE"] = cu.intToString(fixedFieldThreads);
    program = cu.compileProgram(CudaAmoebaKernelSources::multipoleFixedField, defines);
    computeFixedFieldKernel = program->createKernel("computeFixedField");
    computeFixedFieldKernel->addArg(field);
    computeFixedFieldKernel->addArg(fieldPolar);
    computeFixedFieldKernel->addArg(cu.getPosq());
    computeFixedFieldKernel->addArg(covalentFlags);
    computeFixedFieldKernel->addArg(polarizationGroupFlags);
    computeFixedFieldKernel->addArg(nb.getExclusionTiles());
    computeFixedFieldKernel->addArg();
    computeFixedFieldKernel->addArg();
    if (gk != NULL) {
        computeFixedFieldKernel->addArg();
        computeFixedFieldKernel->addArg();
    }
    else if (usePME) {
        computeFixedFieldKernel->addArg(nb.getInteractingTiles());
        computeFixedFieldKernel->addArg(nb.getInteractionCount());
        for (int i = 0; i < 6; i++)
            computeFixedFieldKernel->addArg();
        computeFixedFieldKernel->addArg(nb.getBlockCenters());
        computeFixedFieldKernel->addArg(nb.getInteractingAtoms());
    }
    computeFixedFieldKernel->addArg(labDipoles);
    computeFixedFieldKernel->addArg(labQuadrupoles);
    computeFixedFieldKernel->addArg(dampingAndThole);
    if (polarizationType != AmoebaMultipoleForce::Direct) {
        defines["THREAD_BLOCK_SIZE"] = cu.intToString(inducedFieldThreads);
        defines["MAX_PREV_DIIS_DIPOLES"] = cu.intToString(MaxPrevDIISDipoles);
        program = cu.compileProgram(CudaAmoebaKernelSources::multipoleInducedField, defines);
        computeInducedFieldKernel = program->createKernel("computeInducedField");
        computeInducedFieldKernel->addArg(inducedField);
        computeInducedFieldKernel->addArg(inducedFieldPolar);
        computeInducedFieldKernel->addArg(cu.getPosq());
        computeInducedFieldKernel->addArg(nb.getExclusionTiles());
        computeInducedFieldKernel->addArg(inducedDipole);
        computeInducedFieldKernel->addArg(inducedDipolePolar);
        computeInducedFieldKernel->addArg();
        computeInducedFieldKernel->addArg();
        if (usePME) {
            computeInducedFieldKernel->addArg(nb.getInteractingTiles());
            computeInducedFieldKernel->addArg(nb.getInteractionCount());
            for (int i = 0; i < 6; i++)
                computeInducedFieldKernel->addArg();
            computeInducedFieldKernel->addArg(nb.getBlockCenters());
            computeInducedFieldKernel->addArg(nb.getInteractingAtoms());
        }
        if (gk != NULL) {
            for (int i = 0; i < 5; i++)
                computeInducedFieldKernel->addArg();
        }
        if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
            computeInducedFieldKernel->addArg(inducedDipoleFieldGradient);
            computeInducedFieldKernel->addArg(inducedDipoleFieldGradientPolar);
            if (gk != NULL) {
                computeInducedFieldKernel->addArg(inducedDipoleFieldGradientGk);
                computeInducedFieldKernel->addArg(inducedDipoleFieldGradientGkPolar);
            }
        }
        computeInducedFieldKernel->addArg(dampingAndThole);
        updateInducedFieldKernel = program->createKernel("updateInducedFieldByDIIS");
        for (int i = 0; i < 4; i++)
            updateInducedFieldKernel->addArg();
        updateInducedFieldKernel->addArg(diisCoefficients);
        updateInducedFieldKernel->addArg();
        recordDIISDipolesKernel = program->createKernel("recordInducedDipolesForDIIS");
        recordDIISDipolesKernel->addArg(field);
        recordDIISDipolesKernel->addArg(fieldPolar);
        recordDIISDipolesKernel->addArg(polarizability);
        recordDIISDipolesKernel->addArg(inducedDipoleErrors);
        recordDIISDipolesKernel->addArg(prevErrors);
        recordDIISDipolesKernel->addArg(diisMatrix);
        for (int i = 0; i < 9; i++)
            recordDIISDipolesKernel->addArg();
        buildMatrixKernel = program->createKernel("computeDIISMatrix");
        buildMatrixKernel->addArg(prevErrors);
        buildMatrixKernel->addArg();
        buildMatrixKernel->addArg(diisMatrix);
        solveMatrixKernel = program->createKernel("solveDIISMatrix");
        solveMatrixKernel->addArg();
        solveMatrixKernel->addArg(diisMatrix);
        solveMatrixKernel->addArg(diisCoefficients);
        initExtrapolatedKernel = program->createKernel("initExtrapolatedDipoles");
        initExtrapolatedKernel->addArg(inducedDipole);
        initExtrapolatedKernel->addArg(extrapolatedDipole);
        initExtrapolatedKernel->addArg(inducedDipolePolar);
        initExtrapolatedKernel->addArg(extrapolatedDipolePolar);
        initExtrapolatedKernel->addArg(inducedDipoleFieldGradient);
        initExtrapolatedKernel->addArg(inducedDipoleFieldGradientPolar);
        if (gk != NULL) {
            initExtrapolatedKernel->addArg();
            initExtrapolatedKernel->addArg();
            initExtrapolatedKernel->addArg(extrapolatedDipoleGk);
            initExtrapolatedKernel->addArg(extrapolatedDipoleGkPolar);
            initExtrapolatedKernel->addArg(inducedDipoleFieldGradientGk);
            initExtrapolatedKernel->addArg(inducedDipoleFieldGradientGkPolar);
        }
        iterateExtrapolatedKernel = program->createKernel("iterateExtrapolatedDipoles");
        iterateExtrapolatedKernel->addArg();
        iterateExtrapolatedKernel->addArg(inducedDipole);
        iterateExtrapolatedKernel->addArg(extrapolatedDipole);
        iterateExtrapolatedKernel->addArg(inducedField);
        iterateExtrapolatedKernel->addArg(inducedDipolePolar);
        iterateExtrapolatedKernel->addArg(extrapolatedDipolePolar);
        iterateExtrapolatedKernel->addArg(inducedFieldPolar);
        iterateExtrapolatedKernel->addArg(inducedDipoleFieldGradient);
        iterateExtrapolatedKernel->addArg(inducedDipoleFieldGradientPolar);
        iterateExtrapolatedKernel->addArg(extrapolatedDipoleFieldGradient);
        iterateExtrapolatedKernel->addArg(extrapolatedDipoleFieldGradientPolar);
        if (gk != NULL) {
            iterateExtrapolatedKernel->addArg();
            iterateExtrapolatedKernel->addArg();
            iterateExtrapolatedKernel->addArg(extrapolatedDipoleGk);
            iterateExtrapolatedKernel->addArg(extrapolatedDipoleGkPolar);
            iterateExtrapolatedKernel->addArg(inducedDipoleFieldGradientGk);
            iterateExtrapolatedKernel->addArg(inducedDipoleFieldGradientGkPolar);
            iterateExtrapolatedKernel->addArg();
            iterateExtrapolatedKernel->addArg();
            iterateExtrapolatedKernel->addArg(extrapolatedDipoleFieldGradientGk);
            iterateExtrapolatedKernel->addArg(extrapolatedDipoleFieldGradientGkPolar);
        }
        iterateExtrapolatedKernel->addArg(polarizability);
        computeExtrapolatedKernel = program->createKernel("computeExtrapolatedDipoles");
        computeExtrapolatedKernel->addArg(inducedDipole);
        computeExtrapolatedKernel->addArg(extrapolatedDipole);
        computeExtrapolatedKernel->addArg(inducedDipolePolar);
        computeExtrapolatedKernel->addArg(extrapolatedDipolePolar);
        if (gk != NULL) {
            computeExtrapolatedKernel->addArg();
            computeExtrapolatedKernel->addArg();
            computeExtrapolatedKernel->addArg(extrapolatedDipoleGk);
            computeExtrapolatedKernel->addArg(extrapolatedDipoleGkPolar);
        }
        addExtrapolatedGradientKernel = program->createKernel("addExtrapolatedFieldGradientToForce");
        addExtrapolatedGradientKernel->addArg(cu.getLongForceBuffer());
        addExtrapolatedGradientKernel->addArg(extrapolatedDipole);
        addExtrapolatedGradientKernel->addArg(extrapolatedDipolePolar);
        addExtrapolatedGradientKernel->addArg(extrapolatedDipoleFieldGradient);
        addExtrapolatedGradientKernel->addArg(extrapolatedDipoleFieldGradientPolar);
        if (gk != NULL) {
            addExtrapolatedGradientKernel->addArg(extrapolatedDipoleGk);
            addExtrapolatedGradientKernel->addArg(extrapolatedDipoleGkPolar);
            addExtrapolatedGradientKernel->addArg(extrapolatedDipoleFieldGradientGk);
            addExtrapolatedGradientKernel->addArg(extrapolatedDipoleFieldGradientGkPolar);
        }
    }
    stringstream electrostaticsSource;
    electrostaticsSource << CudaAmoebaKernelSources::sphericalMultipoles;
    if (usePME)
        electrostaticsSource << CudaAmoebaKernelSources::pmeMultipoleElectrostatics;
    else
        electrostaticsSource << CudaAmoebaKernelSources::multipoleElectrostatics;
    electrostaticsThreadMemory = 24*elementSize+3*sizeof(float)+3*sizeof(int)/(double) cu.TileSize;
    electrostaticsThreads = min(maxThreads, cu.computeThreadBlockSize(electrostaticsThreadMemory));
    defines["THREAD_BLOCK_SIZE"] = cu.intToString(electrostaticsThreads);
    program = cu.compileProgram(electrostaticsSource.str(), defines);
    electrostaticsKernel = program->createKernel("computeElectrostatics");
    electrostaticsKernel->addArg(cu.getLongForceBuffer());
    electrostaticsKernel->addArg(torque);
    electrostaticsKernel->addArg(cu.getEnergyBuffer());
    electrostaticsKernel->addArg(cu.getPosq());
    electrostaticsKernel->addArg(covalentFlags);
    electrostaticsKernel->addArg(polarizationGroupFlags);
    electrostaticsKernel->addArg(nb.getExclusionTiles());
    electrostaticsKernel->addArg();
    electrostaticsKernel->addArg();
    if (usePME) {
        electrostaticsKernel->addArg(nb.getInteractingTiles());
        electrostaticsKernel->addArg(nb.getInteractionCount());
        for (int i = 0; i < 6; i++)
            electrostaticsKernel->addArg();
        electrostaticsKernel->addArg(nb.getBlockCenters());
        electrostaticsKernel->addArg(nb.getInteractingAtoms());
    }
    electrostaticsKernel->addArg(sphericalDipoles);
    electrostaticsKernel->addArg(sphericalQuadrupoles);
    electrostaticsKernel->addArg(inducedDipole);
    electrostaticsKernel->addArg(inducedDipolePolar);
    electrostaticsKernel->addArg(dampingAndThole);

    // Set up PME.
    
    if (usePME) {
        // Create the PME kernels.

        map<string, string> pmeDefines;
        pmeDefines["EWALD_ALPHA"] = cu.doubleToString(pmeAlpha);
        pmeDefines["PME_ORDER"] = cu.intToString(PmeOrder);
        pmeDefines["NUM_ATOMS"] = cu.intToString(numMultipoles);
        pmeDefines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        pmeDefines["EPSILON_FACTOR"] = cu.doubleToString(ONE_4PI_EPS0);
        pmeDefines["GRID_SIZE_X"] = cu.intToString(gridSizeX);
        pmeDefines["GRID_SIZE_Y"] = cu.intToString(gridSizeY);
        pmeDefines["GRID_SIZE_Z"] = cu.intToString(gridSizeZ);
        pmeDefines["M_PI"] = cu.doubleToString(M_PI);
        pmeDefines["SQRT_PI"] = cu.doubleToString(sqrt(M_PI));
        if (polarizationType == AmoebaMultipoleForce::Direct)
            pmeDefines["DIRECT_POLARIZATION"] = "";
        else if (polarizationType == AmoebaMultipoleForce::Mutual)
            pmeDefines["MUTUAL_POLARIZATION"] = "";
        else if (polarizationType == AmoebaMultipoleForce::Extrapolated)
            pmeDefines["EXTRAPOLATED_POLARIZATION"] = "";
        program = cu.compileProgram(CudaAmoebaKernelSources::multipolePme, pmeDefines);
        pmeTransformMultipolesKernel = program->createKernel("transformMultipolesToFractionalCoordinates");
        pmeTransformMultipolesKernel->addArg(labDipoles);
        pmeTransformMultipolesKernel->addArg(labQuadrupoles);
        pmeTransformMultipolesKernel->addArg(fracDipoles);
        pmeTransformMultipolesKernel->addArg(fracQuadrupoles);
        for (int i = 0; i < 3; i++)
            pmeTransformMultipolesKernel->addArg();
        pmeTransformPotentialKernel = program->createKernel("transformPotentialToCartesianCoordinates");
        pmeTransformPotentialKernel->addArg();
        pmeTransformPotentialKernel->addArg(pmeCphi);
        for (int i = 0; i < 3; i++)
            pmeTransformPotentialKernel->addArg();
        pmeSpreadFixedMultipolesKernel = program->createKernel("gridSpreadFixedMultipoles");
        pmeSpreadFixedMultipolesKernel->addArg(cu.getPosq());
        pmeSpreadFixedMultipolesKernel->addArg(fracDipoles);
        pmeSpreadFixedMultipolesKernel->addArg(fracQuadrupoles);
        pmeSpreadFixedMultipolesKernel->addArg(pmeGrid);
        for (int i = 0; i < 6; i++)
            pmeSpreadFixedMultipolesKernel->addArg();
        pmeSpreadInducedDipolesKernel = program->createKernel("gridSpreadInducedDipoles");
        pmeSpreadInducedDipolesKernel->addArg(cu.getPosq());
        pmeSpreadInducedDipolesKernel->addArg(inducedDipole);
        pmeSpreadInducedDipolesKernel->addArg(inducedDipolePolar);
        pmeSpreadInducedDipolesKernel->addArg(pmeGrid);
        for (int i = 0; i < 6; i++)
            pmeSpreadInducedDipolesKernel->addArg();
        pmeFinishSpreadChargeKernel = program->createKernel("finishSpreadCharge");
        pmeFinishSpreadChargeKernel->addArg(pmeGrid);
        pmeConvolutionKernel = program->createKernel("reciprocalConvolution");
        pmeConvolutionKernel->addArg(pmeGrid);
        pmeConvolutionKernel->addArg(pmeBsplineModuliX);
        pmeConvolutionKernel->addArg(pmeBsplineModuliY);
        pmeConvolutionKernel->addArg(pmeBsplineModuliZ);
        for (int i = 0; i < 4; i++)
            pmeConvolutionKernel->addArg();
        pmeFixedPotentialKernel = program->createKernel("computeFixedPotentialFromGrid");
        pmeFixedPotentialKernel->addArg(pmeGrid);
        pmeFixedPotentialKernel->addArg(pmePhi);
        pmeFixedPotentialKernel->addArg(field);
        pmeFixedPotentialKernel->addArg(fieldPolar);
        pmeFixedPotentialKernel->addArg(cu.getPosq());
        pmeFixedPotentialKernel->addArg(labDipoles);
        for (int i = 0; i < 6; i++)
            pmeFixedPotentialKernel->addArg();
        pmeInducedPotentialKernel = program->createKernel("computeInducedPotentialFromGrid");
        pmeInducedPotentialKernel->addArg(pmeGrid);
        pmeInducedPotentialKernel->addArg(pmePhid);
        pmeInducedPotentialKernel->addArg(pmePhip);
        pmeInducedPotentialKernel->addArg(pmePhidp);
        pmeInducedPotentialKernel->addArg(cu.getPosq());
        for (int i = 0; i < 6; i++)
            pmeInducedPotentialKernel->addArg();
        pmeFixedForceKernel = program->createKernel("computeFixedMultipoleForceAndEnergy");
        pmeFixedForceKernel->addArg(cu.getPosq());
        pmeFixedForceKernel->addArg(cu.getLongForceBuffer());
        pmeFixedForceKernel->addArg(torque);
        pmeFixedForceKernel->addArg(cu.getEnergyBuffer());
        pmeFixedForceKernel->addArg(labDipoles);
        pmeFixedForceKernel->addArg(labQuadrupoles);
        pmeFixedForceKernel->addArg(fracDipoles);
        pmeFixedForceKernel->addArg(fracQuadrupoles);
        pmeFixedForceKernel->addArg(pmePhi);
        pmeFixedForceKernel->addArg(pmeCphi);
        for (int i = 0; i < 3; i++)
            pmeFixedForceKernel->addArg();
        pmeInducedForceKernel = program->createKernel("computeInducedDipoleForceAndEnergy");
        pmeInducedForceKernel->addArg(cu.getPosq());
        pmeInducedForceKernel->addArg(cu.getLongForceBuffer());
        pmeInducedForceKernel->addArg(torque);
        pmeInducedForceKernel->addArg(cu.getEnergyBuffer());
        pmeInducedForceKernel->addArg(labDipoles);
        pmeInducedForceKernel->addArg(labQuadrupoles);
        pmeInducedForceKernel->addArg(fracDipoles);
        pmeInducedForceKernel->addArg(fracQuadrupoles);
        pmeInducedForceKernel->addArg(inducedDipole);
        pmeInducedForceKernel->addArg(inducedDipolePolar);
        pmeInducedForceKernel->addArg(pmePhi);
        pmeInducedForceKernel->addArg(pmePhid);
        pmeInducedForceKernel->addArg(pmePhip);
        pmeInducedForceKernel->addArg(pmePhidp);
        pmeInducedForceKernel->addArg(pmeCphi);
        for (int i = 0; i < 3; i++)
            pmeInducedForceKernel->addArg();
        pmeRecordInducedFieldDipolesKernel = program->createKernel("recordInducedFieldDipoles");
        pmeRecordInducedFieldDipolesKernel->addArg(pmePhid);
        pmeRecordInducedFieldDipolesKernel->addArg(pmePhip);
        pmeRecordInducedFieldDipolesKernel->addArg(inducedField);
        pmeRecordInducedFieldDipolesKernel->addArg(inducedFieldPolar);
        pmeRecordInducedFieldDipolesKernel->addArg(inducedDipole);
        pmeRecordInducedFieldDipolesKernel->addArg(inducedDipolePolar);
        for (int i = 0; i < 3; i++)
            pmeRecordInducedFieldDipolesKernel->addArg();
        if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
            pmeRecordInducedFieldDipolesKernel->addArg(inducedDipoleFieldGradient);
            pmeRecordInducedFieldDipolesKernel->addArg(inducedDipoleFieldGradientPolar);
        }

        // Create required data structures.

        int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
        pmeGrid.initialize(cu, gridSizeX*gridSizeY*gridSizeZ, 2*elementSize, "pmeGrid");
        cu.addAutoclearBuffer(pmeGrid);
        pmeBsplineModuliX.initialize(cu, gridSizeX, elementSize, "pmeBsplineModuliX");
        pmeBsplineModuliY.initialize(cu, gridSizeY, elementSize, "pmeBsplineModuliY");
        pmeBsplineModuliZ.initialize(cu, gridSizeZ, elementSize, "pmeBsplineModuliZ");
        pmePhi.initialize(cu, 20*numMultipoles, elementSize, "pmePhi");
        pmePhid.initialize(cu, 10*numMultipoles, elementSize, "pmePhid");
        pmePhip.initialize(cu, 10*numMultipoles, elementSize, "pmePhip");
        pmePhidp.initialize(cu, 20*numMultipoles, elementSize, "pmePhidp");
        pmeCphi.initialize(cu, 10*numMultipoles, elementSize, "pmeCphi");
        cufftResult result = cufftPlan3d(&fft, gridSizeX, gridSizeY, gridSizeZ, cu.getUseDoublePrecision() ? CUFFT_Z2Z : CUFFT_C2C);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: "+cu.intToString(result));
        hasInitializedFFT = true;

        // Initialize the B-spline moduli.

        double data[PmeOrder];
        double x = 0.0;
        data[0] = 1.0 - x;
        data[1] = x;
        for (int i = 2; i < PmeOrder; i++) {
            double denom = 1.0/i;
            data[i] = x*data[i-1]*denom;
            for (int j = 1; j < i; j++)
                data[i-j] = ((x+j)*data[i-j-1] + ((i-j+1)-x)*data[i-j])*denom;
            data[0] = (1.0-x)*data[0]*denom;
        }
        int maxSize = max(max(gridSizeX, gridSizeY), gridSizeZ);
        vector<double> bsplines_data(maxSize+1, 0.0);
        for (int i = 2; i <= PmeOrder+1; i++)
            bsplines_data[i] = data[i-2];
        for (int dim = 0; dim < 3; dim++) {
            int ndata = (dim == 0 ? gridSizeX : dim == 1 ? gridSizeY : gridSizeZ);
            vector<double> moduli(ndata);

            // get the modulus of the discrete Fourier transform

            double factor = 2.0*M_PI/ndata;
            for (int i = 0; i < ndata; i++) {
                double sc = 0.0;
                double ss = 0.0;
                for (int j = 1; j <= ndata; j++) {
                    double arg = factor*i*(j-1);
                    sc += bsplines_data[j]*cos(arg);
                    ss += bsplines_data[j]*sin(arg);
                }
                moduli[i] = sc*sc+ss*ss;
            }

            // Fix for exponential Euler spline interpolation failure.

            double eps = 1.0e-7;
            if (moduli[0] < eps)
                moduli[0] = 0.9*moduli[1];
            for (int i = 1; i < ndata-1; i++)
                if (moduli[i] < eps)
                    moduli[i] = 0.9*(moduli[i-1]+moduli[i+1]);
            if (moduli[ndata-1] < eps)
                moduli[ndata-1] = 0.9*moduli[ndata-2];

            // Compute and apply the optimal zeta coefficient.

            int jcut = 50;
            for (int i = 1; i <= ndata; i++) {
                int k = i - 1;
                if (i > ndata/2)
                    k = k - ndata;
                double zeta;
                if (k == 0)
                    zeta = 1.0;
                else {
                    double sum1 = 1.0;
                    double sum2 = 1.0;
                    factor = M_PI*k/ndata;
                    for (int j = 1; j <= jcut; j++) {
                        double arg = factor/(factor+M_PI*j);
                        sum1 += pow(arg, PmeOrder);
                        sum2 += pow(arg, 2*PmeOrder);
                    }
                    for (int j = 1; j <= jcut; j++) {
                        double arg = factor/(factor-M_PI*j);
                        sum1 += pow(arg, PmeOrder);
                        sum2 += pow(arg, 2*PmeOrder);
                    }
                    zeta = sum2/sum1;
                }
                moduli[i-1] = moduli[i-1]*zeta*zeta;
            }
            if (cu.getUseDoublePrecision()) {
                if (dim == 0)
                    pmeBsplineModuliX.upload(moduli);
                else if (dim == 1)
                    pmeBsplineModuliY.upload(moduli);
                else
                    pmeBsplineModuliZ.upload(moduli);
            }
            else {
                vector<float> modulif(ndata);
                for (int i = 0; i < ndata; i++)
                    modulif[i] = (float) moduli[i];
                if (dim == 0)
                    pmeBsplineModuliX.upload(modulif);
                else if (dim == 1)
                    pmeBsplineModuliY.upload(modulif);
                else
                    pmeBsplineModuliZ.upload(modulif);
            }
        }
    }

    // Add an interaction to the default nonbonded kernel.  This doesn't actually do any calculations.  It's
    // just so that CudaNonbondedUtilities will build the exclusion flags and maintain the neighbor list.
    
    cu.getNonbondedUtilities().addInteraction(usePME, usePME, true, force.getCutoffDistance(), exclusions, "", force.getForceGroup());
    cu.getNonbondedUtilities().setUsePadding(false);
    cu.addForce(new ForceInfo(force));
}

void CudaCalcAmoebaMultipoleForceKernel::initializeScaleFactors() {
    hasInitializedScaleFactors = true;
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    
    // Figure out the covalent flag values to use for each atom pair.

    vector<int2> exclusionTiles;
    nb.getExclusionTiles().download(exclusionTiles);
    map<pair<int, int>, int> exclusionTileMap;
    for (int i = 0; i < (int) exclusionTiles.size(); i++) {
        int2 tile = exclusionTiles[i];
        exclusionTileMap[make_pair(tile.x, tile.y)] = i;
    }
    covalentFlags.initialize<uint2>(cu, nb.getExclusions().getSize(), "covalentFlags");
    vector<uint2> covalentFlagsVec(nb.getExclusions().getSize(), make_uint2(0, 0));
    for (int3 values : covalentFlagValues) {
        int atom1 = values.x;
        int atom2 = values.y;
        int value = values.z;
        int x = atom1/CudaContext::TileSize;
        int offset1 = atom1-x*CudaContext::TileSize;
        int y = atom2/CudaContext::TileSize;
        int offset2 = atom2-y*CudaContext::TileSize;
        int f1 = (value == 0 || value == 1 ? 1 : 0);
        int f2 = (value == 0 || value == 2 ? 1 : 0);
        if (x == y) {
            int index = exclusionTileMap[make_pair(x, y)]*CudaContext::TileSize;
            covalentFlagsVec[index+offset1].x |= f1<<offset2;
            covalentFlagsVec[index+offset1].y |= f2<<offset2;
            covalentFlagsVec[index+offset2].x |= f1<<offset1;
            covalentFlagsVec[index+offset2].y |= f2<<offset1;
        }
        else if (x > y) {
            int index = exclusionTileMap[make_pair(x, y)]*CudaContext::TileSize;
            covalentFlagsVec[index+offset1].x |= f1<<offset2;
            covalentFlagsVec[index+offset1].y |= f2<<offset2;
        }
        else {
            int index = exclusionTileMap[make_pair(y, x)]*CudaContext::TileSize;
            covalentFlagsVec[index+offset2].x |= f1<<offset1;
            covalentFlagsVec[index+offset2].y |= f2<<offset1;
        }
    }
    covalentFlags.upload(covalentFlagsVec);
    
    // Do the same for the polarization flags.
    
    polarizationGroupFlags.initialize<unsigned int>(cu, nb.getExclusions().getSize(), "polarizationGroupFlags");
    vector<unsigned int> polarizationGroupFlagsVec(nb.getExclusions().getSize(), 0);
    for (int2 values : polarizationFlagValues) {
        int atom1 = values.x;
        int atom2 = values.y;
        int x = atom1/CudaContext::TileSize;
        int offset1 = atom1-x*CudaContext::TileSize;
        int y = atom2/CudaContext::TileSize;
        int offset2 = atom2-y*CudaContext::TileSize;
        if (x == y) {
            int index = exclusionTileMap[make_pair(x, y)]*CudaContext::TileSize;
            polarizationGroupFlagsVec[index+offset1] |= 1<<offset2;
            polarizationGroupFlagsVec[index+offset2] |= 1<<offset1;
        }
        else if (x > y) {
            int index = exclusionTileMap[make_pair(x, y)]*CudaContext::TileSize;
            polarizationGroupFlagsVec[index+offset1] |= 1<<offset2;
        }
        else {
            int index = exclusionTileMap[make_pair(y, x)]*CudaContext::TileSize;
            polarizationGroupFlagsVec[index+offset2] |= 1<<offset1;
        }
    }
    polarizationGroupFlags.upload(polarizationGroupFlagsVec);
}

void CudaCalcAmoebaMultipoleForceKernel::computeForwardFFT() {
    if (cu.getUseDoublePrecision())
        cufftExecZ2Z(fft, (double2*) pmeGrid.getDevicePointer(), (double2*) pmeGrid.getDevicePointer(), CUFFT_FORWARD);
    else
        cufftExecC2C(fft, (float2*) pmeGrid.getDevicePointer(), (float2*) pmeGrid.getDevicePointer(), CUFFT_FORWARD);
}

void CudaCalcAmoebaMultipoleForceKernel::computeInverseFFT() {
    if (cu.getUseDoublePrecision())
        cufftExecZ2Z(fft, (double2*) pmeGrid.getDevicePointer(), (double2*) pmeGrid.getDevicePointer(), CUFFT_INVERSE);
    else
        cufftExecC2C(fft, (float2*) pmeGrid.getDevicePointer(), (float2*) pmeGrid.getDevicePointer(), CUFFT_INVERSE);
}

double CudaCalcAmoebaMultipoleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!hasInitializedScaleFactors) {
        initializeScaleFactors();
        for (auto impl : context.getForceImpls()) {
            AmoebaGeneralizedKirkwoodForceImpl* gkImpl = dynamic_cast<AmoebaGeneralizedKirkwoodForceImpl*>(impl);
            if (gkImpl != NULL) {
                gkKernel = dynamic_cast<CudaCalcAmoebaGeneralizedKirkwoodForceKernel*>(&gkImpl->getKernel().getImpl());
                recordInducedDipolesKernel->setArg(2, gkKernel->getField());
                recordInducedDipolesKernel->setArg(3, gkKernel->getInducedDipoles());
                recordInducedDipolesKernel->setArg(4, gkKernel->getInducedDipolesPolar());
                computeFixedFieldKernel->setArg(8, gkKernel->getBornRadii());
                computeFixedFieldKernel->setArg(9, gkKernel->getField());
                if (polarizationType != AmoebaMultipoleForce::Direct) {
                    computeInducedFieldKernel->setArg(8, gkKernel->getInducedField());
                    computeInducedFieldKernel->setArg(9,  gkKernel->getInducedFieldPolar());
                    computeInducedFieldKernel->setArg(10, gkKernel->getInducedDipoles());
                    computeInducedFieldKernel->setArg(11, gkKernel->getInducedDipolesPolar());
                    computeInducedFieldKernel->setArg(12, gkKernel->getBornRadii());
                }
                if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
                    initExtrapolatedKernel->setArg(6, gkKernel->getInducedDipoles());
                    initExtrapolatedKernel->setArg(7, gkKernel->getInducedDipolesPolar());
                    iterateExtrapolatedKernel->setArg(11, gkKernel->getInducedDipoles());
                    iterateExtrapolatedKernel->setArg(12, gkKernel->getInducedDipolesPolar());
                    iterateExtrapolatedKernel->setArg(17, gkKernel->getInducedField());
                    iterateExtrapolatedKernel->setArg(18, gkKernel->getInducedFieldPolar());
                    computeExtrapolatedKernel->setArg(4, gkKernel->getInducedDipoles());
                    computeExtrapolatedKernel->setArg(5, gkKernel->getInducedDipolesPolar());
                }
                break;
            }
        }
    }
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    
    // Compute the lab frame moments.

    computeMomentsKernel->execute(cu.getNumAtoms());
    int startTileIndex = nb.getStartTileIndex();
    int numTileIndices = nb.getNumTiles();
    int numForceThreadBlocks = nb.getNumForceThreadBlocks();
    electrostaticsKernel->setArg(7, startTileIndex);
    electrostaticsKernel->setArg(8, numTileIndices);
    computeFixedFieldKernel->setArg(6, startTileIndex);
    computeFixedFieldKernel->setArg(7, numTileIndices);
    if (!pmeGrid.isInitialized()) {
        // Compute induced dipoles.
        
        if (gkKernel != NULL)
            gkKernel->computeBornRadii();
        computeFixedFieldKernel->execute(numForceThreadBlocks*fixedFieldThreads, fixedFieldThreads);
        recordInducedDipolesKernel->execute(cu.getNumAtoms());
        
        // Iterate until the dipoles converge.
        
        if (polarizationType == AmoebaMultipoleForce::Extrapolated)
            computeExtrapolatedDipoles(NULL);
        for (int i = 0; i < maxInducedIterations; i++) {
            computeInducedField(NULL);
            bool converged = iterateDipolesByDIIS(i);
            if (converged)
                break;
        }
        
        // Compute electrostatic force.
        
        electrostaticsKernel->execute(numForceThreadBlocks*electrostaticsThreads, electrostaticsThreads);
        if (gkKernel != NULL)
            gkKernel->finishComputation(torque, labDipoles, labQuadrupoles, inducedDipole, inducedDipolePolar, dampingAndThole, covalentFlags, polarizationGroupFlags);
    }
    else {
        // Compute reciprocal box vectors.
        
        Vec3 a, b, c;
        cu.getPeriodicBoxVectors(a, b, c);
        double determinant = a[0]*b[1]*c[2];
        double scale = 1.0/determinant;
        double3 recipBoxVectors[3];
        recipBoxVectors[0] = make_double3(b[1]*c[2]*scale, 0, 0);
        recipBoxVectors[1] = make_double3(-b[0]*c[2]*scale, a[0]*c[2]*scale, 0);
        recipBoxVectors[2] = make_double3((b[0]*c[1]-b[1]*c[0])*scale, -a[0]*c[1]*scale, a[0]*b[1]*scale);
        float3 recipBoxVectorsFloat[3];
        void* recipBoxVectorPointer[3];
        if (cu.getUseDoublePrecision()) {
            recipBoxVectorPointer[0] = &recipBoxVectors[0];
            recipBoxVectorPointer[1] = &recipBoxVectors[1];
            recipBoxVectorPointer[2] = &recipBoxVectors[2];
            double3 boxVectors[] = {make_double3(a[0], a[1], a[2]), make_double3(b[0], b[1], b[2]), make_double3(c[0], c[1], c[2])};
            pmeConvolutionKernel->setArg(4, make_double4(a[0], b[1], c[2], 0));
            for (int i = 0; i < 3; i++) {
                pmeTransformMultipolesKernel->setArg(4+i, recipBoxVectors[i]);
                pmeTransformPotentialKernel->setArg(2+i, recipBoxVectors[i]);
                pmeSpreadFixedMultipolesKernel->setArg(4+i, boxVectors[i]);
                pmeSpreadFixedMultipolesKernel->setArg(7+i, recipBoxVectors[i]);
                pmeSpreadInducedDipolesKernel->setArg(4+i, boxVectors[i]);
                pmeSpreadInducedDipolesKernel->setArg(7+i, recipBoxVectors[i]);
                pmeConvolutionKernel->setArg(5+i, recipBoxVectors[i]);
                pmeFixedPotentialKernel->setArg(6+i, boxVectors[i]);
                pmeFixedPotentialKernel->setArg(9+i, recipBoxVectors[i]);
                pmeInducedPotentialKernel->setArg(5+i, boxVectors[i]);
                pmeInducedPotentialKernel->setArg(8+i, recipBoxVectors[i]);
                pmeFixedForceKernel->setArg(10+i, recipBoxVectors[i]);
                pmeInducedForceKernel->setArg(15+i, recipBoxVectors[i]);
                pmeRecordInducedFieldDipolesKernel->setArg(6+i, recipBoxVectors[i]);
            }
        }
        else {
            recipBoxVectorsFloat[0] = make_float3((float) recipBoxVectors[0].x, 0, 0);
            recipBoxVectorsFloat[1] = make_float3((float) recipBoxVectors[1].x, (float) recipBoxVectors[1].y, 0);
            recipBoxVectorsFloat[2] = make_float3((float) recipBoxVectors[2].x, (float) recipBoxVectors[2].y, (float) recipBoxVectors[2].z);
            recipBoxVectorPointer[0] = &recipBoxVectorsFloat[0];
            recipBoxVectorPointer[1] = &recipBoxVectorsFloat[1];
            recipBoxVectorPointer[2] = &recipBoxVectorsFloat[2];
            float3 boxVectors[] = {make_float3(a[0], a[1], a[2]), make_float3(b[0], b[1], b[2]), make_float3(c[0], c[1], c[2])};
            pmeConvolutionKernel->setArg(4, make_float4(a[0], b[1], c[2], 0));
            for (int i = 0; i < 3; i++) {
                pmeTransformMultipolesKernel->setArg(4+i, recipBoxVectorsFloat[i]);
                pmeTransformPotentialKernel->setArg(2+i, recipBoxVectorsFloat[i]);
                pmeSpreadFixedMultipolesKernel->setArg(4+i, boxVectors[i]);
                pmeSpreadFixedMultipolesKernel->setArg(7+i, recipBoxVectorsFloat[i]);
                pmeSpreadInducedDipolesKernel->setArg(4+i, boxVectors[i]);
                pmeSpreadInducedDipolesKernel->setArg(7+i, recipBoxVectorsFloat[i]);
                pmeConvolutionKernel->setArg(5+i, recipBoxVectorsFloat[i]);
                pmeFixedPotentialKernel->setArg(6+i, boxVectors[i]);
                pmeFixedPotentialKernel->setArg(9+i, recipBoxVectorsFloat[i]);
                pmeInducedPotentialKernel->setArg(5+i, boxVectors[i]);
                pmeInducedPotentialKernel->setArg(8+i, recipBoxVectorsFloat[i]);
                pmeFixedForceKernel->setArg(10+i, recipBoxVectorsFloat[i]);
                pmeInducedForceKernel->setArg(15+i, recipBoxVectorsFloat[i]);
                pmeRecordInducedFieldDipolesKernel->setArg(6+i, recipBoxVectorsFloat[i]);
            }
        }

        // Reciprocal space calculation.
        
        unsigned int maxTiles = nb.getInteractingTiles().getSize();
        pmeTransformMultipolesKernel->execute(cu.getNumAtoms());
        pmeSpreadFixedMultipolesKernel->execute(cu.getNumAtoms());
        if (cu.getUseDoublePrecision())
            pmeFinishSpreadChargeKernel->execute(pmeGrid.getSize());
        computeForwardFFT();
        pmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ, 256);
        computeInverseFFT();
        pmeFixedPotentialKernel->execute(cu.getNumAtoms());
        pmeTransformPotentialKernel->setArg(0, pmePhi);
        pmeTransformPotentialKernel->execute(cu.getNumAtoms());
        pmeFixedForceKernel->execute(cu.getNumAtoms());

        // Direct space calculation.

        setPeriodicBoxArgs(cu, computeFixedFieldKernel, 10);
        computeFixedFieldKernel->setArg(15, maxTiles);
        computeFixedFieldKernel->execute(numForceThreadBlocks*fixedFieldThreads, fixedFieldThreads);
        recordInducedDipolesKernel->execute(cu.getNumAtoms());

        // Reciprocal space calculation for the induced dipoles.

        cu.clearBuffer(pmeGrid);
        pmeSpreadInducedDipolesKernel->execute(cu.getNumAtoms());
        if (cu.getUseDoublePrecision())
            pmeFinishSpreadChargeKernel->execute(pmeGrid.getSize());
        computeForwardFFT();
        pmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ, 256);
        computeInverseFFT();
        pmeInducedPotentialKernel->execute(cu.getNumAtoms());
        
        // Iterate until the dipoles converge.
        
        if (polarizationType == AmoebaMultipoleForce::Extrapolated)
            computeExtrapolatedDipoles(recipBoxVectorPointer);
        for (int i = 0; i < maxInducedIterations; i++) {
            computeInducedField(recipBoxVectorPointer);
            bool converged = iterateDipolesByDIIS(i);
            if (converged)
                break;
        }
        
        // Compute electrostatic force.
        
        setPeriodicBoxArgs(cu, electrostaticsKernel, 11);
        electrostaticsKernel->setArg(16, maxTiles);
        electrostaticsKernel->execute(numForceThreadBlocks*electrostaticsThreads, electrostaticsThreads);
        pmeTransformPotentialKernel->setArg(0, pmePhidp);
        pmeTransformPotentialKernel->execute(cu.getNumAtoms());
        pmeInducedForceKernel->execute(cu.getNumAtoms());
    }
    
    // If using extrapolated polarization, add in force contributions from µ(m) T µ(n).
    
    if (polarizationType == AmoebaMultipoleForce::Extrapolated)
        addExtrapolatedGradientKernel->execute(numMultipoles);

    // Map torques to force.

    mapTorqueKernel->execute(cu.getNumAtoms());
    
    // Record the current atom positions so we can tell later if they have changed.
    
    cu.getPosq().copyTo(lastPositions);
    multipolesAreValid = true;
    return 0.0;
}

void CudaCalcAmoebaMultipoleForceKernel::computeInducedField(void** recipBoxVectorPointer) {
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    int startTileIndex = nb.getStartTileIndex();
    int numTileIndices = nb.getNumTiles();
    int numForceThreadBlocks = nb.getNumForceThreadBlocks();
    computeInducedFieldKernel->setArg(6, startTileIndex);
    computeInducedFieldKernel->setArg(7, numTileIndices);
    if (usePME) {
        setPeriodicBoxArgs(cu, computeInducedFieldKernel, 10);
        computeInducedFieldKernel->setArg(15, nb.getInteractingTiles().getSize());
    }
    cu.clearBuffer(inducedField);
    cu.clearBuffer(inducedFieldPolar);
    if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
        cu.clearBuffer(inducedDipoleFieldGradient);
        cu.clearBuffer(inducedDipoleFieldGradientPolar);
    }
    if (gkKernel != NULL) {
        cu.clearBuffer(gkKernel->getInducedField());
        cu.clearBuffer(gkKernel->getInducedFieldPolar());
        if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
            cu.clearBuffer(inducedDipoleFieldGradientGk);
            cu.clearBuffer(inducedDipoleFieldGradientGkPolar);
        }
    }
    computeInducedFieldKernel->execute(numForceThreadBlocks*inducedFieldThreads, inducedFieldThreads);
    if (pmeGrid.isInitialized()) {
        cu.clearBuffer(pmeGrid);
        pmeSpreadInducedDipolesKernel->execute(cu.getNumAtoms());
        if (cu.getUseDoublePrecision()) {
            pmeFinishSpreadChargeKernel->execute(pmeGrid.getSize());
        }
        computeForwardFFT();
        pmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ, 256);
        computeInverseFFT();
        pmeInducedPotentialKernel->execute(cu.getNumAtoms());
        if (polarizationType == AmoebaMultipoleForce::Extrapolated) {
            pmeRecordInducedFieldDipolesKernel->execute(cu.getNumAtoms());
        }
        else {
            pmeRecordInducedFieldDipolesKernel->execute(cu.getNumAtoms());
        }
    }
}

bool CudaCalcAmoebaMultipoleForceKernel::iterateDipolesByDIIS(int iteration) {
    void* npt = NULL;

    // Record the dipoles and errors into the lists of previous dipoles.

    recordDIISDipolesKernel->setArg(13, iteration);
    if (gkKernel != NULL) {
        recordDIISDipolesKernel->setArg(6, gkKernel->getField());
        recordDIISDipolesKernel->setArg(7, gkKernel->getInducedField());
        recordDIISDipolesKernel->setArg(8, gkKernel->getInducedFieldPolar());
        recordDIISDipolesKernel->setArg(9, gkKernel->getInducedDipoles());
        recordDIISDipolesKernel->setArg(10, gkKernel->getInducedDipolesPolar());
        recordDIISDipolesKernel->setArg(11, prevDipolesGk);
        recordDIISDipolesKernel->setArg(12, prevDipolesGkPolar);
        recordDIISDipolesKernel->setArg(14, false);
        recordDIISDipolesKernel->execute(cu.getNumThreadBlocks()*64, 64);
    }
    recordDIISDipolesKernel->setArg(6, npt);
    recordDIISDipolesKernel->setArg(7, inducedField);
    recordDIISDipolesKernel->setArg(8, inducedFieldPolar);
    recordDIISDipolesKernel->setArg(9, inducedDipole);
    recordDIISDipolesKernel->setArg(10, inducedDipolePolar);
    recordDIISDipolesKernel->setArg(11, prevDipoles);
    recordDIISDipolesKernel->setArg(12, prevDipolesPolar);
    recordDIISDipolesKernel->setArg(14, true);
    recordDIISDipolesKernel->execute(cu.getNumThreadBlocks()*64, 64);
    float2* errors = (float2*) cu.getPinnedBuffer();
    inducedDipoleErrors.download(errors, false);
    cuEventRecord(syncEvent, cu.getCurrentStream());
    
    // Build the DIIS matrix.
    
    int numPrev = (iteration+1 < MaxPrevDIISDipoles ? iteration+1 : MaxPrevDIISDipoles);
    int threadBlocks = min(numPrev, cu.getNumThreadBlocks());
    int blockSize = 512;
    buildMatrixKernel->setArg(1, iteration);
    buildMatrixKernel->execute(threadBlocks*blockSize, blockSize);
    
    // Solve the matrix.

    solveMatrixKernel->setArg(0, iteration);
    solveMatrixKernel->execute(32, 32);
    
    // Determine whether the iteration has converged.
    
    cuEventSynchronize(syncEvent);
    double total1 = 0.0, total2 = 0.0;
    for (int j = 0; j < inducedDipoleErrors.getSize(); j++) {
        total1 += errors[j].x;
        total2 += errors[j].y;
    }
    if (48.033324*sqrt(max(total1, total2)/cu.getNumAtoms()) < inducedEpsilon)
        return true;
    
    // Compute the dipoles.
    
    updateInducedFieldKernel->setArg(0, inducedDipole);
    updateInducedFieldKernel->setArg(1, inducedDipolePolar);
    updateInducedFieldKernel->setArg(2, prevDipoles);
    updateInducedFieldKernel->setArg(3, prevDipolesPolar);
    updateInducedFieldKernel->setArg(5, numPrev);
    updateInducedFieldKernel->execute(3*cu.getNumAtoms(), 256);
    if (gkKernel != NULL) {
        updateInducedFieldKernel->setArg(0, gkKernel->getInducedDipoles());
        updateInducedFieldKernel->setArg(1, gkKernel->getInducedDipolesPolar());
        updateInducedFieldKernel->setArg(2, prevDipolesGk);
        updateInducedFieldKernel->setArg(3, prevDipolesGkPolar);
        updateInducedFieldKernel->execute(3*cu.getNumAtoms(), 256);
    }
    return false;
}

void CudaCalcAmoebaMultipoleForceKernel::computeExtrapolatedDipoles(void** recipBoxVectorPointer) {
    // Start by storing the direct dipoles as PT0

    initExtrapolatedKernel->execute(extrapolatedDipole.getSize());

    // Recursively apply alpha.Tau to the µ_(n) components to generate µ_(n+1), and store the result

    for (int order = 1; order < maxExtrapolationOrder; ++order) {
        computeInducedField(recipBoxVectorPointer);
        iterateExtrapolatedKernel->setArg(0, order);
        iterateExtrapolatedKernel->execute(extrapolatedDipole.getSize());
    }
    
    // Take a linear combination of the µ_(n) components to form the total dipole

    computeExtrapolatedKernel->execute(extrapolatedDipole.getSize());
    computeInducedField(recipBoxVectorPointer);
}

void CudaCalcAmoebaMultipoleForceKernel::ensureMultipolesValid(ContextImpl& context) {
    if (multipolesAreValid) {
        int numParticles = cu.getNumAtoms();
        if (cu.getUseDoublePrecision()) {
            vector<double4> pos1, pos2;
            cu.getPosq().download(pos1);
            lastPositions.download(pos2);
            for (int i = 0; i < numParticles; i++)
                if (pos1[i].x != pos2[i].x || pos1[i].y != pos2[i].y || pos1[i].z != pos2[i].z) {
                    multipolesAreValid = false;
                    break;
                }
        }
        else {
            vector<float4> pos1, pos2;
            cu.getPosq().download(pos1);
            lastPositions.download(pos2);
            for (int i = 0; i < numParticles; i++)
                if (pos1[i].x != pos2[i].x || pos1[i].y != pos2[i].y || pos1[i].z != pos2[i].z) {
                    multipolesAreValid = false;
                    break;
                }
        }
    }
    if (!multipolesAreValid)
        context.calcForcesAndEnergy(false, false, context.getIntegrator().getIntegrationForceGroups());
}

void CudaCalcAmoebaMultipoleForceKernel::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ensureMultipolesValid(context);
    int numParticles = cu.getNumAtoms();
    dipoles.resize(numParticles);
    const vector<int>& order = cu.getAtomIndex();
    if (cu.getUseDoublePrecision()) {
        vector<double3> labDipoleVec;
        labDipoles.download(labDipoleVec);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(labDipoleVec[i].x, labDipoleVec[i].y, labDipoleVec[i].z);
    }
    else {
        vector<float3> labDipoleVec;
        labDipoles.download(labDipoleVec);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(labDipoleVec[i].x, labDipoleVec[i].y, labDipoleVec[i].z);
    }
}


void CudaCalcAmoebaMultipoleForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ensureMultipolesValid(context);
    int numParticles = cu.getNumAtoms();
    dipoles.resize(numParticles);
    const vector<int>& order = cu.getAtomIndex();
    if (cu.getUseDoublePrecision()) {
        vector<double3> d;
        inducedDipole.download(d);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(d[i].x, d[i].y, d[i].z);
    }
    else {
        vector<float3> d;
        inducedDipole.download(d);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(d[i].x, d[i].y, d[i].z);
    }
}


void CudaCalcAmoebaMultipoleForceKernel::getTotalDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ensureMultipolesValid(context);
    int numParticles = cu.getNumAtoms();
    dipoles.resize(numParticles);
    const vector<int>& order = cu.getAtomIndex();
    if (cu.getUseDoublePrecision()) {
        vector<double4> posqVec;
        vector<double3> labDipoleVec;
        vector<double3> inducedDipoleVec;
        double totalDipoleVecX;
        double totalDipoleVecY;
        double totalDipoleVecZ;
        inducedDipole.download(inducedDipoleVec);
        labDipoles.download(labDipoleVec);
        cu.getPosq().download(posqVec);
        for (int i = 0; i < numParticles; i++) {
            totalDipoleVecX = labDipoleVec[i].x + inducedDipoleVec[i].x;
            totalDipoleVecY = labDipoleVec[i].y + inducedDipoleVec[i].y;
            totalDipoleVecZ = labDipoleVec[i].z + inducedDipoleVec[i].z;
            dipoles[order[i]] = Vec3(totalDipoleVecX, totalDipoleVecY, totalDipoleVecZ);
        }
    }
    else {
        vector<float4> posqVec;
        vector<float3> labDipoleVec;
        vector<float3> inducedDipoleVec;
        float totalDipoleVecX;
        float totalDipoleVecY;
        float totalDipoleVecZ;
        inducedDipole.download(inducedDipoleVec);
        labDipoles.download(labDipoleVec);
        cu.getPosq().download(posqVec);
        for (int i = 0; i < numParticles; i++) {
            totalDipoleVecX = labDipoleVec[i].x + inducedDipoleVec[i].x;
            totalDipoleVecY = labDipoleVec[i].y + inducedDipoleVec[i].y;
            totalDipoleVecZ = labDipoleVec[i].z + inducedDipoleVec[i].z;
            dipoles[order[i]] = Vec3(totalDipoleVecX, totalDipoleVecY, totalDipoleVecZ);
        }
    }
}

void CudaCalcAmoebaMultipoleForceKernel::getElectrostaticPotential(ContextImpl& context, const vector<Vec3>& inputGrid, vector<double>& outputElectrostaticPotential) {
    ensureMultipolesValid(context);
    int numPoints = inputGrid.size();
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    CudaArray points(cu, numPoints, 4*elementSize, "points");
    CudaArray potential(cu, numPoints, elementSize, "potential");
    
    // Copy the grid points to the GPU.
    
    if (cu.getUseDoublePrecision()) {
        vector<double4> p(numPoints);
        for (int i = 0; i < numPoints; i++)
            p[i] = make_double4(inputGrid[i][0], inputGrid[i][1], inputGrid[i][2], 0);
        points.upload(p);
    }
    else {
        vector<float4> p(numPoints);
        for (int i = 0; i < numPoints; i++)
            p[i] = make_float4((float) inputGrid[i][0], (float) inputGrid[i][1], (float) inputGrid[i][2], 0);
        points.upload(p);
    }
    
    // Compute the potential.
    
    computePotentialKernel->setArg(4, points);
    computePotentialKernel->setArg(5, potential);
    computePotentialKernel->setArg(6, numPoints);
    setPeriodicBoxArgs(cu, computePotentialKernel, 7);
    computePotentialKernel->execute(numPoints, 128);
    outputElectrostaticPotential.resize(numPoints);
    if (cu.getUseDoublePrecision())
        potential.download(outputElectrostaticPotential);
    else {
        vector<float> p(numPoints);
        potential.download(p);
        for (int i = 0; i < numPoints; i++)
            outputElectrostaticPotential[i] = p[i];
    }
}

template <class T, class T3, class T4, class M4>
void CudaCalcAmoebaMultipoleForceKernel::computeSystemMultipoleMoments(ContextImpl& context, vector<double>& outputMultipoleMoments) {
    // Compute the local coordinates relative to the center of mass.
    int numAtoms = cu.getNumAtoms();
    vector<T4> posq;
    vector<M4> velm;
    cu.getPosq().download(posq);
    cu.getVelm().download(velm);
    double totalMass = 0.0;
    Vec3 centerOfMass(0, 0, 0);
    for (int i = 0; i < numAtoms; i++) {
        double mass = (velm[i].w > 0 ? 1.0/velm[i].w : 0.0);
        totalMass += mass;
        centerOfMass[0] += mass*posq[i].x;
        centerOfMass[1] += mass*posq[i].y;
        centerOfMass[2] += mass*posq[i].z;
    }
    if (totalMass > 0.0) {
        centerOfMass[0] /= totalMass;
        centerOfMass[1] /= totalMass;
        centerOfMass[2] /= totalMass;
    }
    vector<double4> posqLocal(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        posqLocal[i].x = posq[i].x - centerOfMass[0];
        posqLocal[i].y = posq[i].y - centerOfMass[1];
        posqLocal[i].z = posq[i].z - centerOfMass[2];
        posqLocal[i].w = posq[i].w;
    }

    // Compute the multipole moments.
    
    double totalCharge = 0.0;
    double xdpl = 0.0;
    double ydpl = 0.0;
    double zdpl = 0.0;
    double xxqdp = 0.0;
    double xyqdp = 0.0;
    double xzqdp = 0.0;
    double yxqdp = 0.0;
    double yyqdp = 0.0;
    double yzqdp = 0.0;
    double zxqdp = 0.0;
    double zyqdp = 0.0;
    double zzqdp = 0.0;
    vector<T3> labDipoleVec, inducedDipoleVec;
    vector<T> quadrupoleVec;
    labDipoles.download(labDipoleVec);
    inducedDipole.download(inducedDipoleVec);
    labQuadrupoles.download(quadrupoleVec);
    for (int i = 0; i < numAtoms; i++) {
        totalCharge += posqLocal[i].w;
        double netDipoleX = (labDipoleVec[i].x + inducedDipoleVec[i].x);
        double netDipoleY = (labDipoleVec[i].y + inducedDipoleVec[i].y);
        double netDipoleZ = (labDipoleVec[i].z + inducedDipoleVec[i].z);
        xdpl += posqLocal[i].x*posqLocal[i].w + netDipoleX;
        ydpl += posqLocal[i].y*posqLocal[i].w + netDipoleY;
        zdpl += posqLocal[i].z*posqLocal[i].w + netDipoleZ;
        xxqdp += posqLocal[i].x*posqLocal[i].x*posqLocal[i].w + 2*posqLocal[i].x*netDipoleX;
        xyqdp += posqLocal[i].x*posqLocal[i].y*posqLocal[i].w + posqLocal[i].x*netDipoleY + posqLocal[i].y*netDipoleX;
        xzqdp += posqLocal[i].x*posqLocal[i].z*posqLocal[i].w + posqLocal[i].x*netDipoleZ + posqLocal[i].z*netDipoleX;
        yxqdp += posqLocal[i].y*posqLocal[i].x*posqLocal[i].w + posqLocal[i].y*netDipoleX + posqLocal[i].x*netDipoleY;
        yyqdp += posqLocal[i].y*posqLocal[i].y*posqLocal[i].w + 2*posqLocal[i].y*netDipoleY;
        yzqdp += posqLocal[i].y*posqLocal[i].z*posqLocal[i].w + posqLocal[i].y*netDipoleZ + posqLocal[i].z*netDipoleY;
        zxqdp += posqLocal[i].z*posqLocal[i].x*posqLocal[i].w + posqLocal[i].z*netDipoleX + posqLocal[i].x*netDipoleZ;
        zyqdp += posqLocal[i].z*posqLocal[i].y*posqLocal[i].w + posqLocal[i].z*netDipoleY + posqLocal[i].y*netDipoleZ;
        zzqdp += posqLocal[i].z*posqLocal[i].z*posqLocal[i].w + 2*posqLocal[i].z*netDipoleZ;
    }

    // Convert the quadrupole from traced to traceless form.
 
    double qave = (xxqdp + yyqdp + zzqdp)/3;
    xxqdp = 1.5*(xxqdp-qave);
    xyqdp = 1.5*xyqdp;
    xzqdp = 1.5*xzqdp;
    yxqdp = 1.5*yxqdp;
    yyqdp = 1.5*(yyqdp-qave);
    yzqdp = 1.5*yzqdp;
    zxqdp = 1.5*zxqdp;
    zyqdp = 1.5*zyqdp;
    zzqdp = 1.5*(zzqdp-qave);

    // Add the traceless atomic quadrupoles to the total quadrupole moment.

    for (int i = 0; i < numAtoms; i++) {
        xxqdp = xxqdp + 3*quadrupoleVec[5*i];
        xyqdp = xyqdp + 3*quadrupoleVec[5*i+1];
        xzqdp = xzqdp + 3*quadrupoleVec[5*i+2];
        yxqdp = yxqdp + 3*quadrupoleVec[5*i+1];
        yyqdp = yyqdp + 3*quadrupoleVec[5*i+3];
        yzqdp = yzqdp + 3*quadrupoleVec[5*i+4];
        zxqdp = zxqdp + 3*quadrupoleVec[5*i+2];
        zyqdp = zyqdp + 3*quadrupoleVec[5*i+4];
        zzqdp = zzqdp + -3*(quadrupoleVec[5*i]+quadrupoleVec[5*i+3]);
    }
 
    double debye = 4.80321;
    outputMultipoleMoments.resize(13);
    outputMultipoleMoments[0] = totalCharge;
    outputMultipoleMoments[1] = 10.0*xdpl*debye;
    outputMultipoleMoments[2] = 10.0*ydpl*debye;
    outputMultipoleMoments[3] = 10.0*zdpl*debye;
    outputMultipoleMoments[4] = 100.0*xxqdp*debye;
    outputMultipoleMoments[5] = 100.0*xyqdp*debye;
    outputMultipoleMoments[6] = 100.0*xzqdp*debye;
    outputMultipoleMoments[7] = 100.0*yxqdp*debye;
    outputMultipoleMoments[8] = 100.0*yyqdp*debye;
    outputMultipoleMoments[9] = 100.0*yzqdp*debye;
    outputMultipoleMoments[10] = 100.0*zxqdp*debye;
    outputMultipoleMoments[11] = 100.0*zyqdp*debye;
    outputMultipoleMoments[12] = 100.0*zzqdp*debye;
}


void CudaCalcAmoebaMultipoleForceKernel::getSystemMultipoleMoments(ContextImpl& context, vector<double>& outputMultipoleMoments) {
    ensureMultipolesValid(context);
    if (cu.getUseDoublePrecision())
        computeSystemMultipoleMoments<double, double3, double4, double4>(context, outputMultipoleMoments);
    else if (cu.getUseMixedPrecision())
        computeSystemMultipoleMoments<float, float3, float4, double4>(context, outputMultipoleMoments);
    else
        computeSystemMultipoleMoments<float, float3, float4, float4>(context, outputMultipoleMoments);
}

void CudaCalcAmoebaMultipoleForceKernel::copyParametersToContext(ContextImpl& context, const AmoebaMultipoleForce& force) {
    // Make sure the new parameters are acceptable.
    
    cu.setAsCurrent();
    if (force.getNumMultipoles() != cu.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of multipoles has changed");
    
    // Record the per-multipole parameters.
    
    cu.getPosq().download(cu.getPinnedBuffer());
    float4* posqf = (float4*) cu.getPinnedBuffer();
    double4* posqd = (double4*) cu.getPinnedBuffer();
    vector<float2> dampingAndTholeVec;
    vector<float> polarizabilityVec;
    vector<float> localDipolesVec;
    vector<float> localQuadrupolesVec;
    vector<int4> multipoleParticlesVec;
    for (int i = 0; i < force.getNumMultipoles(); i++) {
        double charge, thole, damping, polarity;
        int axisType, atomX, atomY, atomZ;
        vector<double> dipole, quadrupole;
        force.getMultipoleParameters(i, charge, dipole, quadrupole, axisType, atomZ, atomX, atomY, thole, damping, polarity);
        if (cu.getUseDoublePrecision())
            posqd[i].w = charge;
        else
            posqf[i].w = (float) charge;
        dampingAndTholeVec.push_back(make_float2((float) damping, (float) thole));
        polarizabilityVec.push_back((float) polarity);
        multipoleParticlesVec.push_back(make_int4(atomX, atomY, atomZ, axisType));
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back((float) dipole[j]);
        localQuadrupolesVec.push_back((float) quadrupole[0]);
        localQuadrupolesVec.push_back((float) quadrupole[1]);
        localQuadrupolesVec.push_back((float) quadrupole[2]);
        localQuadrupolesVec.push_back((float) quadrupole[4]);
        localQuadrupolesVec.push_back((float) quadrupole[5]);
    }
    if (!hasQuadrupoles) {
        for (auto q : localQuadrupolesVec)
            if (q != 0.0)
                throw OpenMMException("updateParametersInContext: Cannot set a non-zero quadrupole moment, because quadrupoles were excluded from the kernel");
    }
    for (int i = force.getNumMultipoles(); i < cu.getPaddedNumAtoms(); i++) {
        dampingAndTholeVec.push_back(make_float2(0, 0));
        polarizabilityVec.push_back(0);
        multipoleParticlesVec.push_back(make_int4(0, 0, 0, 0));
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back(0);
        for (int j = 0; j < 5; j++)
            localQuadrupolesVec.push_back(0);
    }
    dampingAndThole.upload(dampingAndTholeVec);
    polarizability.upload(polarizabilityVec);
    multipoleParticles.upload(multipoleParticlesVec);
    localDipoles.upload(localDipolesVec);
    localQuadrupoles.upload(localQuadrupolesVec);
    cu.getPosq().upload(cu.getPinnedBuffer());
    cu.invalidateMolecules();
    multipolesAreValid = false;
}

void CudaCalcAmoebaMultipoleForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (!usePME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    alpha = pmeAlpha;
    nx = gridSizeX;
    ny = gridSizeY;
    nz = gridSizeZ;
}

/* -------------------------------------------------------------------------- *
 *                       AmoebaGeneralizedKirkwood                            *
 * -------------------------------------------------------------------------- */

class CudaCalcAmoebaGeneralizedKirkwoodForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const AmoebaGeneralizedKirkwoodForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double charge1, charge2, radius1, radius2, scale1, scale2;
        force.getParticleParameters(particle1, charge1, radius1, scale1);
        force.getParticleParameters(particle2, charge2, radius2, scale2);
        return (charge1 == charge2 && radius1 == radius2 && scale1 == scale2);
    }
private:
    const AmoebaGeneralizedKirkwoodForce& force;
};

CudaCalcAmoebaGeneralizedKirkwoodForceKernel::CudaCalcAmoebaGeneralizedKirkwoodForceKernel(const std::string& name, const Platform& platform, CudaContext& cu, const System& system) :
           CalcAmoebaGeneralizedKirkwoodForceKernel(name, platform), cu(cu), system(system), hasInitializedKernels(false) {
}

void CudaCalcAmoebaGeneralizedKirkwoodForceKernel::initialize(const System& system, const AmoebaGeneralizedKirkwoodForce& force) {
    cu.setAsCurrent();
    if (cu.getPlatformData().contexts.size() > 1)
        throw OpenMMException("AmoebaGeneralizedKirkwoodForce does not support using multiple CUDA devices");
    const AmoebaMultipoleForce* multipoles = NULL;
    for (int i = 0; i < system.getNumForces() && multipoles == NULL; i++)
        multipoles = dynamic_cast<const AmoebaMultipoleForce*>(&system.getForce(i));
    if (multipoles == NULL)
        throw OpenMMException("AmoebaGeneralizedKirkwoodForce requires the System to also contain an AmoebaMultipoleForce");
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    params.initialize<float2>(cu, paddedNumAtoms, "amoebaGkParams");
    bornRadii .initialize(cu, paddedNumAtoms, elementSize, "bornRadii");
    field .initialize(cu, 3*paddedNumAtoms, sizeof(long long), "gkField");
    bornSum.initialize<long long>(cu, paddedNumAtoms, "bornSum");
    bornForce.initialize<long long>(cu, paddedNumAtoms, "bornForce");
    inducedDipoleS .initialize(cu, 3*paddedNumAtoms, elementSize, "inducedDipoleS");
    inducedDipolePolarS .initialize(cu, 3*paddedNumAtoms, elementSize, "inducedDipolePolarS");
    polarizationType = multipoles->getPolarizationType();
    if (polarizationType != AmoebaMultipoleForce::Direct) {
        inducedField .initialize(cu, 3*paddedNumAtoms, sizeof(long long), "gkInducedField");
        inducedFieldPolar .initialize(cu, 3*paddedNumAtoms, sizeof(long long), "gkInducedFieldPolar");
    }
    cu.addAutoclearBuffer(field);
    cu.addAutoclearBuffer(bornSum);
    cu.addAutoclearBuffer(bornForce);
    vector<float2> paramsVector(paddedNumAtoms);
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, radius, scalingFactor;
        force.getParticleParameters(i, charge, radius, scalingFactor);
        paramsVector[i] = make_float2((float) radius, (float) (scalingFactor*radius));
        
        // Make sure the charge matches the one specified by the AmoebaMultipoleForce.
        
        double charge2, thole, damping, polarity;
        int axisType, atomX, atomY, atomZ;
        vector<double> dipole, quadrupole;
        multipoles->getMultipoleParameters(i, charge2, dipole, quadrupole, axisType, atomZ, atomX, atomY, thole, damping, polarity);
        if (charge != charge2)
            throw OpenMMException("AmoebaGeneralizedKirkwoodForce and AmoebaMultipoleForce must specify the same charge for every atom");
    }
    params.upload(paramsVector);
    
    // Select the number of threads for each kernel.
    
    double computeBornSumThreadMemory = 4*elementSize+3*sizeof(float);
    double gkForceThreadMemory = 24*elementSize;
    double chainRuleThreadMemory = 10*elementSize;
    double ediffThreadMemory = 28*elementSize+2*sizeof(float)+3*sizeof(int)/(double) cu.TileSize;
    int maxThreads = cu.getNonbondedUtilities().getForceThreadBlockSize();
    computeBornSumThreads = min(maxThreads, cu.computeThreadBlockSize(computeBornSumThreadMemory));
    gkForceThreads = min(maxThreads, cu.computeThreadBlockSize(gkForceThreadMemory));
    chainRuleThreads = min(maxThreads, cu.computeThreadBlockSize(chainRuleThreadMemory));
    ediffThreads = min(maxThreads, cu.computeThreadBlockSize(ediffThreadMemory));
    
    // Set preprocessor macros we will use when we create the kernels.
    
    defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(paddedNumAtoms);
    defines["BORN_SUM_THREAD_BLOCK_SIZE"] = cu.intToString(computeBornSumThreads);
    defines["GK_FORCE_THREAD_BLOCK_SIZE"] = cu.intToString(gkForceThreads);
    defines["CHAIN_RULE_THREAD_BLOCK_SIZE"] = cu.intToString(chainRuleThreads);
    defines["EDIFF_THREAD_BLOCK_SIZE"] = cu.intToString(ediffThreads);
    defines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
    defines["GK_C"] = cu.doubleToString(2.455);
    double solventDielectric = force.getSolventDielectric();
    defines["GK_FC"] = cu.doubleToString(1*(1-solventDielectric)/(0+1*solventDielectric));
    defines["GK_FD"] = cu.doubleToString(2*(1-solventDielectric)/(1+2*solventDielectric));
    defines["GK_FQ"] = cu.doubleToString(3*(1-solventDielectric)/(2+3*solventDielectric));
    defines["EPSILON_FACTOR"] = cu.doubleToString(ONE_4PI_EPS0);
    defines["M_PI"] = cu.doubleToString(M_PI);
    defines["ENERGY_SCALE_FACTOR"] = cu.doubleToString(ONE_4PI_EPS0/force.getSoluteDielectric());
    if (polarizationType == AmoebaMultipoleForce::Direct)
        defines["DIRECT_POLARIZATION"] = "";
    else if (polarizationType == AmoebaMultipoleForce::Mutual)
        defines["MUTUAL_POLARIZATION"] = "";
    else if (polarizationType == AmoebaMultipoleForce::Extrapolated)
        defines["EXTRAPOLATED_POLARIZATION"] = "";
    includeSurfaceArea = force.getIncludeCavityTerm();
    if (includeSurfaceArea) {
        defines["SURFACE_AREA_FACTOR"] = cu.doubleToString(force.getSurfaceAreaFactor());
        defines["PROBE_RADIUS"] = cu.doubleToString(force.getProbeRadius());
        defines["DIELECTRIC_OFFSET"] = cu.doubleToString(0.009);
    }
    cu.addForce(new ForceInfo(force));
}

double CudaCalcAmoebaGeneralizedKirkwoodForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // Since GK is so tightly entwined with the electrostatics, this method does nothing, and the force calculation
    // is driven by AmoebaMultipoleForce.
    return 0.0;
}

void CudaCalcAmoebaGeneralizedKirkwoodForceKernel::computeBornRadii() {
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        
        // Create the kernels.
        
        int numExclusionTiles = cu.getNonbondedUtilities().getExclusionTiles().getSize();
        defines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(numExclusionTiles);
        int numContexts = cu.getPlatformData().contexts.size();
        int startExclusionIndex = cu.getContextIndex()*numExclusionTiles/numContexts;
        int endExclusionIndex = (cu.getContextIndex()+1)*numExclusionTiles/numContexts;
        defines["FIRST_EXCLUSION_TILE"] = cu.intToString(startExclusionIndex);
        defines["LAST_EXCLUSION_TILE"] = cu.intToString(endExclusionIndex);
        stringstream forceSource;
        forceSource << CudaKernelSources::vectorOps;
        forceSource << CudaAmoebaKernelSources::amoebaGk;
        forceSource << "#define F1\n";
        forceSource << CudaAmoebaKernelSources::gkPairForce1;
        forceSource << CudaAmoebaKernelSources::gkPairForce2;
        forceSource << CudaAmoebaKernelSources::gkEDiffPairForce;
        forceSource << "#undef F1\n";
        forceSource << "#define F2\n";
        forceSource << CudaAmoebaKernelSources::gkPairForce1;
        forceSource << CudaAmoebaKernelSources::gkPairForce2;
        forceSource << "#undef F2\n";
        forceSource << "#define T1\n";
        forceSource << CudaAmoebaKernelSources::gkPairForce1;
        forceSource << CudaAmoebaKernelSources::gkPairForce2;
        forceSource << CudaAmoebaKernelSources::gkEDiffPairForce;
        forceSource << "#undef T1\n";
        forceSource << "#define T2\n";
        forceSource << CudaAmoebaKernelSources::gkPairForce1;
        forceSource << CudaAmoebaKernelSources::gkPairForce2;
        forceSource << "#undef T2\n";
        forceSource << "#define T3\n";
        forceSource << CudaAmoebaKernelSources::gkEDiffPairForce;
        forceSource << "#undef T3\n";
        forceSource << "#define B1\n";
        forceSource << "#define B2\n";
        forceSource << CudaAmoebaKernelSources::gkPairForce1;
        forceSource << CudaAmoebaKernelSources::gkPairForce2;
        CUmodule module = cu.createModule(forceSource.str(), defines);
        computeBornSumKernel = cu.getKernel(module, "computeBornSum");
        reduceBornSumKernel = cu.getKernel(module, "reduceBornSum");
        gkForceKernel = cu.getKernel(module, "computeGKForces");
        chainRuleKernel = cu.getKernel(module, "computeChainRuleForce");
        ediffKernel = cu.getKernel(module, "computeEDiffForce");
        if (includeSurfaceArea)
            surfaceAreaKernel = cu.getKernel(module, "computeSurfaceAreaForce");
    }
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    int numTiles = nb.getNumTiles();
    int numForceThreadBlocks = nb.getNumForceThreadBlocks();
    void* computeBornSumArgs[] = {&bornSum.getDevicePointer(), &cu.getPosq().getDevicePointer(),
        &params.getDevicePointer(), &numTiles};
    cu.executeKernel(computeBornSumKernel, computeBornSumArgs, numForceThreadBlocks*computeBornSumThreads, computeBornSumThreads);
    void* reduceBornSumArgs[] = {&bornSum.getDevicePointer(), &params.getDevicePointer(), &bornRadii.getDevicePointer()};
    cu.executeKernel(reduceBornSumKernel, reduceBornSumArgs, cu.getNumAtoms());
}

void CudaCalcAmoebaGeneralizedKirkwoodForceKernel::finishComputation(CudaArray& torque, CudaArray& labDipoles, CudaArray& labQuadrupoles,
            CudaArray& inducedDipole, CudaArray& inducedDipolePolar, CudaArray& dampingAndThole, CudaArray& covalentFlags, CudaArray& polarizationGroupFlags) {
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    int startTileIndex = nb.getStartTileIndex();
    int numTileIndices = nb.getNumTiles();
    int numForceThreadBlocks = nb.getNumForceThreadBlocks();
    
    // Compute the GK force.
    
    void* gkForceArgs[] = {&cu.getForce().getDevicePointer(), &torque.getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(),
        &cu.getPosq().getDevicePointer(), &startTileIndex, &numTileIndices, &labDipoles.getDevicePointer(),
        &labQuadrupoles.getDevicePointer(), &inducedDipoleS.getDevicePointer(), &inducedDipolePolarS.getDevicePointer(),
        &bornRadii.getDevicePointer(), &bornForce.getDevicePointer()};
    cu.executeKernel(gkForceKernel, gkForceArgs, numForceThreadBlocks*gkForceThreads, gkForceThreads);

    // Compute the surface area force.
    
    if (includeSurfaceArea) {
        void* surfaceAreaArgs[] = {&bornForce.getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(), &params.getDevicePointer(), &bornRadii.getDevicePointer()};
        cu.executeKernel(surfaceAreaKernel, surfaceAreaArgs, cu.getNumAtoms());
    }
    
    // Apply the remaining terms.
    
    void* chainRuleArgs[] = {&cu.getForce().getDevicePointer(), &cu.getPosq().getDevicePointer(), &startTileIndex, &numTileIndices,
        &params.getDevicePointer(), &bornRadii.getDevicePointer(), &bornForce.getDevicePointer()};
    cu.executeKernel(chainRuleKernel, chainRuleArgs, numForceThreadBlocks*chainRuleThreads, chainRuleThreads);    
    void* ediffArgs[] = {&cu.getForce().getDevicePointer(), &torque.getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(),
        &cu.getPosq().getDevicePointer(), &covalentFlags.getDevicePointer(), &polarizationGroupFlags.getDevicePointer(),
        &nb.getExclusionTiles().getDevicePointer(), &startTileIndex, &numTileIndices,
        &labDipoles.getDevicePointer(), &labQuadrupoles.getDevicePointer(), &inducedDipole.getDevicePointer(),
        &inducedDipolePolar.getDevicePointer(), &inducedDipoleS.getDevicePointer(), &inducedDipolePolarS.getDevicePointer(),
        &dampingAndThole.getDevicePointer()};
    cu.executeKernel(ediffKernel, ediffArgs, numForceThreadBlocks*ediffThreads, ediffThreads);
}

void CudaCalcAmoebaGeneralizedKirkwoodForceKernel::copyParametersToContext(ContextImpl& context, const AmoebaGeneralizedKirkwoodForce& force) {
    // Make sure the new parameters are acceptable.
    
    cu.setAsCurrent();
    if (force.getNumParticles() != cu.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    
    // Record the per-particle parameters.
    
    vector<float2> paramsVector(cu.getPaddedNumAtoms());
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, radius, scalingFactor;
        force.getParticleParameters(i, charge, radius, scalingFactor);
        paramsVector[i] = make_float2((float) radius, (float) (scalingFactor*radius));
    }
    params.upload(paramsVector);
    cu.invalidateMolecules();
}

/* -------------------------------------------------------------------------- *
 *                           HippoNonbondedForce                              *
 * -------------------------------------------------------------------------- */

class CudaCalcHippoNonbondedForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const HippoNonbondedForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double charge1, coreCharge1, alpha1, epsilon1, damping1, c61, pauliK1, pauliQ1, pauliAlpha1, polarizability1;
        double charge2, coreCharge2, alpha2, epsilon2, damping2, c62, pauliK2, pauliQ2, pauliAlpha2, polarizability2;
        int axisType1, multipoleZ1, multipoleX1, multipoleY1;
        int axisType2, multipoleZ2, multipoleX2, multipoleY2;
        vector<double> dipole1, dipole2, quadrupole1, quadrupole2;
        force.getParticleParameters(particle1, charge1, dipole1, quadrupole1, coreCharge1, alpha1, epsilon1, damping1, c61, pauliK1, pauliQ1, pauliAlpha1,
                                    polarizability1, axisType1, multipoleZ1, multipoleX1, multipoleY1);
        force.getParticleParameters(particle2, charge2, dipole2, quadrupole2, coreCharge2, alpha2, epsilon2, damping2, c62, pauliK2, pauliQ2, pauliAlpha2,
                                    polarizability2, axisType2, multipoleZ2, multipoleX2, multipoleY2);
        if (charge1 != charge2 || coreCharge1 != coreCharge2 || alpha1 != alpha2 || epsilon1 != epsilon1 || damping1 != damping2 || c61 != c62 ||
                pauliK1 != pauliK2 || pauliQ1 != pauliQ2 || pauliAlpha1 != pauliAlpha2 || polarizability1 != polarizability2 || axisType1 != axisType2) {
            return false;
        }
        for (int i = 0; i < dipole1.size(); ++i)
            if (dipole1[i] != dipole2[i])
                return false;
        for (int i = 0; i < quadrupole1.size(); ++i)
            if (quadrupole1[i] != quadrupole2[i])
                return false;
        return true;
    }
    int getNumParticleGroups() {
        return force.getNumExceptions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        double multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale, dispersionScale, repulsionScale, chargeTransferScale;
        force.getExceptionParameters(index, particle1, particle2, multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale, dispersionScale, repulsionScale, chargeTransferScale);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2;
        double multipoleMultipoleScale1, dipoleMultipoleScale1, dipoleDipoleScale1, dispersionScale1, repulsionScale1, chargeTransferScale1;
        double multipoleMultipoleScale2, dipoleMultipoleScale2, dipoleDipoleScale2, dispersionScale2, repulsionScale2, chargeTransferScale2;
        force.getExceptionParameters(group1, particle1, particle2, multipoleMultipoleScale1, dipoleMultipoleScale1, dipoleDipoleScale1, dispersionScale1, repulsionScale1, chargeTransferScale1);
        force.getExceptionParameters(group2, particle1, particle2, multipoleMultipoleScale2, dipoleMultipoleScale2, dipoleDipoleScale2, dispersionScale2, repulsionScale2, chargeTransferScale2);
        return (multipoleMultipoleScale1 == multipoleMultipoleScale2 && dipoleMultipoleScale1 == dipoleMultipoleScale2 &&
                dipoleDipoleScale1 == dipoleDipoleScale2 && dispersionScale1 == dispersionScale2 && repulsionScale1 == repulsionScale2 && chargeTransferScale1 == chargeTransferScale2);
    }
private:
    const HippoNonbondedForce& force;
};

class CudaCalcHippoNonbondedForceKernel::TorquePostComputation : public CudaContext::ForcePostComputation {
public:
    TorquePostComputation(CudaCalcHippoNonbondedForceKernel& owner) : owner(owner) {
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        owner.addTorquesToForces();
        return 0.0;
    }
private:
    CudaCalcHippoNonbondedForceKernel& owner;
};

CudaCalcHippoNonbondedForceKernel::CudaCalcHippoNonbondedForceKernel(const std::string& name, const Platform& platform, CudaContext& cu, const System& system) :
        CalcHippoNonbondedForceKernel(name, platform), cu(cu), system(system), sort(NULL), hasInitializedKernels(false), hasInitializedFFT(false), multipolesAreValid(false) {
}

CudaCalcHippoNonbondedForceKernel::~CudaCalcHippoNonbondedForceKernel() {
    cu.setAsCurrent();
    if (sort != NULL)
        delete sort;
    if (hasInitializedFFT) {
        cufftDestroy(fftForward);
        cufftDestroy(fftBackward);
        cufftDestroy(dfftForward);
        cufftDestroy(dfftBackward);
    }
}

void CudaCalcHippoNonbondedForceKernel::initialize(const System& system, const HippoNonbondedForce& force) {
    cu.setAsCurrent();
    extrapolationCoefficients = force.getExtrapolationCoefficients();
    usePME = (force.getNonbondedMethod() == HippoNonbondedForce::PME);

    // Initialize particle parameters.

    numParticles = force.getNumParticles();
    vector<double> coreChargeVec, valenceChargeVec, alphaVec, epsilonVec, dampingVec, c6Vec, pauliKVec, pauliQVec, pauliAlphaVec, polarizabilityVec;
    vector<double> localDipolesVec, localQuadrupolesVec;
    vector<int4> multipoleParticlesVec;
    vector<vector<int> > exclusions(numParticles);
    for (int i = 0; i < numParticles; i++) {
        double charge, coreCharge, alpha, epsilon, damping, c6, pauliK, pauliQ, pauliAlpha, polarizability;
        int axisType, atomX, atomY, atomZ;
        vector<double> dipole, quadrupole;
        force.getParticleParameters(i, charge, dipole, quadrupole, coreCharge, alpha, epsilon, damping, c6, pauliK, pauliQ, pauliAlpha,
                                    polarizability, axisType, atomZ, atomX, atomY);
        coreChargeVec.push_back(coreCharge);
        valenceChargeVec.push_back(charge-coreCharge);
        alphaVec.push_back(alpha);
        epsilonVec.push_back(epsilon);
        dampingVec.push_back(damping);
        c6Vec.push_back(c6);
        pauliKVec.push_back(pauliK);
        pauliQVec.push_back(pauliQ);
        pauliAlphaVec.push_back(pauliAlpha);
        polarizabilityVec.push_back(polarizability);
        multipoleParticlesVec.push_back(make_int4(atomX, atomY, atomZ, axisType));
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back(dipole[j]);
        localQuadrupolesVec.push_back(quadrupole[0]);
        localQuadrupolesVec.push_back(quadrupole[1]);
        localQuadrupolesVec.push_back(quadrupole[2]);
        localQuadrupolesVec.push_back(quadrupole[4]);
        localQuadrupolesVec.push_back(quadrupole[5]);
        exclusions[i].push_back(i);
    }
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    for (int i = numParticles; i < paddedNumAtoms; i++) {
        coreChargeVec.push_back(0);
        valenceChargeVec.push_back(0);
        alphaVec.push_back(0);
        epsilonVec.push_back(0);
        dampingVec.push_back(0);
        c6Vec.push_back(0);
        pauliKVec.push_back(0);
        pauliQVec.push_back(0);
        pauliAlphaVec.push_back(0);
        polarizabilityVec.push_back(0);
        multipoleParticlesVec.push_back(make_int4(0, 0, 0, 0));
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back(0);
        for (int j = 0; j < 5; j++)
            localQuadrupolesVec.push_back(0);
    }
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    coreCharge.initialize(cu, paddedNumAtoms, elementSize, "coreCharge");
    valenceCharge.initialize(cu, paddedNumAtoms, elementSize, "valenceCharge");
    alpha.initialize(cu, paddedNumAtoms, elementSize, "alpha");
    epsilon.initialize(cu, paddedNumAtoms, elementSize, "epsilon");
    damping.initialize(cu, paddedNumAtoms, elementSize, "damping");
    c6.initialize(cu, paddedNumAtoms, elementSize, "c6");
    pauliK.initialize(cu, paddedNumAtoms, elementSize, "pauliK");
    pauliQ.initialize(cu, paddedNumAtoms, elementSize, "pauliQ");
    pauliAlpha.initialize(cu, paddedNumAtoms, elementSize, "pauliAlpha");
    polarizability.initialize(cu, paddedNumAtoms, elementSize, "polarizability");
    multipoleParticles.initialize<int4>(cu, paddedNumAtoms, "multipoleParticles");
    localDipoles.initialize(cu, 3*paddedNumAtoms, elementSize, "localDipoles");
    localQuadrupoles.initialize(cu, 5*paddedNumAtoms, elementSize, "localQuadrupoles");
    lastPositions.initialize(cu, cu.getPosq().getSize(), cu.getPosq().getElementSize(), "lastPositions");
    coreCharge.upload(coreChargeVec, true);
    valenceCharge.upload(valenceChargeVec, true);
    alpha.upload(alphaVec, true);
    epsilon.upload(epsilonVec, true);
    damping.upload(dampingVec, true);
    c6.upload(c6Vec, true);
    pauliK.upload(pauliKVec, true);
    pauliQ.upload(pauliQVec, true);
    pauliAlpha.upload(pauliAlphaVec, true);
    polarizability.upload(polarizabilityVec, true);
    multipoleParticles.upload(multipoleParticlesVec);
    localDipoles.upload(localDipolesVec, true);
    localQuadrupoles.upload(localQuadrupolesVec, true);
    
    // Create workspace arrays.
    
    labDipoles.initialize(cu, paddedNumAtoms, 3*elementSize, "dipole");
    labQuadrupoles[0].initialize(cu, paddedNumAtoms, elementSize, "qXX");
    labQuadrupoles[1].initialize(cu, paddedNumAtoms, elementSize, "qXY");
    labQuadrupoles[2].initialize(cu, paddedNumAtoms, elementSize, "qXZ");
    labQuadrupoles[3].initialize(cu, paddedNumAtoms, elementSize, "qYY");
    labQuadrupoles[4].initialize(cu, paddedNumAtoms, elementSize, "qYZ");
    fracDipoles.initialize(cu, paddedNumAtoms, 3*elementSize, "fracDipoles");
    fracQuadrupoles.initialize(cu, 6*paddedNumAtoms, elementSize, "fracQuadrupoles");
    field.initialize(cu, 3*paddedNumAtoms, sizeof(long long), "field");
    inducedField.initialize(cu, 3*paddedNumAtoms, sizeof(long long), "inducedField");
    torque.initialize(cu, 3*paddedNumAtoms, sizeof(long long), "torque");
    inducedDipole.initialize(cu, paddedNumAtoms, 3*elementSize, "inducedDipole");
    int numOrders = extrapolationCoefficients.size();
    extrapolatedDipole.initialize(cu, 3*numParticles*numOrders, elementSize, "extrapolatedDipole");
    extrapolatedPhi.initialize(cu, 10*numParticles*numOrders, elementSize, "extrapolatedPhi");
    cu.addAutoclearBuffer(field);
    cu.addAutoclearBuffer(torque);
    
    // Record exceptions and exclusions.
    
    vector<double> exceptionScaleVec[6];
    vector<int2> exceptionAtomsVec;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale, dispersionScale, repulsionScale, chargeTransferScale;
        force.getExceptionParameters(i, particle1, particle2, multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale, dispersionScale, repulsionScale, chargeTransferScale);
        exclusions[particle1].push_back(particle2);
        exclusions[particle2].push_back(particle1);
        if (usePME || multipoleMultipoleScale != 0 || dipoleMultipoleScale != 0 || dipoleDipoleScale != 0 || dispersionScale != 0 || repulsionScale != 0 || chargeTransferScale != 0) {
            exceptionAtomsVec.push_back(make_int2(particle1, particle2));
            exceptionScaleVec[0].push_back(multipoleMultipoleScale);
            exceptionScaleVec[1].push_back(dipoleMultipoleScale);
            exceptionScaleVec[2].push_back(dipoleDipoleScale);
            exceptionScaleVec[3].push_back(dispersionScale);
            exceptionScaleVec[4].push_back(repulsionScale);
            exceptionScaleVec[5].push_back(chargeTransferScale);
        }
    }
    if (exceptionAtomsVec.size() > 0) {
        exceptionAtoms.initialize<int2>(cu, exceptionAtomsVec.size(), "exceptionAtoms");
        exceptionAtoms.upload(exceptionAtomsVec);
        for (int i = 0; i < 6; i++) {
            exceptionScales[i].initialize(cu, exceptionAtomsVec.size(), elementSize, "exceptionScales");
            exceptionScales[i].upload(exceptionScaleVec[i], true);
        }
    }
    
    // Create the kernels.

    bool useShuffle = (cu.getComputeCapability() >= 3.0 && !cu.getUseDoublePrecision());
    map<string, string> defines;
    defines["HIPPO"] = "1";
    defines["NUM_ATOMS"] = cu.intToString(numParticles);
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    defines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
    defines["ENERGY_SCALE_FACTOR"] = cu.doubleToString(ONE_4PI_EPS0);
    if (useShuffle)
        defines["USE_SHUFFLE"] = "";
    maxExtrapolationOrder = extrapolationCoefficients.size();
    defines["MAX_EXTRAPOLATION_ORDER"] = cu.intToString(maxExtrapolationOrder);
    stringstream coefficients;
    for (int i = 0; i < maxExtrapolationOrder; i++) {
        if (i > 0)
            coefficients << ",";
        double sum = 0;
        for (int j = i; j < maxExtrapolationOrder; j++)
            sum += extrapolationCoefficients[j];
        coefficients << cu.doubleToString(sum);
    }
    defines["EXTRAPOLATION_COEFFICIENTS_SUM"] = coefficients.str();
    cutoff = force.getCutoffDistance();
    if (usePME) {
        int nx, ny, nz;
        force.getPMEParameters(pmeAlpha, nx, ny, nz);
        if (nx == 0 || pmeAlpha == 0) {
            NonbondedForce nb;
            nb.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
            nb.setCutoffDistance(force.getCutoffDistance());
            NonbondedForceImpl::calcPMEParameters(system, nb, pmeAlpha, gridSizeX, gridSizeY, gridSizeZ, false);
            gridSizeX = CudaFFT3D::findLegalDimension(gridSizeX);
            gridSizeY = CudaFFT3D::findLegalDimension(gridSizeY);
            gridSizeZ = CudaFFT3D::findLegalDimension(gridSizeZ);
        } else {
            gridSizeX = CudaFFT3D::findLegalDimension(nx);
            gridSizeY = CudaFFT3D::findLegalDimension(ny);
            gridSizeZ = CudaFFT3D::findLegalDimension(nz);
        }
        force.getDPMEParameters(dpmeAlpha, nx, ny, nz);
        if (nx == 0 || dpmeAlpha == 0) {
            NonbondedForce nb;
            nb.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
            nb.setCutoffDistance(force.getCutoffDistance());
            NonbondedForceImpl::calcPMEParameters(system, nb, dpmeAlpha, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, true);
            dispersionGridSizeX = CudaFFT3D::findLegalDimension(dispersionGridSizeX);
            dispersionGridSizeY = CudaFFT3D::findLegalDimension(dispersionGridSizeY);
            dispersionGridSizeZ = CudaFFT3D::findLegalDimension(dispersionGridSizeZ);
        } else {
            dispersionGridSizeX = CudaFFT3D::findLegalDimension(nx);
            dispersionGridSizeY = CudaFFT3D::findLegalDimension(ny);
            dispersionGridSizeZ = CudaFFT3D::findLegalDimension(nz);
        }
        defines["EWALD_ALPHA"] = cu.doubleToString(pmeAlpha);
        defines["SQRT_PI"] = cu.doubleToString(sqrt(M_PI));
        defines["USE_EWALD"] = "";
        defines["USE_CUTOFF"] = "";
        defines["USE_PERIODIC"] = "";
        defines["CUTOFF_SQUARED"] = cu.doubleToString(force.getCutoffDistance()*force.getCutoffDistance());
    }
    CUmodule module = cu.createModule(CudaKernelSources::vectorOps+CudaAmoebaKernelSources::hippoMultipoles, defines);
    computeMomentsKernel = cu.getKernel(module, "computeLabFrameMoments");
    recordInducedDipolesKernel = cu.getKernel(module, "recordInducedDipoles");
    mapTorqueKernel = cu.getKernel(module, "mapTorqueToForce");
    module = cu.createModule(CudaKernelSources::vectorOps+CudaAmoebaKernelSources::multipoleInducedField, defines);
    initExtrapolatedKernel = cu.getKernel(module, "initExtrapolatedDipoles");
    iterateExtrapolatedKernel = cu.getKernel(module, "iterateExtrapolatedDipoles");
    computeExtrapolatedKernel = cu.getKernel(module, "computeExtrapolatedDipoles");
    polarizationEnergyKernel = cu.getKernel(module, "computePolarizationEnergy");

    // Set up PME.
    
    if (usePME) {
        // Create the PME kernels.

        map<string, string> pmeDefines;
        pmeDefines["HIPPO"] = "1";
        pmeDefines["EWALD_ALPHA"] = cu.doubleToString(pmeAlpha);
        pmeDefines["DISPERSION_EWALD_ALPHA"] = cu.doubleToString(dpmeAlpha);
        pmeDefines["PME_ORDER"] = cu.intToString(PmeOrder);
        pmeDefines["NUM_ATOMS"] = cu.intToString(numParticles);
        pmeDefines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        pmeDefines["EPSILON_FACTOR"] = cu.doubleToString(ONE_4PI_EPS0);
        pmeDefines["GRID_SIZE_X"] = cu.intToString(gridSizeX);
        pmeDefines["GRID_SIZE_Y"] = cu.intToString(gridSizeY);
        pmeDefines["GRID_SIZE_Z"] = cu.intToString(gridSizeZ);
        pmeDefines["M_PI"] = cu.doubleToString(M_PI);
        pmeDefines["SQRT_PI"] = cu.doubleToString(sqrt(M_PI));
        pmeDefines["EXTRAPOLATION_COEFFICIENTS_SUM"] = coefficients.str();
        pmeDefines["MAX_EXTRAPOLATION_ORDER"] = cu.intToString(maxExtrapolationOrder);
        CUmodule module = cu.createModule(CudaKernelSources::vectorOps+CudaAmoebaKernelSources::multipolePme, pmeDefines);
        pmeTransformMultipolesKernel = cu.getKernel(module, "transformMultipolesToFractionalCoordinates");
        pmeTransformPotentialKernel = cu.getKernel(module, "transformPotentialToCartesianCoordinates");
        pmeSpreadFixedMultipolesKernel = cu.getKernel(module, "gridSpreadFixedMultipoles");
        pmeSpreadInducedDipolesKernel = cu.getKernel(module, "gridSpreadInducedDipoles");
        pmeFinishSpreadChargeKernel = cu.getKernel(module, "finishSpreadCharge");
        pmeConvolutionKernel = cu.getKernel(module, "reciprocalConvolution");
        pmeFixedPotentialKernel = cu.getKernel(module, "computeFixedPotentialFromGrid");
        pmeInducedPotentialKernel = cu.getKernel(module, "computeInducedPotentialFromGrid");
        pmeFixedForceKernel = cu.getKernel(module, "computeFixedMultipoleForceAndEnergy");
        pmeInducedForceKernel = cu.getKernel(module, "computeInducedDipoleForceAndEnergy");
        pmeRecordInducedFieldDipolesKernel = cu.getKernel(module, "recordInducedFieldDipoles");
        pmeSelfEnergyKernel = cu.getKernel(module, "calculateSelfEnergyAndTorque");

        // Create the dispersion PME kernels.

        pmeDefines["EWALD_ALPHA"] = cu.doubleToString(dpmeAlpha);
        pmeDefines["EPSILON_FACTOR"] = "1";
        pmeDefines["GRID_SIZE_X"] = cu.intToString(dispersionGridSizeX);
        pmeDefines["GRID_SIZE_Y"] = cu.intToString(dispersionGridSizeY);
        pmeDefines["GRID_SIZE_Z"] = cu.intToString(dispersionGridSizeZ);
        pmeDefines["RECIP_EXP_FACTOR"] = cu.doubleToString(M_PI*M_PI/(dpmeAlpha*dpmeAlpha));
        pmeDefines["CHARGE"] = "charges[atom]";
        pmeDefines["USE_LJPME"] = "1";
        module = cu.createModule(CudaKernelSources::vectorOps+CudaKernelSources::pme, pmeDefines);
        dpmeFinishSpreadChargeKernel = cu.getKernel(module, "finishSpreadCharge");
        dpmeGridIndexKernel = cu.getKernel(module, "findAtomGridIndex");
        dpmeSpreadChargeKernel = cu.getKernel(module, "gridSpreadCharge");
        dpmeConvolutionKernel = cu.getKernel(module, "reciprocalConvolution");
        dpmeEvalEnergyKernel = cu.getKernel(module, "gridEvaluateEnergy");
        dpmeInterpolateForceKernel = cu.getKernel(module, "gridInterpolateForce");

        // Create required data structures.

        int roundedZSize = PmeOrder*(int) ceil(gridSizeZ/(double) PmeOrder);
        int gridElements = gridSizeX*gridSizeY*roundedZSize;
        roundedZSize = PmeOrder*(int) ceil(dispersionGridSizeZ/(double) PmeOrder);
        gridElements = max(gridElements, dispersionGridSizeX*dispersionGridSizeY*roundedZSize);
        pmeGrid1.initialize(cu, gridElements, elementSize, "pmeGrid1");
        pmeGrid2.initialize(cu, gridElements, 2*elementSize, "pmeGrid2");
        cu.addAutoclearBuffer(pmeGrid1);
        pmeBsplineModuliX.initialize(cu, gridSizeX, elementSize, "pmeBsplineModuliX");
        pmeBsplineModuliY.initialize(cu, gridSizeY, elementSize, "pmeBsplineModuliY");
        pmeBsplineModuliZ.initialize(cu, gridSizeZ, elementSize, "pmeBsplineModuliZ");
        dpmeBsplineModuliX.initialize(cu, dispersionGridSizeX, elementSize, "dpmeBsplineModuliX");
        dpmeBsplineModuliY.initialize(cu, dispersionGridSizeY, elementSize, "dpmeBsplineModuliY");
        dpmeBsplineModuliZ.initialize(cu, dispersionGridSizeZ, elementSize, "dpmeBsplineModuliZ");
        pmePhi.initialize(cu, 20*numParticles, elementSize, "pmePhi");
        pmePhidp.initialize(cu, 20*numParticles, elementSize, "pmePhidp");
        pmeCphi.initialize(cu, 10*numParticles, elementSize, "pmeCphi");
        pmeAtomGridIndex.initialize<int2>(cu, numParticles, "pmeAtomGridIndex");
        sort = new CudaSort(cu, new SortTrait(), cu.getNumAtoms());
        cufftResult result = cufftPlan3d(&fftForward, gridSizeX, gridSizeY, gridSizeZ, cu.getUseDoublePrecision() ? CUFFT_D2Z : CUFFT_R2C);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: "+cu.intToString(result));
        result = cufftPlan3d(&fftBackward, gridSizeX, gridSizeY, gridSizeZ, cu.getUseDoublePrecision() ? CUFFT_Z2D : CUFFT_C2R);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: "+cu.intToString(result));
        result = cufftPlan3d(&dfftForward, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, cu.getUseDoublePrecision() ? CUFFT_D2Z : CUFFT_R2C);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: "+cu.intToString(result));
        result = cufftPlan3d(&dfftBackward, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, cu.getUseDoublePrecision() ? CUFFT_Z2D : CUFFT_C2R);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: "+cu.intToString(result));
        hasInitializedFFT = true;

        // Initialize the B-spline moduli.

        double data[PmeOrder];
        double x = 0.0;
        data[0] = 1.0 - x;
        data[1] = x;
        for (int i = 2; i < PmeOrder; i++) {
            double denom = 1.0/i;
            data[i] = x*data[i-1]*denom;
            for (int j = 1; j < i; j++)
                data[i-j] = ((x+j)*data[i-j-1] + ((i-j+1)-x)*data[i-j])*denom;
            data[0] = (1.0-x)*data[0]*denom;
        }
        int maxSize = max(max(gridSizeX, gridSizeY), gridSizeZ);
        vector<double> bsplines_data(maxSize+1, 0.0);
        for (int i = 2; i <= PmeOrder+1; i++)
            bsplines_data[i] = data[i-2];
        for (int dim = 0; dim < 3; dim++) {
            int ndata = (dim == 0 ? gridSizeX : dim == 1 ? gridSizeY : gridSizeZ);
            vector<double> moduli(ndata);

            // get the modulus of the discrete Fourier transform

            double factor = 2.0*M_PI/ndata;
            for (int i = 0; i < ndata; i++) {
                double sc = 0.0;
                double ss = 0.0;
                for (int j = 1; j <= ndata; j++) {
                    double arg = factor*i*(j-1);
                    sc += bsplines_data[j]*cos(arg);
                    ss += bsplines_data[j]*sin(arg);
                }
                moduli[i] = sc*sc+ss*ss;
            }

            // Fix for exponential Euler spline interpolation failure.

            double eps = 1.0e-7;
            if (moduli[0] < eps)
                moduli[0] = 0.9*moduli[1];
            for (int i = 1; i < ndata-1; i++)
                if (moduli[i] < eps)
                    moduli[i] = 0.9*(moduli[i-1]+moduli[i+1]);
            if (moduli[ndata-1] < eps)
                moduli[ndata-1] = 0.9*moduli[ndata-2];

            // Compute and apply the optimal zeta coefficient.

            int jcut = 50;
            for (int i = 1; i <= ndata; i++) {
                int k = i - 1;
                if (i > ndata/2)
                    k = k - ndata;
                double zeta;
                if (k == 0)
                    zeta = 1.0;
                else {
                    double sum1 = 1.0;
                    double sum2 = 1.0;
                    factor = M_PI*k/ndata;
                    for (int j = 1; j <= jcut; j++) {
                        double arg = factor/(factor+M_PI*j);
                        sum1 += pow(arg, PmeOrder);
                        sum2 += pow(arg, 2*PmeOrder);
                    }
                    for (int j = 1; j <= jcut; j++) {
                        double arg = factor/(factor-M_PI*j);
                        sum1 += pow(arg, PmeOrder);
                        sum2 += pow(arg, 2*PmeOrder);
                    }
                    zeta = sum2/sum1;
                }
                moduli[i-1] = moduli[i-1]*zeta*zeta;
            }
            if (cu.getUseDoublePrecision()) {
                if (dim == 0)
                    pmeBsplineModuliX.upload(moduli);
                else if (dim == 1)
                    pmeBsplineModuliY.upload(moduli);
                else
                    pmeBsplineModuliZ.upload(moduli);
            }
            else {
                vector<float> modulif(ndata);
                for (int i = 0; i < ndata; i++)
                    modulif[i] = (float) moduli[i];
                if (dim == 0)
                    pmeBsplineModuliX.upload(modulif);
                else if (dim == 1)
                    pmeBsplineModuliY.upload(modulif);
                else
                    pmeBsplineModuliZ.upload(modulif);
            }
        }

        // Initialize the b-spline moduli for dispersion PME.

        maxSize = max(max(dispersionGridSizeX, dispersionGridSizeY), dispersionGridSizeZ);
        vector<double> ddata(PmeOrder);
        bsplines_data.resize(maxSize);
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

        for(int dim = 0; dim < 3; dim++) {
            int ndata = (dim == 0 ? dispersionGridSizeX : dim == 1 ? dispersionGridSizeY : dispersionGridSizeZ);
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
                dpmeBsplineModuliX.upload(moduli, true);
            else if (dim == 1)
                dpmeBsplineModuliY.upload(moduli, true);
            else
                dpmeBsplineModuliZ.upload(moduli, true);
        }
    }

    // Add the interaction to the default nonbonded kernel.
    
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    nb.setKernelSource(CudaAmoebaKernelSources::hippoInteractionHeader+CudaAmoebaKernelSources::hippoNonbonded);
    nb.addArgument(CudaNonbondedUtilities::ParameterInfo("torqueBuffers", "unsigned long long", 1, torque.getElementSize(), torque.getDevicePointer(), false));
    nb.addArgument(CudaNonbondedUtilities::ParameterInfo("extrapolatedDipole", "real3", 1, extrapolatedDipole.getElementSize(), extrapolatedDipole.getDevicePointer()));
    nb.addParameter(CudaNonbondedUtilities::ParameterInfo("coreCharge", "real", 1, coreCharge.getElementSize(), coreCharge.getDevicePointer()));
    nb.addParameter(CudaNonbondedUtilities::ParameterInfo("valenceCharge", "real", 1, valenceCharge.getElementSize(), valenceCharge.getDevicePointer()));
    nb.addParameter(CudaNonbondedUtilities::ParameterInfo("alpha", "real", 1, alpha.getElementSize(), alpha.getDevicePointer()));
    nb.addParameter(CudaNonbondedUtilities::ParameterInfo("epsilon", "real", 1, epsilon.getElementSize(), epsilon.getDevicePointer()));
    nb.addParameter(CudaNonbondedUtilities::ParameterInfo("damping", "real", 1, damping.getElementSize(), damping.getDevicePointer()));
    nb.addParameter(CudaNonbondedUtilities::ParameterInfo("c6", "real", 1, c6.getElementSize(), c6.getDevicePointer()));
    nb.addParameter(CudaNonbondedUtilities::ParameterInfo("pauliK", "real", 1, pauliK.getElementSize(), pauliK.getDevicePointer()));
    nb.addParameter(CudaNonbondedUtilities::ParameterInfo("pauliQ", "real", 1, pauliQ.getElementSize(), pauliQ.getDevicePointer()));
    nb.addParameter(CudaNonbondedUtilities::ParameterInfo("pauliAlpha", "real", 1, pauliAlpha.getElementSize(), pauliAlpha.getDevicePointer()));
    nb.addParameter(CudaNonbondedUtilities::ParameterInfo("dipole", "real", 3, labDipoles.getElementSize(), labDipoles.getDevicePointer()));
    nb.addParameter(CudaNonbondedUtilities::ParameterInfo("inducedDipole", "real", 3, inducedDipole.getElementSize(), inducedDipole.getDevicePointer()));
    nb.addParameter(CudaNonbondedUtilities::ParameterInfo("qXX", "real", 1, labQuadrupoles[0].getElementSize(), labQuadrupoles[0].getDevicePointer()));
    nb.addParameter(CudaNonbondedUtilities::ParameterInfo("qXY", "real", 1, labQuadrupoles[1].getElementSize(), labQuadrupoles[1].getDevicePointer()));
    nb.addParameter(CudaNonbondedUtilities::ParameterInfo("qXZ", "real", 1, labQuadrupoles[2].getElementSize(), labQuadrupoles[2].getDevicePointer()));
    nb.addParameter(CudaNonbondedUtilities::ParameterInfo("qYY", "real", 1, labQuadrupoles[3].getElementSize(), labQuadrupoles[3].getDevicePointer()));
    nb.addParameter(CudaNonbondedUtilities::ParameterInfo("qYZ", "real", 1, labQuadrupoles[4].getElementSize(), labQuadrupoles[4].getDevicePointer()));
    map<string, string> replacements;
    replacements["ENERGY_SCALE_FACTOR"] = cu.doubleToString(ONE_4PI_EPS0);
    replacements["SWITCH_CUTOFF"] = cu.doubleToString(force.getSwitchingDistance());
    replacements["SWITCH_C3"] = cu.doubleToString(10/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 3.0));
    replacements["SWITCH_C4"] = cu.doubleToString(15/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 4.0));
    replacements["SWITCH_C5"] = cu.doubleToString(6/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 5.0));
    replacements["MAX_EXTRAPOLATION_ORDER"] = cu.intToString(maxExtrapolationOrder);
    replacements["EXTRAPOLATION_COEFFICIENTS_SUM"] = coefficients.str();
    replacements["USE_EWALD"] = (usePME ? "1" : "0");
    replacements["PME_ALPHA"] = (usePME ? cu.doubleToString(pmeAlpha) : "0");
    replacements["DPME_ALPHA"] = (usePME ? cu.doubleToString(dpmeAlpha) : "0");
    replacements["SQRT_PI"] = cu.doubleToString(sqrt(M_PI));
    string interactionSource = cu.replaceStrings(CudaAmoebaKernelSources::hippoInteraction, replacements);
    nb.addInteraction(usePME, usePME, true, force.getCutoffDistance(), exclusions, interactionSource, force.getForceGroup());
    nb.setUsePadding(false);
    
    // Create the kernel for computing exceptions.
    
    if (exceptionAtoms.isInitialized()) {
        replacements["COMPUTE_INTERACTION"] = interactionSource;
        string exceptionsSrc = CudaKernelSources::vectorOps+CudaAmoebaKernelSources::hippoInteractionHeader+CudaAmoebaKernelSources::hippoNonbondedExceptions;
        exceptionsSrc = cu.replaceStrings(exceptionsSrc, replacements);
        defines["NUM_EXCEPTIONS"] = cu.intToString(exceptionAtoms.getSize());
        module = cu.createModule(exceptionsSrc, defines);
        computeExceptionsKernel = cu.getKernel(module, "computeNonbondedExceptions");
    }
    cu.addForce(new ForceInfo(force));
    cu.addPostComputation(new TorquePostComputation(*this));
}

void CudaCalcHippoNonbondedForceKernel::createFieldKernel(const string& interactionSrc, vector<CudaArray*> params,
            CudaArray& fieldBuffer, CUfunction& kernel, vector<void*>& args, CUfunction& exceptionKernel, vector<void*>& exceptionArgs,
            CudaArray& exceptionScale) {
    // Create the kernel source.

    map<string, string> replacements;
    replacements["COMPUTE_FIELD"] = interactionSrc;
    stringstream extraArgs, atomParams, loadLocal1, loadLocal2, load1, load2, load3;
    for (auto param : params) {
        string name = param->getName();
        string type = (param->getElementSize() == 4 || param->getElementSize() == 8 ? "real" : "real3");
        extraArgs << ", const " << type << "* __restrict__ " << name;
        atomParams << type << " " << name << ";\n";
        loadLocal1 << "localData[localAtomIndex]." << name << " = " << name << "1;\n";
        loadLocal2 << "localData[localAtomIndex]." << name << " = " << name << "[j];\n";
        load1 << type << " " << name << "1 = " << name << "[atom1];\n";
        load2 << type << " " << name << "2 = localData[atom2]." << name << ";\n";
        load3 << type << " " << name << "2 = " << name << "[atom2];\n";
    }
    replacements["PARAMETER_ARGUMENTS"] = extraArgs.str();
    replacements["ATOM_PARAMETER_DATA"] = atomParams.str();
    replacements["LOAD_LOCAL_PARAMETERS_FROM_1"] = loadLocal1.str();
    replacements["LOAD_LOCAL_PARAMETERS_FROM_GLOBAL"] = loadLocal2.str();
    replacements["LOAD_ATOM1_PARAMETERS"] = load1.str();
    replacements["LOAD_ATOM2_PARAMETERS"] = load2.str();
    replacements["LOAD_ATOM2_PARAMETERS_FROM_GLOBAL"] = load3.str();
    string src = cu.replaceStrings(CudaAmoebaKernelSources::hippoComputeField, replacements);

    // Set defines and create the kernel.

    map<string, string> defines;
    if (usePME) {
        defines["USE_CUTOFF"] = "1";
        defines["USE_PERIODIC"] = "1";
        defines["USE_EWALD"] = "1";
        defines["PME_ALPHA"] = cu.doubleToString(pmeAlpha);
        defines["SQRT_PI"] = cu.doubleToString(sqrt(M_PI));
    }
    defines["WARPS_PER_GROUP"] = cu.intToString(cu.getNonbondedUtilities().getForceThreadBlockSize()/CudaContext::TileSize);
    defines["THREAD_BLOCK_SIZE"] = cu.intToString(cu.getNonbondedUtilities().getForceThreadBlockSize());
    defines["CUTOFF"] = cu.doubleToString(cutoff);
    defines["CUTOFF_SQUARED"] = cu.doubleToString(cutoff*cutoff);
    defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    defines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
    defines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
    defines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(cu.getNonbondedUtilities().getExclusionTiles().getSize());
    defines["NUM_EXCEPTIONS"] = cu.intToString(exceptionAtoms.isInitialized() ? exceptionAtoms.getSize() : 0);
    CUmodule module = cu.createModule(CudaKernelSources::vectorOps+src, defines);
    kernel = cu.getKernel(module, "computeField");

    // Build the list of arguments.

    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    args.push_back(&cu.getPosq().getDevicePointer());
    args.push_back(&cu.getNonbondedUtilities().getExclusions().getDevicePointer());
    args.push_back(&cu.getNonbondedUtilities().getExclusionTiles().getDevicePointer());
    args.push_back(&fieldBuffer.getDevicePointer());
    if (nb.getUseCutoff()) {
        args.push_back(&nb.getInteractingTiles().getDevicePointer());
        args.push_back(&nb.getInteractionCount().getDevicePointer());
        args.push_back(cu.getPeriodicBoxSizePointer());
        args.push_back(cu.getInvPeriodicBoxSizePointer());
        args.push_back(cu.getPeriodicBoxVecXPointer());
        args.push_back(cu.getPeriodicBoxVecYPointer());
        args.push_back(cu.getPeriodicBoxVecZPointer());
        args.push_back(&maxTiles);
        args.push_back(&nb.getBlockCenters().getDevicePointer());
        args.push_back(&nb.getBlockBoundingBoxes().getDevicePointer());
        args.push_back(&nb.getInteractingAtoms().getDevicePointer());
    }
    else
        args.push_back(&maxTiles);
    for (auto param : params)
        args.push_back(&param->getDevicePointer());
    
    // If there are any exceptions, build the kernel and arguments to compute them.
    
    if (exceptionAtoms.isInitialized()) {
        exceptionKernel = cu.getKernel(module, "computeFieldExceptions");
        exceptionArgs.push_back(&cu.getPosq().getDevicePointer());
        exceptionArgs.push_back(&fieldBuffer.getDevicePointer());
        exceptionArgs.push_back(&exceptionAtoms.getDevicePointer());
        exceptionArgs.push_back(&exceptionScale.getDevicePointer());
        if (nb.getUseCutoff()) {
            exceptionArgs.push_back(cu.getPeriodicBoxSizePointer());
            exceptionArgs.push_back(cu.getInvPeriodicBoxSizePointer());
            exceptionArgs.push_back(cu.getPeriodicBoxVecXPointer());
            exceptionArgs.push_back(cu.getPeriodicBoxVecYPointer());
            exceptionArgs.push_back(cu.getPeriodicBoxVecZPointer());
        }
        for (auto param : params)
            exceptionArgs.push_back(&param->getDevicePointer());
    }
}

double CudaCalcHippoNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        
        // These kernels can't be compiled in initialize(), because the nonbonded utilities object
        // has not yet been initialized then.

        maxTiles = (nb.getUseCutoff() ? nb.getInteractingTiles().getSize() : cu.getNumAtomBlocks()*(cu.getNumAtomBlocks()+1)/2);
        createFieldKernel(CudaAmoebaKernelSources::hippoFixedField, {&coreCharge, &valenceCharge, &alpha, &labDipoles, &labQuadrupoles[0],
                &labQuadrupoles[1], &labQuadrupoles[2], &labQuadrupoles[3], &labQuadrupoles[4]}, field, fixedFieldKernel, fixedFieldArgs,
                fixedFieldExceptionKernel, fixedFieldExceptionArgs, exceptionScales[1]);
        createFieldKernel(CudaAmoebaKernelSources::hippoMutualField, {&alpha, &inducedDipole}, inducedField, mutualFieldKernel, mutualFieldArgs,
                mutualFieldExceptionKernel, mutualFieldExceptionArgs, exceptionScales[2]);
        if (exceptionAtoms.isInitialized()) {
            computeExceptionsArgs.push_back(&cu.getForce().getDevicePointer());
            computeExceptionsArgs.push_back(&cu.getEnergyBuffer().getDevicePointer());
            computeExceptionsArgs.push_back(&torque.getDevicePointer());
            computeExceptionsArgs.push_back(&cu.getPosq().getDevicePointer());
            computeExceptionsArgs.push_back(&extrapolatedDipole.getDevicePointer());
            computeExceptionsArgs.push_back(&exceptionAtoms.getDevicePointer());
            computeExceptionsArgs.push_back(&exceptionScales[0].getDevicePointer());
            computeExceptionsArgs.push_back(&exceptionScales[1].getDevicePointer());
            computeExceptionsArgs.push_back(&exceptionScales[2].getDevicePointer());
            computeExceptionsArgs.push_back(&exceptionScales[3].getDevicePointer());
            computeExceptionsArgs.push_back(&exceptionScales[4].getDevicePointer());
            computeExceptionsArgs.push_back(&exceptionScales[5].getDevicePointer());
            computeExceptionsArgs.push_back(&coreCharge.getDevicePointer());
            computeExceptionsArgs.push_back(&valenceCharge.getDevicePointer());
            computeExceptionsArgs.push_back(&alpha.getDevicePointer());
            computeExceptionsArgs.push_back(&epsilon.getDevicePointer());
            computeExceptionsArgs.push_back(&damping.getDevicePointer());
            computeExceptionsArgs.push_back(&c6.getDevicePointer());
            computeExceptionsArgs.push_back(&pauliK.getDevicePointer());
            computeExceptionsArgs.push_back(&pauliQ.getDevicePointer());
            computeExceptionsArgs.push_back(&pauliAlpha.getDevicePointer());
            computeExceptionsArgs.push_back(&labDipoles.getDevicePointer());
            computeExceptionsArgs.push_back(&inducedDipole.getDevicePointer());
            computeExceptionsArgs.push_back(&labQuadrupoles[0].getDevicePointer());
            computeExceptionsArgs.push_back(&labQuadrupoles[1].getDevicePointer());
            computeExceptionsArgs.push_back(&labQuadrupoles[2].getDevicePointer());
            computeExceptionsArgs.push_back(&labQuadrupoles[3].getDevicePointer());
            computeExceptionsArgs.push_back(&labQuadrupoles[4].getDevicePointer());
            computeExceptionsArgs.push_back(&extrapolatedDipole.getDevicePointer());
            if (nb.getUseCutoff()) {
                computeExceptionsArgs.push_back(cu.getPeriodicBoxSizePointer());
                computeExceptionsArgs.push_back(cu.getInvPeriodicBoxSizePointer());
                computeExceptionsArgs.push_back(cu.getPeriodicBoxVecXPointer());
                computeExceptionsArgs.push_back(cu.getPeriodicBoxVecYPointer());
                computeExceptionsArgs.push_back(cu.getPeriodicBoxVecZPointer());
            }
        }
    }
    
    // Make sure the arrays for the neighbor list haven't been recreated.

    if (nb.getUseCutoff()) {
        if (maxTiles < nb.getInteractingTiles().getSize()) {
            maxTiles = nb.getInteractingTiles().getSize();
            fixedFieldArgs[4] = &nb.getInteractingTiles().getDevicePointer();
            fixedFieldArgs[14] = &nb.getInteractingAtoms().getDevicePointer();
            mutualFieldArgs[4] = &nb.getInteractingTiles().getDevicePointer();
            mutualFieldArgs[14] = &nb.getInteractingAtoms().getDevicePointer();
        }
    }

    // Compute the lab frame moments.

    void* computeMomentsArgs[] = {&cu.getPosq().getDevicePointer(), &multipoleParticles.getDevicePointer(),
        &localDipoles.getDevicePointer(), &localQuadrupoles.getDevicePointer(),
        &labDipoles.getDevicePointer(), &labQuadrupoles[0].getDevicePointer(),
        &labQuadrupoles[1].getDevicePointer(), &labQuadrupoles[2].getDevicePointer(),
        &labQuadrupoles[3].getDevicePointer(), &labQuadrupoles[4].getDevicePointer()};
    cu.executeKernel(computeMomentsKernel, computeMomentsArgs, cu.getNumAtoms());

    void* recipBoxVectorPointer[3];
    if (usePME) {
        // Compute reciprocal box vectors.
        
        Vec3 boxVectors[3];
        cu.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double determinant = boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2];
        double scale = 1.0/determinant;
        double3 recipBoxVectors[3];
        recipBoxVectors[0] = make_double3(boxVectors[1][1]*boxVectors[2][2]*scale, 0, 0);
        recipBoxVectors[1] = make_double3(-boxVectors[1][0]*boxVectors[2][2]*scale, boxVectors[0][0]*boxVectors[2][2]*scale, 0);
        recipBoxVectors[2] = make_double3((boxVectors[1][0]*boxVectors[2][1]-boxVectors[1][1]*boxVectors[2][0])*scale, -boxVectors[0][0]*boxVectors[2][1]*scale, boxVectors[0][0]*boxVectors[1][1]*scale);
        float3 recipBoxVectorsFloat[3];
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

        // Reciprocal space calculation for electrostatics.
        
        void* pmeTransformMultipolesArgs[] = {&labDipoles.getDevicePointer(), &labQuadrupoles[0].getDevicePointer(),
            &labQuadrupoles[1].getDevicePointer(), &labQuadrupoles[2].getDevicePointer(),
            &labQuadrupoles[3].getDevicePointer(), &labQuadrupoles[4].getDevicePointer(),
            &fracDipoles.getDevicePointer(), &fracQuadrupoles.getDevicePointer(),
            recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeTransformMultipolesKernel, pmeTransformMultipolesArgs, cu.getNumAtoms());
        void* pmeSpreadFixedMultipolesArgs[] = {&cu.getPosq().getDevicePointer(), &fracDipoles.getDevicePointer(), &fracQuadrupoles.getDevicePointer(),
            &pmeGrid1.getDevicePointer(), &coreCharge.getDevicePointer(), &valenceCharge.getDevicePointer(),
            cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeSpreadFixedMultipolesKernel, pmeSpreadFixedMultipolesArgs, cu.getNumAtoms());
        if (cu.getUseDoublePrecision()) {
            void* finishSpreadArgs[] = {&pmeGrid1.getDevicePointer()};
            cu.executeKernel(pmeFinishSpreadChargeKernel, finishSpreadArgs, pmeGrid1.getSize());
            cufftExecD2Z(fftForward, (double*) pmeGrid1.getDevicePointer(), (double2*) pmeGrid2.getDevicePointer());
        }
        else
            cufftExecR2C(fftForward, (float*) pmeGrid1.getDevicePointer(), (float2*) pmeGrid2.getDevicePointer());
        void* pmeConvolutionArgs[] = {&pmeGrid2.getDevicePointer(), &pmeBsplineModuliX.getDevicePointer(), &pmeBsplineModuliY.getDevicePointer(),
            &pmeBsplineModuliZ.getDevicePointer(), cu.getPeriodicBoxSizePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeConvolutionKernel, pmeConvolutionArgs, gridSizeX*gridSizeY*gridSizeZ, 256);
        if (cu.getUseDoublePrecision())
            cufftExecZ2D(fftBackward, (double2*) pmeGrid2.getDevicePointer(), (double*) pmeGrid1.getDevicePointer());
        else
            cufftExecC2R(fftBackward, (float2*) pmeGrid2.getDevicePointer(), (float*) pmeGrid1.getDevicePointer());
        void* pmeFixedPotentialArgs[] = {&pmeGrid1.getDevicePointer(), &pmePhi.getDevicePointer(), &field.getDevicePointer(),
            &cu.getPosq().getDevicePointer(), &labDipoles.getDevicePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(),
            cu.getPeriodicBoxVecZPointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeFixedPotentialKernel, pmeFixedPotentialArgs, cu.getNumAtoms());
        void* pmeTransformFixedPotentialArgs[] = {&pmePhi.getDevicePointer(), &pmeCphi.getDevicePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeTransformPotentialKernel, pmeTransformFixedPotentialArgs, cu.getNumAtoms());
        void* pmeFixedForceArgs[] = {&cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &torque.getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(), &labDipoles.getDevicePointer(), &coreCharge.getDevicePointer(),
            &valenceCharge.getDevicePointer(), &labQuadrupoles[0].getDevicePointer(),
            &labQuadrupoles[1].getDevicePointer(), &labQuadrupoles[2].getDevicePointer(),
            &labQuadrupoles[3].getDevicePointer(), &labQuadrupoles[4].getDevicePointer(),
            &fracDipoles.getDevicePointer(), &fracQuadrupoles.getDevicePointer(), &pmePhi.getDevicePointer(), &pmeCphi.getDevicePointer(),
            recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeFixedForceKernel, pmeFixedForceArgs, cu.getNumAtoms());

        // Reciprocal space calculation for dispersion.

        void* gridIndexArgs[] = {&cu.getPosq().getDevicePointer(), &pmeAtomGridIndex.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(dpmeGridIndexKernel, gridIndexArgs, cu.getNumAtoms());
        sort->sort(pmeAtomGridIndex);
        cu.clearBuffer(pmeGrid2);
        void* spreadArgs[] = {&cu.getPosq().getDevicePointer(), &pmeGrid2.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &pmeAtomGridIndex.getDevicePointer(),
                &c6.getDevicePointer()};
        cu.executeKernel(dpmeSpreadChargeKernel, spreadArgs, cu.getNumAtoms(), 128);
        void* finishSpreadArgs[] = {&pmeGrid2.getDevicePointer(), &pmeGrid1.getDevicePointer()};
        cu.executeKernel(dpmeFinishSpreadChargeKernel, finishSpreadArgs, dispersionGridSizeX*dispersionGridSizeY*dispersionGridSizeZ, 256);
        if (cu.getUseDoublePrecision())
            cufftExecD2Z(dfftForward, (double*) pmeGrid1.getDevicePointer(), (double2*) pmeGrid2.getDevicePointer());
        else
            cufftExecR2C(dfftForward, (float*) pmeGrid1.getDevicePointer(), (float2*) pmeGrid2.getDevicePointer());
        if (includeEnergy) {
            void* computeEnergyArgs[] = {&pmeGrid2.getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(),
                    &dpmeBsplineModuliX.getDevicePointer(), &dpmeBsplineModuliY.getDevicePointer(), &dpmeBsplineModuliZ.getDevicePointer(),
                    cu.getPeriodicBoxSizePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
            cu.executeKernel(dpmeEvalEnergyKernel, computeEnergyArgs, dispersionGridSizeX*dispersionGridSizeY*dispersionGridSizeZ);
        }
        void* convolutionArgs[] = {&pmeGrid2.getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(),
                &dpmeBsplineModuliX.getDevicePointer(), &dpmeBsplineModuliY.getDevicePointer(), &dpmeBsplineModuliZ.getDevicePointer(),
                cu.getPeriodicBoxSizePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(dpmeConvolutionKernel, convolutionArgs, dispersionGridSizeX*dispersionGridSizeY*dispersionGridSizeZ, 256);
        if (cu.getUseDoublePrecision())
            cufftExecZ2D(dfftBackward, (double2*) pmeGrid2.getDevicePointer(), (double*) pmeGrid1.getDevicePointer());
        else
            cufftExecC2R(dfftBackward, (float2*) pmeGrid2.getDevicePointer(), (float*)  pmeGrid1.getDevicePointer());
        void* interpolateArgs[] = {&cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &pmeGrid1.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &pmeAtomGridIndex.getDevicePointer(),
                &c6.getDevicePointer()};
        cu.executeKernel(dpmeInterpolateForceKernel, interpolateArgs, cu.getNumAtoms(), 128);
    }

    // Compute the field from fixed multipoles.

    cu.executeKernel(fixedFieldKernel, &fixedFieldArgs[0], nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize(), nb.getForceThreadBlockSize());
    if (fixedFieldExceptionArgs.size() > 0)
        cu.executeKernel(fixedFieldExceptionKernel, &fixedFieldExceptionArgs[0], exceptionAtoms.getSize());

    // Iterate the induced dipoles.

    computeExtrapolatedDipoles(recipBoxVectorPointer);

    // Add the polarization energy.

    if (includeEnergy) {
        void* polarizationEnergyArgs[] = {&cu.getEnergyBuffer().getDevicePointer(), &inducedDipole.getDevicePointer(),
            &extrapolatedDipole.getDevicePointer(), &polarizability.getDevicePointer()};
        cu.executeKernel(polarizationEnergyKernel, polarizationEnergyArgs, cu.getNumAtoms());
    }

    // Compute the forces due to the reciprocal space PME calculation for induced dipoles.

    if (usePME) {
        void* pmeTransformInducedPotentialArgs[] = {&pmePhidp.getDevicePointer(), &pmeCphi.getDevicePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeTransformPotentialKernel, pmeTransformInducedPotentialArgs, cu.getNumAtoms());
        void* pmeInducedForceArgs[] = {&cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &torque.getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(), &labDipoles.getDevicePointer(), &coreCharge.getDevicePointer(),
            &valenceCharge.getDevicePointer(), &extrapolatedDipole.getDevicePointer(), &extrapolatedPhi.getDevicePointer(),
            &labQuadrupoles[0].getDevicePointer(), &labQuadrupoles[1].getDevicePointer(), &labQuadrupoles[2].getDevicePointer(),
            &labQuadrupoles[3].getDevicePointer(), &labQuadrupoles[4].getDevicePointer(), &fracDipoles.getDevicePointer(),
            &fracQuadrupoles.getDevicePointer(), &inducedDipole.getDevicePointer(), &pmePhi.getDevicePointer(), &pmePhidp.getDevicePointer(),
            &pmeCphi.getDevicePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeInducedForceKernel, pmeInducedForceArgs, cu.getNumAtoms());
        void* pmeSelfEnergyArgs[] = {&torque.getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(),
            &labDipoles.getDevicePointer(), &coreCharge.getDevicePointer(), &valenceCharge.getDevicePointer(), &c6.getDevicePointer(),
            &inducedDipole.getDevicePointer(), &labQuadrupoles[0].getDevicePointer(), &labQuadrupoles[1].getDevicePointer(),
            &labQuadrupoles[2].getDevicePointer(), &labQuadrupoles[3].getDevicePointer(), &labQuadrupoles[4].getDevicePointer()};
        cu.executeKernel(pmeSelfEnergyKernel, pmeSelfEnergyArgs, cu.getNumAtoms());
    }

    // Compute nonbonded exceptions.

    if (exceptionAtoms.isInitialized())
        cu.executeKernel(computeExceptionsKernel, &computeExceptionsArgs[0], exceptionAtoms.getSize());

    // Record the current atom positions so we can tell later if they have changed.
    
    cu.getPosq().copyTo(lastPositions);
    multipolesAreValid = true;
    return 0.0;
}

void CudaCalcHippoNonbondedForceKernel::computeInducedField(void** recipBoxVectorPointer, int optOrder) {
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    cu.clearBuffer(inducedField);
    cu.executeKernel(mutualFieldKernel, &mutualFieldArgs[0], nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize(), nb.getForceThreadBlockSize());
    if (mutualFieldExceptionArgs.size() > 0)
        cu.executeKernel(mutualFieldExceptionKernel, &mutualFieldExceptionArgs[0], exceptionAtoms.getSize());
    if (usePME) {
        cu.clearBuffer(pmeGrid1);
        void* pmeSpreadInducedDipolesArgs[] = {&cu.getPosq().getDevicePointer(), &inducedDipole.getDevicePointer(),
            &pmeGrid1.getDevicePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeSpreadInducedDipolesKernel, pmeSpreadInducedDipolesArgs, cu.getNumAtoms());
        if (cu.getUseDoublePrecision()) {
            void* finishSpreadArgs[] = {&pmeGrid1.getDevicePointer()};
            cu.executeKernel(pmeFinishSpreadChargeKernel, finishSpreadArgs, pmeGrid1.getSize());
            cufftExecD2Z(fftForward, (double*) pmeGrid1.getDevicePointer(), (double2*) pmeGrid2.getDevicePointer());
        }
        else
            cufftExecR2C(fftForward, (float*) pmeGrid1.getDevicePointer(), (float2*) pmeGrid2.getDevicePointer());
        void* pmeConvolutionArgs[] = {&pmeGrid2.getDevicePointer(), &pmeBsplineModuliX.getDevicePointer(), &pmeBsplineModuliY.getDevicePointer(),
            &pmeBsplineModuliZ.getDevicePointer(), cu.getPeriodicBoxSizePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeConvolutionKernel, pmeConvolutionArgs, gridSizeX*gridSizeY*gridSizeZ, 256);
        if (cu.getUseDoublePrecision())
            cufftExecZ2D(fftBackward, (double2*) pmeGrid2.getDevicePointer(), (double*) pmeGrid1.getDevicePointer());
        else
            cufftExecC2R(fftBackward, (float2*) pmeGrid2.getDevicePointer(), (float*) pmeGrid1.getDevicePointer());
        void* pmeInducedPotentialArgs[] = {&pmeGrid1.getDevicePointer(), &extrapolatedPhi.getDevicePointer(), &optOrder,
            &pmePhidp.getDevicePointer(), &cu.getPosq().getDevicePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(),
            cu.getPeriodicBoxVecZPointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeInducedPotentialKernel, pmeInducedPotentialArgs, cu.getNumAtoms());
        void* pmeRecordInducedFieldDipolesArgs[] = {&pmePhidp.getDevicePointer(), &inducedField.getDevicePointer(),
            &inducedDipole.getDevicePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeRecordInducedFieldDipolesKernel, pmeRecordInducedFieldDipolesArgs, cu.getNumAtoms());
    }
}

void CudaCalcHippoNonbondedForceKernel::computeExtrapolatedDipoles(void** recipBoxVectorPointer) {
    // Start by storing the direct dipoles as PT0

    void* recordInducedDipolesArgs[] = {&field.getDevicePointer(), &inducedDipole.getDevicePointer(), &polarizability.getDevicePointer()};
    cu.executeKernel(recordInducedDipolesKernel, recordInducedDipolesArgs, cu.getNumAtoms());
    void* initArgs[] = {&inducedDipole.getDevicePointer(), &extrapolatedDipole.getDevicePointer()};
    cu.executeKernel(initExtrapolatedKernel, initArgs, extrapolatedDipole.getSize());

    // Recursively apply alpha.Tau to the µ_(n) components to generate µ_(n+1), and store the result

    for (int order = 1; order < maxExtrapolationOrder; ++order) {
        computeInducedField(recipBoxVectorPointer, order-1);
        void* iterateArgs[] = {&order, &inducedDipole.getDevicePointer(), &extrapolatedDipole.getDevicePointer(), &inducedField.getDevicePointer(), &polarizability.getDevicePointer()};
        cu.executeKernel(iterateExtrapolatedKernel, iterateArgs, extrapolatedDipole.getSize());
    }
    
    // Take a linear combination of the µ_(n) components to form the total dipole

    void* computeArgs[] = {&inducedDipole.getDevicePointer(), &extrapolatedDipole.getDevicePointer()};
    cu.executeKernel(computeExtrapolatedKernel, computeArgs, extrapolatedDipole.getSize());
    computeInducedField(recipBoxVectorPointer, maxExtrapolationOrder-1);
}

void CudaCalcHippoNonbondedForceKernel::addTorquesToForces() {
    void* mapTorqueArgs[] = {&cu.getForce().getDevicePointer(), &torque.getDevicePointer(),
        &cu.getPosq().getDevicePointer(), &multipoleParticles.getDevicePointer()};
    cu.executeKernel(mapTorqueKernel, mapTorqueArgs, cu.getNumAtoms());
}

void CudaCalcHippoNonbondedForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ensureMultipolesValid(context);
    int numParticles = cu.getNumAtoms();
    dipoles.resize(numParticles);
    const vector<int>& order = cu.getAtomIndex();
    if (cu.getUseDoublePrecision()) {
        vector<double3> d;
        inducedDipole.download(d);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(d[i].x, d[i].y, d[i].z);
    }
    else {
        vector<float3> d;
        inducedDipole.download(d);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(d[i].x, d[i].y, d[i].z);
    }
}

void CudaCalcHippoNonbondedForceKernel::ensureMultipolesValid(ContextImpl& context) {
    if (multipolesAreValid) {
        int numParticles = cu.getNumAtoms();
        if (cu.getUseDoublePrecision()) {
            vector<double4> pos1, pos2;
            cu.getPosq().download(pos1);
            lastPositions.download(pos2);
            for (int i = 0; i < numParticles; i++)
                if (pos1[i].x != pos2[i].x || pos1[i].y != pos2[i].y || pos1[i].z != pos2[i].z) {
                    multipolesAreValid = false;
                    break;
                }
        }
        else {
            vector<float4> pos1, pos2;
            cu.getPosq().download(pos1);
            lastPositions.download(pos2);
            for (int i = 0; i < numParticles; i++)
                if (pos1[i].x != pos2[i].x || pos1[i].y != pos2[i].y || pos1[i].z != pos2[i].z) {
                    multipolesAreValid = false;
                    break;
                }
        }
    }
    if (!multipolesAreValid)
        context.calcForcesAndEnergy(false, false, context.getIntegrator().getIntegrationForceGroups());
}

void CudaCalcHippoNonbondedForceKernel::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ensureMultipolesValid(context);
    int numParticles = cu.getNumAtoms();
    dipoles.resize(numParticles);
    const vector<int>& order = cu.getAtomIndex();
    if (cu.getUseDoublePrecision()) {
        vector<double3> labDipoleVec;
        labDipoles.download(labDipoleVec);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(labDipoleVec[i].x, labDipoleVec[i].y, labDipoleVec[i].z);
    }
    else {
        vector<float3> labDipoleVec;
        labDipoles.download(labDipoleVec);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(labDipoleVec[i].x, labDipoleVec[i].y, labDipoleVec[i].z);
    }
}

void CudaCalcHippoNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const HippoNonbondedForce& force) {
    // Make sure the new parameters are acceptable.
    
    cu.setAsCurrent();
    if (force.getNumParticles() != cu.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    
    // Record the per-particle parameters.
    
    vector<double> coreChargeVec, valenceChargeVec, alphaVec, epsilonVec, dampingVec, c6Vec, pauliKVec, pauliQVec, pauliAlphaVec, polarizabilityVec;
    vector<double> localDipolesVec, localQuadrupolesVec;
    vector<int4> multipoleParticlesVec;
    for (int i = 0; i < numParticles; i++) {
        double charge, coreCharge, alpha, epsilon, damping, c6, pauliK, pauliQ, pauliAlpha, polarizability;
        int axisType, atomX, atomY, atomZ;
        vector<double> dipole, quadrupole;
        force.getParticleParameters(i, charge, dipole, quadrupole, coreCharge, alpha, epsilon, damping, c6, pauliK, pauliQ, pauliAlpha,
                                    polarizability, axisType, atomZ, atomX, atomY);
        coreChargeVec.push_back(coreCharge);
        valenceChargeVec.push_back(charge-coreCharge);
        alphaVec.push_back(alpha);
        epsilonVec.push_back(epsilon);
        dampingVec.push_back(damping);
        c6Vec.push_back(c6);
        pauliKVec.push_back(pauliK);
        pauliQVec.push_back(pauliQ);
        pauliAlphaVec.push_back(pauliAlpha);
        polarizabilityVec.push_back(polarizability);
        multipoleParticlesVec.push_back(make_int4(atomX, atomY, atomZ, axisType));
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back(dipole[j]);
        localQuadrupolesVec.push_back(quadrupole[0]);
        localQuadrupolesVec.push_back(quadrupole[1]);
        localQuadrupolesVec.push_back(quadrupole[2]);
        localQuadrupolesVec.push_back(quadrupole[4]);
        localQuadrupolesVec.push_back(quadrupole[5]);
    }
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    for (int i = numParticles; i < paddedNumAtoms; i++) {
        coreChargeVec.push_back(0);
        valenceChargeVec.push_back(0);
        alphaVec.push_back(0);
        epsilonVec.push_back(0);
        dampingVec.push_back(0);
        c6Vec.push_back(0);
        pauliKVec.push_back(0);
        pauliQVec.push_back(0);
        pauliAlphaVec.push_back(0);
        polarizabilityVec.push_back(0);
        multipoleParticlesVec.push_back(make_int4(0, 0, 0, 0));
        for (int j = 0; j < 3; j++)
            localDipolesVec.push_back(0);
        for (int j = 0; j < 5; j++)
            localQuadrupolesVec.push_back(0);
    }
    coreCharge.upload(coreChargeVec, true);
    valenceCharge.upload(valenceChargeVec, true);
    alpha.upload(alphaVec, true);
    epsilon.upload(epsilonVec, true);
    damping.upload(dampingVec, true);
    c6.upload(c6Vec, true);
    pauliK.upload(pauliKVec, true);
    pauliQ.upload(pauliQVec, true);
    pauliAlpha.upload(pauliAlphaVec, true);
    polarizability.upload(polarizabilityVec, true);
    multipoleParticles.upload(multipoleParticlesVec);
    localDipoles.upload(localDipolesVec, true);
    localQuadrupoles.upload(localQuadrupolesVec, true);
    
    // Record the per-exception parameters.

    vector<double> exceptionScaleVec[6];
    vector<int2> exceptionAtomsVec;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale, dispersionScale, repulsionScale, chargeTransferScale;
        force.getExceptionParameters(i, particle1, particle2, multipoleMultipoleScale, dipoleMultipoleScale, dipoleDipoleScale, dispersionScale, repulsionScale, chargeTransferScale);
        if (usePME || multipoleMultipoleScale != 0 || dipoleMultipoleScale != 0 || dipoleDipoleScale != 0 || dispersionScale != 0 || repulsionScale != 0 || chargeTransferScale != 0) {
            exceptionAtomsVec.push_back(make_int2(particle1, particle2));
            exceptionScaleVec[0].push_back(multipoleMultipoleScale);
            exceptionScaleVec[1].push_back(dipoleMultipoleScale);
            exceptionScaleVec[2].push_back(dipoleDipoleScale);
            exceptionScaleVec[3].push_back(dispersionScale);
            exceptionScaleVec[4].push_back(repulsionScale);
            exceptionScaleVec[5].push_back(chargeTransferScale);
        }
    }
    if (exceptionAtomsVec.size() > 0) {
        if (!exceptionAtoms.isInitialized() || exceptionAtoms.getSize() != exceptionAtomsVec.size())
            throw OpenMMException("updateParametersInContext: The number of exceptions has changed");
        exceptionAtoms.upload(exceptionAtomsVec);
        for (int i = 0; i < 6; i++)
            exceptionScales[i].upload(exceptionScaleVec[i], true);
    }
    else if (exceptionAtoms.isInitialized())
        throw OpenMMException("updateParametersInContext: The number of exceptions has changed");
    cu.invalidateMolecules();
    multipolesAreValid = false;
}

void CudaCalcHippoNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = pmeAlpha;
    nx = gridSizeX;
    ny = gridSizeY;
    nz = gridSizeZ;
}

void CudaCalcHippoNonbondedForceKernel::getDPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = dpmeAlpha;
    nx = dispersionGridSizeX;
    ny = dispersionGridSizeY;
    nz = dispersionGridSizeZ;
}
