/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2011-2012 Stanford University and the Authors.      *
 * Portions copyright (C) 2020 Advanced Micro Devices, Inc. All Rights        *
 * Reserved.                                                                  *
 * Authors: Peter Eastman, Nicholas Curtis                                    *
 * Contributors:                                                              *
 * Portions Copyright© 2020 Advanced Micro Devices, Inc. All rights reserved  *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

/**
 * This tests a system with multiple forces, to make sure HipBondedUtilities is
 * processing them correctly.
 */

#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "HipPlatform.h"
#include "ReferencePlatform.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/PeriodicTorsionForce.h"
#include "openmm/RBTorsionForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "SimTKOpenMMRealType.h"
#include "sfmt/SFMT.h"
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

HipPlatform platform;

const double TOL = 1e-4;

void testForces() {
    const int numParticles = 100;
    System system;
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    HarmonicBondForce* bonds = new HarmonicBondForce();
    for (int i = 0; i < numParticles-1; i++)
        bonds->addBond(i, i+1, 1.0, 1.5);
    system.addForce(bonds);
    HarmonicAngleForce* angles = new HarmonicAngleForce();
    for (int i = 0; i < numParticles-2; i++)
        angles->addAngle(i, i+1, i+2, 2.0, 1.5);
    system.addForce(angles);
    PeriodicTorsionForce* periodic = new PeriodicTorsionForce();
    for (int i = 0; i < numParticles-3; i++)
        periodic->addTorsion(i, i+1, i+2, i+3, 2, PI_M/3, 1.1);
    system.addForce(periodic);
    RBTorsionForce* rb = new RBTorsionForce();
    for (int i = 0; i < numParticles-3; i += 3)
        rb->addTorsion(i, i+1, i+2, i+3, 1.0, 1.1, 1.2, 0.3, 0.4, 0.5);
    system.addForce(rb);
    ReferencePlatform ref;
    VerletIntegrator integrator1(0.01);
    Context context1(system, integrator1, ref);
    VerletIntegrator integrator2(0.01);
    Context context2(system, integrator2, platform);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++)
        positions[i] = Vec3(i, genrand_real2(sfmt), genrand_real2(sfmt));
    context1.setPositions(positions);
    context2.setPositions(positions);
    State state1 = context1.getState(State::Forces | State::Energy);
    State state2 = context2.getState(State::Forces | State::Energy);
    const vector<Vec3>& forces1 = state1.getForces();
    const vector<Vec3>& forces2 = state2.getForces();
    for (int i = 0; i < numParticles; i++)
        ASSERT_EQUAL_VEC(forces1[i], forces2[i], TOL);
    ASSERT_EQUAL_TOL(state1.getPotentialEnergy(), state2.getPotentialEnergy(), TOL);
}

int main(int argc, char* argv[]) {
    try {
        if (argc > 1)
            platform.setPropertyDefaultValue("HipPrecision", string(argv[1]));
        testForces();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
