/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2012-2015 Stanford University and the Authors.      *
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

#include "HipTests.h"
#include "TestVirtualSites.h"

/**
 * Make sure that atom reordering respects virtual sites.
 */
void testReordering() {
    const double cutoff = 2.0;
    const double boxSize = 20.0;
    System system;
    NonbondedForce* nonbonded = new NonbondedForce();
    system.addForce(nonbonded);
    nonbonded->setNonbondedMethod(NonbondedForce::CutoffNonPeriodic);
    nonbonded->setCutoffDistance(cutoff);
    vector<Vec3> positions;
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);

    // Create linear molecules with TwoParticleAverage virtual sites.

    for (int i = 0; i < 50; i++) {
        int start = system.getNumParticles();
        system.addParticle(1.0);
        system.addParticle(1.0);
        system.addParticle(0.0);
        system.setVirtualSite(start+2, new TwoParticleAverageSite(start, start+1, 0.4, 0.6));
        system.addConstraint(start, start+1, 2.0);
        for (int i = 0; i < 3; i++) {
            nonbonded->addParticle(0, 0.2, 1);
            for (int j = 0; j < i; j++)
                nonbonded->addException(start+i, start+j, 0, 1, 0);
        }
        Vec3 pos(boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt));
        positions.push_back(pos);
        positions.push_back(pos+Vec3(2, 0, 0));
        positions.push_back(Vec3());
    }

    // Create planar molecules with ThreeParticleAverage virtual sites.

    for (int i = 0; i < 50; i++) {
        int start = system.getNumParticles();
        system.addParticle(1.0);
        system.addParticle(1.0);
        system.addParticle(1.0);
        system.addParticle(0.0);
        system.setVirtualSite(start+3, new ThreeParticleAverageSite(start, start+1, start+2, 0.3, 0.5, 0.2));
        system.addConstraint(start, start+1, 1.0);
        system.addConstraint(start, start+2, 1.0);
        system.addConstraint(start+1, start+2, sqrt(2.0));
        for (int i = 0; i < 4; i++) {
            nonbonded->addParticle(0, 0.2, 1);
            for (int j = 0; j < i; j++)
                nonbonded->addException(start+i, start+j, 0, 1, 0);
        }
        Vec3 pos(boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt));
        positions.push_back(pos);
        positions.push_back(pos+Vec3(1, 0, 0));
        positions.push_back(pos+Vec3(0, 1, 0));
        positions.push_back(Vec3());
    }

    // Create tetrahedral molecules with OutOfPlane virtual sites.

    for (int i = 0; i < 50; i++) {
        int start = system.getNumParticles();
        system.addParticle(1.0);
        system.addParticle(1.0);
        system.addParticle(1.0);
        system.addParticle(0.0);
        system.setVirtualSite(start+3, new OutOfPlaneSite(start, start+1, start+2, 0.3, 0.5, 0.2));
        system.addConstraint(start, start+1, 1.0);
        system.addConstraint(start, start+2, 1.0);
        system.addConstraint(start+1, start+2, sqrt(2.0));
        for (int i = 0; i < 4; i++) {
            nonbonded->addParticle(0, 0.2, 1);
            for (int j = 0; j < i; j++)
                nonbonded->addException(start+i, start+j, 0, 1, 0);
        }
        Vec3 pos(boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt));
        positions.push_back(pos);
        positions.push_back(pos+Vec3(1, 0, 0));
        positions.push_back(pos+Vec3(0, 1, 0));
        positions.push_back(Vec3());
    }

    // Simulate it and check conservation laws.

    LangevinIntegrator integrator(300.0, 0.1, 0.002);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.applyConstraints(0.0001);
    for (int i = 0; i < 1000; i++) {
        State state = context.getState(State::Positions);
        const vector<Vec3>& pos = state.getPositions();
        for (int j = 0; j < 150; j += 3)
            ASSERT_EQUAL_VEC(pos[j]*0.4+pos[j+1]*0.6, pos[j+2], 1e-5);
        for (int j = 150; j < 350; j += 4)
            ASSERT_EQUAL_VEC(pos[j]*0.3+pos[j+1]*0.5+pos[j+2]*0.2, pos[j+3], 1e-5);
        for (int j = 350; j < 550; j += 4) {
            Vec3 v12 = pos[j+1]-pos[j];
            Vec3 v13 = pos[j+2]-pos[j];
            Vec3 cross = v12.cross(v13);
            ASSERT_EQUAL_VEC(pos[j]+v12*0.3+v13*0.5+cross*0.2, pos[j+3], 1e-5);
        }
        integrator.step(1);
    }
}

void runPlatformTests() {
    testReordering();
}
