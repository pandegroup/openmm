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
#include "TestCheckpoints.h"

void testCheckpoint() {
    const int numParticles = 100;
    const double boxSize = 5.0;
    const double temperature = 200.0;
    System system;
    system.addForce(new AndersenThermostat(0.0, 100.0));
    NonbondedForce* nonbonded = new NonbondedForce();
    system.addForce(nonbonded);
    nonbonded->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        nonbonded->addParticle(i%2 == 0 ? 0.1 : -0.1, 0.2, 0.1);
        bool clash;
        do {
            clash = false;
            positions[i] = Vec3(boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt));
            for (int j = 0; j < i; j++) {
                Vec3 delta = positions[i]-positions[j];
                if (sqrt(delta.dot(delta)) < 0.1)
                    clash = true;
            }
        } while (clash);
    }
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    context.setParameter(AndersenThermostat::Temperature(), temperature);

    // Run for a little while.

    integrator.step(100);

    // Record the current state and make a checkpoint.

    State s1 = context.getState(State::Positions | State::Velocities | State::Parameters);
    stringstream stream1(ios_base::out | ios_base::in | ios_base::binary);
    context.createCheckpoint(stream1);

    // Continue the simulation for a few more steps and record the state again.

    integrator.step(10);
    State s2 = context.getState(State::Positions | State::Velocities | State::Parameters);

    // Restore from the checkpoint and see if everything gets restored correctly.

    context.setPeriodicBoxVectors(Vec3(2*boxSize, 0, 0), Vec3(0, 2*boxSize, 0), Vec3(0, 0, 2*boxSize));
    context.setParameter(AndersenThermostat::Temperature(), temperature+10);
    context.loadCheckpoint(stream1);
    State s3 = context.getState(State::Positions | State::Velocities | State::Parameters);
    compareStates(s1, s3);

    // Now simulate from there and see if the trajectory is identical.

    integrator.step(10);
    State s4 = context.getState(State::Positions | State::Velocities | State::Parameters);
    compareStates(s2, s4);

    // Create a new Context that uses multiple devices.

    string deviceIndex = platform.getPropertyValue(context, HipPlatform::HipDeviceIndex());
    map<string, string> props;
    props[HipPlatform::HipDeviceIndex()] = deviceIndex+","+deviceIndex;
    VerletIntegrator integrator2(0.001);
    Context context2(system, integrator2, platform, props);
    context2.setPositions(positions);
    context2.setPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    context2.setParameter(AndersenThermostat::Temperature(), temperature);

    // Now repeat all of the above tests with it.

    integrator2.step(100);
    State s5 = context2.getState(State::Positions | State::Velocities | State::Parameters);
    stringstream stream2(ios_base::out | ios_base::in | ios_base::binary);
    context2.createCheckpoint(stream2);
    integrator2.step(10);
    State s6 = context2.getState(State::Positions | State::Velocities | State::Parameters);
    context2.setPeriodicBoxVectors(Vec3(2*boxSize, 0, 0), Vec3(0, 2*boxSize, 0), Vec3(0, 0, 2*boxSize));
    context2.setParameter(AndersenThermostat::Temperature(), temperature+10);
    context2.loadCheckpoint(stream2);
    State s7 = context2.getState(State::Positions | State::Velocities | State::Parameters);
    compareStates(s5, s7);
    integrator2.step(10);
    State s8 = context2.getState(State::Positions | State::Velocities | State::Parameters);
    compareStates(s6, s8);

    // See if a checkpoint created from one Context can be loaded into a different one.

    VerletIntegrator integrator3(0.001);
    Context context3(system, integrator3, platform);
    stream1.seekg(0, stream1.beg);
    context3.loadCheckpoint(stream1);
    State s9 = context3.getState(State::Positions | State::Velocities | State::Parameters | State::Energy);
    compareStates(s1, s9);
}

void runPlatformTests() {
    testCheckpoint();
}
