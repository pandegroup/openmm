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
 * This tests the Hip implementation of random number generation.
 */

#include "openmm/internal/AssertionUtilities.h"
#include "HipArray.h"
#include "HipContext.h"
#include "HipIntegrationUtilities.h"
#include "openmm/System.h"
#include "openmm/Context.h"
#include "HipPlatform.h"
#include "openmm/VerletIntegrator.h"
#include "SimTKOpenMMRealType.h"
#include <iostream>

using namespace OpenMM;
using namespace std;

HipPlatform platform;

void testGaussian() {
    int numAtoms = 5000;
    System system;
    for (int i = 0; i < numAtoms; i++)
        system.addParticle(1.0);
    HipPlatform::PlatformData platformData(NULL, system, "", "true", platform.getPropertyDefaultValue("HipPrecision"), "false",
            platform.getPropertyDefaultValue(HipPlatform::HipCompiler()), platform.getPropertyDefaultValue(HipPlatform::HipTempDirectory()),
            platform.getPropertyDefaultValue(HipPlatform::HipHostCompiler()), platform.getPropertyDefaultValue(HipPlatform::HipDisablePmeStream()), "false", 1, NULL);
    HipContext& context = *platformData.contexts[0];
    context.initialize();
    context.getIntegrationUtilities().initRandomNumberGenerator(0);
    HipArray& random = context.getIntegrationUtilities().getRandom();
    context.getIntegrationUtilities().prepareRandomNumbers(random.getSize());
    const int numValues = random.getSize()*4;
    vector<float4> values(numValues);
    random.download(values);
    float* data = reinterpret_cast<float*>(&values[0]);
    double mean = 0.0;
    double var = 0.0;
    double skew = 0.0;
    double kurtosis = 0.0;
    for (int i = 0; i < numValues; i++) {
        double value = data[i];
        mean += value;
        var += value*value;
        skew += value*value*value;
        kurtosis += value*value*value*value;
    }
    mean /= numValues;
    var /= numValues;
    skew /= numValues;
    kurtosis /= numValues;
    double c2 = var-mean*mean;
    double c3 = skew-3*var*mean+2*mean*mean*mean;
    double c4 = kurtosis-4*skew*mean-3*var*var+12*var*mean*mean-6*mean*mean*mean*mean;
    ASSERT_EQUAL_TOL(0.0, mean, 3.0/sqrt((double)numValues));
    ASSERT_EQUAL_TOL(1.0, c2, 3.0/pow(numValues, 1.0/3.0));
    ASSERT_EQUAL_TOL(0.0, c3, 3.0/pow(numValues, 1.0/4.0));
    ASSERT_EQUAL_TOL(0.0, c4, 3.0/pow(numValues, 1.0/4.0));
}

void testRandomVelocities() {
    // Create a system.

    const int numParticles = 10000;
    const double temperture = 100.0;
    System system;
    VerletIntegrator integrator(0.01);
    for (int i = 0; i < numParticles; ++i)
        system.addParticle(10.0+sin(0.1*i));
    for (int i = 0; i < numParticles-1; ++i)
        system.addConstraint(i, i+1, 1.0);
    Context context(system, integrator, platform);
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; ++i)
        positions[i] = Vec3(i/2, (i+1)/2, 0);
    context.setPositions(positions);

    // Ask the context to generate random velocities.

    context.setVelocitiesToTemperature(temperture);
    State state = context.getState(State::Velocities);

    // See if they respect constraints.

    for (int i = 1; i < numParticles; i++) {
        Vec3 v1 = state.getVelocities()[i-1];
        Vec3 v2 = state.getVelocities()[i];
        double vel = (v1-v2).dot(positions[i-1]-positions[i]);
        ASSERT_EQUAL_TOL(0.0, vel, 2e-5);
    }

    // See if the temperature is correct.

    double ke = 0;
    for (int i = 0; i < numParticles; i++) {
        Vec3 v = state.getVelocities()[i];
        ke += 0.5*system.getParticleMass(i)*v.dot(v);
    }
    double expected = 0.5*(numParticles*3-system.getNumConstraints())*BOLTZ*temperture;
    ASSERT_USUALLY_EQUAL_TOL(expected, ke, 4/sqrt((double) numParticles));
}

int main(int argc, char* argv[]) {
    try {
        if (argc > 1)
            platform.setPropertyDefaultValue("HipPrecision", string(argv[1]));
        testGaussian();
        testRandomVelocities();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
