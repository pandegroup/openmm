/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2015 Stanford University and the Authors.      *
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
#include "TestCustomIntegrator.h"

/**
 * Make sure random numbers are computed correctly when steps get merged.
 */
void testMergedRandoms() {
    const int numParticles = 10;
    const int numSteps = 10;
    System system;
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    CustomIntegrator integrator(0.1);
    integrator.addPerDofVariable("dofUniform1", 0);
    integrator.addPerDofVariable("dofUniform2", 0);
    integrator.addPerDofVariable("dofGaussian1", 0);
    integrator.addPerDofVariable("dofGaussian2", 0);
    integrator.addGlobalVariable("globalUniform1", 0);
    integrator.addGlobalVariable("globalUniform2", 0);
    integrator.addGlobalVariable("globalGaussian1", 0);
    integrator.addGlobalVariable("globalGaussian2", 0);
    integrator.addComputePerDof("dofUniform1", "uniform");
    integrator.addComputePerDof("dofUniform2", "uniform");
    integrator.addComputePerDof("dofGaussian1", "gaussian");
    integrator.addComputePerDof("dofGaussian2", "gaussian");
    integrator.addComputeGlobal("globalUniform1", "uniform");
    integrator.addComputeGlobal("globalUniform2", "uniform");
    integrator.addComputeGlobal("globalGaussian1", "gaussian");
    integrator.addComputeGlobal("globalGaussian2", "gaussian");
    Context context(system, integrator, platform);

    // See if the random numbers are computed correctly.

    vector<Vec3> values1, values2;
    for (int i = 0; i < numSteps; i++) {
        integrator.step(1);
        integrator.getPerDofVariable(0, values1);
        integrator.getPerDofVariable(1, values2);
        for (int i = 0; i < numParticles; i++)
            for (int j = 0; j < 3; j++) {
                double v1 = values1[i][j];
                double v2 = values2[i][j];
                ASSERT(v1 >= 0 && v1 < 1);
                ASSERT(v2 >= 0 && v2 < 1);
                ASSERT(v1 != v2);
            }
        integrator.getPerDofVariable(2, values1);
        integrator.getPerDofVariable(3, values2);
        for (int i = 0; i < numParticles; i++)
            for (int j = 0; j < 3; j++) {
                double v1 = values1[i][j];
                double v2 = values2[i][j];
                ASSERT(v1 >= -10 && v1 < 10);
                ASSERT(v2 >= -10 && v2 < 10);
                ASSERT(v1 != v2);
            }
        double v1 = integrator.getGlobalVariable(0);
        double v2 = integrator.getGlobalVariable(1);
        ASSERT(v1 >= 0 && v1 < 1);
        ASSERT(v2 >= 0 && v2 < 1);
        ASSERT(v1 != v2);
        v1 = integrator.getGlobalVariable(2);
        v2 = integrator.getGlobalVariable(3);
        ASSERT(v1 >= -10 && v1 < 10);
        ASSERT(v2 >= -10 && v2 < 10);
        ASSERT(v1 != v2);
    }
}

void runPlatformTests() {
    testMergedRandoms();
}
