/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2010-2020 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
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

#include "openmm/LocalEnergyMinimizer.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/VerletIntegrator.h"
#include "lbfgs.h"
#include <cmath>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace OpenMM;
using namespace std;

struct MinimizerData {
    Context& context;
    double k;
    bool checkLargeForces;
    VerletIntegrator cpuIntegrator;
    Context* cpuContext;
    MinimizerData(Context& context, double k) : context(context), k(k), cpuIntegrator(1.0), cpuContext(NULL) {
        string platformName = context.getPlatform().getName();
        checkLargeForces = (platformName == "CUDA" || platformName == "OpenCL" || platformName == "HIP");
    }
    ~MinimizerData() {
        if (cpuContext != NULL)
            delete cpuContext;
    }
    Context& getCpuContext() {
        // Get an alternate context that runs on the CPU and doesn't place any limits
        // on the magnitude of forces.

        if (cpuContext == NULL) {
            Platform* cpuPlatform;
            try {
                cpuPlatform = &Platform::getPlatformByName("CPU");
            }
            catch (...) {
                cpuPlatform = &Platform::getPlatformByName("Reference");
            }
            cpuContext = new Context(context.getSystem(), cpuIntegrator, *cpuPlatform);
            cpuContext->setState(context.getState(State::Positions | State::Velocities | State::Parameters));
        }
        return *cpuContext;
    }
};

static double computeForcesAndEnergy(Context& context, const vector<Vec3>& positions, lbfgsfloatval_t *g) {
    context.setPositions(positions);
    context.computeVirtualSites();
    State state = context.getState(State::Forces | State::Energy, false, context.getIntegrator().getIntegrationForceGroups());
    const vector<Vec3>& forces = state.getForces();
    const System& system = context.getSystem();
    for (int i = 0; i < forces.size(); i++) {
        if (system.getParticleMass(i) == 0) {
            g[3*i] = 0.0;
            g[3*i+1] = 0.0;
            g[3*i+2] = 0.0;
        }
        else {
            g[3*i] = -forces[i][0];
            g[3*i+1] = -forces[i][1];
            g[3*i+2] = -forces[i][2];
        }
    }
    return state.getPotentialEnergy();
}

static lbfgsfloatval_t evaluate(void *instance, const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n, const lbfgsfloatval_t step) {
    MinimizerData* data = reinterpret_cast<MinimizerData*>(instance);
    Context& context = data->context;
    const System& system = context.getSystem();
    int numParticles = system.getNumParticles();

    // Compute the force and energy for this configuration.

    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++)
        positions[i] = Vec3(x[3*i], x[3*i+1], x[3*i+2]);
    double energy = computeForcesAndEnergy(context, positions, g);
    if (data->checkLargeForces) {
        // The CUDA, HIP and OpenCL platforms accumulate forces in fixed point, so they
        // can't handle very large forces.  Check for problematic forces (very large,
        // infinite, or NaN) and if necessary recompute them on the CPU.

        for (int i = 0; i < 3*numParticles; i++) {
            if (!(fabs(g[i]) < 2e9)) {
                energy = computeForcesAndEnergy(data->getCpuContext(), positions, g);
                break;
            }
        }
    }

    // Add harmonic forces for any constraints.

    int numConstraints = system.getNumConstraints();
    double k = data->k;
    for (int i = 0; i < numConstraints; i++) {
        int particle1, particle2;
        double distance;
        system.getConstraintParameters(i, particle1, particle2, distance);
        Vec3 delta = positions[particle2]-positions[particle1];
        double r2 = delta.dot(delta);
        double r = sqrt(r2);
        delta *= 1/r;
        double dr = r-distance;
        double kdr = k*dr;
        energy += 0.5*kdr*dr;
        g[3*particle1] -= kdr*delta[0];
        g[3*particle1+1] -= kdr*delta[1];
        g[3*particle1+2] -= kdr*delta[2];
        g[3*particle2] += kdr*delta[0];
        g[3*particle2+1] += kdr*delta[1];
        g[3*particle2+2] += kdr*delta[2];
    }
    return energy;
}

void LocalEnergyMinimizer::minimize(Context& context, double tolerance, int maxIterations) {
    const System& system = context.getSystem();
    int numParticles = system.getNumParticles();
    double constraintTol = context.getIntegrator().getConstraintTolerance();
    double workingConstraintTol = std::max(1e-4, constraintTol);
    double k = 100/workingConstraintTol;
    lbfgsfloatval_t *x = lbfgs_malloc(numParticles*3);
    if (x == NULL)
        throw OpenMMException("LocalEnergyMinimizer: Failed to allocate memory");
    try {

        // Initialize the minimizer.

        lbfgs_parameter_t param;
        lbfgs_parameter_init(&param);
        if (!context.getPlatform().supportsDoublePrecision())
            param.xtol = 1e-7;
        param.max_iterations = maxIterations;
        param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;

        // Make sure the initial configuration satisfies all constraints.

        context.applyConstraints(workingConstraintTol);

        // Record the initial positions and determine a normalization constant for scaling the tolerance.

        vector<Vec3> initialPos = context.getState(State::Positions).getPositions();
        double norm = 0.0;
        for (int i = 0; i < numParticles; i++) {
            x[3*i] = initialPos[i][0];
            x[3*i+1] = initialPos[i][1];
            x[3*i+2] = initialPos[i][2];
            norm += initialPos[i].dot(initialPos[i]);
        }
        norm /= numParticles;
        norm = (norm < 1 ? 1 : sqrt(norm));
        param.epsilon = tolerance/norm;

        // Repeatedly minimize, steadily increasing the strength of the springs until all constraints are satisfied.

        double prevMaxError = 1e10;
        MinimizerData data(context, k);
        while (true) {
            // Perform the minimization.

            lbfgsfloatval_t fx;
            lbfgs(numParticles*3, x, &fx, evaluate, NULL, &data, &param);

            // Check whether all constraints are satisfied.

            vector<Vec3> positions = context.getState(State::Positions).getPositions();
            int numConstraints = system.getNumConstraints();
            double maxError = 0.0;
            for (int i = 0; i < numConstraints; i++) {
                int particle1, particle2;
                double distance;
                system.getConstraintParameters(i, particle1, particle2, distance);
                Vec3 delta = positions[particle2]-positions[particle1];
                double r = sqrt(delta.dot(delta));
                double error = fabs(r-distance);
                if (error > maxError)
                    maxError = error;
            }
            if (maxError <= workingConstraintTol)
                break; // All constraints are satisfied.
            context.setPositions(initialPos);
            if (maxError >= prevMaxError)
                break; // Further tightening the springs doesn't seem to be helping, so just give up.
            prevMaxError = maxError;
            data.k *= 10;
            if (maxError > 100*workingConstraintTol) {
                // We've gotten far enough from a valid state that we might have trouble getting
                // back, so reset to the original positions.

                for (int i = 0; i < numParticles; i++) {
                    x[3*i] = initialPos[i][0];
                    x[3*i+1] = initialPos[i][1];
                    x[3*i+2] = initialPos[i][2];
                }
            }
        }
    }
    catch (...) {
        lbfgs_free(x);
        throw;
    }
    lbfgs_free(x);
    
    // If necessary, do a final constraint projection to make sure they are satisfied
    // to the full precision requested by the user.
    
    if (constraintTol < workingConstraintTol)
        context.applyConstraints(workingConstraintTol);
}

