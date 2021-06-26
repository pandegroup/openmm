#ifndef OPENMM_CUSTOMTORSIONFORCE_H_
#define OPENMM_CUSTOMTORSIONFORCE_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2010-2016 Stanford University and the Authors.      *
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

#include "Force.h"
#include "Vec3.h"
#include <vector>
#include "internal/windowsExport.h"

namespace OpenMM {

/**
 * This class implements interactions between sets of four particles that depend on the torsion angle between them.
 * Unlike PeriodicTorsionForce, the functional form of the interaction is completely customizable, and may
 * involve arbitrary algebraic expressions.  In addition to the angle formed by the particles, it may depend
 * on arbitrary global and per-torsion parameters.
 *
 * To use this class, create a CustomTorsionForce object, passing an algebraic expression to the constructor
 * that defines the interaction energy between each set of particles.  The expression may depend on theta, the torsion angle
 * formed by the particles defined on the domain [-pi,+pi), as well as on any parameters you choose.  Then call addPerTorsionParameter() to define per-torsion
 * parameters, and addGlobalParameter() to define global parameters.  The values of per-torsion parameters are specified as
 * part of the system definition, while values of global parameters may be modified during a simulation by calling Context::setParameter().
 * Finally, call addTorsion() once for each torsion.  After an torsion has been added, you can modify its parameters by calling setTorsionParameters().
 * This will have no effect on Contexts that already exist unless you call updateParametersInContext().
 * Note that theta is guaranteed to be in the range [-pi,+pi], which may cause issues with force discontinuities if the energy function does not respect this domain.
 *
 * As an example, the following code creates a CustomTorsionForce that implements a periodic potential:
 *
 * <tt>CustomTorsionForce* force = new CustomTorsionForce("0.5*k*(1-cos(theta-theta0))");</tt>
 *
 * This force depends on two parameters: the spring constant k and equilibrium angle theta0.  The following code defines these parameters:
 *
 * <tt><pre>
 * force->addPerTorsionParameter("k");
 * force->addPerTorsionParameter("theta0");
 * </pre></tt>
 *
 * If a harmonic restraint is desired, it is important to be careful of the domain for theta, using an idiom like this:
 *
 * <tt>CustomTorsionForce* force = new CustomTorsionForce("0.5*k*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta-theta0); pi = 3.1415926535");</tt>
 *
 * This class also has the ability to compute derivatives of the potential energy with respect to global parameters.
 * Call addEnergyParameterDerivative() to request that the derivative with respect to a particular parameter be
 * computed.  You can then query its value in a Context by calling getState() on it.
 *
 * Expressions may involve the operators + (add), - (subtract), * (multiply), / (divide), and ^ (power), and the following
 * functions: sqrt, exp, log, sin, cos, sec, csc, tan, cot, asin, acos, atan, atan2, sinh, cosh, tanh, erf, erfc, min, max, abs, floor, ceil, step, delta, select.  All trigonometric functions
 * are defined in radians, and log is the natural logarithm.  step(x) = 0 if x is less than 0, 1 otherwise.  delta(x) = 1 if x is 0, 0 otherwise.
 * select(x,y,z) = z if x = 0, y otherwise.
 */

class OPENMM_EXPORT CustomTorsionForce : public Force {
public:
    /**
     * Create a CustomTorsionForce.
     *
     * @param energy    an algebraic expression giving the interaction energy between three particles as a function
     *                  of theta, the torsion angle between them
     */
    explicit CustomTorsionForce(const std::string& energy);
    /**
     * Get the number of torsions for which force field parameters have been defined.
     */
    int getNumTorsions() const {
        return torsions.size();
    }
    /**
     * Get the number of per-torsion parameters that the interaction depends on.
     */
    int getNumPerTorsionParameters() const {
        return parameters.size();
    }
    /**
     * Get the number of global parameters that the interaction depends on.
     */
    int getNumGlobalParameters() const {
        return globalParameters.size();
    }
    /**
     * Get the number of global parameters with respect to which the derivative of the energy
     * should be computed.
     */
    int getNumEnergyParameterDerivatives() const {
        return energyParameterDerivatives.size();
    }
    /**
     * Get the algebraic expression that gives the interaction energy for each torsion
     */
    const std::string& getEnergyFunction() const;
    /**
     * Set the algebraic expression that gives the interaction energy for each torsion
     */
    void setEnergyFunction(const std::string& energy);
    /**
     * Add a new per-torsion parameter that the interaction may depend on.
     *
     * @param name      the name of the parameter
     * @return the index of the parameter that was added
     */
    int addPerTorsionParameter(const std::string& name);
    /**
     * Get the name of a per-torsion parameter.
     *
     * @param index     the index of the parameter for which to get the name
     * @return the parameter name
     */
    const std::string& getPerTorsionParameterName(int index) const;
    /**
     * Set the name of a per-torsion parameter.
     *
     * @param index          the index of the parameter for which to set the name
     * @param name           the name of the parameter
     */
    void setPerTorsionParameterName(int index, const std::string& name);
    /**
     * Add a new global parameter that the interaction may depend on.  The default value provided to
     * this method is the initial value of the parameter in newly created Contexts.  You can change
     * the value at any time by calling setParameter() on the Context.
     *
     * @param name             the name of the parameter
     * @param defaultValue     the default value of the parameter
     * @return the index of the parameter that was added
     */
    int addGlobalParameter(const std::string& name, double defaultValue);
    /**
     * Get the name of a global parameter.
     *
     * @param index     the index of the parameter for which to get the name
     * @return the parameter name
     */
    const std::string& getGlobalParameterName(int index) const;
    /**
     * Set the name of a global parameter.
     *
     * @param index          the index of the parameter for which to set the name
     * @param name           the name of the parameter
     */
    void setGlobalParameterName(int index, const std::string& name);
    /**
     * Get the default value of a global parameter.
     *
     * @param index     the index of the parameter for which to get the default value
     * @return the parameter default value
     */
    double getGlobalParameterDefaultValue(int index) const;
    /**
     * Set the default value of a global parameter.
     *
     * @param index          the index of the parameter for which to set the default value
     * @param defaultValue   the default value of the parameter
     */
    void setGlobalParameterDefaultValue(int index, double defaultValue);
    /**
     * Request that this Force compute the derivative of its energy with respect to a global parameter.
     * The parameter must have already been added with addGlobalParameter().
     *
     * @param name             the name of the parameter
     */
    void addEnergyParameterDerivative(const std::string& name);
    /**
     * Get the name of a global parameter with respect to which this Force should compute the
     * derivative of the energy.
     *
     * @param index     the index of the parameter derivative, between 0 and getNumEnergyParameterDerivatives()
     * @return the parameter name
     */
    const std::string& getEnergyParameterDerivativeName(int index) const;
    /**
     * Add a torsion term to the force field.
     *
     * @param particle1     the index of the first particle connected by the torsion
     * @param particle2     the index of the second particle connected by the torsion
     * @param particle3     the index of the third particle connected by the torsion
     * @param particle4     the index of the fourth particle connected by the torsion
     * @param parameters    the list of parameters for the new torsion
     * @return the index of the torsion that was added
     */
    int addTorsion(int particle1, int particle2, int particle3, int particle4, const std::vector<double>& parameters=std::vector<double>());
    /**
     * Get the force field parameters for a torsion term.
     *
     * @param index              the index of the torsion for which to get parameters
     * @param[out] particle1     the index of the first particle connected by the torsion
     * @param[out] particle2     the index of the second particle connected by the torsion
     * @param[out] particle3     the index of the third particle connected by the torsion
     * @param[out] particle4     the index of the fourth particle connected by the torsion
     * @param[out] parameters    the list of parameters for the torsion
     */
    void getTorsionParameters(int index, int& particle1, int& particle2, int& particle3, int& particle4, std::vector<double>& parameters) const;
    /**
     * Set the force field parameters for a torsion term.
     *
     * @param index         the index of the torsion for which to set parameters
     * @param particle1     the index of the first particle connected by the torsion
     * @param particle2     the index of the second particle connected by the torsion
     * @param particle3     the index of the third particle connected by the torsion
     * @param particle4     the index of the fourth particle connected by the torsion
     * @param parameters    the list of parameters for the torsion
     */
    void setTorsionParameters(int index, int particle1, int particle2, int particle3, int particle4, const std::vector<double>& parameters=std::vector<double>());
    /**
     * Update the per-torsion parameters in a Context to match those stored in this Force object.  This method provides
     * an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call setTorsionParameters() to modify this object's parameters, then call updateParametersInContext()
     * to copy them over to the Context.
     *
     * This method has several limitations.  The only information it updates is the values of per-torsion parameters.
     * All other aspects of the Force (such as the energy function) are unaffected and can only be changed by reinitializing
     * the Context.  The set of particles involved in a torsion cannot be changed, nor can new torsions be added.
     */
    void updateParametersInContext(Context& context);
    /**
     * Set whether this force should apply periodic boundary conditions when calculating displacements.
     * Usually this is not appropriate for bonded forces, but there are situations when it can be useful.
     */
    void setUsesPeriodicBoundaryConditions(bool periodic);
    /**
     * Returns whether or not this force makes use of periodic boundary
     * conditions.
     *
     * @returns true if force uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const;
protected:
    ForceImpl* createImpl() const;
private:
    class TorsionInfo;
    class TorsionParameterInfo;
    class GlobalParameterInfo;
    std::string energyExpression;
    std::vector<TorsionParameterInfo> parameters;
    std::vector<GlobalParameterInfo> globalParameters;
    std::vector<TorsionInfo> torsions;
    std::vector<int> energyParameterDerivatives;
    bool usePeriodic;
};

/**
 * This is an internal class used to record information about a torsion.
 * @private
 */
class CustomTorsionForce::TorsionInfo {
public:
    int particle1, particle2, particle3, particle4;
    std::vector<double> parameters;
    TorsionInfo() : particle1(-1), particle2(-1), particle3(-1), particle4(-1) {
    }
    TorsionInfo(int particle1, int particle2, int particle3, int particle4, const std::vector<double>& parameters) :
            particle1(particle1), particle2(particle2), particle3(particle3), particle4(particle4), parameters(parameters) {
    }
};

/**
 * This is an internal class used to record information about a per-torsion parameter.
 * @private
 */
class CustomTorsionForce::TorsionParameterInfo {
public:
    std::string name;
    TorsionParameterInfo() {
    }
    TorsionParameterInfo(const std::string& name) : name(name) {
    }
};

/**
 * This is an internal class used to record information about a global parameter.
 * @private
 */
class CustomTorsionForce::GlobalParameterInfo {
public:
    std::string name;
    double defaultValue;
    GlobalParameterInfo() {
    }
    GlobalParameterInfo(const std::string& name, double defaultValue) : name(name), defaultValue(defaultValue) {
    }
};

} // namespace OpenMM

#endif /*OPENMM_CUSTOMTORSIONFORCE_H_*/
