"""
metadynamics.py: Well-tempered metadynamics

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2018-2019 Stanford University and the Authors.
Authors: Peter Eastman

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import openmm as mm
import openmm.unit as unit
from collections import namedtuple
from functools import reduce
import os
import re
try:
    import numpy as np
except:
    pass


class Metadynamics(object):
    """Performs metadynamics.

    This class implements well-tempered metadynamics, as described in Barducci et al.,
    "Well-Tempered Metadynamics: A Smoothly Converging and Tunable Free-Energy Method"
    (https://doi.org/10.1103/PhysRevLett.100.020603).  You specify from one to three
    collective variables whose sampling should be accelerated.  A biasing force that
    depends on the collective variables is added to the simulation.  Initially the bias
    is zero.  As the simulation runs, Gaussian bumps are periodically added to the bias
    at the current location of the simulation.  This pushes the simulation away from areas
    it has already explored, encouraging it to sample other regions.  At the end of the
    simulation, the bias function can be used to calculate the system's free energy as a
    function of the collective variables.

    To use the class you create a Metadynamics object, passing to it the System you want
    to simulate and a list of BiasVariable objects defining the collective variables.
    It creates a biasing force and adds it to the System.  You then run the simulation
    as usual, but call step() on the Metadynamics object instead of on the Simulation.

    You can optionally specify a directory on disk where the current bias function should
    periodically be written.  In addition, it loads biases from any other files in the
    same directory and includes them in the simulation.  It loads files when the
    Metqdynamics object is first created, and also checks for any new files every time it
    updates its own bias on disk.

    This serves two important functions.  First, it lets you stop a metadynamics run and
    resume it later.  When you begin the new simulation, it will load the biases computed
    in the earlier simulation and continue adding to them.  Second, it provides an easy
    way to parallelize metadynamics sampling across many computers.  Just point all of
    them to a shared directory on disk.  Each process will save its biases to that
    directory, and also load in and apply the biases added by other processes.
    """

    def __init__(self, system, variables, temperature, biasFactor, height, frequency, saveFrequency=None, biasDir=None):
        """Create a Metadynamics object.

        Parameters
        ----------
        system: System
            the System to simulate.  A CustomCVForce implementing the bias is created and
            added to the System.
        variables: list of BiasVariables
            the collective variables to sample
        temperature: temperature
            the temperature at which the simulation is being run.  This is used in computing
            the free energy.
        biasFactor: float
            used in scaling the height of the Gaussians added to the bias.  The collective
            variables are sampled as if the effective temperature of the simulation were
            temperature*biasFactor.
        height: energy
            the initial height of the Gaussians to add
        frequency: int
            the interval in time steps at which Gaussians should be added to the bias potential
        saveFrequency: int (optional)
            the interval in time steps at which to write out the current biases to disk.  At
            the same time it writes biases, it also checks for updated biases written by other
            processes and loads them in.  This must be a multiple of frequency.
        biasDir: str (optional)
            the directory to which biases should be written, and from which biases written by
            other processes should be loaded
        """
        if not unit.is_quantity(temperature):
            temperature = temperature*unit.kelvin
        if not unit.is_quantity(height):
            height = height*unit.kilojoules_per_mole
        if biasFactor < 1.0:
            raise ValueError('biasFactor must be >= 1')
        if (saveFrequency is None and biasDir is not None) or (saveFrequency is not None and biasDir is None):
            raise ValueError('Must specify both saveFrequency and biasDir')
        if saveFrequency is not None and (saveFrequency < frequency or saveFrequency%frequency != 0):
            raise ValueError('saveFrequency must be a multiple of frequency')
        self.variables = variables
        self.temperature = temperature
        self.biasFactor = biasFactor
        self.height = height
        self.frequency = frequency
        self.biasDir = biasDir
        self.saveFrequency = saveFrequency
        self._id = np.random.randint(0x7FFFFFFF)
        self._saveIndex = 0
        self._selfBias = np.zeros(tuple(v.gridWidth for v in reversed(variables)))
        self._totalBias = np.zeros(tuple(v.gridWidth for v in reversed(variables)))
        self._loadedBiases = {}
        self._syncWithDisk()
        self._deltaT = temperature*(biasFactor-1)
        varNames = ['cv%d' % i for i in range(len(variables))]
        self._force = mm.CustomCVForce('table(%s)' % ', '.join(varNames))
        for name, var in zip(varNames, variables):
            self._force.addCollectiveVariable(name, var.force)
        self._widths = [v.gridWidth for v in variables]
        self._limits = sum(([v.minValue, v.maxValue] for v in variables), [])
        numPeriodics = sum(v.periodic for v in variables)
        if numPeriodics not in [0, len(variables)]:
            raise ValueError('Metadynamics cannot handle mixed periodic/non-periodic variables')
        periodic = numPeriodics == len(variables)
        if len(variables) == 1:
            self._table = mm.Continuous1DFunction(self._totalBias.flatten(), *self._limits, periodic)
        elif len(variables) == 2:
            self._table = mm.Continuous2DFunction(*self._widths, self._totalBias.flatten(), *self._limits, periodic)
        elif len(variables) == 3:
            self._table = mm.Continuous3DFunction(*self._widths, self._totalBias.flatten(), *self._limits, periodic)
        else:
            raise ValueError('Metadynamics requires 1, 2, or 3 collective variables')
        self._force.addTabulatedFunction('table', self._table)
        freeGroups = set(range(32)) - set(force.getForceGroup() for force in system.getForces())
        if len(freeGroups) == 0:
            raise RuntimeError('Cannot assign a force group to the metadynamics force. '
                               'The maximum number (32) of the force groups is already used.')
        self._force.setForceGroup(max(freeGroups))
        system.addForce(self._force)

    def step(self, simulation, steps):
        """Advance the simulation by integrating a specified number of time steps.

        Parameters
        ----------
        simulation: Simulation
            the Simulation to advance
        steps: int
            the number of time steps to integrate
        """
        stepsToGo = steps
        forceGroup = self._force.getForceGroup()
        while stepsToGo > 0:
            nextSteps = stepsToGo
            if simulation.currentStep % self.frequency == 0:
                nextSteps = min(nextSteps, self.frequency)
            else:
                nextSteps = min(nextSteps, self.frequency - simulation.currentStep % self.frequency)
            simulation.step(nextSteps)
            if simulation.currentStep % self.frequency == 0:
                position = self._force.getCollectiveVariableValues(simulation.context)
                energy = simulation.context.getState(getEnergy=True, groups={forceGroup}).getPotentialEnergy()
                height = self.height*np.exp(-energy/(unit.MOLAR_GAS_CONSTANT_R*self._deltaT))
                self._addGaussian(position, height, simulation.context)
            if self.saveFrequency is not None and simulation.currentStep % self.saveFrequency == 0:
                self._syncWithDisk()
            stepsToGo -= nextSteps

    def getFreeEnergy(self):
        """Get the free energy of the system as a function of the collective variables.

        The result is returned as a N-dimensional NumPy array, where N is the number of collective
        variables.  The values are in kJ/mole.  The i'th position along an axis corresponds to
        minValue + i*(maxValue-minValue)/gridWidth.
        """
        return -((self.temperature+self._deltaT)/self._deltaT)*self._totalBias*unit.kilojoules_per_mole

    def getCollectiveVariables(self, simulation):
        """Get the current values of all collective variables in a Simulation."""
        return self._force.getCollectiveVariableValues(simulation.context)

    def getHillHeight(self, simulation):
        """Get the current height of the Gaussian hill in kJ/mol"""
        forceGroup = self._force.getForceGroup()
        energy = simulation.context.getState(getEnergy=True, groups={forceGroup}).getPotentialEnergy()
        currentHillHeight = self.height*np.exp(-energy/(unit.MOLAR_GAS_CONSTANT_R*self._deltaT))
        scaledHillHeight = ((self.temperature+self._deltaT)/self._deltaT)*currentHillHeight
        return scaledHillHeight.value_in_unit(unit.kilojoules_per_mole)

    def _addGaussian(self, position, height, context):
        """Add a Gaussian to the bias function."""
        # Compute a Gaussian along each axis.

        axisGaussians = []
        for i,v in enumerate(self.variables):
            x = (position[i]-v.minValue) / (v.maxValue-v.minValue)
            if v.periodic:
                x = x % 1.0
            dist = np.abs(np.linspace(0, 1.0, num=v.gridWidth) - x)
            if v.periodic:
                dist = np.min(np.array([dist, np.abs(dist-1)]), axis=0)
                dist[-1] = dist[0]
            axisGaussians.append(np.exp(-0.5*dist*dist/v._scaledVariance))

        # Compute their outer product.

        if len(self.variables) == 1:
            gaussian = axisGaussians[0]
        else:
            gaussian = reduce(np.multiply.outer, reversed(axisGaussians))

        # Add it to the bias.

        height = height.value_in_unit(unit.kilojoules_per_mole)
        self._selfBias += height*gaussian
        self._totalBias += height*gaussian
        if len(self.variables) == 1:
            self._table.setFunctionParameters(self._totalBias.flatten(), *self._limits)
        else:
            self._table.setFunctionParameters(*self._widths, self._totalBias.flatten(), *self._limits)
        self._force.updateParametersInContext(context)

    def _syncWithDisk(self):
        """Save biases to disk, and check for updated files created by other processes."""
        if self.biasDir is None:
            return

        # Check for any files updated by other processes first.

        fileLoaded = False
        pattern = re.compile('bias_(.*)_(.*)\.npy')
        for filename in os.listdir(self.biasDir):
            match = pattern.match(filename)
            if match is not None:
                matchId = int(match.group(1))
                matchIndex = int(match.group(2))
                if matchId != self._id and (matchId not in self._loadedBiases or matchIndex > self._loadedBiases[matchId].index):
                    try:
                        data = np.load(os.path.join(self.biasDir, filename))
                        self._loadedBiases[matchId] = _LoadedBias(matchId, matchIndex, data)
                        fileLoaded = True
                    except IOError:
                        # There's a tiny chance the file could get deleted by another process between when
                        # we check the directory and when we try to load it.  If so, just ignore the error
                        # and keep using whatever version of that process' biases we last loaded.
                        pass

        # If we loaded any files, recompute the total bias from all processes.

        if fileLoaded:
            self._totalBias = self._selfBias
            for bias in self._loadedBiases.values():
                diffInBias = bias.bias - self._totalBias
                diffInBias[diffInBias < 0] = 0
                self._totalBias += diffInBias

            self._selfBias = self._totalBias

        # Use a safe save to write out the biases to disk, then delete the older file.

        oldName = os.path.join(self.biasDir, 'bias_%d_%d.npy' % (self._id, self._saveIndex))
        self._saveIndex += 1
        tempName = os.path.join(self.biasDir, 'temp_%d_%d.npy' % (self._id, self._saveIndex))
        fileName = os.path.join(self.biasDir, 'bias_%d_%d.npy' % (self._id, self._saveIndex))
        np.save(tempName, self._selfBias)
        os.rename(tempName, fileName)
        if os.path.exists(oldName):
            os.remove(oldName)

class BiasVariable(object):
    """A collective variable that can be used to bias a simulation with metadynamics."""

    def __init__(self, force, minValue, maxValue, biasWidth, periodic=False, gridWidth=None):
        """Create a BiasVariable.

        Parameters
        ----------
        force: Force
            the Force object whose potential energy defines the collective variable
        minValue: float or unit.Quantity
            the minimum value the collective variable can take.  If it should ever go below this,
            the bias force will be set to 0.
        maxValue: float or unit.Quantity
            the maximum value the collective variable can take.  If it should ever go above this,
            the bias force will be set to 0.
        biasWidth: float or unit.Quantity
            the width (standard deviation) of the Gaussians added to the bias during metadynamics
        periodic: bool (optional)
            whether this is a periodic variable, such that minValue and maxValue are physical equivalent
        gridWidth: int (optional)
            the number of grid points to use when tabulating the bias function.  If this is omitted,
            a reasonable value is chosen automatically.
        """
        self.force = force
        self.minValue = self._standardize(minValue)
        self.maxValue = self._standardize(maxValue)
        self.biasWidth = self._standardize(biasWidth)
        if not isinstance(periodic, bool):
            raise ValueError("BiasVariable: invalid argument")
        self.periodic = periodic
        if gridWidth is None:
            self.gridWidth = int(np.ceil(5*(maxValue-minValue)/biasWidth))
        else:
            self.gridWidth = gridWidth
        self._scaledVariance = (self.biasWidth/(self.maxValue-self.minValue))**2

    def _standardize(self, quantity):
        if unit.is_quantity(quantity):
            return quantity.value_in_unit_system(unit.md_unit_system)
        else:
            return quantity

_LoadedBias = namedtuple('LoadedBias', ['id', 'index', 'bias'])
