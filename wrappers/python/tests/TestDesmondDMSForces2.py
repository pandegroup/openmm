import os
import sys
import tempfile
import shutil
import contextlib
import unittest
import distutils
from subprocess import check_output

import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

DESMOND_PATH = distutils.spawn.find_executable('desmond')


class TestDesmondDMSForces2(unittest.TestCase):
    def setUp(self):
        """Set up the tests by loading the input files."""

        # alanine dipeptide with explicit water
        self.path = os.path.join(os.path.dirname(__file__), 'systems/alanine-dipeptide-explicit-amber99SBILDN-tip3p.dms')
        self.dms = DesmondDMSFile(self.path)

    @unittest.skipIf(DESMOND_PATH is None, "desmond is required to be available in your PATH")
    def testForces(self):
        with print_options(suppress=True):
            cutoffs = [8, 9, 10, 11, 12]*angstrom
            for cutoff in cutoffs:
                print 'OpenMM nonbondedCutoff = %s' % cutoff
                print self._mmForces(nonbondedCutoff=cutoff)[:10]

            print '\n\n'
            for cutoff in cutoffs:
                print 'Desmond nonbondedCutoff = %s' % cutoff
                print self._desmondForces(nonbondedCutoff=cutoff)[:10]



    def _mmForces(self, nonbondedCutoff):
        system = self.dms.createSystem(nonbondedMethod=PME, nonbondedCutoff=nonbondedCutoff)
        context = Context(system, VerletIntegrator(0.0))
        context.setPositions(self.dms.positions)
        forces = np.array(context.getState(getForces=True).getForces(asNumpy=True))

        nonbonded = [system.getForce(i) for i in range(system.getNumForces()) if isinstance(system.getForce(i), NonbondedForce)][0]

        totalCharge = sum([nonbonded.getParticleParameters(i)[0].value_in_unit(elementary_charge) for i in range(system.getNumParticles())])
        assert totalCharge == 0
        return forces

    def _desmondForces(self, nonbondedCutoff):
        if DESMOND_PATH is None:
            raise ImportError("Need desmond")
        sys.path.insert(0, os.path.join(os.path.dirname(DESMOND_PATH), '..', 'lib', 'python'))
        import framesettools

        dms_path = os.path.abspath(self.path)
        with temporary_directory():
            with open('desmond.cfg', 'w') as f:
                f.write(self._desmondConfig(nonbondedCutoff))
            check_output('desmond --include desmond.cfg --cfg boot.file=%s' % dms_path, shell=True)

            fframe = [f for f in framesettools.FrameSet('forces.dtr')][0]
            tframe = [f for f in framesettools.FrameSet('trajectory.dtr')][0]

        forces = Quantity(fframe.FORCES.reshape(-1, 3), kilocalories_per_mole/angstrom)
        positions = Quantity(tframe.POSITION.reshape(-1,3), angstroms)

        # Make sure that the geometry used by OpenMM and Desmond is the same
        u = self.dms.topology.getUnitCellDimensions()[0].value_in_unit(angstrom)
        p1 = (np.array(positions.value_in_unit(angstrom)) + u) % u
        p2 = (np.array(self.dms.positions.value_in_unit(angstroms)) + u) % u
        assert np.sqrt(np.mean(np.square(np.sum(p1-p2, axis=1)))) < 1e-5

        return forces.value_in_unit(kilojoules_per_mole/nanometer)

    def _desmondConfig(self, nonbondedCutoff):
        return """
        app = mdsim

        mdsim = { title = "Compute Forces"
                  last_time = 0.000 }
        global_cell={ reference_time=0.0
                      partition = [ 0 0 0 ]
                      margin = 0
                      r_clone = %(clone)s }
        force = { term.list = []
                  bonded = {}
                  virtual = {}
                  constraint = none
                  nonbonded = { sigma = %(sigma)s
                                r_cut = %(cutoff)s
                                n_zone = 1
                                near = { type = default
                                         taper = none }
                                far = { type = pme
                                        order = [4 4 4]
                                        n_k = [128 128 128] }
                              }
                  ignore_com_dofs = false }
        migration = { first = 0.0
                      interval = 0.02 }
        integrator = { type = V_NVE
                       dt = 0.002
                       respa = { near_timesteps = 1
                                 far_timesteps = 1
                                 outer_timesteps = 1 }
                       V_NVE = {} }
        mdsim {
            plugin = { list = [ compute_forces trajectory ]
                       compute_forces = { type = compute_forces
                                          name = forces.dtr
                                          mode = clobber
                                          first = 0
                                          interval = 0.04 }
                       trajectory = { type=trajectory
                                      first = 0
                                      interval = 0.04
                                      mode = clobber
                                      write_velocity  = false
                                      write_last_step = false
                                      periodicfix  = false
                                      glue = []
                                      center = []
                                      name = trajectory.dtr }
          }
          checkpt = none
        }
        """ % {'cutoff': nonbondedCutoff.value_in_unit(angstroms),
               'clone': nonbondedCutoff.value_in_unit(angstroms)/2 + 1,
               'sigma': nonbondedCutoff.value_in_unit(angstroms)/(3*np.sqrt(2))}



@contextlib.contextmanager
def temporary_directory():
    """A context manager which changes the working directory to a
    temporary directory, then cleans it up and cds back
    """
    prev_cwd = os.getcwd()
    tempdir = tempfile.mkdtemp()
    os.chdir(tempdir)
    yield
    os.chdir(prev_cwd)
    shutil.rmtree(tempdir)

@contextlib.contextmanager
def print_options(**opts):
    oldopts = np.get_printoptions()
    newopts = oldopts.copy()
    newopts.update(opts)
    try:
        np.set_printoptions(**newopts)
        yield
    finally:
        np.set_printoptions(**oldopts)
