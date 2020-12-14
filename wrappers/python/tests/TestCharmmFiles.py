import unittest
from validateConstraints import *
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import simtk.openmm.app.element as elem
import math
import os
import tempfile
import warnings
if sys.version_info >= (3,0):
    from io import StringIO
else:
    from cStringIO import StringIO

class TestCharmmFiles(unittest.TestCase):

    """Test the GromacsTopFile.createSystem() method."""

    def setUp(self):
        """Set up the tests by loading the input files."""

        # alanine tripeptide; no waters
        self.psf_c = CharmmPsfFile('systems/ala_ala_ala.psf')
        self.psf_x = CharmmPsfFile('systems/ala_ala_ala.xpsf')
        self.psf_v = CharmmPsfFile('systems/ala_ala_ala.vpsf')
        self.params = CharmmParameterSet(
                            'systems/charmm22.rtf', 'systems/charmm22.par')
        self.pdb = PDBFile('systems/ala_ala_ala.pdb')

    def test_NonbondedMethod(self):
        """Test both non-periodic methods for the systems"""

        methodMap = {NoCutoff:NonbondedForce.NoCutoff,
                     CutoffNonPeriodic:NonbondedForce.CutoffNonPeriodic}
        for top in (self.psf_c, self.psf_x, self.psf_v):
            for method in methodMap:
                system = top.createSystem(self.params, nonbondedMethod=method)
                forces = system.getForces()
                self.assertTrue(any(isinstance(f, NonbondedForce) and
                                    f.getNonbondedMethod()==methodMap[method]
                                    for f in forces))

    def test_Cutoff(self):
        """Test to make sure the nonbondedCutoff parameter is passed correctly."""

        for top in (self.psf_c, self.psf_x, self.psf_v):
            for method in [CutoffNonPeriodic]:
                system = top.createSystem(self.params, nonbondedMethod=method,
                                          nonbondedCutoff=2*nanometer,
                                          constraints=HBonds)
                cutoff_distance = 0.0*nanometer
                cutoff_check = 2.0*nanometer
                for force in system.getForces():
                    if isinstance(force, NonbondedForce):
                        cutoff_distance = force.getCutoffDistance()
                self.assertEqual(cutoff_distance, cutoff_check)

    def test_RemoveCMMotion(self):
        """Test both options (True and False) for the removeCMMotion parameter."""

        for b in [True, False]:
            system = self.psf_c.createSystem(self.params, removeCMMotion=b)
            self.assertEqual(any(isinstance(f, CMMotionRemover) for f in system.getForces()), b)

    def test_ImplicitSolvent(self):
        """Test implicit solvent using the implicitSolvent parameter.

        """
        system = self.psf_v.createSystem(self.params, implicitSolvent=OBC2, gbsaModel='ACE')
        self.assertTrue(any(isinstance(f, CustomGBForce) for f in system.getForces()))

    def test_ImplicitSolventParameters(self):
        """Test that solventDielectric and soluteDielectric are passed correctly.

        """
        system = self.psf_x.createSystem(self.params, implicitSolvent=GBn,
                                         solventDielectric=50.0,
                                         soluteDielectric = 0.9)
        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                self.assertEqual(force.getReactionFieldDielectric(), 1.0)

    def test_HydrogenMass(self):
        """Test that altering the mass of hydrogens works correctly."""

        topology = self.psf_v.topology
        hydrogenMass = 4*amu
        system1 = self.psf_v.createSystem(self.params)
        system2 = self.psf_v.createSystem(self.params, hydrogenMass=hydrogenMass)
        for atom in topology.atoms():
            if atom.element == elem.hydrogen:
                self.assertNotEqual(hydrogenMass, system1.getParticleMass(atom.index))
                self.assertEqual(hydrogenMass, system2.getParticleMass(atom.index))
        totalMass1 = sum([system1.getParticleMass(i) for i in range(system1.getNumParticles())]).value_in_unit(amu)
        totalMass2 = sum([system2.getParticleMass(i) for i in range(system2.getNumParticles())]).value_in_unit(amu)
        self.assertAlmostEqual(totalMass1, totalMass2)

    def test_DrudeMass(self):
        """Test that setting the mass of Drude particles works correctly."""

        psf = CharmmPsfFile('systems/ala3_solv_drude.psf')
        crd = CharmmCrdFile('systems/ala3_solv_drude.crd')
        params = CharmmParameterSet('systems/toppar_drude_master_protein_2013e.str')
        system = psf.createSystem(params, drudeMass=0)
        trueMass = [system.getParticleMass(i) for i in range(system.getNumParticles())]
        drudeMass = 0.3*amu
        system = psf.createSystem(params, drudeMass=drudeMass)
        adjustedMass = [system.getParticleMass(i) for i in range(system.getNumParticles())]
        drudeForce = [f for f in system.getForces() if isinstance(f, DrudeForce)][0]
        drudeParticles = set()
        parentParticles = set()
        for i in range(drudeForce.getNumParticles()):
            params = drudeForce.getParticleParameters(i)
            drudeParticles.add(params[0])
            parentParticles.add(params[1])
        for i in range(system.getNumParticles()):
            if i in drudeParticles:
                self.assertEqual(0*amu, trueMass[i])
                self.assertEqual(drudeMass, adjustedMass[i])
            elif i in parentParticles:
                self.assertEqual(trueMass[i]-drudeMass, adjustedMass[i])
            else:
                self.assertEqual(trueMass[i], adjustedMass[i])

    def test_NBFIX(self):
        """Tests CHARMM systems with NBFIX Lennard-Jones modifications"""
        warnings.filterwarnings('ignore', category=CharmmPSFWarning)
        psf = CharmmPsfFile('systems/ala3_solv.psf', unitCellDimensions=Vec3(32.7119500, 32.9959600, 33.0071500)*angstroms)
        crd = CharmmCrdFile('systems/ala3_solv.crd')
        params = CharmmParameterSet('systems/par_all36_prot.prm',
                                    'systems/toppar_water_ions.str')

        # Turn off charges so we only test the Lennard-Jones energies
        for a in psf.atom_list:
            a.charge = 0.0

        # Now compute the full energy
        plat = Platform.getPlatformByName('Reference')
        system = psf.createSystem(params, nonbondedMethod=PME,
                                  nonbondedCutoff=8*angstroms)

        con = Context(system, VerletIntegrator(2*femtoseconds), plat)
        con.setPositions(crd.positions)

        state = con.getState(getEnergy=True, enforcePeriodicBox=True)
        ene = state.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
        self.assertAlmostEqual(ene, 15559.71602, delta=0.05)

    def test_NBFIX_14(self):
        """Tests CHARMM systems with NBFIX 1-4 special Lennard-Jones modifications"""
        warnings.filterwarnings('ignore', category=CharmmPSFWarning)
        psf = CharmmPsfFile('systems/rna_drude.psf')
        crd = CharmmCrdFile('systems/rna_drude.crd')
        params = CharmmParameterSet('systems/toppar_drude_master_protein_2020b_mod.str',
                                    'systems/toppar_drude_nucleic_acid_2020b.str')
        # Box dimensions (cubic box)
        psf.setBox(65*angstroms, 65*angstroms, 65*angstroms)

        # Now compute the full energy
        plat = Platform.getPlatformByName('Reference')
        system = psf.createSystem(params, nonbondedMethod=PME, ewaldErrorTolerance=0.00005)
        integrator = DrudeLangevinIntegrator(300*kelvin, 1.0/picosecond, 1*kelvin, 10/picosecond, 0.001*picoseconds)
        con = Context(system, integrator, plat)
        con.setPositions(crd.positions)

        state = con.getState(getEnergy=True, enforcePeriodicBox=True)
        ene = state.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
        self.assertAlmostEqual(ene, -1103.11598, delta=1.0)

    def test_Drude(self):
        """Test CHARMM systems with Drude force field"""
        warnings.filterwarnings('ignore', category=CharmmPSFWarning)
        psf = CharmmPsfFile('systems/ala3_solv_drude.psf')
        crd = CharmmCrdFile('systems/ala3_solv_drude.crd')
        params = CharmmParameterSet('systems/toppar_drude_master_protein_2013e.str')
        # Box dimensions (cubic box)
        psf.setBox(33.2*angstroms, 33.2*angstroms, 33.2*angstroms)

        # Now compute the full energy
        plat = Platform.getPlatformByName('Reference')
        system = psf.createSystem(params, nonbondedMethod=PME, ewaldErrorTolerance=0.00005)
        integrator = DrudeLangevinIntegrator(300*kelvin, 1.0/picosecond, 1*kelvin, 10/picosecond, 0.001*picoseconds)
        con = Context(system, integrator, plat)
        con.setPositions(crd.positions)

        state = con.getState(getEnergy=True, enforcePeriodicBox=True)
        ene = state.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
        self.assertAlmostEqual(ene, -1788.36644, delta=1.0)

    def test_Lonepair(self):
        """Test the lonepair facilities, in particular the colinear type of lonepairs"""
        warnings.filterwarnings('ignore', category=CharmmPSFWarning)
        psf = CharmmPsfFile('systems/chlb_cgenff.psf')
        crd = CharmmCrdFile('systems/chlb_cgenff.crd')
        params = CharmmParameterSet('systems/top_all36_cgenff.rtf',
                                    'systems/par_all36_cgenff.prm')
        plat = Platform.getPlatformByName('Reference')
        system = psf.createSystem(params)
        con = Context(system, VerletIntegrator(2*femtoseconds), plat)
        con.setPositions(crd.positions)
        init_coor = con.getState(getPositions=True).getPositions()
        # move the position of the lonepair and recompute its coordinates
        plp=12
        crd.positions[plp] = Vec3(0.5, 1.0, 1.5) * angstrom
        con.setPositions(crd.positions)
        con.computeVirtualSites()
        new_coor = con.getState(getPositions=True).getPositions()
        
        self.assertAlmostEqual(init_coor[plp][0]/nanometers, new_coor[plp][0]/nanometers)
        self.assertAlmostEqual(init_coor[plp][1]/nanometers, new_coor[plp][1]/nanometers)
        self.assertAlmostEqual(init_coor[plp][2]/nanometers, new_coor[plp][2]/nanometers)

    def test_InsCode(self):
        """ Test the parsing of PSF files that contain insertion codes in their residue numbers """
        psf = CharmmPsfFile('systems/4TVP-dmj_wat-ion.psf')
        self.assertEqual(len(list(psf.topology.atoms())), 66264)
        self.assertEqual(len(list(psf.topology.residues())), 20169)
        self.assertEqual(len(list(psf.topology.bonds())), 46634)

    def testSystemOptions(self):
        """ Test various options in CharmmPsfFile.createSystem """
        warnings.filterwarnings('ignore', category=CharmmPSFWarning)
        psf = CharmmPsfFile('systems/ala3_solv.psf',
                            periodicBoxVectors=(Vec3(32.7119500, 0, 0)*angstroms, Vec3(0, 32.9959600, 0)*angstroms, Vec3(0, 0, 33.0071500)*angstroms))
        crd = CharmmCrdFile('systems/ala3_solv.crd')
        params = CharmmParameterSet('systems/par_all36_prot.prm',
                                    'systems/toppar_water_ions.str')

        # Check some illegal options
        self.assertRaises(ValueError, lambda:
                    psf.createSystem(params, nonbondedMethod=5))
        self.assertRaises(TypeError, lambda:
                    psf.createSystem(params, nonbondedMethod=PME,
                                     nonbondedCutoff=1*radian)
        )
        self.assertRaises(TypeError, lambda:
                    psf.createSystem(params, nonbondedMethod=PME,
                                     switchDistance=1*radian)
        )

        # Check what should be some legal options
        psf.createSystem(params, nonbondedMethod=PME, switchDistance=0.8,
                         nonbondedCutoff=1.2)
        psf.createSystem(params, nonbondedMethod=PME, switchDistance=0.8,
                         nonbondedCutoff=1.2*nanometer)

    def test_ImplicitSolventForces(self):
        """Compute forces for different implicit solvent types, and compare them to ones generated with a previous version of OpenMM to ensure they haven't changed."""
        solventType = [HCT, OBC1, OBC2, GBn, GBn2]
        nonbondedMethod = [NoCutoff, CutoffNonPeriodic, CutoffNonPeriodic, NoCutoff, NoCutoff]
        salt = [0.0, 0.0, 0.5, 0.5, 0.0]*(moles/liter)
        file = ['HCT_NoCutoff', 'OBC1_NonPeriodic', 'OBC2_NonPeriodic_Salt', 'GBn_NoCutoff_Salt', 'GBn2_NoCutoff']
        for i in range(5):
            system = self.psf_c.createSystem(self.params, implicitSolvent=solventType[i], nonbondedMethod=nonbondedMethod[i], implicitSolventSaltConc=salt[i])
            integrator = VerletIntegrator(0.001)
            context = Context(system, integrator, Platform.getPlatformByName("Reference"))
            context.setPositions(self.pdb.positions)
            state1 = context.getState(getForces=True)
            #out = open('systems/ala-ala-ala-implicit-forces/'+file[i]+'.xml', 'w')
            #out.write(XmlSerializer.serialize(state1))
            #out.close()
            with open('systems/ala-ala-ala-implicit-forces/'+file[i]+'.xml') as xml:
                state2 = XmlSerializer.deserialize(xml.read())
            for f1, f2, in zip(state1.getForces().value_in_unit(kilojoules_per_mole/nanometer), state2.getForces().value_in_unit(kilojoules_per_mole/nanometer)):
                diff = norm(f1-f2)
                self.assertTrue(diff < 0.1 or diff/norm(f1) < 1e-4)

    def test_PermissiveRead(self):
        """Compare permissive and strict reading of Charmm parameters"""

        psf = CharmmPsfFile('systems/5dhfr_cube.psf')
        pdb = PDBFile('systems/5dhfr_cube.pdb')

        params_strict     = CharmmParameterSet('systems/par_all22_prot_with_mass.inp')
        params_permissive = CharmmParameterSet('systems/par_all22_prot.inp', permissive=True)
        # Box dimensions (found from bounding box)
        psf.setBox(62.23*angstroms, 62.23*angstroms, 62.23*angstroms)

        # Turn off charges so we only test the Lennard-Jones energies
        for a in psf.atom_list:
            a.charge = 0.0

        # Now compute the full energy
        plat = Platform.getPlatformByName('Reference')

        system_strict     = psf.createSystem(params_strict    , nonbondedMethod=PME,
                                  nonbondedCutoff=8*angstroms)
        system_permissive = psf.createSystem(params_permissive, nonbondedMethod=PME,
                                  nonbondedCutoff=8*angstroms)

        con_strict     = Context(system_strict    , VerletIntegrator(2*femtoseconds), plat)
        con_permissive = Context(system_permissive, VerletIntegrator(2*femtoseconds), plat)

        con_strict.setPositions(pdb.positions)
        con_permissive.setPositions(pdb.positions)

        state_strict     = con_strict.getState(getEnergy=True, enforcePeriodicBox=True)
        state_permissive = con_permissive.getState(getEnergy=True, enforcePeriodicBox=True)

        ene_strict     = state_strict.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
        ene_permissive = state_permissive.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
        self.assertAlmostEqual(ene_strict, ene_permissive, delta=0.00001)
    
    def test_Impropers(self):
        """Test CHARMM improper torsions."""
        psf = CharmmPsfFile('systems/improper.psf')
        system = psf.createSystem(self.params)
        force = [f for f in system.getForces() if isinstance(f, CustomTorsionForce)][0]
        group = force.getForceGroup()
        integrator = VerletIntegrator(0.001)
        context = Context(system, integrator, Platform.getPlatformByName("Reference"))
        angle = 0.1
        pos1 = [Vec3(0,0,0), Vec3(1,0,0), Vec3(1,1,0), Vec3(0,1,math.tan(angle))] # theta = angle
        pos2 = [Vec3(0,0,0), Vec3(1,0,0), Vec3(1,1,0), Vec3(2,1,math.tan(angle))] # theta = pi-angle
        pos3 = [Vec3(0,0,0), Vec3(1,0,0), Vec3(1,1,0), Vec3(2,1,-math.tan(angle))] # theta = -pi+angle
        for theta0 in (0, math.pi):
            force.setTorsionParameters(0, 0, 1, 2, 3, [1.0, theta0])
            force.updateParametersInContext(context)
            for pos in (pos1, pos2, pos3):
                context.setPositions(pos)
                energy = context.getState(getEnergy=True, groups={group}).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
                if (theta0 == 0 and pos == pos1) or (theta0 == math.pi and pos in (pos2, pos3)):
                    dtheta = angle
                else:
                    dtheta = math.pi-angle
                self.assertAlmostEqual(energy, dtheta**2, delta=1e-5)

    def test_Residues(self):
        """Test that residues are read correctly, even if they have the same RESID while being in separate segments."""
        m14 = (["C{}".format(i) for i in range(1,14)]
               + ["H{}".format(i) for i in range(1,12)]
               + ["N{}".format(i) for i in range(1,4)]
               )
        hoh = ["O", "H1", "H2"]
        pot = ["POT"]
        cla = ["CLA"]
        psf = CharmmPsfFile('systems/charmm-solvated/isa_wat.3_kcl.m14.psf')
        for residue in psf.topology.residues():
            atoms = [atom.name for atom in residue.atoms()]
            if residue.name == "M14":
                self.assertEqual(sorted(m14), sorted(atoms))
            elif residue.name == "HOH":
                self.assertEqual(sorted(hoh), sorted(atoms))
            elif residue.name == "POT":
                self.assertEqual(sorted(pot), sorted(atoms))
            elif residue.name == "CLA":
                self.assertEqual(sorted(cla), sorted(atoms))
            else:
                self.assertTrue(False)

    def test_NoLongRangeCorrection(self):
        """Test that long range correction is disabled."""
        parameters = CharmmParameterSet(
            'systems/charmm-solvated/envi.str',
            'systems/charmm-solvated/m14.rtf',
            'systems/charmm-solvated/m14.prm'
        )
        psf = CharmmPsfFile('systems/charmm-solvated/isa_wat.3_kcl.m14.psf')
        psf.setBox(3.0584*nanometers,3.0584*nanometers,3.0584*nanometers)
        system = psf.createSystem(parameters, nonbondedMethod=PME)
        for force in system.getForces():
            if isinstance(force, CustomNonbondedForce):
                self.assertFalse(force.getUseLongRangeCorrection())
            if isinstance(force, NonbondedForce):
                self.assertFalse(force.getUseDispersionCorrection())

    def test_NoPsfWarning(self):
        """Test that PSF warning is not thrown."""
        parameters = CharmmParameterSet(
            'systems/charmm-solvated/envi.str',
            'systems/charmm-solvated/m14.rtf',
            'systems/charmm-solvated/m14.prm'
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", CharmmPSFWarning)
            psf = CharmmPsfFile('systems/charmm-solvated/isa_wat.3_kcl.m14.psf')
            psf.setBox(3.0584*nanometers,3.0584*nanometers,3.0584*nanometers)
            psf.createSystem(parameters, nonbondedMethod=PME)

    def test_NBXMod(self):
        """Test that all values of NBXMod are interpreted correctly."""
        crd = CharmmCrdFile('systems/ala_ala_ala.crd')
        with open('systems/charmm22.par') as parfile:
            par = parfile.read()
        # The following values were computed with CHARMM.
        modeEnergy = {0: 754318.20507, 1: 754318.20507, 2: 908.35224, 3: 59.65279, 4: -241.12856, 5: 39.13169}
        for nbxmod in range(-5, 6):
            with tempfile.NamedTemporaryFile(suffix='.par', mode='w', delete=False) as parfile:
                parfile.write(par.replace('nbxmod  5', 'nbxmod %d' % nbxmod))
                parfile.close()
                params = CharmmParameterSet('systems/charmm22.rtf', parfile.name)
                os.remove(parfile.name)
            system = self.psf_c.createSystem(params, nonbondedMethod=NoCutoff)
            context = Context(system, VerletIntegrator(1*femtoseconds), Platform.getPlatformByName('Reference'))
            context.setPositions(crd.positions)
            energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilocalories_per_mole)
            self.assertAlmostEqual(energy, modeEnergy[abs(nbxmod)], delta=1e-3*abs(energy))

    def test_Nonbonded_Exclusion(self):
        """Test that the 1-2, 1-3 and 1-4 pairs are correctly excluded or scaled."""
        psf = CharmmPsfFile('systems/MoS2.psf')
        pdb = PDBFile('systems/MoS2.pdb')
        params = CharmmParameterSet('systems/MoS2.prm')
        system = psf.createSystem(params, nonbondedMethod=NoCutoff)
        context = Context(system, VerletIntegrator(1*femtoseconds), Platform.getPlatformByName('Reference'))
        context.setPositions(pdb.positions)
        energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilocalories_per_mole)
        # Compare with value computed with NAMD.
        self.assertAlmostEqual(energy, -2154.5539, delta=1e-3*abs(energy))

    def test_Constraints(self):
        """Test that bond and angles constraints are correctly added into the system"""
        psf = CharmmPsfFile('systems/water_methanol.psf')
        params = CharmmParameterSet('systems/water_methanol.prm')
        # the system is made of one water molecule and one methanol molecule
        hBonds_water = [[0, 1], [1, 2]]
        hAngles_water = [[0, 2]]
        hBonds_methanol = [[3, 4], [3, 5], [3, 6], [7, 8]]
        allBonds_methanol = hBonds_methanol + [[3, 7]]
        hAngles_methanol = [[4, 5], [4, 6], [5, 6], [3, 8]]
        system = psf.createSystem(params, constraints=None, rigidWater=False)
        self.assertEqual(system.getNumConstraints(), 0)
        system = psf.createSystem(params, constraints=None, rigidWater=True)
        self.assertEqual(sorted(system.getConstraintParameters(i)[:2] for i in range(system.getNumConstraints())),
                         sorted(hBonds_water + hAngles_water))
        system = psf.createSystem(params, constraints=HBonds, rigidWater=False)
        self.assertEqual(sorted(system.getConstraintParameters(i)[:2] for i in range(system.getNumConstraints())),
                         sorted(hBonds_water + hBonds_methanol))
        system = psf.createSystem(params, constraints=HBonds, rigidWater=True)
        self.assertEqual(sorted(system.getConstraintParameters(i)[:2] for i in range(system.getNumConstraints())),
                         sorted(hBonds_water + hAngles_water + hBonds_methanol))
        system = psf.createSystem(params, constraints=AllBonds, rigidWater=False)
        self.assertEqual(sorted(system.getConstraintParameters(i)[:2] for i in range(system.getNumConstraints())),
                         sorted(hBonds_water + allBonds_methanol))
        system = psf.createSystem(params, constraints=AllBonds, rigidWater=True)
        self.assertEqual(sorted(system.getConstraintParameters(i)[:2] for i in range(system.getNumConstraints())),
                         sorted(hBonds_water + hAngles_water + allBonds_methanol))
        system = psf.createSystem(params, constraints=HAngles, rigidWater=False)
        self.assertEqual(sorted(system.getConstraintParameters(i)[:2] for i in range(system.getNumConstraints())),
                         sorted(hBonds_water + hAngles_water + allBonds_methanol + hAngles_methanol))
        system = psf.createSystem(params, constraints=HAngles, rigidWater=True)
        self.assertEqual(sorted(system.getConstraintParameters(i)[:2] for i in range(system.getNumConstraints())),
                         sorted(hBonds_water + hAngles_water + allBonds_methanol + hAngles_methanol))

    def test_Constraints_charmm(self):
        """Tests that CHARMM and OpenMM implementation of CHARMM force field produce the same constraints and energy"""
        warnings.filterwarnings('ignore', category=CharmmPSFWarning)
        psf = CharmmPsfFile('systems/ala3_solv.psf',
                            unitCellDimensions=Vec3(32.7119500, 32.9959600, 33.0071500) * angstroms)
        crd = CharmmCrdFile('systems/ala3_solv.crd')
        params = CharmmParameterSet('systems/par_all36_prot.prm',
                                    'systems/toppar_water_ions.str')
        plat = Platform.getPlatformByName('Reference')
        system_charmm = psf.createSystem(params, nonbondedMethod=PME,
                                  nonbondedCutoff=8 * angstroms)
        topology = psf.topology
        forcefield = ForceField('charmm36.xml', 'charmm36/water.xml')
        system_openmm = forcefield.createSystem(topology, nonbondedMethod=PME,
                                  nonbondedCutoff=8 * angstroms)
        # Test different combinations of constrains/rigidWater parameters
        system_charmm = psf.createSystem(params, constraints=None, rigidWater=False, nonbondedMethod=PME,
                                         nonbondedCutoff=8 * angstroms)
        system_openmm = forcefield.createSystem(topology, constraints=None, rigidWater=False, nonbondedMethod=PME,
                                                nonbondedCutoff=8 * angstroms)
        self.assertEqual(system_charmm.getNumConstraints(), 0)
        self.assertEqual(system_openmm.getNumConstraints(), 0)
        con_charmm = Context(system_charmm, VerletIntegrator(2 * femtoseconds), plat)
        con_charmm.setPositions(crd.positions)
        con_openmm = Context(system_openmm, VerletIntegrator(2 * femtoseconds), plat)
        con_openmm.setPositions(crd.positions)
        state_charmm = con_charmm.getState(getEnergy=True, enforcePeriodicBox=True)
        ene_charmm = state_charmm.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
        state_openmm = con_openmm.getState(getEnergy=True, enforcePeriodicBox=True)
        ene_openmm = state_openmm.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
        self.assertAlmostEqual(ene_charmm, ene_openmm, delta=0.05)

        system_charmm = psf.createSystem(params, constraints=None, rigidWater=True, nonbondedMethod=PME,
                                         nonbondedCutoff=8 * angstroms)
        system_openmm = forcefield.createSystem(topology, constraints=None, rigidWater=True, nonbondedMethod=PME,
                                                nonbondedCutoff=8 * angstroms)
        self.assertEqual(
            sorted(system_charmm.getConstraintParameters(i)[:2] for i in range(system_charmm.getNumConstraints())),
            sorted(system_openmm.getConstraintParameters(j)[:2] for j in range(system_openmm.getNumConstraints())))
        for i in range(system_charmm.getNumConstraints()):
            self.assertAlmostEqual(system_charmm.getConstraintParameters(i)[2],
                                   system_openmm.getConstraintParameters(i)[2], delta=1e-7 * nanometers)

        system_charmm = psf.createSystem(params, constraints=HBonds, rigidWater=False, nonbondedMethod=PME,
                                         nonbondedCutoff=8 * angstroms)
        system_openmm = forcefield.createSystem(topology, constraints=HBonds, rigidWater=False, nonbondedMethod=PME,
                                                nonbondedCutoff=8 * angstroms)
        self.assertEqual(
            sorted(system_charmm.getConstraintParameters(i)[:2] for i in range(system_charmm.getNumConstraints())),
            sorted(system_openmm.getConstraintParameters(j)[:2] for j in range(system_openmm.getNumConstraints())))
        for i in range(system_charmm.getNumConstraints()):
            self.assertAlmostEqual(system_charmm.getConstraintParameters(i)[2],
                                   system_openmm.getConstraintParameters(i)[2], delta=1e-7 * nanometers)

        system_charmm = psf.createSystem(params, constraints=HBonds, rigidWater=True, nonbondedMethod=PME,
                                         nonbondedCutoff=8 * angstroms)
        system_openmm = forcefield.createSystem(topology, constraints=HBonds, rigidWater=True, nonbondedMethod=PME,
                                                nonbondedCutoff=8 * angstroms)
        self.assertEqual(
            sorted(system_charmm.getConstraintParameters(i)[:2] for i in range(system_charmm.getNumConstraints())),
            sorted(system_openmm.getConstraintParameters(j)[:2] for j in range(system_openmm.getNumConstraints())))
        for i in range(system_charmm.getNumConstraints()):
            self.assertAlmostEqual(system_charmm.getConstraintParameters(i)[2],
                                   system_openmm.getConstraintParameters(i)[2], delta=1e-7 * nanometers)

        system_charmm = psf.createSystem(params, constraints=AllBonds, rigidWater=False, nonbondedMethod=PME,
                                         nonbondedCutoff=8 * angstroms)
        system_openmm = forcefield.createSystem(topology, constraints=AllBonds, rigidWater=False, nonbondedMethod=PME,
                                                nonbondedCutoff=8 * angstroms)
        self.assertEqual(
            sorted(system_charmm.getConstraintParameters(i)[:2] for i in range(system_charmm.getNumConstraints())),
            sorted(system_openmm.getConstraintParameters(j)[:2] for j in range(system_openmm.getNumConstraints())))
        for i in range(system_charmm.getNumConstraints()):
            self.assertAlmostEqual(system_charmm.getConstraintParameters(i)[2],
                                   system_openmm.getConstraintParameters(i)[2], delta=1e-7 * nanometers)

        system_charmm = psf.createSystem(params, constraints=AllBonds, rigidWater=True, nonbondedMethod=PME,
                                         nonbondedCutoff=8 * angstroms)
        system_openmm = forcefield.createSystem(topology, constraints=AllBonds, rigidWater=True, nonbondedMethod=PME,
                                                nonbondedCutoff=8 * angstroms)
        self.assertEqual(
            sorted(system_charmm.getConstraintParameters(i)[:2] for i in range(system_charmm.getNumConstraints())),
            sorted(system_openmm.getConstraintParameters(j)[:2] for j in range(system_openmm.getNumConstraints())))
        for i in range(system_charmm.getNumConstraints()):
            self.assertAlmostEqual(system_charmm.getConstraintParameters(i)[2],
                                   system_openmm.getConstraintParameters(i)[2], delta=1e-7 * nanometers)

        system_charmm = psf.createSystem(params, constraints=HAngles, rigidWater=False, nonbondedMethod=PME,
                                         nonbondedCutoff=8 * angstroms)
        system_openmm = forcefield.createSystem(topology, constraints=HAngles, rigidWater=False, nonbondedMethod=PME,
                                                nonbondedCutoff=8 * angstroms)
        self.assertEqual(
            sorted(system_charmm.getConstraintParameters(i)[:2] for i in range(system_charmm.getNumConstraints())),
            sorted(system_openmm.getConstraintParameters(j)[:2] for j in range(system_openmm.getNumConstraints())))
        for i in range(system_charmm.getNumConstraints()):
            self.assertAlmostEqual(system_charmm.getConstraintParameters(i)[2],
                                   system_openmm.getConstraintParameters(i)[2], delta=1e-7 * nanometers)

        system_charmm = psf.createSystem(params, constraints=HAngles, rigidWater=True, nonbondedMethod=PME,
                                         nonbondedCutoff=8 * angstroms)
        system_openmm = forcefield.createSystem(topology, constraints=HAngles, rigidWater=True, nonbondedMethod=PME,
                                                nonbondedCutoff=8 * angstroms)
        self.assertEqual(
            sorted(system_charmm.getConstraintParameters(i)[:2] for i in range(system_charmm.getNumConstraints())),
            sorted(system_openmm.getConstraintParameters(j)[:2] for j in range(system_openmm.getNumConstraints())))
        for i in range(system_charmm.getNumConstraints()):
            self.assertAlmostEqual(system_charmm.getConstraintParameters(i)[2],
                                   system_openmm.getConstraintParameters(i)[2], delta=1e-7 * nanometers)

if __name__ == '__main__':
    unittest.main()

