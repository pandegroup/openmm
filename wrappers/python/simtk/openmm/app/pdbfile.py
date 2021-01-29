"""
pdbfile.py: Used for loading PDB files.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2012-2020 Stanford University and the Authors.
Authors: Peter Eastman
Contributors:

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
from __future__ import print_function, division, absolute_import
__author__ = "Peter Eastman"
__version__ = "1.0"

import os
import sys
import math
import xml.etree.ElementTree as etree
from copy import copy
from datetime import date
from simtk.openmm import Vec3, Platform
from simtk.openmm.app.internal.pdbstructure import PdbStructure
from simtk.openmm.app.internal.unitcell import computeLengthsAndAngles
from simtk.openmm.app import Topology
from simtk.unit import nanometers, angstroms, is_quantity, norm, Quantity, dot
from . import element as elem
try:
    import numpy
except ImportError:
    pass

class PDBFile(object):
    """PDBFile parses a Protein Data Bank (PDB) file and constructs a Topology and a set of atom positions from it.

    This class also provides methods for creating PDB files.  To write a file containing a single model, call
    writeFile().  You also can create files that contain multiple models.  To do this, first call writeHeader(),
    then writeModel() once for each model in the file, and finally writeFooter() to complete the file."""

    _residueNameReplacements = {}
    _atomNameReplacements = {}
    _standardResidues = ['ALA', 'ASN', 'CYS', 'GLU', 'HIS', 'LEU', 'MET', 'PRO', 'THR', 'TYR',
                         'ARG', 'ASP', 'GLN', 'GLY', 'ILE', 'LYS', 'PHE', 'SER', 'TRP', 'VAL',
                         'A', 'G', 'C', 'U', 'I', 'DA', 'DG', 'DC', 'DT', 'DI', 'HOH']

    def __init__(self, file, extraParticleIdentifier='EP'):
        """Load a PDB file.

        The atom positions and Topology can be retrieved by calling getPositions() and getTopology().

        Parameters
        ----------
        file : string or file
            the name of the file to load.  Alternatively you can pass an open file object.
        extraParticleIdentifier : string='EP'
            if this value appears in the element column for an ATOM record, the Atom's element will be set to None to mark it as an extra particle
        """
        
        metalElements = ['Al','As','Ba','Ca','Cd','Ce','Co','Cs','Cu','Dy','Fe','Gd','Hg','Ho','In','Ir','K','Li','Mg',
        'Mn','Mo','Na','Ni','Pb','Pd','Pt','Rb','Rh','Sm','Sr','Te','Tl','V','W','Yb','Zn']
        
        top = Topology()
        ## The Topology read from the PDB file
        self.topology = top

        # Load the PDB file

        if isinstance(file, PdbStructure):
            pdb = file
        else:
            inputfile = file
            own_handle = False
            if isinstance(file, str):
                inputfile = open(file)
                own_handle = True
            pdb = PdbStructure(inputfile, load_all_models=True, extraParticleIdentifier=extraParticleIdentifier)
            if own_handle:
                inputfile.close()
        PDBFile._loadNameReplacementTables()

        # Build the topology

        atomByNumber = {}
        for chain in pdb.iter_chains():
            c = top.addChain(chain.chain_id)
            for residue in chain.iter_residues():
                resName = residue.get_name()
                if resName in PDBFile._residueNameReplacements:
                    resName = PDBFile._residueNameReplacements[resName]
                r = top.addResidue(resName, c, str(residue.number), residue.insertion_code)
                if resName in PDBFile._atomNameReplacements:
                    atomReplacements = PDBFile._atomNameReplacements[resName]
                else:
                    atomReplacements = {}
                for atom in residue.iter_atoms():
                    atomName = atom.get_name()
                    if atomName in atomReplacements:
                        atomName = atomReplacements[atomName]
                    atomName = atomName.strip()
                    element = atom.element
                    if element == 'EP':
                        element = None
                    elif element is None:
                        # Try to guess the element.

                        upper = atomName.upper()
                        while len(upper) > 1 and upper[0].isdigit():
                            upper = upper[1:]
                        if upper.startswith('CL'):
                            element = elem.chlorine
                        elif upper.startswith('BR'):
                            element = elem.bromine
                        elif upper.startswith('NA'):
                            element = elem.sodium
                        elif upper.startswith('MG'):
                            element = elem.magnesium
                        elif upper.startswith('BE'):
                            element = elem.beryllium
                        elif upper.startswith('LI'):
                            element = elem.lithium
                        elif upper.startswith('K'):
                            element = elem.potassium
                        elif upper.startswith('ZN'):
                            element = elem.zinc
                        elif len(residue) == 1 and upper.startswith('CA'):
                            element = elem.calcium
                        elif upper.startswith('D') and any(a.name == atomName[1:] for a in residue.iter_atoms()):
                            pass # A Drude particle
                        else:
                            try:
                                element = elem.get_by_symbol(upper[0])
                            except KeyError:
                                pass
                    newAtom = top.addAtom(atomName, element, r, str(atom.serial_number))
                    atomByNumber[atom.serial_number] = newAtom
        self._positions = []
        for model in pdb.iter_models(True):
            coords = []
            for chain in model.iter_chains():
                for residue in chain.iter_residues():
                    for atom in residue.iter_atoms():
                        pos = atom.get_position().value_in_unit(nanometers)
                        coords.append(Vec3(pos[0], pos[1], pos[2]))
            self._positions.append(coords*nanometers)
        ## The atom positions read from the PDB file.  If the file contains multiple frames, these are the positions in the first frame.
        self.positions = self._positions[0]
        self.topology.setPeriodicBoxVectors(pdb.get_periodic_box_vectors())
        self.topology.createStandardBonds()
        self.topology.createDisulfideBonds(self.positions)
        self._numpyPositions = None

        # Add bonds based on CONECT records. Bonds between metals of elements specified in metalElements and residues in standardResidues are not added.

        connectBonds = []
        for connect in pdb.models[-1].connects:
            i = connect[0]
            for j in connect[1:]:
                if i in atomByNumber and j in atomByNumber:    
                    if atomByNumber[i].element is not None and atomByNumber[j].element is not None:
                        if atomByNumber[i].element.symbol not in metalElements and atomByNumber[j].element.symbol not in metalElements:
                            connectBonds.append((atomByNumber[i], atomByNumber[j])) 
                        elif atomByNumber[i].element.symbol in metalElements and atomByNumber[j].residue.name not in PDBFile._standardResidues:
                            connectBonds.append((atomByNumber[i], atomByNumber[j])) 
                        elif atomByNumber[j].element.symbol in metalElements and atomByNumber[i].residue.name not in PDBFile._standardResidues:
                            connectBonds.append((atomByNumber[i], atomByNumber[j]))     
                    else:
                        connectBonds.append((atomByNumber[i], atomByNumber[j]))         
        if len(connectBonds) > 0:
            # Only add bonds that don't already exist.
            existingBonds = set(top.bonds())
            for bond in connectBonds:
                if bond not in existingBonds and (bond[1], bond[0]) not in existingBonds:
                    top.addBond(bond[0], bond[1])
                    existingBonds.add(bond)

    def getTopology(self):
        """Get the Topology of the model."""
        return self.topology

    def getNumFrames(self):
        """Get the number of frames stored in the file."""
        return len(self._positions)

    def getPositions(self, asNumpy=False, frame=0):
        """Get the atomic positions.

        Parameters
        ----------
        asNumpy : boolean=False
            if true, the values are returned as a numpy array instead of a list
            of Vec3s
        frame : int=0
            the index of the frame for which to get positions
        """
        if asNumpy:
            if self._numpyPositions is None:
                self._numpyPositions = [None]*len(self._positions)
            if self._numpyPositions[frame] is None:
                self._numpyPositions[frame] = Quantity(numpy.array(self._positions[frame].value_in_unit(nanometers)), nanometers)
            return self._numpyPositions[frame]
        return self._positions[frame]

    @staticmethod
    def _loadNameReplacementTables():
        """Load the list of atom and residue name replacements."""
        if len(PDBFile._residueNameReplacements) == 0:
            tree = etree.parse(os.path.join(os.path.dirname(__file__), 'data', 'pdbNames.xml'))
            allResidues = {}
            proteinResidues = {}
            nucleicAcidResidues = {}
            for residue in tree.getroot().findall('Residue'):
                name = residue.attrib['name']
                if name == 'All':
                    PDBFile._parseResidueAtoms(residue, allResidues)
                elif name == 'Protein':
                    PDBFile._parseResidueAtoms(residue, proteinResidues)
                elif name == 'Nucleic':
                    PDBFile._parseResidueAtoms(residue, nucleicAcidResidues)
            for atom in allResidues:
                proteinResidues[atom] = allResidues[atom]
                nucleicAcidResidues[atom] = allResidues[atom]
            for residue in tree.getroot().findall('Residue'):
                name = residue.attrib['name']
                for id in residue.attrib:
                    if id == 'name' or id.startswith('alt'):
                        PDBFile._residueNameReplacements[residue.attrib[id]] = name
                if 'type' not in residue.attrib:
                    atoms = copy(allResidues)
                elif residue.attrib['type'] == 'Protein':
                    atoms = copy(proteinResidues)
                elif residue.attrib['type'] == 'Nucleic':
                    atoms = copy(nucleicAcidResidues)
                else:
                    atoms = copy(allResidues)
                PDBFile._parseResidueAtoms(residue, atoms)
                PDBFile._atomNameReplacements[name] = atoms

    @staticmethod
    def _parseResidueAtoms(residue, map):
        for atom in residue.findall('Atom'):
            name = atom.attrib['name']
            for id in atom.attrib:
                map[atom.attrib[id]] = name

    @staticmethod
    def writeFile(topology, positions, file=sys.stdout, keepIds=False, extraParticleIdentifier='EP'):
        """Write a PDB file containing a single model.

        Parameters
        ----------
        topology : Topology
            The Topology defining the model to write
        positions : list
            The list of atomic positions to write
        file : file=stdout
            A file to write to
        keepIds : bool=False
            If True, keep the residue and chain IDs specified in the Topology
            rather than generating new ones.  Warning: It is up to the caller to
            make sure these are valid IDs that satisfy the requirements of the
            PDB format.  Otherwise, the output file will be invalid.
        extraParticleIdentifier : string='EP'
            String to write in the element column of the ATOM records for atoms whose element is None (extra particles)
        """
        PDBFile.writeHeader(topology, file)
        PDBFile.writeModel(topology, positions, file, keepIds=keepIds, extraParticleIdentifier=extraParticleIdentifier)
        PDBFile.writeFooter(topology, file)

    @staticmethod
    def writeHeader(topology, file=sys.stdout):
        """Write out the header for a PDB file.

        Parameters
        ----------
        topology : Topology
            The Topology defining the molecular system being written
        file : file=stdout
            A file to write the file to
        """
        print("REMARK   1 CREATED WITH OPENMM %s, %s" % (Platform.getOpenMMVersion(), str(date.today())), file=file)
        vectors = topology.getPeriodicBoxVectors()
        if vectors is not None:
            a, b, c, alpha, beta, gamma = computeLengthsAndAngles(vectors)
            RAD_TO_DEG = 180/math.pi
            print("CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1 " % (
                    a*10, b*10, c*10, alpha*RAD_TO_DEG, beta*RAD_TO_DEG, gamma*RAD_TO_DEG), file=file)

    @staticmethod
    def writeModel(topology, positions, file=sys.stdout, modelIndex=None, keepIds=False, extraParticleIdentifier='EP'):
        """Write out a model to a PDB file.

        Parameters
        ----------
        topology : Topology
            The Topology defining the model to write
        positions : list
            The list of atomic positions to write
        file : file=stdout
            A file to write the model to
        modelIndex : int=None
            If not None, the model will be surrounded by MODEL/ENDMDL records
            with this index
        keepIds : bool=False
            If True, keep the residue and chain IDs specified in the Topology
            rather than generating new ones.  Warning: It is up to the caller to
            make sure these are valid IDs that satisfy the requirements of the
            PDB format.  No guarantees are made about what will happen if they
            are not, and the output file could be invalid.
        extraParticleIdentifier : string='EP'
            String to write in the element column of the ATOM records for atoms whose element is None (extra particles)
        """

        if len(list(topology.atoms())) != len(positions):
            raise ValueError('The number of positions must match the number of atoms')
        if is_quantity(positions):
            positions = positions.value_in_unit(angstroms)
        if any(math.isnan(norm(pos)) for pos in positions):
            raise ValueError('Particle position is NaN')
        if any(math.isinf(norm(pos)) for pos in positions):
            raise ValueError('Particle position is infinite')
        nonHeterogens = PDBFile._standardResidues[:]
        nonHeterogens.remove('HOH')
        atomIndex = 1
        posIndex = 0
        if modelIndex is not None:
            print("MODEL     %4d" % modelIndex, file=file)
        for (chainIndex, chain) in enumerate(topology.chains()):
            if keepIds and len(chain.id) == 1:
                chainName = chain.id
            else:
                chainName = chr(ord('A')+chainIndex%26)
            residues = list(chain.residues())
            for (resIndex, res) in enumerate(residues):
                if len(res.name) > 3:
                    resName = res.name[:3]
                else:
                    resName = res.name
                if keepIds and len(res.id) < 5:
                    resId = res.id
                else:
                    resId = "%4d" % ((resIndex+1)%10000)
                if len(res.insertionCode) == 1:
                    resIC = res.insertionCode
                else:
                    resIC = " "
                if res.name in nonHeterogens:
                    recordName = "ATOM  "
                else:
                    recordName = "HETATM"
                for atom in res.atoms():
                    if atom.element is not None:
                        symbol = atom.element.symbol
                    else:
                        symbol = extraParticleIdentifier
                    if len(atom.name) < 4 and atom.name[:1].isalpha() and len(symbol) < 2:
                        atomName = ' '+atom.name
                    elif len(atom.name) > 4:
                        atomName = atom.name[:4]
                    else:
                        atomName = atom.name
                    coords = positions[posIndex]
                    line = "%s%5d %-4s %3s %s%4s%1s   %s%s%s  1.00  0.00          %2s  " % (
                        recordName, atomIndex%100000, atomName, resName, chainName, resId, resIC, _format_83(coords[0]),
                        _format_83(coords[1]), _format_83(coords[2]), symbol)
                    if len(line) != 80:
                        raise ValueError('Fixed width overflow detected')
                    print(line, file=file)
                    posIndex += 1
                    atomIndex += 1
                if resIndex == len(residues)-1:
                    print("TER   %5d      %3s %s%4s" % (atomIndex, resName, chainName, resId), file=file)
                    atomIndex += 1
        if modelIndex is not None:
            print("ENDMDL", file=file)

    @staticmethod
    def writeFooter(topology, file=sys.stdout):
        """Write out the footer for a PDB file.

        Parameters
        ----------
        topology : Topology
            The Topology defining the molecular system being written
        file : file=stdout
            A file to write the file to
        """
        # Identify bonds that should be listed as CONECT records.

        conectBonds = []
        for atom1, atom2 in topology.bonds():
            if atom1.residue.name not in PDBFile._standardResidues or atom2.residue.name not in PDBFile._standardResidues:
                conectBonds.append((atom1, atom2))
            elif atom1.name == 'SG' and atom2.name == 'SG' and atom1.residue.name == 'CYS' and atom2.residue.name == 'CYS':
                conectBonds.append((atom1, atom2))
        if len(conectBonds) > 0:

            # Work out the index used in the PDB file for each atom.

            atomIndex = {}
            nextAtomIndex = 0
            prevChain = None
            for chain in topology.chains():
                for atom in chain.atoms():
                    if atom.residue.chain != prevChain:
                        nextAtomIndex += 1
                        prevChain = atom.residue.chain
                    atomIndex[atom] = nextAtomIndex
                    nextAtomIndex += 1

            # Record which other atoms each atom is bonded to.

            atomBonds = {}
            for atom1, atom2 in conectBonds:
                index1 = atomIndex[atom1]
                index2 = atomIndex[atom2]
                if index1 not in atomBonds:
                    atomBonds[index1] = []
                if index2 not in atomBonds:
                    atomBonds[index2] = []
                atomBonds[index1].append(index2)
                atomBonds[index2].append(index1)

            # Write the CONECT records.

            for index1 in sorted(atomBonds):
                bonded = atomBonds[index1]
                while len(bonded) > 4:
                    print("CONECT%5d%5d%5d%5d" % (index1, bonded[0], bonded[1], bonded[2]), file=file)
                    del bonded[:4]
                line = "CONECT%5d" % index1
                for index2 in bonded:
                    line = "%s%5d" % (line, index2)
                print(line, file=file)
        print("END", file=file)


def _format_83(f):
    """Format a single float into a string of width 8, with ideally 3 decimal
    places of precision. If the number is a little too large, we can
    gracefully degrade the precision by lopping off some of the decimal
    places. If it's much too large, we throw a ValueError"""
    if -999.999 < f < 9999.999:
        return '%8.3f' % f
    if -9999999 < f < 99999999:
        return ('%8.3f' % f)[:8]
    raise ValueError('coordinate "%s" could not be represented '
                     'in a width-8 field' % f)
