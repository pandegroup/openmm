"""
Provides a Python class for parsing a PSF file and setting up a system
structure for it within the OpenMM framework.

This file is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of Biological
Structures at Stanford, funded under the NIH Roadmap for Medical Research,
grant U54 GM072970. See https://simtk.org.  This code was originally part of
the ParmEd program and was ported for use with OpenMM.

Copyright (c) 2014-2020 the Authors

Author: Jason M. Swails
Contributors: Jing Huang

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
from __future__ import division, absolute_import, print_function

from functools import wraps
from math import pi, cos, sin, sqrt
import os
import re
import sys
import simtk.openmm as mm
from simtk.openmm.vec3 import Vec3
import simtk.unit as u
from simtk.openmm.app import (forcefield as ff, Topology, element, PDBFile)
from simtk.openmm.app.amberprmtopfile import HCT, OBC1, OBC2, GBn, GBn2
from simtk.openmm.app.internal.customgbforces import (GBSAHCTForce,
                GBSAOBC1Force, GBSAOBC2Force, GBSAGBnForce, GBSAGBn2Force)
from simtk.openmm.app.internal.unitcell import computePeriodicBoxVectors
# CHARMM imports
from simtk.openmm.app.internal.charmm.topologyobjects import (
                ResidueList, AtomList, TrackedList, Bond, Angle, Dihedral,
                Improper, AcceptorDonor, Group, Cmap, UreyBradley,
                NoUreyBradley)
from simtk.openmm.app.internal.charmm.exceptions import (
                CharmmPSFError, MoleculeError, CharmmPSFWarning,
                MissingParameter, CharmmPsfEOF)
import warnings

TINY = 1e-8
WATNAMES = ('WAT', 'HOH', 'TIP3', 'TIP4', 'TIP5', 'SPCE', 'SPC', 'SWM4', 'SWM6')
if sys.version_info >= (3, 0):
    xrange = range

def _catchindexerror(func):
    """
    Protects a function from raising an index error, and replace that exception
    with a CharmmPSFError instead
    """
    @wraps(func)
    def newfunc(*args, **kwargs):
        """ Catch the index error """
        try:
            return func(*args, **kwargs)
        except IndexError as e:
            raise CharmmPSFError('Array is too short: %s' % e)

    return newfunc

class _ZeroDict(dict):
    """
    Contains a dict that returns dummy (zero) arguments when a key is not
    present rather than raising a KeyError.  The return value for non-existent
    items is (0, []). It also special-case sections that have multiple pointers
    to avoid index errors if those are not present in the PSF file
    """
    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            if key.startswith('NGRP'):
                for k in self:
                    if k.startswith('NGRP'):
                        return dict.__getitem__(self, k)
                return [0, 0], []
            elif key.startswith('NUMLP'):
                for k in self:
                    if k.startswith('NUMLP'):
                        return dict.__getitem__(self, k)
                return [0, 0], []
            return 0, []

def _strip_optunit(thing, unit):
    """
    Strips optional units, converting to specified unit type. If no unit
    present, it just returns the number
    """
    if u.is_quantity(thing):
        return thing.value_in_unit(unit)
    return thing

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

_resre = re.compile(r'(-?\d+)([a-zA-Z]*)')

class CharmmPsfFile(object):
    """A chemical structure instantiated from CHARMM files.

    This structure has numerous attributes that are lists of the elements of
    this structure, including atoms, bonds, torsions, etc. The attributes are

    - residue_list
    - atom_list
    - bond_list
    - angle_list
    - dihedral_list
    - dihedral_parameter_list
    - improper_list
    - cmap_list
    - donor_list    # hbonds donors?
    - acceptor_list # hbond acceptors?
    - group_list    # list of nonbonded interaction groups

    Four additional lists for Drude psf:
    - drudeconsts_list
    - drudepair_list
    - lonepair_list
    - aniso_list

    Additional attribute is available if a CharmmParameterSet is loaded into
    this structure.

    - urey_bradley_list

    The lengths of each of these lists gives the pointers (e.g., natom, nres,
    etc.)

    Examples
    --------
    >>> cs = CharmmPsfFile("testfiles/test.psf")
    >>> len(cs.atom_list)
    33
    >>> len(cs.bond_list)
    32
    """
    # Define default force groups for all of the bonded terms. This allows them
    # to be turned on and off selectively. This is a way to implement per-term
    # energy decomposition to compare individual components

    BOND_FORCE_GROUP = 0
    ANGLE_FORCE_GROUP = 1
    DIHEDRAL_FORCE_GROUP = 2
    UREY_BRADLEY_FORCE_GROUP = 3
    IMPROPER_FORCE_GROUP = 4
    CMAP_FORCE_GROUP = 5
    NONBONDED_FORCE_GROUP = 6
    GB_FORCE_GROUP = 6
    DRUDE_FORCE_GROUP = 7

    @_catchindexerror
    def __init__(self, psf_name, periodicBoxVectors=None, unitCellDimensions=None):
        """Opens and parses a PSF file, then instantiates a CharmmPsfFile
        instance from the data.

        Parameters
        ----------
        psf_name : str
            Name of the PSF file (it must exist)
        periodicBoxVectors : tuple of Vec3
            the vectors defining the periodic box
        unitCellDimensions : Vec3
            the dimensions of the crystallographic unit cell.  For
            non-rectangular unit cells, specify periodicBoxVectors instead.

        Raises
        ------
        IOError : If file "psf_name" does not exist
        CharmmPSFError: If any parsing errors are encountered
        """
        conv = CharmmPsfFile._convert
        # Make sure the file exists
        if not os.path.exists(psf_name):
            raise IOError('Could not find PSF file %s' % psf_name)
        # Open the PSF and read the first line. It must start with "PSF"
        with open(psf_name, 'r') as psf:
            line = psf.readline()
            if not line.startswith('PSF'):
                raise CharmmPSFError('Unrecognized PSF file. First line is %s' %
                                     line.strip())
            # Store the flags
            psf_flags = line.split()[1:]
            # Determine whether it's a Drude polarizable system
            IsDrudePSF = 'DRUDE' in psf_flags
            # Now get all of the sections and store them in a dict
            psf.readline()
            # Now get all of the sections
            psfsections = _ZeroDict()
            while True:
                try:
                    sec, ptr, data = CharmmPsfFile._parse_psf_section(psf)
                except CharmmPsfEOF:
                    break
                psfsections[sec] = (ptr, data)
        # store the title
        title = psfsections['NTITLE'][1]
        # Next is the number of atoms
        natom = conv(psfsections['NATOM'][0], int, 'natom')
        # Parse all of the atoms
        residue_list = ResidueList()
        atom_list = AtomList()
        if IsDrudePSF:
            drudeconsts_list = TrackedList()
        PDBFile._loadNameReplacementTables()
        for i in xrange(natom):
            words = psfsections['NATOM'][1][i].split()
            system = words[1]
            rematch = _resre.match(words[2])
            if not rematch:
                raise RuntimeError('Could not parse residue number %s' %
                                   words[2])
            resid, inscode = rematch.groups()
            resid = int(resid)
            resname = words[3]
            name = words[4]
            attype = words[5]
            # Try to convert the atom type to an integer a la CHARMM
            try:
                attype = int(attype)
            except ValueError:
                pass
            charge = conv(words[6], float, 'partial charge')
            mass = conv(words[7], float, 'atomic mass')
            props = words[8:]
            if resname in PDBFile._residueNameReplacements:
                resname = PDBFile._residueNameReplacements[resname]
            if resname in PDBFile._atomNameReplacements:
                atomReplacements = PDBFile._atomNameReplacements[resname]
                if name in atomReplacements:
                    name = atomReplacements[name]
            atom = residue_list.add_atom(system, resid, resname, name,
                            attype, charge, mass, inscode, props)
            atom_list.append(atom)
            if IsDrudePSF:
                alpha = conv(words[9], float, 'alpha')
                thole = conv(words[10], float, 'thole')
                drudeconsts_list.append([alpha, thole])
        atom_list.assign_indexes()
        atom_list.changed = False
        # Now get the number of bonds
        nbond = conv(psfsections['NBOND'][0], int, 'number of bonds')
        holder = psfsections['NBOND'][1]
        bond_list = TrackedList()
        if IsDrudePSF:
            drudepair_list = TrackedList()
        if len(holder) != nbond * 2:
            raise CharmmPSFError('Got %d indexes for %d bonds' %
                                 (len(holder), nbond))
        for i in range(nbond):
            id1 = holder[2*i  ] - 1
            id2 = holder[2*i+1] - 1
            # ignore any bond pair involving Drude or lonepairs: possible using atom's prop
            if (atom_list[id1].name[0]=='D' or atom_list[id2].name[0]=='D'):
                drudepair_list.append([min(id1,id2), max(id1,id2)])
            elif (atom_list[id1].name[0:2]=='LP' or atom_list[id2].name[0:2]=='LP' or atom_list[id1].name=='OM' or atom_list[id2].name=='OM'):
                pass
            # Ignore H-H bond in water if present
            elif atom_list[id1].name[0]=='H' and atom_list[id2].name[0]=='H' and (atom_list[id1].residue.resname in WATNAMES):
                pass
            else:
                bond_list.append(Bond(atom_list[id1], atom_list[id2]))
        bond_list.changed = False
        # Now get the number of angles and the angle list
        ntheta = conv(psfsections['NTHETA'][0], int, 'number of angles')
        holder = psfsections['NTHETA'][1]
        angle_list = TrackedList()
        if len(holder) != ntheta * 3:
            raise CharmmPSFError('Got %d indexes for %d angles' %
                                 (len(holder), ntheta))
        for i in range(ntheta):
            id1 = holder[3*i  ] - 1
            id2 = holder[3*i+1] - 1
            id3 = holder[3*i+2] - 1
            angle_list.append(
                    Angle(atom_list[id1], atom_list[id2], atom_list[id3])
            )
        angle_list.changed = False
        # Now get the number of torsions and the torsion list
        nphi = conv(psfsections['NPHI'][0], int, 'number of torsions')
        holder = psfsections['NPHI'][1]
        dihedral_list = TrackedList()
        if len(holder) != nphi * 4:
            raise CharmmPSFError('Got %d indexes for %d torsions' %
                                 (len(holder), nphi))
        for i in range(nphi):
            id1 = holder[4*i  ] - 1
            id2 = holder[4*i+1] - 1
            id3 = holder[4*i+2] - 1
            id4 = holder[4*i+3] - 1
            dihedral_list.append(
                    Dihedral(atom_list[id1], atom_list[id2], atom_list[id3],
                             atom_list[id4])
            )
        dihedral_list.changed = False
        # Now get the number of improper torsions
        nimphi = conv(psfsections['NIMPHI'][0], int, 'number of impropers')
        holder = psfsections['NIMPHI'][1]
        improper_list = TrackedList()
        if len(holder) != nimphi * 4:
            raise CharmmPSFError('Got %d indexes for %d impropers' %
                                 (len(holder), nimphi))
        for i in range(nimphi):
            id1 = holder[4*i  ] - 1
            id2 = holder[4*i+1] - 1
            id3 = holder[4*i+2] - 1
            id4 = holder[4*i+3] - 1
            improper_list.append(
                    Improper(atom_list[id1], atom_list[id2], atom_list[id3],
                             atom_list[id4])
            )
        improper_list.changed = False
        # Now handle the donors (what is this used for??)
        ndon = conv(psfsections['NDON'][0], int, 'number of donors')
        holder = psfsections['NDON'][1]
        donor_list = TrackedList()
        if len(holder) != ndon * 2:
            raise CharmmPSFError('Got %d indexes for %d donors' %
                                 (len(holder), ndon))
        for i in range(ndon):
            id1 = holder[2*i  ] - 1
            id2 = holder[2*i+1] - 1
            donor_list.append(AcceptorDonor(atom_list[id1], atom_list[id2]))
        donor_list.changed = False
        # Now handle the acceptors (what is this used for??)
        nacc = conv(psfsections['NACC'][0], int, 'number of acceptors')
        holder = psfsections['NACC'][1]
        acceptor_list = TrackedList()
        if len(holder) != nacc * 2:
            raise CharmmPSFError('Got %d indexes for %d acceptors' %
                                 (len(holder), ndon))
        for i in range(nacc):
            id1 = holder[2*i  ] - 1
            id2 = holder[2*i+1] - 1
            acceptor_list.append(AcceptorDonor(atom_list[id1], atom_list[id2]))
        acceptor_list.changed = False
        # Now get the group sections
        group_list = TrackedList()
        try:
            ngrp, nst2 = psfsections['NGRP NST2'][0]
        except ValueError:
            raise CharmmPSFError('Could not unpack GROUP pointers')
        holder = psfsections['NGRP NST2'][1]
        group_list.nst2 = nst2
        # Now handle the groups
        if len(holder) != ngrp * 3:
            raise CharmmPSFError('Got %d indexes for %d groups' %
                                 (len(holder), ngrp))
        for i in range(ngrp):
            i1 = holder[3*i  ]
            i2 = holder[3*i+1]
            i3 = holder[3*i+2]
            group_list.append(Group(i1, i2, i3))
        group_list.changed = False
        # Assign all of the atoms to molecules recursively
        holder = psfsections['MOLNT'][1]
        set_molecules(atom_list)
        molecule_list = [atom.marked for atom in atom_list]
        if len(holder) == len(atom_list):
            if len(molecule_list) != len(holder):
                # The MOLNT section is only used for fluctuating charge models,
                # which are currently not supported anyway.
                # Therefore, we only check the lengths of the lists now rather than their contents.
                warnings.warn('Detected PSF molecule section that is WRONG. '
                              'Resetting molecularity.', CharmmPSFWarning)
        # We have a CHARMM PSF file; now do NUMLP/NUMLPH sections
        numlp, numlph = psfsections['NUMLP NUMLPH'][0]
        holder = psfsections['NUMLP NUMLPH'][1]
        lonepair_list = TrackedList()
        if numlp != 0 or numlph != 0:
            lp_hostnum_list=[]
            lp_distance_list=[]
            lp_angle_list=[]
            lp_dihe_list=[]
            for i in range(numlp):
                lpline = holder[i].split()
                if len(lpline)!=6 or lpline[2] != 'F' :
                    raise CharmmPSFError('Lonepair format error')
                else:
                    lp_hostnum_list.append(int(lpline[0]))
                    lp_distance_list.append(float(lpline[3]))
                    lp_angle_list.append(float(lpline[4]))
                    lp_dihe_list.append(float(lpline[5]))
            lp_atom_counter=0
            for i in range(numlp):
                idall=[]
                for j in range(lp_hostnum_list[i]+1):
                    iline = int((lp_atom_counter+j)/8)+numlp
                    icolumn = (lp_atom_counter+j)%8
                    idall.append(int(holder[iline].split()[icolumn])-1)
                if len(idall)==3:
                    idall.append(-1) # use id4=-1 to mark colinear
                lonepair_list.append([idall[0], idall[1], idall[2], idall[3], lp_distance_list[i], lp_angle_list[i], lp_dihe_list[i]])
                lp_atom_counter += lp_hostnum_list[i]+1
        # In Drude psf, here comes anisotropic section
        if IsDrudePSF:
            numaniso = psfsections['NUMANISO'][0]
            holder = psfsections['NUMANISO'][1]
            aniso_list = TrackedList()
            if numaniso != 0:
               k_list=[]
               for i in range(numaniso):
                   lpline = holder[i].split()
                   k_list.append([float(lpline[0]),float(lpline[1]),float(lpline[2])])
               for i in range(numaniso):
                    lpline = holder[int(i/2)+numaniso].split()
                    icolumn = (i%2) * 4
                    id1=int(lpline[icolumn])  -1
                    id2=int(lpline[icolumn+1])-1
                    id3=int(lpline[icolumn+2])-1
                    id4=int(lpline[icolumn+3])-1
                    aniso_list.append([id1, id2, id3, id4, k_list[i][0], k_list[i][1], k_list[i][2]])
        # Now do the CMAPs
        ncrterm = conv(psfsections['NCRTERM'][0], int, 'Number of cross-terms')
        holder = psfsections['NCRTERM'][1]
        cmap_list = TrackedList()
        if len(holder) != ncrterm * 8:
            raise CharmmPSFError('Got %d CMAP indexes for %d cmap terms' %
                                 (len(holder), ncrterm))
        for i in range(ncrterm):
            id1 = holder[8*i  ] - 1
            id2 = holder[8*i+1] - 1
            id3 = holder[8*i+2] - 1
            id4 = holder[8*i+3] - 1
            id5 = holder[8*i+4] - 1
            id6 = holder[8*i+5] - 1
            id7 = holder[8*i+6] - 1
            id8 = holder[8*i+7] - 1
            cmap_list.append(
                    Cmap(atom_list[id1], atom_list[id2], atom_list[id3],
                         atom_list[id4], atom_list[id5], atom_list[id6],
                         atom_list[id7], atom_list[id8])
            )
        cmap_list.changed = False

        self.residue_list = residue_list
        self.atom_list = atom_list
        self.bond_list = bond_list
        self.angle_list = angle_list
        self.dihedral_list = dihedral_list
        self.dihedral_parameter_list = TrackedList()
        self.improper_list = improper_list
        self.lonepair_list = lonepair_list
        if IsDrudePSF:
            self.drudeconsts_list = drudeconsts_list
            self.drudepair_list = drudepair_list
            self.aniso_list = aniso_list
        self.cmap_list = cmap_list
        self.donor_list = donor_list
        self.acceptor_list = acceptor_list
        self.group_list = group_list
        self.title = title
        self.flags = psf_flags
        if unitCellDimensions is not None:
            if periodicBoxVectors is not None:
                raise ValueError("specify either periodicBoxVectors or unitCellDimensions, but not both")
            if u.is_quantity(unitCellDimensions):
                unitCellDimensions = unitCellDimensions.value_in_unit(u.nanometers)
            self.box_vectors = (Vec3(unitCellDimensions[0], 0, 0), Vec3(0, unitCellDimensions[1], 0), Vec3(0, 0, unitCellDimensions[2]))*u.nanometers
        else:
            self.box_vectors = periodicBoxVectors

    def _build_exclusion_list(self):
        pair_12_set = set()
        pair_13_set = set()
        pair_14_set = set()
        for bond in self.bond_list:
            a1, a2 = bond.atom1, bond.atom2
            pair = (min(a1.idx, a2.idx), max(a1.idx, a2.idx),)
            pair_12_set.add(pair)
        for bond in self.bond_list:
            a2, a3 = bond.atom1, bond.atom2
            for a1 in a2.bond_partners:
                pair = (min(a1.idx, a3.idx), max(a1.idx, a3.idx),)
                if a1 != a3:
                    pair_13_set.add(pair)
            for a4 in a3.bond_partners:
                pair = (min(a2.idx, a4.idx), max(a2.idx, a4.idx),)
                if a2 != a4:
                    pair_13_set.add(pair)
        for bond in self.bond_list:
            a2, a3 = bond.atom1, bond.atom2
            for a1 in a2.bond_partners:
                for a4 in a3.bond_partners:
                    pair = (min(a1.idx, a4.idx), max(a1.idx, a4.idx),)
                    if a1 != a3 and a2 != a4 and a1 != a4:
                        pair_14_set.add(pair)

        # in case there are 3,4,5-member rings
        self.pair_12_list = list(sorted(pair_12_set))
        self.pair_13_list = list(sorted(pair_13_set - pair_12_set))
        self.pair_14_list = list(sorted(pair_14_set - pair_13_set.union(pair_12_set)))

    @staticmethod
    def _convert(string, type, message):
        """Converts a string to a specific type, making sure to raise
        CharmmPSFError with the given message in the event of a failure.

        Parameters
        ----------
        string : str
            Input string to process
        type : type
            Type of data to convert to
        message : str
            Error message to put in exception if failed
        """
        try:
            return type(string)
        except ValueError as e:
            print(e)
            raise CharmmPSFError('Could not convert %s' % message)

    @staticmethod
    def _parse_psf_section(psf):
        """This method parses a section of the PSF file

        Parameters
        ----------
         psf : CharmmFile
             Open file that is pointing to the first line of the section
             that is to be parsed

        Returns
        --------
        str
            The label of the PSF section we are parsing
        int/tuple of ints
            If one pointer is set, pointers is simply the integer that is
            value of that pointer. Otherwise it is a tuple with every pointer
            value defined in the first line
        list
            A list of all data in the parsed section converted to `dtype'
        """
        conv = CharmmPsfFile._convert
        line = psf.readline()
        while not line.strip():
            if not line:
                raise CharmmPsfEOF('Unexpected EOF in PSF file')
            else:
                line = psf.readline()
        if '!' in line:
            words = line[:line.index('!')].split()
            title = line[line.index('!')+1:].strip().upper()
            # Strip out description
            if ':' in title:
                title = title[:title.index(':')]
        else:
            raise CharmmPSFError('Could not determine section title')
        if len(words) == 1:
            pointers = conv(words[0], int, 'pointer')
        else:
            pointers = tuple([conv(w, int, 'pointer') for w in words])
        line = psf.readline().strip()
        if not line and title.startswith('NNB'):
            # This will correctly handle the NNB section (which has a spurious
            # blank line) as well as any sections that have 0 members.
            line = psf.readline().strip()
        data = []
        if title == 'NATOM' or title == 'NTITLE' or title == 'NUMLP NUMLPH' or title == 'NUMANISO':
            # Store these four sections as strings (ATOM section we will parse
            # later). The rest of the sections are integer pointers
            while line:
                data.append(line)
                line = psf.readline().strip()
        else:
            while line:
                words = line.split()
                data.extend([conv(w, int, 'PSF data') for w in words])
                line = psf.readline().strip()
        return title, pointers, data

    def loadParameters(self, parmset):
        """Loads parameters from a parameter set that was loaded via CHARMM RTF,
        PAR, and STR files.

        Parameters
        ----------
        parmset : CharmmParameterSet
            List of all parameters

        Notes
        -----
        - If any parameters that are necessary cannot be found, a
          MissingParameter exception is raised.
        - If any dihedral or improper parameters cannot be found, I will try
          inserting wildcards (at either end for dihedrals and as the two
          central atoms in impropers) and see if that matches.  Wild-cards
          will apply ONLY if specific parameters cannot be found.
        - This method will expand the dihedral_parameter_list attribute by
          adding a separate Dihedral object for each term for types that
          have a multi-term expansion
        """
        # First load the atom types
        types_are_int = False
        for atom in self.atom_list:
            try:
                if isinstance(atom.attype, int):
                    atype = parmset.atom_types_int[atom.attype]
                    types_are_int = True # if we have to change back
                else:
                    atype = parmset.atom_types_str[atom.attype]
            except KeyError:
                raise MissingParameter('Could not find atom type for %s' %
                                       atom.attype)
            atom.type = atype
            # Change to string attype to look up the rest of the parameters
            atom.type_to_str()

        # Next load all of the bonds
        for bond in self.bond_list:
            # Construct the key
            key = (min(bond.atom1.attype, bond.atom2.attype),
                   max(bond.atom1.attype, bond.atom2.attype))
            try:
                bond.bond_type = parmset.bond_types[key]
            except KeyError:
                raise MissingParameter('Missing bond type for %r' % bond)
        # Next load all of the angles. If a Urey-Bradley term is defined for
        # this angle, also build the urey_bradley and urey_bradley_type lists
        self.urey_bradley_list = TrackedList()
        for ang in self.angle_list:
            # Construct the key
            key = (min(ang.atom1.attype, ang.atom3.attype), ang.atom2.attype,
                   max(ang.atom1.attype, ang.atom3.attype))
            try:
                ang.angle_type = parmset.angle_types[key]
                ubt = parmset.urey_bradley_types[key]
                if ubt is not NoUreyBradley:
                    ub = UreyBradley(ang.atom1, ang.atom3, ubt)
                    self.urey_bradley_list.append(ub)
            except KeyError:
                raise MissingParameter('Missing angle type for %r' % ang)
        # Next load all of the dihedrals.
        self.dihedral_parameter_list = TrackedList()
        for dih in self.dihedral_list:
            # Store the atoms
            a1, a2, a3, a4 = dih.atom1, dih.atom2, dih.atom3, dih.atom4
            at1, at2, at3, at4 = a1.attype, a2.attype, a3.attype, a4.attype
            # First see if the exact dihedral is specified
            key = min((at1,at2,at3,at4), (at4,at3,at2,at1))
            if not key in parmset.dihedral_types:
                # Check for wild-cards
                key = min(('X',at2,at3,'X'), ('X',at3,at2,'X'))
                if not key in parmset.dihedral_types:
                    raise MissingParameter('No dihedral parameters found for '
                                           '%r' % dih)
            dtlist = parmset.dihedral_types[key]
            for i, dt in enumerate(dtlist):
                self.dihedral_parameter_list.append(Dihedral(a1,a2,a3,a4,dt))
                # See if we include the end-group interactions for this
                # dihedral. We do IFF it is the last or only dihedral term and
                # it is NOT in the angle/bond partners
                if i != len(dtlist) - 1:
                    self.dihedral_parameter_list[-1].end_groups_active = False
                elif a1 in a4.bond_partners or a1 in a4.angle_partners:
                    self.dihedral_parameter_list[-1].end_groups_active = False
        # Now do the impropers
        for imp in self.improper_list:
            # Store the atoms
            a1, a2, a3, a4 = imp.atom1, imp.atom2, imp.atom3, imp.atom4
            at1, at2, at3, at4 = a1.attype, a2.attype, a3.attype, a4.attype
            key = tuple(sorted([at1, at2, at3, at4]))
            if not key in parmset.improper_types:
                # Check for wild-cards
                for anchor in (at2, at3, at4):
                    key = tuple(sorted([at1, anchor, 'X', 'X']))
                    if key in parmset.improper_types:
                        break # This is the right key
            try:
                imp.improper_type = parmset.improper_types[key]
            except KeyError:
                raise MissingParameter('No improper parameters found for %r' %
                                       imp)
        # Now do the cmaps. These will not have wild-cards
        for cmap in self.cmap_list:
            # Store the atoms for easy reference
            if cmap.consecutive:
                a1, a2, a3, a4 = cmap.atom1, cmap.atom2, cmap.atom3, cmap.atom4
                a5, a6, a7, a8 = cmap.atom2, cmap.atom3, cmap.atom4, cmap.atom5
            else:
                a1, a2, a3, a4 = cmap.atom1, cmap.atom2, cmap.atom3, cmap.atom4
                a5, a6, a7, a8 = cmap.atom5, cmap.atom6, cmap.atom7, cmap.atom8
            at1, at2, at3, at4 = a1.attype, a2.attype, a3.attype, a4.attype
            at5, at6, at7, at8 = a5.attype, a6.attype, a7.attype, a8.attype
            # Construct the keys
            k1 = list(min((at1,at2,at3,at4), (at4,at3,at2,at1)))
            k2 = list(min((at5,at6,at7,at8), (at8,at7,at6,at5)))
            key = tuple(k1 + k2)
            try:
                cmap.cmap_type = parmset.cmap_types[key]
            except KeyError:
                raise MissingParameter('No CMAP parameters found for %r' % cmap)
        # If the types started out as integers, change them back
        if types_are_int:
            for atom in self.atom_list: atom.type_to_int()

    def setBox(self, a, b, c, alpha=90.0*u.degrees, beta=90.0*u.degrees,
               gamma=90.0*u.degrees):
        """Sets the periodic box boundary conditions.

        Parameters
        ----------
        a : length
            Lengths of the periodic cell
        b : length
            Lengths of the periodic cell
        c : length
            Lengths of the periodic cell
        alpha : floats, optional
            Angles between the periodic cell vectors.
        beta : floats, optional
            Angles between the periodic cell vectors.
        gamma : floats, optional
            Angles between the periodic cell vectors.
        """
        try:
            # Since we are setting the box, delete the cached box lengths if we
            # have them to make sure they are recomputed if desired.
            del self._boxLengths
        except AttributeError:
            pass
        self.box_vectors = computePeriodicBoxVectors(a, b, c, alpha, beta, gamma)
        # If we already have a _topology instance, then we have possibly changed
        # the existence of box information (whether or not this is a periodic
        # system), so delete any cached reference to a topology so it's
        # regenerated with updated information
        if hasattr(self, '_topology'):
            del self._topology

    @property
    def topology(self):
        """ Create an OpenMM Topology object from the stored bonded network """
        try:
            return self._topology
        except AttributeError:
            # If none exists, we need to create it
            pass
        # Cache the topology for easy returning later
        self._topology = topology = Topology()

        last_chain = None
        last_residue = None
        # Add each chain (separate 'system's) and residue
        for atom in self.atom_list:
            resid = '%d%s' % (atom.residue.idx, atom.residue.inscode)
            if atom.system != last_chain:
                chain = topology.addChain(atom.system)
                last_chain = atom.system
                last_residue = None
            if resid != last_residue:
                last_residue = resid
                residue = topology.addResidue(atom.residue.resname, chain, str(atom.residue.idx), atom.residue.inscode)
            if atom.type is not None:
                # This is the most reliable way of determining the element
                atomic_num = atom.type.atomic_number
                if atomic_num != 0:
                    elem = element.Element.getByAtomicNumber(atomic_num)
            else:
                # Figure it out from the mass
                elem = element.Element.getByMass(atom.mass)
            topology.addAtom(atom.name, elem, residue)

        # Add all of the bonds
        atoms = list(topology.atoms())
        # Assign atom indexes to make sure they're current
        self.atom_list.assign_indexes()
        for bond in self.bond_list:
            topology.addBond(atoms[bond.atom1.idx], atoms[bond.atom2.idx])

        # Add the periodic box if there is one
        if self.box_vectors is not None:
            topology.setUnitCellDimensions(self.boxLengths)

        return topology

    def createSystem(self, params, nonbondedMethod=ff.NoCutoff,
                     nonbondedCutoff=1.0*u.nanometer,
                     switchDistance=0.0*u.nanometer,
                     constraints=None,
                     rigidWater=True,
                     implicitSolvent=None,
                     implicitSolventKappa=None,
                     implicitSolventSaltConc=0.0*u.moles/u.liter,
                     temperature=298.15*u.kelvin,
                     soluteDielectric=1.0,
                     solventDielectric=78.5,
                     removeCMMotion=True,
                     hydrogenMass=None,
                     ewaldErrorTolerance=0.0005,
                     flexibleConstraints=True,
                     verbose=False,
                     gbsaModel=None,
                     drudeMass=0.4*u.amu):
        """Construct an OpenMM System representing the topology described by the
        prmtop file. You MUST have loaded a parameter set into this PSF before
        calling createSystem. If not, AttributeError will be raised. ValueError
        is raised for illegal input.

        Parameters
        ----------
        params : CharmmParameterSet
            The parameter set to use to parametrize this molecule
        nonbondedMethod : object=NoCutoff
            The method to use for nonbonded interactions. Allowed values are
            NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, PME, or LJPME.
        nonbondedCutoff : distance=1*nanometer
            The cutoff distance to use for nonbonded interactions.
        switchDistance : distance=0*nanometer
            The distance at which the switching function is active for nonbonded
            interactions. If the switchDistance evaluates to boolean False (if
            it is 0), no switching function will be used. Illegal values will
            raise a ValueError
        constraints : object=None
            Specifies which bonds or angles should be implemented with
            constraints. Allowed values are None, HBonds, AllBonds, or HAngles.
        rigidWater : boolean=True
            If true, water molecules will be fully rigid regardless of the value
            passed for the constraints argument
        implicitSolvent : object=None
            If not None, the implicit solvent model to use. Allowed values are
            HCT, OBC1, OBC2, or GBn
        implicitSolventKappa : float=None
            Debye screening parameter to model salt concentrations in GB
            solvent.
        implicitSolventSaltConc : float=0.0*u.moles/u.liter
            Salt concentration for GB simulations. Converted to Debye length
            ``kappa``
        temperature : float=298.15*u.kelvin
            Temperature used in the salt concentration-to-kappa conversion for
            GB salt concentration term
        soluteDielectric : float=1.0
            The solute dielectric constant to use in the implicit solvent model.
        solventDielectric : float=78.5
            The solvent dielectric constant to use in the implicit solvent
            model.
        removeCMMotion : boolean=True
            If true, a CMMotionRemover will be added to the System.
        hydrogenMass : mass=None
            The mass to use for hydrogen atoms bound to heavy atoms. Any mass
            added to a hydrogen is subtracted from the heavy atom to keep their
            total mass the same.
        ewaldErrorTolerance : float=0.0005
            The error tolerance to use if the nonbonded method is Ewald, PME, or LJPME.
        flexibleConstraints : bool=True
            If True, parameters for constrained degrees of freedom will be added to the System
        verbose : bool=False
            Optionally prints out a running progress report
        gbsaModel : str=None
            Can be ACE (to use the ACE solvation model) or None. Other values
            raise a ValueError
        drudeMass : mass=0.4*amu
            The mass to use for Drude particles.  Any mass added to a Drude particle is
            subtracted from its parent atom to keep their total mass the same.
        """
        # Load the parameter set
        self.loadParameters(params)
        hasbox = self.topology.getUnitCellDimensions() is not None
        # Check GB input parameters
        if implicitSolvent is not None and gbsaModel not in ('ACE', None):
            raise ValueError('gbsaModel must be ACE or None')
        # Set the cutoff distance in nanometers
        cutoff = None
        if nonbondedMethod is not ff.NoCutoff:
            cutoff = nonbondedCutoff
            # Remove units from cutoff
            if u.is_quantity(cutoff):
                cutoff = cutoff.value_in_unit(u.nanometers)

        if nonbondedMethod not in (ff.NoCutoff, ff.CutoffNonPeriodic,
                                   ff.CutoffPeriodic, ff.Ewald, ff.PME, ff.LJPME):
            raise ValueError('Illegal value for nonbonded method')
        if not hasbox and nonbondedMethod in (ff.CutoffPeriodic,
                                              ff.Ewald, ff.PME, ff.LJPME):
            raise ValueError('Illegal nonbonded method for a '
                             'non-periodic system')
        if implicitSolvent not in (HCT, OBC1, OBC2, GBn, GBn2, None):
            raise ValueError('Illegal implicit solvent model choice.')
        if not constraints in (None, ff.HAngles, ff.HBonds, ff.AllBonds):
            raise ValueError('Illegal constraints choice')

        # Define conversion factors
        length_conv = u.angstrom.conversion_factor_to(u.nanometer)
        _chmfrc = u.kilocalorie_per_mole/(u.angstrom*u.angstrom)
        _openmmfrc = u.kilojoule_per_mole/(u.nanometer*u.nanometer)
        bond_frc_conv = _chmfrc.conversion_factor_to(_openmmfrc)
        _chmfrc = u.kilocalorie_per_mole/(u.radians*u.radians)
        _openmmfrc = u.kilojoule_per_mole/(u.radians*u.radians)
        angle_frc_conv = _chmfrc.conversion_factor_to(_openmmfrc)
        dihe_frc_conv = u.kilocalorie_per_mole.conversion_factor_to(
                            u.kilojoule_per_mole)
        ene_conv = dihe_frc_conv

        # Create the system and determine if any of our atoms have NBFIX or NBTHOLE (and
        # therefore requires a CustomNonbondedForce instead)
        typenames = set()
        system = mm.System()
        if verbose: print('Adding particles...')
        for atom in self.atom_list:
            typenames.add(atom.type.name)
            system.addParticle(atom.mass)
        has_nbfix_terms = False
        typenames = list(typenames)
        try:
            for i, typename in enumerate(typenames):
                typ = params.atom_types_str[typename]
                for j in range(i, len(typenames)):
                    if typenames[j] in typ.nbfix:
                        has_nbfix_terms = True
                        raise StopIteration
        except StopIteration:
            pass
        has_nbthole_terms = False
        try:
            for i, typename in enumerate(typenames):
                typ = params.atom_types_str[typename]
                for j in range(i, len(typenames)):
                    if typenames[j] in typ.nbthole:
                        has_nbthole_terms = True
                        raise StopIteration
        except StopIteration:
            pass
        # test if the system containing the Drude particles
        has_drude_particle = False
        try:
            if self.drudeconsts_list:
                has_drude_particle = True
        except AttributeError:
            pass

        # Set up the constraints
        def _is_bond_in_water(bond):
            return bond.atom1.residue.resname in WATNAMES and \
                   tuple(sorted([bond.atom1.type.atomic_number, bond.atom2.type.atomic_number])) == (1, 8)

        n_cons_bond = n_cons_angle = 0
        if verbose and (constraints is not None or rigidWater):
            print('Adding constraints...')

        for bond in self.bond_list:
            if constraints in (ff.AllBonds, ff.HAngles):
                system.addConstraint(bond.atom1.idx, bond.atom2.idx,
                                     bond.bond_type.req*length_conv)
                n_cons_bond += 1
            elif constraints is ff.HBonds:
                if bond.atom1.type.atomic_number == 1 or bond.atom2.type.atomic_number == 1:
                    system.addConstraint(bond.atom1.idx, bond.atom2.idx,
                                         bond.bond_type.req*length_conv)
                    n_cons_bond += 1
            elif rigidWater:
                if _is_bond_in_water(bond):
                    system.addConstraint(bond.atom1.idx, bond.atom2.idx,
                                         bond.bond_type.req*length_conv)
                    n_cons_bond += 1

        # Add virtual sites
        if hasattr(self, 'lonepair_list'):
            if verbose: print('Adding lonepairs...')
            for lpsite in self.lonepair_list:
                index=lpsite[0]
                atom1=lpsite[1]
                atom2=lpsite[2]
                atom3=lpsite[3]
                if atom3 >= 0: 
                    if lpsite[4] > 0 : # relative lonepair type
                        r = lpsite[4] /10.0 # in nanometer
                        xweights = [-1.0, 0.0, 1.0]
                    elif lpsite[4] < 0: # bisector lonepair type
                        r = lpsite[4] / (-10.0)
                        xweights = [-1.0, 0.5, 0.5]
                    theta = lpsite[5] * pi / 180.0
                    phi = (180.0 - lpsite[6]) * pi / 180.0
                    p = [r*cos(theta), r*sin(theta)*cos(phi), r*sin(theta)*sin(phi)]
                    p = [x if abs(x) > 1e-10 else 0 for x in p] # Avoid tiny numbers caused by roundoff error
                    system.setVirtualSite(index, mm.LocalCoordinatesSite(atom1, atom3, atom2, mm.Vec3(1.0, 0.0, 0.0), mm.Vec3(xweights[0],xweights[1],xweights[2]), mm.Vec3(0.0, -1.0, 1.0), mm.Vec3(p[0],p[1],p[2])))
                else: # colinear lonepair type
                    # find a real atom to be the third one for LocalCoordinatesSite
                    for bond in self.bond_list:
                        if (bond.atom1.idx == atom2 and bond.atom2.idx != atom1):
                            a3=bond.atom2.idx
                        elif (bond.atom2.idx == atom2 and bond.atom1.idx != atom1):
                            a3=bond.atom1.idx
                    r = lpsite[4] / 10.0 # in nanometer
                    system.setVirtualSite(index, mm.LocalCoordinatesSite(atom1, atom2, a3, mm.Vec3(1.0, 0.0, 0.0), mm.Vec3(1.0,-1.0, 0.0), mm.Vec3(0.0, -1.0, 1.0), mm.Vec3(r,0.0,0.0)))   
        # Add Bond forces
        if verbose: print('Adding bonds...')
        force = mm.HarmonicBondForce()
        force.setForceGroup(self.BOND_FORCE_GROUP)
        # See which, if any, energy terms we omit
        omit_all = not flexibleConstraints and constraints in (ff.AllBonds, ff.HAngles)
        omit_h = not flexibleConstraints and constraints is not None
        omit_h_in_water = not flexibleConstraints and (constraints is not None or rigidWater)
        for bond in self.bond_list:
            if omit_all: continue
            if omit_h and (bond.atom1.type.atomic_number == 1 or bond.atom2.type.atomic_number == 1): continue
            if omit_h_in_water and _is_bond_in_water(bond): continue
            force.addBond(bond.atom1.idx, bond.atom2.idx,
                          bond.bond_type.req*length_conv,
                          2*bond.bond_type.k*bond_frc_conv)
        system.addForce(force)
        # Add Angle forces
        if verbose: print('Adding angles...')
        force = mm.HarmonicAngleForce()
        force.setForceGroup(self.ANGLE_FORCE_GROUP)
        if constraints is ff.HAngles:
            num_constrained_bonds = system.getNumConstraints()
            atom_constraints = [[]] * system.getNumParticles()
            for i in range(num_constrained_bonds):
                c = system.getConstraintParameters(i)
                dist = c[2].value_in_unit(u.nanometer)
                atom_constraints[c[0]].append((c[1], dist))
                atom_constraints[c[1]].append((c[0], dist))
        for angle in self.angle_list:
            # Only constrain angles including hydrogen here
            if (angle.atom1.type.atomic_number != 1 and angle.atom3.type.atomic_number != 1):
                continue
            a1 = angle.atom1.type.atomic_number
            a2 = angle.atom2.type.atomic_number
            a3 = angle.atom3.type.atomic_number
            nh = int(a1==1) + int(a3==1)
            if constraints is ff.HAngles:
                constrained = (nh == 2 or (nh == 1 and a2 == 8))
            elif rigidWater:
                constrained = (nh == 2 and a2 == 8 and angle.atom1.residue.resname in WATNAMES)
            else:
                constrained = False # no constraints
            if constrained:
                l1 = l2 = None
                for bond in angle.atom2.bonds:
                    if bond.atom1 is angle.atom1 or bond.atom2 is angle.atom1:
                        l1 = bond.bond_type.req * length_conv
                    elif bond.atom1 is angle.atom3 or bond.atom2 is angle.atom3:
                        l2 = bond.bond_type.req * length_conv
                # Compute the distance between the atoms and add a constraint
                length = sqrt(l1*l1 + l2*l2 - 2*l1*l2*
                              cos(angle.angle_type.theteq*pi/180))
                system.addConstraint(angle.atom1.idx, angle.atom3.idx, length)
                n_cons_angle += 1
            if flexibleConstraints or not constrained:
                force.addAngle(angle.atom1.idx, angle.atom2.idx,
                               angle.atom3.idx, angle.angle_type.theteq*pi/180,
                               2*angle.angle_type.k*angle_frc_conv)
        if verbose and (constraints is not None or rigidWater):
            print('    Number of bond constraints:', n_cons_bond)
            print('    Number of angle constraints:', n_cons_angle)
        for angle in self.angle_list:
            # Already did the angles with hydrogen above. So skip those here
            if (angle.atom1.type.atomic_number == 1 or angle.atom3.type.atomic_number == 1):
                continue
            force.addAngle(angle.atom1.idx, angle.atom2.idx,
                           angle.atom3.idx, angle.angle_type.theteq*pi/180,
                           2*angle.angle_type.k*angle_frc_conv)
        system.addForce(force)

        # Add the urey-bradley terms
        if verbose: print('Adding Urey-Bradley terms')
        force = mm.HarmonicBondForce()
        force.setForceGroup(self.UREY_BRADLEY_FORCE_GROUP)
        for ub in self.urey_bradley_list:
            force.addBond(ub.atom1.idx, ub.atom2.idx,
                          ub.ub_type.req*length_conv,
                          2*ub.ub_type.k*bond_frc_conv)
        system.addForce(force)

        # Add dihedral forces
        if verbose: print('Adding torsions...')
        force = mm.PeriodicTorsionForce()
        force.setForceGroup(self.DIHEDRAL_FORCE_GROUP)
        for tor in self.dihedral_parameter_list:
            force.addTorsion(tor.atom1.idx, tor.atom2.idx, tor.atom3.idx,
                             tor.atom4.idx, tor.dihedral_type.per,
                             tor.dihedral_type.phase*pi/180,
                             tor.dihedral_type.phi_k*dihe_frc_conv)
        system.addForce(force)

        if verbose: print('Adding impropers...')
        # Ick. OpenMM does not have an improper torsion class. Need to
        # construct one from CustomTorsionForce that respects toroidal boundaries
        energy_function = 'k*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta-theta0);'
        energy_function += 'pi = %f;' % pi
        force = mm.CustomTorsionForce(energy_function)
        force.addPerTorsionParameter('k')
        force.addPerTorsionParameter('theta0')
        force.setForceGroup(self.IMPROPER_FORCE_GROUP)
        for imp in self.improper_list:
            force.addTorsion(imp.atom1.idx, imp.atom2.idx,
                             imp.atom3.idx, imp.atom4.idx,
                             (imp.improper_type.k*dihe_frc_conv,
                              imp.improper_type.phieq*pi/180)
            )
        system.addForce(force)

        if hasattr(self, 'cmap_list'):
            if verbose: print('Adding CMAP coupled torsions...')
            force = mm.CMAPTorsionForce()
            force.setForceGroup(self.CMAP_FORCE_GROUP)
            # First get the list of cmap maps we're going to use. Just store the
            # IDs so we have simple integer comparisons to do later
            cmap_type_list = []
            cmap_map = dict()
            for cmap in self.cmap_list:
                if not id(cmap.cmap_type) in cmap_type_list:
                    ct = cmap.cmap_type
                    cmap_type_list.append(id(ct))
                    # Our torsion correction maps need to go from 0 to 360
                    # degrees
                    grid = ct.grid.switch_range().T
                    m = force.addMap(ct.resolution, [x*ene_conv for x in grid])
                    cmap_map[id(ct)] = m
            # Now add in all of the cmaps
            for cmap in self.cmap_list:
                if cmap.consecutive:
                    id1, id2 = cmap.atom1.idx, cmap.atom2.idx
                    id3, id4 = cmap.atom3.idx, cmap.atom4.idx
                    id5, id6 = cmap.atom2.idx, cmap.atom3.idx
                    id7, id8 = cmap.atom4.idx, cmap.atom5.idx
                else:
                    id1, id2 = cmap.atom1.idx, cmap.atom2.idx
                    id3, id4 = cmap.atom3.idx, cmap.atom4.idx
                    id5, id6 = cmap.atom5.idx, cmap.atom6.idx
                    id7, id8 = cmap.atom7.idx, cmap.atom8.idx
                force.addTorsion(cmap_map[id(cmap.cmap_type)],
                                 id1, id2, id3, id4, id5, id6, id7, id8)
            system.addForce(force)
        # Add nonbonded terms now
        if verbose: print('Adding nonbonded interactions...')
        force = mm.NonbondedForce()
        force.setUseDispersionCorrection(False)
        force.setForceGroup(self.NONBONDED_FORCE_GROUP)
        if not hasbox: # non-periodic
            if nonbondedMethod is ff.NoCutoff:
                force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
            elif nonbondedMethod is ff.CutoffNonPeriodic:
                if cutoff is None:
                    raise ValueError('No cutoff value specified')
                force.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
                force.setCutoffDistance(cutoff)
            else:
                raise ValueError('Illegal nonbonded method for non-periodic '
                                 'system')

            # See if we need to use a switching function
            if switchDistance and nonbondedMethod is not ff.NoCutoff:
                # make sure it's legal
                if (_strip_optunit(switchDistance, u.nanometer) >=
                        _strip_optunit(nonbondedCutoff, u.nanometer)):
                    raise ValueError('switchDistance is too large compared '
                                     'to the cutoff!')
                if _strip_optunit(switchDistance, u.nanometer) < 0:
                    # Detects negatives for both Quantity and float
                    raise ValueError('switchDistance must be non-negative!')
                force.setUseSwitchingFunction(True)
                force.setSwitchingDistance(switchDistance)

        else: # periodic
            # Set up box vectors (from inpcrd if available, or fall back to
            # prmtop definitions
            system.setDefaultPeriodicBoxVectors(*self.box_vectors)

            # Set cutoff
            if cutoff is None:
                # Compute cutoff automatically
                box = self.boxLengths
                min_box_width = min((box[0]/u.nanometers,
                                     box[1]/u.nanometers,
                                     box[2]/u.nanometers))
                CLEARANCE_FACTOR = 0.97
                cutoff = u.Quantity((min_box_width*CLEARANCE_FACTOR)/2.0,
                                    u.nanometers)
            if nonbondedMethod is not ff.NoCutoff:
                force.setCutoffDistance(cutoff)

            # Set nonbonded method.
            if nonbondedMethod is ff.NoCutoff:
                force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
            elif nonbondedMethod is ff.CutoffNonPeriodic:
                force.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
            elif nonbondedMethod is ff.CutoffPeriodic:
                force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
            elif nonbondedMethod is ff.Ewald:
                force.setNonbondedMethod(mm.NonbondedForce.Ewald)
            elif nonbondedMethod is ff.PME:
                force.setNonbondedMethod(mm.NonbondedForce.PME)
            elif nonbondedMethod is ff.LJPME:
                force.setNonbondedMethod(mm.NonbondedForce.LJPME)
            else:
                raise ValueError('Cutoff method is not understood')

            # See if we need to use a switching function
            if switchDistance and nonbondedMethod is not ff.NoCutoff:
                # make sure it's legal
                if (_strip_optunit(switchDistance, u.nanometer) >=
                        _strip_optunit(nonbondedCutoff, u.nanometer)):
                    raise ValueError('switchDistance is too large compared '
                                     'to the cutoff!')
                if _strip_optunit(switchDistance, u.nanometer) < 0:
                    # Detects negatives for both Quantity and float
                    raise ValueError('switchDistance must be non-negative!')
                force.setUseSwitchingFunction(True)
                force.setSwitchingDistance(switchDistance)

            if ewaldErrorTolerance is not None:
                force.setEwaldErrorTolerance(ewaldErrorTolerance)

        # Add per-particle nonbonded parameters (LJ params)
        sigma_scale = 2**(-1/6) * 2
        if not has_nbfix_terms:
            for atm in self.atom_list:
                force.addParticle(atm.charge, sigma_scale*atm.type.rmin*length_conv,
                                  abs(atm.type.epsilon*ene_conv))
        else:
            for atm in self.atom_list:
                force.addParticle(atm.charge, 1.0, 0.0)
            # Now add the custom nonbonded force that implements NBFIX. First
            # thing we need to do is condense our number of types
            lj_idx_list = [0 for atom in self.atom_list]
            lj_radii, lj_depths = [], []
            num_lj_types = 0
            lj_type_list = []
            for i, atom in enumerate(self.atom_list):
                atom = atom.type
                if lj_idx_list[i]: continue # already assigned
                num_lj_types += 1
                lj_idx_list[i] = num_lj_types
                ljtype = (atom.rmin, atom.epsilon)
                lj_type_list.append(atom)
                lj_radii.append(atom.rmin)
                lj_depths.append(atom.epsilon)
                for j in range(i+1, len(self.atom_list)):
                    atom2 = self.atom_list[j].type
                    if lj_idx_list[j] > 0: continue # already assigned
                    if atom2 is atom:
                        lj_idx_list[j] = num_lj_types
                    elif not atom.nbfix and not atom.nbthole:
                        # Only non-NBFIXed and non-NBTholed atom types can be compressed
                        ljtype2 = (atom2.rmin, atom2.epsilon)
                        if ljtype == ljtype2:
                            lj_idx_list[j] = num_lj_types
            # Now everything is assigned. Create the A-coefficient and
            # B-coefficient arrays
            acoef = [0 for i in range(num_lj_types*num_lj_types)]
            bcoef = acoef[:]
            for i in range(num_lj_types):
                for j in range(num_lj_types):
                    namej = lj_type_list[j].name
                    try:
                        rij, wdij, rij14, wdij14 = lj_type_list[i].nbfix[namej]
                    except KeyError:
                        rij = (lj_radii[i] + lj_radii[j]) * length_conv
                        wdij = sqrt(lj_depths[i] * lj_depths[j]) * ene_conv
                    else:
                        rij *= length_conv
                        wdij *= ene_conv
                    acoef[i+num_lj_types*j] = sqrt(wdij) * rij**6
                    bcoef[i+num_lj_types*j] = 2 * wdij * rij**6
            cforce = mm.CustomNonbondedForce('(a/r6)^2-b/r6; r6=r^6;'
                                             'a=acoef(type1, type2);'
                                             'b=bcoef(type1, type2)')
            cforce.addTabulatedFunction('acoef',
                    mm.Discrete2DFunction(num_lj_types, num_lj_types, acoef))
            cforce.addTabulatedFunction('bcoef',
                    mm.Discrete2DFunction(num_lj_types, num_lj_types, bcoef))
            cforce.addPerParticleParameter('type')
            cforce.setForceGroup(self.NONBONDED_FORCE_GROUP)
            if (nonbondedMethod in (ff.PME, ff.LJPME, ff.Ewald, ff.CutoffPeriodic)):
                cforce.setNonbondedMethod(cforce.CutoffPeriodic)
                cforce.setCutoffDistance(nonbondedCutoff)
            elif nonbondedMethod is ff.NoCutoff:
                cforce.setNonbondedMethod(cforce.NoCutoff)
            elif nonbondedMethod is ff.CutoffNonPeriodic:
                cforce.setNonbondedMethod(cforce.CutoffNonPeriodic)
                cforce.setCutoffDistance(nonbondedCutoff)
            else:
                raise ValueError('Unrecognized nonbonded method')
            if switchDistance and nonbondedMethod is not ff.NoCutoff:
                # make sure it's legal
                if (_strip_optunit(switchDistance, u.nanometer) >=
                        _strip_optunit(nonbondedCutoff, u.nanometer)):
                    raise ValueError('switchDistance is too large compared '
                                     'to the cutoff!')
                if _strip_optunit(switchDistance, u.nanometer) < 0:
                    # Detects negatives for both Quantity and float
                    raise ValueError('switchDistance must be non-negative!')
                cforce.setUseSwitchingFunction(True)
                cforce.setSwitchingDistance(switchDistance)
            for i in lj_idx_list:
                cforce.addParticle((i - 1,)) # adjust for indexing from 0

        # Add NBTHOLE terms
        if has_drude_particle and has_nbthole_terms:
            nbt_idx_list = [0 for atom in self.atom_list]
            nbt_alpha_list = [0 for atom in self.atom_list] # only save alpha for NBThole pairs
            num_nbt_types = 0 
            nbt_type_list = []
            nbt_set_list = []
            for i, atom in enumerate(self.atom_list):
                atom = atom.type
                if not atom.nbthole: continue # get them as zero
                if nbt_idx_list[i]: continue # already assigned
                num_nbt_types += 1
                nbt_idx_list[i] = num_nbt_types
                nbt_idx_list[i+1] = num_nbt_types
                nbt_alpha_list[i] = pow(-1*self.drudeconsts_list[i][0],-1./6.)
                nbt_alpha_list[i+1] = pow(-1*self.drudeconsts_list[i][0],-1./6.)
                nbt_type_list.append(atom)
                nbt_set_list.append([i,i+1])
                for j in range(i+1, len(self.atom_list)):
                    atom2 = self.atom_list[j].type
                    if nbt_idx_list[j] > 0: continue # already assigned
                    if atom2 is atom:
               	       nbt_idx_list[j] = num_nbt_types
               	       nbt_idx_list[j+1] = num_nbt_types
                       nbt_alpha_list[j] = pow(-1*self.drudeconsts_list[j][0],-1./6.)
                       nbt_alpha_list[j+1] = pow(-1*self.drudeconsts_list[j][0],-1./6.)
                       nbt_set_list[num_nbt_types-1].append(j)
                       nbt_set_list[num_nbt_types-1].append(j+1)
            num_total_nbt=num_nbt_types+1 # use zero index for all the atoms with no nbthole
            nbt_interset_list=[]
            # need to get all other particles as an independent group, so in total num_nbt_types+1
            coef = [0 for i in range(num_total_nbt*num_total_nbt)]
            for i in range(num_nbt_types):
                for j in range(num_nbt_types):
                    namej = nbt_type_list[j].name
                    nbt_value = nbt_type_list[i].nbthole.get(namej,0)
                    if abs(nbt_value)>TINY and i<j :  nbt_interset_list.append([i+1,j+1])
                    coef[i+1+num_total_nbt*(j+1)]=nbt_value
            nbtforce = mm.CustomNonbondedForce('-138.935456*charge1*charge2*(1.0+0.5*screen*r)*exp(-1.0*screen*r)/r; screen=coef(type1, type2) * alpha1*alpha2*10.0')
            nbtforce.addTabulatedFunction('coef', mm.Discrete2DFunction(num_total_nbt, num_total_nbt, coef))
            nbtforce.addPerParticleParameter('charge')
            nbtforce.addPerParticleParameter('alpha')
            nbtforce.addPerParticleParameter('type')
            nbtforce.setForceGroup(self.NONBONDED_FORCE_GROUP)
            # go through all the particles to set up per-particle parameters
            for i in range(system.getNumParticles()):
                c=force.getParticleParameters(i)
                cc=c[0]/u.elementary_charge
                aa=nbt_alpha_list[i]
                ti=nbt_idx_list[i]
                nbtforce.addParticle([cc,aa,ti])
            # set interaction group
            for a in nbt_interset_list:
                ai=a[0]
                aj=a[1]
                nbtforce.addInteractionGroup(nbt_set_list[ai-1],nbt_set_list[aj-1])
            nbtforce.setNonbondedMethod(nbtforce.CutoffPeriodic)
            nbtforce.setCutoffDistance(0.5*u.nanometer)
            nbtforce.setUseSwitchingFunction(False)
            # now, add the actual force to the system
            system.addForce(nbtforce)

        # build 1-2, 1-3 and 1-4 pairs from connectivity
        if verbose:
            print('Build exclusion list...')
        self._build_exclusion_list()
        if verbose:
            print('    Number of 1-2 pairs: %i' % len(self.pair_12_list))
            print('    Number of 1-3 pairs: %i' % len(self.pair_13_list))
            print('    Number of 1-4 pairs: %i' % len(self.pair_14_list))

        # Add 1-4 interactions
        sigma_scale = 2**(-1/6)
        nbxmod = abs(params.nbxmod)
        if nbxmod == 4:
            for ia1, ia4 in self.pair_14_list:
                force.addException(ia1, ia4, 0.0, 0.1, 0.0)
        if nbxmod == 5:
            for ia1, ia4 in self.pair_14_list:
                atom1 = self.atom_list[ia1]
                atom4 = self.atom_list[ia4]
                charge_prod = (atom1.charge * atom4.charge)
                if str(atom4.type) in atom1.type.nbfix:
                    epsilon = (atom1.type.nbfix[str(atom4.type)][3]) * ene_conv
                    sigma = (atom1.type.nbfix[str(atom4.type)][2]) * (length_conv * sigma_scale)
                else:
                    epsilon = sqrt(abs(atom1.type.epsilon_14 * atom4.type.epsilon_14)) * ene_conv
                    sigma = (atom1.type.rmin_14 + atom4.type.rmin_14) * (length_conv * sigma_scale)
                force.addException(ia1, ia4, charge_prod, sigma, epsilon)

        # Add excluded atoms
        # Drude and lonepairs will be excluded based on their parent atoms
        parent_exclude_list=[[] for _ in self.atom_list]
        for lpsite in self.lonepair_list:
            idx = lpsite[1]
            idxa = lpsite[0]
            parent_exclude_list[idx].append(idxa)
            force.addException(idx, idxa, 0.0, 0.1, 0.0)
        if has_drude_particle:
            for pair in self.drudepair_list:
                idx = pair[0]
                idxa = pair[1]
                parent_exclude_list[idx].append(idxa)
                force.addException(idx, idxa, 0.0, 0.1, 0.0)
            # If lonepairs and Drude particles are bonded to the same parent atom, add exception
            for excludeterm in parent_exclude_list:
                if(len(excludeterm) >= 2):
                    for i in range(len(excludeterm)):
                        for j in range(i):
                            force.addException(excludeterm[j], excludeterm[i], 0.0, 0.1, 0.0)
        # Exclude 1-2 and 1-3 pairs as well as the lonepair/Drude attached onto them
        if nbxmod > 1:
            for ia1, ia2 in self.pair_12_list:
                for excludeatom in [ia1]+parent_exclude_list[ia1]:
                    for excludeatom2 in [ia2]+parent_exclude_list[ia2]:
                        force.addException(excludeatom, excludeatom2, 0.0, 0.1, 0.0)
        if nbxmod > 2:
            for ia1, ia3 in self.pair_13_list:
                for excludeatom in [ia1]+parent_exclude_list[ia1]:
                    for excludeatom2 in [ia3]+parent_exclude_list[ia3]:
                        force.addException(excludeatom, excludeatom2, 0.0, 0.1, 0.0)
        system.addForce(force)

        # Add Drude particles (Drude force)
        if has_drude_particle:
            if verbose: print('Adding Drude force and Thole screening...')
            drudeforce = mm.DrudeForce()
            drudeforce.setForceGroup(self.DRUDE_FORCE_GROUP)
            for pair in self.drudepair_list:
                parentatom=pair[0]
                drudeatom=pair[1]
                p=[-1, -1, -1]
                a11 = 0 
                a22 = 0
                charge = self.atom_list[drudeatom].charge
                polarizability = self.drudeconsts_list[parentatom][0]/(-1000.0)
                for aniso in self.aniso_list: # not smart to do another loop, but the number of aniso is small anyway
                    if (parentatom==aniso[0]):
                        p[0]=aniso[1]
                        p[1]=aniso[2]
                        p[2]=aniso[3]
                        k11=aniso[4]
                        k22=aniso[5]
                        k33=aniso[6]
                        # solve out DrudeK, which should equal 500.0
                        a = k11+k22+3*k33
                        b = 2*k11*k22+4*k11*k33+4*k22*k33+6*k33*k33
                        c = 3*k33*(k11+k33)*(k22+k33)
                        DrudeK = (sqrt(b*b-4*a*c)-b)/2/a
                        a11=round(DrudeK/(k11+k33+DrudeK),5)
                        a22=round(DrudeK/(k22+k33+DrudeK),5)
                drudeforce.addParticle(drudeatom, parentatom, p[0], p[1], p[2], charge, polarizability, a11, a22 )
            system.addForce(drudeforce)
            particleMap = {}
            for i in range(drudeforce.getNumParticles()):
                particleMap[drudeforce.getParticleParameters(i)[0]] = i

            # Apply thole screening for 1-2 and 1-3 pairs
            for ia1, ia2 in self.pair_12_list + self.pair_13_list:
                alpha1 = self.drudeconsts_list[ia1][0]
                alpha2 = self.drudeconsts_list[ia2][0]
                if abs(alpha1) > TINY and abs(alpha2) > TINY: # both are Drude parent atoms
                    thole1 = self.drudeconsts_list[ia1][1]
                    thole2 = self.drudeconsts_list[ia2][1]
                    drude1 = ia1 + 1 # CHARMM psf has hard-coded rule that the Drude is next to its parent
                    drude2 = ia2 + 1
                    drudeforce.addScreenedPair(particleMap[drude1], particleMap[drude2], thole1+thole2)

            # Set the masses of Drude particles.
            if not u.is_quantity(drudeMass):
                drudeMass *= u.dalton
            for i in range(drudeforce.getNumParticles()):
                params = drudeforce.getParticleParameters(i)
                particle = params[0]
                parent = params[1]
                transferMass = drudeMass-system.getParticleMass(particle)
                system.setParticleMass(particle, drudeMass)
                system.setParticleMass(parent, system.getParticleMass(parent)-transferMass)

        # If we needed a CustomNonbondedForce, map all of the exceptions from
        # the NonbondedForce to the CustomNonbondedForce
        if has_nbfix_terms:
            for i in range(force.getNumExceptions()):
                ii, jj, q, eps, sig = force.getExceptionParameters(i)
                cforce.addExclusion(ii, jj)
            system.addForce(cforce)
        if has_drude_particle and has_nbthole_terms:
            for i in range(force.getNumExceptions()):
                ii, jj, q, eps, sig = force.getExceptionParameters(i)
                nbtforce.addExclusion(ii, jj)

        # Add GB model if we're doing one
        if implicitSolvent is not None:
            if verbose: print('Adding GB parameters...')
            # If implicitSolventKappa is None, compute it from salt
            # concentration
            if implicitSolventKappa is None:
                if u.is_quantity(implicitSolventSaltConc):
                    sc = implicitSolventSaltConc.value_in_unit(u.moles/u.liter)
                    implicitSolventSaltConc = sc
                if u.is_quantity(temperature):
                    temperature = temperature.value_in_unit(u.kelvin)
                # The constant is 1 / sqrt( epsilon_0 * kB / (2 * NA * q^2 *
                # 1000) ) where NA is avogadro's number, epsilon_0 is the
                # permittivity of free space, q is the elementary charge (this
                # number matches sander/pmemd's kappa conversion factor)
                implicitSolventKappa = 50.33355 * sqrt(implicitSolventSaltConc /
                                                solventDielectric / temperature)
                # Multiply by 0.73 to account for ion exclusions, and multiply
                # by 10 to convert to 1/nm from 1/angstroms
                implicitSolventKappa *= 7.3
            elif implicitSolvent is None:
                implicitSolventKappa = 0.0

            if u.is_quantity(implicitSolventKappa):
                implicitSolventKappa = implicitSolventKappa.value_in_unit(
                                            (1.0/u.nanometer).unit)
            if implicitSolvent is HCT:
                gb = GBSAHCTForce(solventDielectric, soluteDielectric, gbsaModel,
                                  cutoff, kappa=implicitSolventKappa)
            elif implicitSolvent is OBC1:
                gb = GBSAOBC1Force(solventDielectric, soluteDielectric, gbsaModel,
                                   cutoff, kappa=implicitSolventKappa)
            elif implicitSolvent is OBC2:
                gb = GBSAOBC2Force(solventDielectric, soluteDielectric, gbsaModel,
                                   cutoff, kappa=implicitSolventKappa)
            elif implicitSolvent is GBn:
                gb = GBSAGBnForce(solventDielectric, soluteDielectric, gbsaModel,
                                  cutoff, kappa=implicitSolventKappa)
            elif implicitSolvent is GBn2:
                gb = GBSAGBn2Force(solventDielectric, soluteDielectric, gbsaModel,
                                   cutoff, kappa=implicitSolventKappa)
            gb_parms = gb.getStandardParameters(self.topology)
            for atom, gb_parm in zip(self.atom_list, gb_parms):
                gb.addParticle([atom.charge] + list(gb_parm))
            # Set cutoff method
            if nonbondedMethod is ff.NoCutoff:
                gb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
            elif nonbondedMethod is ff.CutoffNonPeriodic:
                gb.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
                gb.setCutoffDistance(cutoff)
            elif nonbondedMethod is ff.CutoffPeriodic:
                gb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
                gb.setCutoffDistance(cutoff)
            else:
                raise ValueError('Illegal nonbonded method for use with GBSA')
            gb.setForceGroup(self.GB_FORCE_GROUP)
            gb.finalize()
            system.addForce(gb)
            force.setReactionFieldDielectric(1.0) # applies to NonbondedForce

        # See if we repartition the hydrogen masses
        if hydrogenMass is not None:
            for bond in self.bond_list:
                # Only take the ones with at least one hydrogen
                if (bond.atom1.type.atomic_number != 1 and
                    bond.atom2.type.atomic_number != 1):
                    continue
                atom1, atom2 = bond.atom1, bond.atom2
                if atom1.type.atomic_number == 1:
                    atom1, atom2 = atom2, atom1 # now atom2 is hydrogen for sure
                if atom1.type.atomic_number != 1:
                    transfer_mass = hydrogenMass - atom2.mass
                    new_mass1 = (system.getParticleMass(atom1.idx) -
                                 transfer_mass)
                    system.setParticleMass(atom2.idx, hydrogenMass)
                    system.setParticleMass(atom1.idx, new_mass1)
        # See if we want to remove COM motion
        if removeCMMotion:
            system.addForce(mm.CMMotionRemover())

        # Cache our system for easy access
        self._system = system

        return system

    @property
    def system(self):
        """
        Return the cached system class -- it needs to be initialized via
        "createSystem" first!
        """
        try:
            return self._system
        except AttributeError:
            raise AttributeError('You must initialize the system with '
                                 'createSystem before accessing the cached '
                                 'object.')

    @property
    def boxLengths(self):
        """ Return tuple of 3 units """
        try:
            # See if we have a cached version
            return self._boxLengths
        except AttributeError:
            pass
        if self.box_vectors is not None:
            # Get the lengths of each vector
            if u.is_quantity(self.box_vectors):
                # Unlikely -- each vector is like a quantity
                vecs = self.box_vectors.value_in_unit(u.nanometers)
            elif u.is_quantity(self.box_vectors[0]):
                # Assume all box vectors are quantities
                vecs = [x.value_in_unit(u.nanometers) for x in self.box_vectors]
            else:
                # Assume nanometers
                vecs = self.box_vectors
            a = sqrt(vecs[0][0]*vecs[0][0] + vecs[0][1]*vecs[0][1] +
                     vecs[0][2]*vecs[0][2])
            b = sqrt(vecs[1][0]*vecs[1][0] + vecs[1][1]*vecs[1][1] +
                     vecs[1][2]*vecs[1][2])
            c = sqrt(vecs[2][0]*vecs[2][0] + vecs[2][1]*vecs[2][1] +
                     vecs[2][2]*vecs[2][2])
            self._boxLengths = (a, b, c) * u.nanometers
            return self._boxLengths
        return None

    @boxLengths.setter
    def boxLengths(self, stuff):
        raise RuntimeError('Use setBox to set a box with lengths and angles '
                           'or set the boxVectors attribute with box vectors')

    @property
    def boxVectors(self):
        """ Return the box vectors """
        return self.box_vectors

    @boxVectors.setter
    def boxVectors(self, stuff):
        """ Sets the box vectors """
        try:
            # We may be changing the box, so delete the cached box lengths to
            # make sure they are recomputed if desired
            del self._boxLengths
        except AttributeError:
            pass
        self.box_vectors = stuff

    def deleteCmap(self):
        """ Deletes the CMAP terms from the CHARMM PSF """
        self.cmap_list = TrackedList()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def set_molecules(atom_list):
    """
    Correctly sets the molecularity of the system based on connectivity
    """
    from sys import setrecursionlimit, getrecursionlimit
    # Since we use a recursive function here, we make sure that the recursion
    # limit is large enough to handle the maximum possible recursion depth we'll
    # need (NATOM). We don't want to shrink it, though, since we use list
    # comprehensions in list constructors in some places that have an implicit
    # (shallow) recursion, therefore, reducing the recursion limit too much here
    # could raise a recursion depth exceeded exception during a _Type/Atom/XList
    # creation. Therefore, set the recursion limit to the greater of the current
    # limit or the number of atoms
    setrecursionlimit(max(len(atom_list), getrecursionlimit()))

    # Unmark all atoms so we can track which molecule each goes into
    atom_list.unmark()

    # The molecule "ownership" list
    owner = []
    # The way I do this is via a recursive algorithm, in which
    # the "set_owner" method is called for each bonded partner an atom
    # has, which in turn calls set_owner for each of its partners and
    # so on until everything has been assigned.
    molecule_number = 1 # which molecule number we are on
    for i in range(len(atom_list)):
        # If this atom has not yet been "owned", make it the next molecule
        # However, we only increment which molecule number we're on if
        # we actually assigned a new molecule (obviously)
        if not atom_list[i].marked:
            tmp = [i]
            _set_owner(atom_list, tmp, i, molecule_number)
            # Make sure the atom indexes are sorted
            tmp.sort()
            owner.append(tmp)
            molecule_number += 1
    return owner

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def _set_owner(atom_list, owner_array, atm, mol_id):
    """ Sets ownership of given atom and all bonded partners """
    # This could be written more simply as a recursive function, but that leads
    # to stack overflows, so I flattened it into an iterative one.
    partners = [atom_list[atm].bond_partners]
    loop_index = [0]
    atom_list[atm].marked = mol_id
    while len(partners) > 0:
        if loop_index[-1] >= len(partners[-1]):
            partners.pop()
            loop_index.pop()
            continue
        partner = partners[-1][loop_index[-1]]
        loop_index[-1] += 1
        if not partner.marked:
            owner_array.append(partner.idx)
            partner.marked = mol_id
            partners.append(partner.bond_partners)
            loop_index.append(0)
        elif partner.marked != mol_id:
            raise MoleculeError('Atom %d in multiple molecules' % partner.idx)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Routines that set the necessary radii lists based on a list of atoms with
# their connectivities

def _bondi_radii(atom_list):
    """ Sets the bondi radii """
    radii = [0.0 for atom in atom_list]
    for i, atom in enumerate(atom_list):
        if atom.type.atomic_number == 6:
            radii[i] = 1.7
        elif atom.type.atomic_number == 1:
            radii[i] = 1.2
        elif atom.type.atomic_number == 7:
            radii[i] = 1.55
        elif atom.type.atomic_number == 8:
            radii[i] = 1.5
        elif atom.type.atomic_number == 9:
            radii[i] = 1.5
        elif atom.type.atomic_number == 14:
            radii[i] = 2.1
        elif atom.type.atomic_number == 15:
            radii[i] = 1.85
        elif atom.type.atomic_number == 16:
            radii[i] = 1.8
        elif atom.type.atomic_number == 17:
            radii[i] = 1.5
        else:
            radii[i] = 1.5
    return radii  # converted to nanometers above

def _mbondi_radii(atom_list):
    """ Sets the mbondi radii """
    radii = [0.0 for atom in atom_list]
    for i, atom in enumerate(atom_list):
        # Radius of H atom depends on element it is bonded to
        if atom.type.atomic_number == 1:
            bondeds = list(atom.bond_partners)
            if bondeds[0].type.atomic_number in (6, 7): # C or N
                radii[i] = 1.3
            elif bondeds[0].type.atomic_number in (8, 16): # O or S
                radii[i] = 0.8
            else:
                radii[i] = 1.2
        # Radius of C atom depends on what type it is
        elif atom.type.atomic_number == 6:
            radii[i] = 1.7
        # All other elements have fixed radii for all types/partners
        elif atom.type.atomic_number == 7:
            radii[i] = 1.55
        elif atom.type.atomic_number == 8:
            radii[i] = 1.5
        elif atom.type.atomic_number == 9:
            radii[i] = 1.5
        elif atom.type.atomic_number == 14:
            radii[i] = 2.1
        elif atom.type.atomic_number == 15:
            radii[i] = 1.85
        elif atom.type.atomic_number == 16:
            radii[i] = 1.8
        elif atom.type.atomic_number == 17:
            radii[i] = 1.5
        else:
            radii[i] = 1.5
    return radii  # converted to nanometers above

def _mbondi2_radii(atom_list):
    """ Sets the mbondi2 radii """
    radii = [0.0 for atom in atom_list]
    for i, atom in enumerate(atom_list):
        # Radius of H atom depends on element it is bonded to
        if atom.type.atomic_number == 1:
            if atom.bond_partners[0].type.atomic_number == 7:
                radii[i] = 1.3
            else:
                radii[i] = 1.2
        # Radius of C atom depends on what type it is
        elif atom.type.atomic_number == 6:
            radii[i] = 1.7
        # All other elements have fixed radii for all types/partners
        elif atom.type.atomic_number == 7:
            radii[i] = 1.55
        elif atom.type.atomic_number == 8:
            radii[i] = 1.5
        elif atom.type.atomic_number == 9:
            radii[i] = 1.5
        elif atom.type.atomic_number == 14:
            radii[i] = 2.1
        elif atom.type.atomic_number == 15:
            radii[i] = 1.85
        elif atom.type.atomic_number == 16:
            radii[i] = 1.8
        elif atom.type.atomic_number == 17:
            radii[i] = 1.5
        else:
            radii[i] = 1.5
    return radii  # Converted to nanometers above

def _mbondi3_radii(atom_list):
    """ Sets the mbondi3 radii """
    radii = _mbondi2_radii(atom_list)
    for i, atom in enumerate(atom_list):
        # Adjust OE (GLU), OD (ASP) and HH/HE (ARG)
        if atom.residue.resname in ('GLU', 'ASP'):
            if atom.name.startswith('OE') or atom.name.startswith('OD'):
                radii[i] = 1.4
        elif atom.residue.resname == 'ARG':
            if atom.name.startswith('HH') or atom.name.startswith('HE'):
                radii[i] = 1.17
        # Adjust carboxylate O radii on C-termini
        if atom.name == 'OXT':
            radii[i] = 1.4
    return radii  # Converted to nanometers above

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    import doctest
    doctest.testmod()
