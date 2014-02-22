'''Create an index file for specified dihedral angles in a PDB
'''
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division

import os
import numpy as np
import mdtraj as md
from mdtraj.geometry.dihedral import (_atom_sequence, PHI_ATOMS, PSI_ATOMS,
    OMEGA_ATOMS, CHI1_ATOMS, CHI2_ATOMS, CHI3_ATOMS, CHI4_ATOMS)

from mixtape.cmdline import Command, argument, argument_group

__all__ = ['DihedralIndices']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class DihedralIndices(Command):
    description="Create index file for dihedral angles."
    pdb = argument('-p', '--pdb', required=True, help='Path to PDB file')
    out = argument('-o', '--out', required=True, help='Path to output file')
    
    section2 = argument_group(description='Selection Criteria: Choose One or More')
    section2.add_argument('--phi', action='store_true', help='''Backbone phi
        (C-N-CA-C) angles''')
    section2.add_argument('--psi', action='store_true', help='''Backbone psi
        (N-CA-C-N) angles''')
    section2.add_argument('--omega', action='store_true', help='''Backbone omega
        (CA-C-N-CA) angles''')
    section2.add_argument('--chi1', action='store_true', help='''Chi1 is the
        first side chain torsion angle formed between the 4 atoms over the
        CA-CB axis.''')
    section2.add_argument('--chi2', action='store_true', help='''Chi2 is the
        second side chain torsion angle formed between the corresponding 4
        atoms over the CB-CG axis.''')
    section2.add_argument('--chi3', action='store_true', help='''Chi3 is the
        third side chain torsion angle formed between the corresponding 4 atoms
        over the CG-CD axis (only the residues ARG, GLN, GLU, LYS & MET have
        these atoms)''')
    section2.add_argument('--chi4', action='store_true', help='''Chi4 is the
        fourth side chain torsion angle formed between the corresponding 4
        atoms over the CD-CE or CD-NE axis (only ARG & LYS residues have these
        atoms)''')
    
    def __init__(self, args):
        self.args = args
        if os.path.exists(args.out):
            self.error('IOError: file exists: %s' % args.out)
        self.pdb = md.load(args.pdb)
        print('Loaded pdb containing (%d) chains, (%d) residues, (%d) atoms.' %
            (self.pdb.topology.n_chains, self.pdb.topology.n_residues,
             self.pdb.topology.n_atoms))
    
    def start(self):
        dihedral_atom_types = []
        if self.args.phi:
            dihedral_atom_types.append(PHI_ATOMS)
        if self.args.psi:
            dihedral_atom_types.append(PSI_ATOMS)
        if self.args.omega:
            dihedral_atom_types.append(OMEGA_ATOMS)
        if self.args.chi1:
            dihedral_atom_types.extend(CHI1_ATOMS)
        if self.args.chi2:
            dihedral_atom_types.extend(CHI2_ATOMS)
        if self.args.chi3:
            dihedral_atom_types.extend(CHI3_ATOMS)
        if self.args.chi4:
            dihedral_atom_types.extend(CHI4_ATOMS)

        rids, indices = zip(*(_atom_sequence(self.pdb, atoms) for atoms in dihedral_atom_types))
        rids = np.concatenate(rids)
        id_sort = np.argsort(rids)
        if not any(x.size for x in indices):
            self.error('No dihedral angles matched.')
        indices = np.vstack(x for x in indices if x.size)[id_sort]
        print('Selected (%d) dihedrals from (%d) unique residues.' % (len(indices),
            len(np.unique(rids))))
        np.savetxt(self.args.out, indices, '%d')
