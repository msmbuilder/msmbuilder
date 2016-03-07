"""Create index file for atoms or distance pairs.

This index file can be used by for RMSD distance calculations, to specify
pairs of atoms for AtomPairsFeaturizer, or to specify particular atoms
for SuperposeFeaturizer.

The format of the index file is flat text, with each line containing either
1 or 2 0-based atom indices.
"""
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

from __future__ import print_function, division, absolute_import
import os
import itertools

import mdtraj as md
import numpy as np
from ..cmdline import Command, argument, argument_group, exttype


class AtomIndices(Command):
    _group = '0-Support'
    _concrete = True
    description = __doc__
    pdb = argument('-p', '--pdb', required=True, help='Path to PDB file')
    out = argument('-o', '--out', required=True, help='Path to output file',
                   type=exttype('.txt'))

    section1 = argument_group(description='Mode: Choose One')
    group1 = section1.add_mutually_exclusive_group(required=True)
    group1.add_argument('-d', '--distance-pairs', action='store_true', help='''
        Create a 2-dimensional index file with (N choose 2) rows and 2
        columns, where each row specifies a pair of indices. All (N choose 2)
        pairs of the selected atoms will be written.''')
    group1.add_argument('-a', '--atoms', action='store_true', help='''
        Create a 1-dimensional index file containing the indices of the
        selected atoms.''')

    section2 = argument_group(description='Selection Criteria: Choose One')
    group2 = section2.add_mutually_exclusive_group(required=True)
    group2.add_argument('--minimal', action='store_true', help='''Keep the
        atoms in protein residues with names CA, CB, C, N, O, (recommended).''')
    group2.add_argument('--heavy', action='store_true', help='''All
        non-hydrogen atoms that are not symmetry equivalent. By symmetry
        equivalent, we mean atoms identical under an exchange of labels. For
        example, heavy will exclude the two pairs of equivalent carbons (CD,
        CE) in a PHE ring.''')
    group2.add_argument('--alpha', action='store_true', help='''Only alpha
        carbons.''')
    group2.add_argument('--water', action='store_true', help='''Water oxygen
        atoms.''')
    group2.add_argument('--all', action='store_true', help='''Selection
        includes every atom.''')

    def __init__(self, args):
        self.args = args
        if os.path.exists(args.out):
            self.error('IOError: file exists: %s' % args.out)
        self.pdb = md.load(os.path.expanduser(args.pdb))
        print('Loaded pdb containing (%d) chains, (%d) residues, (%d) atoms.' %
              (self.pdb.topology.n_chains, self.pdb.topology.n_residues,
               self.pdb.topology.n_atoms))

    def start(self):
        if self.args.all:
            s = 'all'
        elif self.args.alpha:
            s = 'alpha'
        elif self.args.minimal:
            s = 'minimal'
        elif self.args.heavy:
            s = 'heavy'
        elif self.args.water:
            s = 'water'
        else:
            raise RuntimeError()

        atom_indices = self.pdb.topology.select_atom_indices(s)
        n_atoms = len(atom_indices)
        n_residues = len(np.unique(
            [self.pdb.topology.atom(i).residue.index for i in atom_indices]))
        print('Selected (%d) atoms from (%d) unique residues.' % (
            n_atoms, n_residues))

        if self.args.distance_pairs:
            out = np.array(list(itertools.combinations(atom_indices, 2)))
        elif self.args.atoms:
            out = np.array(atom_indices)
        else:
            raise RuntimeError
        np.savetxt(self.args.out, out, '%d')
