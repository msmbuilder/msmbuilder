import itertools

import numpy as np

ATOM_NAMES = ["N", "CA", "CB", "C", "O", "H"]


def get_atompair_indices(reference_traj, keep_atoms=None,
                         exclude_atoms=None, reject_bonded=True):
    """Get a list of acceptable atom pairs.

    Parameters
    ----------
    reference_traj : mdtraj.Trajectory
        Trajectory to grab atom pairs from
    keep_atoms : np.ndarray, dtype=string, optional
        Select only these atom names. Defaults to N, CA, CB, C, O, H
    exclude_atoms : np.ndarray, dtype=string, optional
        Exclude these atom names
    reject_bonded : bool, default=True
        If True, exclude bonded atompairs.

    Returns
    -------
    atom_indices : np.ndarray, dtype=int
        The atom indices that pass your criteria
    pair_indices : np.ndarray, dtype=int, shape=(N, 2)
        Pairs of atom indices that pass your criteria.

    Notes
    -----
    This function has been optimized for speed.  A naive implementation
    can be slow (~minutes) for large proteins.
    """
    if keep_atoms is None:
        keep_atoms = ATOM_NAMES

    top, bonds = reference_traj.top.to_dataframe()

    if keep_atoms is not None:
        atom_indices = top[top.name.isin(keep_atoms) == True].index.values

    if exclude_atoms is not None:
        atom_indices = top[top.name.isin(exclude_atoms) == False].index.values

    pair_indices = np.array(list(itertools.combinations(atom_indices, 2)))

    if reject_bonded:
        a_list = bonds.min(1)
        b_list = bonds.max(1)

        n = atom_indices.max() + 1

        bond_hashes = a_list + b_list * n
        pair_hashes = pair_indices[:, 0] + pair_indices[:, 1] * n

        not_bonds = ~np.in1d(pair_hashes, bond_hashes)

        pair_indices = np.array([(a, b) for k, (a, b)
                                 in enumerate(pair_indices)
                                 if not_bonds[k]])

    return atom_indices, pair_indices
