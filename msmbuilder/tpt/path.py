# Author(s): TJ Lane (tjlane@stanford.edu) and Christian Schwantes
#            (schwancr@stanford.edu)
# Contributors: Vince Voelz, Kyle Beauchamp, Robert McGibbon
# Copyright (c) 2014, Stanford University
# All rights reserved.

"""
Functions for enumerating paths through an MSM for a given
set of sink and source states.

These are the canonical references for TPT. Note that TPT
is really a specialization of ideas very familiar to the
mathematical study of Markov chains, and there are many
books, manuscripts in the mathematical literature that
cover the same concepts.

In addition, the original paper from Dijkstra describing the
path finding algorithm we employ is listed below.

References
----------
.. [1] Weinan, E. and Vanden-Eijnden, E. Towards a theory of
       transition paths. J. Stat. Phys. 123, 503-523 (2006).
.. [2] Metzner, P., Schutte, C. & Vanden-Eijnden, E.
       Transition path theory for Markov jump processes.
       Multiscale Model. Simul. 7, 1192-1219 (2009).
.. [3] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive
       flux and folding pathways in network models of
       coarse-grained protein dynamics. J. Chem. Phys.
       130, 205102 (2009).
.. [4] Dijkstra, E. W. A Note on Two Problems in Connexion with Graphs.
       Numeriche Mathematik 1, 269-271 (1959).
.. [5] Noe, Frank, et al. "Constructing the equilibrium ensemble of folding
       pathways from short off-equilibrium simulations." PNAS 106.45 (2009):
       19011-19016.
"""
from __future__ import print_function, division, absolute_import
import numpy as np
import copy

__all__ = ['paths', 'top_path']


def top_path(sources, sinks, net_flux):
    """
    Use the Dijkstra algorithm for finding the shortest path
    connecting a set of source states from a set of sink states.

    Parameters
    ----------
    sources : array_like, int
        One-dimensional list of nodes to define the source states.
    sinks : array_like, int
        One-dimensional list of nodes to define the sink states.
    net_flux : np.ndarray, shape = [n_states, n_states]
        Net flux of the MSM

    Returns
    -------
    top_path : np.ndarray
        Array corresponding to the top path between sources and
        sinks. It is an array of states visited along the path.
    flux : float
        Flux traveling through this path -- this is equal to the
        minimum flux over edges in the path.

    See Also
    --------
    msmbuilder.tpt.paths : function for calculating many high
        flux paths through a network.

    References
    ----------
    .. [1] Weinan, E. and Vanden-Eijnden, E. Towards a theory of
           transition paths. J. Stat. Phys. 123, 503-523 (2006).
    .. [2] Metzner, P., Schutte, C. & Vanden-Eijnden, E.
           Transition path theory for Markov jump processes.
           Multiscale Model. Simul. 7, 1192-1219 (2009).
    .. [3] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive
           flux and folding pathways in network models of
           coarse-grained protein dynamics. J. Chem. Phys.
           130, 205102 (2009).
    .. [4] Dijkstra, E. W. A Note on Two Problems in Connexion with Graphs.
           Numeriche Mathematik 1, 269-271 (1959).
    .. [5] Noe, Frank, et al. "Constructing the equilibrium ensemble of folding
           pathways from short off-equilibrium simulations." PNAS 106.45 (2009):
           19011-19016.
    """
    sources = np.array(sources, dtype=np.int).reshape((-1,))
    sinks = np.array(sinks, dtype=np.int).reshape((-1,))

    n_states = net_flux.shape[0]

    queue = list(sources)
    # nodes to check (the "queue")
    # going to use list.pop method so I can't keep it as an array

    visited = np.zeros(n_states).astype(np.bool)
    # have we already checked this node?

    previous_node = np.ones(n_states).astype(np.int) * -1
    # what node was found before finding this one

    min_fluxes = np.ones(n_states) * -1 * np.inf
    # what is the flux of the highest flux path
    # from this node to the source set.

    min_fluxes[sources] = np.inf
    # source states are connected to the source
    # so this distance is zero which means the flux is infinite

    while len(queue) > 0: # iterate until there's nothing to check anymore

        test_node = queue.pop(min_fluxes[queue].argmax())
        # find the node in the queue that has the
        # highest flux path to it from the source set

        visited[test_node] = True

        if np.all(visited[sinks]):
            # if we've visited all of the sink states, then we just have to choose
            # the path that goes to the sink state that is closest to the source
            break

        # if test_node in sinks: # I *think* we want to break ... or are there paths we still
        # need to check?
        # continue
        # I think if sinks is more than one state we have to check everything

        # now update the distances for each neighbor of the test_node:
        neighbors = np.where(net_flux[test_node, :] > 0)[0]
        if len(neighbors) == 0:
            continue

        new_fluxes = net_flux[test_node, neighbors].flatten()
        # flux from test_node to each neighbor

        new_fluxes[np.where(new_fluxes > min_fluxes[test_node])] = min_fluxes[test_node]
        # previous step to get to test_node was lower flux, so that is still the path flux

        ind = np.where((1 - visited[neighbors]) & (new_fluxes > min_fluxes[neighbors]))
        min_fluxes[neighbors[ind]] = new_fluxes[ind]

        previous_node[neighbors[ind]] = test_node
        # each of these neighbors came from this test_node
        # we don't want to update the nodes that have already been visited

        queue.extend(neighbors[ind])

    top_path = []
    # populate the path in reverse
    top_path.append(int(sinks[min_fluxes[sinks].argmax()]))
    # find the closest sink state

    while previous_node[top_path[-1]] != -1:
        top_path.append(previous_node[top_path[-1]])

    return np.array(top_path[::-1]), min_fluxes[top_path[0]]


def _remove_bottleneck(net_flux, path):
    """
    Internal function for modifying the net flux matrix by removing
    a particular edge, corresponding to the bottleneck of a particular
    path.
    """
    net_flux = copy.copy(net_flux)

    bottleneck_ind = net_flux[path[:-1], path[1:]].argmin()

    net_flux[path[bottleneck_ind], path[bottleneck_ind + 1]] = 0.0

    return net_flux


def _subtract_path_flux(net_flux, path):
    """
    Internal function for modifying the net flux matrix by subtracting
    a path's flux from every edge in the path.
    """

    net_flux = copy.copy(net_flux)

    net_flux[path[:-1], path[1:]] -= net_flux[path[:-1], path[1:]].min()

    # The above *should* make the bottleneck have zero flux, but
    # numerically that may not be the case, so just set it to zero
    # to be sure.
    bottleneck_ind = net_flux[path[:-1], path[1:]].argmin()
    net_flux[path[bottleneck_ind], path[bottleneck_ind + 1]] = 0.0

    return net_flux


def paths(sources, sinks, net_flux, remove_path='subtract',
          num_paths=np.inf, flux_cutoff=(1-1E-10)):
    """
    Get the top N paths by iteratively performing Dijkstra's
    algorithm.

    Parameters
    ----------
    sources : array_like, int
        One-dimensional list of nodes to define the source states.
    sinks : array_like, int
        One-dimensional list of nodes to define the sink states.
    net_flux : np.ndarray
        Net flux of the MSM
    remove_path : str or callable, optional
        Function for removing a path from the net flux matrix.
        (if str, one of {'subtract', 'bottleneck'})
        See note below for more details.
    num_paths : int, optional
        Number of paths to find
    flux_cutoff : float, optional
        Quit looking for paths once the explained flux is greater
        than this cutoff (as a percentage of the total).

    Returns
    -------
    paths : list
        List of paths. Each item is an array of nodes visited
        in the path.
    fluxes : np.ndarray, shape = [n_paths,]
        Flux of each path returned.

    Notes
    -----
    The Dijkstra algorithm only allows for computing the
    *single* top flux pathway through the net flux matrix. If
    we want many paths, there are many ways of finding the
    *second* highest flux pathway.

    The algorithm proceeds as follows:

    1. Using the Djikstra algorithm, find the highest flux
       pathway from the sources to the sink states
    2. Remove that pathway from the net flux matrix by
       some criterion
    3. Repeat (1) with the modified net flux matrix

    Currently, there are two schemes for step (2):

    - 'subtract' : Remove the path by subtracting the flux
      of the path from every edge in the path. This was
      suggested by Metzner, Schutte, and Vanden-Eijnden.
      Transition Path Theory for Markov Jump Processes.
      Multiscale Model. Simul. 7, 1192-1219 (2009).
    - 'bottleneck' : Remove the path by only removing
      the edge that corresponds to the bottleneck of the
      path.

    If a new scheme is desired, the user may pass a function
    that takes the net_flux and the path to remove and returns
    the new net flux matrix.

    See Also
    --------
    msmbuilder.tpt.top_path : function for computing the single
        highest flux pathway through a network.

    References
    ----------
    .. [1] Weinan, E. and Vanden-Eijnden, E. Towards a theory of
           transition paths. J. Stat. Phys. 123, 503-523 (2006).
    .. [2] Metzner, P., Schutte, C. & Vanden-Eijnden, E.
           Transition path theory for Markov jump processes.
           Multiscale Model. Simul. 7, 1192-1219 (2009).
    .. [3] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive
           flux and folding pathways in network models of
           coarse-grained protein dynamics. J. Chem. Phys.
           130, 205102 (2009).
    .. [4] Dijkstra, E. W. A Note on Two Problems in Connexion with Graphs.
           Numeriche Mathematik 1, 269-271 (1959).
    .. [5] Noe, Frank, et al. "Constructing the equilibrium ensemble of folding
           pathways from short off-equilibrium simulations." PNAS 106.45 (2009):
           19011-19016.
    """

    if not callable(remove_path):
        if remove_path == 'subtract':
            remove_path = _subtract_path_flux
        elif remove_path == 'bottleneck':
            remove_path = _remove_bottleneck
        else:
            raise ValueError("remove_path_func (%s) must be a callable or one of ['subtract', 'bottleneck']" % str(remove_path))

    net_flux = copy.copy(net_flux)

    paths = []
    fluxes = []

    total_flux = net_flux[sources, :].sum()
    # total flux is the total flux coming from the sources (or going into the sinks)

    not_done = True
    counter = 0
    expl_flux = 0.0
    while not_done:
        path, flux = top_path(sources, sinks, net_flux)
        if np.isinf(flux):
            break

        paths.append(path)
        fluxes.append(flux)

        expl_flux += flux / total_flux
        counter += 1

        if counter >= num_paths or expl_flux >= flux_cutoff:
            break

        # modify the net_flux matrix
        net_flux = remove_path(net_flux, path)

    fluxes = np.array(fluxes)

    return paths, fluxes
