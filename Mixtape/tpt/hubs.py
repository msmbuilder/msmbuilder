# This file is part of MSMBuilder.
#
# Copyright 2011 Stanford University
#
# MSMBuilder is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""
Functions for performing Transition Path Theory calculations. 

Written and maintained by TJ Lane <tjlane@stanford.edu>
Contributions from Kyle Beauchamp, Robert McGibbon, Vince Voelz,
Christian Schwantes.

These are the cannonical references for TPT. Note that TPT is really a
specialization of ideas very framiliar to the mathematical study of Markov
chains, and there are many books, manuscripts in the mathematical literature
that cover the same concepts.

References
----------
.. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory
       for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219
       (2009).
.. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
       pathways in network models of coarse-grained protein dynamics. J. 
       Chem. Phys. 130, 205102 (2009).
"""
from __future__ import print_function, division, absolute_import
import numpy as np
import scipy.sparse

from msmbuilder import MSMLib
from msmbuilder import msm_analysis
from msmbuilder.utils import deprecated
import itertools
import copy

import logging
logger = logging.getLogger(__name__)

# turn on debugging printout
# logger.setLogLevel(logging.DEBUG)

###############################################################################
# Typechecking/Utility Functions
#


def _ensure_iterable(arg):
    if not hasattr(arg, '__iter__'):
        arg = list([int(arg)])
        logger.debug("Passed object was not iterable,"
                     " converted it to: %s" % str(arg))
    assert hasattr(arg, '__iter__')
    return arg

def _check_sources_sinks(sources, sinks):
    sources = _ensure_iterable(sources)
    sinks = _ensure_iterable(sinks)

    for s in sources:
        if s in sinks:
            raise ValueError("sources and sinks are not disjoint")

    return sources, sinks


###############################################################################
# Path Finding Functions
#

def get_top_path_scipy(sources, sinks, net_flux):
    """
    Use the Dijkstra algorithm for finding the shortest path connecting
    sources and sinks

    Parameters
    ----------
    sources : array_like
        nodes to define the source states
    sinks : array_like
        nodes to define the sink states
    net_flux : scipy.sparse matrix
        net flux of the MSM

    Returns
    -------
    top_path : np.ndarray
        array corresponding to the top path between sources and sinks
    
    """

    _check_sources_sinks(sources, sinks)

    if not scipy.sparse.issparse(net_flux):
        logger.warn("only sparse matrices are currently supported.")
        net_flux = scipy.sparse.lil_matrix(net_flux)

    net_flux = net_flux.tolil()
    n_states = net_flux.shape[0]

    sources = np.array(sources, dtype=np.int).flatten()
    sinks = np.array(sinks, dtype=np.int).flatten()

    Q = list(sources) # nodes to check (the "queue")
                      # going to use list.pop method so I can't keep it as an array

def get_top_path(sources, sinks, net_flux):
    """
    Use the Dijkstra algorithm for finding the shortest path connecting
    sources and sinks

    Parameters
    ----------
    sources : array_like
        nodes to define the source states
    sinks : array_like
        nodes to define the sink states
    net_flux : scipy.sparse matrix
        net flux of the MSM

    Returns
    -------
    top_path : np.ndarray
        array corresponding to the top path between sources and sinks
    
    """

    _check_sources_sinks(sources, sinks)

    if not scipy.sparse.issparse(net_flux):
        logger.warn("only sparse matrices are currently supported.")
        net_flux = scipy.sparse.lil_matrix(net_flux)

    net_flux = net_flux.tolil()
    n_states = net_flux.shape[0]

    sources = np.array(sources, dtype=np.int).flatten()
    sinks = np.array(sinks, dtype=np.int).flatten()

    Q = list(sources) # nodes to check (the "queue")
                      # going to use list.pop method so I can't keep it as an array
    visited = np.zeros(n_states).astype(np.bool) # have we already checked this node?
    previous_node = np.ones(n_states).astype(np.int) * -1 # what node was found before finding this one
    min_fluxes = np.ones(n_states) * -1 * np.inf # what is the flux of the highest flux path
                                            # from this node to the source set.

    min_fluxes[sources] = np.inf # source states are connected to the source
                                 # so this distance is zero which means the flux is infinite
    
    while len(Q) > 0: # iterate until there's nothing to check anymore

        test_node = Q.pop(min_fluxes[Q].argmax()) # find the node in the queue that has the 
                                                  # highest flux path to it from the source set
        visited[test_node] = True

        if np.all(visited[sinks]):
            # if we've visited all of the sink states, then we just have to choose
            # the path that goes to the sink state that is closest to the source
            break

        #if test_node in sinks: # I *think* we want to break ... or are there paths we still 
        #                       # need to check?
        #    continue # I think if sinks is more than one state we have to check everything

        # now update the distances for each neighbor of the test_node:
        neighbors = np.where(net_flux[test_node, :].toarray() > 0)[1]
        if len(neighbors) == 0:
            continue
        new_fluxes = net_flux[test_node, neighbors].toarray().flatten()
            # flux from test_node to each neighbor

        new_fluxes[np.where(new_fluxes > min_fluxes[test_node])] = min_fluxes[test_node]
            # previous step to get to test_node was lower flux, so that is still the path flux

        ind = np.where((1 - visited[neighbors]) & (new_fluxes > min_fluxes[neighbors]))
        min_fluxes[neighbors[ind]] = new_fluxes[ind]

        previous_node[neighbors[ind]] = test_node # each of these neighbors came from this test_node
        # we don't want to update the nodes that have already been visited

        Q.extend(neighbors[ind])

    top_path = []
    # populate the path in reverse
    top_path.append(int(sinks[min_fluxes[sinks].argmax()]))
    # find the closest sink state

    while previous_node[top_path[-1]] != -1:
        top_path.append(previous_node[top_path[-1]])

    return np.array(top_path[::-1]), min_fluxes[top_path[0]]


def get_paths(sources, sinks, net_flux, num_paths=np.inf, flux_cutoff=(1-1E-10)):
    """
    Get the top N paths by iteratively performing Dijkstra's
    algorithm, but at each step modifying the net flux matrix
    by subtracting the previously found path's flux from each
    edge in that path.
    
    Parameters
    ----------
    sources : array_like
        nodes to define the source states
    sinks : array_like
        nodes to define the sink states
    net_flux : scipy.sparse matrix
        net flux of the MSM
    num_paths : int, optional
        number of paths to find
    flux_cutoff : float, optional
        quit finding paths once the explained flux is greater
        this cutoff (as a percentage of the total)
    
    Returns
    -------
    paths : np.ndarray
        list of paths
    fluxes : np.ndarray
        flux of each path returned
    """
    
    _check_sources_sinks(sources, sinks)
    
    if not scipy.sparse.issparse(net_flux):
        logger.warn("only sparse matrices are currently supported")
        net_flux = scipy.sparse.lil_matrix(net_flux)

    net_flux = copy.deepcopy(net_flux.tolil())
    
    paths = []
    fluxes = []
    
    total_flux = net_flux[sources, :].sum()
    # total flux is the total flux coming from the sources (or going into the sinks)
    
    not_done = True
    counter = 0
    expl_flux = 0.0
    while not_done:
        path, flux = get_top_path(sources, sinks, net_flux)
        if np.isinf(flux):
            logger.info("no more paths exist")
            break
        
        paths.append(path)
        fluxes.append(flux)
        
        expl_flux += flux / total_flux
        
        counter += 1
        logger.info("Found next path (%d) with flux %.4e (%.2f%% of total)" % (counter, flux, expl_flux * 100))
        
        if counter >= num_paths or expl_flux >= flux_cutoff:
            break

        # modify the net_flux matrix
        for k in xrange(len(path) - 1):
            net_flux[path[k], path[k+1]] -= flux
        # since flux is the bottleneck, this will never be negative
        # though... this might lead to CLOS which is bad...
        # I'll fix this (^^^^) later.. for now let's get it working

    
    max_len = np.max([len(p) for p in paths])
    temp = np.ones((len(paths), max_len)) * -1
    for i in range(len(paths)):
        temp[i][:len(paths[i])] = paths[i]
    
    fluxes = np.array(fluxes)
    
    return temp, fluxes


def _get_enumerated_paths(sources, sinks, net_flux, num_paths=1):
    """
    get all possible paths, sorted from highest to lowest flux

    Parameters
    ----------
    sources : array_like
        nodes to define the source states
    sinks : array_like
        nodes to define the sink states
    net_flux : scipy.sparse matrix
        net flux of the MSM

    Returns
    -------
    paths : np.ndarray
        list of paths
    fluxes : np.ndarray
        flux of each path returned
    """

    paths = []
    edges = []
    fluxes = []

    temp_net_flux = copy.deepcopy(net_flux).tolil()

    path, flux = get_top_path(sources, sinks, temp_net_flux)
    paths.append(path)
    edges.append([[path[i], path[i + 1]] for i in xrange(len(path) - 1)])
    fluxes.append(flux)

    for i in xrange(1, num_paths):
        #print "found %s - %.4e" % (str(paths[-1]), fluxes[-1])
        test_path = None
        test_flux = - np.inf

        edge_list = itertools.product(*edges)
        # this is a list of sets of edges to remove from net flux

        for edges_to_remove in edge_list:
            temp_net_flux = copy.deepcopy(net_flux).tolil()
            for edge in edges_to_remove:
                temp_net_flux[edge[0], edge[1]] = 0.0

            path, flux = get_top_path(sources, sinks, temp_net_flux)
            if flux > test_flux:
                test_path = path
                test_flux = flux

        paths.append(test_path)
        edges.append([[test_path[i], test_path[i + 1]] for i in xrange(len(test_path) - 1)])
        fluxes.append(test_flux)
            
    max_len = np.max([len(p) for p in paths])
    temp = np.ones((len(paths), max_len)) * -1
    for i in range(len(paths)):
        temp[i][:len(paths[i])] = paths[i]

    fluxes = np.array(fluxes)

    return temp, fluxes


def _find_top_paths_cut(sources, sinks, tprob, num_paths=10, node_wipe=False, net_flux=None):
    r"""
    Calls the Dijkstra algorithm to find the top 'NumPaths'.

    Does this recursively by first finding the top flux path, then cutting that
    path and relaxing to find the second top path. Continues until NumPaths
    have been found.

    Parameters
    ----------
    sources : array_like, int
        The indices of the source states
    sinks : array_like, int
        Indices of sink states
    num_paths : int
        The number of paths to find

    Returns
    -------
    Paths : list of lists
        The nodes transversed in each path
    Bottlenecks : list of tuples
        The nodes between which exists the path bottleneck
    Fluxes : list of floats
        The flux through each path

    Optional Parameters
    -------------------
    node_wipe : bool
        If true, removes the bottleneck-generating node from the graph, instead
        of just the bottleneck (not recommended, a debugging functionality)
    net_flux : sparse matrix
        Matrix of the net flux from `sources` to `sinks`, see function `net_flux`.
        If not provided, is calculated from scratch. If provided, `tprob` is
        ignored.

    To Do
    -----
    -- Add periodic flow check

    References
    ----------
    .. [1] Dijkstra, E. W. (1959). "A note on two problems in connexion with 
           graphs." Numerische Mathematik 1: 269-271. doi:10.1007/BF01386390.
    """

    # first, do some checking on the input, esp. `sources` and `sinks`
    # we want to make sure all objects are iterable and the sets are disjoint
    sources, sinks = _check_sources_sinks(sources, sinks)
    msm_analysis.check_transition(tprob)

    # check to see if we get net_flux for free, otherwise calculate it
    if not net_flux:
        net_flux = calculate_net_fluxes(sources, sinks, tprob)

    # initialize objects
    paths = []
    fluxes = []
    bottlenecks = []

    if scipy.sparse.issparse(net_flux):
        net_flux = net_flux.tolil()

    # run the initial Dijkstra pass
    pi, b = _Dijkstra(sources, sinks, net_flux)

    logger.info("Path Num | Path | Bottleneck | Flux")

    i = 1
    done = False
    while not done:

        # First find the highest flux pathway
        (path, (b1, b2), flux) = _backtrack(sinks, b, pi, net_flux)

        # Add each result to a Paths, Bottlenecks, Fluxes list
        if flux == 0:
            logger.info("Only %d possible pathways found. Stopping backtrack.", i)
            break
        paths.append(path)
        bottlenecks.append((b1, b2))
        fluxes.append(flux)
        logger.info("%s | %s | %s | %s ", i, path, (b1, b2), flux)

        # Cut the bottleneck, start relaxing from B side of the cut
        if node_wipe:
            net_flux[:, b2] = 0
            logger.info("Wiped node: %s", b2)
        else:
            net_flux[b1, b2] = 0

        G = scipy.sparse.find(net_flux)
        Q = [b2]
        b, pi, net_flux = _back_relax(b2, b, pi, net_flux)

        # Then relax the graph and repeat
        # But only if we still need to
        if i != num_paths - 1:
            while len(Q) > 0:
                w = Q.pop()
                for v in G[1][np.where(G[0] == w)]:
                    if pi[v] == w:
                        b, pi, net_flux = _back_relax(v, b, pi, net_flux)
                        Q.append(v)
                Q = sorted(Q, key=lambda v: b[v])

        i += 1
        if i == num_paths + 1:
            done = True
        if flux == 0:
            logger.info("Only %d possible pathways found. Stopping backtrack.", i)
            done = True

    return paths, bottlenecks, fluxes


def _Dijkstra(sources, sinks, net_flux):
    r""" A modified Dijkstra algorithm that dynamically computes the cost
    of all paths from A to B, weighted by NFlux.

    Parameters
    ----------
    sources : array_like, int
        The indices of the source states (i.e. for state A in rxn A -> B)
    sinks : array_like, int
        Indices of sink states (state B)
    NFlux : sparse matrix
        Matrix of the net flux from A to B, see function GetFlux

    Returns
    -------
    pi : array_like
        The paths from A->B, pi[i] = node preceeding i
    b : array_like
        The flux passing through each node

    See Also
    --------
    DijkstraTopPaths : child function
        `DijkstraTopPaths` is probably the function you want to call to find
         paths through an MSM network. This is a utility function called by
         `DijkstraTopPaths`, but may be useful in some specific cases

     References
     ----------
     .. [1] Dijkstra, E. W. (1959). "A note on two problems in connexion with 
            graphs." Numerische Mathematik 1: 269-271. doi:10.1007/BF01386390.
    """

    sources, sinks = _check_sources_sinks(sources, sinks)

    # initialize data structures
    if scipy.sparse.issparse(net_flux):
        net_flux = net_flux.tolil()
    else:
        net_flux = scipy.sparse.lil_matrix(net_flux)

    G = scipy.sparse.find(net_flux)
    N = net_flux.shape[0]
    b = np.zeros(N)
    b[sources] = 1000
    pi = np.zeros(N, dtype=int)
    pi[sources] = -1
    U = []

    Q = sorted(list(range(N)), key=lambda v: b[v])
    for v in sinks:
        Q.remove(v)

    # run the Dijkstra algorithm
    while len(Q) > 0:
        w = Q.pop()
        U.append(w)

        # relax
        for v in G[1][np.where(G[0] == w)]:
            if b[v] < min(b[w], net_flux[w, v]):
                b[v] = min(b[w], net_flux[w, v])
                pi[v] = w

        Q = sorted(Q, key=lambda v: b[v])

    logger.info("Searched %s nodes", len(U) + len(sinks))

    return pi, b


def _back_relax(s, b, pi, NFlux):
    r"""
    Updates a Djikstra calculation once a bottleneck is cut, quickly
    recalculating only cost of nodes that change due to the cut.

    Cuts & relaxes the B-side (sink side) of a cut edge (b2) to source from the
    adjacent node with the most flux flowing to it. If there are no
    adjacent source nodes, cuts the node out of the graph and relaxes the
    nodes that were getting fed by b2 (the cut node).

    Parameters
    ----------
    s : int
        the node b2
    b : array_like
        the cost function
    pi : array_like
        the backtrack array, a list such that pi[i] = source node of node i
    NFlux : sparse matrix
        Net flux matrix

    Returns
    -------
    b : array_like
        updated cost function
    pi : array_like
        updated backtrack array
    NFlux : sparse matrix
        net flux matrix

    See Also
    --------
    DijkstraTopPaths : child function
        `DijkstraTopPaths` is probably the function you want to call to find
         paths through an MSM network. This is a utility function called by
         `DijkstraTopPaths`, but may be useful in some specific cases
    """

    G = scipy.sparse.find(NFlux)
    if len(G[0][np.where(G[1] == s)]) > 0:

        # For all nodes connected upstream to the node `s` in question,
        # Re-source that node from the best option (lowest cost) one level lower
        # Notation: j is node one level below, s is the one being considered

        b[s] = 0                                 # set the cost to zero
        for j in G[0][np.where(G[1] == s)]:    # for each upstream node
            if b[s] < min(b[j], NFlux[j, s]):   # if that node has a lower cost
                b[s] = min(b[j], NFlux[j, s])   # then set the cost to that node
                pi[s] = j                        # and the source comes from there

    # if there are no nodes connected to this one, then we need to go one
    # level up and work there first
    else:
        for sprime in G[1][np.where(G[0] == s)]:
            NFlux[s, sprime] = 0
            b, pi, NFlux = _back_relax(sprime, b, pi, NFlux)

    return b, pi, NFlux


def _backtrack(B, b, pi, NFlux):
    """
    Works backwards to pull out a path from pi, where pi is a list such that
    pi[i] = source node of node i. Begins at the largest staring incoming flux
    point in B.

    Parameters
    ----------
    B : array_like, int
        Indices of sink states (state B)
    b : array_like
        the cost function
    pi : array_like
        the backtrack array, a list such that pi[i] = source node of node i
    NFlux : sparse matrix
        net flux matrix

    Returns
    -------
    bestpath : list
        the list of nodes forming the highest flux path
    bottleneck : tuple
        a tupe of nodes, between which is the bottleneck
    bestflux : float
        the flux through `bestpath`

    See Also
    --------
    DijkstraTopPaths : child function
        `DijkstraTopPaths` is probably the function you want to call to find
        paths through an MSM network. This is a utility function called by
        `DijkstraTopPaths`, but may be useful in some specific cases
    """

    # Select starting location
    bestflux = 0
    for Bnode in B:
        path = [Bnode]
        NotDone = True
        while NotDone:
            if pi[path[-1]] == -1:
                break
            else:
                path.append(pi[path[-1]])
        path.reverse()

        bottleneck, flux = find_path_bottleneck(path, NFlux)

        logger.debug('In Backtrack: Flux %s, bestflux %s', flux, bestflux)

        if flux > bestflux:
            bestpath = path
            bestbottleneck = bottleneck
            bestflux = flux

    if flux == 0:
        bestpath = []
        bottleneck = (np.nan, np.nan)
        bestflux = 0

    return (bestpath, bestbottleneck, bestflux)


def find_path_bottleneck(path, net_flux):
    """
    Simply finds the bottleneck along a path.

    This is the point at which the cost function first goes up along the path,
    backtracking from B to A.

    Parameters
    ----------
    path : list
        a list of nodes along the path of interest
    net_flux : matrix
        the net flux matrix

    Returns
    -------
    bottleneck : tuple
        a tuple of the nodes on either end of the bottleneck
    flux : float
        the flux at the bottleneck

    See Also
    --------
    find_top_paths : child function
        `find_top_paths` is probably the function you want to call to find
         paths through an MSM network. This is a utility function called by
         `find_top_paths`, but may be useful in some specific cases.

     References
     ----------
     .. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
            for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219
            (2009).
     .. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
            pathways in network models of coarse-grained protein dynamics. J. 
            Chem. Phys. 130, 205102 (2009).
    """

    if scipy.sparse.issparse(net_flux):
        net_flux = net_flux.tolil()

    flux = 100000.  # initialize as large value

    for i in range(len(path) - 1):
        if net_flux[path[i], path[i + 1]] < flux:
            flux = net_flux[path[i], path[i + 1]]
            b1 = path[i]
            b2 = path[i + 1]

    return (b1, b2), flux


def calculate_fluxes(sources, sinks, tprob, populations=None, committors=None):
    """
    Compute the transition path theory flux matrix.

    Parameters
    ----------
    sources : array_like, int
        The set of unfolded/reactant states.
    sinks : array_like, int
        The set of folded/product states.
    tprob : mm_matrix
        The transition matrix.

    Returns
    ------
    fluxes : mm_matrix
        The flux matrix

    Optional Parameters
    -------------------
    populations : nd_array, float
        The equilibrium populations, if not provided is re-calculated
    committors : nd_array, float
        The committors associated with `sources`, `sinks`, and `tprob`.
        If not provided, is calculated from scratch. If provided, `sources`
        and `sinks` are ignored.

    References
    ----------
    .. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
           for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219
           (2009).
    .. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
           pathways in network models of coarse-grained protein dynamics. J. 
           Chem. Phys. 130, 205102 (2009).
    """

    sources, sinks = _check_sources_sinks(sources, sinks)
    msm_analysis.check_transition(tprob)

    if scipy.sparse.issparse(tprob):
        dense = False
    else:
        dense = True

    # check if we got the populations
    if populations is None:
        eigens = msm_analysis.get_eigenvectors(tprob, 5)
        if np.count_nonzero(np.imag(eigens[1][:, 0])) != 0:
            raise ValueError('First eigenvector has imaginary components')
        populations = np.real(eigens[1][:, 0])

    # check if we got the committors
    if committors is None:
        committors = calculate_committors(sources, sinks, tprob)

    # perform the flux computation
    Indx, Indy = tprob.nonzero()

    n = tprob.shape[0]

    if dense:
        X = np.zeros((n, n))
        Y = np.zeros((n, n))
        X[(np.arange(n), np.arange(n))] = populations * (1.0 - committors)
        Y[(np.arange(n), np.arange(n))] = committors
    else:
        X = scipy.sparse.lil_matrix((n, n))
        Y = scipy.sparse.lil_matrix((n, n))
        X.setdiag(populations * (1.0 - committors))
        Y.setdiag(committors)

    if dense:
        fluxes = np.dot(np.dot(X, tprob), Y)
        fluxes[(np.arange(n), np.arange(n))] = np.zeros(n)
    else:
        fluxes = (X.tocsr().dot(tprob.tocsr())).dot(Y.tocsr())
        # This should be the same as below, but it's a bit messy...
        #fluxes = np.dot(np.dot(X.tocsr(), tprob.tocsr()), Y.tocsr())
        fluxes = fluxes.tolil()
        fluxes.setdiag(np.zeros(n))

    return fluxes


def calculate_net_fluxes(sources, sinks, tprob, populations=None, committors=None):
    """
    Computes the transition path theory net flux matrix.

    Parameters
    ----------
    sources : array_like, int
        The set of unfolded/reactant states.
    sinks : array_like, int
        The set of folded/product states.
    tprob : mm_matrix
        The transition matrix.

    Returns
    ------
    net_fluxes : mm_matrix
        The net flux matrix

    Optional Parameters
    -------------------
    populations : nd_array, float
        The equilibrium populations, if not provided is re-calculated
    committors : nd_array, float
        The committors associated with `sources`, `sinks`, and `tprob`.
        If not provided, is calculated from scratch. If provided, `sources`
        and `sinks` are ignored.

    References
    ----------
    .. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
           for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219
           (2009).
    .. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
           pathways in network models of coarse-grained protein dynamics. J. 
           Chem. Phys. 130, 205102 (2009).
    """

    sources, sinks = _check_sources_sinks(sources, sinks)
    msm_analysis.check_transition(tprob)

    if scipy.sparse.issparse(tprob):
        dense = False
    else:
        dense = True

    n = tprob.shape[0]

    flux = calculate_fluxes(sources, sinks, tprob, populations, committors)
    ind = flux.nonzero()

    if dense:
        net_flux = np.zeros((n, n))
    else:
        net_flux = scipy.sparse.lil_matrix((n, n))

    for k in range(len(ind[0])):
        i, j = ind[0][k], ind[1][k]
        forward = flux[i, j]
        reverse = flux[j, i]
        net_flux[i, j] = max(0, forward - reverse)

    return net_flux


###############################################################################
# MFPT & Committor Finding Functions
#

def calculate_ensemble_mfpt(sources, sinks, tprob, lag_time):
    """
    Calculates the average 'Folding Time' of an MSM defined by T and a LagTime.
    The Folding Time is the average of the MFPTs (to F) of all the states in U.

    Note here 'Folding Time' is defined as the avg MFPT of {U}, to {F}.
    Consider this carefully. This is probably NOT the experimental folding time!

    Parameters
    ----------
    sources : array, int
        indices of the source states
    sinks : array, int
        indices of the sink states
    tprob : matrix
        transition probability matrix
    lag_time : float
        the lag time used to create T (dictates units of the answer)

    Returns
    -------
    avg : float
        the average of the MFPTs
    std : float
        the standard deviation of the MFPTs

    References
    ----------
    .. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
           for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219
           (2009).
    .. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
           pathways in network models of coarse-grained protein dynamics. J. 
           Chem. Phys. 130, 205102 (2009).
    """

    sources, sinks = _check_sources_sinks(sources, sinks)
    msm_analysis.check_transition(tprob)

    X = calculate_mfpt(sinks, tprob, lag_time)
    times = np.zeros(len(sources))
    for i in range(len(sources)):
        times[i] = X[sources[i]]

    return np.average(times), np.std(times)


def calculate_avg_TP_time(sources, sinks, tprob, lag_time):
    """
    Calculates the Average Transition Path Time for MSM with: T, LagTime.
    The TPTime is the average of the MFPTs (to F) of all the states
    immediately adjacent to U, with the U states effectively deleted.

    Note here 'TP Time' is defined as the avg MFPT of all adjacent states to {U},
    to {F}, ignoring {U}.

    Consider this carefully.

    Parameters
    ----------
    sources : array, int
        indices of the unfolded states
    sinks : array, int
        indices of the folded states
    tprob : matrix
        transition probability matrix
    lag_time : float
        the lag time used to create T (dictates units of the answer)

    Returns
    -------
    avg : float
        the average of the MFPTs
    std : float
        the standard deviation of the MFPTs

    References
    ----------
    .. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
           for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219 
           (2009).
    .. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
           pathways in network models of coarse-grained protein dynamics. J. 
           Chem. Phys. 130, 205102 (2009).
    """

    sources, sinks = _check_sources_sinks(sources, sinks)
    msm_analysis.check_transition(tprob)

    n = tprob.shape[0]
    if scipy.sparse.issparse(tprob):
        T = tprob.tolil()
        P = scipy.sparse.lil_matrix((n, n))
    else:
        P = np.zeros((n, n))

    for u in sources:
        for i in range(n):
            if i not in sources:
                P[u, i] = T[u, i]

    for u in sources:
        T[u, :] = np.zeros(n)
        T[:, u] = 0

    for i in sources:
        N = T[i, :].sum()
        T[i, :] = T[i, :] / N

    X = calculate_mfpt(sinks, tprob, lag_time)
    TP = P * X.T
    TPtimes = []

    for time in TP:
        if time != 0:
            TPtimes.append(time)

    return np.average(TPtimes), np.std(TPtimes)


def calculate_mfpt(sinks, tprob, lag_time=1.):
    """
    Gets the Mean First Passage Time (MFPT) for all states to a *set*
    of sinks.

    Parameters
    ----------
    sinks : array, int
        indices of the sink states
    tprob : matrix
        transition probability matrix
    LagTime : float
        the lag time used to create T (dictates units of the answer)

    Returns
    -------
    MFPT : array, float
        MFPT in time units of LagTime, for each state (in order of state index)

    See Also
    --------
    calculate_all_to_all_mfpt : function
        A more efficient way to calculate all the MFPTs in a network

    References
    ----------
    .. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
           for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219 
           (2009).
    .. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
           pathways in network models of coarse-grained protein dynamics. J. 
           Chem. Phys. 130, 205102 (2009).
    """

    sinks = _ensure_iterable(sinks)
    msm_analysis.check_transition(tprob)

    n = tprob.shape[0]

    if scipy.sparse.isspmatrix(tprob):
        tprob = tprob.tolil()

    for state in sinks:
        tprob[state, :] = 0.0
        tprob[state, state] = 2.0

    if scipy.sparse.isspmatrix(tprob):
        tprob = tprob - scipy.sparse.eye(n, n)
        tprob = tprob.tocsr()
    else:
        tprob = tprob - np.eye(n)

    RHS = -1 * np.ones(n)
    for state in sinks:
        RHS[state] = 0.0

    if scipy.sparse.isspmatrix(tprob):
        MFPT = lag_time * scipy.sparse.linalg.spsolve(tprob, RHS)
    else:
        MFPT = lag_time * np.linalg.solve(tprob, RHS)

    return MFPT


def calculate_all_to_all_mfpt(tprob, populations=None):
    """
    Calculate the all-states by all-state matrix of mean first passage
    times.

    This uses the fundamental matrix formalism, and should be much faster
    than GetMFPT for calculating many MFPTs.

    Parameters
    ----------
    tprob : matrix
        transition probability matrix
    populations : array_like, float
        optional argument, the populations of each state. If  not supplied,
        it will be computed from scratch

    Returns
    -------
    MFPT : array, float
        MFPT in time units of LagTime, square array for MFPT from i -> j

    See Also
    --------
    GetMFPT : function
        for calculating a subset of the MFPTs, with functionality for including
        a set of sinks

    References
    ----------
    .. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
           for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219 
           (2009).
    .. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
           pathways in network models of coarse-grained protein dynamics. J. 
           Chem. Phys. 130, 205102 (2009).
    """

    msm_analysis.check_transition(tprob)

    if scipy.sparse.issparse(tprob):
        tprob = tprob.toarray()
        logger.warning('calculate_all_to_all_mfpt does not support sparse linear algebra')

    if populations is None:
        eigens = msm_analysis.get_eigenvectors(tprob, 1)
        if np.count_nonzero(np.imag(eigens[1][:, 0])) != 0:
            raise ValueError('First eigenvector has imaginary parts')
        populations = np.real(eigens[1][:, 0])

    # ensure that tprob is a transition matrix
    msm_analysis.check_transition(tprob)
    num_states = len(populations)
    if tprob.shape[0] != num_states:
        raise ValueError("Shape of tprob and populations vector don't match")

    eye = np.transpose(np.matrix(np.ones(num_states)))
    limiting_matrix = eye * populations
    #z = scipy.linalg.inv(scipy.sparse.eye(num_states, num_states) - (tprob - limiting_matrix))
    z = scipy.linalg.inv(np.eye(num_states) - (tprob - limiting_matrix))

    # mfpt[i,j] = z[j,j] - z[i,j] / pi[j]
    mfpt = -z
    for j in range(num_states):
        mfpt[:, j] += z[j, j]
        mfpt[:, j] /= populations[j]

    return mfpt


def calculate_committors(sources, sinks, tprob):
    """
    Get the forward committors of the reaction sources -> sinks.

    Parameters
    ----------
    sources : array_like, int
        The set of unfolded/reactant states.
    sinks : array_like, int
        The set of folded/product states.
    tprob : mm_matrix
        The transition matrix.

    Returns
    -------
    Q : array_like
        The forward committors for the reaction U -> F.

    References
    ----------
    .. [1] Metzner, P., Schutte, C. & Vanden-Eijnden, E. Transition path theory 
           for Markov jump processes. Multiscale Model. Simul. 7, 1192-1219 
           (2009).
    .. [2] Berezhkovskii, A., Hummer, G. & Szabo, A. Reactive flux and folding 
           pathways in network models of coarse-grained protein dynamics. J. 
           Chem. Phys. 130, 205102 (2009).
    """

    sources, sinks = _check_sources_sinks(sources, sinks)
    msm_analysis.check_transition(tprob)

    if scipy.sparse.issparse(tprob):
        dense = False
        tprob = tprob.tolil()
    else:
        dense = True

    # construct the committor problem
    n = tprob.shape[0]

    if dense:
        T = np.eye(n) - tprob
    else:
        T = scipy.sparse.eye(n, n, 0, format='lil') - tprob
        T = T.tolil()

    for a in sources:
        T[a, :] = 0.0  # np.zeros(n)
        T[:, a] = 0.0
        T[a, a] = 1.0

    for b in sinks:
        T[b, :] = 0.0  # np.zeros(n)
        T[:, b] = 0.0
        T[b, b] = 1.0

    IdB = np.zeros(n)
    IdB[sinks] = 1.0

    if dense:
        RHS = np.dot(tprob, IdB)
    else:
        RHS = tprob.dot(IdB)
        # This should be the same as below
        #RHS = tprob * IdB

    RHS[sources] = 0.0
    RHS[sinks] = 1.0

    # solve for the committors
    if dense == False:
        Q = scipy.sparse.linalg.spsolve(T.tocsr(), RHS)
    else:
        Q = np.linalg.solve(T, RHS)

    epsilon = 0.001
    assert np.all(Q <= 1.0 + epsilon)
    assert np.all(Q >= 0.0 - epsilon)

    return Q


######################################################################
# Functions for computing hub scores, conditional committors, and
# related quantities
#


def calculate_fraction_visits(tprob, waypoint, source, sink, return_cond_Q=False):
    """
    Calculate the fraction of times a walker on `tprob` going from `sources`
    to `sinks` will travel through the set of states `waypoints` en route.

    Computes the conditional committors q^{ABC^+} and uses them to find the
    fraction of paths mentioned above. The conditional committors can be

    Note that in the notation of Dickson et. al. this computes h_c(A,B), with
        sources   = A
        sinks     = B
        waypoint  = C

    Parameters
    ----------
    tprob : matrix
        The transition probability matrix
    waypoint : int
        The index of the intermediate state
    sources : nd_array, int or int
        The indices of the source state(s)
    sinks : nd_array, int or int
        The indices of the sink state(s)
    return_cond_Q : bool
        Whether or not to return the conditional committors

    Returns
    -------
    fraction_paths : float
        The fraction of times a walker going from `sources` -> `sinks` stops
        by `waypoints` on its way.
    cond_Q : nd_array, float (optional)
        Optionally returned (`return_cond_Q`)

    See Also
    --------
    calculate_hub_score : function
        Compute the 'hub score', the weighted fraction of visits for an
        entire network.
    calculate_all_hub_scores : function
        Wrapper to compute all the hub scores in a network.

    Notes
    -----
    Employs dense linear algebra,
      memory use scales as N^2
      cycle use scales as N^3

    References
    ----------
    ..[1] Dickson & Brooks (2012), J. Chem. Theory Comput.,
          Article ASAP DOI: 10.1021/ct300537s
    """

    # do some typechecking - we need to be sure that the lumped sources are in
    # the second to last row, and the lumped sinks are in the last row
    # check `tprob`
    msm_analysis.check_transition(tprob)
    if type(tprob) != np.ndarray:
        try:
            tprob = tprob.todense()
        except AttributeError as e:
            raise TypeError('Argument `tprob` must be convertable to a dense'
                            'numpy array. \n%s' % e)

    # typecheck
    for data in [source, sink, waypoint]:
        if type(data) == int:
            pass
        elif hasattr(data, 'len'):
            if len(data) == 1:
                data = data[0]
        else:
            raise TypeError('Arguments source/sink/waypoint must be an int')

    if (source == waypoint) or (sink == waypoint) or (sink == source):
        raise ValueError('source, sink, waypoint must all be disjoint!')

    N = tprob.shape[0]
    Q = calculate_committors([source], [sink], tprob)

    # permute the transition matrix into cannonical form - send waypoint the the
    # last row, and source + sink to the end after that
    Bsink_indices = [source, sink, waypoint]
    perm = np.arange(N)
    perm = np.delete(perm, Bsink_indices)
    perm = np.append(perm, Bsink_indices)
    T = MSMLib.permute_mat(tprob, perm)

    # extract P, R
    n = N - len(Bsink_indices)
    P = T[:n, :n]
    R = T[:n, n:]

    # calculate the conditional committors ( B = N*R ), B[i,j] is the prob
    # state i ends in j, where j runs over the source + sink + waypoint
    # (waypoint is position -1)
    B = np.dot(np.linalg.inv(np.eye(n) - P), R)
    # Not sure if this is sparse or not...

    # add probs for the sinks, waypoint / b[i] is P( i --> {C & not A, B} )
    b = np.append(B[:, -1].flatten(), [0.0] * (len(Bsink_indices) - 1) + [1.0])
    cond_Q = b * Q[waypoint]

    epsilon = 1e-6  # some numerical give, hard-coded
    assert cond_Q.shape == (N,)
    assert np.all(cond_Q <= 1.0 + epsilon)
    assert np.all(cond_Q >= 0.0 - epsilon)
    assert np.all(cond_Q <= Q[perm] + epsilon)

    # finally, calculate the fraction of paths h_C(A,B) (eq. 7 in [1])
    fraction_paths = np.sum(T[-3, :] * cond_Q) / np.sum(T[-3, :] * Q[perm])

    assert fraction_paths <= 1.0
    assert fraction_paths >= 0.0

    if return_cond_Q:
        cond_Q = cond_Q[np.argsort(perm)]  # put back in orig. order
        return fraction_paths, cond_Q
    else:
        return fraction_paths


def calculate_hub_score(tprob, waypoint):
    """
    Calculate the hub score for the states `waypoint`.

    The "hub score" is a measure of how well traveled a certain state or
    set of states is in a network. Specifically, it is the fraction of
    times that a walker visits a state en route from some state A to another
    state B, averaged over all combinations of A and B.


    Parameters
    ----------
    tprob : matrix
        The transition probability matrix
    waypoints : int
        The indices of the intermediate state(s)

    Returns
    -------
    Hc : float
        The hub score for the state composed of `waypoints`

    See Also
    --------
    calculate_fraction_visits : function
        Calculate the fraction of times a state is visited on pathways going
        from a set of "sources" to a set of "sinks".
    calculate_all_hub_scores : function
        A more efficient way to compute the hub score for every state in a
        network.

    Notes
    -----
    Employs dense linear algebra,
      memory use scales as N^2
      cycle use scales as N^5

    References
    ----------
    ..[1] Dickson & Brooks (2012), J. Chem. Theory Comput.,
        Article ASAP DOI: 10.1021/ct300537s
    """

    msm_analysis.check_transition(tprob)

    # typecheck
    if type(waypoint) != int:
        if hasattr(waypoint, '__len__'):
            if len(waypoint) == 1:
                waypoint = waypoint[0]
            else:
                raise ValueError('Must pass waypoints as int or list/array of ints')
        else:
            raise ValueError('Must pass waypoints as int or list/array of ints')

    # find out which states to include in A, B (i.e. everything but C)
    N = tprob.shape[0]
    states_to_include = list(range(N))
    states_to_include.remove(waypoint)

    # calculate the hub score
    Hc = 0.0
    for s1 in states_to_include:
        for s2 in states_to_include:
            if (s1 != s2) and (s1 != waypoint) and (s2 != waypoint):
                Hc += calculate_fraction_visits(tprob, waypoint,
                                                s1, s2, return_cond_Q=False)

    Hc /= ((N - 1) * (N - 2))

    return Hc


def calculate_all_hub_scores(tprob):
    """
    Calculate the hub scores for all states in a network defined by `tprob`.

    The "hub score" is a measure of how well traveled a certain state or
    set of states is in a network. Specifically, it is the fraction of
    times that a walker visits a state en route from some state A to another
    state B, averaged over all combinations of A and B.

    Parameters
    ----------
    tprob : matrix
        The transition probability matrix

    Returns
    -------
    Hc_array : nd_array, float
        The hub score for each state in `tprob`

    See Also
    --------
    calculate_fraction_visits : function
        Calculate the fraction of times a state is visited on pathways going
        from a set of "sources" to a set of "sinks".
    calculate_hub_score : function
        A function that computes just one hub score, can compute the hub score
        for a set of states.

    Notes
    -----
    Employs dense linear algebra,
      memory use scales as N^2
      cycle use scales as N^6

    References
    ----------
    ..[1] Dickson & Brooks (2012), J. Chem. Theory Comput.,
        Article ASAP DOI: 10.1021/ct300537s
    """

    N = tprob.shape[0]
    states = list(range(N))

    # calculate the hub score
    Hc_array = np.zeros(N)

    # loop over each state and compute it's hub score
    for i, waypoint in enumerate(states):

        Hc = 0.0

        # now loop over all combinations of sources/sinks and average
        for s1 in states:
            if waypoint != s1:
                for s2 in states:
                    if s1 != s2:
                        if waypoint != s2:
                            Hc += calculate_fraction_visits(tprob, waypoint, s1, s2)

        # store the hub score in an array
        Hc_array[i] = Hc / ((N - 1) * (N - 2))

    return Hc_array
