# Author: Brooke Husic <brookehusic@gmail.com>
# Contributors: Greg Bowman
# Copyright (c) 2017, Stanford University and the Authors
# All rights reserved.

from __future__ import print_function, division, absolute_import

import copy
import numpy as np
from msmbuilder.msm import MarkovStateModel
import functools
import multiprocessing
import os


class BACE(MarkovStateModel):
    """Bayesian Agglomerative Clustering Engine (BACE) for coarse-graining (lumping)
    microstates into macrostates.

    Parameters
    ----------
    n_macrostates : int or None
        The desired number of macrostates in the lumped model. If None,
        only the linkages are calcluated (see ``use_scipy``)
    filter : float (default=1.1)
        Prune states with Bayes factors less than this number. Default is
        approximation log(3), so states with less than a 3:1 ratio are pruned.
        NOTE : MSMs not built directly from pairwise RMSD, we recommend
        setting this parameter to zero.
    save_all_maps : boolean (default=True)
        Creates a dicitonary where the keys are the number of macrostates
        and the values are the mapping for that macrostate. Since BACE
        uses agglomerative clustering, the mapping for every number of
        macrostates higher than the one chosen is automatically computed,
        and setting this to True saves them all.
    n_proc : int (default=1)
        Number of processors to use
    kwargs : optional
        Additional keyword arguments to be passed to MarkovStateModel.  See
        msmbuilder.msm.MarkovStateModel for possible options.

    Attributes
    ----------
    microstate_mapping_ : np.array, [number of microstates]
    map_dict : dictionary, [n_macrostates : microstate_mapping_]

    Notes
    -----
    BACE was implemented with as much fidelity to the MSMBuilder 2 version
    as possible. BACE was implemented before sliding windows were used;
    therefore, if a sliding window was used to build the MSM, then the
    counts matrix is multiplied by the lag time before proceeding. This 
    is because the prior probability for each transition is hard-coded
    to be 1 over the number of microstates. In order to produce identical
    results to the MSMB2 version, states with Bayes factors less than 1.1
    are merged with the kinetically closest state before clustering begins.
    When coarse-graining MSMs not built from RMSD-based clustering, however,
    we recommend setting filter=0.

    The MSMB2 version of BACE included an option for sparse matrices, which
    we have removed in this implementation. However, the reciprocal of the
    distance matrix is still taken during the calculation, which was 
    needed for the sparse formulation. For this reason, there might be
    some precision issues when compared to coding BACE separatey.

    BACE is a subclass of MarkovStateModel.  However, the MSM properties
    and attributes on BACE refer to the MICROSTATE properties--e.g.
    mvca.transmat_ is the microstate transition matrix.  To get the
    macrostate transition matrix, you must fit a new MarkovStateModel
    object on the output (assignments) of BACE().
    BACE will scale poorly with the number of microstates. Consider
    use_scipy=False and n_landmarks < number of microstates.
    """

    def __init__(self, n_macrostates, filter=1.1, save_all_maps=True,
                 n_proc=1, chunk_size=100, **kwargs):
        self.n_macrostates = n_macrostates
        self.filter = filter
        self.save_all_maps = save_all_maps
        self.n_proc = n_proc
        self.chunk_size = chunk_size

        if self.save_all_maps:
            self.map_dict = {}

        super(BACE, self).__init__(**kwargs)

    def fit(self, sequences, y=None):
        """Fit a BACE lumping model using a sequence of cluster assignments.

        Parameters
        ----------
        sequences : list(np.ndarray(dtype='int'))
            List of arrays of cluster assignments
        y : None
            Unused, present for sklearn compatibility only.

        Returns
        -------
        self
        """
        super(BACE, self).fit(sequences, y=y)
        if self.n_macrostates is not None:
            self._do_lumping()
        else:
            raise RuntimeError('n_macrostates must not be None to fit')

        return self

    def _do_lumping(self):
        """Do the BACE lumping.
        """
        c = self.countsmat_
        if self.sliding_window:
            c *= self.lag_time

        c, macro_map, statesKeep = self.filterFunc(c)

        w = np.array(c.sum(axis=1)).flatten()
        w[statesKeep] += 1

        unmerged = np.zeros(w.shape[0], dtype=np.int8)
        unmerged[statesKeep] = 1

        # get nonzero indices in upper triangle
        indRecalc = self.getInds(c, statesKeep)
        dMat = np.zeros(c.shape, dtype=np.float32)

        i = 0
        nCurrentStates = statesKeep.shape[0]

        self.fBayesFact = {}

        dMat, minX, minY = self.calcDMat(c, w, indRecalc, dMat,
                                         statesKeep, unmerged)

        while nCurrentStates > self.n_macrostates:
            c, w, indRecalc, dMat, macro_map, statesKeep, unmerged, minX, minY = self.mergeTwoClosestStates(
                c, w, indRecalc, dMat, macro_map,
                statesKeep, minX, minY, unmerged)

            nCurrentStates -= 1

            if self.save_all_maps:
                saved_map = copy.deepcopy(macro_map)
                self.map_dict[nCurrentStates] = saved_map

            if nCurrentStates - 1 == self.n_macrostates:
                self.microstate_mapping_ = macro_map

    def partial_transform(self, sequence, mode='clip'):
        if self.n_macrostates is None:
            raise RuntimeError('n_macrostates must not be None to transform')
        trimmed_sequence = super(BACE, self).partial_transform(sequence, mode)
        if mode == 'clip':
            return [self.microstate_mapping_[seq] for seq in trimmed_sequence]
        elif mode == 'fill':
            def nan_get(x):
                try:
                    x = int(x)
                    return self.microstate_mapping_[x]
                except ValueError:
                    return np.nan

            return np.asarray([nan_get(x) for x in trimmed_sequence])
        else:
            raise ValueError

    @classmethod
    def from_msm(cls, msm, n_macrostates, filter=1.1, save_all_maps=True,
                 n_proc=1, chunk_size=100):
        """Create and fit lumped model from pre-existing MSM.

        Parameters
        ----------
        msm : MarkovStateModel
            The input microstate msm to use.
        n_macrostates : int
            The number of macrostates

        Returns
        -------
        lumper : cls
            The fit MVCA object.
        """
        params = msm.get_params()
        lumper = cls(n_macrostates, filter, save_all_maps, n_proc,
                     chunk_size, **params)

        lumper.transmat_ = msm.transmat_
        lumper.populations_ = msm.populations_
        lumper.mapping_ = msm.mapping_
        lumper.countsmat_ = msm.countsmat_
        lumper.n_states_ = msm.n_states_

        if n_macrostates is not None:
            lumper._do_lumping()

        return lumper

    def getInds(self, c, stateInds, updateSingleState=None):
        indices = []
        for s in stateInds:
            dest = np.where(c[s, :] > 1)[0]

            if updateSingleState != None:
                dest = dest[np.where(dest != updateSingleState)[0]]
            else:
                dest = dest[np.where(dest > s)[0]]

            if dest.shape[0] == 0:
                continue
            elif dest.shape[0] < self.chunk_size:
                indices.append((s, dest))
            else:
                i = 0
                while dest.shape[0] > i:
                    if i + self.chunk_size > dest.shape[0]:
                        indices.append((s, dest[i:]))
                    else:
                        indices.append((s, dest[i:i + self.chunk_size]))
                    i += self.chunk_size
        return indices

    def mergeTwoClosestStates(self, c, w, indRecalc, dMat,
                              macro_map, statesKeep, minX, minY,
                              unmerged):
        if unmerged[minX]:
            c[minX, statesKeep] += unmerged[statesKeep] * 1.0 / c.shape[0]
            unmerged[minX] = 0
            c[statesKeep, minX] += unmerged[statesKeep] * 1.0 / c.shape[0]
        if unmerged[minY]:
            c[minY, statesKeep] += unmerged[statesKeep] * 1.0 / c.shape[0]
            unmerged[minY] = 0
            c[statesKeep, minY] += unmerged[statesKeep] * 1.0 / c.shape[0]
        c[minX, statesKeep] += c[minY, statesKeep]
        c[statesKeep, minX] += c[statesKeep, minY]
        c[minY, statesKeep] = 0
        c[statesKeep, minY] = 0
        dMat[minX, :] = 0
        dMat[:, minX] = 0
        dMat[minY, :] = 0
        dMat[:, minY] = 0
        w[minX] += w[minY]
        w[minY] = 0
        statesKeep = statesKeep[np.where(statesKeep != minY)[0]]
        indChange = np.where(macro_map == macro_map[minY])[0]
        macro_map = self.renumberMap(macro_map, macro_map[minY])
        macro_map[indChange] = macro_map[minX]
        indRecalc = self.getInds(c, [minX], updateSingleState=minX)
        dMat, minX, minY = self.calcDMat(c, w, indRecalc, dMat, statesKeep,
                                         unmerged)
        return c, w, indRecalc, dMat, macro_map, statesKeep, unmerged, minX, minY

    def renumberMap(self, macro_map, stateDrop):
        for i in range(macro_map.shape[0]):
            if macro_map[i] >= stateDrop:
                macro_map[i] -= 1
        return macro_map

    def calcDMat(self, c, w, indRecalc, dMat, statesKeep, unmerged):
        nRecalc = len(indRecalc)
        if nRecalc > 1 and self.n_proc > 1:
            if nRecalc < self.n_proc:
                self.n_proc = nRecalc
            pool = multiprocessing.Pool(processes=self.n_proc)
            n = len(indRecalc)
            stepSize = int(n / self.n_proc)
            if n % stepSize > 3:
                dlims = zip(range(0, n, stepSize), range(
                    stepSize, n, stepSize) + [n])
            else:
                dlims = zip(range(0, n - stepSize, stepSize),
                            range(stepSize, n - stepSize, stepSize) + [n])
            args = []
            for start, stop in dlims:
                args.append(indRecalc[start:stop])
            result = pool.map_async(functools.partial(self.multiDist,
                                                      c=c, w=w, statesKeep=statesKeep, unmerged=unmerged,
                                                      chunk_size=self.chunk_size), args)
            result.wait()
            d = np.vstack(result.get())
            pool.close()
        else:
            d = self.multiDist(indRecalc, c, w, statesKeep, unmerged)
        for i in range(len(indRecalc)):
            dMat[indRecalc[i][0], indRecalc[i][1]
                 ] = d[i][:len(indRecalc[i][1])]

        # BACE BF inverted so can use sparse matrices
        indMin = dMat.argmax()
        minX = int(np.floor(indMin / dMat.shape[1]))
        minY = int(indMin % dMat.shape[1])

        self.fBayesFact[statesKeep.shape[0] - 1] = 1. / dMat[minX, minY]

        #fBayesFact.write("%d %f\n" % (statesKeep.shape[0]-1, 1./dMat[minX,minY]))
        return dMat, minX, minY

    def multiDist(self, indicesList, c, w, statesKeep, unmerged):
        d = np.zeros((len(indicesList), self.chunk_size), dtype=np.float32)
        for j in range(len(indicesList)):
            indices = indicesList[j]
            ind1 = indices[0]
            c1 = c[ind1, statesKeep] + unmerged[ind1] * \
                unmerged[statesKeep] * 1.0 / c.shape[0]
            d[j, :indices[1].shape[0]] = 1. / \
                self.multiDistHelper(
                    indices[1], c1, w[ind1], c, w, statesKeep, unmerged)
            # BACE BF inverted so can use sparse matrices
        return d

    def multiDistHelper(self, indices, c1, w1, c, w, statesKeep, unmerged):
        d = np.zeros(indices.shape[0], dtype=np.float32)
        p1 = c1 / w1
        for i in range(indices.shape[0]):
            ind2 = indices[i]
            c2 = c[ind2, statesKeep] + unmerged[ind2] * \
                unmerged[statesKeep] * 1.0 / c.shape[0]
            p2 = c2 / w[ind2]
            cp = c1 + c2
            cp /= (w1 + w[ind2])
            d[i] = c1.dot(np.log(p1 / cp)) + c2.dot(np.log(p2 / cp))
        return d

    def filterFunc(self, c):
        # get num counts in each state (or weight)
        w = np.array(c.sum(axis=1)).flatten()
        w += 1

        # init map from micro to macro states
        macro_map = np.arange(c.shape[0], dtype=np.int32)

        # pseudo-state (just pseudo counts)
        pseud = np.ones(c.shape[0], dtype=np.float32)
        pseud /= c.shape[0]

        indices = np.arange(c.shape[0], dtype=np.int32)
        statesKeep = np.arange(c.shape[0], dtype=np.int32)
        unmerged = np.ones(c.shape[0], dtype=np.float32)

        nInd = len(indices)
        if nInd > 1 and self.n_proc > 1:
            if nInd < self.n_proc:
                self.n_proc = nInd
            pool = multiprocessing.Pool(processes=self.n_proc)
            stepSize = int(nInd / self.n_proc)
            if nInd % stepSize > 3:
                dlims = zip(range(0, nInd, stepSize), range(
                    stepSize, nInd, stepSize) + [nInd])
            else:
                dlims = zip(range(0, nInd - stepSize, stepSize),
                            range(stepSize, nInd - stepSize, stepSize) + [nInd])
            args = []
            for start, stop in dlims:
                args.append(indices[start:stop])
            result = pool.map_async(functools.partial(self.multiDistHelper,
                                                      c1=pseud, w1=1, c=c, w=w, statesKeep=statesKeep,
                                                      unmerged=unmerged), args)
            result.wait()
            d = np.concatenate(result.get())
            pool.close()
        else:
            d = self.multiDistHelper(
                indices, pseud, 1, c, w, statesKeep, unmerged)

        statesPrune = np.where(d < self.filter)[0]
        statesKeep = np.where(d >= self.filter)[0]

        for s in statesPrune:
            dest = c[s, :].argmax()
            c[dest, :] += c[s, :]
            c[s, :] = 0
            c[:, s] = 0
            macro_map = self.renumberMap(macro_map, macro_map[s])
            macro_map[s] = macro_map[dest]

        return c, macro_map, statesKeep
