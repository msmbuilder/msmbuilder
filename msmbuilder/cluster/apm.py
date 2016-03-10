from __future__ import absolute_import, print_function, division
__author__ = 'LIU Song <stephenliu1989@gmail.com>'
__contributors__ = "Fu Kit SHEONG, Xuhui HUANG"
__version__ = "0.91"
# Copyright (c) 2015, Hong Kong University of Science and Technology (HKUST)
# All rights reserved.
# ===============================================================================
# GLOBAL IMPORTS:
import time
import copy
import numpy as np
# ===============================================================================
# LOCAL IMPORTS:
from .kcenters import KCenters
from ..lumping import PCCA
from ..msm import MarkovStateModel
from . import MultiSequenceClusterMixin
from ..base import BaseEstimator
# ===============================================================================

class APM(BaseEstimator):
    '''
    APM
    This Program is a python package which implements Automatic State Partitioning
    for Multibody Systems (APM) algorithm to cluster trajectories and model the
    transitions based on a Markov State Model (MSM).
    Parameters
    ----------
    metric : {"euclidean", "sqeuclidean", "cityblock", "chebyshev", "canberra",
              "braycurtis", "hamming", "jaccard", "cityblock", "rmsd"}
        The distance metric to use. metric = "rmsd" requires that sequences
        passed to ``fit()`` be ```md.Trajectory```; other distance metrics
        require ``np.ndarray``s.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
        default : none
    n_macrostates : int
        The desired number of macrostates in the lumped model.
        default : 6
    max_iter : int
        Max number of iterations, default: 20
    lag_time: int
        The lag time of Markov State Model.
    sub_clus : int
        The cluster number when splitting by k-centers
        default : 2
    Attributes
    ----------
    MacroAssignments_ :  array, [n_samples,]
        The label of each point in an integer in [0, n_macrostates]
    labels_ :  array, [n_samples,]
        The label of each point in an integer in [0, n_microstates]
    '''

    def __init__(self,
                 metric='rmsd',
                 random_state=None,
                 n_macrostates=6,
                 max_iter=50,
                 lag_time=10,
                 sub_clus=2):
        # lag time in number of entries in assignment file (int).

        self.n_macrostates = n_macrostates
        self.lag_time = lag_time
        self.sub_clus = sub_clus
        self.max_iter = max_iter

        self.X = None
        self.metric = metric
        self.random_state = random_state

        self.__max_state = -1
        self.__micro_stack = []

        #Attributes:
        self.labels_ = None
        self.__temp_labels_ = None
        self.MacroAssignments_ = None
        #self.transmat_ = None

    def fit(self, X, y=None):
        """Perform clustering.
        Parameters
        -----------
        X : array-like, shape=[n_samples, n_features]
            Samples to cluster.
        """
        # X = check_array(X)
        t0 = time.time()
        self.X = X
        self._run()
        t1 = time.time()
        print("APM clustering Time Cost:", t1 - t0)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y).MacroAssignments_

    def _run(self):
        """Do the APM lumping.
        """
        print("Doing APM Clustering...")
        # Start looping for maxIter times
        n_macrostates = 1  # initialized as 1 because no macrostate exist in loop 0
        metaQ = -1.0
        prevQ = -1.0
        global_maxQ = -1.0
        local_maxQ = -1.0

        for iter in range(self.max_iter):
            self.__max_state = -1
            self.__micro_stack = []
            for k in range(n_macrostates):
                self._do_split(micro_state=k, sub_clus=self.sub_clus)
                self._do_time_clustering(macro_state=k)

            # do Lumping
            n_micro_states = np.amax(self.__temp_labels_) + 1
            if n_micro_states > self.n_macrostates:
                print("PCCA Lumping...", n_micro_states, "microstates")
                self.__temp_MacroAssignments_ = self._do_lumping(
                    n_macrostates=n_macrostates)
                #self.__temp_labels_ = [copy.copy(element) for element in self.__temp_MacroAssignments_]

                #Calculate Metastabilty
                prevQ = metaQ
                metaQ = self.__temp_transmat_.diagonal().sum()
                metaQ /= len(self.__temp_transmat_)
            else:
                self.__temp_MacroAssignments_ = [
                    copy.copy(element) for element in self.__temp_labels_
                ]

            # Optimization / Monte-Carlo
            acceptedMove = False
            MCacc = np.exp(metaQ * metaQ - prevQ * prevQ)

            if MCacc > 1.0:
                MCacc = 1.0
            optLim = 0.95

            if MCacc > optLim:
                acceptedMove = True

            if acceptedMove:
                local_maxQ = metaQ
                if metaQ > global_maxQ:
                    global_maxQ = metaQ
                    self.MacroAssignments_ = [
                        copy.copy(element)
                        for element in self.__temp_MacroAssignments_
                    ]
                    self.labels_ = [copy.copy(element)
                                    for element in self.__temp_labels_]
                    self.transmat_ = self.__temp_transmat_

            print("Loop:", iter, "AcceptedMove?", acceptedMove, "metaQ:",
                  metaQ, "prevQ:", prevQ, "global_maxQ:", global_maxQ,
                  "local_maxQ:", local_maxQ, "macroCount:", n_macrostates)
            #set n_macrostates
            n_macrostates = self.n_macrostates
            self.__temp_labels_ = [copy.copy(element)
                                   for element in self.__temp_MacroAssignments_
                                   ]

    def _get_Lagtime(self, micro_state=None, macro_state=None):
        if self._get_RelaxProb(micro_state, macro_state) > 0.6321206:
            return self.lag_time
        else:
            return 0  #infinity in this case

    def _get_RelaxProb(self, micro_state=None, macro_state=None):
        count_trans = 0
        count_relax = 0
        for k in range(len(self.X)):
            X_len = len(self.X[k])
            for i in range(X_len - self.lag_time):
                # if it starts at the desired state and ends at the same trajectory, count as one transition
                if self.__temp_labels_[k][
                        i] == micro_state and self.__temp_MacroAssignments_[k][
                            i] == macro_state:
                    count_trans += 1
                    # if it does not end at the same state, count as one relaxation
                    if self.__temp_labels_[k][
                            i +
                            self.lag_time] != micro_state or self.__temp_MacroAssignments_[
                                k][i + self.lag_time] != macro_state:
                        count_relax += 1
        if count_trans > 0:
            relax_prob = float(count_relax) / float(count_trans)
            return relax_prob
        else:
            return 1.0

    def _do_time_clustering(self, macro_state=None):
        print("Doing time clustering...")
        if not self.__micro_stack:
            #print "Stack is emtpy"
            return
        else:
            print("Stack:", self.__micro_stack)
            micro_state = self.__micro_stack[
                -1
            ]  # last element of self.__micro_stack
            if self._get_Lagtime(micro_state, macro_state) is 0:
                # split if the relaxation time is too long
                self._do_split(micro_state=micro_state, sub_clus=self.sub_clus)
            else:
                # accept if the relaxation time is fine
                self.__micro_stack.pop(-1)

            self._do_time_clustering(
                macro_state=macro_state)  # Note: recursion
        return

    def _do_split(self, micro_state=None, sub_clus=2):
        micro_clusterer = KCenters(n_clusters=sub_clus,
                                   metric=self.metric,
                                   random_state=0)
        if self.__temp_labels_ is not None:
            sub_X = []
            sub_indices = []

            #Get sub trjas
            for i in range(len(self.X)):
                sub_X.append([])
                sub_indices.append([])
                sub_indices[i] = np.where(self.__temp_labels_[i] ==
                                          micro_state)
                sub_X[i] = self.X[i][sub_indices[i]]

            micro_clusterer.fit(sub_X)
            sub_labels_ = micro_clusterer.labels_
            for i in range(len(self.__temp_labels_)):
                local_max_state = max(self.__temp_labels_[i])
                if local_max_state > self.__max_state:
                    self.__max_state = local_max_state

            #rename the cluster number on self.__temp_labels_
            for i in range(len(sub_labels_)):
                new_index = 0
                for j in sub_indices[i][0]:
                    if sub_labels_[i][new_index] == 0:
                        sub_labels_[i][new_index] = micro_state
                    else:
                        sub_labels_[i][new_index] += self.__max_state
                    self.__temp_labels_[i][j] = sub_labels_[i][new_index]
                    new_index += 1

                states = list(set(sub_labels_[i]))
                for k in states:
                    if k not in self.__micro_stack:
                        self.__micro_stack.append(k)
        else:
            micro_clusterer.fit(self.X)
            self.__temp_labels_ = micro_clusterer.labels_

    def _do_lumping(self, n_macrostates=3):
        msm = MarkovStateModel()
        msm.fit(self.__temp_labels_)
        algorithm = PCCA.from_msm(msm, n_macrostates)
        macro_assignments = algorithm.fit_transform(self.__temp_labels_)
        self.__temp_transmat_ = algorithm.transmat_
        return macro_assignments