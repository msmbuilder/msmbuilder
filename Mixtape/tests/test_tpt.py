from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import sys
import os

import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.io

from mdtraj import io
from msmbuilder import tpt
from msmbuilder.scripts import FindPaths
from msmbuilder.testing import get

def tpt_get(filename):
    """ a little hack to save headache -- returns the path to a file in 
        the hub_ref subdir inside reference/ """
    return get( os.path.join('transition_path_theory_reference', filename), just_filename=True )
        
  
def hub_get(filename):
    """ a little hack to save headache -- returns the path to a file in 
        the hub_ref subdir inside reference/ """
    return get( os.path.join('transition_path_theory_reference', 'hub_ref', filename), just_filename=True )
    


class TestTPT():
    """ Test the transition_path_theory library """

    def setUp(self):
        
        # load in the reference data
        self.tprob = get("transition_path_theory_reference/tProb.mtx")
        self.sources   = [0]   # chosen arbitarily by TJL
        self.sinks     = [70]  # chosen arbitarily by TJL
        self.waypoints = [60]  # chosen arbitarily by TJL
        self.lag_time  = 1.0   # chosen arbitarily by TJL

        self.multi_sources = get("transition_path_theory_reference/many_state/sources.dat").astype(int)
        self.multi_sinks = get("transition_path_theory_reference/many_state/sinks.dat").astype(int)

        self.num_paths = 10

        # set up the reference data for hub scores
        K = np.loadtxt( hub_get('ratemat_1.dat') )
        #self.hub_T = scipy.linalg.expm( K ) # delta-t should not affect hub scores
        self.hub_T = np.transpose( np.genfromtxt( hub_get('mat_1.dat') )[:,:-3] )
        
        for i in range(self.hub_T.shape[0]):
            self.hub_T[i,:] /= np.sum(self.hub_T[i,:])
        
        self.hc = np.loadtxt( hub_get('fraction_visited.dat') )
        self.Hc = np.loadtxt( hub_get('hub_scores.dat') )[:,2]
        

    def test_committors(self):
        Q = tpt.calculate_committors(self.sources, self.sinks, self.tprob)
        Q_ref = io.loadh(tpt_get("committors.h5"), 'Data')
        npt.assert_array_almost_equal(Q, Q_ref)
        
        
    def test_flux(self):
        
        flux = tpt.calculate_fluxes(self.sources, self.sinks, self.tprob)
        flux_ref = io.loadh(tpt_get("flux.h5"), 'Data')
        npt.assert_array_almost_equal(flux.toarray(), flux_ref)
        
        net_flux = tpt.calculate_net_fluxes(self.sources, self.sinks, self.tprob)
        net_flux_ref = io.loadh(tpt_get("net_flux.h5"), 'Data')
        npt.assert_array_almost_equal(net_flux.toarray(), net_flux_ref)
        
    
    # this test is for the original bottleneck cutting code    
    def test_path_calculations(self):
        path_output = tpt._find_top_paths_cut(self.sources, self.sinks, self.tprob)
    
        paths_ref = io.loadh( tpt_get("dijkstra_paths.h5"), 'Data')
        fluxes_ref = io.loadh( tpt_get("dijkstra_fluxes.h5"), 'Data')
        bottlenecks_ref = io.loadh( tpt_get("dijkstra_bottlenecks.h5"), 'Data')

        for i in range(len(paths_ref)):
            npt.assert_array_almost_equal(path_output[0][i], paths_ref[i])
        npt.assert_array_almost_equal(path_output[1], bottlenecks_ref)
        npt.assert_array_almost_equal(path_output[2], fluxes_ref)


    #def test_multi_state_path_calculations(self):
    #    path_output = FindPaths.run(self.tprob, self.multi_sources, self.multi_sinks, self.num_paths)

    #    path_result_ref = io.loadh(tpt_get("many_state/Paths.h5"))

    #    paths_ref = path_result_ref['Paths']
    #    bottlenecks_ref = path_result_ref['Bottlenecks']
    #    fluxes_ref = path_result_ref['fluxes']

    #    npt.assert_array_almost_equal(path_output[0], paths_ref)
    #    npt.assert_array_almost_equal(path_output[1], bottlenecks_ref)
    #    npt.assert_array_almost_equal(path_output[2], fluxes_ref)


    def test_mfpt(self):
        
        mfpt = tpt.calculate_mfpt(self.sinks, self.tprob, lag_time=self.lag_time)
        mfpt_ref = io.loadh( tpt_get("mfpt.h5"), 'Data')
        npt.assert_array_almost_equal(mfpt, mfpt_ref)
        
        ensemble_mfpt = tpt.calculate_ensemble_mfpt(self.sources, self.sinks, self.tprob, self.lag_time)
        ensemble_mfpt_ref = io.loadh( tpt_get("ensemble_mfpt.h5"), 'Data')
        npt.assert_array_almost_equal(ensemble_mfpt, ensemble_mfpt_ref)
        
        all_to_all_mfpt = tpt.calculate_all_to_all_mfpt(self.tprob)
        all_to_all_mfpt_ref = io.loadh( tpt_get("all_to_all_mfpt.h5"), 'Data')
        npt.assert_array_almost_equal(all_to_all_mfpt, all_to_all_mfpt_ref)
        
        
    def test_TP_time(self):
        tp_time = tpt.calculate_avg_TP_time(self.sources, self.sinks, self.tprob, self.lag_time)
        tp_time_ref = io.loadh( tpt_get("tp_time.h5"), 'Data')
        npt.assert_array_almost_equal(tp_time, tp_time_ref)
        
        
    def test_fraction_visits(self):
        
        num_to_test = 11**2 # this can be changed to shorten the test a little
        
        for i in range(num_to_test):
            waypoint = int(self.hc[i,0])
            source   = int(self.hc[i,1])
            sink     = int(self.hc[i,2])
            hc = tpt.calculate_fraction_visits(self.hub_T, waypoint, source, sink)
            assert np.abs(hc - self.hc[i,3]) < 0.0001
        

    def test_hub_scores(self):
        
        all_hub_scores = tpt.calculate_all_hub_scores(self.hub_T)
        npt.assert_array_almost_equal( all_hub_scores, self.Hc )
        
        for waypoint in range(self.hub_T.shape[0]):
            hub_score = tpt.calculate_hub_score(self.hub_T, waypoint)
            assert np.abs(hub_score - all_hub_scores[waypoint]) < 0.0001
        
        

class TestTPT2():
    def setUp(self):
        
        self.net_flux = np.array([[0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.3, 0.0, 0.2],
                                  [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.sources = np.array([0])
        self.sinks = np.array([4, 5])

        self.ref_paths = np.array([[0, 2, 4, -1],
                                   [0, 1, 3,  5],
                                   [0, 1, 5, -1]])

        self.ref_fluxes = np.array([0.5, 0.3, 0.2])

    def testA(self):
        
        paths, fluxes = tpt.get_paths(self.sources, self.sinks, self.net_flux)

        npt.assert_array_almost_equal(fluxes, self.ref_fluxes)

        npt.assert_array_equal(paths, self.ref_paths)

        
    
