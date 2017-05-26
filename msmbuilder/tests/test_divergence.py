import numpy as np
from msmbuilder.cluster import LandmarkAgglomerative
from msmbuilder.utils.divergence import *
from scipy.stats import entropy

def _get_random_prob_dist(n):
    P = np.random.random([n,n])
    P = (P.T / np.sum(P,axis=1)).T
    return P


def test_kullback_leibler_manual():
    P = _get_random_prob_dist(4)
    Q = _get_random_prob_dist(4)

    vec = []
    for row in range(P.shape[0]):
        temp = 0
        for i, entry in enumerate(P[row]):
            temp += entry * np.log(entry/Q[row][i])
        vec.append(temp)
    manual_kl = np.array(vec)

    msmb_kl = kl_divergence(P, Q, manual=False, scalar=False)

    assert np.allclose(manual_kl, msmb_kl)


def test_kullback_leibler_scipy():
    P = _get_random_prob_dist(4)
    Q = _get_random_prob_dist(4)

    scipy_kl = entropy(P.T, Q.T)
    msmb_kl = kl_divergence(P, Q, manual=False, scalar=False)  

    assert np.allclose(scipy_kl, msmb_kl)


def test_js_correspondence():
    P = _get_random_prob_dist(4)
    Q = _get_random_prob_dist(4)

    assert np.sum(np.sqrt(js_divergence(P,Q))) == np.sum(js_metric(P,Q))


def test_array_vs_msm():
    my_list = [_get_random_prob_dist(4) for i in range(100)]

    my_0 = np.array([x[0] for x in my_list])
    my_1 = np.array([x[1] for x in my_list])    
    my_2 = np.array([x[2] for x in my_list])    
    my_3 = np.array([x[3] for x in my_list])    

    my_flat = np.array([x.flatten() for x in my_list])

    ind = np.random.randint(100)

    dist0 = kl_divergence_array(my_0, my_0, ind)
    dist1 = kl_divergence_array(my_1, my_1, ind)
    dist2 = kl_divergence_array(my_2, my_2, ind)
    dist3 = kl_divergence_array(my_3, my_3, ind)
    dist_sum = dist0 + dist1 + dist2 + dist3

    dist_all = kl_divergence_msm(my_flat, my_flat, ind)

    assert np.allclose(dist_sum, dist_all)


def test_array_vs_msm_sym():
    my_list = [_get_random_prob_dist(4) for i in range(100)]

    my_0 = np.array([x[0] for x in my_list])
    my_1 = np.array([x[1] for x in my_list])    
    my_2 = np.array([x[2] for x in my_list])    
    my_3 = np.array([x[3] for x in my_list])    

    my_flat = np.array([x.flatten() for x in my_list])

    ind = np.random.randint(100)

    dist0 = sym_kl_divergence_array(my_0, my_0, ind)
    dist1 = sym_kl_divergence_array(my_1, my_1, ind)
    dist2 = sym_kl_divergence_array(my_2, my_2, ind)
    dist3 = sym_kl_divergence_array(my_3, my_3, ind)
    dist_sum = dist0 + dist1 + dist2 + dist3

    dist_all = sym_kl_divergence_msm(my_flat, my_flat, ind)

    assert np.allclose(dist_sum, dist_all)


def test_array_vs_msm_js():
    my_list = [_get_random_prob_dist(4) for i in range(100)]

    my_0 = np.array([x[0] for x in my_list])
    my_1 = np.array([x[1] for x in my_list])    
    my_2 = np.array([x[2] for x in my_list])    
    my_3 = np.array([x[3] for x in my_list])    

    my_flat = np.array([x.flatten() for x in my_list])

    ind = np.random.randint(100)

    dist0 = js_divergence_array(my_0, my_0, ind)
    dist1 = js_divergence_array(my_1, my_1, ind)
    dist2 = js_divergence_array(my_2, my_2, ind)
    dist3 = js_divergence_array(my_3, my_3, ind)
    dist_sum = dist0 + dist1 + dist2 + dist3

    dist_all = js_divergence_msm(my_flat, my_flat, ind)

    assert np.allclose(dist_sum, dist_all)


def test_agglom_with_metric_array():
    my_list = [_get_random_prob_dist(4) for i in range(100)]
    my_stationary = np.array([x[0] for x in my_list])
    model = LandmarkAgglomerative(n_clusters=2,
                                  metric=sym_kl_divergence_array,
                                  linkage='complete')
    assert model.fit_predict([my_stationary])[0].shape == (100,)


def test_agglom_with_metric_msm():
    my_list = [_get_random_prob_dist(4) for i in range(100)]
    my_flat = np.array([x.flatten() for x in my_list])
    model = LandmarkAgglomerative(n_clusters=2,
                                  metric=sym_kl_divergence_msm,
                                  linkage='complete')
    assert model.fit_predict([my_flat])[0].shape == (100,)
