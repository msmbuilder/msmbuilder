
def test_plusmin_mstep():
    l = MetastableSwitchingLDS(plusmin.K, plusmin.x_dim,
            n_hotstart=0)
    l.fit(obs_sequences)
    mslds_score = l.score(obs_sequences)
    print("MSLDS Log-Likelihood = %f" %  mslds_score)
    # Fit Gaussian HMM for comparison
    g = GaussianFusionHMM(plusmin.K, plusmin.x_dim)
    g.fit(obs_sequences)
    hmm_score = g.score(obs_sequences)
    print("HMM Log-Likelihood = %f" %  hmm_score)
    sim_xs, sim_Ss = l.sample(T, init_state=0, init_obs=plusmin.mus[0])
    sim_xs = np.reshape(sim_xs, (n_seq, T, plusmin.x_dim))

    plt.close('all')
    plt.figure(1)
    plt.plot(range(T), obs_sequences[0], label="Observations")
    plt.plot(range(T), sim_xs[0], label='Sampled Observations')
    plt.legend()
    plt.show()
    pass
def test_sample():
    assert True == False

def test_predict():
    assert True == False

def test_fit():
    assert True == False

def test_means_update():
    assert True == False

def test_transmat_update():
    assert True == False

def test_A_update():
    assert True == False

def test_Q_update():
    assert True == False

def test_b_update():
    assert True == False

