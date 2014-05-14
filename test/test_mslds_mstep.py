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

def test_muller_potential_mstep():
    sim_T = 1000
    l.fit(obs_sequences)
    mslds_score = l.score(obs_sequences)
    print("MSLDS Log-Likelihood = %f" %  mslds_score)
    # Fit Gaussian HMM for comparison
    g = GaussianFusionHMM(muller.K, muller.x_dim)
    g.fit(obs_sequences)
    hmm_score = g.score(obs_sequences)
    print("HMM Log-Likelihood = %f" %  hmm_score)
    # Clear Display
    plt.cla()
    plt.plot(trajectory[start:, 0], trajectory[start:, 1], color='k')
    plt.scatter(l.means_[:, 0], l.means_[:, 1], color='r', zorder=10)
    plt.scatter(obs_sequences[0, :, 0], obs_sequences[0,:, 1],
            edgecolor='none', facecolor='k', zorder=1)
    Delta = 0.5
    minx = min(obs_sequences[0, :, 0])
    maxx = max(obs_sequences[0, :, 0])
    miny = min(obs_sequences[0, :, 1])
    maxy = max(obs_sequences[0, :, 1])
    sim_xs, sim_Ss = l.sample(sim_T, init_state=0, init_obs=l.means_[0])

    minx = min(min(sim_xs[:, 0]), minx) - Delta
    maxx = max(max(sim_xs[:, 0]), maxx) + Delta
    miny = min(min(sim_xs[:, 1]), miny) - Delta
    maxy = max(max(sim_xs[:, 1]), maxy) + Delta
    plt.scatter(sim_xs[:, 0], sim_xs[:, 1], edgecolor='none',
               zorder=5, facecolor='g')
    plt.plot(sim_xs[:, 0], sim_xs[:, 1], zorder=5, color='g')


    MullerForce.plot(ax=plt.gca(), minx=minx, maxx=maxx,
            miny=miny, maxy=maxy)
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

