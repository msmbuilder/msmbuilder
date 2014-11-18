

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
