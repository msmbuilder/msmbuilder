import numpy as np


def create_perturb_params(countsmat):
    '''
    Computes transition probabilities and standard errors of the transition probabilities due to 
    finite sampling using the MSM counts matrix. First, the transition probabilities are computed 
    by dividing the each element c_ij by the row-sumemd counts of row i. THe standard errors are then
    computed by first computing the standard deviation of the transition probability, treating each count 
    as a Bernoulli process with p = t_ij (std = (t_ij - t_ij ^2)^0.5). This is then divided by the 
    square root of the row-summed counts of row i to obtain the standard error.
    
    Parameters:
    ----------
    countsmat: np.ndarray
        The msm counts matrix

    Returns:
    -----------
    transmat, np.ndarray:
        The MSM transition matrix
    scale, np.ndarray:
        The matrix of standard errors for each transition probability
    '''
    norm = np.sum(countsmat, axis=1)
    transmat = (countsmat.transpose() / norm).transpose()
    counts = (np.ones((len(transmat), len(transmat))) * norm).transpose()
    scale = ((transmat - transmat ** 2) ** 0.5 / counts ** 0.5) + 10 ** -15
    return transmat, scale


def perturb_tmat(transmat, scale):
    '''
    Perturbs each nonzero entry in the MSM transition matrix by treating it as a Gaussian random variable
    with mean t_ij and standard deviation equal to the standard error computed using "create_perturb_params".
    Returns a sampled transition matrix that takes into consideration errors due to finite sampling
    (useful for boostrapping, etc.)

    Parameters:
    ----------
    transmat: np.ndarray:
        The transition matrix, whose elements serve as the means of the Gaussian random variables
    scale: np.ndarray:
        The matrix of standard errors. For transition probability t_ij, this is assumed to be the standard 
        error of the mean of a binomial distribution with p = transition probability and number of observations 
        equal to the summed counts in row i.

    '''
    output = np.vectorize(np.random.normal)(transmat, scale)
    output[np.where(output < 0)] = 0
    return (output.transpose() / np.sum(output, axis=1)).transpose()

