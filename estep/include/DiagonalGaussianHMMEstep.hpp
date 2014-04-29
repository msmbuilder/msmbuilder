/*****************************************************************/
/*    Copyright (c) 2014, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#ifndef MIXTAPE_DIAGONALGAUSSIANHMMESTEP_H
#define MIXTAPE_DIAGONALGAUSSIANHMMESTEP_H
#include "typedefs.hpp"
#include "HMMEstep.hpp"
#include "DiagonalGaussianHMMSufficientStats.hpp"

namespace Mixtape {

class DiagonalGaussianHMMEstep : public HMMEstep {
public:
    /**
     * Create the E-step for an HMM with a Gaussian emission
     * distribution with a diagonal covariance matrix
     *
     * Pameters
     * --------
     * numStates : int
     * transmat : array of shape (numStates, numStates)
     * startprob : array of shape (numStates)
     * numFeatures : int
     * means : array of shape (numStates, numFeatures)
     * variances : array of shape (numStates, numFeatures)
     */
    DiagonalGaussianHMMEstep(const int numStates,
                             const DoubleArray2D& transmat,
                             const DoubleArray1D& startProb,
                             const int numFeatures,
                             const DoubleArray2D& means,
                             const DoubleArray2D& variances);


    /**
     * Initialize the sufficient statistics container for this
     * distribution
     */
    DiagonalGaussianHMMSufficientStats* initializeSufficientStats();


    /**
     * Compute the log-likelihood of the diagonal multivariate gaussian
     * evaluated at each point in a data sequence.
     *
     * Parameters
     * ----------
     * X : 2d array of shaep (numSamples, numFeatures)
     *     X is a single timeseries, the "raw data" being fit by the model.
     *
     * Returns
     * -------
     * logl : 2d array of shape (numSamples, numStates)
     *     logl[i][j] log-likelihood of state `j`'s Gaussian
     *     distribution evaluated at the point X[i]
     */
    FloatArray2D emissionLogLikelihood(const FloatArray2D& X);

protected:
    const int numFeatures_;

private:
    const DoubleArray2D means_;
    const DoubleArray2D variances_;
    DoubleArray2D logVariances_;
    DoubleArray2D meansOverVariances_;
    DoubleArray2D means2OverVariances_;
};


void doDiagonalGaussianHMMEstep(const int numStates,
                                const double* transmat,
                                const double* startProb,
                                const int numFeatures,
                                const double* means,
                                const double* variances,
                                const int numSequences,
                                const int* sequenceLengths,
                                const float** sequences);
/*                                float* transcounts,
                                float* posts,
                                float* obs,
                                float* obs2);*/

} // namespace
#endif
