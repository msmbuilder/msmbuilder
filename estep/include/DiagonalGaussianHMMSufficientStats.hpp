/*****************************************************************/
/*    Copyright (c) 2014, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#ifndef MIXTAPE_DIAGONALGAUSSIANHMMSUFFICIENTSTATS_H
#define MIXTAPE_DIAGONALGAUSSIANHMMSUFFICIENTSTATS_H
#include "typedefs.hpp"
#include "HMMEstep.hpp"
#include "HMMSufficientStats.hpp"

namespace Mixtape {
class DiagonalGaussianHMMSufficientStats : public HMMSufficientStats {
public:
    /**
     * Initialize the container for the sufficient statistics of a gaussain
     * HMM with a diagonal covariance matrix. You should not call this constructor
     * directly -- it should be called by initializeSufficientStatistics in
     * the DiagonalGaussianHMMEstep class, and then returned to the client
     * as the final return value from execute()
     */
    DiagonalGaussianHMMSufficientStats(const int numStates, const int numFeatures)
        : HMMSufficientStats(numStates)
        , numFeatures_(numFeatures)
        , posts_(boost::extents[numStates])
        , obs_(boost::extents[numStates][numFeatures])
        , obs2_(boost::extents[numStates][numFeatures])
    { };


    /**
     * The posterior weight of each state
     *
     *    posts[i] = \sum_t posteriors[t][i]
     *
     * where `posteriors` is the posterior probabilty that the sequence being trained
     * on was in state `i` at time `t`.
     */
    DoubleArray1D& posts() { return posts_; }

    /**
     * Sum of the posterior-weighted data across the sequences
     *
     *     obs[i][j] = \sum_t posteriors[i][j] * X[t][j]
     *
     * where `posteriors` is the posterior probabilty that the sequence being trained
     * on was in state `i` at time `t`, and X is a training data sequence of length `t`
     * with features indexed by `j`.
     */
    DoubleArray2D& obs() { return obs_; }

    /**
     * Sum of the posterior-weighted square of the data across the sequences
     *
     *     obs[i][j] = \sum_t posteriors[i][j] * (X[t][j])^2
     *
     * where `posteriors` is the posterior probabilty that the sequence being trained
     * on was in state `i` at time `t`, and X is a training data sequence of length `t`
     * with features indexed by `j`.
     */
    DoubleArray2D& obs2() { return obs2_; }

    void increment(const HMMEstep* estep,
                   const FloatArray2D& seq,
                   const FloatArray2D& frameLogProb,
                   const FloatArray2D& posteriors,
                   const DoubleArray2D& fwdLattice,
                   const DoubleArray2D& bwdLattice);

private:
    int numFeatures_;
    DoubleArray1D posts_;
    DoubleArray2D obs_;
    DoubleArray2D obs2_;


};

} // namespace
#endif
