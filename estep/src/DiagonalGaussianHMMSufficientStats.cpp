/*****************************************************************/
/*    Copyright (c) 2014, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#include <cstdio>
#include "cblas.h"
#include "DiagonalGaussianHMMSufficientStats.hpp"
namespace Mixtape {


void DiagonalGaussianHMMSufficientStats::increment(const HMMEstep* estep,
                                                   const FloatArray2D& seq,
                                                   const FloatArray2D& frameLogProb,
                                                   const FloatArray2D& posteriors,
                                                   const DoubleArray2D& fwdLattice,
                                                   const DoubleArray2D& bwdLattice)
{
    HMMSufficientStats::increment(estep, seq, frameLogProb, posteriors, fwdLattice, bwdLattice);

    const float alpha = 1.0;
    const float beta = 1.0;
    const int length = seq.shape()[0];
    assert(seq.shape()[1] == numFeatures_);

    FloatArray2D obs(boost::extents[numStates_][numFeatures_]);
    FloatArray2D obs2(boost::extents[numStates_][numFeatures_]);
    FloatArray2D seq2(boost::extents[length][numFeatures_]);

    typedef FloatArray2D::index index;
    for (index i = 0; i < length; i++)
        for (index j = 0; j < numFeatures_; j++)
            seq2[i][j] = seq[i][j] * seq[i][j];

    sgemm_("N", "T", &numFeatures_, &numStates_, &length, &alpha, &seq[0][0],
           &numFeatures_, &posteriors[0][0], &numStates_, &beta, &obs[0][0],
           &numFeatures_);
    sgemm_("N", "T", &numFeatures_, &numStates_, &length, &alpha, &seq2[0][0],
           &numFeatures_, &posteriors[0][0], &numStates_, &beta, &obs2[0][0],
           &numFeatures_);

    // increment posts
    for (index j = 0; j < length; j++)
        for (index k = 0; k < numStates_; k++)
            posts_[k] += posteriors[j][k];

    // increment obs and obs**2
    for (index i = 0; i < numStates_; i++) {
        for (index j = 0; j < numFeatures_; j++) {
            obs_[i][j] += obs[i][j];
            obs2_[i][j] += obs2[i][j];
        }
    }
}


}  // namespace Mixtape
