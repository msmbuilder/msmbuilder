/*****************************************************************/
/*    Copyright (c) 2014, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#include "HMMEstep.hpp"
#include "HMMSufficientStats.hpp"
#include "kernels/TransitionCounts.hpp"

namespace Mixtape {

void HMMSufficientStats::increment(const HMMEstep* estep,
                                   const FloatArray2D& seq,
                                   const FloatArray2D& frameLogProb,
                                   const FloatArray2D& posteriors,
                                   const DoubleArray2D& fwdLattice,
                                   const DoubleArray2D& bwdLattice)
{
    double logProb = 0;
    const int length = seq.shape()[0];

    DoubleArray2D transCounts(boost::extents[numStates_][numStates_]);
    transitioncounts<double>(&fwdLattice[0][0], &bwdLattice[0][0], &estep->logTransmat_[0][0], &frameLogProb[0][0], length, numStates_, &transCounts[0][0], &logProb);

    // increment transcounts
    typedef DoubleArray2D::index index;
    for (index i = 0; i < numStates_; i++)
        for (index j = 0; j < numStates_; j++)
            transCounts_[i][j] += transCounts[i][j];

    // increment logprob
    logProb_ += logProb;

    numObservations_ += length;

    for (index i = 0; i < numStates_; i++)
        startCounts_[i] += posteriors[0][i];
}

}  // namespace Mixtape
