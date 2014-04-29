/*****************************************************************/
/*    Copyright (c) 2014, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#ifndef MIXTAPE_HMMSUFFICIENTSTATS_H
#define MIXTAPE_HMMSUFFICIENTSTATS_H
#include "typedefs.hpp"
#include "HMMEstep.hpp"

namespace Mixtape {
class HMMSufficientStats {
public:
    HMMSufficientStats(int numStates)
        : numStates_(numStates)
        , logProb_(0)
        , startCounts_(boost::extents[numStates])
        , transCounts_(boost::extents[numStates][numStates])
    { };

    /**
     * Total number of data points that we trained on. (sum across all
     * the sequences)
     */
    int numObservations() { return numObservations_; }

    /**
     * Overall log probability of the data
     */
    double logProb() { return logProb_; }

    /**
     * Expected number of sequences that started in each state
     */
    DoubleArray1D& startCounts() { return startCounts_; }

    /**
     * Expected number of transitions from state to state
     */
    DoubleArray2D& transCounts() { return transCounts_; }

    virtual void increment(const HMMEstep* estep,
                           const FloatArray2D& seq,
                           const FloatArray2D& frameLogProb,
                           const FloatArray2D& posteriors,
                           const DoubleArray2D& fwdLattice,
                           const DoubleArray2D& bwdLattice);

protected:
    int numStates_;

private:
    int numObservations_;
    double logProb_;
    DoubleArray1D startCounts_;
    DoubleArray2D transCounts_;
};


} // namespace
#endif
