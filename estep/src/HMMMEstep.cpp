/*****************************************************************/
/*    Copyright (c) 2014, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#include <vector>
#include <cmath>
#include <cassert>
#include "HMMEstep.hpp"
#include "HMMSufficientStats.hpp"
#include "kernels/forward.hpp"
#include "kernels/backward.hpp"
#include "kernels/posteriors.hpp"

namespace Mixtape {

HMMEstep::HMMEstep(const int numStates, const DoubleArray2D& transMat, const DoubleArray1D& startProb)
    : numStates_(numStates)
    , logStartProb_(boost::extents[numStates_])
    , logTransmat_(boost::extents[numStates_][numStates_])
    , logTransmat_T_(boost::extents[numStates_][numStates_])
{
    if (numStates_ != transMat.shape()[0] || numStates_ != transMat.shape()[1])
        throw MixtapeException("transmat has wrong shape");
    if (numStates_ != startProb.shape()[0])
        throw MixtapeException("startprob has wrong shape");


    typedef DoubleArray2D::index index;
    for (index i = 0; i < numStates_; i++) {
        for (index j = 0; j < numStates_; j++) {
            logTransmat_[i][j] = log(transMat[i][j]);
            logTransmat_T_[j][i] = log(transMat[i][j]);
        }
    }

    for (index i = 0; i < numStates_; i++)
        logStartProb_[i] = log(startProb[i]);
}

void HMMEstep::addSequence(const FloatArray2D& X) {
    const FloatArray2D* x = &X;
    sequences_.push_back(x);
}


HMMSufficientStats* HMMEstep::initializeSufficientStats() {
    HMMSufficientStats* stats = new HMMSufficientStats(numStates_);
    return stats;
}

DoubleArray2D HMMEstep::forwardPass(const FloatArray2D& frameLogProb) {
    assert(frameLogProb.shape()[1] == numStates_);
    DoubleArray2D forwardLattice(boost::extents[frameLogProb.shape()[0]][numStates_]);
    forward<double>(&logTransmat_T_[0][0], &logStartProb_[0], &frameLogProb[0][0], frameLogProb.shape()[0],
                    frameLogProb.shape()[1], &forwardLattice[0][0]);
    return forwardLattice;
}

DoubleArray2D HMMEstep::backwardPass(const FloatArray2D& frameLogProb) {
    assert(frameLogProb.shape()[1] == numStates_);
    DoubleArray2D backwardLattice(boost::extents[frameLogProb.shape()[0]][numStates_]);
    backward<double>(&logTransmat_[0][0], &logStartProb_[0], &frameLogProb[0][0], frameLogProb.shape()[0],
                    frameLogProb.shape()[1], &backwardLattice[0][0]);
    return backwardLattice;
}

FloatArray2D HMMEstep::computePosteriors(const DoubleArray2D& fwdLattice, const DoubleArray2D& bwdLattice) {
    assert(fwdLattice.shape()[0] == bwdLattice.shape()[0]);
    assert(fwdLattice.shape()[1] == bwdLattice.shape()[1]);
    assert(fwdLattice.shape()[1] == numStates_);

    FloatArray2D posteriors(boost::extents[fwdLattice.shape()[0]][numStates_]);
    compute_posteriors<double>(&fwdLattice[0][0], &bwdLattice[0][0], fwdLattice.shape()[0], numStates_, &posteriors[0][0]);
    return posteriors;
}

HMMSufficientStats* HMMEstep::execute() {
    HMMSufficientStats* stats = initializeSufficientStats();
    for (int i = 0; i < sequences_.size(); i++) {
        const FloatArray2D& sequence = *sequences_[i];

        FloatArray2D frameLogProb = emissionLogLikelihood(sequence);
        DoubleArray2D fwdLattice = forwardPass(frameLogProb);
        DoubleArray2D bwdLattice = backwardPass(frameLogProb);
        FloatArray2D posteriors = computePosteriors(fwdLattice, bwdLattice);

        stats->increment(this, sequence, frameLogProb, posteriors, fwdLattice, bwdLattice);
    }

    return stats;
}


}  // namespace Mixtape
