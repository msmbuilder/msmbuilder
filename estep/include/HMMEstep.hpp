/*****************************************************************/
/*    Copyright (c) 2014, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#ifndef MIXTAPE_HMMESTEP_H
#define MIXTAPE_HMMESTEP_H
#include <map>
#include <vector>
#include "typedefs.hpp"
#include "MixtapeException.hpp"


namespace Mixtape {

/* Forward declare this, becuase otherwise we get a recursive #include
   that breaks the build :( */
class HMMSufficientStats;

class HMMEstep {
friend class HMMSufficientStats;
public:
    /**
     * Construct a HMMEstep
     */
    HMMEstep(const int numStates, const DoubleArray2D& transMat, const DoubleArray1D& startProb);

    /**
     * Register a new data sequence with the model.
     *
     * Parameters
     * ----------
     * X : 2d array of shaep (numSamples, numFeatures)
     *     X is a pointer single timeseries, one of the data sequences to
     *     train the model on
     *
     * Memory Management
     * -----------------
     * HMMEstep does *not* assume ownership over the memory in X, and will
     * not free it. It's the caller's responsibility to make sure that X is
     * not cleaned up before HMMEstep is deleted.
     */
    void addSequence(const FloatArray2D& X);


    /**
     * Run the E-step of the HMM inference. The return value is a pointer to a
     * container with all of the sufficient statistics.
     *
     * Memory Management
     * -----------------
     * This function allocates a new object on the heap, and returns a pointer
     * to it. HMMEstep _abjures_ ownership of this pointer. It is the
     * caller's responsibility to clean it up. (The reason for this is that the
     * objected pointed to may be a subclass of HMMSufficientStats, and the
     * return value for all the subclasses needs to be set here at compile time,
     * so I couldn't get this to work with references).
     *
     * Returns
     * -------
     * A pointer to an instance of HMMSufficientStats. When you call execute
     * on a subclass of HMMEstep, the type of the return pointer will actually
     * be a subclass of HMMSufficientStats.
     */
    HMMSufficientStats* execute();

    /*-----------------------------------------------------------------------*/
    /* Virtual functions that must be implemented by subclasses              */
    /*-----------------------------------------------------------------------*/

    /**
     * Compute the log-likelihood of the emssion distribution evaluated at
     * each point in a data sequence.
     *
     * Parameters
     * ----------
     * X : 2d array of shaep (numSamples, numFeatures)
     *     X is a single timeseries, the "raw data" being fit by the model.
     *
     * Returns
     * -------
     * logl : 2d array of shape (numSamples, numStates)
     *     logl[i][j] should be the log-likelihood of state `j`'s emission
     *     distribution evaluated at the point X[i]
     */
    virtual FloatArray2D emissionLogLikelihood(const FloatArray2D& X) = 0;
    virtual HMMSufficientStats* initializeSufficientStats();


    /*-----------------------------------------------------------------------*/
    /* Methods for the forward-backward algorithim. Clients should not need  */
    /* to access these directly.                                             */
    /*-----------------------------------------------------------------------*/

    /**
     * Run the forward portion of the forward-backward algorithm
     */
    DoubleArray2D forwardPass(const FloatArray2D& frameLogProb);

    /**
     * Run the backward part of the forwarf-backward algorithim
     */
    DoubleArray2D backwardPass(const FloatArray2D& frameLogProb);

    /**
     * Compute the log-probability that the sequence was, at time `t`, in state `i`. This
     * is the last part of the forward-backward algorithim.
     */
    FloatArray2D computePosteriors(const DoubleArray2D& fwdLattice, const DoubleArray2D& bwdLattice);


protected:
    int numStates_;

private:
    std::vector<const FloatArray2D*> sequences_;
    DoubleArray2D logTransmat_;
    DoubleArray2D logTransmat_T_;
    DoubleArray1D logStartProb_;
};



} // namespace

#endif
