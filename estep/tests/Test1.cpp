#include <iostream>
#include <cassert>
#include "DiagonalGaussianHMMEstep.hpp"

#define ASSERT_EQUAL_TOL(expected, found, tol) {double _scale_ = std::fabs(expected) > 1.0 ? std::fabs(expected) : 1.0; if (!(std::fabs((expected)-(found))/_scale_ <= (tol))) throw exception();};

using namespace Mixtape;
using namespace std;


void assertArrayEqual(const FloatArray2D& a1, const FloatArray2D& a2) {
    assert(a1.shape()[0] == a2.shape()[0]);
    assert(a1.shape()[1] == a2.shape()[1]);

    for (int i = 0; i < a1.shape()[0]; i++)
        for (int j = 0; j < a1.shape()[1]; j++)
            ASSERT_EQUAL_TOL(a1[i][j], a2[i][j], 1e-6);
}

void assertArrayEqual(const DoubleArray2D& a1, const FloatArray2D& a2) {
    assert(a1.shape()[0] == a2.shape()[0]);
    assert(a1.shape()[1] == a2.shape()[1]);

    for (int i = 0; i < a1.shape()[0]; i++)
        for (int j = 0; j < a1.shape()[1]; j++)
            ASSERT_EQUAL_TOL(a1[i][j], a2[i][j], 1e-6);
}

void assertArrayEqual(const DoubleArray2D& a1, const DoubleArray2D& a2) {
    assert(a1.shape()[0] == a2.shape()[0]);
    assert(a1.shape()[1] == a2.shape()[1]);

    for (int i = 0; i < a1.shape()[0]; i++)
        for (int j = 0; j < a1.shape()[1]; j++)
            ASSERT_EQUAL_TOL(a1[i][j], a2[i][j], 1e-6);
}





int main() {
    DoubleArray2D transMat(boost::extents[2][2]);
    DoubleArray1D startProb(boost::extents[2]);
    DoubleArray2D means(boost::extents[2][1]);
    DoubleArray2D variances(boost::extents[2][1]);
    FloatArray2D sequence(boost::extents[10][1]);

    transMat[0][0] = 0.7;  transMat[0][1] = 0.3;
    transMat[1][0] = 0.4;  transMat[1][1] = 0.6;
    startProb[0] = 0.6; startProb[1] = 0.4;
    means[0][0] = 0.0; means[1][0] = 2.0;
    variances[0][0] = 1.0; variances[1][0] = 1.0;
    for (int i = 0; i < 10; i++)
        sequence[i][0] = sin(i) + 1;

    DiagonalGaussianHMMEstep estep = DiagonalGaussianHMMEstep(2, transMat, startProb, 1, means, variances);
    estep.addSequence(sequence);


    /* training data from python */
    FloatArray2D refFrameLogProb(boost::extents[10][2]);
    refFrameLogProb[0][0] = -1.418939;  refFrameLogProb[0][1] = -1.418939;
    refFrameLogProb[1][0] = -2.614446;  refFrameLogProb[1][1] = -0.931504;
    refFrameLogProb[2][0] = -2.741647;  refFrameLogProb[2][1] = -0.923052;
    refFrameLogProb[3][0] = -1.570016;  refFrameLogProb[3][1] = -1.287776;
    refFrameLogProb[4][0] = -0.948511;  refFrameLogProb[4][1] = -2.462116;
    refFrameLogProb[5][0] = -0.919782;  refFrameLogProb[5][1] = -2.837631;
    refFrameLogProb[6][0] = -1.178560;  refFrameLogProb[6][1] = -1.737391;
    refFrameLogProb[7][0] = -2.291741;  refFrameLogProb[7][1] = -0.977768;
    refFrameLogProb[8][0] = -2.897712;  refFrameLogProb[8][1] = -0.918995;
    refFrameLogProb[9][0] = -1.915978;  refFrameLogProb[9][1] = -1.091741;
    DoubleArray2D refForward(boost::extents[10][2]);
    refForward[0][0] = -1.929764;  refForward[0][1] = -2.335229;
    refForward[1][0] = -4.578112;  refForward[1][1] = -3.217943;
    refForward[2][0] = -6.504952;  refForward[2][1] = -4.531101;
    refForward[3][0] = -6.799790;  refForward[3][1] = -6.262548;
    refForward[4][0] = -7.422953;  refForward[4][1] = -8.979160;
    refForward[5][0] = -8.585605;  refForward[5][1] = -11.112585;
    refForward[6][0] = -10.076194;  refForward[6][1] = -11.378721;
    refForward[7][0] = -12.580216;  refForward[7][1] = -11.823760;
    refForward[8][0] = -15.038200;  refForward[8][1] = -13.042782;
    refForward[9][0] = -15.661615;  refForward[9][1] = -14.579581;
    DoubleArray2D refBackward(boost::extents[10][2]);
    refBackward[0][0] = -13.114006;  refBackward[0][1] = -12.586475;
    refBackward[1][0] = -11.686156;  refBackward[1][1] = -11.218921;
    refBackward[2][0] = -9.849801;  refBackward[2][1] = -9.891946;
    refBackward[3][0] = -8.239344;  refBackward[3][1] = -8.663277;
    refBackward[4][0] = -6.994044;  refBackward[4][1] = -7.418369;
    refBackward[5][0] = -5.777281;  refBackward[5][1] = -5.800654;
    refBackward[6][0] = -4.575883;  refBackward[6][1] = -4.095302;
    refBackward[7][0] = -3.243900;  refBackward[7][1] = -2.706240;
    refBackward[8][0] = -1.590970;  refBackward[8][1] = -1.346081;
    refBackward[9][0] = 0.000000;  refBackward[9][1] = 0.000000;
    FloatArray2D refPosteriors(boost::extents[10][2]);
    refPosteriors[0][0] = 0.469522;  refPosteriors[0][1] = 0.530478;
    refPosteriors[1][0] = 0.138548;  refPosteriors[1][1] = 0.861452;
    refPosteriors[2][0] = 0.126562;  refPosteriors[2][1] = 0.873438;
    refPosteriors[3][0] = 0.471703;  refPosteriors[3][1] = 0.528297;
    refPosteriors[4][0] = 0.878738;  refPosteriors[4][1] = 0.121262;
    refPosteriors[5][0] = 0.927597;  refPosteriors[5][1] = 0.072403;
    refPosteriors[6][0] = 0.694649;  refPosteriors[6][1] = 0.305351;
    refPosteriors[7][0] = 0.215157;  refPosteriors[7][1] = 0.784843;
    refPosteriors[8][0] = 0.096189;  refPosteriors[8][1] = 0.903811;
    refPosteriors[9][0] = 0.253121;  refPosteriors[9][1] = 0.746879;
    FloatArray2D refTrans(boost::extents[2][2]);
    refTrans[0][0] = 2.446473;  refTrans[0][1] = 1.572192;
    refTrans[1][0] = 1.355792;  refTrans[1][1] = 3.625544;
    FloatArray1D refPosts(boost::extents[2]);
    refPosts[0] = 4.271786;  refPosts[1] = 5.728214;
    FloatArray2D refObs(boost::extents[2][1]);
    refObs[0][0] = 3.162234;
    refObs[1][0] = 8.792976;
    FloatArray2D refObs2(boost::extents[2][1]);
    refObs2[0][0] = 3.905322;
    refObs2[1][0] = 14.710568;
    FloatArray1D refStart(boost::extents[2]);
    refStart[0] = 0.469522;  refStart[1] = 0.530478;


    HMMSufficientStats* stats = estep.execute();
    DiagonalGaussianHMMSufficientStats* dstats = static_cast<DiagonalGaussianHMMSufficientStats*>(stats);
    if (dstats == NULL)
        throw MixtapeException("sdfbhklsdbjklsdf");


    assertArrayEqual(estep.emissionLogLikelihood(sequence), refFrameLogProb);
    assertArrayEqual(estep.forwardPass(refFrameLogProb), refForward);
    assertArrayEqual(estep.backwardPass(refFrameLogProb), refBackward);
    assertArrayEqual(estep.computePosteriors(refForward, refBackward), refPosteriors);
    assertArrayEqual(dstats->transCounts(), refTrans);
    assertArrayEqual(dstats->obs(), refObs);
    assertArrayEqual(dstats->obs2(), refObs2);
    ASSERT_EQUAL_TOL(dstats->posts()[0], refPosts[0], 1e-6);
    ASSERT_EQUAL_TOL(dstats->posts()[1], refPosts[1], 1e-6);
    ASSERT_EQUAL_TOL(dstats->startCounts()[0], refStart[0], 1e-6);
    ASSERT_EQUAL_TOL(dstats->startCounts()[1], refStart[1], 1e-6);
    assert(dstats->numObservations() == 10);

    cout << "Done" << endl;
    delete stats;
    return 1;
}
