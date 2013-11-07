#include "math.h"
#include "cephes.h"
#include "stdio.h"
#include "gammautils.h"

double invpsi(double y) {
    /*
    Inverse digamma (psi) function.  The digamma function is the
    derivative of the log gamma function.
    
    Adapted from matlab code in PMTK (https://code.google.com/p/pmtk3),
    copyright Kevin Murphy and available under the MIT license.
    https://code.google.com/p/pmtk3/source/browse/trunk/misc/fastfit/inv_digamma.m?r=773
    */

    /* never more than 5 iterations required */
    static double const TOLERANCE = 1e-6;
    static double const MAX_ITERS = 5;
    static double const PSI_1 = -0.57721566490153287;  // =psi(1.0)

    int i;
    double x, delta_x;
    if (y <= -2.22)
        x = 1.0 / (y * PSI_1);
    else
        x = exp(y) + 0.5;

    /* Newton iteration to solve digamma(x)-y = 0 */
    for (i = 0; i < MAX_ITERS; i++) {
        delta_x = (psi(x) - y) / zeta(2, x);
        if (fabs(delta_x) < TOLERANCE)
            return x - delta_x;
        x -= delta_x;
    }
    return x;
}


double weightlogsumexp(double nums[], double weight[], size_t ct) {
  double max_exp = nums[0], sum = 0.0;
  size_t i;

  for (i = 1 ; i < ct ; i++)
    if (nums[i] > max_exp)
      max_exp = nums[i];

  for (i = 0; i < ct ; i++)
    sum += weight[i] * exp(nums[i] - max_exp);

  return log(sum) + max_exp;
}
