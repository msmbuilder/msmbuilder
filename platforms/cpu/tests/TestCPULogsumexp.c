
#include <stdlib.h>

#include <emmintrin.h>
#include "logsumexp.h"
#include "assertions.h"

int main() {
    int i;
    float buf[33];
    __m128 bufv[8];
    for (i = 0; i < 33; i++)
        buf[i] = pow(2.0, -i);
    for (i = 0; i < 8; i++)
        bufv[i] = _mm_loadu_ps(buf + 4*i);

    // computed with python: scipy.misc.logsumexp(2.0**(-np.arange(N)))
    float correct1 = 1.0;
    float correct2 = 1.4740769841801067;
    float correct3 = 1.7318375667944002;
    float correct4 = 1.9145929843689586;
    float correct5 = 2.0603442726085728;
    float correct31 = 3.5237638716431539;
    float correct32 = 3.5528256922619614;
    float correct33 = 3.5810667210265881;
    ASSERT_TOL(logsumexp(buf, 1), correct1, 1e-6);
    ASSERT_TOL(logsumexp(buf, 2), correct2, 1e-6);
    ASSERT_TOL(logsumexp(buf, 3), correct3, 1e-6);
    ASSERT_TOL(logsumexp(buf, 4), correct4, 1e-6);
    ASSERT_TOL(logsumexp(buf, 5), correct5, 1e-2);
    ASSERT_TOL(logsumexp(buf, 31), correct31, 1e-6);
    ASSERT_TOL(logsumexp(buf, 32), correct32, 1e-6);
    ASSERT_TOL(logsumexp(buf, 33), correct33, 1e-6);
    ASSERT_TOL(_mm_logsumexp(bufv, 8), correct32, 1e-6);

    return 1;
}
