/** 
 * There's really no portable way to link against BLAS's SDOT in a cross
 * platform way, at least on OSX. Aparently the Accelerate framework is
 * compiled with the g77 ABI, which means that the return value of their
 * SDOT is actually a double, even though it's **s**dot. But other BLASs
 * use the gfortran ABI so you have a float return value. Scipy handles
 * all of this at compile time and builds wrappers, but the _cpointer that
 * it exposes to SDOT doesn't have any of these wrappers. So all of the
 * dynamism that we're tying to achieve by using these function pointers
 * is for naught if we can't trust the ABI.
 * https://groups.google.com/forum/#!topic/cython-users/V_DR1xi5Ang
 *
 * The easiest thing is just to supply our own damn version
 */

float sdot_(const int N, const float* x, const float* y)
{
    int i;
    float dot1 = 0, dot2 = 0, dot3 = 0, dot4 = 0;

    for (i = 0; i < N-4; i += 4) {
        dot1 += x[i] * y[i];
        dot2 += x[i + 1] * y[i + 1];
        dot3 += x[i + 2] * y[i + 2];
        dot4 += x[i + 3] * y[i + 3];
    }

    for (; i < N; i++) {
        dot1 += x[i] * y[i];
    }

    return dot1 + dot2 + dot3 + dot4;
}
