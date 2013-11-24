#ifndef MIXTAPE_CUDA_STATIC_ASSERT
#define MIXTAPE_CUDA_STATIC_ASSERT

/**
 * Device-side static assert
 */
template <bool b> struct static_assert{};
template <> struct static_assert<true> { __device__ static void valid_expression() {}; };
__inline__ __device__ int divU(int numerator, int denominator){
    return (numerator+denominator-1)/denominator;
}

#endif