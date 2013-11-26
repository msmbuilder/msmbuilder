#include "stdio.h"
#include "staticassert.cuh"

/**
 * Compute a set of specialized matrix-matrix products of the form
 * A^T x B for C-major matrices with *many* more rows than columns.
 *
 * Parameters
 * ----------
 * A : matrix in contiguous c order of shape [nRows, nColsA] where
 *     nRows >> nColsA
 * B : matrix in contiguous c order of shape [nRows, nColsB] where
 *     nRows >> nColsB
 *
 * Template Parameters
 * ------------------- 
 * W : In each iteration, we use team of W^2 theads to load a W*W block from
 *     A and B into shared memory. We're processing W rows and W columns
 *     at a time. It makes the most sense for W to be the largest power of
 *     two less than min(nColsA, nColsB)
 * BLOCK_SIZE : The actual dimension of the block that this kernel is
 *              invoked with. Note, you should only use a 1-dimensional
 *              block size.
 *
 * Returns
 * -------
 * C1 : matrix in contiguous C order of shape [nColsA, nColsB]
 *      will be incremented by the matrix product (A^T) dot (B).
 * C2 : matrix in contiguous C order of shape [nColsA, nColsB]
 *      will be incremennted by  the matrix product (A^T) dot (B.*2)
 *      where B.*2 denotes the element-wise square of B.
 * D  : matrix in contiguous C order of shape [nColsA] containing
 *      the column sums of A.
 *
 * Notes
 * -----
 * The code is optimizd for nColsB > nColsA, but can be changed
 * the other way by reversing the order of two loops.
 *
 * Equivalent Python Code
 * ----------------------
 * >>> C1 = np.dot(A.T, B)
 * >>> C2 = np.dot(A.T, B**2)
 * >>> D = np.sum(A, axis=1)
 */
template <unsigned int W, unsigned int BLOCK_SIZE>
__global__ void sufficientstatistics(
const float* A,
const float* B,
const int nRows,
const int nColsA,
const int nColsB,
float* C1,
float* C2,
float* D)
{
    static_assert<(W*W <= BLOCK_SIZE)>::valid_expression();

    // The columns of A and B are grouped into groups of W, and these are the
    // indices over them.
    unsigned int aColBlock, bColBlock;
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

    // During each other loop iteration, we use W^2 threads to processes one
    // block along nRows and all nColsA/W * nColsB/W blocks along the two
    // column dimensions.
    while (gid/(W*W) < (nRows+W-1)/W) {
        // When W^2 < BLOCK_SIZE, we need to make sure that the size-W^2
        // teams processing an individual block don't overwrite each
        // other's shared memory by giving each team an index within the block
        const unsigned int teamid = (gid % BLOCK_SIZE) / (W*W);
        const unsigned int rowBlock = gid / (W*W);
        const unsigned int lid = gid % (W*W);
        const unsigned int tRow = lid / W;
        const unsigned int tCol = lid % W;
        // To keep track of spillover, when nRows is not a multple of W
        // we do an extra iteration at the end, but where the invalid
        // entries will just be set to zero.
        const bool validRow = (rowBlock*W+tRow) < nRows;

        __shared__ float As[BLOCK_SIZE/(W*W)][W][W];
        __shared__ float Bs[BLOCK_SIZE/(W*W)][W][W];

        // You could reverse whether the loop over the A blocks or
        // the blocks B is on the outside of the double loop, based
        // on whether you expect A or B to have more columns
        for (aColBlock = 0; aColBlock < (nColsA+W-1)/W; aColBlock++) {
            const bool validColA = (aColBlock*W + tCol) < nColsA;
            if (W > 4)
                __syncthreads();
            As[teamid][tRow][tCol] = (validColA && validRow) ? A[(rowBlock*W*nColsA) + tRow*nColsA + (aColBlock*W + tCol)] : 0.0f;
        for (bColBlock = 0; bColBlock < (nColsB+W-1)/W; bColBlock++) {
            const bool validColB = (bColBlock*W + tCol) < nColsB;
            if (W > 4)
                __syncthreads();
            Bs[teamid][tRow][tCol] = (validColB && validRow) ? B[(rowBlock*W*nColsB) + tRow*nColsB + (bColBlock*W + tCol)] : 0.0f;
            // this is not required when W <= 4, because we're 
            // implicitly warp-cynchronous at that point
            if (W > 4)
                __syncthreads();  


            float Ds  = 0.0f;
            float C1s = 0.0f;
            float C2s = 0.0f;
            #pragma unroll
            for (int k = 0; k < W; k++) {
                Ds  += As[teamid][k][tRow];
                C1s += As[teamid][k][tRow] * Bs[teamid][k][tCol];
                C2s += As[teamid][k][tRow] * (Bs[teamid][k][tCol] * Bs[teamid][k][tCol]);
           }

            const unsigned int offset = (tRow+aColBlock*W)*nColsB + (bColBlock*W + tCol);     
            if (offset < nColsA * nColsB) {
            atomicAdd(C1 + offset, C1s);
            atomicAdd(C2 + offset, C2s);
            if ((tCol == 0) && (bColBlock == 0))
                // By computing the Ds term on every thread, we've done it too many times.
                // but since the rate limiting step is memory access and we've got it here
                // we mind as well
                atomicAdd(D + tRow + aColBlock*W, Ds);
            }
       } }
    gid += gridDim.x*blockDim.x;
    }
}
