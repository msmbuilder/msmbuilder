from jinja2 import Template
tpl = Template('''#include <stdio.h>

__global__ void log_diag_mvn_likelihood(
const float* __restrict__ sequences,
const float* __restrict__ means,
const float* __restrict__ variances,
const float* __restrict__ logvariances,
const int n_samples,
const int n_states,
const int n_features,
float* __restrict__ loglikelihoods)
{
    /* W1 and W2 are the two chunk size parameters that control the dimensions
     * of the submatrices that are loaded into shared memory. The product W1*W2
     * currently must be equal to the size of the block that the kernel is
     * invoked with.
     *
     * From `sequences`, submatrices with W1 samples and W2 features are loaded,
     * and from means/variances/logvariances, submatrices with W1 states and
     * W2 features are loaded. In general, bigger blocks are more efficient but
     * only for bigger matrices, because spillover, where, for example,
     * n_features % W2 != 0 causes some inefficiency.
     *
     *
     * This kernel is optimized for n_samples >> (n_states ~ n_features) because
     * it deploys all of its threads along the n_samples dimension, and each thread
     * has to iterate in the blocked n_states/n_featues space, computing a 
     * total of (n_states/W1) * (n_features/W2) entries per thread gid iteration.
     */ 
    const unsigned int W1 = {{W1}};
    const unsigned int W2 = {{W2}};

    __shared__ float SEQ[W1][W2];
    __shared__ float MU[W1][W2];
    __shared__ float SIG2[W1][W2];
    __shared__ float LOGSIG2[W1][W2];
const float MINUS_HALF_LOG_2_PI = -0.91893853320467267f;
unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
while (gid/(W1*W2) < (n_samples+W1-1)/W1) {
    const unsigned int lid = gid % (W1*W2);
    const unsigned int loadRow = lid / W2;
    const unsigned int loadCol = lid % W2;
    const unsigned int samplesBlock = W1*(gid / (W1*W2));
    const bool validSample = (samplesBlock+loadRow) < n_samples;

for (unsigned int featuresBlock = 0; featuresBlock < ((n_features+W2-1)/W2)*W2; featuresBlock += W2) {
    const bool validFeature = (featuresBlock+loadCol) < n_features;
    SEQ[loadRow][loadCol] = (validSample && validFeature) ? sequences[(samplesBlock+loadRow)*n_features + (featuresBlock+loadCol)] : 0.0f;
for (unsigned int statesBlock = 0; statesBlock < ((n_states+W1-1)/W1)*W1; statesBlock += W1) {
    // Load 64 W1*W2 items into shared memory from the global arrays
    const bool validState = (statesBlock+loadRow) < n_states;
    if (validState && validFeature) {
        MU[loadRow][loadCol] = means[(statesBlock+loadRow)*n_features + (featuresBlock+loadCol)];
        SIG2[loadRow][loadCol] = variances[(statesBlock+loadRow)*n_features + (featuresBlock+loadCol)];
        LOGSIG2[loadRow][loadCol] = logvariances[(statesBlock+loadRow)*n_features + (featuresBlock+loadCol)];
    } else {
        // Use sig < 0 as a sentinel for the block containing an out of bound
        // index, which is necessary for the boundary blocks without making an
        // explicit epilogue to handle them.
        MU[loadRow][loadCol] = 0;
        SIG2[loadRow][loadCol] = -1;
        LOGSIG2[loadRow][loadCol] = 0;
    }
    __syncthreads();


    // Now, we need to compute W1^2 results, each of which is a sum of W2 entries
    // between the W1*W2 threads on this. So each thread needs to do W1 entries
    // in the sum.
    
    // Say W1=2, W2=32. There are four sums of 32 elements to compute. Each
    // sum is split up into W2/W1=16 chunks of 2 elements each.
    
    // thread 0 takes (0, 0) from 0 to W1
    // thread 1 takes (0, 0) from W1 to 2*W1
    // ... 
    // thread   takes (0, 0) from .. to W2
    // ...
    // takes (W1-1, W1-1) from W2-W1 to W2
    
    const unsigned int offset = lid % (W2 / W1);
    const unsigned int i = (lid / (W2/W1)) % W1;
    const unsigned int j = (lid / (W2/W1)) / W1;
    float temp = 0;
    #pragma unroll
    for (int k = W1*offset; k < W1*(offset+1); k++) {
        float f = SEQ[i][k] - MU[j][k];
        float loglike = MINUS_HALF_LOG_2_PI + -0.5f * (LOGSIG2[j][k] + (f*f / SIG2[j][k]));
        // using the sentinel to avoid summing in invalid indices
        temp += loglike * (SIG2[j][k] > 0);
    }
    atomicAdd(loglikelihoods + (samplesBlock+i)*n_states + statesBlock+j, temp);
} }
    gid += gridDim.x*blockDim.x;
}
}
''')
import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

def speed(n_samples, n_states, n_features, W1, W2):
    module = SourceModule(tpl.render(W1=W1, W2=W2))
    func = module.get_function('log_diag_mvn_likelihood')

    sequences = np.zeros((n_samples, n_features), dtype=np.float32)
    means = np.zeros((n_states, n_features), dtype=np.float32)
    variances = np.ones((n_states, n_features), dtype=np.float32)
    logvariances = np.ones((n_states, n_features), dtype=np.float32)
    loglikelihoods = np.zeros((n_samples, n_states), dtype=np.float32)

    N_THREADS = 4096

    start = cuda.Event()
    end = cuda.Event()


    start.record()
    func(cuda.In(sequences), cuda.In(means), cuda.In(variances), cuda.In(np.log(variances)),
         np.int32(n_samples), np.int32(n_states), np.int32(n_features),
         cuda.InOut(loglikelihoods),
         block=(W1*W2,1,1), grid=(N_THREADS/(W1*W2),1))
    end.record()
    end.synchronize()
    
    return {'n_samples': n_samples, 'n_states': n_states,
            'n_features': n_features, 'W1': W1, 'W2': W2,
            'time': np.around(start.time_till(end), decimals=3)}
    

def test():
    func = SourceModule(source).get_function('log_diag_mvn_likelihood')
    n_samples = 8
    n_states = 9
    n_features = 33
    np.random.seed(42)
    sequences = np.random.rand(n_samples, n_features).astype(np.float32)
    means = np.random.rand(n_states, n_features).astype(np.float32)
    variances = np.random.rand(n_states, n_features).astype(np.float32)
    loglikelihoods = np.zeros((n_samples, n_states), dtype=np.float32)
    
    func(cuda.In(sequences), cuda.In(means), cuda.In(variances), cuda.In(np.log(variances)),
         np.int32(n_samples), np.int32(n_states), np.int32(n_features),
         cuda.InOut(loglikelihoods),
         block=(64,1,1), grid=(1,1))

    print 'loglikelihoods'
    print loglikelihoods

    print 'sklearn'
    from sklearn.mixture.gmm import _log_multivariate_normal_density_diag
    r = _log_multivariate_normal_density_diag(sequences, means, variances)
    print r
    print np.abs(r - loglikelihoods) < 1e-4


if __name__ == '__main__':
    for n_states in [4, 8, 16, 32]:
        for n_features in [8, 16, 32, 64, 128]:
            best = None
            for W1 in [4, 8, 16, 32]:
                for W2 in [4, 8, 16, 32]:
                    if W1 <= W2:
                        r = speed(n_samples=1000000, n_states=n_states,
                                  n_features=n_features, W1=W1, W2=W2)
                        if best is None or r['time'] < best['time']:
                            best = r
            print 'n_states=%d, n_features=%d Opt W1=%d, W2=%d' % \
                (n_states, n_features, best['W1'], best['W2'])
