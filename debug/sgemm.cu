#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cublas_v2.h>


int main() {
    int n_frames = 100;
    int n_features = 20;
    int n_states = 30;
    float* trajs = (float*) malloc(n_frames*n_features*sizeof(float));
    float* posts = (float*) malloc(n_frames*n_states*sizeof(float));
    float* results = (float*) malloc(n_states*n_features*sizeof(float));
    float* results2 = (float*) malloc(n_states*n_features*sizeof(float));
    float* d_trajs, *d_posts, *d_result;

    for (int i = 0; i < n_frames; i++)
        for (int j = 0; j < n_features; j++)
            trajs[i*n_features + j] = drand48();
    for (int i = 0; i < n_frames; i++)
        for (int j = 0; j < n_states; j++)
            posts[i*n_states + j] = drand48();
    for (int i = 0; i < n_states; i++)
        for (int j = 0; j < n_features; j++)
            results[i*n_features+j] = 0;
    
    double correct00 = 0.0;
    double correct01 = 0.0;
    double correct11 = 0.0;
    for (int i = 0; i < n_frames; i++) {
        correct00 += posts[i*n_states + 0] * trajs[i*n_features + 0];
        correct01 += posts[i*n_states + 0] * trajs[i*n_features + 1];
        correct11 += posts[i*n_states + 1] * trajs[i*n_features + 1];

    }

    cudaMalloc((void**) &d_trajs, n_frames*n_features*sizeof(float));
    cudaMalloc((void**) &d_posts, n_frames*n_states*sizeof(float));
    cudaMalloc((void**) &d_result, n_states*n_features*sizeof(float));
    cudaMemcpy(d_trajs, trajs, n_frames*n_features*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posts, posts, n_frames*n_states*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, results, n_states*n_features*sizeof(float), cudaMemcpyHostToDevice);
    
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);

    float alpha = 1.0;
    float beta = 1.0;
    int p1[] = {n_frames, n_features, n_states};
    int p2[] = {n_frames, n_features, n_states};
    int p3[] = {n_frames, n_features, n_states};
    int p4[] = {n_frames, n_features, n_states};
    int p5[] = {n_frames, n_features, n_states};
    int p6[] = {n_frames, n_features, n_states};
    for (int i = 0; i < 3; i++)
        for (int ii = 0; ii < 3; ii++)
            for (int j = 0; j < 3; j++)
                for (int jj = 0; jj < 3; jj++)
                    for (int k = 0; k < 3; k++)
                        for (int kk = 0; kk < 3; kk++) {
                            cudaMemcpy(d_result, results, n_states*n_features*sizeof(float), cudaMemcpyHostToDevice);
                            status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, p1[i], p2[ii], p3[j],
                                                 &alpha, d_trajs, p4[jj], d_posts, p5[k],
                                                 &beta, d_result, p6[kk]);
                            
                            cudaMemcpy(results2, d_result, n_states*n_features*sizeof(float), cudaMemcpyDeviceToHost);
                            if (fabs(results2[0]-correct00) < 1e-5 && fabs(results2[1]-correct01) < 1e-5 && fabs(results2[n_features*1+1] - correct11) < 1e-5)
                                printf("Correct result when arguments are (%d, %d, %d, %d, %d, %d)\n",
                                       p1[i], p2[ii], p3[j], p4[jj], p5[k], p6[kk]);
    
            }

    cublasDestroy(handle);
    printf("sizeof(handle)=%d", sizeof(handle));
}
