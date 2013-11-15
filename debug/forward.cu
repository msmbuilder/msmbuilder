#include <stdio.h>
#include <math.h>
#define N_STATES 32
#define N_OBSERVATIONS 3
#define N_TRAJS 1



int host_forward(const float* __restrict__ log_transmat_T, const float* __restrict__ log_startprob,
                 const float* __restrict__ frame_logprob, float* __restrict__ fwdlattice)
{
    int s, t, i, j;
    float work_buffer[N_STATES];
    for (s = 0; s < N_TRAJS; s++) {
        for (j = 0; j < N_STATES; j++)
            fwdlattice[s*N_OBSERVATIONS*N_STATES + 0*N_STATES + j] = log_startprob[j]
                                                                     + frame_logprob[s*N_OBSERVATIONS + 0*N_STATES + j];

        for (t = 1; t < N_OBSERVATIONS; t++) {
            for (j = 0; j < N_STATES; j++) {
                for (i = 0; i < N_STATES; i++)
                    work_buffer[i] = fwdlattice[s*N_OBSERVATIONS*N_STATES + (t-1)*N_STATES + i] + log_transmat_T[j*N_STATES + i];
                fwdlattice[s*N_OBSERVATIONS*N_STATES + t*N_STATES + j] = logsumexp(work_buffer, N_STATES)
                                                                         + frame_logprob[s*N_OBSERVATIONS*N_STATES + t*N_STATES+j];
            }
        }

    }
    return 1;
}



__global__ void warplogsumexpdriver(float* __restrict__ result)
{
    const unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int lid = gid % 32;
    float value = -1.0f * (float) lid;
    warplogsumexp(value, result);
}


__global__ void device_forward(const float* __restrict__ log_transmat_T, const float* __restrict__ log_startprob,
                               const float* __restrict__ frame_logprob, float* __restrict__ fwdlattice)
{

    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int warpWidth = 32;
    const unsigned int lid = gid % warpWidth;
    unsigned int t, j;
    float work_buffer = 0;

    while(gid < N_TRAJS*N_STATES) {
        // s = the trajectory index we're working on. (up to N_TRAJS)
        const int s = gid / warpWidth;
        // initial t=0 condition of fwdlattice
        fwdlattice[s*N_OBSERVATIONS*N_STATES + 0*N_STATES + lid] = log_startprob[lid] +
                                                                   frame_logprob[s*N_OBSERVATIONS*N_STATES + 0*N_STATES + lid];

        for(t = 1; t < N_OBSERVATIONS; t++) {
            for(j = 0; j < N_STATES; j++) {
                work_buffer = fwdlattice[s*N_OBSERVATIONS*N_STATES + (t-1)*N_STATES + lid] + log_transmat_T[j*N_STATES+lid];
                warplogsumexp(work_buffer, &work_buffer);
                if (lid == 0) {
                    fwdlattice[s*N_OBSERVATIONS*N_STATES + t*N_STATES+j] = work_buffer +
                                                                           frame_logprob[s*N_OBSERVATIONS*N_STATES + t*N_STATES + j];
                }
            }
        }

        gid += gridDim.x*blockDim.x;
    }
}



int main() {
    int i, j, k;

    float* log_transmat_T = (float*) malloc(N_STATES * N_STATES * sizeof(float));
    if (log_transmat_T == NULL)
        exit(EXIT_FAILURE);
    for (i = 0; i < N_STATES; i++)
        for (j = 0; j < N_STATES; j++)
            log_transmat_T[i*N_STATES + j] = 1.0/(i+j+1);

    float* log_startprob = (float*) malloc(N_STATES * sizeof(float));
    if (log_startprob == NULL)
        exit(EXIT_FAILURE);
    for (i = 0; i < N_STATES; i++)
        log_startprob[i] = 0.12;

    float* frame_logprob = (float*) malloc(N_TRAJS * N_OBSERVATIONS * N_STATES * sizeof(float));
    if (frame_logprob == NULL)
        exit(EXIT_FAILURE);
    for (i = 0; i < N_TRAJS; i++)
        for (j = 0; j < N_OBSERVATIONS; j++)
            for (k = 0; k < N_STATES; k++)
                frame_logprob[i*N_OBSERVATIONS*N_STATES + j*N_STATES + k] = 1.0 / (j+1)*k;

    float* fwdlattice = (float*) malloc(N_TRAJS * N_OBSERVATIONS * N_STATES * sizeof(float));
    if (fwdlattice == NULL)
        exit(EXIT_FAILURE);
    float* fwdlattice2 = (float*) malloc(N_TRAJS * N_OBSERVATIONS * N_STATES * sizeof(float));
    if (fwdlattice2 == NULL)
        exit(EXIT_FAILURE);




    float *device_log_transmat_T, *device_log_startprob, *device_frame_logprob, *device_fwdlattice;
    cudaMalloc((void **) &device_log_transmat_T, N_STATES*N_STATES*sizeof(float));
    cudaMalloc((void **) &device_log_startprob, N_STATES*sizeof(float));
    cudaMalloc((void **) &device_frame_logprob, N_OBSERVATIONS*N_STATES*sizeof(float));
    cudaMalloc((void **) &device_fwdlattice, N_OBSERVATIONS*N_STATES*sizeof(float));

    cudaMemcpy(device_log_transmat_T, log_transmat_T, N_STATES*N_STATES*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_log_startprob, log_startprob, N_STATES*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_frame_logprob, frame_logprob, N_OBSERVATIONS*N_STATES*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_fwdlattice, fwdlattice, N_OBSERVATIONS*N_STATES*sizeof(float), cudaMemcpyHostToDevice);


    host_forward(log_transmat_T, log_startprob, frame_logprob, fwdlattice);
    device_forward<<<1, 32>>>(device_log_transmat_T, device_log_startprob, device_frame_logprob, device_fwdlattice);
    cudaMemcpy(fwdlattice2, device_fwdlattice, N_OBSERVATIONS*N_STATES*sizeof(float), cudaMemcpyDeviceToHost);

    for (i = 0; i < N_TRAJS; i++)
        for (j = 0; j < N_OBSERVATIONS; j++)
            for (int k = 0; k < N_STATES; k++)
                printf("fwdlatice[%d, %d, %d] = %f, %f \n", i, j, k, fwdlattice[i*N_OBSERVATIONS*N_STATES + j*N_STATES + k],
                       fwdlattice2[i*N_OBSERVATIONS*N_STATES + j*N_STATES + k]);


    /*
    float* device_out;
    float* out = (float*) malloc(32*sizeof(float));
    cudaMalloc((void **) &device_out, 32*sizeof(float));
    warplogsumexpdriver<<<1, 32>>>(device_out);
    cudaMemcpy(out, device_out, 1*sizeof(float), cudaMemcpyDeviceToHost);
    printf("out[%d] = %f\n", 0, out[0]);

    float hostdata[32];
    for (i = 0; i < 32; i++)
        hostdata[i] = -i;
    printf("host out = %f\n", logsumexp(hostdata, 32));
    */
}
