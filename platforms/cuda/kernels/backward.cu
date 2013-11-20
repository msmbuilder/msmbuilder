#include "logsumexp.cu"
#include <stdlib.h>

__global__ void backward4(
const float* __restrict__ log_transmat,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const int* __restrict__ sequence_lengths,
const int* __restrict__ cum_sequence_lengths,
const int n_trajs,
float* __restrict__ bwdlattice)
{
    const int n_states = 4;
    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    float work_buffer;
    int t;

    while (gid/16 < n_trajs) {
        const unsigned int hid = gid % 16;
        const unsigned int s = gid / 16;
        const float* _frame_logprob = frame_logprob + cum_sequence_lengths[s]*n_states;
        float* _bwdlattice = bwdlattice + cum_sequence_lengths[s]*n_states;
        
        if (hid < 4)
            _bwdlattice[(sequence_lengths[s]-1)*n_states + hid] = 0;
        
        for (t = sequence_lengths[s]-2; t >= 0; t--) {
            work_buffer = _bwdlattice[(t+1)*n_states + hid%4] + log_transmat[hid] + _frame_logprob[(t+1)*n_states + hid%4];
            work_buffer = logsumexp<4>(work_buffer);
            if (hid % 4 == 0)
                _bwdlattice[t*n_states + hid/4] = work_buffer;
        }
        gid += gridDim.x*blockDim.x;
    }
}


__global__ void backward8(
const float* __restrict__ log_transmat,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const int* __restrict__ sequence_lengths,
const int* __restrict__ cum_sequence_lengths,
const int n_trajs,
float* __restrict__ bwdlattice)
{
    const int n_states = 8;
    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    float work_buffer1, work_buffer2;
    int t;

    while (gid/32 < n_trajs) {
        const unsigned int lid = gid % 32;
        const unsigned int s = gid / 32;
        const float* _frame_logprob = frame_logprob + cum_sequence_lengths[s]*n_states;
        float* _bwdlattice = bwdlattice + cum_sequence_lengths[s]*n_states;
        const int i1 = lid / 8;
        const int i2 = lid / 8 + 4;
        const int j = lid % 8;

        if (lid < 8)
            _bwdlattice[(sequence_lengths[s]-1)*n_states + lid] = 0;

        for (t = sequence_lengths[s]-2; t >= 0; --t) {
            work_buffer1 = _bwdlattice[(t+1)*n_states + j] + log_transmat[i1*n_states + j] + _frame_logprob[(t+1)*n_states + j];
            work_buffer2 = _bwdlattice[(t+1)*n_states + j] + log_transmat[i2*n_states + j] + _frame_logprob[(t+1)*n_states + j];
            work_buffer1 = logsumexp<8>(work_buffer1);
            work_buffer2 = logsumexp<8>(work_buffer2);

            if (lid % 8 == 0) {
                _bwdlattice[t*n_states + i1] = work_buffer1;
                _bwdlattice[t*n_states + i2] = work_buffer2;
            }
        }
        gid += gridDim.x*blockDim.x;
    }
}