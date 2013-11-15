#include "utils.h"
#include <stdio.h>

void pprintarray(const float* a, int N, int M) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++)
            printf("%.3g  ", a[i*M+ j]);
        printf("\n");
    }
}
