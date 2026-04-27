#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>

void initializeMatrix(float* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = rand() % 10;
    }
}

#endif
