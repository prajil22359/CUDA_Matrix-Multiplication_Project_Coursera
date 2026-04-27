#include <iostream>
#include <chrono>
#include "matrix_cpu.h"
#include "matrix_gpu.h"
#include "utils.h"

#define N 1024

int main() {
    size_t size = N * N * sizeof(float);

    float *A, *B, *C_cpu, *C_gpu;
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C_cpu = (float*)malloc(size);
    C_gpu = (float*)malloc(size);

    initializeMatrix(A, N);
    initializeMatrix(B, N);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(A, B, C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_time.count() << " seconds\n";

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(N / 16, N / 16);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    matrixMultiplyGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;
    std::cout << "GPU Time: " << gpu_time.count() << " seconds\n";

    cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);

    return 0;
}
