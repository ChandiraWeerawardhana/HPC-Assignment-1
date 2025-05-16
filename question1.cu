#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

__global__ void vecAdd(float* A, float* B, float* C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    const int N = 1024 * 10;
    float *a, *b, *c;
    float *dev_a, *dev_b, *dev_c;

    size_t size = N * sizeof(float);

    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        a[i] = rand() % 1000;
        b[i] = rand() % 1000;
    }

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    bool error = false;

    for (int threads = 1; threads <= 2048; threads++) {

        vecAdd<<<1, threads>>>(dev_a, dev_b, dev_c, N);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            if (!error) {
                printf("\nKernel launch failed at %d threads per block.\n", threads);
                printf("Therefore, the maximum supported threads per block is %d.\n", threads - 1);
                error = true;
            }
            break;
        }
    }

    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    printf("\nSample output for N=%d elements:\n", N);
    for (int i = 0; i < 10; i++) {
        printf("%.1f + %.1f = %.1f\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);

    return 0;
}
