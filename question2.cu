#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

__global__ void vecAdd2(float* A, float* B, float* C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    const int threadsPerBlock = 1024;
    const int N = 1024 * 100000;
    float *a, *b, *c, *dev_a, *dev_b, *dev_c;
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

    bool foundLimit = false;

    for (int blocks = 1; blocks <= 200000; blocks *= 2) {
        vecAdd2<<<blocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, N);
        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess) {
            printf("\nKernel launch failed at %d blocks per grid\n", blocks);
            printf("Therefore, maximum blocks per grid = %d\n", blocks - 1);
            printf("Maximum number of elements(threads) per grid = %d\n", (blocks-1)*1024);
            foundLimit = true;
            break;
        }
    }

    if (!foundLimit) {
        printf("\nSuccessfully tested up to maximum limit. No error occurred.\n");
    }

    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

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
