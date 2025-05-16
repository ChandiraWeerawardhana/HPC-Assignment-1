#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

__global__ void vecAdd3(float* A, float* B, float* C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    const int threadsPerBlock = 1024;
    const int N = 1024 * 100000;
    size_t size = N * sizeof(float);
    float *a, *b, *c;
    float *dev_a, *dev_b, *dev_c;

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

    cudaEvent_t startUpload, endUpload, startKernel, endKernel, startDownload, endDownload;
    cudaEventCreate(&startUpload);
    cudaEventCreate(&endUpload);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&endKernel);
    cudaEventCreate(&startDownload);
    cudaEventCreate(&endDownload);

    //to measure upload time (Host to Device)
    cudaEventRecord(startUpload);
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    cudaEventRecord(endUpload);
    //to measure kernel execution time
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    cudaEventRecord(startKernel);
    vecAdd3<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, N);
    cudaEventRecord(endKernel);

    //to measure download time (Device to Host)
    cudaEventRecord(startDownload);
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(endDownload);

    cudaEventSynchronize(endUpload);
    cudaEventSynchronize(endKernel);
    cudaEventSynchronize(endDownload);

    float timeUpload, timeKernel, timeDownload;
    cudaEventElapsedTime(&timeUpload, startUpload, endUpload);
    cudaEventElapsedTime(&timeKernel, startKernel, endKernel);
    cudaEventElapsedTime(&timeDownload, startDownload, endDownload);

    float timeTotal = timeUpload + timeKernel + timeDownload;

    printf("Upload Time (Host to Device): %.4f ms\n", timeUpload);
    printf("Kernel Execution Time: %.4f ms\n", timeKernel);
    printf("Download Time (Device to Host): %.4f ms\n", timeDownload);
    printf("Total Time Taken: %.4f seconds\n", timeTotal/1000);

    for (int i = 0; i < 10; i++) {
        printf("%.1f + %.1f = %.1f\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);

    cudaEventDestroy(startUpload);
    cudaEventDestroy(endUpload);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(endKernel);
    cudaEventDestroy(startDownload);
    cudaEventDestroy(endDownload);

    return 0;
}
