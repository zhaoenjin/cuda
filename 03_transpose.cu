// Shared memory 矩阵转置：演示 block 内协作
#include <cstdio>
#include <cuda_runtime.h>

#define TILE_DIM 32

// 朴素版本：直接转置，内存访问不连续
__global__ void transposeNaive(float *out, float *in, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        out[x * height + y] = in[y * width + x];
    }
}

// 优化版本：使用 shared memory 作为中转，避免不连续的全局内存访问
__global__ void transposeShared(float *out, float *in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 避免 bank conflict

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 连续读入
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }

    __syncthreads();  // 确保整个 tile 都读完

    // 转置后连续写出
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    const int width = 1024;
    const int height = 1024;
    size_t size = width * height * sizeof(float);

    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    // 初始化
    for (int i = 0; i < width * height; i++) {
        h_in[i] = i;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((width + TILE_DIM - 1) / TILE_DIM,
              (height + TILE_DIM - 1) / TILE_DIM);

    // 创建 event 用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 朴素版本
    cudaEventRecord(start);
    transposeNaive<<<grid, block>>>(d_out, d_in, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msNaive;
    cudaEventElapsedTime(&msNaive, start, stop);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (h_out[x * height + y] != h_in[y * width + x]) {
                printf("Naive version error at (%d, %d)\n", x, y);
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }
    printf("Naive version: %.3f ms, %s\n", msNaive, correct ? "correct" : "failed");

    // 优化版本
    cudaEventRecord(start);
    transposeShared<<<grid, block>>>(d_out, d_in, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msShared;
    cudaEventElapsedTime(&msShared, start, stop);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    correct = true;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (h_out[x * height + y] != h_in[y * width + x]) {
                printf("Shared version error at (%d, %d)\n", x, y);
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }
    printf("Shared version: %.3f ms, %s\n", msShared, correct ? "correct" : "failed");
    printf("Speedup: %.2fx\n", msNaive / msShared);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
