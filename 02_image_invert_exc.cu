#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vert(unsigned char* d_data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(x < width && y < height) {
        int i = y * width + x;
        d_data[i] = 255 - d_data[i];
    }
}

int main() {

    int width = 1280;
    int height = 720;

    size_t size = 1280 * 720;
    unsigned char* h_data = (unsigned char*)malloc(size);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_data[y * width + x] = (x + y) % 256;
        }
    }

    unsigned char* d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    vert<<<grid, block>>>(d_data, width, height);

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    return 0;
}