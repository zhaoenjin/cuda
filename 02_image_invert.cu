// 图像反色：演示二维线程模型
#include <cstdio>
#include <cuda_runtime.h>

__global__ void invertImage(unsigned char *img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int i = y * width + x;
        img[i] = 255 - img[i];
    }
}

int main() {
    const int width = 1024;
    const int height = 768;
    size_t size = width * height;

    // 分配内存
    unsigned char *h_img = (unsigned char*)malloc(size);
    unsigned char *d_img;
    cudaMalloc(&d_img, size);

    // 初始化：生成渐变图
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_img[y * width + x] = (x + y) % 256;
        }
    }

    printf("Original pixel [100, 100]: %d\n", h_img[100 * width + 100]);

    // 拷贝到 device
    cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice);

    // 二维 grid 和 block
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    printf("Grid: (%d, %d), Block: (%d, %d)\n",
           grid.x, grid.y, block.x, block.y);

    // 启动 kernel
    invertImage<<<grid, block>>>(d_img, width, height);

    // 拷回
    cudaMemcpy(h_img, d_img, size, cudaMemcpyDeviceToHost);

    printf("Inverted pixel [100, 100]: %d\n", h_img[100 * width + 100]);

    // 验证：原像素 + 反色像素 应该等于 255
    bool correct = true;
    for (int i = 0; i < size; i++) {
        int original = (i / width + i % width) % 256;
        if (h_img[i] != 255 - original) {
            printf("Error at index %d\n", i);
            correct = false;
            break;
        }
    }
    if (correct) printf("Result verified: all pixels inverted correctly\n");

    cudaFree(d_img);
    free(h_img);
    return 0;
}
