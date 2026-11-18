// 实验 5：block 大小对性能的影响
// 同一个向量加法，只改 block 大小（32 ~ 1024），用 cudaEvent 计时对比。
//
// 运行前先预测：哪个 block 大小最快？为什么？
//
// 观察：
//   1. 中间大小（128~512）通常最快，两端变慢
//      - 32 太小：GPU 一个 SM 最多同时驻留 16 个 block(sm_75)，
//        32*16 = 512 线程/SM，占用率低，SM 吃不饱
//      - 1024 太大：调度粒度粗，最后一个 block 拖尾
//   2. 最后一列"有效带宽"：64 以上都稳定在 ~178 GB/s（这张 GTX 1650 是
//      GDDR6 版，理论极限 192 GB/s），说明 kernel 是"内存受限"的 ——
//      计算太简单，瓶颈在搬数据。只有 32 明显掉速，正是占用率不足的信号。
//      这也解释了为什么 block 大小的影响看起来不算剧烈。
//
// 编译运行：
//   nvcc 05_block_size_timing.cu -o 05_block_size_timing && ./05_block_size_timing

#include <cstdio>
#include <cuda_runtime.h>

__global__ void init(float* a, float* b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { a[i] = 1.0f; b[i] = 2.0f; }
}

__global__ void add(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

static float time_kernel(int blockSize, const float* a, const float* b,
                         float* c, int n, int reps)
{
    int gridSize = (n + blockSize - 1) / blockSize;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    add<<<gridSize, blockSize>>>(a, b, c, n);   // 预热一次，排除首次启动开销
    cudaEventRecord(start);
    for (int k = 0; k < reps; k++)
        add<<<gridSize, blockSize>>>(a, b, c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / reps;
}

int main()
{
    const int n = 1 << 24;   // 1600 万个元素
    float *a, *b, *c;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&b, n * sizeof(float));
    cudaMalloc(&c, n * sizeof(float));
    init<<<(n + 255) / 256, 256>>>(a, b, n);
    cudaDeviceSynchronize();

    printf("向量加法: %d 个元素, 每次 kernel 读写 %.0f MB\n\n",
           n, 3.0 * n * sizeof(float) / 1e6);
    printf("%-8s %-10s %-12s %-12s\n", "block", "grid", "平均耗时", "有效带宽");

    int sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int s : sizes) {
        float ms = time_kernel(s, a, b, c, n, 20);
        int grid = (n + s - 1) / s;
        float gbs = 3.0 * n * sizeof(float) / (ms * 1e6);   // GB/s
        printf("%-8d %-10d %-12.3f %-10.1f GB/s\n", s, grid, ms, gbs);
    }

    printf("\n练习:\n");
    printf("  1. 把 n 改成 1<<20 (100万), 差异比例会变大还是变小?\n");
    printf("  2. 把 add 里的加法换成 sinf(a[i]) + cosf(b[i]),\n");
    printf("     kernel 变成计算受限, 再看 block 大小的影响是否更明显\n");
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
