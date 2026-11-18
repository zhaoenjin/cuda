// 实验 6：warp 分支发散的代价
// 硬件以 32 个线程为一组(warp)执行，同一 warp 必须执行同一条指令。
// 如果 warp 内一半线程走 if、一半走 else，硬件只能串行执行两条路径。
//
// 下面两个 kernel 的"总工作量"完全相同（一半线程干重活，一半干轻活），
// 唯一区别是分支怎么切：
//   uniform_branch : 按整个 warp 切 —— warp 0 干重活, warp 1 干轻活, 互不影响
//   divergent      : 在 warp 内部切 —— 每个 warp 都是一半重一半轻,
//                    硬件被迫把两条路径各执行一遍(空闲一半线程)
//
// 预测：两个各跑多少时间？差距多大？
//
// 编译运行：
//   nvcc 06_warp_divergence.cu -o 06_warp_divergence && ./06_warp_divergence

#include <cstdio>
#include <cuda_runtime.h>

// 分支 A：重活，iters 次链式 sinf（必须链式，防止编译器把循环折叠掉）
// 分支 B：轻活，只有 iters/4
// 两个 kernel 里 A、B 的工作量完全一样，只是线程的分配方式不同。

__global__ void uniform_branch(float* x, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = x[i];
    if (((threadIdx.x >> 5) & 1) == 0) {          // 偶数 warp 整体走 A
        for (int k = 0; k < iters; k++) v = sinf(v);
    } else {                                      // 奇数 warp 整体走 B
        for (int k = 0; k < iters / 4; k++) v = sinf(v);
    }
    x[i] = v;
}

__global__ void divergent_branch(float* x, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = x[i];
    if (threadIdx.x % 2 == 0) {                   // 每个 warp 内部对半分
        for (int k = 0; k < iters; k++) v = sinf(v);
    } else {
        for (int k = 0; k < iters / 4; k++) v = sinf(v);
    }
    x[i] = v;
}

template <typename K>
static float time_it(K kernel, float* x, int iters, int n, int reps)
{
    int grid = (n + 255) / 256;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    kernel<<<grid, 256>>>(x, iters);              // 预热
    cudaEventRecord(start);
    for (int r = 0; r < reps; r++)
        kernel<<<grid, 256>>>(x, iters);
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
    const int n = 1 << 20;      // 100 万线程
    const int iters = 2000;

    float* x;
    cudaMalloc(&x, n * sizeof(float));
    cudaMemset(x, 0, n * sizeof(float));
    // 注: sinf(0)=0, 结果收敛到 0, 但工作量不变, 计时有效

    float t_uniform = time_it(uniform_branch, x, iters, n, 10);
    float t_diverge = time_it(divergent_branch, x, iters, n, 10);

    printf("总工作量相同, 线程数相同 (%d threads, %d iters vs %d/4 iters)\n\n",
           n, iters, iters);
    printf("按 warp 整体分支 : %8.3f ms\n", t_uniform);
    printf("warp 内部对半分支: %8.3f ms   (慢 %.2f 倍)\n",
           t_diverge, t_diverge / t_uniform);

    printf("\n解释:\n");
    printf("  uniform: 每个 warp 只走一条路径, 一半 warp 忙、一半 warp 快速做完\n");
    printf("  divergent: 每个 warp 都要把两条路径串行执行, 且各有一半线程空转\n");
    printf("练习: 把 iters/4 改成 iters/2 (两条路径工作量接近), 差距会变小还是变大?\n");

    cudaFree(x);
    return 0;
}
