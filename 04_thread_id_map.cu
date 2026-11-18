// 实验 4：亲眼"看见"线程编号
// 线程模型的核心问题：每个线程怎么知道自己该处理哪个数据？
// 答案：blockIdx / threadIdx / blockDim 三个内置变量组合算出全局编号。
//
// 运行后观察：
//   1. gid = blockIdx.x * blockDim.x + threadIdx.x 的计算过程
//   2. 二维时 (x, y) 怎么映射回一维的内存地址
//   3. 输出顺序可能乱序 —— block 的执行顺序是不保证的！
//
// 编译运行：
//   nvcc 04_thread_id_map.cu -o 04_thread_id_map && ./04_thread_id_map

#include <cstdio>
#include <cuda_runtime.h>

// 一维：3 个 block × 每个 8 线程 = 24 个线程
__global__ void ids_1d()
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("block %d . thread %d  ->  gid = %2d\n",
           blockIdx.x, threadIdx.x, gid);
}

// 二维：2x2 个 block × 每个 4x4 线程，模拟处理一张宽 8 高 4 的图
__global__ void ids_2d()
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // 列
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // 行
    int flat = y * 8 + x;                            // 内存中按行存储 -> 线性地址
    printf("block(%d,%d) thread(%d,%d)  ->  pixel(x=%d, y=%d)  flat = %2d\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, x, y, flat);
}

int main()
{
    printf("=== 一维: grid = 3 blocks, block = 8 threads ===\n\n");
    ids_1d<<<3, 8>>>();
    cudaDeviceSynchronize();

    printf("\n=== 二维: grid = 2x2 blocks, block = 4x4 threads, image width = 8 ===\n\n");
    dim3 block(4, 4);   // 16 threads
    dim3 grid(2, 2);    // 4 blocks
    ids_2d<<<grid, block>>>();
    cudaDeviceSynchronize();

    printf("\n练习:\n");
    printf("  1. 把 ids_1d<<<3, 8>>> 改成 <<<2, 16>>>, 先预测输出, 再运行验证\n");
    printf("  2. 在 kernel 里打印 gridDim.x, 看看它是什么\n");
    printf("  3. 多跑几次, 观察 gid 顺序是否每次都一样\n");
    return 0;
}
