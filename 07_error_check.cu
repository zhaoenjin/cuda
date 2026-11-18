// 实验 7：CUDA 的错误检查 —— 为什么必须有、错误藏在哪里
//
// CUDA 错误的三个特性，决定了检查代码必须这样写：
//   1. API 调用返回 cudaError_t，但不检查就被静默忽略（不抛异常、不崩溃）
//   2. kernel 启动是异步的：<<<>>> 立即返回，kernel 内部的错误（如越界写）
//      要等到下一次同步（cudaDeviceSynchronize / cudaMemcpy）才浮出水面
//      —— 所以报错行号往往离真正的出错点很远！
//   3. 严重错误（illegal memory access）会"粘住"：context 被污染，
//      之后所有 CUDA 调用都返回同一个错误，进程只能退出
//
// 四种运行模式（分别演示）：
//   ./07_error_check good     正常路径，检查全部通过
//   ./07_error_check config   启动配置错误（每 block 超 1024 线程）
//                             —— cudaGetLastError 立即捕获，可清除，context 存活
//   ./07_error_check oob      野指针写 —— 启动后检查"没问题"，错误延迟到 sync
//                             才出现，且粘住后续所有调用
//   ./07_error_check subtle   轻微越界 —— 硬件根本不报错，数据被悄悄写坏（最可怕）
//
//   用 compute-sanitizer 抓 subtle 这种硬件抓不到的越界（能定位到具体线程）：
//   /usr/local/cuda/bin/compute-sanitizer ./07_error_check subtle
//
// 编译：nvcc 07_error_check.cu -o 07_error_check

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// ---------------- 生产环境用的检查宏（复制这个） ----------------
// 用法：CUDA_CHECK(cudaMalloc(&d, size));
//       kernel<<<g, b>>>(...);
//       CUDA_CHECK(cudaGetLastError());        // 启动配置错误在这里现形
//       CUDA_CHECK(cudaDeviceSynchronize());   // 执行期错误在这里现形
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err_));         \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// ---------------- 教学版：只打印不退出，便于观察错误的行为 ----------------
static cudaError_t g_last;
#define SHOW(call, desc)                                        \
    do {                                                        \
        g_last = (call);                                        \
        printf("  L%-4d %-30s -> %s\n", __LINE__, desc,         \
               cudaGetErrorString(g_last));                     \
    } while (0)

// 正确的 kernel：有边界检查
__global__ void good_kernel(int* d, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = i * 2;
}

// 教材级错误：分配恰好 1000 个 int，却启动 1016 个线程，还忘了边界检查
// 越界只超出 16 个 int —— 落在显存池已映射的页里，硬件不会报错！
__global__ void overflow_kernel(int* d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d[i] = i;                       // 忘了 if (i < n)
}

// 写明显非法的地址 —— 硬件必然捕获，用来演示"错误延迟出现"
__global__ void wild_write_kernel(int* d)
{
    int* bad = reinterpret_cast<int*>(0x12345678);
    *bad = threadIdx.x;             // 这行必然触发 illegal memory access
    d[threadIdx.x] = threadIdx.x;   // 永远执行不到
}

int main(int argc, char** argv)
{
    const char* mode = argc > 1 ? argv[1] : "good";
    int* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, 1000 * sizeof(int)));

    if (!strcmp(mode, "good")) {
        printf("[good] 正常路径\n");
        good_kernel<<<4, 256>>>(d, 1000);
        SHOW(cudaGetLastError(), "启动后立即检查");
        SHOW(cudaDeviceSynchronize(), "同步等待执行结果");

    } else if (!strcmp(mode, "config")) {
        printf("[config] 启动配置错误：每 block 最多 1024 线程，这里给了 2048\n");
        good_kernel<<<1, 2048>>>(d, 1000);
        SHOW(cudaGetLastError(), "启动后立即检查");   // 立即捕获！
        // 这种错误不破坏 context，被 cudaGetLastError 读走后就清除了
        good_kernel<<<4, 256>>>(d, 1000);
        SHOW(cudaGetLastError(), "重新正确启动");
        SHOW(cudaDeviceSynchronize(), "同步");
        printf("  -> context 没坏，程序可以继续跑\n");

    } else if (!strcmp(mode, "oob")) {
        printf("[oob] kernel 里写野指针：观察错误什么时候浮出来\n");
        wild_write_kernel<<<1, 64>>>(d);
        SHOW(cudaGetLastError(), "启动后立即检查");   // 竟然是 success！
        SHOW(cudaDeviceSynchronize(), "同步等待执行结果");   // 这里才爆
        printf("  -> 报错的是 sync 那一行，真正的错误在 kernel 里，离得很远\n");
        printf("  -> 再跑一个完全正确的 kernel：\n");
        good_kernel<<<4, 256>>>(d, 1000);
        SHOW(cudaGetLastError(), "正确 kernel 启动后检查");
        SHOW(cudaDeviceSynchronize(), "正确 kernel 同步");
        printf("  -> 错误粘住了：context 已被污染，之后的调用全报错\n");

    } else if (!strcmp(mode, "subtle")) {
        printf("[subtle] 轻微越界（多写 16 个 int），看硬件能不能发现\n");
        overflow_kernel<<<1, 1016>>>(d);
        SHOW(cudaGetLastError(), "启动后立即检查");
        SHOW(cudaDeviceSynchronize(), "同步等待执行结果");   // 还是 success！
        printf("  -> 全程没有任何报错，但分配之外的内存已被写坏，受害者未知\n");
        printf("  -> 这是最可怕的失败模式，也是 compute-sanitizer 存在的意义\n");
    }

    cudaFree(d);
    return 0;
}
