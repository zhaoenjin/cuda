# CUDA 线程模型

核心思路：kernel 描述"一个线程要做的事"，然后启动成千上万个线程同时执行，每个线程靠自己的编号决定处理哪份数据。

## 1. 三层结构：Grid → Block → Thread

```cpp
add<<<numBlocks, threadsPerBlock>>>(...);
```

- **Thread**：最小执行单位，每个线程执行一遍 kernel。
- **Block**：一组线程，最多 1024 个。同一个 block 内的线程可以协作。
- **Grid**：一次 kernel 启动的所有 block。

例：`<<<4, 256>>>` 是 4 个 block，每个 256 个线程，共 1024 个线程。

## 2. 线程如何确定自己的工作

| 内置变量 | 含义 |
|---|---|
| `threadIdx.x` | 线程在 block 内的编号 |
| `blockIdx.x` | block 在 grid 内的编号 |
| `blockDim.x` | 每个 block 的线程数 |
| `gridDim.x` | grid 内的 block 数 |

全局唯一编号：

```cpp
__global__ void add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)              // 线程总数可能多于数据量，多出的线程不做事
        c[i] = a[i] + b[i];
}
```

CPU 写法是一个循环，这里的循环被拆给 n 个线程，每个线程只算一个元素。

启动时线程数向上取整：

```cpp
int block = 256;
int grid = (n + block - 1) / block;
add<<<grid, block>>>(a, b, c, n);
```

## 3. 为什么分 Block：对应硬件

GPU 由多个 **SM（流式多处理器）** 组成。

- 一个 block 只会在一个 SM 上执行，不会拆开。所以 block 内的线程可以共享该 SM 的 **shared memory**，并用 `__syncthreads()` 同步。
- 不同 block 相互独立，执行顺序不确定，不能直接同步。因此 GPU 可以把 block 分配给任意空闲的 SM，同一份程序在 SM 更多的显卡上自动更快。

## 4. Warp：实际执行单位

硬件以 **32 个线程为一组（warp）** 执行，同一 warp 内的线程同一时刻执行同一条指令（SIMT）。

- **block 大小取 32 的倍数**（如 128、256），否则最后一个 warp 有空闲线程。
- **避免 warp 内分支发散**：同一 warp 内一部分线程走 `if`、一部分走 `else`，硬件要把两个分支依次执行。

```cpp
if (threadIdx.x % 2 == 0) {...}   // 差：每个 warp 内部都分叉
if (threadIdx.x / 32 == 0) {...}  // 好：按 warp 整体分支，不发散
```

## 5. 内存层次与线程层次对应

| 线程层次 | 可访问的内存 | 速度 |
|---|---|---|
| 单个线程 | 寄存器、局部变量 | 最快 |
| 同一 Block | shared memory | 很快 |
| 整个 Grid | global memory（`cudaMalloc` 分配的显存） | 慢 |

优化的主要方向是让数据尽量留在上两层，减少对显存的访问。

## 6. 二维、三维线程模型

多维只是**编号方式**不同，线程总数是各维度相乘。二维 block `(16, 16)` 与一维 `256` 的线程数相同。数据本身是多维时（图像、体数据），用多维编号更直观。

用 `dim3` 指定：

```cpp
dim3 block(16, 16);
dim3 grid((width + 15) / 16, (height + 15) / 16);
kernel<<<grid, block>>>(...);
```

可以理解为铺瓷砖：图像是墙，block 是一块 16×16 的瓷砖，grid 是铺满墙需要的瓷砖数，每个线程负责瓷砖上的一个点。

### 二维：像素坐标

```cpp
__global__ void invert(unsigned char *img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 列
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 行
    if (x < width && y < height) {
        int i = y * width + x;   // 内存是一维的，换算回线性地址
        img[i] = 255 - img[i];
    }
}
```

### 三维：体数据

```cpp
dim3 block(8, 8, 8);   // 512 个线程
dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
int i = (z * ny + y) * nx + x;
```

### x 是变化最快的维度

硬件先把 block 内线程按 x → y → z 的顺序排成一维，再每 32 个切成一个 warp。所以 `x` 应对应内存中连续的维度，这样同一 warp 内相邻线程访问相邻地址，可合并为一次内存事务（coalescing）。

```cpp
img[y * width + x]   // 好：相邻线程访问相邻地址
img[x * height + y]  // 差：地址相隔一整行，访问分散，浪费带宽
```

### 维度上限

| | x | y | z |
|---|---|---|---|
| Block 每维最大 | 1024 | 1024 | 64 |
| Block 线程总数 | x·y·z ≤ 1024 | | |
| Grid 每维最大 | 2³¹−1 | 65535 | 65535 |

常用配置：一维 256，二维 `(16,16)` 或 `(32,8)`，三维 `(8,8,8)`。

## 7. 与图形 API 对照

| CUDA | Vulkan / GLSL | D3D / HLSL |
|---|---|---|
| `blockDim` | `local_size_x/y/z` | `[numthreads(x,y,z)]` |
| `gridDim` | `vkCmdDispatch(x,y,z)` | `Dispatch(x,y,z)` |
| `threadIdx` | `gl_LocalInvocationID` | `SV_GroupThreadID` |
| 全局坐标 | `gl_GlobalInvocationID` | `SV_DispatchThreadID` |

## 8. 编译与运行

```bash
nvcc hello.cu -o hello
./hello
```
