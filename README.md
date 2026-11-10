# CUDA 学习示例

CUDA 13.3 安装在 WSL Ubuntu 26.04，显卡为 GeForce GTX 1650 (sm_75)。

## 文档

- [CUDA线程模型.md](CUDA线程模型.md) - 线程层次、内存模型、多维编号

## 代码示例

### 01_vector_add.cu - 向量加法
演示一维线程模型的基本用法。

```bash
nvcc 01_vector_add.cu -o 01_vector_add
./01_vector_add
```

关键点：
- 全局线程编号：`blockIdx.x * blockDim.x + threadIdx.x`
- grid 大小向上取整：`(n + blockSize - 1) / blockSize`
- 边界检查：`if (i < n)`

### 02_image_invert.cu - 图像反色
演示二维线程模型，每个线程对应一个像素。

```bash
nvcc 02_image_invert.cu -o 02_image_invert
./02_image_invert
```

关键点：
- 二维编号：`(x, y)` 分别由 `.x` 和 `.y` 计算
- `dim3` 定义 grid 和 block 的多维结构
- 线性地址换算：`y * width + x`

### 03_transpose.cu - 矩阵转置与 shared memory
对比朴素版本和使用 shared memory 优化的版本。

```bash
nvcc 03_transpose.cu -o 03_transpose
./03_transpose
```

关键点：
- **朴素版本**：直接转置导致写入时内存访问不连续（跨行），带宽利用率低。
- **优化版本**：用 shared memory 做中转，读写全局内存时都保持连续访问（coalescing）。
- `__shared__` 声明的数组被同一 block 内所有线程共享。
- `__syncthreads()` 同步 block 内所有线程，确保数据都读进 shared memory 后再写出。
- `tile[TILE_DIM][TILE_DIM + 1]`：多加一列避免 bank conflict（shared memory 被分为 32 个 bank，同时访问同一 bank 的不同地址会串行化）。

在 GTX 1650 上测试结果：优化版本比朴素版本快约 30 倍。

## 编译技巧

查看生成的 PTX（中间代码）：
```bash
nvcc -ptx file.cu
```

查看生成的汇编（SASS）：
```bash
nvcc -cubin file.cu
cuobjdump -sass file.cubin
```

指定计算能力（GTX 1650 是 sm_75）：
```bash
nvcc -arch=sm_75 file.cu
```

## 性能分析

用 `nvprof`（已废弃，但 WSL 里可能还能用）：
```bash
nvprof ./program
```

或用 Nsight Compute（推荐）：
```bash
ncu ./program
```

## 常用 API 速查

| 操作 | API |
|---|---|
| 分配显存 | `cudaMalloc(&ptr, size)` |
| 释放显存 | `cudaFree(ptr)` |
| 拷贝数据 | `cudaMemcpy(dst, src, size, kind)` |
| 设置显存 | `cudaMemset(ptr, value, size)` |
| 同步 | `cudaDeviceSynchronize()` |
| 检查错误 | `cudaGetLastError()` |

`cudaMemcpy` 的 `kind` 参数：
- `cudaMemcpyHostToDevice`：CPU → GPU
- `cudaMemcpyDeviceToHost`：GPU → CPU
- `cudaMemcpyDeviceToDevice`：GPU → GPU
