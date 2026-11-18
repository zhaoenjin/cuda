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

### 04_thread_id_map.cu - 看见线程编号
打印每个线程的 `blockIdx` / `threadIdx`，亲眼看到编号如何映射到全局编号和像素坐标。

```bash
nvcc 04_thread_id_map.cu -o 04_thread_id_map
./04_thread_id_map
```

关键点：
- 一维：`gid = blockIdx.x * blockDim.x + threadIdx.x`
- 二维：`(x, y)` 各自计算，最后换算回线性地址 `y * width + x`
- **实测发现**：block 2 的输出比 block 1 先出现 —— block 的执行顺序不保证，
  kernel 里绝不能假设"前面的 block 先跑完"。

### 05_block_size_timing.cu - block 大小对性能的影响
同一个向量加法，只改 block 大小（32~1024），用 `cudaEvent` 计时。

```bash
nvcc 05_block_size_timing.cu -o 05_block_size_timing
./05_block_size_timing
```

GTX 1650 实测（1600 万元素）：

| block | 平均耗时 | 有效带宽 |
|---|---|---|
| 32 | 2.200 ms | 91.5 GB/s |
| 64~1024 | ~1.13 ms | ~178 GB/s |

关键点：
- **block=32 带宽直接掉一半**：sm_75 每个 SM 最多同时驻留 16 个 block，
  32×16 = 512 线程/SM，占用率不足，SM 吃不饱。
- 64 以上全部打满 ~178 GB/s（GDDR6 版理论 192 GB/s），说明这是**内存受限**
  的 kernel —— 瓶颈在搬数据，不在计算，所以中间大小的差异很小。
- `cudaEvent` 是 GPU 侧计时的标准做法，比 CPU 的 `clock()` 准确。

### 06_warp_divergence.cu - warp 分支发散的代价
两个 kernel 总工作量完全相同（一半线程干重活、一半干轻活），只是分支切法不同：
按整个 warp 切 vs 在 warp 内部对半切。

```bash
nvcc 06_warp_divergence.cu -o 06_warp_divergence
./06_warp_divergence
```

GTX 1650 实测（100 万线程）：

| 分支方式 | 耗时 |
|---|---|
| 按 warp 整体分支 | 19.5 ms |
| warp 内部对半分支 | 30.0 ms（慢 1.54 倍） |

关键点：
- warp 内分支发散时，硬件把两条路径**串行**执行，每次都有一半线程空转。
- 写 kernel 时尽量让同一个 warp 走同一条路径（比如按 `threadIdx.x / 32` 分支，
  而不是 `threadIdx.x % 2`）。

### 07_error_check.cu - 错误检查的四种失败模式
演示 CUDA 错误的三个特性：静默返回、异步延迟、粘住不散。

```bash
nvcc 07_error_check.cu -o 07_error_check
./07_error_check good      # 正常路径
./07_error_check config    # 启动配置错误：立即捕获，可恢复
./07_error_check oob       # 野指针写：错误延迟到 sync 才出现，且粘住后续调用
./07_error_check subtle    # 轻微越界：硬件完全不报错，数据被悄悄写坏
```

关键点：
- API 调用返回 `cudaError_t`，不检查就等于没有错误处理
- kernel 启动是异步的：启动后要用 `cudaGetLastError()` 查配置错误，
  用 `cudaDeviceSynchronize()` 查执行期错误——报错行号往往离出错点很远
- illegal memory access 会污染 context，之后所有调用报同一个错误，只能退出
- 轻微越界（超出分配几字节）硬件不报错，只能靠 `compute-sanitizer` 抓
- 定位异步错误的快捷方式：`CUDA_LAUNCH_BLOCKING=1 ./app` 让启动变成同步，
  错误直接在启动行爆出来

## 建议学习顺序

1. 读 [CUDA线程模型.md](CUDA线程模型.md) 建立整体概念
2. 01 → 02：一维、二维编号怎么写
3. 04：跑一遍，亲眼看到编号映射和 block 乱序执行
4. 03：shared memory 的威力
5. 05 → 06：开始关注性能 —— 占用率、内存带宽、warp 发散
6. 07：养成错误检查的习惯，学会用 compute-sanitizer
7. 每个实验先做文件头部注释里的"预测"，再运行验证

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
