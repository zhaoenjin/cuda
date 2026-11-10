#include <stdio.h>

__global__ void hello_from_gpu() {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int id = threadId + blockId * blockDim.x;

    printf("current thread: block %d, thread %d, id %d\n", blockId, threadId, id);
}

int main() {
    //一个核函数对应一个网格，一个网格对应多个线程块，一个线程块对应多个线程
    hello_from_gpu<<<3, 2>>>();
    cudaDeviceSynchronize();
    return 0;
}

