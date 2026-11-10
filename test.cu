#include <stdio.h>

__global__ void hello_from_gpu() {
    printf("Hello world from gpu\n");
}

int main() {
    //一个核函数对应一个网格，一个网格对应多个线程块，一个线程块对应多个线程
    hello_from_gpu<<<2, 2>>>();
    cudaDeviceSynchronize();
}