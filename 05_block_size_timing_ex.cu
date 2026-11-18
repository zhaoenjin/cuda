#include <stdio.h>
#include <cuda_runtime.h>

#define ERROR_CHECK(call)                                                                 \
    do {                                                                                  \
        cudaError_t err = call;                                                           \
        if (err != cudaSuccess) {                                                         \
            printf("has error: name: %s, string: %s, in: %s, %d\n",                       \
                    cudaGetErrorName(err), cudaGetErrorString(err), __FILE__, __LINE__);  \
            exit(-1);                                                                     \
        }                                                                                 \
    } while (0)
    

int main() {
    int n = 1 << 20;
    int *a, *b, *c;
    ERROR_CHECK(cudaMalloc(&a, n * sizeof(int)));
    ERROR_CHECK(cudaMalloc(&b, n * sizeof(int)));
    ERROR_CHECK(cudaMalloc(&c, n * sizeof(int)));

    return 0;
}