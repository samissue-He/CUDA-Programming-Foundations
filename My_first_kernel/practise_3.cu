#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16

// 多维动态内存的“行数”通常是隐式的
// // 假设 TILE_SIZE = 32
// // 每行占用 TILE_SIZE + 1 个 float 的空间
// extern __shared__ float s_mem[];

// // s_a 指向起始位置，每行宽度为 TILE_SIZE + 1
// float (*s_a)[TILE_SIZE + 1] = (float (*)[TILE_SIZE + 1])s_mem;

// // s_b 的偏移量：跳过 s_a 占用的总元素个数
// // s_a 有 TILE_SIZE 行，每行宽 TILE_SIZE + 1
// float (*s_b)[TILE_SIZE + 1] = (float (*)[TILE_SIZE + 1])&s_mem[TILE_SIZE * (TILE_SIZE + 1)];

// // 在 Host 端分配的共享内存大小也需要相应修改：
// size_t sMemSize = 2 * TILE_SIZE * (TILE_SIZE + 1) * sizeof(float);
// matrixmultiply<<<gridDim, blockDim, sMemSize>>>(...);

__global__ void matrixmultiply(float* A, float* B, float* C, int M, int K, int N){
    __shared__ float s_a[TILE_SIZE][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE];
    float sum = 0.0f;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    for(int t=0; t < (K+TILE_SIZE-1) / TILE_SIZE; t++){
        if(row < M && (t*TILE_SIZE+tx < K)){
            s_a[ty][tx] = A[row*K + t*TILE_SIZE + tx];
        }else
            s_a[ty][tx] = 0.0f;

        if(col < N && (t*TILE_SIZE + ty) < K){
            s_b[ty][tx] = B[(t*TILE_SIZE+ty)*N + col];
        }else
            s_b[ty][tx] = 0.0f;

        __syncthreads();

        for(int j=0; j<TILE_SIZE; j++){
            sum += s_a[ty][j] * s_b[j][tx];
        }

        __syncthreads();

    }

    if(col < N && row < M)
        C[row * N + col] = sum;

}


int main(){
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;

    // Allocate cuda memory
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Kernel block dimension
    dim3 blockDim(TILE_SIZE,TILE_SIZE);
    dim3 gridDim((N+TILE_SIZE-1)/TILE_SIZE,(M+TILE_SIZE-1)/TILE_SIZE);

    matrixmultiply<<<gridDim,blockDim>>>(d_A, d_B, d_C, M, K, N);

    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Check for any cuda error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error is %d", error); // error 竟然是 int 类型
        return -1;
    }

    return 0;
}

/*

cd ~/cuda-course-master/05_Writing_your_First_Kernels
nvcc -arch=native practise_3.cu -o p3 
nsys profile -o ./report_p3 --force-overwrite true ./p3
ncu --set full -o ./ncu_report_p3 -f ./p3

*/