#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256 // Number of rows in A and C
#define K 512 // Number of columns in A and rows in B
#define N 256 // Number of columns in B and C
#define BLOCK_SIZE 32

// CPU matrix multiplication
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float sum = 0.0f;
            for (int l = 0; l < k; l++)
            {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// CUDA kernel for matrix multipication 只需要写最核心的一层即可,其他循环被线程并行代替(可以视为row,col代替?)
// 一次执行只有一个网格，公式映射下的 row 和 col 必然是唯一的，不会越界覆盖
__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 修正：将 blockIdx.y 用于行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 修正：将 blockIdx.x 用于列索引

    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (int i = 0; i < k; i++)
        {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Initialize matirx with random values
void init_matrix(float *mat, int rows, int cols)
{
    for (int i = 0; i < cols * rows; i++)
    {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main()
{
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate host memory
    h_A = (float *)malloc(size_A);
    h_B = (float *)malloc(size_B);
    h_C_cpu = (float *)malloc(size_C);
    h_C_gpu = (float *)malloc(size_C);

    // Initialize the matrixs
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // Allocate device memory
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);                                              // Corrected: 2D block dimensions
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE); // 修正：gridDim.x 对应 N (列), gridDim.y 对应 M (行)

    // Warm-up runs
    printf("Running for %d\n", BLOCK_SIZE); 
    for (int i = 0; i < 3; i++)
    {
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N); // Corrected: Kernel launch configuration
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    double cpu_total_time = 0.0f;
    for (int i = 0; i < 20; i++)
    {
        double start_time = get_time();
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        double end_time = get_time();
        cpu_total_time += (end_time - start_time);
    }
    double cpu_avg_time = cpu_total_time / 20;

    // Benchmark GPU implementation
    double gpu_total_time = 0.0f;
    for (int i = 0; i < 20; i++)
    {
        double start_time = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N); // Corrected: Kernel launch configuration
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += (end_time - start_time);
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Print result
    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Free variables
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/*
nvcc -arch=native "/home/sami/cuda-course-master/05_Writing_your_First_Kernels/practise_2.cu" -o practise_2 && ./practise_2
*/
