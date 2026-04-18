#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <nvtx3/nvtx3.hpp>

#define TILE_SIZE 32

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *func, const char *file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%d code = %d(%s) \" %s \" \n ", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

// M Number of rows in A and C
// K Number of columns in A and rows in B
// N Number of columns in B and C
 
// 不能在核函数中使用NVTX, NVTX只能在CPU中调用
__global__ void matrixmultiply(float *A, float *B, float *C, int M, int N, int K)
{
    extern __shared__ float s_rem[];
    float (*s_a)[TILE_SIZE + 1] = (float (*)[TILE_SIZE + 1]) s_rem; // 数组指针,指向一个数组
    float (*s_b)[TILE_SIZE + 1] = (float (*)[TILE_SIZE + 1]) &s_rem[TILE_SIZE * (TILE_SIZE + 1)];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * blockDim.x + tx;
    int row = blockDim.y * blockIdx.y + ty;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
 
        if (row < M && (t * TILE_SIZE + tx) < K)
        {
            s_a[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        }
        else
            s_a[ty][tx] = 0.0f;

        if (col < N && (t * TILE_SIZE + ty) < K)
        {
            s_b[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        }
        else
            s_b[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++)
        {
            sum += s_a[ty][i] * s_b[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

void CUDART_CB callback(cudaStream_t stream, cudaError_t status, void *userdata)
{
    printf("Stream callback: Operation completed\n");
}

void init_matrix(float *A, int M, int K)
{
    nvtx3::scoped_range rm("Initialization");
    srand(time(NULL));
    for (int i = 0; i < M * K; i++)
    {
        A[i] = (float)rand() / RAND_MAX;
    }
}

void checkdiff(float *C, float *D, int M, int K)
{
    nvtx3::scoped_range ru("checking difference");
    int flag = 0;
    for (int j = 0; j < M * K; j++)
    {
        if (fabs(C[j] - D[j]) > 1e-5)
        {
            printf("There's Different in element no. %d", j);
            flag = 1;
            break;
        }
    }
    if (flag == 0)
        printf("There's the same between two calculations!\n");
}

int main(int argc, char **argv)
{
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A, *h_B, *h_C, *h_D;
    float *d_A, *d_B, *d_C, *d_D;

    CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_A, size_A));
    CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_B, size_B));
    CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_C, size_C));
    CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_D, size_C));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size_A));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size_B));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size_C));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_D, size_C));

    init_matrix(h_A, M, K);
    init_matrix(h_B, M, K);

    cudaStream_t stream1;
    cudaStream_t stream2;
    nvtxMarkA("Event starts.");
    cudaEvent_t event;
    cudaEvent_t start1, start2;
    cudaEvent_t stop1, stop2;

    int leastpriority, greatestpriority;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastpriority, &greatestpriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastpriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestpriority));

    CHECK_CUDA_ERROR(cudaEventCreate(&event));
    CHECK_CUDA_ERROR(cudaEventCreate(&start1));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop1));
    CHECK_CUDA_ERROR(cudaEventCreate(&start2));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop2));
    std::cout << event << std::endl;
{
    nvtx3::scoped_range ro("Copy data host to device");
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A, h_A, size_A, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B, h_B, size_B, cudaMemcpyHostToDevice, stream2));
}
    CHECK_CUDA_ERROR(cudaEventRecord(event, stream2));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream1, event, 0));

    // Do multiplication
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    size_t s_size = 2 * TILE_SIZE * (TILE_SIZE + 1) * sizeof(float);

    // 增加时间测算,需要保留事件同步,导致无法异步传输数据回host

    // CHECK_CUDA_ERROR(cudaEventRecord(start1, stream1));
    // matrixmultiply<<<gridDim, blockDim, s_size, stream1>>>(d_A, d_B, d_C, M, N, K);
    // CHECK_CUDA_ERROR(cudaEventRecord(stop1, stream1));
    // CHECK_CUDA_ERROR(cudaEventSynchronize(stop1));
    // float time1 = 0.5f;
    // CHECK_CUDA_ERROR(cudaEventElapsedTime(&time1, start1, stop1));

    // CHECK_CUDA_ERROR(cudaEventRecord(start2, stream2));
    // matrixmultiply<<<gridDim, blockDim, s_size, stream2>>>(d_A, d_B, d_D, M, N, K);
    // CHECK_CUDA_ERROR(cudaEventRecord(stop2, stream2));
    // CHECK_CUDA_ERROR(cudaEventSynchronize(stop2));
    // float time2 = 0.5f;
    // CHECK_CUDA_ERROR(cudaEventElapsedTime(&time2, start2, stop2));
 
    CHECK_CUDA_ERROR(cudaStreamAddCallback(stream2, callback, NULL, 0));

    // printf("Stream1 time is %f\n", time1);
    // printf("Stream2 time is %f\n", time2);

{    
    nvtx3::scoped_range rd("Kernal starts.");
    matrixmultiply<<<gridDim, blockDim, s_size, stream1>>>(d_A, d_B, d_C, M, N, K);
    matrixmultiply<<<gridDim, blockDim, s_size, stream2>>>(d_A, d_B, d_D, M, N, K);
}
    
{    
    nvtx3::scoped_range re("Transfer data back to host");
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C, d_C, size_C, cudaMemcpyDeviceToHost, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_D, d_D, size_C, cudaMemcpyDeviceToHost, stream2));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));
}

    checkdiff(h_C, h_D, M, K);

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_D);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(event);

    return 0;
}

/*

cd ~/cuda-course-master/05_Writing_your_First_Kernels
nvcc -arch=native practise_4.cu -o p4
nsys profile -o ./report_p4 --force-overwrite true ./p4
ncu --set full -o ./ncu_report_p4 -f ./p4

*/
