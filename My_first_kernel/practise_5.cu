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
        fprintf(stderr, "CUDA error at %s:%d code = %d(%s) \" %s \" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

// M Number of rows in A and C
// K Number of columns in A and rows in B
// N Number of columns in B and C

__global__ void matrixmultiply(float *A, float *B, float *C, int M, int N, int K)
{
    extern __shared__ float s_rem[];
    float (*s_a)[TILE_SIZE + 1] = (float (*)[TILE_SIZE + 1]) s_rem;
    float (*s_b)[TILE_SIZE + 1] = (float (*)[TILE_SIZE + 1]) & s_rem[TILE_SIZE * (TILE_SIZE + 1)];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockDim.x * blockIdx.x + tx;
    int row = blockDim.y * blockIdx.y + ty;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        if (row < M && (t * TILE_SIZE + tx) < K)
        {
            s_a[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        }
        else
        {
            s_a[ty][tx] = 0.0f;
        }

        if (col < N && (t * TILE_SIZE + ty) < K)
        {
            s_b[ty][tx] = B[col + N * (t * TILE_SIZE + ty)];
        }
        else
        {
            s_b[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++)
        {
            sum += s_a[ty][j] * s_b[j][tx];
        }

        __syncthreads();
    }

    if (col < N && row < M)
    {
        C[N * row + col] = sum;
    }
}

void CUDART_CB callback(cudaStream_t stream, cudaError_t status, void *userdata)
{
    printf("Stream callback: Operation completed\n");
}

void init_matrix(float *A, int M, int K)
{
    nvtx3::scoped_range r1("Initialzation");
    srand(time(NULL));
    for (int i = 0; i < M * K; i++)
    {
        A[i] = (float)rand() / RAND_MAX;
    }
}

void checkdiff(float *C, float *D, int M, int K)
{
    nvtx3::scoped_range r2("Check difference");
    int flag = 0;
    for (int m = 0; m < M * K; m++)
    {
        if (fabs(C[m] - D[m]) > 1e-5)
        {
            flag = 1;
            std::cout << "There's difference" << std::endl;
            break;
        }
    }
    if (flag == 0)
    {
        std::cout << "There's no difference" << std::endl;
    }
}

int main()
{
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    float *h_A, *h_B, *h_C, *h_D;
    float *d_A, *d_B, *d_C, *d_D;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_A, size_A));
    CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_B, size_B));
    CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_C, size_C));
    CHECK_CUDA_ERROR(cudaMallocHost((void **)&h_D, size_C));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size_A));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size_B));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size_C));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_D, size_C));

    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaEvent_t event;

    int leastpriority, greatestpriority;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastpriority, &greatestpriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastpriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestpriority));

    CHECK_CUDA_ERROR(cudaEventCreate(&event));

    nvtxRangePushA("Copy data from host to device");
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A, h_A, size_A, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B, h_B, size_B, cudaMemcpyHostToDevice, stream2));
    nvtxRangePop();

    // Must apply in pair
    CHECK_CUDA_ERROR(cudaEventRecord(event, stream1));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream2, event, 0));

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    size_t s_size = 2 * TILE_SIZE * (TILE_SIZE + 1) * sizeof(float);

    {
        nvtx3::scoped_range r3("Calculation");
        matrixmultiply<<<grid_size, block_size, s_size, stream1>>>(d_A, d_B, d_C, M, N, K);
        matrixmultiply<<<grid_size, block_size, s_size, stream2>>>(d_A, d_B, d_D, M, N, K);
    }

    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C, d_C, size_C, cudaMemcpyDeviceToHost, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_D, d_D, size_C, cudaMemcpyDeviceToHost, stream2));
    CHECK_CUDA_ERROR(cudaStreamAddCallback(stream1, callback, NULL, 0));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    {
        nvtx3::scoped_range r4("Check difference");
        checkdiff(h_C, h_D, M, K);
    }

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
nvcc -arch=native practise_5.cu -o p5
nsys profile -o ./report_p5 --force-overwrite true ./p5
ncu --set full -o ./ncu_report_p5 -f ./p5

*/