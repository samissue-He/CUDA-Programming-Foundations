#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define M 3
#define K 4
#define N 2

#define CHECK_CUDA(val) check_cuda((val), #val, __FILE__, __LINE__)
#define CHECK_CUBLAS(val) check_cublas((val), #val, __FILE__, __LINE__)
#undef PRINT_MATRIX
#define PRINT_MATRIX(mat, row, cols) print_matrix(mat, row, cols)

void check_cuda(cudaError_t err, const char* func, const char* file, const int line){
    if(err != cudaSuccess){
        fprintf(stderr, "CUDA error at %s:%d code = %d(%s) \" %s \" \n", file , line, static_cast<unsigned int> (err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

void check_cublas(cublasStatus_t err, const char* func, const char* file, const int line){
    if(err != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "cuBLAS error at %s:%d code = %d \" %s \" \n", file , line, static_cast<unsigned int> (err), func);
        exit(EXIT_FAILURE);
    }
}

void print_matrix(const thrust::host_vector<float> &mat, int rows, int cols){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            printf("%8.3f", mat[i*cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Today we try to use thrust library to adapt modern C++
struct floattohalf{
    __device__ half operator()(const float &f) const {return __float2half(f);}
};

struct halftohost{  // Remember transform in host
    __host__ float operator()(const half&h) const{return __half2float(h);}
};

int main(){
    // row precedence
    float a[M * K] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float b[K * N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    thrust::host_vector<float> A(a, a+M*K);
    thrust::host_vector<float> B(b, b+K*N);
    thrust::host_vector<float> C_cublas_s(M*N);
    thrust::host_vector<float> C_cublas_h(M*N);
    thrust::host_vector<half> C(M*N);

    // cublas set-up
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // cublas_s
    thrust::device_vector<float> d_a = A;
    thrust::device_vector<float> d_b = B;
    thrust::device_vector<float> d_c(M*N);

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, thrust::raw_pointer_cast(d_b.data()), N, thrust::raw_pointer_cast(d_a.data()), K, &beta, thrust::raw_pointer_cast(d_c.data()), N));
    C_cublas_s = d_c;

    // cublas_h
    thrust::device_vector<half> d_a_h(M*K);
    thrust::device_vector<half> d_b_h(K*N);
    thrust::device_vector<half> d_c_h(M*N);

    thrust::transform(d_a.begin(), d_a.end(), d_a_h.begin(), floattohalf());
    thrust::transform(d_b.begin(), d_b.end(), d_b_h.begin(), floattohalf());

    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);
    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, thrust::raw_pointer_cast(d_b_h.data()), N, thrust::raw_pointer_cast(d_a_h.data()), K, &beta_h, thrust::raw_pointer_cast(d_c_h.data()), N));
    C = d_c_h;

    thrust::transform(C.begin(), C.end(), C_cublas_h.begin(), halftohost());

    // Print results
    printf("Matrix A (%dx%d):\n", M, K);
    PRINT_MATRIX(A, M, K);
    printf("Matrix B (%dx%d):\n", K, N);
    PRINT_MATRIX(B, K, N);
    printf("cuBLAS SGEMM Result (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cublas_s, M, N);
    printf("cuBLAS HGEMM Result (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cublas_h, M, N);

    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}

/*

cd ~/cuda-course-master/06_CUDA_APIs
nvcc -arch=native practise_1.cu -o p1 -lcublas
nsys profile -o ./report_p1 --force-overwrite true ./p1
ncu --set full -o ./ncu_report_p1 -f ./p1

*/