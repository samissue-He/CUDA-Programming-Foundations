#include <iostream>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <random>
#include <thrust/system/cuda/memory.h>
#include <nvtx3/nvtx3.hpp>
#include <future> // cpu-async

#define CHECK_CUBLAS(val) check((val), #val, __FILE__, __LINE__)
#define M 8192
#define N 8192
#define K 8192

void check(cublasStatus_t err, const char* func, const char* file, const int line){
    if(err != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "cuBLAS error at %s:%d code = %d \" %s \" \n", file , line, static_cast<unsigned int> (err), func);
        exit(EXIT_FAILURE);
    }
}

struct floattohalf{
    __device__ half operator()(const float &f) const {return __float2half(f);}
};

struct halftohost{  // Remember transform in host
    __host__ float operator()(const half&h) const{return __half2float(h);}
};

struct dellay{
    void operator()(cublasLtMatrixLayout_t a) const {CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(a));}
};
using uniquelayout = std::unique_ptr<std::remove_pointer<cublasLtMatrixLayout_t>::type, dellay>;

using pinned_vector = thrust::host_vector<float, thrust::mr::stateless_resource_allocator<float,thrust::system::cuda::universal_host_pinned_memory_resource>>;

struct delstr{
    void operator()(cudaStream_t t) const {cudaStreamDestroy(t);}
};

void initializeMatrix(pinned_vector& matrix, int rows, int cols) {
    nvtx3::scoped_range r("Initialization");
    std::random_device rd; // It will not generate normally distributed data like randn.
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(dis(gen));
    }
}
int main(){
    // row precedence
    pinned_vector A(M*K);
    pinned_vector B(K*N);
    thrust::host_vector<float> C_cublas_h(M*N);
    thrust::host_vector<half> C(M*N);

    // Initialization
    auto featureA{std::async(std::launch::async, [&](){ initializeMatrix(A, M, K);})};
    auto featureB(std::async(std::launch::async, [&](){ initializeMatrix(B, K, N);}));

    featureA.get();
    featureB.get();

    // Streams
    cudaStream_t stream1 = nullptr;
    cudaStream_t stream2 = nullptr;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    thrust::device_vector<float> d_a(M*K);
    thrust::device_vector<float> d_b(K*N);
    thrust::device_vector<float> d_c(M*N);

    // Copy data from pinned memory to device
    thrust::copy(thrust::cuda::par.on(stream1), A.begin(), A.end(), d_a.begin());
    thrust::copy(thrust::cuda::par.on(stream2), B.begin(), B.end(), d_b.begin());

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Construct handle (pinned memory --- handle can be binded with stream)
    cublasLtHandle_t lthandle1 = nullptr;
    CHECK_CUBLAS(cublasLtCreate(&lthandle1));

    // For cuBLAS, not for cuBLASLt
    // cublasSetStream_v2(lthandle1, stream1);
    // cublasSetStream_v2(lthandle2, stream2);

    // Construct layout
    cublasLtMatrixLayout_t adesc = nullptr;
    cublasLtMatrixLayout_t bdesc = nullptr;
    cublasLtMatrixLayout_t r_cdesc = nullptr;

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&adesc, CUDA_R_16F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_16F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&r_cdesc, CUDA_R_16F, N, M, N));
    uniquelayout cdesc(r_cdesc); // use smart pointer to manage

    // Cunstruct desc
    cublasLtMatmulDesc_t operation = nullptr;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F)); // Often scale-type is the same as the compute-type

    // cublas_h
    thrust::device_vector<half> d_a_h(M*K);
    thrust::device_vector<half> d_b_h(K*N);
    thrust::device_vector<half> d_c_h(M*N);

    thrust::transform(d_a.begin(), d_a.end(), d_a_h.begin(), floattohalf());
    thrust::transform(d_b.begin(), d_b.end(), d_b_h.begin(), floattohalf());

    float alpha_h = 1.0f;
    float beta_h = 0.0f;
    // The third matrix parameter is usually the input bias matrix. Since we only perform matrix multiplication here, we can simply overwrite the C matrix.
    CHECK_CUBLAS(cublasLtMatmul(lthandle1, operation, &alpha_h, thrust::raw_pointer_cast(d_b_h.data()), bdesc, thrust::raw_pointer_cast(d_a_h.data()), adesc, &beta_h, thrust::raw_pointer_cast(d_c_h.data()), cdesc.get(), thrust::raw_pointer_cast(d_c_h.data()), cdesc.get(), nullptr, nullptr, 0, stream1));
    C = d_c_h;

    thrust::transform(C.begin(), C.end(), C_cublas_h.begin(), halftohost());

    CHECK_CUBLAS(cublasLtDestroy(lthandle1));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(adesc));
    dellay()(bdesc); 
    delstr()(stream1);
    delstr()(stream2);
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operation));

    return 0;
}

/*

cd ~/cuda-course-master/06_CUDA_APIs
nvcc -arch=native practise_2.cu -o p2 -lcublas -lcublasLt
nsys profile -o ./report_p2_pinned --force-overwrite true ./p2
ncu --set full -o ./ncu_report_p2_pinned -f ./p2

*/