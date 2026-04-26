#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <random>
#include <future>
#include <thrust/system/cuda/memory.h>
#include <nvtx3/nvtx3.hpp>
#include <cublasLt.h>

#define CHECK_CUBLAS(val) check((val), #val, __FILE__, __LINE__)
#define M 8192
#define N 8192
#define K 8192

void check(cublasStatus_t err, const char* func, const char* file, const int line){
    if(err != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "cuBLASLt error at %s:%d code = %d \" %s \" \n", file, line, static_cast<unsigned int>(err), func);
        exit(EXIT_FAILURE);
    }
}

struct floattohalf{
    __device__ half operator()(const float& f) const { return __float2half(f);};
};

struct halftofloat{
    __device__ float operator()(const half& f) const { return __half2float(f);};
};

struct dellay{
    void operator()(cublasLtMatrixLayout_t& f) const { CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(f));};
};

struct delstr{
    void operator()(cudaStream_t& f) const { cudaStreamDestroy(f);};
};

using pinned_vector = thrust::host_vector<float, thrust::mr::stateless_resource_allocator<float, thrust::system::cuda::universal_host_pinned_memory_resource>>;

void initializematrix(pinned_vector& vec, int rows, int cols){
    nvtx3::scoped_range f("Initialization");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    for(int i=0; i< rows * cols; i++){
        vec[i] = static_cast<float>(dis(gen)); 
    }
}

int main(){
    // Set up pinned vector
    pinned_vector A(M*K);
    pinned_vector B(K*N);
    thrust::host_vector<float> C(M*N);

    // Initialization
    auto featureA{std::async(std::launch::async, [&](){ initializematrix(A, M, K);})};
    auto featureB{std::async(std::launch::async, [&](){ initializematrix(B, K, N);})};

    featureA.get();
    featureB.get();

    // Steams
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // device_vector
    thrust::device_vector<float> d_a(M*K);
    thrust::device_vector<float> d_b(K*N);
    thrust::device_vector<float> d_c(M*N);
    
    // Copy data from host to device
    {
        nvtx3::scoped_range l("Copying");
        thrust::copy(thrust::cuda::par.on(stream1), A.begin(), A.end(), d_a.begin());
        thrust::copy(thrust::cuda::par.on(stream2), B.begin(), B.end(), d_b.begin());
    }


    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cublasLtHandle_t Lthandle;
    cublasLtCreate(&Lthandle);

    // Create Layout
    cublasLtMatrixLayout_t adesc;
    cublasLtMatrixLayout_t bdesc;
    cublasLtMatrixLayout_t cdesc;

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&adesc, CUDA_R_16F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_16F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_16F, N, M, N));

    // Construct desc
    cublasLtMatmulDesc_t operation;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F)); // alpha & beta 's type

    // Construct half device_vector
    thrust::device_vector<half> d_a_h(M*K);
    thrust::device_vector<half> d_b_h(K*N);
    thrust::device_vector<half> d_c_h(M*N);

    // transform data from float to half
    thrust::transform(d_a.begin(), d_a.end(), d_a_h.begin(), floattohalf());
    thrust::transform(d_b.begin(), d_b.end(), d_b_h.begin(), floattohalf());
  
    // Preference for algo
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));

    // Set-up Workspace's max size
    size_t workspacesize = 128 * 1024 * 1024; // 32MB
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspacesize, sizeof(workspacesize)));

    // Search algo(Heuristic)
    cublasLtMatmulHeuristicResult_t heuristicresult;
    int returncount = 0;
{
    nvtx3::scoped_range p("Searching for algo");
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(Lthandle, operation, bdesc, adesc, cdesc, cdesc, preference, 1, &heuristicresult, &returncount));
    if (returncount == 0){
        fprintf(stderr, "Cannot find the algo.");
        exit(EXIT_FAILURE);
    }
}
    // Assign workspace
    thrust::device_vector<uint8_t> work; // Bytes, if you place "int", you actually apply 4* space
    if(heuristicresult.workspaceSize > 0){
        work.resize(heuristicresult.workspaceSize);
    };

    float alpha = 1.0f;
    float beta = 0.0f;
{
    nvtx3::scoped_range h("Calculation");
    CHECK_CUBLAS(cublasLtMatmul(Lthandle, operation, &alpha, thrust::raw_pointer_cast(d_b_h.data()), bdesc, thrust::raw_pointer_cast(d_a_h.data()), adesc, &beta, thrust::raw_pointer_cast(d_c_h.data()), cdesc, thrust::raw_pointer_cast(d_c_h.data()), cdesc, &heuristicresult.algo, thrust::raw_pointer_cast(work.data()), heuristicresult.workspaceSize, stream1));
}

    thrust::transform(d_c_h.begin(), d_c_h.end(), d_c.begin(), halftofloat());
    C = d_c;

    CHECK_CUBLAS(cublasLtDestroy(Lthandle));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operation));
    dellay()(adesc);
    dellay()(bdesc);
    dellay()(cdesc);
    delstr()(stream1);
    delstr()(stream2);

    exit(EXIT_SUCCESS);
}

/*

cd ~/cuda-course-master/06_CUDA_APIs
nvcc -arch=native practise_3.cu -o p3 -lcublas -lcublasLt
nsys profile -o ./report_p3 --force-overwrite true ./p3
ncu --set full -o ./ncu_report_p3 -f ./p3

*/
