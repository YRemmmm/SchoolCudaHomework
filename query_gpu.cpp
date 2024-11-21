#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "There are no available CUDA devices." << std::endl;
        return 1;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Total global memory: " << prop.totalGlobalMem / 1024 / 1024 << " MB" << std::endl;
        std::cout << "  Number of SMs: " << prop.multiProcessorCount << std::endl;

        int cudaCoresPerSM = 128; // For Ampere architecture
        int totalCudaCores = prop.multiProcessorCount * cudaCoresPerSM;

        std::cout << "  CUDA Cores per SM: " << cudaCoresPerSM << std::endl;
        std::cout << "  Total CUDA Cores: " << totalCudaCores << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    }

    return 0;
}
