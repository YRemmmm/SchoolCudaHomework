#include <cuda_runtime.h>
#include <iostream>

void printSMProperties(const cudaDeviceProp& prop) {
    std::cout << "  Threads per Warp: " << 32 << std::endl;
    std::cout << "  Max Warps per Multiprocessor: " << prop.maxThreadsPerMultiProcessor / 32 << std::endl;
    std::cout << "  Max Thread Blocks per Multiprocessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "  Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Maximum Thread Block Size: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Registers per Multiprocessor: " << prop.regsPerMultiprocessor << std::endl;
    std::cout << "  Max Registers per Thread Block: " << prop.regsPerBlock << std::endl;
    std::cout << "  Max Registers per Thread: " << 255 << std::endl;
    std::cout << "  Shared Memory per Multiprocessor (bytes): " << prop.sharedMemPerMultiprocessor << std::endl;
    std::cout << "  Max Shared Memory per Block: " << prop.sharedMemPerBlock << std::endl;
}

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
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;

        printSMProperties(prop);
    }

    return 0;
}
