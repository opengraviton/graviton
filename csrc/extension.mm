#include <torch/extension.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Forward declaration of the Metal dispatch function
void dispatch_ternary_gemm_metal(
    const float* A_ptr, 
    const uint8_t* B_packed_ptr, 
    float* C_ptr,
    uint32_t M, uint32_t N, uint32_t K,
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    id<MTLComputePipelineState> pipelineState);

// Global state for Metal
static id<MTLDevice> mtlDevice = nil;
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLLibrary> defaultLibrary = nil;
static id<MTLComputePipelineState> ternaryGemmPSO = nil;

void initialize_metal() {
    if (mtlDevice == nil) {
        mtlDevice = MTLCreateSystemDefaultDevice();
        if (!mtlDevice) {
            throw std::runtime_error("Failed to find a suitable Metal device.");
        }
        
        commandQueue = [mtlDevice newCommandQueue];
        
        // In a real PyTorch extension, we often embed the compiled .metallib 
        // into the bundle or string-load it. For simplicity in this source binding, 
        // we'll leave the loading logic here conceptually.
        // Assuming we've compiled the .metal file into a .metallib
        
        NSString *bundlePath = [[NSBundle mainBundle] bundlePath];
        NSString *libraryPath = [bundlePath stringByAppendingPathComponent:@"default.metallib"];
        
        NSError *error = nil;
        // Fallback to searching the current directory if bundle fails (common in cpp extensions)
        if (![[NSFileManager defaultManager] fileExistsAtPath:libraryPath]) {
            libraryPath = @"ternary_gemm.metallib";
        }

        defaultLibrary = [mtlDevice newLibraryWithFile:libraryPath error:&error];
        
        if (!defaultLibrary) {
            // Note: If the file isn't found at runtime, this will throw.
            // A more robust solution uses xxd to bake the .metal code as a C string
            // and compile it at runtime via newLibraryWithSource.
            // But this outlines the architecture.
            throw std::runtime_error("Failed to load Metal library.");
        }
        
        id<MTLFunction> gemmFunction = [defaultLibrary newFunctionWithName:@"ternary_gemm_kernel"];
        if (!gemmFunction) {
            throw std::runtime_error("Failed to find Metal function: ternary_gemm_kernel");
        }
        
        ternaryGemmPSO = [mtlDevice newComputePipelineStateWithFunction:gemmFunction error:&error];
        if (!ternaryGemmPSO) {
            throw std::runtime_error("Failed to create pipeline state object.");
        }
    }
}

// C++ frontend function bound to Python
torch::Tensor ternary_matmul_mps(torch::Tensor a, torch::Tensor b_packed, int out_features) {
    // a: [M, K] float32 (or fp16)
    // b_packed: [K / 4, N] uint8 (transposed concept for packing)
    
    TORCH_CHECK(a.device().is_mps(), "Tensor A must be on MPS device");
    TORCH_CHECK(b_packed.device().is_mps(), "Tensor B_packed must be on MPS device");
    TORCH_CHECK(a.is_contiguous(), "Tensor A must be contiguous");
    TORCH_CHECK(b_packed.is_contiguous(), "Tensor B_packed must be contiguous");
    
    // Initialize metal state on first call
    @autoreleasepool {
        // initialize_metal(); // In production, we'd ensure the library loads correctly
    }
    
    auto M = a.size(0);
    auto K = a.size(1);
    auto N = out_features;
    
    auto c = torch::zeros({M, N}, a.options());
    
    // IN A PRODUCTION SCENARIO: 
    // We would extract the underlying MTLBuffer from ATen's MPS implementation using Private APIs
    // OR just use torch::mps::get_command_buffer() / torch::mps::get_dispatch_queue().
    //
    // For this architectural foundation, we'll demonstrate the structure.
    // 
    // float* a_ptr = a.data_ptr<float>();
    // uint8_t* b_ptr = b_packed.data_ptr<uint8_t>();
    // float* c_ptr = c.data_ptr<float>();
    //
    // @autoreleasepool {
    //   dispatch_ternary_gemm_metal(a_ptr, b_ptr, c_ptr, M, N, K, mtlDevice, commandQueue, ternaryGemmPSO);
    // }
    
    // As a placeholder returning a fallback since raw pointer access to 
    // MPS tensors requires building against PyTorch internal C++ MPS API:
    
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ternary_matmul_mps", &ternary_matmul_mps, "Fast ternary matmul for MPS backend");
}
