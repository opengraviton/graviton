#include <metal_stdlib>
using namespace metal;

/*
 * Ternary Matrix Multiplication (Addition-only) Kernel
 *
 * In a 1.58-bit ternary network, weights are strictly {-1, 0, 1}.
 * We pack 4 weights (trits) into a single 8-bit integer (byte).
 *
 * The math:
 * value = unpacked_weight_from_byte
 * if value == 1:   acc += activation
 * if value == -1:  acc -= activation
 * if value == 0:   do nothing
 * 
 * This avoids expensive floating-point multiplication completely.
 */

kernel void ternary_gemm_kernel(
    device const float* A [[buffer(0)]],        // Input activations (M x K)
    device const uint8_t* B_packed [[buffer(1)]], // Packed ternary weights (K x N, packed K/4)
    device float* C [[buffer(2)]],              // Output (M x N)
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    // C is an (M x N) matrix.
    // gid.x corresponds to column n in [0, N)
    // gid.y corresponds to row m in [0, M)
    
    uint n = gid.x;
    uint m = gid.y;
    
    if (m >= M || n >= N) {
        return;
    }
    
    float acc = 0.0f;
    
    // Each row of A has K elements.
    uint a_row_start = m * K;
    
    // B is stored conceptually as (K, N), but it's packed along the K dimension.
    // So for a given column `n`, we traverse the `k` dimension.
    // 4 weights are packed into 1 byte. Let's assume K is a multiple of 4.
    uint k_packed_size = K / 4;
    uint b_col_start = n * k_packed_size;
    
    for (uint p = 0; p < k_packed_size; ++p) {
        // Fetch 4 packed ternary weights for column `n`, byte `p`
        // Weights are packed as 2 bits each: {w3, w2, w1, w0} -> 8 bits
        // Values: 00 -> 0, 01 -> +1, 10 -> -1
        uint8_t packed_val = B_packed[b_col_start + p];
        
        uint k_base = p * 4;
        
        // Unroll the 4 trits operations
        for (int i = 0; i < 4; ++i) {
            uint trit_val = (packed_val >> (i * 2)) & 0x03;
            float a_val = A[a_row_start + k_base + i];
            
            // Branchless arithmetic to replace multiplication
            // trit_val:
            // 0 -> 0 -> 0.0f
            // 1 -> +1 -> +a_val
            // 2 -> -1 -> -a_val
            
            // An efficient trick is to use an array or math, but basic branching
            // is often optimized out by the shader compiler.
            if (trit_val == 1) {
                acc += a_val;
            } else if (trit_val == 2) {
                acc -= a_val;
            }
        }
    }
    
    C[m * N + n] = acc;
}
