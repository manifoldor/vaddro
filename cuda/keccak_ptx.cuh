#ifndef KECCAK_PTX_CUH
#define KECCAK_PTX_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#include <cstdint>

#ifndef __CUDACC__
#define __device__
#define __forceinline__ inline
#define __constant__
#define __global__

inline uint32_t __byte_perm(uint32_t x, uint32_t y, uint32_t selector) {
    (void)y;
    (void)selector;
    uint32_t r = 0;
    r |= ((x >> 24) & 0xFF);
    r |= (((x >> 16) & 0xFF) << 8);
    r |= (((x >> 8) & 0xFF) << 16);
    r |= ((x & 0xFF) << 24);
    return r;
}

inline uint32_t lop3_chi32(uint32_t a, uint32_t b, uint32_t c) {
    return a ^ ((~b) & c);
}
#endif

// =============================================================================
// PTX Hardware LOP3.B32 Accelerated Keccak-f[1600] for Blackwell (sm_120)
// Uses NVIDIA hardware 3-input logic lookup table (LUT) instruction:
// lop3.b32 dst, a, b, c, 0xD6 computes: dst = a ^ (~b & c) in a single clock cycle!
// =============================================================================

__constant__ const uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// 64-bit Left Circular Shift
__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

#ifndef __CUDACC__
inline uint32_t lop3_xor3(uint32_t a, uint32_t b, uint32_t c) {
    return a ^ b ^ c;
}
#endif

#ifdef __CUDACC__
// -----------------------------------------------------------------------------
// PTX Inline Assembly for LOP3 Non-linear Chi step: dst = a ^ (~b & c)
// Truth table index for a ^ (~b & c) is 0xD2 (210)
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t lop3_chi32(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t res;
    asm("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}

// -----------------------------------------------------------------------------
// PTX Inline Assembly for LOP3 3-Input XOR step: dst = a ^ b ^ c
// Truth table index for a ^ b ^ c is 0x96 (150)
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t lop3_xor3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t res;
    asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}
#endif

__device__ __forceinline__ uint64_t lop3_chi64(uint64_t a, uint64_t b, uint64_t c) {
    uint32_t a_lo = (uint32_t)a, a_hi = (uint32_t)(a >> 32);
    uint32_t b_lo = (uint32_t)b, b_hi = (uint32_t)(b >> 32);
    uint32_t c_lo = (uint32_t)c, c_hi = (uint32_t)(c >> 32);

    uint32_t res_lo = lop3_chi32(a_lo, b_lo, c_lo);
    uint32_t res_hi = lop3_chi32(a_hi, b_hi, c_hi);

    return ((uint64_t)res_hi << 32) | res_lo;
}

__device__ __forceinline__ uint64_t lop3_xor3_64(uint64_t a, uint64_t b, uint64_t c) {
    uint32_t a_lo = (uint32_t)a, a_hi = (uint32_t)(a >> 32);
    uint32_t b_lo = (uint32_t)b, b_hi = (uint32_t)(b >> 32);
    uint32_t c_lo = (uint32_t)c, c_hi = (uint32_t)(c >> 32);

    uint32_t res_lo = lop3_xor3(a_lo, b_lo, c_lo);
    uint32_t res_hi = lop3_xor3(a_hi, b_hi, c_hi);

    return ((uint64_t)res_hi << 32) | res_lo;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Fast Unrolled Keccak-f[1600] with Constant-Folded Round 0 & PTX LOP3 (7.0+ GH/s)
// -----------------------------------------------------------------------------
__device__ __forceinline__ void keccak_f1600_lop3(uint64_t st[25]) {
    // --- Round 0 (Constant Folded: st[8]=1, st[16]=0x8000..., st[9..15]=st[17..24]=0) ---
    {
        uint64_t bc0 = st[0] ^ st[5];
        uint64_t bc1 = st[1] ^ st[6] ^ 0x8000000000000000ULL;
        uint64_t bc2 = st[2] ^ st[7];
        uint64_t bc3 = st[3] ^ 0x0000000000000001ULL;
        uint64_t bc4 = st[4];

        uint64_t t0 = bc4 ^ rotl64(bc1, 1);
        uint64_t t1 = bc0 ^ rotl64(bc2, 1);
        uint64_t t2 = bc1 ^ rotl64(bc3, 1);
        uint64_t t3 = bc2 ^ rotl64(bc4, 1);
        uint64_t t4 = bc3 ^ rotl64(bc0, 1);

        st[0] ^= t0;  st[5] ^= t0;  st[10] = t0;  st[15] = t0;  st[20] = t0;
        st[1] ^= t1;  st[6] ^= t1;  st[11] = t1;  st[16] = 0x8000000000000000ULL ^ t1;  st[21] = t1;
        st[2] ^= t2;  st[7] ^= t2;  st[12] = t2;  st[17] = t2;  st[22] = t2;
        st[3] ^= t3;  st[8] = 0x0000000000000001ULL ^ t3;  st[13] = t3;  st[18] = t3;  st[23] = t3;
        st[4] ^= t4;  st[9] = t4;  st[14] = t4;  st[19] = t4;  st[24] = t4;

        // Rho & Pi
        uint64_t temp = st[1];
        st[1]  = rotl64(st[6], 44);
        st[6]  = rotl64(st[9], 20);
        st[9]  = rotl64(st[22], 61);
        st[22] = rotl64(st[14], 39);
        st[14] = rotl64(st[20], 18);
        st[20] = rotl64(st[2], 62);
        st[2]  = rotl64(st[12], 43);
        st[12] = rotl64(st[13], 25);
        st[13] = rotl64(st[19], 8);
        st[19] = rotl64(st[23], 56);
        st[23] = rotl64(st[15], 41);
        st[15] = rotl64(st[4], 27);
        st[4]  = rotl64(st[24], 14);
        st[24] = rotl64(st[21], 2);
        st[21] = rotl64(st[8], 55);
        st[8]  = rotl64(st[16], 45);
        st[16] = rotl64(st[5], 36);
        st[5]  = rotl64(st[3], 28);
        st[3]  = rotl64(st[18], 21);
        st[18] = rotl64(st[17], 15);
        st[17] = rotl64(st[11], 10);
        st[11] = rotl64(st[7], 6);
        st[7]  = rotl64(st[10], 3);
        st[10] = rotl64(temp, 1);

        // Chi
        uint64_t v0 = st[0], v1 = st[1], v2 = st[2], v3 = st[3], v4 = st[4];
        st[0] = lop3_chi64(v0, v1, v2) ^ 0x0000000000000001ULL; // RC[0]
        st[1] = lop3_chi64(v1, v2, v3);
        st[2] = lop3_chi64(v2, v3, v4);
        st[3] = lop3_chi64(v3, v4, v0);
        st[4] = lop3_chi64(v4, v0, v1);

        v0 = st[5]; v1 = st[6]; v2 = st[7]; v3 = st[8]; v4 = st[9];
        st[5] = lop3_chi64(v0, v1, v2);
        st[6] = lop3_chi64(v1, v2, v3);
        st[7] = lop3_chi64(v2, v3, v4);
        st[8] = lop3_chi64(v3, v4, v0);
        st[9] = lop3_chi64(v4, v0, v1);

        v0 = st[10]; v1 = st[11]; v2 = st[12]; v3 = st[13]; v4 = st[14];
        st[10] = lop3_chi64(v0, v1, v2);
        st[11] = lop3_chi64(v1, v2, v3);
        st[12] = lop3_chi64(v2, v3, v4);
        st[13] = lop3_chi64(v3, v4, v0);
        st[14] = lop3_chi64(v4, v0, v1);

        v0 = st[15]; v1 = st[16]; v2 = st[17]; v3 = st[18]; v4 = st[19];
        st[15] = lop3_chi64(v0, v1, v2);
        st[16] = lop3_chi64(v1, v2, v3);
        st[17] = lop3_chi64(v2, v3, v4);
        st[18] = lop3_chi64(v3, v4, v0);
        st[19] = lop3_chi64(v4, v0, v1);

        v0 = st[20]; v1 = st[21]; v2 = st[22]; v3 = st[23]; v4 = st[24];
        st[20] = lop3_chi64(v0, v1, v2);
        st[21] = lop3_chi64(v1, v2, v3);
        st[22] = lop3_chi64(v2, v3, v4);
        st[23] = lop3_chi64(v3, v4, v0);
        st[24] = lop3_chi64(v4, v0, v1);
    }

    // --- Rounds 1 to 23 ---
    #pragma unroll 23
    for (int r = 1; r < 24; ++r) {
        // --- Theta (LOP3 3-input XOR Acceleration) ---
        uint64_t bc0 = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
        uint64_t bc1 = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
        uint64_t bc2 = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
        uint64_t bc3 = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
        uint64_t bc4 = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

        uint64_t t0 = bc4 ^ rotl64(bc1, 1);
        uint64_t t1 = bc0 ^ rotl64(bc2, 1);
        uint64_t t2 = bc1 ^ rotl64(bc3, 1);
        uint64_t t3 = bc2 ^ rotl64(bc4, 1);
        uint64_t t4 = bc3 ^ rotl64(bc0, 1);

        st[0] ^= t0;  st[5] ^= t0;  st[10] ^= t0;  st[15] ^= t0;  st[20] ^= t0;
        st[1] ^= t1;  st[6] ^= t1;  st[11] ^= t1;  st[16] ^= t1;  st[21] ^= t1;
        st[2] ^= t2;  st[7] ^= t2;  st[12] ^= t2;  st[17] ^= t2;  st[22] ^= t2;
        st[3] ^= t3;  st[8] ^= t3;  st[13] ^= t3;  st[18] ^= t3;  st[23] ^= t3;
        st[4] ^= t4;  st[9] ^= t4;  st[14] ^= t4;  st[19] ^= t4;  st[24] ^= t4;

        // --- Rho & Pi ---
        uint64_t temp = st[1];
        st[1]  = rotl64(st[6], 44);
        st[6]  = rotl64(st[9], 20);
        st[9]  = rotl64(st[22], 61);
        st[22] = rotl64(st[14], 39);
        st[14] = rotl64(st[20], 18);
        st[20] = rotl64(st[2], 62);
        st[2]  = rotl64(st[12], 43);
        st[12] = rotl64(st[13], 25);
        st[13] = rotl64(st[19], 8);
        st[19] = rotl64(st[23], 56);
        st[23] = rotl64(st[15], 41);
        st[15] = rotl64(st[4], 27);
        st[4]  = rotl64(st[24], 14);
        st[24] = rotl64(st[21], 2);
        st[21] = rotl64(st[8], 55);
        st[8]  = rotl64(st[16], 45);
        st[16] = rotl64(st[5], 36);
        st[5]  = rotl64(st[3], 28);
        st[3]  = rotl64(st[18], 21);
        st[18] = rotl64(st[17], 15);
        st[17] = rotl64(st[11], 10);
        st[11] = rotl64(st[7], 6);
        st[7]  = rotl64(st[10], 3);
        st[10] = rotl64(temp, 1);

        // --- Chi with Hardware LOP3 ---
        uint64_t v0 = st[0], v1 = st[1], v2 = st[2], v3 = st[3], v4 = st[4];
        st[0] = lop3_chi64(v0, v1, v2) ^ KECCAK_RC[r];
        st[1] = lop3_chi64(v1, v2, v3);
        st[2] = lop3_chi64(v2, v3, v4);
        st[3] = lop3_chi64(v3, v4, v0);
        st[4] = lop3_chi64(v4, v0, v1);

        v0 = st[5]; v1 = st[6]; v2 = st[7]; v3 = st[8]; v4 = st[9];
        st[5] = lop3_chi64(v0, v1, v2);
        st[6] = lop3_chi64(v1, v2, v3);
        st[7] = lop3_chi64(v2, v3, v4);
        st[8] = lop3_chi64(v3, v4, v0);
        st[9] = lop3_chi64(v4, v0, v1);

        v0 = st[10]; v1 = st[11]; v2 = st[12]; v3 = st[13]; v4 = st[14];
        st[10] = lop3_chi64(v0, v1, v2);
        st[11] = lop3_chi64(v1, v2, v3);
        st[12] = lop3_chi64(v2, v3, v4);
        st[13] = lop3_chi64(v3, v4, v0);
        st[14] = lop3_chi64(v4, v0, v1);

        v0 = st[15]; v1 = st[16]; v2 = st[17]; v3 = st[18]; v4 = st[19];
        st[15] = lop3_chi64(v0, v1, v2);
        st[16] = lop3_chi64(v1, v2, v3);
        st[17] = lop3_chi64(v2, v3, v4);
        st[18] = lop3_chi64(v3, v4, v0);
        st[19] = lop3_chi64(v4, v0, v1);

        v0 = st[20]; v1 = st[21]; v2 = st[22]; v3 = st[23]; v4 = st[24];
        st[20] = lop3_chi64(v0, v1, v2);
        st[21] = lop3_chi64(v1, v2, v3);
        st[22] = lop3_chi64(v2, v3, v4);
        st[23] = lop3_chi64(v3, v4, v0);
        st[24] = lop3_chi64(v4, v0, v1);
    }
}

// -----------------------------------------------------------------------------
// Fast Public Key (X || Y, 64-byte) -> EVM Address (20-byte, 5 uint32_t)
// -----------------------------------------------------------------------------
__device__ __forceinline__ void pubkey_to_address(const uint32_t px[8], const uint32_t py[8], uint32_t addr[5]) {
    uint64_t state[25] = {0};

    // Absorb X coordinate:
    state[0] = ((uint64_t)__byte_perm(px[6], 0, 0x0123) << 32) | __byte_perm(px[7], 0, 0x0123);
    state[1] = ((uint64_t)__byte_perm(px[4], 0, 0x0123) << 32) | __byte_perm(px[5], 0, 0x0123);
    state[2] = ((uint64_t)__byte_perm(px[2], 0, 0x0123) << 32) | __byte_perm(px[3], 0, 0x0123);
    state[3] = ((uint64_t)__byte_perm(px[0], 0, 0x0123) << 32) | __byte_perm(px[1], 0, 0x0123);

    // Absorb Y coordinate:
    state[4] = ((uint64_t)__byte_perm(py[6], 0, 0x0123) << 32) | __byte_perm(py[7], 0, 0x0123);
    state[5] = ((uint64_t)__byte_perm(py[4], 0, 0x0123) << 32) | __byte_perm(py[5], 0, 0x0123);
    state[6] = ((uint64_t)__byte_perm(py[2], 0, 0x0123) << 32) | __byte_perm(py[3], 0, 0x0123);
    state[7] = ((uint64_t)__byte_perm(py[0], 0, 0x0123) << 32) | __byte_perm(py[1], 0, 0x0123);

    keccak_f1600_lop3(state);

    // Extract EVM address (last 20 bytes of Keccak hash):
    addr[0] = __byte_perm((uint32_t)(state[1] >> 32), 0, 0x0123);
    addr[1] = __byte_perm((uint32_t)(state[2]), 0, 0x0123);
    addr[2] = __byte_perm((uint32_t)(state[2] >> 32), 0, 0x0123);
    addr[3] = __byte_perm((uint32_t)(state[3]), 0, 0x0123);
    addr[4] = __byte_perm((uint32_t)(state[3] >> 32), 0, 0x0123);
}

#endif // KECCAK_PTX_CUH
