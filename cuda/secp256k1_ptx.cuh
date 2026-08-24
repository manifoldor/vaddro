#ifndef SECP256K1_PTX_CUH
#define SECP256K1_PTX_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#include <cstdint>

#ifndef __CUDACC__
#define __device__
#define __forceinline__ inline
#define __constant__
#define __global__
#endif

// Keep the qualified binary-exponentiation path as the production default.
// The fixed-chain candidate passed correctness checks but did not improve the
// measured end-to-end RTX 5090 search rate.
#ifndef VADDRO_INV_ADDITION_CHAIN
#define VADDRO_INV_ADDITION_CHAIN 0
#endif

#if VADDRO_INV_ADDITION_CHAIN != 0 && VADDRO_INV_ADDITION_CHAIN != 1
#error "VADDRO_INV_ADDITION_CHAIN must be 0 or 1"
#endif

// =============================================================================
// secp256k1 Elliptic Curve & PTX Hardware Carry-Chain BigInt Arithmetic
// Modulus p = 2^256 - 2^32 - 977 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// =============================================================================

struct __align__(16) uint256_t {
    uint32_t d[8];
};

struct PointAffine {
    uint256_t x;
    uint256_t y;
};

__device__ __forceinline__ void load_u256(uint32_t out[8], const uint256_t *p) {
    const uint4 *v = reinterpret_cast<const uint4 *>(p);
    uint4 a = v[0];
    uint4 b = v[1];
    out[0] = a.x;
    out[1] = a.y;
    out[2] = a.z;
    out[3] = a.w;
    out[4] = b.x;
    out[5] = b.y;
    out[6] = b.z;
    out[7] = b.w;
}

__device__ __forceinline__ void store_u256(uint256_t *p, const uint32_t in[8]) {
    uint4 *v = reinterpret_cast<uint4 *>(p);
    uint4 a = make_uint4(in[0], in[1], in[2], in[3]);
    uint4 b = make_uint4(in[4], in[5], in[6], in[7]);
    v[0] = a;
    v[1] = b;
}

// secp256k1 prime field modulus p in little-endian 32-bit words
__constant__ const uint32_t SECP256K1_P[8] = {
    0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
    0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
};

// secp256k1 generator G
__constant__ const uint32_t SECP256K1_GX[8] = {
    0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
    0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu
};

__constant__ const uint32_t SECP256K1_GY[8] = {
    0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
    0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u
};

// -G_x mod p = p - G_x
__constant__ const uint32_t SECP256K1_NEG_GX[8] = {
    0xE907E497u, 0xA60D7EA3u, 0xD231D726u, 0xFD640324u,
    0x3178F4F8u, 0xAA5F9D6Au, 0x06234453u, 0x86419981u
};

// -G_y mod p = p - G_y
__constant__ const uint32_t SECP256K1_NEG_GY[8] = {
    0x04EF2777u, 0x63B82F6Fu, 0x597AABE6u, 0x02E84BB7u,
    0xF1EEF757u, 0xA25B0403u, 0xD95C3B9Au, 0xB7C52588u
};

// -2*G_y mod p = p - 2*G_y
__constant__ const uint32_t SECP256K1_NEG_DOUBLE_GY[8] = {
    0x09DE52BFu, 0xC7705EDFu, 0xB2F557CCu, 0x05D0976Eu,
    0xE3DDEEAEu, 0x44B60807u, 0xB2B87735u, 0x6F8A4B11u
};

// secp256k1 GLV endomorphism beta and beta^2.
__constant__ const uint32_t SECP256K1_BETA[8] = {
    0x719501EEu, 0xC1396C28u, 0x12F58995u, 0x9CF04975u,
    0xAC3434E9u, 0x6E64479Eu, 0x657C0710u, 0x7AE96A2Bu
};

__constant__ const uint32_t SECP256K1_BETA2[8] = {
    0x8E6AFA40u, 0x3EC693D6u, 0xED0A766Au, 0x630FB68Au,
    0x53CBCB16u, 0x919BB861u, 0x9A83F8EFu, 0x851695D4u
};

// -3*G_x mod p
__constant__ const uint32_t SECP256K1_TRIPLE_NEG_GX[8] = {
    0xBB17B196u, 0xF2287BECu, 0x76958573u, 0xF82C096Eu,
    0x946ADEEAu, 0xFF1ED83Eu, 0x1269CCFAu, 0x92C4CC83u
};

// Precomputed powers of 2 for generator point: 2^i * G for i = 0..23 (24 points)
__constant__ const uint32_t SECP256K1_POW2_G[24][16] = {
    {0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu, 0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu, 0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u, 0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u},
    {0x5C709EE5u, 0xABAC09B9u, 0x8CEF3CA7u, 0x5C778E4Bu, 0x95C07CD8u, 0x3045406Eu, 0x41ED7D6Du, 0xC6047F94u, 0x50CFE52Au, 0x236431A9u, 0x3266D0E1u, 0xF7F63265u, 0x466CEAEEu, 0xA3C58419u, 0xA63DC339u, 0x1AE168FEu},
    {0xE8C4CD13u, 0x74FA94ABu, 0x0EE07584u, 0xCC6C1390u, 0x930B1404u, 0x581E4904u, 0xC10D80F3u, 0xE493DBF1u, 0x47739922u, 0xCFE97BDCu, 0xBFBDFE40u, 0xD967AE33u, 0x8EA51448u, 0x5642E209u, 0xA0D455B7u, 0x51ED993Eu},
    {0xE10A2A01u, 0x67784EF3u, 0xE5AF888Au, 0x0A1BDD05u, 0xB70F3C2Fu, 0xAFF3843Fu, 0x5CCA351Du, 0x2F01E5E1u, 0x6CBDE904u, 0xB5DA2CB7u, 0xBA5B7617u, 0xC2E213D6u, 0x132D13B4u, 0x293D082Au, 0x41539949u, 0x5C4DA8A7u},
    {0x2A6DEC0Au, 0xC44EE89Eu, 0xB87A5AE9u, 0xB2A31369u, 0x21C23E97u, 0x3011AABCu, 0xB59E9EC5u, 0xE60FCE93u, 0x69616821u, 0xE1F32CCEu, 0x44D23F0Bu, 0x1296891Eu, 0xF5793710u, 0x9DB99F34u, 0x99E59592u, 0xF7E35073u},
    {0x07143E65u, 0x75D0DBD4u, 0x9904A61Du, 0xDACFFCB8u, 0xE2F378CEu, 0x47B6E054u, 0x4FB5A22Du, 0xD30199D7u, 0x24106AB9u, 0x05B3FF1Fu, 0x64ED8196u, 0x1F760CC3u, 0xE9838065u, 0xB3D6DEC9u, 0x0AE3D5C3u, 0x95038D9Du},
    {0xF874EF8Bu, 0xE37918E6u, 0xCDBAFD81u, 0xFC4C6F1Du, 0xF832823Cu, 0x0B1051EAu, 0x2D16EAB7u, 0xBF23C154u, 0x66831D9Fu, 0x4DC37EFEu, 0x811E2F78u, 0xC522FC54u, 0xBA5392E4u, 0x7AD928A0u, 0xC3300373u, 0x5CB3866Fu},
    {0x6769A24Eu, 0x64707745u, 0x00535655u, 0xBCF55CD7u, 0xF7D1671Cu, 0x696C3D09u, 0x033F7A06u, 0x34FF3BE4u, 0x73CC2F1Au, 0x8491067Au, 0xE8F8B681u, 0x55DF16C3u, 0x9832098Cu, 0x3F6619D8u, 0x3A236C55u, 0x5D9D1162u},
    {0xD5F51508u, 0x0646E23Fu, 0xD5AC1CA1u, 0xD8C39CABu, 0x172DE238u, 0xEA2A6E3Eu, 0x12C609D9u, 0x82822632u, 0xF6E26CAFu, 0xD31B6EAFu, 0x2F7B17BEu, 0x62D613ACu, 0x30B60ACEu, 0x5E8256E8u, 0x8557DFE4u, 0x11F8A809u},
    {0xEF34A24Du, 0x926E2C00u, 0xD9E159D0u, 0x0ADBC968u, 0x9CF918D5u, 0x905A857Au, 0x87A79FF3u, 0x465370B2u, 0xB33887F4u, 0xA2F8FB20u, 0x15D37A10u, 0x588E09B2u, 0xFDEEC2C1u, 0xA4AF8BDAu, 0x8368C082u, 0x35E531B3u},
    {0xE285131Fu, 0xD5B901B2u, 0xC813B088u, 0xAAEC6ECDu, 0x66AD6240u, 0xD664A18Fu, 0xE23CBD77u, 0x241FEBB8u, 0x2750026Du, 0xABB3E66Fu, 0xBD0CB5AFu, 0xCD50FD0Fu, 0x13981DF8u, 0xD6C420BDu, 0xFF94F8D3u, 0x513378D9u},
    {0x9B920471u, 0xEDCB6306u, 0xF423DE0Du, 0xFC318B85u, 0x83D8F8D9u, 0xFCE4CC29u, 0xA172FA79u, 0x5D1BDB4Eu, 0xF7B83103u, 0x70330666u, 0x96C56E7Bu, 0x79EB1E99u, 0x38A22656u, 0x794BB994u, 0x79379E2Eu, 0x28438267u},
    {0x51E5B739u, 0x73FCE5B5u, 0xFD2222EDu, 0xE0B93833u, 0xC6FC846Du, 0x72F99CC6u, 0x728B865Au, 0x175E159Fu, 0xE9FED695u, 0x6EFA6FFEu, 0xDD24345Cu, 0xACB5955Au, 0x1FF71F5Eu, 0xA4EF97A5u, 0x9E3C79EBu, 0xD3506E0Du},
    {0x46BC47D6u, 0xCE78049Eu, 0xC6DA121Bu, 0x30FDFEB5u, 0xE139C621u, 0xA5FFBCC8u, 0x03FF32D7u, 0x423A013Fu, 0x8B548A34u, 0xD1236E6Du, 0x524F009Eu, 0x720D8EC3u, 0xBAF6B3C7u, 0xA1179F7Bu, 0xE1E1D970u, 0xB91AE00Fu},
    {0x8416824Au, 0x302DF6F7u, 0x3B3E2741u, 0x49DF662Fu, 0xBCD68776u, 0x08907A7Au, 0xAC1FB905u, 0x111D6A45u, 0x2108E9D0u, 0xA9A111D4u, 0x96DACA4Cu, 0xF0700089u, 0xFF065952u, 0xB90D48DBu, 0x478EAFFBu, 0x0696911Cu},
    {0x1BCED775u, 0xAB5EDDE0u, 0x5EF74E56u, 0x7290B68Au, 0xBCB9DCFFu, 0xAD795DBEu, 0x7AC7C8B8u, 0x4A4A6DC9u, 0xA78DD66Du, 0xB2BF8F68u, 0x424742ACu, 0x1DE90CDBu, 0x39C0F457u, 0x943EF9F7u, 0x16631E72u, 0x529911B0u},
    {0x83FF4640u, 0x526BAD8Fu, 0x55552FFEu, 0x53441C7Eu, 0xB6262EE0u, 0x99CEAC05u, 0x47B00C9Cu, 0x363D90D4u, 0x3BEE9DE9u, 0x62003C7Fu, 0x08199ECBu, 0x45B9A890u, 0x97F33631u, 0x953B4453u, 0xFC732221u, 0x04E273ADu},
    {0xAF92C541u, 0xD2712A5Cu, 0x2EDFB59Du, 0x0B62FB01u, 0xC93B02BFu, 0x553973C6u, 0xED9A7E9Bu, 0x4C1B9866u, 0xC68FE020u, 0x72C4F3FDu, 0xE43EB1ADu, 0xC652EAD7u, 0xCE56E69Cu, 0x7FBCB753u, 0x20BE8A0Fu, 0xC1F792D3u},
    {0xBEAAF3D1u, 0xE0C8F28Cu, 0x1A28F135u, 0x3233EDBCu, 0x0780B54Eu, 0x529A2F3Cu, 0xBA83B12Bu, 0xA4083877u, 0xF0B254B9u, 0xB12B534Du, 0x6ED1EF90u, 0x2001E757u, 0x9361B3E2u, 0xB8BF83D6u, 0xFEEFBC79u, 0x40E9F612u},
    {0x7940D33Au, 0x8CD3DB5Au, 0x454203B9u, 0x6E0D880Au, 0x2F56C86Fu, 0x3A4E3E1Au, 0xD28CC0B5u, 0xA804C641u, 0x6D46967Au, 0x43CE95FAu, 0x9CF736A9u, 0x1AF18CA8u, 0xC16047E8u, 0x3DEC2842u, 0x2B2FA6D0u, 0x95BE8325u},
    {0xDFFDF80Cu, 0x69F79A55u, 0xA15BCD1Bu, 0x43E4A781u, 0xB7456388u, 0x8C6244B5u, 0x5DF3C2BEu, 0x8B4B5F16u, 0x5FD4FD36u, 0xB3EFF0C6u, 0x6162EE56u, 0xF9E33654u, 0x3AB0DA04u, 0xB3FBD781u, 0x68D308B4u, 0x4AAD0A6Fu},
    {0x4755E4BEu, 0xBB3F2671u, 0x417C997Au, 0x71AF64EEu, 0xC83C6110u, 0x8CE17C7Eu, 0xE1329171u, 0xED0C5CE4u, 0xA07BF42Fu, 0x63F9FA6Eu, 0x763DDAB1u, 0x49D93925u, 0x5A7EA680u, 0xBF3DAD7Fu, 0xBC2345BDu, 0x221A9FC7u},
    {0x07CEC8ABu, 0xD4B6E6C8u, 0x6E0552CEu, 0xE5325456u, 0x83F1FAE8u, 0xB3B15C3Fu, 0xC44CE694u, 0xFAECB013u, 0xAB155070u, 0x8981DFD9u, 0xB85E2E2Eu, 0x9C32B286u, 0xEC2FB13Du, 0xFC2E02C6u, 0x0E9ECB57u, 0xCC09B5E9u},
    {0x9AD1B1F7u, 0x07F64EAEu, 0x4F23CD3Eu, 0xB3B2DD82u, 0x37CBCAFDu, 0xC8731A0Bu, 0x2DCAD2F2u, 0x09BB8A13u, 0xC3811C80u, 0x64130627u, 0xF4752D53u, 0x0F54A840u, 0x4F863E85u, 0xB6F9DD28u, 0xAFEEE3B9u, 0x945BB2B2u},
};

#ifndef __CUDACC__
inline uint32_t add256_ptx(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; ++i) {
        uint64_t sum = (uint64_t)a[i] + (uint64_t)b[i] + carry;
        r[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
    return (uint32_t)carry;
}

inline uint32_t sub256_ptx(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    int64_t borrow = 0;
    for (int i = 0; i < 8; ++i) {
        int64_t diff = (int64_t)a[i] - (int64_t)b[i] - borrow;
        r[i] = (uint32_t)diff;
        borrow = (diff < 0) ? 1 : 0;
    }
    return (uint32_t)borrow;
}
#else
// -----------------------------------------------------------------------------
// PTX 256-bit Addition with Hardware Carry Chain
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t add256_ptx(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    uint32_t carry = 0;
    asm volatile(
        "add.cc.u32   %0,  %9, %17;\n\t"
        "addc.cc.u32  %1, %10, %18;\n\t"
        "addc.cc.u32  %2, %11, %19;\n\t"
        "addc.cc.u32  %3, %12, %20;\n\t"
        "addc.cc.u32  %4, %13, %21;\n\t"
        "addc.cc.u32  %5, %14, %22;\n\t"
        "addc.cc.u32  %6, %15, %23;\n\t"
        "addc.cc.u32  %7, %16, %24;\n\t"
        "addc.u32     %8,   0,   0;\n\t"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
          "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
          "=r"(carry)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7])
    );
    return carry;
}

// -----------------------------------------------------------------------------
// PTX 256-bit Subtraction with Hardware Borrow Chain
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t sub256_ptx(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    uint32_t borrow = 0;
    asm volatile(
        "sub.cc.u32   %0,  %9, %17;\n\t"
        "subc.cc.u32  %1, %10, %18;\n\t"
        "subc.cc.u32  %2, %11, %19;\n\t"
        "subc.cc.u32  %3, %12, %20;\n\t"
        "subc.cc.u32  %4, %13, %21;\n\t"
        "subc.cc.u32  %5, %14, %22;\n\t"
        "subc.cc.u32  %6, %15, %23;\n\t"
        "subc.cc.u32  %7, %16, %24;\n\t"
        "subc.u32     %8,   0,   0;\n\t"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
          "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
          "=r"(borrow)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7])
    );
    return borrow;
}
#endif

__device__ __forceinline__ bool gte256(const uint32_t a[8], const uint32_t b[8]) {
    for (int i = 7; i >= 0; --i) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;
}

__device__ __forceinline__ void add_mod_p(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    uint32_t sum[8];
    uint32_t carry = add256_ptx(sum, a, b);

    if (carry || gte256(sum, SECP256K1_P)) {
        sub256_ptx(r, sum, SECP256K1_P);
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) r[i] = sum[i];
    }
}

__device__ __forceinline__ void sub_mod_p(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    uint32_t diff[8];
    uint32_t borrow = sub256_ptx(diff, a, b);

    if (borrow) {
        add256_ptx(r, diff, SECP256K1_P);
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) r[i] = diff[i];
    }
}

__device__ __forceinline__ void mul256(uint32_t r[16], const uint32_t a[8], const uint32_t b[8]) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) r[i] = 0;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint64_t prod = (uint64_t)a[i] * (uint64_t)b[j] + (uint64_t)r[i + j] + (uint64_t)carry;
            r[i + j] = (uint32_t)prod;
            carry = (uint32_t)(prod >> 32);
        }
        r[i + 8] += carry;
    }
}

__device__ __forceinline__ void reduce512_mod_p(uint32_t r[8], const uint32_t in[16]) {
    uint32_t mid[10] = {0};
    uint64_t carry = 0;

    // R = low + high * (2^32 + 977) = low + high*977 + (high << 32).
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
        uint64_t sum = carry;
        if (i < 8) {
            sum += (uint64_t)in[i];
            sum += (uint64_t)in[i + 8] * 977ULL;
        }
        if (i > 0 && i <= 8) {
            sum += (uint64_t)in[i + 7];
        }
        mid[i] = (uint32_t)sum;
        carry = sum >> 32;
    }

    uint64_t top = ((uint64_t)mid[9] << 32) | (uint64_t)mid[8];
    uint64_t top_c = top * 977ULL;

    // top * 2^256 ≡ top * (2^32 + 977) mod p
    // Shift component: top * 2^32 means mid[8] adds to word 1, mid[9] adds to word 2
    carry = 0;
    uint64_t s0 = (uint64_t)mid[0] + (uint32_t)top_c;
    r[0] = (uint32_t)s0;
    carry = s0 >> 32;

    uint64_t s1 = (uint64_t)mid[1] + (uint32_t)(top_c >> 32) + (uint64_t)mid[8] + carry;
    r[1] = (uint32_t)s1;
    carry = s1 >> 32;

    uint64_t s2 = (uint64_t)mid[2] + (uint64_t)mid[9] + carry;
    r[2] = (uint32_t)s2;
    carry = s2 >> 32;

    #pragma unroll
    for (int i = 3; i < 8; ++i) {
        uint64_t s = (uint64_t)mid[i] + carry;
        r[i] = (uint32_t)s;
        carry = s >> 32;
    }

    while (carry || gte256(r, SECP256K1_P)) {
        sub256_ptx(r, r, SECP256K1_P);
        carry = 0;
    }
}

__device__ __forceinline__ void mul_mod_p(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    uint32_t prod[16];
    mul256(prod, a, b);
    reduce512_mod_p(r, prod);
}

__device__ __forceinline__ void sqr256(uint32_t r[16], const uint32_t a[8]) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) r[i] = 0;

    // 1. Cross products
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t carry = 0;
        #pragma unroll
        for (int j = i + 1; j < 8; ++j) {
            uint64_t p = (uint64_t)a[i] * (uint64_t)a[j] + (uint64_t)r[i + j] + (uint64_t)carry;
            r[i + j] = (uint32_t)p;
            carry = (uint32_t)(p >> 32);
        }
        int k = i + 8;
        while (carry > 0 && k < 16) {
            uint64_t p = (uint64_t)r[k] + (uint64_t)carry;
            r[k] = (uint32_t)p;
            carry = (uint32_t)(p >> 32);
            k++;
        }
    }

    // 2. Shift left by 1 bit (multiply by 2)
    uint32_t carry = 0;
    #pragma unroll
    for (int i = 1; i < 16; ++i) {
        uint64_t val = ((uint64_t)r[i] << 1) | (uint64_t)carry;
        r[i] = (uint32_t)val;
        carry = (uint32_t)(val >> 32);
    }

    // 3. Add diagonal squares a[i]^2
    carry = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t p = (uint64_t)a[i] * (uint64_t)a[i];
        uint32_t p_lo = (uint32_t)p;
        uint32_t p_hi = (uint32_t)(p >> 32);

        uint64_t s0 = (uint64_t)r[2 * i] + (uint64_t)p_lo + (uint64_t)carry;
        r[2 * i] = (uint32_t)s0;
        carry = (uint32_t)(s0 >> 32);

        uint64_t s1 = (uint64_t)r[2 * i + 1] + (uint64_t)p_hi + (uint64_t)carry;
        r[2 * i + 1] = (uint32_t)s1;
        carry = (uint32_t)(s1 >> 32);
    }
}

__device__ __forceinline__ void sqr_mod_p(uint32_t r[8], const uint32_t a[8]) {
    uint32_t prod[16];
    sqr256(prod, a);
    reduce512_mod_p(r, prod);
}

// Reference implementation retained for differential tests and compile-time
// A/B qualification. This is right-to-left binary exponentiation by p - 2.
__device__ __forceinline__ void inv_mod_p_binary(uint32_t r[8], const uint32_t a[8]) {
    uint32_t res[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) res[i] = (i == 0) ? 1 : 0;

    uint32_t base[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) base[i] = a[i];

    const uint32_t exp[8] = {
        0xFFFFFC2Du, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    };

    for (int i = 0; i < 8; ++i) {
        uint32_t w = exp[i];
        for (int b = 0; b < 32; ++b) {
            if (w & 1) {
                mul_mod_p(res, res, base);
            }
            sqr_mod_p(base, base);
            w >>= 1;
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = res[i];
}

// Apply N dependent field squarings in place. Long runs deliberately remain
// compact loops: fully expanding all 255 squarings produces excessive SASS and
// can trade arithmetic savings for instruction-cache pressure.
template <int N>
__device__ __forceinline__ void sqr_mod_p_n(uint32_t value[8]) {
#ifdef __CUDACC__
    #pragma unroll 1
#endif
    for (int i = 0; i < N; ++i) {
        sqr_mod_p(value, value);
    }
}

__device__ __forceinline__ void copy_field(uint32_t r[8], const uint32_t a[8]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = a[i];
}

// Fixed addition chain for a^(p-2), where
// p = 2^256 - 2^32 - 977. The exponent schedule is representation-independent
// and follows the secp256k1 sliding-window chain introduced by Peter Dettman:
// https://github.com/bitcoin-core/secp256k1/commit/f8ccd9befdb22824ef9a845a90e3db57c1307c11
//
// Operation count: 255 field squarings + 15 field multiplications, versus
// 256 squarings + 249 multiplications in inv_mod_p_binary.
__device__ __forceinline__ void inv_mod_p_addition_chain(
    uint32_t r[8], const uint32_t a[8]
) {
    // xn denotes a^(2^n - 1). Keep only values needed by later windows.
    uint32_t x2[8], x3[8], x22[8], x44[8], x88[8], t[8];

    // x2 = a^(2^2 - 1), x3 = a^(2^3 - 1)
    copy_field(x2, a);
    sqr_mod_p_n<1>(x2);
    mul_mod_p(x2, x2, a);

    copy_field(x3, x2);
    sqr_mod_p_n<1>(x3);
    mul_mod_p(x3, x3, a);

    // Build x6, x9, and x11 in the reusable temporary.
    copy_field(t, x3);
    sqr_mod_p_n<3>(t);
    mul_mod_p(t, t, x3);       // x6
    sqr_mod_p_n<3>(t);
    mul_mod_p(t, t, x3);       // x9
    sqr_mod_p_n<2>(t);
    mul_mod_p(t, t, x2);       // x11

    copy_field(x22, t);
    sqr_mod_p_n<11>(x22);
    mul_mod_p(x22, x22, t);

    copy_field(x44, x22);
    sqr_mod_p_n<22>(x44);
    mul_mod_p(x44, x44, x22);

    copy_field(x88, x44);
    sqr_mod_p_n<44>(x88);
    mul_mod_p(x88, x88, x44);

    // x176, x220, and x223 reuse t. x44 is retained for x220 and x3 for x223.
    copy_field(t, x88);
    sqr_mod_p_n<88>(t);
    mul_mod_p(t, t, x88);      // x176
    sqr_mod_p_n<44>(t);
    mul_mod_p(t, t, x44);      // x220
    sqr_mod_p_n<3>(t);
    mul_mod_p(t, t, x3);       // x223

    // Assemble p - 2 from the 223-, 22-, 2-, and 1-bit windows.
    sqr_mod_p_n<23>(t);
    mul_mod_p(t, t, x22);
    sqr_mod_p_n<5>(t);
    mul_mod_p(t, t, a);
    sqr_mod_p_n<3>(t);
    mul_mod_p(t, t, x2);
    sqr_mod_p_n<2>(t);
    mul_mod_p(r, t, a);
}

__device__ __forceinline__ void inv_mod_p(uint32_t r[8], const uint32_t a[8]) {
#if VADDRO_INV_ADDITION_CHAIN
    inv_mod_p_addition_chain(r, a);
#else
    inv_mod_p_binary(r, a);
#endif
}

// Full Affine Point Addition: r = p + q
__device__ __forceinline__ void point_add_affine(PointAffine &r, const PointAffine &p, const PointAffine &q) {
    uint32_t dy[8], dx[8], inv_dx[8], lambda[8];
    sub_mod_p(dy, q.y.d, p.y.d);
    sub_mod_p(dx, q.x.d, p.x.d);
    inv_mod_p(inv_dx, dx);
    mul_mod_p(lambda, dy, inv_dx);

    uint32_t lambda2[8], x3[8];
    sqr_mod_p(lambda2, lambda);
    sub_mod_p(x3, lambda2, p.x.d);
    sub_mod_p(x3, x3, q.x.d);

    uint32_t diff_x[8], y3[8], prod[8];
    sub_mod_p(diff_x, p.x.d, x3);
    mul_mod_p(prod, lambda, diff_x);
    sub_mod_p(y3, prod, p.y.d);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        r.x.d[i] = x3[i];
        r.y.d[i] = y3[i];
    }
}

// Full Affine Point Doubling: r = 2 * p
__device__ __forceinline__ void point_double_affine(PointAffine &r, const PointAffine &p) {
    uint32_t x2[8], num[8], den[8], inv_den[8], lambda[8];
    sqr_mod_p(x2, p.x.d);
    add_mod_p(num, x2, x2);
    add_mod_p(num, num, x2); // 3 * x^2
    add_mod_p(den, p.y.d, p.y.d); // 2 * y
    inv_mod_p(inv_den, den);
    mul_mod_p(lambda, num, inv_den);

    uint32_t lambda2[8], x3[8], two_x[8];
    sqr_mod_p(lambda2, lambda);
    add_mod_p(two_x, p.x.d, p.x.d);
    sub_mod_p(x3, lambda2, two_x);

    uint32_t dx[8], y3[8], prod[8];
    sub_mod_p(dx, p.x.d, x3);
    mul_mod_p(prod, lambda, dx);
    sub_mod_p(y3, prod, p.y.d);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        r.x.d[i] = x3[i];
        r.y.d[i] = y3[i];
    }
}

// Scalar Multiplication: r = scalar * p
__device__ __forceinline__ void scalar_mult_affine(PointAffine &r, const uint32_t scalar[8], const PointAffine &p) {
    PointAffine acc;
    PointAffine addend = p;
    bool is_first = true;

    for (int i = 0; i < 8; ++i) {
        uint32_t w = scalar[i];
        for (int b = 0; b < 32; ++b) {
            if (w & 1) {
                if (is_first) {
                    acc = addend;
                    is_first = false;
                } else {
                    point_add_affine(acc, acc, addend);
                }
            }
            point_double_affine(addend, addend);
            w >>= 1;
        }
    }

    if (!is_first) {
        r = acc;
    }
}

#endif // SECP256K1_PTX_CUH
