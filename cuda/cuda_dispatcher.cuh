#ifndef CUDA_DISPATCHER_CUH
#define CUDA_DISPATCHER_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

#include "secp256k1_ptx.cuh"
#include "keccak_ptx.cuh"

#define PROFANITY_INVERSE_SIZE 255
#define MAX_RESULTS 512

#ifndef VADDRO_REUSE_GLV_X
#define VADDRO_REUSE_GLV_X 1
#endif

#if VADDRO_REUSE_GLV_X != 0 && VADDRO_REUSE_GLV_X != 1
#error "VADDRO_REUSE_GLV_X must be 0 or 1"
#endif

// Experimental two-stage search path. The qualified fused iterate kernel
// remains the production default until same-source RTX 5090 A/B testing shows
// that lower hash-kernel register pressure outweighs the basis-buffer traffic.
#ifndef VADDRO_SPLIT_VARIANT_HASH
#define VADDRO_SPLIT_VARIANT_HASH 0
#endif

#if VADDRO_SPLIT_VARIANT_HASH != 0 && VADDRO_SPLIT_VARIANT_HASH != 1
#error "VADDRO_SPLIT_VARIANT_HASH must be 0 or 1"
#endif

#if VADDRO_SPLIT_VARIANT_HASH && !VADDRO_REUSE_GLV_X
#error "VADDRO_SPLIT_VARIANT_HASH requires VADDRO_REUSE_GLV_X=1"
#endif

constexpr uint32_t VADDRO_VARIANT_COUNT = 6;
constexpr uint32_t VADDRO_VARIANT_TILE_POINTS = 32;
constexpr uint32_t VADDRO_VARIANT_BASIS_COORDS = 5;

enum VariantBasisCoordinate : uint32_t {
    VADDRO_BASIS_X = 0,
    VADDRO_BASIS_BETA_X = 1,
    VADDRO_BASIS_BETA2_X = 2,
    VADDRO_BASIS_Y = 3,
    VADDRO_BASIS_NEG_Y = 4
};

struct SearchTarget {
    int mode;               // 0: benchmark, 1: word-matching, 2: same-char-ends
    int required_score;     // fixed-pattern match threshold; not a stop policy
    int same_char_ends;     // length of same char ends
    uint32_t word_mask[5];  // 32-bit bitmasks for addr[0..4]
    uint32_t word_val[5];   // 32-bit expected values for addr[0..4]
    uint32_t prefix_filter_mask; // Quick reject mask for addr[0]
    uint32_t prefix_filter_val;  // Quick reject val for addr[0]
};

struct MatchResult {
    uint32_t found;
    uint32_t score;
    uint32_t variant;
    uint32_t address[5];    // 20 bytes (5 uint32_t in big-endian)
    uint32_t share_id;      // Global point index in this batch
    uint64_t round_found;   // Exact round when this match was found
};

__device__ __forceinline__ bool check_point_on_curve(const uint32_t px[8], const uint32_t py[8]);
__device__ __forceinline__ void transform_x_variant(
    uint32_t out_x[8], const uint32_t px[8], uint32_t x_variant
);
__device__ __forceinline__ void transform_point_variant(
    uint32_t out_x[8], uint32_t out_y[8],
    const uint32_t px[8], const uint32_t py[8], uint32_t variant
);

__device__ __forceinline__ bool words_equal(const uint32_t a[8], const uint32_t b[8]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// GPU cryptographic known-answer tests. Each bit in `status` represents a
// separately verified primitive so release tests can fail closed on real PTX.
// -----------------------------------------------------------------------------
__global__ void crypto_self_test_kernel(uint32_t *status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint32_t passed = 0;
    if (check_point_on_curve(SECP256K1_GX, SECP256K1_GY)) {
        passed |= 1u << 0;
    }

    uint32_t address[5];
    pubkey_to_address(SECP256K1_GX, SECP256K1_GY, address);
    const uint32_t expected_address[5] = {
        0x7E5F4552u, 0x091A6912u, 0x5D5DFCB7u, 0xB8C26590u, 0x29395BDFu
    };
    bool address_ok = true;
    #pragma unroll
    for (int i = 0; i < 5; ++i) address_ok &= address[i] == expected_address[i];
    if (address_ok) passed |= 1u << 1;

    const uint32_t a[8] = {
        0x87654321u, 0x0FEDCBA9u, 0xCDDEEFF0u, 0x899AABBCu,
        0x45566778u, 0x01122334u, 0x9ABCDEF0u, 0x12345678u
    };
    const uint32_t b[8] = {
        0x43322110u, 0x87766554u, 0x4B5A6978u, 0x0F1E2D3Cu,
        0x89ABCDEFu, 0x01234567u, 0x76543210u, 0xFEDCBA98u
    };
    const uint32_t expected_mul[8] = {
        0x9EBE9F32u, 0x966F2408u, 0xE5ABA8CCu, 0x2D1FED2Eu,
        0xC04F507Au, 0x10240DBAu, 0x6C817CA6u, 0xC36E7905u
    };
    const uint32_t expected_sqr[8] = {
        0xEF9B3759u, 0x95BF708Bu, 0xA5730313u, 0xF4F1A585u,
        0x69BEE43Bu, 0xD1442E90u, 0x6111D6F6u, 0x8A357A56u
    };
    const uint32_t expected_inv[8] = {
        0xA6B037E2u, 0x8BEC79A0u, 0xF1EE162Eu, 0xCA6FC4E0u,
        0x762B2A14u, 0x5607289Au, 0x9BCADC0Eu, 0xB6D844EBu
    };
    uint32_t value[8];
    mul_mod_p(value, a, b);
    if (words_equal(value, expected_mul)) passed |= 1u << 2;
    sqr_mod_p(value, a);
    if (words_equal(value, expected_sqr)) passed |= 1u << 3;

    // The active implementation must match the known answer. Independently
    // exercise both compile-time paths on boundary and nontrivial inputs so an
    // incorrect chain cannot be hidden by selecting the reference build.
    inv_mod_p(value, a);
    bool inverse_ok = words_equal(value, expected_inv);
    const uint32_t inverse_inputs[4][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {1, 0, 0, 0, 0, 0, 0, 0},
        {0xFFFFFC2Eu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
         0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu},
        {0x87654321u, 0x0FEDCBA9u, 0xCDDEEFF0u, 0x899AABBCu,
         0x45566778u, 0x01122334u, 0x9ABCDEF0u, 0x12345678u}
    };
    const uint32_t field_zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t field_one[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    for (int sample = 0; sample < 4; ++sample) {
        uint32_t binary_inv[8], chain_inv[8], product[8];
        inv_mod_p_binary(binary_inv, inverse_inputs[sample]);
        inv_mod_p_addition_chain(chain_inv, inverse_inputs[sample]);
        inverse_ok &= words_equal(binary_inv, chain_inv);
        if (sample == 0) {
            inverse_ok &= words_equal(chain_inv, field_zero);
        } else {
            mul_mod_p(product, inverse_inputs[sample], chain_inv);
            inverse_ok &= words_equal(product, field_one);
        }
    }
    if (inverse_ok) passed |= 1u << 4;

    const uint32_t expected_variant_addresses[6][5] = {
        {0x7E5F4552u, 0x091A6912u, 0x5D5DFCB7u, 0xB8C26590u, 0x29395BDFu},
        {0x1869DBF1u, 0x8B0786E3u, 0xB2D8D768u, 0x8FD5F6B9u, 0xC1D5AD17u},
        {0x6889E05Eu, 0xFEFDEEC6u, 0x24BA68BBu, 0x8D6343FAu, 0x1F8911C0u},
        {0x80C0DBF2u, 0x39224071u, 0xC59DD897u, 0x0AB9D542u, 0xE3414AB2u},
        {0xD8955DF1u, 0xD306FCFBu, 0xC4F570FEu, 0xF413D0DCu, 0x954894E8u},
        {0x1B7915F7u, 0x776594DEu, 0x59625CB9u, 0xE8B31506u, 0x335EE3FBu}
    };
    bool variants_ok = true;
    for (uint32_t variant = 0; variant < 6; ++variant) {
        uint32_t vx[8], vy[8];
        transform_point_variant(vx, vy, SECP256K1_GX, SECP256K1_GY, variant);
        variants_ok &= check_point_on_curve(vx, vy);
        uint32_t variant_address[5];
        pubkey_to_address(vx, vy, variant_address);
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            variants_ok &= variant_address[i] == expected_variant_addresses[variant][i];
        }
    }

    // Exercise the exact production schedule for both compile-time A/B modes.
    // Reuse-on pairs 0/3, 1/4, 2/5; reuse-off retains the original 0..5 order.
    uint32_t vx[8];
    for (uint32_t step = 0; step < 6; ++step) {
        uint32_t vy[8];
#if VADDRO_REUSE_GLV_X
        const uint32_t x_variant = step >> 1;
        const uint32_t sign_variant = step & 1u;
        const uint32_t variant = x_variant + sign_variant * 3u;
        if (sign_variant == 0) {
            transform_x_variant(vx, SECP256K1_GX, x_variant);
        }
        if (sign_variant == 0) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) vy[i] = SECP256K1_GY[i];
        } else {
            uint32_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            sub_mod_p(vy, zero, SECP256K1_GY);
        }
#else
        const uint32_t variant = step;
        transform_point_variant(vx, vy, SECP256K1_GX, SECP256K1_GY, variant);
#endif
        uint32_t scheduled_address[5];
        pubkey_to_address(vx, vy, scheduled_address);
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            variants_ok &= scheduled_address[i] == expected_variant_addresses[variant][i];
        }
    }
    if (variants_ok) passed |= 1u << 5;
    *status = passed;
}

// -----------------------------------------------------------------------------
// Kernel 1: Montgomery Batch Inversion Kernel (1 inverse per 255 items)
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(64, 4) profanity_inverse_kernel(
    const uint256_t * __restrict__ pDeltaX,
    uint256_t * __restrict__ pInverse,
    uint32_t num_inverses
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_inverses) return;

    uint32_t id = gid * PROFANITY_INVERSE_SIZE;

    uint32_t buffer[PROFANITY_INVERSE_SIZE][8];
    uint32_t buffer2[PROFANITY_INVERSE_SIZE][8];

    #pragma unroll
    for (int j = 0; j < 8; ++j) buffer[0][j] = pDeltaX[id].d[j];

    for (int i = 1; i < PROFANITY_INVERSE_SIZE; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) buffer2[i][j] = pDeltaX[id + i].d[j];
        mul_mod_p(buffer[i], buffer2[i], buffer[i - 1]);
    }

    uint32_t copy1[8];
    inv_mod_p(copy1, buffer[PROFANITY_INVERSE_SIZE - 1]);
    mul_mod_p(copy1, copy1, SECP256K1_NEG_DOUBLE_GY);

    uint32_t copy2[8];
    for (int i = PROFANITY_INVERSE_SIZE - 1; i > 0; --i) {
        mul_mod_p(copy2, copy1, buffer[i - 1]);
        mul_mod_p(copy1, copy1, buffer2[i]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) pInverse[id + i].d[j] = copy2[j];
    }

    #pragma unroll
    for (int j = 0; j < 8; ++j) pInverse[id].d[j] = copy1[j];
}

// -----------------------------------------------------------------------------
// Helper: Strict Secp256k1 Curve Integrity Validator
// -----------------------------------------------------------------------------
__device__ __forceinline__ bool check_point_on_curve(const uint32_t px[8], const uint32_t py[8]) {
    uint32_t y2[8], x2[8], x3[8], rhs[8], seven[8] = {7, 0, 0, 0, 0, 0, 0, 0};
    sqr_mod_p(y2, py);
    sqr_mod_p(x2, px);
    mul_mod_p(x3, x2, px);
    add_mod_p(rhs, x3, seven);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (y2[i] != rhs[i]) return false;
    }
    return true;
}

__device__ __forceinline__ void transform_point_variant(
    uint32_t out_x[8],
    uint32_t out_y[8],
    const uint32_t px[8],
    const uint32_t py[8],
    uint32_t variant
) {
    const uint32_t x_variant = variant >= 3 ? variant - 3 : variant;
    transform_x_variant(out_x, px, x_variant);

    uint32_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (int i = 0; i < 8; ++i) out_y[i] = py[i];

    if (variant >= 3) {
        sub_mod_p(out_y, zero, out_y);
    }
}

// Transform only the x-coordinate. The production search evaluates positive
// and negative y variants as a pair so each expensive GLV field multiplication
// can be shared by both points.
__device__ __forceinline__ void transform_x_variant(
    uint32_t out_x[8], const uint32_t px[8], uint32_t x_variant
) {
    if (x_variant == 1) {
        mul_mod_p(out_x, px, SECP256K1_BETA);
    } else if (x_variant == 2) {
        mul_mod_p(out_x, px, SECP256K1_BETA2);
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) out_x[i] = px[i];
    }
}

__device__ __forceinline__ uint32_t address_nibble(const uint32_t addr[5], int index) {
    int word_idx = index / 8;
    int nibble_idx = index % 8;
    return (addr[word_idx] >> (28 - 4 * nibble_idx)) & 0xFu;
}

#if VADDRO_SPLIT_VARIANT_HASH
// Advance one recurrence lane and materialize the five coordinates shared by
// the six GLV/negation variants. Each value is stored in its own SoA array so
// the consumer loader warps can use aligned uint4 transactions.
__global__ void __launch_bounds__(256, 1) profanity_point_basis_kernel(
    uint256_t * __restrict__ pDeltaX,
    const uint256_t * __restrict__ pInverse,
    uint256_t * __restrict__ pPrevLambda,
    uint256_t * __restrict__ pBasisX,
    uint256_t * __restrict__ pBasisBetaX,
    uint256_t * __restrict__ pBasisBeta2X,
    uint256_t * __restrict__ pBasisY,
    uint256_t * __restrict__ pBasisNegY,
    uint32_t total_points
) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_points) return;

    uint32_t dX[8], inv[8], lambda[8];
    load_u256(dX, &pDeltaX[gid]);
    load_u256(inv, &pInverse[gid]);
    load_u256(lambda, &pPrevLambda[gid]);

    uint32_t new_lambda[8];
    sub_mod_p(new_lambda, inv, lambda);

    uint32_t tmp[8];
    sqr_mod_p(tmp, new_lambda);

    uint32_t new_dX[8];
    sub_mod_p(new_dX, dX, tmp);
    sub_mod_p(new_dX, SECP256K1_TRIPLE_NEG_GX, new_dX);
    store_u256(&pDeltaX[gid], new_dX);
    store_u256(&pPrevLambda[gid], new_lambda);

    uint32_t term[8];
    mul_mod_p(term, new_lambda, new_dX);

    uint32_t ry[8], rx[8];
    sub_mod_p(ry, SECP256K1_NEG_GY, term);
    sub_mod_p(rx, new_dX, SECP256K1_NEG_GX);

    store_u256(&pBasisX[gid], rx);
    store_u256(&pBasisY[gid], ry);

    // Keep only one transformed coordinate live at a time. This makes the
    // register-lifetime reduction explicit instead of relying on NVCC to
    // coalesce three independent 256-bit temporaries.
    const uint32_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    sub_mod_p(term, zero, ry);
    store_u256(&pBasisNegY[gid], term);
    mul_mod_p(term, rx, SECP256K1_BETA);
    store_u256(&pBasisBetaX[gid], term);
    mul_mod_p(term, rx, SECP256K1_BETA2);
    store_u256(&pBasisBeta2X[gid], term);
}

// One block covers 32 points and all six variants. threadIdx.y selects a
// variant, so every consumer warp follows one uniform X/Y selection path.
// The first five warps each stage one unique coordinate in parallel using a
// bank-friendly [coordinate][limb][lane] layout.
__global__ void __launch_bounds__(192, 4) profanity_variant_hash_kernel(
    const uint256_t * __restrict__ pBasisX,
    const uint256_t * __restrict__ pBasisBetaX,
    const uint256_t * __restrict__ pBasisBeta2X,
    const uint256_t * __restrict__ pBasisY,
    const uint256_t * __restrict__ pBasisNegY,
    SearchTarget target,
    MatchResult *results,
    uint32_t *result_count,
    uint32_t total_points,
    uint64_t current_round
) {
    __shared__ uint32_t basis[VADDRO_VARIANT_BASIS_COORDS][8][VADDRO_VARIANT_TILE_POINTS];

    const uint32_t lane = threadIdx.x;
    const uint32_t variant = threadIdx.y;
    const uint32_t gid = blockIdx.x * VADDRO_VARIANT_TILE_POINTS + lane;

    if (variant < VADDRO_VARIANT_BASIS_COORDS) {
        uint32_t value[8] = {};
        if (gid < total_points) {
            if (variant == VADDRO_BASIS_X) {
                load_u256(value, &pBasisX[gid]);
            } else if (variant == VADDRO_BASIS_BETA_X) {
                load_u256(value, &pBasisBetaX[gid]);
            } else if (variant == VADDRO_BASIS_BETA2_X) {
                load_u256(value, &pBasisBeta2X[gid]);
            } else if (variant == VADDRO_BASIS_Y) {
                load_u256(value, &pBasisY[gid]);
            } else {
                load_u256(value, &pBasisNegY[gid]);
            }
        }
        #pragma unroll
        for (int limb = 0; limb < 8; ++limb) basis[variant][limb][lane] = value[limb];
    }
    __syncthreads();

    if (gid >= total_points || variant >= VADDRO_VARIANT_COUNT) return;

    const uint32_t x_coord = variant < 3 ? variant : variant - 3;
    const uint32_t y_coord = variant < 3 ? VADDRO_BASIS_Y : VADDRO_BASIS_NEG_Y;
    uint32_t vx[8], vy[8];
    #pragma unroll
    for (int limb = 0; limb < 8; ++limb) {
        vx[limb] = basis[x_coord][limb][lane];
        vy[limb] = basis[y_coord][limb][lane];
    }

    uint32_t addr[5];
    pubkey_to_address(vx, vy, addr);

    if (target.mode == 1) {
        const bool has_prefix_filter = target.prefix_filter_mask != 0u;
        const bool passed_prefix_filter =
            !has_prefix_filter ||
            ((addr[0] & target.prefix_filter_mask) == target.prefix_filter_val);

        if (passed_prefix_filter) {
            const bool full_match =
                ((addr[0] & target.word_mask[0]) == target.word_val[0]) &&
                ((addr[4] & target.word_mask[4]) == target.word_val[4]) &&
                ((addr[1] & target.word_mask[1]) == target.word_val[1]) &&
                ((addr[2] & target.word_mask[2]) == target.word_val[2]) &&
                ((addr[3] & target.word_mask[3]) == target.word_val[3]);

            if (full_match && check_point_on_curve(vx, vy)) {
                const uint32_t slot = atomicAdd(result_count, 1);
                if (slot < MAX_RESULTS) {
                    results[slot].found = 1;
                    results[slot].score = target.required_score;
                    results[slot].variant = variant;
                    #pragma unroll
                    for (int i = 0; i < 5; ++i) results[slot].address[i] = addr[i];
                    results[slot].share_id = gid;
                    results[slot].round_found = current_round;
                }
            }
        }
    } else if (target.mode == 2) {
        const int n = target.same_char_ends;
        if (n >= 1 && n <= 8) {
            const uint32_t c = address_nibble(addr, 0);
            bool ends_match = true;
            for (int k = 0; k < n; ++k) {
                if (address_nibble(addr, k) != c ||
                    address_nibble(addr, 39 - k) != c) {
                    ends_match = false;
                    break;
                }
            }
            if (ends_match && check_point_on_curve(vx, vy)) {
                const uint32_t slot = atomicAdd(result_count, 1);
                if (slot < MAX_RESULTS) {
                    results[slot].found = 1;
                    results[slot].score = static_cast<uint32_t>(n * 2);
                    results[slot].variant = variant;
                    #pragma unroll
                    for (int i = 0; i < 5; ++i) results[slot].address[i] = addr[i];
                    results[slot].share_id = gid;
                    results[slot].round_found = current_round;
                }
            }
        }
    }
}

// Populate real global basis buffers for the production hash-kernel KAT.
__global__ void split_variant_self_test_prepare_kernel(
    uint256_t *basis_x,
    uint256_t *basis_beta_x,
    uint256_t *basis_beta2_x,
    uint256_t *basis_y,
    uint256_t *basis_neg_y
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t x[8], beta_x[8], beta2_x[8], y[8], neg_y[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        x[i] = SECP256K1_GX[i];
        y[i] = SECP256K1_GY[i];
    }
    mul_mod_p(beta_x, x, SECP256K1_BETA);
    mul_mod_p(beta2_x, x, SECP256K1_BETA2);
    const uint32_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    sub_mod_p(neg_y, zero, y);
    store_u256(basis_x, x);
    store_u256(basis_beta_x, beta_x);
    store_u256(basis_beta2_x, beta2_x);
    store_u256(basis_y, y);
    store_u256(basis_neg_y, neg_y);
}

#endif

// -----------------------------------------------------------------------------
// Kernel 2: Point Iteration & LOP3 Keccak Address Matching Kernel
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 1) profanity_iterate_kernel(
    uint256_t * __restrict__ pDeltaX,
    const uint256_t * __restrict__ pInverse,
    uint256_t * __restrict__ pPrevLambda,
    SearchTarget target,
    MatchResult *results,
    uint32_t *result_count,
    uint32_t total_points,
    uint64_t current_round
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_points) return;

    uint32_t dX[8];
    uint32_t inv[8];
    uint32_t lambda[8];

    load_u256(dX, &pDeltaX[gid]);
    load_u256(inv, &pInverse[gid]);
    load_u256(lambda, &pPrevLambda[gid]);

    // Step 1: Compute new_lambda = inv - lambda (lambda_next)
    uint32_t new_lambda[8];
    sub_mod_p(new_lambda, inv, lambda);

    // Step 2: Compute tmp = new_lambda * new_lambda (SQR 1)
    uint32_t tmp[8];
    sqr_mod_p(tmp, new_lambda);

    // Step 3: Compute new_dX = -3*G_x - (dX - tmp)
    uint32_t new_dX[8];
    sub_mod_p(new_dX, dX, tmp);
    sub_mod_p(new_dX, SECP256K1_TRIPLE_NEG_GX, new_dX);

    store_u256(&pDeltaX[gid], new_dX);
    store_u256(&pPrevLambda[gid], new_lambda);

    // Step 4: Compute (Rx, Ry) using new_lambda and new_dX (MUL 2)
    uint32_t term[8];
    mul_mod_p(term, new_lambda, new_dX);

    uint32_t Ry[8];
    sub_mod_p(Ry, SECP256K1_NEG_GY, term);

    uint32_t Rx[8];
    sub_mod_p(Rx, new_dX, SECP256K1_NEG_GX);

    // Step 5: PTX LOP3 Accelerated Keccak-256 Hash -> EVM Address.
    // Search all six GLV/negation variants for the generated point.
    if (target.mode != 0) {
        // Pair the positive and negative y variants for each x mapping. With
        // VADDRO_REUSE_GLV_X enabled, beta*x and beta^2*x are each computed once
        // instead of being recomputed for the corresponding negative point.
        uint32_t vx[8];
        #pragma unroll
        for (uint32_t step = 0; step < 6; ++step) {
            uint32_t vy[8];
#if VADDRO_REUSE_GLV_X
            const uint32_t x_variant = step >> 1;
            const uint32_t sign_variant = step & 1u;
            const uint32_t variant = x_variant + sign_variant * 3u;
            if (sign_variant == 0) {
                transform_x_variant(vx, Rx, x_variant);
            }
            if (sign_variant == 0) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) vy[i] = Ry[i];
            } else {
                uint32_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                sub_mod_p(vy, zero, Ry);
            }
#else
            const uint32_t variant = step;
            transform_point_variant(vx, vy, Rx, Ry, variant);
#endif

            uint32_t addr[5];
            pubkey_to_address(vx, vy, addr);

            if (target.mode == 1) {
                // Word Pattern Matching
                const bool has_prefix_filter = target.prefix_filter_mask != 0u;
                const bool passed_prefix_filter =
                    !has_prefix_filter ||
                    ((addr[0] & target.prefix_filter_mask) == target.prefix_filter_val);

                if (passed_prefix_filter) {
                    bool full_match = ((addr[0] & target.word_mask[0]) == target.word_val[0]) &&
                                      ((addr[4] & target.word_mask[4]) == target.word_val[4]) &&
                                      ((addr[1] & target.word_mask[1]) == target.word_val[1]) &&
                                      ((addr[2] & target.word_mask[2]) == target.word_val[2]) &&
                                      ((addr[3] & target.word_mask[3]) == target.word_val[3]);

                    if (full_match && check_point_on_curve(vx, vy)) {
                        uint32_t slot = atomicAdd(result_count, 1);
                        if (slot < MAX_RESULTS) {
                            results[slot].found = 1;
                            results[slot].score = target.required_score;
                            results[slot].variant = variant;
                            #pragma unroll
                            for (int i = 0; i < 5; ++i) results[slot].address[i] = addr[i];
                            results[slot].share_id = gid;
                            results[slot].round_found = current_round;
                        }
                    }
                }
            } else if (target.mode == 2) {
                int n = target.same_char_ends;
                if (n >= 1 && n <= 8) {
                    uint32_t c = address_nibble(addr, 0);
                    bool ends_match = true;
                    for (int k = 0; k < n; ++k) {
                        if (address_nibble(addr, k) != c ||
                            address_nibble(addr, 39 - k) != c) {
                            ends_match = false;
                            break;
                        }
                    }
                    if (ends_match && check_point_on_curve(vx, vy)) {
                        uint32_t slot = atomicAdd(result_count, 1);
                        if (slot < MAX_RESULTS) {
                            results[slot].found = 1;
                            results[slot].score = (uint32_t)(n * 2);
                            results[slot].variant = variant;
                            #pragma unroll
                            for (int i = 0; i < 5; ++i) results[slot].address[i] = addr[i];
                            results[slot].share_id = gid;
                            results[slot].round_found = current_round;
                        }
                    }
                }
            }
        }
    }
}

// Reconstruct the current point represented by one recurrence lane without
// advancing it. The host compares this address with an independently rebuilt
// scalar path as a low-overhead long-running canary.
__global__ void state_address_kernel(
    const uint256_t * __restrict__ pDeltaX,
    const uint256_t * __restrict__ pPrevLambda,
    uint32_t gid,
    uint32_t *address
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t d_x[8], lambda[8], term[8], rx[8], ry[8];
    load_u256(d_x, &pDeltaX[gid]);
    load_u256(lambda, &pPrevLambda[gid]);
    mul_mod_p(term, lambda, d_x);
    sub_mod_p(ry, SECP256K1_NEG_GY, term);
    sub_mod_p(rx, d_x, SECP256K1_NEG_GX);
    pubkey_to_address(rx, ry, address);
}

// -----------------------------------------------------------------------------
// Kernel 3: Point Initialization using Precomputed Secp256k1 Base Point Table
// -----------------------------------------------------------------------------
__device__ void profanity_init_seed_cuda(
    const PointAffine * __restrict__ precomp,
    PointAffine &p,
    bool &is_first,
    uint32_t precomp_offset,
    uint64_t seed_val
) {
    for (int i = 0; i < 8; ++i) {
        uint32_t shift = i * 8;
        uint32_t byte_val = (seed_val >> shift) & 0xFF;
        if (byte_val != 0) {
            PointAffine o = precomp[precomp_offset + i * 255 + byte_val - 1];
            if (is_first) {
                p = o;
                is_first = false;
            } else {
                point_add_affine(p, p, o);
            }
        }
    }
}

__global__ void profanity_init_kernel(
    const PointAffine * __restrict__ precomp,
    PointAffine base_point,
    uint64_t seed0,
    uint64_t seed1,
    uint64_t seed2,
    uint64_t seed3,
    uint256_t *pDeltaX,
    uint256_t *pPrevLambda,
    uint32_t total_points
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_points) return;

    PointAffine p_random;
    bool is_first = true;

    profanity_init_seed_cuda(precomp, p_random, is_first, 8 * 255 * 0, seed0);
    profanity_init_seed_cuda(precomp, p_random, is_first, 8 * 255 * 1, seed1);
    profanity_init_seed_cuda(precomp, p_random, is_first, 8 * 255 * 2, seed2);
    profanity_init_seed_cuda(precomp, p_random, is_first, 8 * 255 * 3, seed3 + (uint64_t)gid);

    PointAffine p_cur;
    point_add_affine(p_cur, base_point, p_random);

    PointAffine p_next;
    point_add_affine(p_next, p_cur, precomp[0]);

    // pDeltaX[gid] = p_next.x - G.x
    uint32_t dx[8];
    sub_mod_p(dx, p_next.x.d, SECP256K1_GX);

    // pPrevLambda[gid] = (p_cur.y - G.y) / (p_cur.x - G.x)
    uint32_t dy[8], denom[8], inv_denom[8], lambda[8];
    sub_mod_p(dy, p_cur.y.d, SECP256K1_GY);
    sub_mod_p(denom, p_cur.x.d, SECP256K1_GX);
    inv_mod_p(inv_denom, denom);
    mul_mod_p(lambda, dy, inv_denom);

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        pDeltaX[gid].d[j] = dx[j];
        pPrevLambda[gid].d[j] = lambda[j];
    }
}

// -----------------------------------------------------------------------------
// Kernel 4: Zero-Defect Instant Verification Kernel
// Derives the exact point and EVM address directly from (s0, s1, s2, s3)
// -----------------------------------------------------------------------------
__global__ void verify_scalar_kernel(
    const PointAffine * __restrict__ d_precomp,
    PointAffine base_point,
    uint64_t s0,
    uint64_t s1,
    uint64_t s2,
    uint64_t s3,
    uint32_t variant,
    uint32_t *d_verified_addr
) {
    PointAffine p_cur;
    bool is_first = true;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t b0 = (s0 >> (i * 8)) & 0xFF;
        if (b0) { PointAffine o = d_precomp[8 * 255 * 0 + i * 255 + b0 - 1]; if (is_first) { p_cur = o; is_first = false; } else point_add_affine(p_cur, p_cur, o); }
        uint32_t b1 = (s1 >> (i * 8)) & 0xFF;
        if (b1) { PointAffine o = d_precomp[8 * 255 * 1 + i * 255 + b1 - 1]; if (is_first) { p_cur = o; is_first = false; } else point_add_affine(p_cur, p_cur, o); }
        uint32_t b2 = (s2 >> (i * 8)) & 0xFF;
        if (b2) { PointAffine o = d_precomp[8 * 255 * 2 + i * 255 + b2 - 1]; if (is_first) { p_cur = o; is_first = false; } else point_add_affine(p_cur, p_cur, o); }
        uint32_t b3 = (s3 >> (i * 8)) & 0xFF;
        if (b3) { PointAffine o = d_precomp[8 * 255 * 3 + i * 255 + b3 - 1]; if (is_first) { p_cur = o; is_first = false; } else point_add_affine(p_cur, p_cur, o); }
    }
    point_add_affine(p_cur, base_point, p_cur);
    uint32_t vx[8], vy[8];
    transform_point_variant(vx, vy, p_cur.x.d, p_cur.y.d, variant);
    pubkey_to_address(vx, vy, d_verified_addr);
}

// -----------------------------------------------------------------------------
// Kernel 5: Validate the split public key before starting the search loop.
// -----------------------------------------------------------------------------
__global__ void verify_base_point_kernel(PointAffine base_point, uint32_t *valid) {
    const bool in_field = !gte256(base_point.x.d, SECP256K1_P) &&
                          !gte256(base_point.y.d, SECP256K1_P);
    *valid = (in_field && check_point_on_curve(base_point.x.d, base_point.y.d)) ? 1u : 0u;
}

#endif // CUDA_DISPATCHER_CUH
