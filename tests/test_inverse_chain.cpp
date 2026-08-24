#include <cstdint>
#include <cstdio>

// Minimal CUDA-vector compatibility for exercising the header's portable
// arithmetic path with a regular C++ compiler.
#define __align__(n) alignas(n)
struct uint4 { uint32_t x, y, z, w; };
inline uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    return uint4{x, y, z, w};
}

#include "../cuda/secp256k1_ptx.cuh"

static bool equal_words(const uint32_t a[8], const uint32_t b[8]) {
    for (int i = 0; i < 8; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

static bool test_inverse(const uint32_t value[8], bool is_zero) {
    const uint32_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t one[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    uint32_t binary[8], chain[8], selected[8], product[8];
    inv_mod_p_binary(binary, value);
    inv_mod_p_addition_chain(chain, value);
    inv_mod_p(selected, value);
    if (!equal_words(binary, chain) || !equal_words(selected, chain)) return false;
    if (is_zero) return equal_words(chain, zero);
    mul_mod_p(product, value, chain);
    return equal_words(product, one);
}

static uint64_t xorshift64(uint64_t &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

int main() {
    const uint32_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t one[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t p_minus_one[8] = {
        0xFFFFFC2Eu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    };
    const uint32_t known_input[8] = {
        0x87654321u, 0x0FEDCBA9u, 0xCDDEEFF0u, 0x899AABBCu,
        0x45566778u, 0x01122334u, 0x9ABCDEF0u, 0x12345678u
    };
    const uint32_t known_inverse[8] = {
        0xA6B037E2u, 0x8BEC79A0u, 0xF1EE162Eu, 0xCA6FC4E0u,
        0x762B2A14u, 0x5607289Au, 0x9BCADC0Eu, 0xB6D844EBu
    };

    if (!test_inverse(zero, true) ||
        !test_inverse(one, false) ||
        !test_inverse(p_minus_one, false) ||
        !test_inverse(known_input, false)) {
        std::fprintf(stderr, "Inverse addition-chain boundary test failed.\n");
        return 1;
    }

    uint32_t known_result[8];
    inv_mod_p_addition_chain(known_result, known_input);
    if (!equal_words(known_result, known_inverse)) {
        std::fprintf(stderr, "Inverse addition-chain known answer failed.\n");
        return 1;
    }

    uint64_t random_state = 0xD1B54A32D192ED03ULL;
    for (int sample = 0; sample < 256; ++sample) {
        uint32_t value[8];
        for (int i = 0; i < 8; ++i) {
            value[i] = static_cast<uint32_t>(xorshift64(random_state));
        }
        // Keep the generated value strictly below p without introducing a
        // dependency on the reduction implementation under test.
        value[7] &= 0x7FFFFFFFu;
        value[0] |= 1u;
        if (!test_inverse(value, false)) {
            std::fprintf(stderr, "Inverse addition-chain differential test failed at sample %d.\n", sample);
            return 1;
        }
    }

    std::printf("Inverse addition-chain differential test: 260 values OK (selected=%s)\n",
                VADDRO_INV_ADDITION_CHAIN ? "chain" : "binary");
    return 0;
}
