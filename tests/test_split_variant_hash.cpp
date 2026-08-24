#include <cstdint>
#include <cstdio>

// Minimal CUDA-vector compatibility for exercising the headers' portable
// arithmetic and Keccak paths with a regular C++ compiler.
#define __align__(n) alignas(n)
struct uint4 { uint32_t x, y, z, w; };
inline uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    return uint4{x, y, z, w};
}

#include "../cuda/secp256k1_ptx.cuh"
#include "../cuda/keccak_ptx.cuh"

static bool equal_words(const uint32_t *a, const uint32_t *b, int count) {
    for (int i = 0; i < count; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

static bool on_curve(const uint32_t x[8], const uint32_t y[8]) {
    uint32_t y2[8], x2[8], x3[8], rhs[8];
    const uint32_t seven[8] = {7, 0, 0, 0, 0, 0, 0, 0};
    sqr_mod_p(y2, y);
    sqr_mod_p(x2, x);
    mul_mod_p(x3, x2, x);
    add_mod_p(rhs, x3, seven);
    return equal_words(y2, rhs, 8);
}

int main() {
    uint32_t basis_x[3][8] = {};
    uint32_t basis_y[2][8] = {};
    for (int i = 0; i < 8; ++i) {
        basis_x[0][i] = SECP256K1_GX[i];
        basis_y[0][i] = SECP256K1_GY[i];
    }
    mul_mod_p(basis_x[1], basis_x[0], SECP256K1_BETA);
    mul_mod_p(basis_x[2], basis_x[0], SECP256K1_BETA2);
    const uint32_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    sub_mod_p(basis_y[1], zero, basis_y[0]);

    const uint32_t expected[6][5] = {
        {0x7E5F4552u, 0x091A6912u, 0x5D5DFCB7u, 0xB8C26590u, 0x29395BDFu},
        {0x1869DBF1u, 0x8B0786E3u, 0xB2D8D768u, 0x8FD5F6B9u, 0xC1D5AD17u},
        {0x6889E05Eu, 0xFEFDEEC6u, 0x24BA68BBu, 0x8D6343FAu, 0x1F8911C0u},
        {0x80C0DBF2u, 0x39224071u, 0xC59DD897u, 0x0AB9D542u, 0xE3414AB2u},
        {0xD8955DF1u, 0xD306FCFBu, 0xC4F570FEu, 0xF413D0DCu, 0x954894E8u},
        {0x1B7915F7u, 0x776594DEu, 0x59625CB9u, 0xE8B31506u, 0x335EE3FBu}
    };

    for (uint32_t variant = 0; variant < 6; ++variant) {
        const uint32_t x_index = variant < 3 ? variant : variant - 3;
        const uint32_t y_index = variant < 3 ? 0 : 1;
        if (!on_curve(basis_x[x_index], basis_y[y_index])) {
            std::fprintf(stderr, "split variant %u is not on secp256k1\n", variant);
            return 1;
        }
        uint32_t address[5];
        pubkey_to_address(basis_x[x_index], basis_y[y_index], address);
        if (!equal_words(address, expected[variant], 5)) {
            std::fprintf(stderr, "split variant %u address mismatch\n", variant);
            return 1;
        }
    }

    std::puts("Split variant basis/address test: 6 variants OK");
    return 0;
}
