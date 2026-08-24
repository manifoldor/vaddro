#include <cstdint>
#include <iostream>
#include <string>

#include "search_shard.hpp"

int main() {
    constexpr uint32_t points = 32768u * 255u;
    vaddro::SearchShardRange ranges[4];
    uint64_t selected_high[4] = {};

    for (uint32_t index = 0; index < 4; ++index) {
        std::string error;
        if (!vaddro::calculate_search_shard(index, 4, points, ranges[index], error)) {
            std::cerr << error << '\n';
            return 1;
        }
        uint64_t seed[4] = {1, 2, 3, 0xFEDCBA9876543210ULL};
        vaddro::apply_search_shard(seed, ranges[index]);
        selected_high[index] = seed[3];

        if (seed[0] != 1 || seed[1] != 2 || seed[2] != 3 ||
            seed[3] < ranges[index].begin || seed[3] >= ranges[index].end) {
            std::cerr << "seed was not mapped into its shard\n";
            return 1;
        }
        const uint64_t maximum_high = seed[3] + points;
        if (maximum_high >= ranges[index].end) {
            std::cerr << "shard does not reserve lane and round-carry headroom\n";
            return 1;
        }
        if (index > 0 && ranges[index - 1].end != ranges[index].begin) {
            std::cerr << "shard ranges are not contiguous\n";
            return 1;
        }
    }

    if (ranges[0].begin != 0 || ranges[3].end != vaddro::SEARCH_SHARD_HIGH_SPACE) {
        std::cerr << "four shards do not cover the high-limb namespace\n";
        return 1;
    }
    for (uint32_t left = 0; left < 4; ++left) {
        for (uint32_t right = left + 1; right < 4; ++right) {
            if (selected_high[left] == selected_high[right]) {
                std::cerr << "same randomness selected the same high limb in two shards\n";
                return 1;
            }
        }
    }

    std::string error;
    vaddro::SearchShardRange invalid;
    if (vaddro::calculate_search_shard(0, 3, points, invalid, error) ||
        vaddro::calculate_search_shard(4, 4, points, invalid, error) ||
        vaddro::calculate_search_shard(0, 65536, UINT32_MAX, invalid, error)) {
        std::cerr << "invalid shard configuration was accepted\n";
        return 1;
    }

    std::cout << "Search shard partition test: OK\n";
    return 0;
}
