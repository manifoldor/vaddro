#pragma once

#include <cstdint>
#include <string>

namespace vaddro {

constexpr uint64_t SEARCH_SHARD_HIGH_SPACE = (1ULL << 48);
constexpr uint32_t SEARCH_SHARD_MAX_COUNT = 65536;

struct SearchShardRange {
    uint64_t begin = 0;
    uint64_t end = SEARCH_SHARD_HIGH_SPACE;
    uint64_t seed_slots = 0;
};

inline bool calculate_search_shard(
    uint32_t shard_index,
    uint32_t shard_count,
    uint32_t total_points,
    SearchShardRange &range,
    std::string &error
) {
    if (shard_count == 0 || shard_count > SEARCH_SHARD_MAX_COUNT ||
        (shard_count & (shard_count - 1)) != 0) {
        error = "shard-count must be a power of two from 1 through " +
                std::to_string(SEARCH_SHARD_MAX_COUNT);
        return false;
    }
    if (shard_index >= shard_count) {
        error = "shard-index must be smaller than shard-count";
        return false;
    }
    if (total_points == 0) {
        error = "search batch must contain at least one point";
        return false;
    }

    const uint64_t width = SEARCH_SHARD_HIGH_SPACE / shard_count;
    const uint64_t required_width = static_cast<uint64_t>(total_points) + 1ULL;
    if (width <= required_width) {
        error = "search batch is too large for the configured shard-count";
        return false;
    }

    range.begin = width * shard_index;
    range.end = range.begin + width;
    // The initial high limb may use any slot that leaves room for every lane
    // plus the single carry possible from the 64-bit round counter.
    range.seed_slots = width - total_points;
    return true;
}

inline void apply_search_shard(uint64_t seed[4], const SearchShardRange &range) {
    seed[3] = range.begin + (seed[3] % range.seed_slots);
}

}  // namespace vaddro
