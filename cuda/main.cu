#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>
#include <ctime>
#include <utility>
#include <unistd.h>
#include <cuda_runtime.h>

#include "secp256k1_ptx.cuh"
#include "keccak_ptx.cuh"
#include "cuda_dispatcher.cuh"
#include "precomp.hpp"
#include "result_io.hpp"
#include "search_shard.hpp"

#ifndef VADDRO_VERSION
#define VADDRO_VERSION "1.0.0"
#endif

static volatile sig_atomic_t g_running = 1;

void signal_handler(int) {
    g_running = 0;
}

struct AuditContext {
    std::string worker_id;
    std::string run_id;
    std::string build_commit;
    std::string gpu_name;
    std::string driver_version;
    std::string mode;
    int gpu_index = 0;
    uint32_t shard_index = 0;
    uint32_t shard_count = 1;
};

static std::string utc_now_iso8601() {
    std::time_t now = std::time(nullptr);
    std::tm utc_tm{};
    gmtime_r(&now, &utc_tm);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc_tm);
    return std::string(buffer);
}

static std::string json_escape(const std::string &value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec;
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

static bool append_json_event(
    const std::string &path,
    const std::string &event,
    const AuditContext &context,
    const std::vector<std::pair<std::string, std::string>> &text_fields = {},
    const std::vector<std::pair<std::string, std::string>> &raw_fields = {}
) {
    if (path.empty()) return true;
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) return false;
    out << "{\"ts\":\"" << utc_now_iso8601() << "\""
        << ",\"event\":\"" << json_escape(event) << "\""
        << ",\"worker_id\":\"" << json_escape(context.worker_id) << "\""
        << ",\"run_id\":\"" << json_escape(context.run_id) << "\""
        << ",\"version\":\"" << VADDRO_VERSION << "\""
        << ",\"build_commit\":\"" << json_escape(context.build_commit) << "\""
        << ",\"gpu_index\":" << context.gpu_index
        << ",\"shard_index\":" << context.shard_index
        << ",\"shard_count\":" << context.shard_count
        << ",\"gpu_name\":\"" << json_escape(context.gpu_name) << "\""
        << ",\"driver_version\":\"" << json_escape(context.driver_version) << "\""
        << ",\"mode\":\"" << json_escape(context.mode) << "\"";
    for (const auto &field : text_fields) {
        out << ",\"" << json_escape(field.first) << "\":\""
            << json_escape(field.second) << "\"";
    }
    for (const auto &field : raw_fields) {
        out << ",\"" << json_escape(field.first) << "\":" << field.second;
    }
    out << "}\n";
    out.flush();
    return out.good();
}

static std::string format_cuda_version(int version) {
    if (version <= 0) return "unknown";
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

static uint8_t parse_hex_char(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

static bool is_hex_string(const std::string &text) {
    if (text.empty()) return false;
    for (char c : text) {
        if (!((c >= '0' && c <= '9') ||
              (c >= 'a' && c <= 'f') ||
              (c >= 'A' && c <= 'F'))) {
            return false;
        }
    }
    return true;
}

static bool is_safe_audit_label(const std::string &text, size_t max_length) {
    if (text.empty() || text.length() > max_length) return false;
    for (unsigned char c : text) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-')) {
            return false;
        }
    }
    return true;
}

static bool parse_u32_arg(const std::string &text, uint32_t &value) {
    if (text.empty()) return false;
    errno = 0;
    char *end = nullptr;
    unsigned long parsed = std::strtoul(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0' || parsed > UINT32_MAX) {
        return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

static bool parse_int_arg(const std::string &text, int &value) {
    if (text.empty()) return false;
    errno = 0;
    char *end = nullptr;
    long parsed = std::strtol(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0' ||
        parsed < INT_MIN || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

static void cuda_check(cudaError_t error, const char *what) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << what << ": "
                  << cudaGetErrorString(error) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static void parse_hex_256(const std::string &hex, uint32_t words[8]) {
    for (int i = 0; i < 8; ++i) {
        std::string byte_chunk = hex.substr((7 - i) * 8, 8);
        uint32_t val = 0;
        for (char c : byte_chunk) {
            val = (val << 4) | parse_hex_char(c);
        }
        words[i] = val;
    }
}

static std::string format_hex_32(uint32_t val) {
    std::ostringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(8) << val;
    return ss.str();
}

static std::string format_address(const uint32_t addr[5]) {
    std::string res;
    for (int i = 0; i < 5; ++i) {
        res += format_hex_32(addr[i]);
    }
    return res;
}

static bool write_text_atomic(
    const std::string &path, const std::string &content, std::string &error
) {
    static uint64_t sequence = 0;
    const std::string temp_path = path + ".tmp." + std::to_string((long long)getpid()) +
                                  "." + std::to_string(++sequence);
    {
        std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            error = "could not open temporary result file '" + temp_path + "'";
            return false;
        }
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
        out.flush();
        if (!out.good()) {
            error = "failed while writing temporary result file '" + temp_path + "'";
            out.close();
            std::remove(temp_path.c_str());
            return false;
        }
    }
    if (std::rename(temp_path.c_str(), path.c_str()) != 0) {
        error = "could not atomically publish '" + path + "': " + std::strerror(errno);
        std::remove(temp_path.c_str());
        return false;
    }
    return true;
}

static bool fill_worker_seed(
    uint64_t seed[4], const vaddro::SearchShardRange &shard, std::string &error
) {
    std::ifstream random_source("/dev/urandom", std::ios::binary);
    if (!random_source.is_open()) {
        error = "could not open /dev/urandom";
        return false;
    }
    random_source.read(reinterpret_cast<char *>(seed), sizeof(uint64_t) * 4);
    if (random_source.gcount() != static_cast<std::streamsize>(sizeof(uint64_t) * 4)) {
        error = "could not read 256 bits from /dev/urandom";
        return false;
    }
    vaddro::apply_search_shard(seed, shard);
    return true;
}

static std::string format_time_hms(double seconds) {
    long s = static_cast<long>(seconds);
    long h = s / 3600;
    long m = (s % 3600) / 60;
    long sec = s % 60;
    char buf[32];
    snprintf(buf, sizeof(buf), "%02ld:%02ld:%02ld", h, m, sec);
    return std::string(buf);
}

static std::string format_key_count(uint64_t keys) {
    char buf[64];
    if (keys >= 1000000000000ULL) {
        snprintf(buf, sizeof(buf), "%.2f Trillion", (double)keys / 1e12);
    } else if (keys >= 1000000000ULL) {
        snprintf(buf, sizeof(buf), "%.2f Billion", (double)keys / 1e9);
    } else if (keys >= 1000000ULL) {
        snprintf(buf, sizeof(buf), "%.2f Million", (double)keys / 1e6);
    } else {
        snprintf(buf, sizeof(buf), "%llu", (unsigned long long)keys);
    }
    return std::string(buf);
}

static std::string format_hashrate(double mhs) {
    char buf[64];
    if (mhs >= 1000.0) {
        snprintf(buf, sizeof(buf), "%.2f GH/s", mhs / 1000.0);
    } else {
        snprintf(buf, sizeof(buf), "%.2f MH/s", mhs);
    }
    return std::string(buf);
}

static std::string format_hit_progress(uint64_t hits, uint32_t max_hits) {
    if (max_hits == 0) {
        return std::to_string(hits) + "/unlimited";
    }
    return std::to_string(hits) + "/" + std::to_string(max_hits);
}

static std::string pad_exact(const std::string &s, size_t target_width) {
    if (s.length() >= target_width) return s.substr(0, target_width);
    return s + std::string(target_width - s.length(), ' ');
}

static std::string abbreviate_middle(const std::string &value, size_t max_length) {
    if (value.length() <= max_length) return value;
    if (max_length <= 3) return value.substr(0, max_length);
    const size_t head_length = (max_length - 3) / 2;
    const size_t tail_length = max_length - 3 - head_length;
    return value.substr(0, head_length) + "..." +
           value.substr(value.length() - tail_length);
}

static bool should_clear_interactive_screen() {
    if (isatty(fileno(stdout)) == 0) return false;
    const char *term = std::getenv("TERM");
    return term != nullptr && term[0] != '\0' && std::strcmp(term, "dumb") != 0;
}

static int count_fixed_pattern(const std::string &pattern) {
    int count = 0;
    for (char c : pattern) {
        if (c != 'X' && c != 'x' && c != '.') {
            ++count;
        }
    }
    return count;
}

struct ScalarWords64 {
    uint64_t s0;
    uint64_t s1;
    uint64_t s2;
    uint64_t s3;
};

static ScalarWords64 reconstruct_share(
    const uint64_t seed[4], uint64_t offset, uint64_t high_offset
) {
    ScalarWords64 result{};
#if defined(__SIZEOF_INT128__)
    unsigned __int128 sum0 = (unsigned __int128)seed[0] + offset;
    result.s0 = (uint64_t)sum0;
    unsigned __int128 sum1 = (unsigned __int128)seed[1] + (uint64_t)(sum0 >> 64);
    result.s1 = (uint64_t)sum1;
    unsigned __int128 sum2 = (unsigned __int128)seed[2] + (uint64_t)(sum1 >> 64);
    result.s2 = (uint64_t)sum2;
    unsigned __int128 sum3 = (unsigned __int128)seed[3] + (uint64_t)(sum2 >> 64) + high_offset;
    result.s3 = (uint64_t)sum3;
#else
    uint64_t carry = 0;
    result.s0 = seed[0] + offset;
    carry = result.s0 < offset;
    result.s1 = seed[1] + carry;
    carry = result.s1 < carry;
    result.s2 = seed[2] + carry;
    carry = result.s2 < carry;
    result.s3 = seed[3] + carry + high_offset;
#endif
    return result;
}

static void print_usage(const char *program) {
    std::cout
        << "Vaddro " << VADDRO_VERSION << " - Vanity Address Online\n"
        << "Seed Offline. Compute Online.\n\n"
        << "Usage:\n"
        << "  " << program << " --self-test [-g GPU]\n"
        << "  " << program << " --benchmark -z PUBLIC_KEY [tuning options]\n"
        << "  " << program << " --matching PATTERN -z PUBLIC_KEY -o DIR [tuning options]\n"
        << "  " << program << " --same-char-ends N -z PUBLIC_KEY -o DIR [tuning options]\n\n"
        << "Options:\n"
        << "  --self-test                 Run GPU cryptographic known-answer tests\n"
        << "  --profile                   Synchronize and report per-stage timing (diagnostic only)\n"
        << "  --benchmark-rounds N        Benchmark loop count (default: 500)\n"
        << "  --canary-interval N         Verify one live recurrence lane every N rounds (0 disables)\n"
        << "  --match-score N             Fixed-pattern match threshold (normally auto-derived)\n"
        << "  --max-hits N                Stop after N verified hits; 0 means unlimited (default: 1)\n"
        << "  --json-log PATH             Append structured JSONL runtime events\n"
        << "  --worker-id ID              Worker label recorded in results and logs\n"
        << "  --run-id ID                 Unique run label recorded in results and logs\n"
        << "  --shard-index N             Zero-based worker shard index (default: 0)\n"
        << "  --shard-count N             Power-of-two worker shard count (default: 1)\n"
        << "  --build-commit SHA          Source revision recorded in results and logs\n"
        << "  --driver-version VERSION    NVIDIA driver package version for audit metadata\n"
        << "  -g, --gpu N                 CUDA device index\n"
        << "  -w, --work-size N           Threads per block (32..256, multiple of 32)\n"
        << "  -I, --inverse-multiple N    Number of 255-point inverse batches\n"
        << "  -s, --streams N             CUDA streams (1..8)\n"
        << "  -h, --help                  Show this help\n"
        << "  --version                   Show the version\n";
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::string pubkey_str;
    std::string pattern_str;
    std::string output_dir;
    std::string json_log_path;
    std::string worker_id = "standalone";
    std::string run_id = "standalone";
    std::string build_commit = "unknown";
    std::string driver_version_override;
    int match_score = 0;
    int same_char_ends = 0;
    bool benchmark_mode = false;
    bool profile_mode = false;
    bool self_test_mode = false;
    int device_id = 0;
    uint32_t num_inverses = 32768; // 32768 * 255 = 8,355,840 points (8.35M items, optimal cache fit for 5090)
    uint32_t work_size = 256;
    uint32_t inverse_size = PROFANITY_INVERSE_SIZE;
    uint32_t num_streams = 8;
    uint32_t canary_interval = 10000;
    uint32_t benchmark_rounds = 500;
    uint32_t max_hits = 1;
    uint32_t shard_index = 0;
    uint32_t shard_count = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--version") {
            std::cout << VADDRO_VERSION << std::endl;
            return 0;
        } else if (arg == "--self-test") {
            self_test_mode = true;
        } else if ((arg == "-z" || arg == "-p" || arg == "--publicKey" || arg == "--public-key") && i + 1 < argc) {
            pubkey_str = argv[++i];
        } else if ((arg == "-m" || arg == "--matching") && i + 1 < argc) {
            pattern_str = argv[++i];
        } else if ((arg == "-q" || arg == "--match-score" || arg == "--quit-score") && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], match_score)) {
                std::cerr << "ERROR: Invalid match score." << std::endl;
                return 1;
            }
        } else if ((arg == "-E" || arg == "--same-char-ends") && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], same_char_ends)) {
                std::cerr << "ERROR: Invalid same-char-ends value." << std::endl;
                return 1;
            }
        } else if ((arg == "-o" || arg == "--output-dir") && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "-0" || arg == "--benchmark") {
            benchmark_mode = true;
        } else if (arg == "--profile") {
            profile_mode = true;
        } else if ((arg == "-g" || arg == "--gpu") && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], device_id) || device_id < 0) {
                std::cerr << "ERROR: Invalid GPU index." << std::endl;
                return 1;
            }
        } else if ((arg == "-I" || arg == "--inverse-multiple") && i + 1 < argc) {
            if (!parse_u32_arg(argv[++i], num_inverses)) {
                std::cerr << "ERROR: Invalid inverse-multiple value." << std::endl;
                return 1;
            }
        } else if ((arg == "-w" || arg == "--work-size") && i + 1 < argc) {
            if (!parse_u32_arg(argv[++i], work_size)) {
                std::cerr << "ERROR: Invalid work-size value." << std::endl;
                return 1;
            }
        } else if ((arg == "-i" || arg == "--inverse-size") && i + 1 < argc) {
            if (!parse_u32_arg(argv[++i], inverse_size)) {
                std::cerr << "ERROR: Invalid inverse-size value." << std::endl;
                return 1;
            }
        } else if ((arg == "-s" || arg == "--streams") && i + 1 < argc) {
            if (!parse_u32_arg(argv[++i], num_streams)) {
                std::cerr << "ERROR: Invalid streams value." << std::endl;
                return 1;
            }
        } else if (arg == "--canary-interval" && i + 1 < argc) {
            if (!parse_u32_arg(argv[++i], canary_interval)) {
                std::cerr << "ERROR: Invalid canary interval." << std::endl;
                return 1;
            }
        } else if (arg == "--benchmark-rounds" && i + 1 < argc) {
            if (!parse_u32_arg(argv[++i], benchmark_rounds) || benchmark_rounds == 0) {
                std::cerr << "ERROR: Invalid benchmark round count." << std::endl;
                return 1;
            }
        } else if (arg == "--max-hits" && i + 1 < argc) {
            if (!parse_u32_arg(argv[++i], max_hits)) {
                std::cerr << "ERROR: Invalid max-hits value." << std::endl;
                return 1;
            }
        } else if (arg == "--json-log" && i + 1 < argc) {
            json_log_path = argv[++i];
        } else if (arg == "--worker-id" && i + 1 < argc) {
            worker_id = argv[++i];
        } else if (arg == "--run-id" && i + 1 < argc) {
            run_id = argv[++i];
        } else if (arg == "--shard-index" && i + 1 < argc) {
            if (!parse_u32_arg(argv[++i], shard_index)) {
                std::cerr << "ERROR: Invalid shard index." << std::endl;
                return 1;
            }
        } else if (arg == "--shard-count" && i + 1 < argc) {
            if (!parse_u32_arg(argv[++i], shard_count)) {
                std::cerr << "ERROR: Invalid shard count." << std::endl;
                return 1;
            }
        } else if (arg == "--build-commit" && i + 1 < argc) {
            build_commit = argv[++i];
        } else if (arg == "--driver-version" && i + 1 < argc) {
            driver_version_override = argv[++i];
        } else {
            std::cerr << "ERROR: Unknown or incomplete argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (benchmark_mode && (!pattern_str.empty() || same_char_ends != 0)) {
        std::cerr << "ERROR: Benchmark mode cannot be combined with a search mode." << std::endl;
        return 1;
    }

    if (!is_safe_audit_label(worker_id, 64) || !is_safe_audit_label(run_id, 96) ||
        !is_safe_audit_label(build_commit, 64) ||
        (!driver_version_override.empty() && !is_safe_audit_label(driver_version_override, 32))) {
        std::cerr << "ERROR: worker-id, run-id, and build-commit must use only "
                  << "letters, digits, dots, underscores, or hyphens." << std::endl;
        return 1;
    }

    if (!self_test_mode && !benchmark_mode && pattern_str.empty() && same_char_ends == 0) {
        std::cerr << "ERROR: No search mode specified. Use --matching or --same-char-ends." << std::endl;
        return 1;
    }

    if (same_char_ends != 0 && (same_char_ends < 1 || same_char_ends > 8)) {
        std::cerr << "ERROR: same-char-ends must be between 1 and 8." << std::endl;
        return 1;
    }

    if (work_size < 32 || work_size > 256 || (work_size % 32) != 0) {
        std::cerr << "ERROR: work-size must be a multiple of 32 between 32 and 256." << std::endl;
        return 1;
    }

    if (inverse_size != PROFANITY_INVERSE_SIZE) {
        std::cerr << "ERROR: this build supports only inverse-size " << PROFANITY_INVERSE_SIZE
                  << ", got " << inverse_size << "." << std::endl;
        return 1;
    }

    if (num_streams < 1 || num_streams > 8) {
        std::cerr << "ERROR: streams must be between 1 and 8." << std::endl;
        return 1;
    }

    if (canary_interval != 0 && canary_interval < 20) {
        std::cerr << "ERROR: canary-interval must be 0 or at least 20 rounds." << std::endl;
        return 1;
    }

    if (num_inverses < num_streams || (num_inverses % num_streams) != 0 ||
        num_inverses > (UINT32_MAX / PROFANITY_INVERSE_SIZE)) {
        std::cerr << "ERROR: inverse-multiple must be a positive multiple of streams ("
                  << num_streams << ") no larger than "
                  << (UINT32_MAX / PROFANITY_INVERSE_SIZE) << "." << std::endl;
        return 1;
    }

    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        std::cerr << "ERROR: No CUDA devices found!" << std::endl;
        return 1;
    }

    if (device_id >= device_count) {
        std::cerr << "ERROR: Invalid GPU index " << device_id << std::endl;
        return 1;
    }

    cuda_check(cudaSetDevice(device_id), "cudaSetDevice");
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");
    int cuda_driver_version = 0;
    cuda_check(cudaDriverGetVersion(&cuda_driver_version), "cudaDriverGetVersion");

    cudaFuncAttributes iterate_attributes{};
    cuda_check(cudaFuncGetAttributes(&iterate_attributes, profanity_iterate_kernel),
               "query profanity_iterate_kernel attributes");
    int iterate_active_blocks_per_sm = 0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &iterate_active_blocks_per_sm,
                   profanity_iterate_kernel,
                   static_cast<int>(work_size),
                   0),
               "query profanity_iterate_kernel occupancy");
    const int iterate_active_threads_per_sm =
        iterate_active_blocks_per_sm * static_cast<int>(work_size);

    cudaFuncAttributes point_basis_attributes{};
    cudaFuncAttributes variant_hash_attributes{};
    int point_basis_active_blocks_per_sm = 0;
    int variant_hash_active_blocks_per_sm = 0;
    int variant_hash_active_warps_per_sm = 0;
#if VADDRO_SPLIT_VARIANT_HASH
    cuda_check(cudaFuncGetAttributes(&point_basis_attributes, profanity_point_basis_kernel),
               "query profanity_point_basis_kernel attributes");
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &point_basis_active_blocks_per_sm,
                   profanity_point_basis_kernel,
                   static_cast<int>(work_size),
                   0),
               "query profanity_point_basis_kernel occupancy");
    cuda_check(cudaFuncGetAttributes(&variant_hash_attributes, profanity_variant_hash_kernel),
               "query profanity_variant_hash_kernel attributes");
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &variant_hash_active_blocks_per_sm,
                   profanity_variant_hash_kernel,
                   static_cast<int>(VADDRO_VARIANT_TILE_POINTS * VADDRO_VARIANT_COUNT),
                   0),
               "query profanity_variant_hash_kernel occupancy");
    variant_hash_active_warps_per_sm =
        variant_hash_active_blocks_per_sm *
        static_cast<int>((VADDRO_VARIANT_TILE_POINTS * VADDRO_VARIANT_COUNT) / 32);
#endif

    cudaFuncAttributes inverse_attributes{};
    cuda_check(cudaFuncGetAttributes(&inverse_attributes, profanity_inverse_kernel),
               "query profanity_inverse_kernel attributes");
    int inverse_active_blocks_per_sm = 0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &inverse_active_blocks_per_sm,
                   profanity_inverse_kernel,
                   64,
                   0),
               "query profanity_inverse_kernel occupancy");

    if (self_test_mode) {
        uint32_t *d_status = nullptr;
        uint32_t status = 0;
        cuda_check(cudaMalloc(&d_status, sizeof(uint32_t)), "allocate self-test status");
        crypto_self_test_kernel<<<1, 1>>>(d_status);
        cuda_check(cudaGetLastError(), "crypto_self_test_kernel launch");
        cuda_check(cudaMemcpy(&status, d_status, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                   "copy self-test status");
        cudaFree(d_status);
        constexpr uint32_t expected_status = (1u << 6) - 1u;
        if (status != expected_status) {
            std::cerr << "GPU cryptographic self-test: FAILED (status=0x"
                      << std::hex << status << ", expected=0x" << expected_status << std::dec << ")\n"
                      << "Bits: curve, keccak, mul, square, inverse differential, GLV variants" << std::endl;
            return 2;
        }
#if VADDRO_SPLIT_VARIANT_HASH
        uint256_t *d_split_basis[VADDRO_VARIANT_BASIS_COORDS] = {};
        MatchResult *d_split_results = nullptr;
        uint32_t *d_split_result_count = nullptr;
        for (uint32_t coord = 0; coord < VADDRO_VARIANT_BASIS_COORDS; ++coord) {
            cuda_check(cudaMalloc(&d_split_basis[coord], sizeof(uint256_t)),
                       "allocate split self-test basis");
        }
        cuda_check(cudaMalloc(&d_split_results,
                              sizeof(MatchResult) * MAX_RESULTS),
                   "allocate split self-test results");
        cuda_check(cudaMalloc(&d_split_result_count, sizeof(uint32_t)),
                   "allocate split self-test result count");
        cuda_check(cudaMemset(d_split_results, 0,
                              sizeof(MatchResult) * MAX_RESULTS),
                   "initialize split self-test results");
        cuda_check(cudaMemset(d_split_result_count, 0, sizeof(uint32_t)),
                   "initialize split self-test result count");
        split_variant_self_test_prepare_kernel<<<1, 1>>>(
            d_split_basis[VADDRO_BASIS_X],
            d_split_basis[VADDRO_BASIS_BETA_X],
            d_split_basis[VADDRO_BASIS_BETA2_X],
            d_split_basis[VADDRO_BASIS_Y],
            d_split_basis[VADDRO_BASIS_NEG_Y]);
        cuda_check(cudaGetLastError(), "split_variant_self_test_prepare_kernel launch");

        SearchTarget split_target{};
        split_target.mode = 1;
        split_target.required_score = 40;
        constexpr uint64_t split_test_round = 7;
        const dim3 split_test_block(VADDRO_VARIANT_TILE_POINTS, VADDRO_VARIANT_COUNT);
        profanity_variant_hash_kernel<<<1, split_test_block>>>(
            d_split_basis[VADDRO_BASIS_X],
            d_split_basis[VADDRO_BASIS_BETA_X],
            d_split_basis[VADDRO_BASIS_BETA2_X],
            d_split_basis[VADDRO_BASIS_Y],
            d_split_basis[VADDRO_BASIS_NEG_Y],
            split_target,
            d_split_results,
            d_split_result_count,
            1,
            split_test_round);
        cuda_check(cudaGetLastError(), "split self-test variant-hash launch");

        MatchResult split_results[VADDRO_VARIANT_COUNT] = {};
        uint32_t split_result_count = 0;
        cuda_check(cudaMemcpy(&split_result_count, d_split_result_count, sizeof(uint32_t),
                              cudaMemcpyDeviceToHost),
                   "copy split self-test result count");
        if (split_result_count == VADDRO_VARIANT_COUNT) {
            cuda_check(cudaMemcpy(split_results, d_split_results, sizeof(split_results),
                                  cudaMemcpyDeviceToHost),
                       "copy split self-test results");
        }
        for (uint32_t coord = 0; coord < VADDRO_VARIANT_BASIS_COORDS; ++coord) {
            cudaFree(d_split_basis[coord]);
        }
        cudaFree(d_split_results);
        cudaFree(d_split_result_count);

        const uint32_t split_expected[VADDRO_VARIANT_COUNT][5] = {
            {0x7E5F4552u, 0x091A6912u, 0x5D5DFCB7u, 0xB8C26590u, 0x29395BDFu},
            {0x1869DBF1u, 0x8B0786E3u, 0xB2D8D768u, 0x8FD5F6B9u, 0xC1D5AD17u},
            {0x6889E05Eu, 0xFEFDEEC6u, 0x24BA68BBu, 0x8D6343FAu, 0x1F8911C0u},
            {0x80C0DBF2u, 0x39224071u, 0xC59DD897u, 0x0AB9D542u, 0xE3414AB2u},
            {0xD8955DF1u, 0xD306FCFBu, 0xC4F570FEu, 0xF413D0DCu, 0x954894E8u},
            {0x1B7915F7u, 0x776594DEu, 0x59625CB9u, 0xE8B31506u, 0x335EE3FBu}
        };
        uint32_t split_status = 0;
        if (split_result_count == VADDRO_VARIANT_COUNT) {
            for (uint32_t i = 0; i < VADDRO_VARIANT_COUNT; ++i) {
                const MatchResult &result = split_results[i];
                bool result_ok = result.found == 1u &&
                                 result.score == 40u &&
                                 result.variant < VADDRO_VARIANT_COUNT &&
                                 result.share_id == 0u &&
                                 result.round_found == split_test_round;
                if (result.variant < VADDRO_VARIANT_COUNT) {
                    for (int word = 0; word < 5; ++word) {
                        result_ok &= result.address[word] == split_expected[result.variant][word];
                    }
                }
                const uint32_t variant_bit = result.variant < VADDRO_VARIANT_COUNT
                    ? (1u << result.variant)
                    : 0u;
                result_ok &= (split_status & variant_bit) == 0u;
                if (result_ok) split_status |= variant_bit;
            }
        }
        if (split_status != ((1u << VADDRO_VARIANT_COUNT) - 1u)) {
            std::cerr << "Split production hash self-test: FAILED (count="
                      << split_result_count << ", status=0x" << std::hex
                      << split_status << std::dec << ")" << std::endl;
            return 2;
        }
#endif
        std::cout << "GPU cryptographic self-test: OK\n"
                  << "Device: " << prop.name << " (sm_" << prop.major << prop.minor << ")\n"
                  << "GLV X reuse: " << (VADDRO_REUSE_GLV_X ? "enabled" : "disabled") << "\n"
                  << "Inverse addition chain: "
                  << (VADDRO_INV_ADDITION_CHAIN ? "enabled" : "disabled") << "\n"
                  << "Split variant hash: "
                  << (VADDRO_SPLIT_VARIANT_HASH ? "enabled" : "disabled") << "\n"
                  << "Inverse resources: " << inverse_attributes.numRegs << " regs/thread, "
                  << inverse_attributes.localSizeBytes << " local bytes/thread, "
                  << inverse_active_blocks_per_sm << " active blocks/SM at block=64\n"
                  << "Iterate resources: " << iterate_attributes.numRegs << " regs/thread, "
                  << iterate_active_blocks_per_sm << " active blocks/SM at block=" << work_size << "\n";
#if VADDRO_SPLIT_VARIANT_HASH
        std::cout << "Point-basis resources: " << point_basis_attributes.numRegs << " regs/thread, "
                  << point_basis_attributes.localSizeBytes << " local bytes/thread, "
                  << point_basis_active_blocks_per_sm << " active blocks/SM at block=" << work_size << "\n"
                  << "Variant-hash resources: " << variant_hash_attributes.numRegs << " regs/thread, "
                  << variant_hash_attributes.localSizeBytes << " local bytes/thread, "
                  << variant_hash_active_blocks_per_sm << " active blocks/SM, "
                  << variant_hash_active_warps_per_sm << " active warps/SM at block=192\n"
                  << "Split production hash self-test: OK\n";
#endif
        std::cout
                  << "Version: " << VADDRO_VERSION << std::endl;
        return 0;
    }

    if (pubkey_str.length() != 128 || !is_hex_string(pubkey_str)) {
        std::cerr << "ERROR: Public key must be exactly 128 hexadecimal characters!" << std::endl;
        return 1;
    }

    PointAffine base_point;
    parse_hex_256(pubkey_str.substr(0, 64), base_point.x.d);
    parse_hex_256(pubkey_str.substr(64, 64), base_point.y.d);

    uint32_t *d_base_valid = nullptr;
    cuda_check(cudaMalloc(&d_base_valid, sizeof(uint32_t)), "allocate base-point validity flag");
    verify_base_point_kernel<<<1, 1>>>(base_point, d_base_valid);
    cuda_check(cudaGetLastError(), "verify_base_point_kernel launch");
    uint32_t base_valid = 0;
    cuda_check(cudaMemcpy(&base_valid, d_base_valid, sizeof(uint32_t), cudaMemcpyDeviceToHost),
               "copy base-point validity flag");
    cudaFree(d_base_valid);
    if (base_valid != 1u) {
        std::cerr << "ERROR: Split public key is not a valid secp256k1 affine point." << std::endl;
        return 1;
    }

    SearchTarget target;
    std::memset(&target, 0, sizeof(target));

    if (benchmark_mode) {
        target.mode = 0;
    } else if (same_char_ends != 0) {
        target.mode = 2;
        target.same_char_ends = same_char_ends;
    } else {
        if (pattern_str.length() != 40) {
            std::cerr << "ERROR: search pattern must be exactly 40 characters." << std::endl;
            return 1;
        }
        for (char c : pattern_str) {
            if (!((c >= '0' && c <= '9') ||
                  (c >= 'a' && c <= 'f') ||
                  (c >= 'A' && c <= 'F') ||
                  c == 'X' || c == 'x' || c == '.')) {
                std::cerr << "ERROR: search pattern contains an invalid character: '" << c << "'." << std::endl;
                return 1;
            }
        }

        int pattern_score = count_fixed_pattern(pattern_str);
        if (pattern_score == 0) {
            std::cerr << "ERROR: an all-wildcard pattern would match every generated address." << std::endl;
            return 1;
        }
        if (match_score <= 0 || match_score > pattern_score) {
            match_score = pattern_score;
        }

        target.mode = 1;
        target.required_score = match_score;

        for (size_t i = 0; i < pattern_str.length(); ++i) {
            char c = pattern_str[i];
            if (c != 'X' && c != 'x' && c != '.') {
                int word_idx = static_cast<int>(i / 8);
                int nibble_idx = static_cast<int>(i % 8);
                int shift = (7 - nibble_idx) * 4;
                uint32_t nibble_val = parse_hex_char(c);

                target.word_mask[word_idx] |= (0xFu << shift);
                target.word_val[word_idx]  |= (nibble_val << shift);
            }
        }
    }

    const uint64_t total_points64 =
        static_cast<uint64_t>(num_inverses) * PROFANITY_INVERSE_SIZE;
    const uint32_t total_points = static_cast<uint32_t>(total_points64);
    vaddro::SearchShardRange shard_range;
    std::string shard_error;
    if (!vaddro::calculate_search_shard(
            shard_index, shard_count, total_points, shard_range, shard_error)) {
        std::cerr << "ERROR: " << shard_error << "." << std::endl;
        return 1;
    }

    AuditContext audit;
    audit.worker_id = worker_id;
    audit.run_id = run_id;
    audit.build_commit = build_commit;
    audit.gpu_name = prop.name;
    audit.driver_version = driver_version_override.empty()
        ? format_cuda_version(cuda_driver_version)
        : driver_version_override;
    audit.gpu_index = device_id;
    audit.shard_index = shard_index;
    audit.shard_count = shard_count;
    if (benchmark_mode) {
        audit.mode = "benchmark";
    } else if (same_char_ends != 0) {
        audit.mode = "same-char-ends:" + std::to_string(same_char_ends);
    } else {
        audit.mode = "pattern:" + pattern_str;
    }
    const std::string search_started_utc = utc_now_iso8601();
    if (!append_json_event(json_log_path, "started", audit,
                           {{"output_dir", output_dir},
                            {"glv_x_reuse", VADDRO_REUSE_GLV_X ? "enabled" : "disabled"},
                            {"inverse_addition_chain",
                             VADDRO_INV_ADDITION_CHAIN ? "enabled" : "disabled"},
                            {"split_variant_hash",
                             VADDRO_SPLIT_VARIANT_HASH ? "enabled" : "disabled"}},
                           {{"max_hits", std::to_string(max_hits)},
                            {"inverse_registers_per_thread", std::to_string(inverse_attributes.numRegs)},
                            {"inverse_local_bytes_per_thread", std::to_string(inverse_attributes.localSizeBytes)},
                            {"inverse_active_blocks_per_sm", std::to_string(inverse_active_blocks_per_sm)},
                            {"iterate_registers_per_thread", std::to_string(iterate_attributes.numRegs)},
                            {"iterate_active_blocks_per_sm", std::to_string(iterate_active_blocks_per_sm)},
                            {"iterate_active_threads_per_sm", std::to_string(iterate_active_threads_per_sm)},
                            {"point_basis_registers_per_thread", std::to_string(point_basis_attributes.numRegs)},
                            {"point_basis_local_bytes_per_thread", std::to_string(point_basis_attributes.localSizeBytes)},
                            {"point_basis_active_blocks_per_sm", std::to_string(point_basis_active_blocks_per_sm)},
                            {"variant_hash_registers_per_thread", std::to_string(variant_hash_attributes.numRegs)},
                            {"variant_hash_local_bytes_per_thread", std::to_string(variant_hash_attributes.localSizeBytes)},
                            {"variant_hash_active_blocks_per_sm", std::to_string(variant_hash_active_blocks_per_sm)},
                            {"variant_hash_active_warps_per_sm", std::to_string(variant_hash_active_warps_per_sm)}})) {
        std::cerr << "ERROR: could not initialize JSONL log: " << json_log_path << std::endl;
        return 1;
    }

    const bool is_interactive_tty = (isatty(fileno(stdout)) != 0);
    // Format strings for Vaddro Header and Telemetry Dashboard
    std::string pubkey_abbr = pubkey_str.substr(0, 24) + "..." + pubkey_str.substr(pubkey_str.length() - 16);
    std::string pattern_disp;
    if (benchmark_mode) {
        pattern_disp = "BENCHMARK MODE (Raw EC Point Generation)";
    } else if (same_char_ends != 0) {
        pattern_disp = "0xXXXXXX...XXXXXX (Same " + std::to_string(same_char_ends) + " chars at both ends)";
    } else {
        pattern_disp = "0x" + pattern_str + " (Score " + std::to_string(match_score) + ")";
    }

    std::ostringstream gpu_ss;
    gpu_ss << "GPU #" << device_id << " - " << prop.name << " ("
           << (prop.totalGlobalMem / (1024 * 1024 * 1024)) << "GB, sm_"
           << prop.major << prop.minor << ")";
    std::string gpu_disp = gpu_ss.str();

    std::ostringstream driver_ss;
    driver_ss << audit.driver_version << " (CUDA API "
              << format_cuda_version(cuda_driver_version) << ")";
    std::string driver_disp = driver_ss.str();

    std::ostringstream kernel_ss;
    kernel_ss << "b" << work_size << " inv" << num_inverses << "x"
              << PROFANITY_INVERSE_SIZE << " s" << num_streams
              << " glv:" << (VADDRO_REUSE_GLV_X ? "on" : "off")
              << " ichain:" << (VADDRO_INV_ADDITION_CHAIN ? "on" : "off")
              << " split:" << (VADDRO_SPLIT_VARIANT_HASH ? "on" : "off");
#if VADDRO_SPLIT_VARIANT_HASH
    kernel_ss << " pr" << point_basis_attributes.numRegs
              << " hr" << variant_hash_attributes.numRegs
              << " hwarp/sm:" << variant_hash_active_warps_per_sm;
#else
    kernel_ss << " r" << iterate_attributes.numRegs
              << " blk/sm:" << iterate_active_blocks_per_sm;
#endif
    std::string kernel_disp = kernel_ss.str();

    const uint64_t candidates_per_round = total_points64 * (benchmark_mode ? 1ULL : 6ULL);
    std::string batch_disp = format_key_count(candidates_per_round) +
        (benchmark_mode ? " points/round" : " candidates/round (6 GLV variants)");

    std::string canary_disp = canary_interval == 0
        ? "DISABLED"
        : "Every " + std::to_string(canary_interval) + " rounds";
    const std::string shard_disp = std::to_string(shard_index) + " / " +
        std::to_string(shard_count) + " (zero-based index / total)";

    const std::string output_disp = output_dir.empty()
        ? "current directory"
        : abbreviate_middle(output_dir, 57);
    const std::string json_log_disp = json_log_path.empty()
        ? "disabled"
        : abbreviate_middle(json_log_path, 57);

    std::string hit_policy_disp;
    if (benchmark_mode) {
        hit_policy_disp = "Not applicable (benchmark mode)";
    } else if (max_hits == 0) {
        hit_policy_disp = "Continuous (MAX_HITS=0)";
    } else {
        hit_policy_disp = "Stop after " + std::to_string(max_hits) + " verified hit(s)";
    }

    std::string border_line = "+" + std::string(78, '-') + "+";

    if (should_clear_interactive_screen()) {
        // Clear only the visible terminal viewport. Redirected logs remain untouched.
        std::cout << "\033[2J\033[H";
    }

    std::cout << border_line << "\n"
              << "| " << pad_exact(" __      __       _     _", 76) << " |\n"
              << "| " << pad_exact(" \\ \\    / /      | |   | |                 [*] V A D D R O", 76) << " |\n"
              << "| " << pad_exact("  \\ \\  / /_ _  __| | __| |_ __ ___         Vanity Address Online", 76) << " |\n"
              << "| " << pad_exact(std::string("   \\ \\/ / _` |/ _` |/ _` | '__/ _ \\        v") + VADDRO_VERSION + " (" + build_commit.substr(0, 12) + ")", 76) << " |\n"
              << "| " << pad_exact("    \\  / (_| | (_| | (_| | | | (_) |       Seed Offline. Compute Online.", 76) << " |\n"
              << "| " << pad_exact("     \\/ \\__,_|\\__,_|\\__,_|_|  \\___/      github.com/manifoldor/vaddro", 76) << " |\n"
              << border_line << "\n"
              << "| " << pad_exact("[WORKER ID]      | " + worker_id, 76) << " |\n"
              << "| " << pad_exact("[RUN ID]         | " + run_id, 76) << " |\n"
              << "| " << pad_exact("[STARTED UTC]    | " + search_started_utc, 76) << " |\n"
              << "| " << pad_exact("[SOURCE REV]     | " + build_commit, 76) << " |\n"
              << border_line << "\n"
              << "| " << pad_exact("[TARGET PUBKEY]  | " + pubkey_abbr, 76) << " |\n"
              << "| " << pad_exact("[SEARCH MODE]    | " + pattern_disp, 76) << " |\n"
              << "| " << pad_exact("[SEARCH SHARD]   | " + shard_disp, 76) << " |\n"
              << "| " << pad_exact("[HIT POLICY]     | " + hit_policy_disp, 76) << " |\n"
              << "| " << pad_exact("[RESULTS DIR]    | " + output_disp, 76) << " |\n"
              << "| " << pad_exact("[EVENT LOG]      | " + json_log_disp, 76) << " |\n"
              << border_line << "\n"
              << "| " << pad_exact("[WORKER DEVICE]  | " + gpu_disp, 76) << " |\n"
              << "| " << pad_exact("[NVIDIA DRIVER]  | " + driver_disp, 76) << " |\n"
              << "| " << pad_exact("[KERNEL CONFIG]  | " + kernel_disp, 76) << " |\n"
              << "| " << pad_exact("[SEARCH BATCH]   | " + batch_disp, 76) << " |\n"
              << "| " << pad_exact("[SAFETY CANARY]  | " + canary_disp, 76) << " |\n"
              << border_line << "\n"
              << "| " << pad_exact(" HASHRATE           TOTAL EXPLORED        TIME ELAPSED        HITS GENERATED", 76) << " |\n"
              << border_line << "\n"
              << std::flush;

    if (target.mode == 1) {
        target.prefix_filter_mask = target.word_mask[0] & 0xFFFFFF00u;
        target.prefix_filter_val  = target.word_val[0]  & 0xFFFFFF00u;
        if (target.prefix_filter_mask == 0) {
            target.prefix_filter_mask = target.word_mask[0];
            target.prefix_filter_val  = target.word_val[0];
        }
    }

    // Multi-Stream Non-Blocking Pipelined Architecture
    constexpr int MAX_STREAMS = 8;
    const int num_streams_int = static_cast<int>(num_streams);
    uint32_t stream_points = total_points / num_streams;
    uint32_t stream_inverses = stream_points / PROFANITY_INVERSE_SIZE;

    cudaStream_t streams[MAX_STREAMS];
    uint256_t *d_delta_x[MAX_STREAMS];
    uint256_t *d_inverse[MAX_STREAMS];
    uint256_t *d_prev_lambda[MAX_STREAMS];
#if VADDRO_SPLIT_VARIANT_HASH
    uint256_t *d_basis_x[MAX_STREAMS] = {};
    uint256_t *d_basis_beta_x[MAX_STREAMS] = {};
    uint256_t *d_basis_beta2_x[MAX_STREAMS] = {};
    uint256_t *d_basis_y[MAX_STREAMS] = {};
    uint256_t *d_basis_neg_y[MAX_STREAMS] = {};
#endif
    MatchResult *d_results[MAX_STREAMS];
    uint32_t *d_result_count[MAX_STREAMS];

    PointAffine *d_precomp = nullptr;
    cuda_check(cudaMalloc(&d_precomp, sizeof(PointAffine) * 8160), "allocate precomputed points");
    cuda_check(cudaMemcpy(d_precomp, g_precomp, sizeof(PointAffine) * 8160, cudaMemcpyHostToDevice),
               "copy precomputed points");

    for (int s = 0; s < num_streams_int; ++s) {
        cuda_check(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        cuda_check(cudaMalloc(&d_delta_x[s], sizeof(uint256_t) * stream_points), "allocate d_delta_x");
        cuda_check(cudaMalloc(&d_inverse[s], sizeof(uint256_t) * stream_points), "allocate d_inverse");
        cuda_check(cudaMalloc(&d_prev_lambda[s], sizeof(uint256_t) * stream_points), "allocate d_prev_lambda");
#if VADDRO_SPLIT_VARIANT_HASH
        if (!benchmark_mode) {
            cuda_check(cudaMalloc(&d_basis_x[s], sizeof(uint256_t) * stream_points),
                       "allocate d_basis_x");
            cuda_check(cudaMalloc(&d_basis_beta_x[s], sizeof(uint256_t) * stream_points),
                       "allocate d_basis_beta_x");
            cuda_check(cudaMalloc(&d_basis_beta2_x[s], sizeof(uint256_t) * stream_points),
                       "allocate d_basis_beta2_x");
            cuda_check(cudaMalloc(&d_basis_y[s], sizeof(uint256_t) * stream_points),
                       "allocate d_basis_y");
            cuda_check(cudaMalloc(&d_basis_neg_y[s], sizeof(uint256_t) * stream_points),
                       "allocate d_basis_neg_y");
        }
#endif
        cuda_check(cudaMalloc(&d_results[s], sizeof(MatchResult) * MAX_RESULTS), "allocate d_results");
        cuda_check(cudaMalloc(&d_result_count[s], sizeof(uint32_t)), "allocate d_result_count");
        cuda_check(cudaMemsetAsync(d_results[s], 0, sizeof(MatchResult) * MAX_RESULTS, streams[s]),
                   "initialize d_results");
        cuda_check(cudaMemsetAsync(d_result_count[s], 0, sizeof(uint32_t), streams[s]),
                   "initialize d_result_count");
    }

    uint64_t seed_u64[4];
    std::string random_error;
    if (!fill_worker_seed(seed_u64, shard_range, random_error)) {
        std::cerr << "ERROR: could not obtain worker search randomness: "
                  << random_error << "." << std::endl;
        return 1;
    }

    // Initialize Points across all streams
    for (int s = 0; s < num_streams_int; ++s) {
        profanity_init_kernel<<<(stream_points + 255) / 256, 256, 0, streams[s]>>>(
            d_precomp,
            base_point,
            seed_u64[0],
            seed_u64[1],
            seed_u64[2],
            seed_u64[3] + (uint64_t)(s * stream_points),
            d_delta_x[s],
            d_prev_lambda[s],
            stream_points
        );
        cuda_check(cudaGetLastError(), "profanity_init_kernel launch");
    }
    cuda_check(cudaDeviceSynchronize(), "profanity_init_kernel synchronize");

    MatchResult h_results[MAX_RESULTS];
    uint32_t h_result_count = 0;
    uint32_t *d_verify_addr = nullptr;
    uint32_t *d_state_addr = nullptr;
    cuda_check(cudaMalloc(&d_verify_addr, sizeof(uint32_t) * 5), "allocate d_verify_addr");
    cuda_check(cudaMalloc(&d_state_addr, sizeof(uint32_t) * 5), "allocate d_state_addr");

    auto start_time = std::chrono::steady_clock::now();
    uint64_t total_hashes = 0;
    uint64_t round = 0;
    uint64_t match_count = 0;
    double profile_inverse_seconds = 0.0;
    double profile_iterate_seconds = 0.0;
    double profile_point_basis_seconds = 0.0;
    double profile_variant_hash_seconds = 0.0;
    uint64_t profile_rounds = 0;
    // Validate the initialized recurrence quickly, then continue at the
    // configured low-overhead production interval.
    uint64_t next_canary_round = canary_interval == 0 ? UINT64_MAX : 20;
    uint64_t canary_checks = 0;
    bool canary_failed = false;
    bool result_write_failed = false;
    bool json_log_warning_emitted = false;
    std::string last_canary_utc = "never";
    std::string stop_reason = benchmark_mode ? "benchmark_complete" : "running";
#if VADDRO_SPLIT_VARIANT_HASH
    const dim3 variant_hash_block(VADDRO_VARIANT_TILE_POINTS, VADDRO_VARIANT_COUNT);
    const dim3 variant_hash_grid(
        (stream_points + VADDRO_VARIANT_TILE_POINTS - 1) / VADDRO_VARIANT_TILE_POINTS);
#endif

    while (g_running) {
        round++;

        if (profile_mode) {
            auto inverse_start = std::chrono::steady_clock::now();
            for (int s = 0; s < num_streams_int; ++s) {
                profanity_inverse_kernel<<<(stream_inverses + 63) / 64, 64, 0, streams[s]>>>(
                    d_delta_x[s], d_inverse[s], stream_inverses
                );
                cuda_check(cudaGetLastError(), "profanity_inverse_kernel launch");
            }
            cuda_check(cudaDeviceSynchronize(), "profile inverse synchronize");
            auto inverse_end = std::chrono::steady_clock::now();

            auto iterate_start = inverse_end;
#if VADDRO_SPLIT_VARIANT_HASH
            auto iterate_end = iterate_start;
            if (!benchmark_mode) {
                for (int s = 0; s < num_streams_int; ++s) {
                    profanity_point_basis_kernel<<<
                        (stream_points + work_size - 1) / work_size,
                        work_size, 0, streams[s]>>>(
                            d_delta_x[s], d_inverse[s], d_prev_lambda[s],
                            d_basis_x[s], d_basis_beta_x[s], d_basis_beta2_x[s],
                            d_basis_y[s], d_basis_neg_y[s], stream_points);
                    cuda_check(cudaGetLastError(), "profanity_point_basis_kernel launch");
                }
                cuda_check(cudaDeviceSynchronize(), "profile point-basis synchronize");
                const auto point_basis_end = std::chrono::steady_clock::now();

                for (int s = 0; s < num_streams_int; ++s) {
                    profanity_variant_hash_kernel<<<variant_hash_grid, variant_hash_block, 0, streams[s]>>>(
                        d_basis_x[s], d_basis_beta_x[s], d_basis_beta2_x[s],
                        d_basis_y[s], d_basis_neg_y[s], target,
                        d_results[s], d_result_count[s], stream_points, round);
                    cuda_check(cudaGetLastError(), "profanity_variant_hash_kernel launch");
                }
                cuda_check(cudaDeviceSynchronize(), "profile variant-hash synchronize");
                iterate_end = std::chrono::steady_clock::now();
                profile_point_basis_seconds +=
                    std::chrono::duration<double>(point_basis_end - iterate_start).count();
                profile_variant_hash_seconds +=
                    std::chrono::duration<double>(iterate_end - point_basis_end).count();
            } else {
                for (int s = 0; s < num_streams_int; ++s) {
                    profanity_iterate_kernel<<<
                        (stream_points + work_size - 1) / work_size,
                        work_size, 0, streams[s]>>>(
                            d_delta_x[s], d_inverse[s], d_prev_lambda[s], target,
                            d_results[s], d_result_count[s], stream_points, round);
                    cuda_check(cudaGetLastError(), "profanity_iterate_kernel launch");
                }
                cuda_check(cudaDeviceSynchronize(), "profile iterate synchronize");
                iterate_end = std::chrono::steady_clock::now();
            }
#else
            for (int s = 0; s < num_streams_int; ++s) {
                profanity_iterate_kernel<<<(stream_points + work_size - 1) / work_size, work_size, 0, streams[s]>>>(
                    d_delta_x[s],
                    d_inverse[s],
                    d_prev_lambda[s],
                    target,
                    d_results[s],
                    d_result_count[s],
                    stream_points,
                    round
                );
                cuda_check(cudaGetLastError(), "profanity_iterate_kernel launch");
            }
            cuda_check(cudaDeviceSynchronize(), "profile iterate synchronize");
            auto iterate_end = std::chrono::steady_clock::now();
#endif

            profile_inverse_seconds += std::chrono::duration<double>(inverse_end - inverse_start).count();
            profile_iterate_seconds += std::chrono::duration<double>(iterate_end - iterate_start).count();
            profile_rounds++;
        } else {
            // Launch concurrent non-blocking kernels across all streams
            for (int s = 0; s < num_streams_int; ++s) {
                profanity_inverse_kernel<<<(stream_inverses + 63) / 64, 64, 0, streams[s]>>>(
                    d_delta_x[s], d_inverse[s], stream_inverses
                );
                cuda_check(cudaGetLastError(), "profanity_inverse_kernel launch");
#if VADDRO_SPLIT_VARIANT_HASH
                if (!benchmark_mode) {
                    profanity_point_basis_kernel<<<
                        (stream_points + work_size - 1) / work_size,
                        work_size, 0, streams[s]>>>(
                            d_delta_x[s], d_inverse[s], d_prev_lambda[s],
                            d_basis_x[s], d_basis_beta_x[s], d_basis_beta2_x[s],
                            d_basis_y[s], d_basis_neg_y[s], stream_points);
                    cuda_check(cudaGetLastError(), "profanity_point_basis_kernel launch");
                    profanity_variant_hash_kernel<<<variant_hash_grid, variant_hash_block, 0, streams[s]>>>(
                        d_basis_x[s], d_basis_beta_x[s], d_basis_beta2_x[s],
                        d_basis_y[s], d_basis_neg_y[s], target,
                        d_results[s], d_result_count[s], stream_points, round);
                    cuda_check(cudaGetLastError(), "profanity_variant_hash_kernel launch");
                } else {
                    profanity_iterate_kernel<<<
                        (stream_points + work_size - 1) / work_size,
                        work_size, 0, streams[s]>>>(
                            d_delta_x[s], d_inverse[s], d_prev_lambda[s], target,
                            d_results[s], d_result_count[s], stream_points, round);
                    cuda_check(cudaGetLastError(), "profanity_iterate_kernel launch");
                }
#else
                profanity_iterate_kernel<<<(stream_points + work_size - 1) / work_size, work_size, 0, streams[s]>>>(
                    d_delta_x[s],
                    d_inverse[s],
                    d_prev_lambda[s],
                    target,
                    d_results[s],
                    d_result_count[s],
                    stream_points,
                    round
                );
                cuda_check(cudaGetLastError(), "profanity_iterate_kernel launch");
#endif
            }
        }

        total_hashes += total_points;

        if (target.mode != 0 && (round % 20 == 0 || !g_running)) {
            for (int s = 0;
                 s < num_streams_int && !result_write_failed &&
                     (max_hits == 0 || match_count < max_hits);
                 ++s) {
                cuda_check(cudaMemcpyAsync(&h_result_count, d_result_count[s], sizeof(uint32_t),
                                           cudaMemcpyDeviceToHost, streams[s]),
                           "copy result count");
                cuda_check(cudaStreamSynchronize(streams[s]), "synchronize result stream");
                if (h_result_count > 0) {
                    uint32_t copy_count = h_result_count > MAX_RESULTS ? MAX_RESULTS : h_result_count;
                    cuda_check(cudaMemcpy(h_results, d_results[s], sizeof(MatchResult) * copy_count,
                                          cudaMemcpyDeviceToHost), "copy match results");
                    cuda_check(cudaMemsetAsync(d_result_count[s], 0, sizeof(uint32_t), streams[s]),
                               "reset result count");

                    for (uint32_t i = 0;
                         i < copy_count && !result_write_failed &&
                             (max_hits == 0 || match_count < max_hits);
                         ++i) {
                        std::string addr_str = format_address(h_results[i].address);

                        uint64_t offset = (uint64_t)h_results[i].round_found + 1;
                        uint64_t global_share_id = (uint64_t)h_results[i].share_id + (uint64_t)(s * stream_points);
                        ScalarWords64 share = reconstruct_share(seed_u64, offset, global_share_id);

                        // Zero-Defect Instant Hardware Verification Check
                        uint32_t variant = h_results[i].variant;
                        verify_scalar_kernel<<<1, 1>>>(d_precomp, base_point, share.s0, share.s1,
                                                       share.s2, share.s3, variant, d_verify_addr);
                        cuda_check(cudaGetLastError(), "verify_scalar_kernel launch");
                        uint32_t h_verify_addr[5];
                        cuda_check(cudaMemcpy(h_verify_addr, d_verify_addr, sizeof(uint32_t) * 5,
                                              cudaMemcpyDeviceToHost), "copy verified address");
                        std::string verified_addr_str = format_address(h_verify_addr);

                        if (verified_addr_str != addr_str) {
                            std::cerr << "\n⚠️ [REJECTED ARTIFACT] Share verification failed ("
                                      << verified_addr_str << " != " << addr_str << "). Discarding false match." << std::endl;
                            continue;
                        }

                        std::ostringstream ss;
                        ss << std::hex << std::setfill('0');
                        ss << std::setw(16) << share.s3 << std::setw(16) << share.s2
                           << std::setw(16) << share.s1 << std::setw(16) << share.s0;
                        std::string share_str = ss.str();

                    std::string variant_desc;
                    switch(variant) {
                        case 0: variant_desc = "0 (Affine Point (x, y))"; break;
                        case 1: variant_desc = "1 (GLV Endomorphism (β x, y))"; break;
                        case 2: variant_desc = "2 (GLV Endomorphism (β² x, y))"; break;
                        case 3: variant_desc = "3 (Point Negation (x, -y))"; break;
                        case 4: variant_desc = "4 (GLV Negation (β x, -y))"; break;
                        case 5: variant_desc = "5 (GLV Negation (β² x, -y))"; break;
                        default: variant_desc = std::to_string(variant); break;
                    }

                    match_count++;
                    const std::string found_utc = utc_now_iso8601();
                    std::string item_path = output_dir.empty()
                        ? ("split-result-0x" + addr_str + ".txt")
                        : (output_dir + "/split-result-0x" + addr_str + ".txt");

                    if (!output_dir.empty()) {
                        std::string write_error;
                        if (!vaddro::ensure_output_directory(output_dir, write_error)) {
                            std::cerr << "FATAL: " << write_error << std::endl;
                            result_write_failed = true;
                            stop_reason = "result_write_failed";
                            g_running = 0;
                            break;
                        }
                        std::ostringstream result_content;
                        result_content << "Split-Key Search Result v1\n"
                                       << "Address: 0x" << addr_str << "\n"
                                       << "Seed Public Key: " << pubkey_str << "\n"
                                       << "Search Share: " << share_str << "\n"
                                       << "Variant: " << variant << "\n"
                                       << "Worker ID: " << audit.worker_id << "\n"
                                       << "Run ID: " << audit.run_id << "\n"
                                       << "Version: " << VADDRO_VERSION << "\n"
                                       << "Build Commit: " << audit.build_commit << "\n"
                                       << "GPU: " << audit.gpu_name << "\n"
                                       << "GPU Index: " << audit.gpu_index << "\n"
                                       << "Shard Index: " << audit.shard_index << "\n"
                                       << "Shard Count: " << audit.shard_count << "\n"
                                       << "NVIDIA Driver: " << audit.driver_version << "\n"
                                       << "Started UTC: " << search_started_utc << "\n"
                                       << "Found UTC: " << found_utc << "\n";
                        if (target.mode == 1) {
                            result_content << "Pattern: " << pattern_str << "\n";
                        } else {
                            result_content << "Pattern: same-char-ends:" << target.same_char_ends << "\n";
                        }
                        result_content << "Scanner: Native CUDA + PTX\n"
                                       << "OS: Linux\n";

                        std::string res_path = output_dir + "/result-share.txt";
                        if (!write_text_atomic(item_path, result_content.str(), write_error) ||
                            !write_text_atomic(res_path, result_content.str(), write_error)) {
                            std::cerr << "FATAL: " << write_error << std::endl;
                            result_write_failed = true;
                            stop_reason = "result_write_failed";
                            g_running = 0;
                            break;
                        }
                        std::string all_path = output_dir + "/all-matches.txt";
                        std::ofstream all_out(all_path, std::ios::app);
                        if (all_out.is_open()) {
                            all_out << "0x" << addr_str << " 0x" << share_str << "\n";
                        }
                    }

                    if (!append_json_event(json_log_path, "hit", audit,
                                           {{"address", "0x" + addr_str},
                                            {"found_utc", found_utc},
                                            {"result_path", item_path}},
                                           {{"hit_count", std::to_string(match_count)},
                                            {"max_hits", std::to_string(max_hits)},
                                            {"variant", std::to_string(variant)}}) &&
                        !json_log_warning_emitted) {
                        std::cerr << "WARNING: could not append to JSONL log: " << json_log_path << std::endl;
                        json_log_warning_emitted = true;
                    }

                    std::string match_border = "+" + std::string(78, '=') + "+";
                    if (is_interactive_tty) {
                        std::cout << "\n";
                    }
                    std::cout << match_border << "\n"
                              << "| " << pad_exact("[*] [VADDRO] VERIFIED VANITY MATCH #" + std::to_string(match_count) + " FOUND!", 76) << " |\n"
                              << match_border << "\n"
                              << "| " << pad_exact("  Matched Address : 0x" + addr_str, 76) << " |\n"
                              << "| " << pad_exact("  Search Share    : 0x" + share_str, 76) << " |\n"
                              << "| " << pad_exact("  GLV Variant     : " + variant_desc, 76) << " |\n"
                              << "| " << pad_exact("  Result Saved To : " + item_path, 76) << " |\n"
                              << "| " << pad_exact("  Hit Progress    : " + format_hit_progress(match_count, max_hits), 76) << " |\n"
                              << "| " << pad_exact("  Security        : Verified share; combine only on the safe admin machine.", 76) << " |\n"
                              << match_border << "\n" << std::flush;

                    if (max_hits != 0 && match_count >= max_hits) {
                        std::cout << "[VADDRO] Hit limit reached (" << match_count << "/"
                                  << max_hits << "). Stopping cleanly." << std::endl;
                        stop_reason = "hit_limit_reached";
                        g_running = 0;
                        break;
                    }
                }
            }
        }
    }

        // Periodic Status Update (Every 20 rounds ~ 200ms)
        if (round % 20 == 0 || !g_running) {
            cuda_check(cudaDeviceSynchronize(), "periodic device synchronize");

            if (canary_interval != 0 && round >= next_canary_round && g_running) {
                const uint64_t canary_index = canary_checks++;
                const int sample_stream = static_cast<int>(canary_index % num_streams);
                const uint32_t sample_gid = static_cast<uint32_t>(
                    (canary_index * 0x9E3779B97F4A7C15ULL) % stream_points
                );
                const uint64_t global_share_id =
                    static_cast<uint64_t>(sample_stream) * stream_points + sample_gid;
                ScalarWords64 expected_share = reconstruct_share(seed_u64, round + 1, global_share_id);

                state_address_kernel<<<1, 1>>>(d_delta_x[sample_stream], d_prev_lambda[sample_stream],
                                                sample_gid, d_state_addr);
                cuda_check(cudaGetLastError(), "state_address_kernel launch");
                verify_scalar_kernel<<<1, 1>>>(d_precomp, base_point,
                                               expected_share.s0, expected_share.s1,
                                               expected_share.s2, expected_share.s3,
                                               0, d_verify_addr);
                cuda_check(cudaGetLastError(), "canary verify_scalar_kernel launch");

                uint32_t state_address[5], exact_address[5];
                cuda_check(cudaMemcpy(state_address, d_state_addr, sizeof(state_address),
                                      cudaMemcpyDeviceToHost), "copy canary state address");
                cuda_check(cudaMemcpy(exact_address, d_verify_addr, sizeof(exact_address),
                                      cudaMemcpyDeviceToHost), "copy canary exact address");
                if (format_address(state_address) != format_address(exact_address)) {
                    std::cerr << "\nFATAL: recurrence canary mismatch at round " << round
                              << ", stream " << sample_stream << ", gid " << sample_gid
                              << ". Search stopped to prevent silent invalid work." << std::endl;
                    canary_failed = true;
                    stop_reason = "canary_mismatch";
                    g_running = 0;
                } else {
                    last_canary_utc = utc_now_iso8601();
                }
                if (!append_json_event(json_log_path, "canary", audit,
                                       {{"status", canary_failed ? "failed" : "ok"},
                                        {"last_canary_utc", last_canary_utc}},
                                       {{"round", std::to_string(round)},
                                        {"checks", std::to_string(canary_checks)}}) &&
                    !json_log_warning_emitted) {
                    std::cerr << "WARNING: could not append to JSONL log: " << json_log_path << std::endl;
                    json_log_warning_emitted = true;
                }
                next_canary_round = round + canary_interval;
            }

            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            if (elapsed > 0.0) {
                const uint64_t effective_hashes = total_hashes * (target.mode == 0 ? 1ULL : 6ULL);
                double speed_mhs = (double)effective_hashes / (elapsed * 1000000.0);

                std::string c1 = pad_exact(" " + format_hashrate(speed_mhs), 19);
                std::string c2 = pad_exact(format_key_count(effective_hashes), 22);
                std::string c3 = pad_exact(format_time_hms(elapsed), 20);
                std::string c4 = pad_exact(format_hit_progress(match_count, max_hits), 15);

                if (is_interactive_tty) {
                    std::cout << "\r| " << pad_exact(c1 + c2 + c3 + c4, 76) << " |" << std::flush;
                } else {
                    if (round % 1000 == 0 || !g_running) {
                        std::cout << "[VADDRO] [*] Speed: " << format_hashrate(speed_mhs)
                                  << " | Total: " << format_key_count(effective_hashes)
                                  << " | Time: " << format_time_hms(elapsed)
                                  << " | Hits: " << format_hit_progress(match_count, max_hits) << std::endl;
                    }
                }
                if ((round % 1000 == 0 || !g_running) &&
                    !append_json_event(json_log_path, "status", audit,
                                       {{"hashrate", format_hashrate(speed_mhs)},
                                        {"elapsed", format_time_hms(elapsed)},
                                        {"hit_progress", format_hit_progress(match_count, max_hits)},
                                        {"last_canary_utc", last_canary_utc}},
                                       {{"speed_mhs", std::to_string(speed_mhs)},
                                        {"effective_keys", std::to_string(effective_hashes)},
                                        {"elapsed_seconds", std::to_string(elapsed)},
                                        {"hits", std::to_string(match_count)},
                                        {"max_hits", std::to_string(max_hits)},
                                        {"canary_checks", std::to_string(canary_checks)}}) &&
                    !json_log_warning_emitted) {
                    std::cerr << "WARNING: could not append to JSONL log: " << json_log_path << std::endl;
                    json_log_warning_emitted = true;
                }
            }
        }

        if (benchmark_mode && round >= benchmark_rounds) {
            stop_reason = "benchmark_complete";
            break;
        }
    }

    if (is_interactive_tty) {
        std::cout << "\n" << border_line << "\n";
    }

    cuda_check(cudaDeviceSynchronize(), "final device synchronize");
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time).count();
    const uint64_t effective_total = total_hashes * (target.mode == 0 ? 1ULL : 6ULL);
    if (stop_reason == "running") stop_reason = "signal";
    const int final_exit_code = canary_failed ? 3 : (result_write_failed ? 4 : 0);
    if (!append_json_event(json_log_path, "completed", audit,
                           {{"reason", stop_reason},
                            {"elapsed", format_time_hms(elapsed)},
                            {"last_canary_utc", last_canary_utc}},
                           {{"exit_code", std::to_string(final_exit_code)},
                            {"effective_keys", std::to_string(effective_total)},
                            {"elapsed_seconds", std::to_string(elapsed)},
                            {"hits", std::to_string(match_count)},
                            {"canary_checks", std::to_string(canary_checks)}}) &&
        !json_log_warning_emitted) {
        std::cerr << "WARNING: could not append final JSONL event: " << json_log_path << std::endl;
    }
    std::cout << "\n[VADDRO] Search completed. Effective keys explored: " << effective_total << " in " << elapsed << "s\n" << std::endl;
    if (profile_mode && profile_rounds > 0) {
        std::cout << "PROFILE inverse_total_ms=" << (profile_inverse_seconds * 1000.0)
                  << " iterate_total_ms=" << (profile_iterate_seconds * 1000.0)
                  << " rounds=" << profile_rounds
                  << " inverse_avg_ms=" << ((profile_inverse_seconds * 1000.0) / profile_rounds)
                  << " iterate_avg_ms=" << ((profile_iterate_seconds * 1000.0) / profile_rounds)
                  << " point_basis_avg_ms=" << ((profile_point_basis_seconds * 1000.0) / profile_rounds)
                  << " variant_hash_avg_ms=" << ((profile_variant_hash_seconds * 1000.0) / profile_rounds)
                  << std::endl;
    }

    for (int s = 0; s < num_streams_int; ++s) {
        cudaFree(d_delta_x[s]);
        cudaFree(d_inverse[s]);
        cudaFree(d_prev_lambda[s]);
#if VADDRO_SPLIT_VARIANT_HASH
        if (d_basis_x[s] != nullptr) cudaFree(d_basis_x[s]);
        if (d_basis_beta_x[s] != nullptr) cudaFree(d_basis_beta_x[s]);
        if (d_basis_beta2_x[s] != nullptr) cudaFree(d_basis_beta2_x[s]);
        if (d_basis_y[s] != nullptr) cudaFree(d_basis_y[s]);
        if (d_basis_neg_y[s] != nullptr) cudaFree(d_basis_neg_y[s]);
#endif
        cudaFree(d_results[s]);
        cudaFree(d_result_count[s]);
        cuda_check(cudaStreamDestroy(streams[s]), "cudaStreamDestroy");
    }
    cudaFree(d_precomp);
    cudaFree(d_verify_addr);
    cudaFree(d_state_addr);

    return final_exit_code;
}
