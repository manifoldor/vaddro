# Vaddro

> **Vanity Address Online — Seed Offline. Compute Online.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/manifoldor/vaddro/actions/workflows/ci.yml/badge.svg)](https://github.com/manifoldor/vaddro/actions/workflows/ci.yml)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
![GPU](https://img.shields.io/badge/Optimized_for-RTX_5090-black?logo=nvidia)

[English](README.md) · [简体中文](README_CN.md)

**Vaddro (Vanity Address Online)** is a native CUDA/PTX engine for generating custom EVM addresses. Its Split-Key workflow keeps the Seed private scalar offline while connected GPU workers search from the corresponding secp256k1 public point and return a verified additive Search Share. The final private key is reconstructed and checked offline

> **RTX 5090 measured performance:** **5.20–5.26 GH/s** effective address checks, **3.45 GH/s** raw point generation, and approximately **5.1 GiB** working GPU memory with the production configuration.

## Why Vaddro

| Capability | Implementation | Practical benefit |
| --- | --- | --- |
| Split-key workflow | The worker searches from a public secp256k1 point and outputs a share | The seed private scalar never needs to be copied to a GPU worker |
| Native GPU engine | CUDA C++ with hand-tuned PTX carry chains and modular arithmetic | High throughput without OpenCL or third-party runtime dependencies |
| Six candidates per EC state | GLV endomorphism and point negation produce six address variants | More address checks per expensive elliptic-curve iteration |
| 255-point batch inversion | One field inversion is amortized across each 255-point batch | Replaces repeated inversions with cheaper field multiplications |
| Pipelined execution | Eight asynchronous CUDA streams with vectorized memory access | Keeps the GPU busy across inversion and iteration workloads |
| Layered verification | Build checks, GPU self-test, runtime canary, hit reconstruction, public verifier | Detects corrupted tables, arithmetic faults, state drift, and invalid results |
| Multi-worker operations | Deterministic Search Share shards, per-GPU locks, exact PID ownership, isolated logs | Prevents base-scan overlap and keeps one or many workers observable |
| Portable offline tools | `seed.py`, `verify_result.py`, and `combine.py` use the Python standard library | Seed generation, public verification, and final combination work across macOS, Linux, and Windows |

## RTX 5090 performance

The recorded qualification used the following environment:

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090, 32 GB |
| Build target | Blackwell `sm_120` |
| Driver / CUDA compiler | 595.71.05 / 12.8.93 |
| Kernel configuration | `WORK_SIZE=224`, `INVERSE_MULTIPLE=32768`, `INVERSE_SIZE=255`, `STREAMS=8` |
| Raw point generation | approximately **3.45 GH/s** |
| Effective six-variant search | approximately **5.20–5.26 GH/s** |
| Working GPU memory | approximately **5,106 MiB** |

“Raw point generation” measures elliptic-curve states. “Effective search” measures address candidates after Keccak-256 and pattern evaluation across the six GLV/negation variants. The effective number is the production-search metric; it is measured rather than inferred by multiplying the raw benchmark.

Four alternating same-source A/B runs measured a **5.096 GH/s** median with GLV X reuse disabled and **5.219 GH/s** with it enabled, a **2.41%** improvement. A 178-second stability run measured **5.222 GH/s** with all periodic canaries passing.

As an illustrative scale-out example—not a required cluster size or a description of the release environment—four equivalent RTX 5090 workers would provide approximately **20.8–21.0 GH/s** aggregate at the measured range. For a target with 13 fixed hexadecimal nibbles (`16^-13 = 2^-52`), that corresponds to a theoretical mean of roughly **59.5–60.1 hours** and a median of roughly **41.2–41.7 hours**. Search time is probabilistic; hardware power limits, cooling, clocks, and target shape affect observed throughput.

Reproduce the raw benchmark on your worker:

~~~bash
./benchmark.sh
~~~

## Split-key architecture

~~~text
Offline machine                                      GPU worker
---------------                                      ----------
seed.py
  -> seed private scalar (kept offline)
  -> seed public point  ---------------------------> public point + search rule
                                                     CUDA/PTX search
                                                     verified search share
verify_result.py  <--------------------------------- result + public metadata
combine.py
  -> seed private scalar + verified share
  -> final private key
  -> independently derived EVM address
~~~

Let the offline seed be `k_seed` and its public point be:

~~~text
P_seed = k_seed · G
~~~

The worker searches additive shares `s` from the public state:

~~~text
P_candidate = P_seed + s · G
~~~

When a transformed candidate matches the requested address pattern, the worker records the share and its GLV/negation variant. The offline combiner applies the same variant to the scalar path, derives the final address independently, and writes the final wallet only after the result matches.

## Deterministic multi-worker sharding

Multiple GPU workers should share one Seed Public Key and divide the base Search Share namespace with deterministic shards. Set the same power-of-two `SHARD_COUNT` on every worker and assign each worker a unique zero-based `SHARD_INDEX`. The following four-worker layout is only a configuration example:

| Worker | `SHARD_INDEX` | `SHARD_COUNT` |
| --- | ---: | ---: |
| worker-01 | `0` | `4` |
| worker-02 | `1` | `4` |
| worker-03 | `2` | `4` |
| worker-04 | `3` | `4` |

The shard mapping partitions the 48-bit high-limb namespace into disjoint contiguous regions and reserves enough headroom for all lane offsets and round-counter carry. This prevents active workers with unique shard indices from scanning overlapping base Search Share regions. A separate public key per worker is unnecessary and would create additional offline Seed material to manage.

`SHARD_COUNT` must be a power of two from `1` to `65536`; `SHARD_INDEX` must be in `0..SHARD_COUNT-1`. Keep both values stable for the lifetime of a search task: changing the count repartitions the namespace. A restart selects a new random starting point inside the assigned shard, so sharding coordinates concurrent workers but is not a persistent checkpoint system.

The startup console and `05_status.sh` display the active shard. The same fields are written to run state, JSONL events, and result files. Public verification checks the declared shard range when metadata is present. GLV/negation transformations occur after the base partition; an equivalent effective candidate appearing across shards remains theoretically possible, but only with cryptographically negligible probability.

## CUDA engine design

### secp256k1 arithmetic

The engine implements 256-bit field arithmetic, Montgomery-style batch inversion, affine point iteration, and scalar-based verification directly in CUDA. Critical multiplication, reduction, carry propagation, and Keccak-256 paths use CUDA intrinsics and PTX operations such as `lop3.b32`, `mad.lo.cc.u32`, and `madc.hi.u32`.

### Six-way GLV search

Each elliptic-curve state is transformed into six equivalent address candidates:

~~~text
0: (x, y)                  3: (x, -y)
1: (βx, y)                 4: (βx, -y)
2: (β²x, y)                5: (β²x, -y)
~~~

This expands address coverage without performing six independent point-iteration pipelines.

Each positive point and its negation share the same X coordinate. The default build evaluates `(0,3)`, `(1,4)`, and `(2,5)` as pairs, retaining only the current X mapping so `βx` and `β²x` each require one field multiplication. Variant numbering, derived addresses, and the offline verification protocol are unchanged. Run `make ARCH=sm_120 ab-build` to create reuse-on and reuse-off binaries from identical sources, or `make ARCH=sm_120 ab-self-test` to build and verify both; the real gain must be established with production-search A/B measurements on an RTX 5090.

### Multi-stream pipeline

The production configuration divides the search state across eight non-blocking CUDA streams. Each stream alternates batched inversion and point iteration while vectorized `uint4` loads and stores reduce memory-transaction overhead.

### Fast matching

Exact 40-nibble templates support fixed hexadecimal characters and `X`/`x`/`.` wildcards. A prefix mask rejects most candidates early before the full pattern comparison. A second mode searches addresses whose first and last `N` hexadecimal characters are the same.

## Verification pipeline

| Stage | Check |
| --- | --- |
| Package/build | Verifies the SHA-256 of the embedded table and independently validates all 8,160 secp256k1 precomputation points |
| GPU startup | Runs known-answer tests for curve membership, Keccak-256, multiplication, squaring, inversion, and all six variants |
| Live search | Periodically rebuilds a sampled recurrence lane through an independent scalar path |
| Hit handling | Reconstructs every reported share and recomputes its address before writing a result |
| Public verification | `verify_result.py` validates the result using public data only |
| Offline combination | `combine.py` checks seed/public-point consistency, combines the share, and derives the EVM address again |

Recorded `memcheck`, `initcheck`, `racecheck`, and `synccheck` runs cover the self-test and a real synthetic-hit path.

## Requirements

For the optimized RTX 5090 path:

- Linux x86_64, tested on Ubuntu 22.04
- NVIDIA RTX 5090 or another Blackwell GPU
- NVIDIA driver compatible with CUDA 12.8
- CUDA Toolkit 12.8 or newer
- Bash, GNU Make, and Python 3

The engine also supports RTX 4090 (`sm_89`) and RTX 3090 (`sm_86`). Both paths can be compiled and run, but their functional, performance, and long-duration validation is currently less complete than the RTX 5090 path. Published qualification and performance figures in this document apply to RTX 5090 / `sm_120` unless stated otherwise. The Makefile also contains targets for `sm_75`, `sm_80`, and `sm_90`.

## Quick start

### 1. Get the source

~~~bash
git clone https://github.com/manifoldor/vaddro.git
cd vaddro
~~~

### 2. Generate the seed offline

Run this on the machine that will retain the seed private scalar:

~~~bash
python3 seed.py
~~~

The command writes `seeds/split-seed.txt` and prints the 128-hex-character public point used by GPU workers.

### 3. Configure a GPU worker

~~~bash
cp config/miner.env.example config/miner.env
~~~

Edit `config/miner.env` and set the public point plus exactly one search mode:

~~~ini
PUBLIC_KEY="YOUR_128_HEX_SECP256K1_PUBLIC_POINT"

# Exact 40-nibble pattern; X/x/. are wildcards.
PATTERN="0000000000000XXXXXXXXXXXXXXXXXXXXXXXXXXX"
SAME_CHAR_ENDS=0

# Stop after the first verified hit.
MAX_HITS=1

WORKER_ID="worker-01"
SHARD_INDEX=0
SHARD_COUNT=1
GPU_INDEX=0
~~~

Or search for the same character at both ends:

~~~ini
PATTERN=""
SAME_CHAR_ENDS=6
~~~

### 4. Diagnose, build, and start

~~~bash
./00_doctor.sh
./01_build.sh
./02_start_search.sh
~~~

The build script detects the installed GPU and selects a native architecture target. On Blackwell it compiles `sm_120` and immediately runs the GPU cryptographic self-test.

### 5. Monitor and control

~~~bash
./05_status.sh
./03_check_result.sh
./04_stop_search.sh --gpu 0
~~~

`02_start_search.sh` enforces one Vaddro process per GPU. Verified results use a flat output directory, while each run receives a unique ID and isolated structured logs:

~~~text
result/
logs/<worker-id>/<run-id>/events.jsonl
~~~

The main console shows the version, source revision, worker/run identity, active shard, search mode, output paths, GPU/driver details, kernel configuration, canary interval, hashrate, explored candidates, elapsed time, and hit progress.

## Verify and combine a result

On any machine with the public result:

~~~bash
python3 verify_result.py result/split-result-0x<address>.txt
~~~

After public verification, copy the result to the offline machine and combine it with the seed:

~~~bash
python3 combine.py \
  seeds/split-seed.txt \
  split-result-0x<address>.txt \
  wallets/wallet.txt
~~~

The combiner verifies the seed public point, share, variant, and final EVM address before atomically writing a mode-`0600` wallet file with an EIP-55 checksum address.

## Search modes and hit policy

| Setting | Meaning |
| --- | --- |
| `PATTERN` | Exact 40-nibble template with hexadecimal characters and `X`/`x`/`.` wildcards |
| `SAME_CHAR_ENDS=N` | Match any address whose first and last `N` nibbles are the same character (`1..8`) |
| `MAX_HITS=0` | Continue until manually stopped |
| `MAX_HITS=1` | Stop after the first verified result |
| `MAX_HITS=N` | Stop after exactly `N` verified results |

## Repository guide

- [DEPLOYMENT.md](DEPLOYMENT.md) / [DEPLOYMENT_CN.md](DEPLOYMENT_CN.md) — English and Chinese deployment guides
- [SECURITY.md](SECURITY.md) / [SECURITY_CN.md](SECURITY_CN.md) — English and Chinese security policies
- [CHANGELOG.md](CHANGELOG.md) — release history

## Contact

General contact: yishan ([linyishan@gmail.com](mailto:linyishan@gmail.com)). For security vulnerabilities, use the private reporting channels in [SECURITY.md](SECURITY.md).

## Buy me a coffee

If Vaddro is useful to you, you can support its continued development through this EVM address:

~~~text
0x00000000000003eb5045a0868cdb96a9926e2328
~~~

## License

[MIT](LICENSE) © 2026 yishan.
