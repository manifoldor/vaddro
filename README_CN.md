# Vaddro

> **Vanity Address Online —— Seed 离线，算力在线。**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/manifoldor/vaddro/actions/workflows/ci.yml/badge.svg)](https://github.com/manifoldor/vaddro/actions/workflows/ci.yml)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
![GPU](https://img.shields.io/badge/Optimized_for-RTX_5090-black?logo=nvidia)

[English](README.md) · [简体中文](README_CN.md)

**Vaddro（Vanity Address Online）**使用原生 CUDA/PTX 搜索定制 EVM 地址。它采用 Split-Key 工作流：Seed 私有标量始终留在离线端，联网 GPU Worker 基于对应的 secp256k1 公钥点搜索并返回经过验证的加法 Search Share，最终私钥只在离线环境中合并和校验。

> **RTX 5090 实测性能：5.20–5.26 GH/s** 六变体有效地址检查、**3.45 GH/s** 原始点生成，生产参数下工作显存约 **5.1 GiB**。

## 为什么选择 Vaddro

| 能力 | 实现 | 实际价值 |
| --- | --- | --- |
| Split-Key 工作流 | 算力端基于 secp256k1 公钥点搜索并输出 Share | Seed 私有标量无需复制到 GPU 算力机 |
| 原生 GPU 引擎 | CUDA C++、手写 PTX 进位链与模运算 | 不依赖 OpenCL 或第三方 GPU 运行时即可获得高吞吐 |
| 每个 EC 状态检查 6 个候选 | GLV 自同态结合点取负生成六个地址变体 | 一次昂贵的椭圆曲线迭代覆盖更多地址 |
| 255 点批量求逆 | 每组 255 个点共享一次有限域求逆 | 用成本更低的有限域乘法替代大量独立求逆 |
| 流水线执行 | 8 个异步 CUDA Stream 与向量化内存访问 | 让批量求逆和点迭代持续占满 GPU |
| 分层正确性校验 | 构建检查、GPU 自检、运行 canary、命中复算、公开验证 | 及时发现表损坏、运算错误、状态漂移和无效结果 |
| 多机运维能力 | 确定性 Search Share 分片、GPU 单实例锁、精确 PID、独立日志 | 避免基础扫描重叠，并稳定管理单机或多台 GPU Worker |
| 跨平台离线工具 | `seed.py`、`verify_result.py`、`combine.py` 仅使用 Python 标准库 | macOS、Linux、Windows 均可生成 Seed、验 Share 和合并钱包 |

## RTX 5090 实测性能

实机验证环境如下：

| 项目 | 配置或结果 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090，32 GB |
| 编译目标 | Blackwell `sm_120` |
| 驱动 / CUDA 编译器 | 595.71.05 / 12.8.93 |
| 内核参数 | `WORK_SIZE=224`、`INVERSE_MULTIPLE=32768`、`INVERSE_SIZE=255`、`STREAMS=8` |
| 原始点生成 | 约 **3.45 GH/s** |
| 六变体有效搜索 | 约 **5.20–5.26 GH/s** |
| 工作显存 | 约 **5,106 MiB** |

“原始点生成”统计椭圆曲线状态；“有效搜索”统计经过六种 GLV/取负变换、Keccak-256 和 Pattern 判断后的实际地址候选。生产搜索应看有效搜索速度，它是实测值，不是简单把原始 Benchmark 乘以 6 得出的理论数字。

GLV X 复用使用同源开关在单张 5090 上交替测试四轮：关闭版中位数为 **5.096 GH/s**，开启版为 **5.219 GH/s**，提升 **2.41%**；178 秒长稳测试为 **5.222 GH/s**，周期 canary 全部通过。

作为扩展能力的算例——并非固定集群规模，也不代表项目正式发布环境——四台同规格 RTX 5090 按实测区间可得到约 **20.8–21.0 GH/s** 聚合算力。搜索 13 个固定十六进制字符时，单次命中概率为 `16^-13 = 2^-52`，对应理论平均等待约 **59.5–60.1 小时**、中位数约 **41.2–41.7 小时**。实际搜索具有随机性，功耗限制、散热、频率和目标形态都会影响速度。

在算力机上复现原始性能：

~~~bash
./benchmark.sh
~~~

## Split-Key 架构

~~~text
离线机                                             GPU 算力机
------                                             ----------
seed.py
  -> Seed 私有标量（留在离线机）
  -> Seed 公钥点  -------------------------------> 公钥点 + 搜索规则
                                                    CUDA/PTX 搜索
                                                    已验证 Search Share
verify_result.py  <------------------------------- 结果 + 公开元数据
combine.py
  -> Seed 私有标量 + 已验证 Share
  -> 最终私钥
  -> 独立推导 EVM 地址
~~~

设离线 Seed 为 `k_seed`，对应公钥点为：

~~~text
P_seed = k_seed · G
~~~

算力端从公开状态搜索加法 Share `s`：

~~~text
P_candidate = P_seed + s · G
~~~

当某个变体命中地址规则时，Worker 记录 Share 及其 GLV/取负 Variant。离线合并器在标量路径应用同一 Variant，独立推导最终地址，只有结果一致时才写出最终钱包。

## 多 Worker 确定性分片

多台 GPU Worker 应共用同一个 Seed 公钥，并通过确定性分片划分基础 Search Share 命名空间。所有 Worker 设置相同且为 2 的幂的 `SHARD_COUNT`，每台分配唯一的零基 `SHARD_INDEX`。下方四 Worker 配置仅作为分片方法示例：

| Worker | `SHARD_INDEX` | `SHARD_COUNT` |
| --- | ---: | ---: |
| 算力机 1 | `0` | `4` |
| 算力机 2 | `1` | `4` |
| 算力机 3 | `2` | `4` |
| 算力机 4 | `3` | `4` |

分片会把 48 位高 limb 命名空间确定性地划分成互不相交的连续区间，并为全部 lane 偏移和 round 计数低位进位预留空间。因此，使用唯一分片索引的在线 Worker 不会扫描重叠的基础 Search Share 区域。无需为每台 Worker 单独生成公钥，否则只会增加离线 Seed 的管理负担。

`SHARD_COUNT` 必须是 `1..65536` 范围内的 2 的幂；`SHARD_INDEX` 必须位于 `0..SHARD_COUNT-1`。同一搜索任务期间应保持这两个值不变，因为修改总分片数会重新划分命名空间。Worker 重启时会在所属分片内重新选择随机起点，因此分片负责协调并发 Worker，但不等同于持久化断点续扫。

启动主界面和 `05_status.sh` 都会显示当前分片，Run State、JSONL 事件和结果文件也会记录相同字段；公开验证器会在结果包含这些元数据时检查分片范围。GLV/取负变换发生在基础空间分片之后，跨分片出现同一有效候选在理论上仍可能发生，但概率在密码学意义上可忽略。

## CUDA 引擎设计

### secp256k1 原生运算

引擎在 CUDA 中直接实现 256 位有限域运算、Montgomery 风格批量求逆、仿射点迭代和基于标量的独立验证。乘法、约减、进位传播与 Keccak-256 热点使用 CUDA Intrinsic 和 `lop3.b32`、`mad.lo.cc.u32`、`madc.hi.u32` 等 PTX 指令。

### GLV 六变体搜索

每个椭圆曲线状态会变换为六个地址候选：

~~~text
0: (x, y)                  3: (x, -y)
1: (βx, y)                 4: (βx, -y)
2: (β²x, y)                5: (β²x, -y)
~~~

这让同一条点迭代流水线覆盖六种地址变体，而无需执行六套独立的椭圆曲线迭代。

正点与对应负点共享相同的 X 坐标。默认构建按 `(0,3)`、`(1,4)`、`(2,5)` 成对计算，只生成并保留当前一组 X 映射，使 `βx` 和 `β²x` 各只进行一次有限域乘法。Variant 编号、地址结果与离线验证协议保持不变。可用 `make ARCH=sm_120 ab-build` 从同一源码生成开启/关闭复用的两个二进制，或以 `make ARCH=sm_120 ab-self-test` 构建并验证两版；实际提升以 RTX 5090 上的生产搜索 A/B 数据为准。

### 多 Stream 流水线

生产参数把搜索状态分配到 8 个非阻塞 CUDA Stream。每个 Stream 交替执行批量求逆与点迭代，`uint4` 向量化加载/存储减少显存事务开销。

### 快速 Pattern 匹配

40 位精确模板支持十六进制字符和 `X`/`x`/`.` 通配符。前缀掩码会优先排除绝大部分不可能命中的候选，再进入完整 Pattern 比较。另一个模式可搜索地址首尾 `N` 位均为同一个字符的结果。

## 六层验证链

| 阶段 | 校验内容 |
| --- | --- |
| 发布包与构建 | 校验内嵌表 SHA-256，并独立验证全部 8,160 个 secp256k1 预计算点 |
| GPU 启动 | 检查曲线成员、Keccak-256、乘法、平方、求逆和六种 Variant 已知答案 |
| 实时搜索 | 周期性抽样 recurrence lane，并从独立标量路径重建比较 |
| 命中处理 | 写出结果前重建 Share 并重新计算地址 |
| 公开验证 | `verify_result.py` 仅使用公开数据验证 Worker 结果 |
| 离线合并 | `combine.py` 检查 Seed/公钥点一致性、合并 Share，并再次推导 EVM 地址 |

已记录的 `memcheck`、`initcheck`、`racecheck` 和 `synccheck` 覆盖 GPU 自检和真实合成命中路径。

## 环境要求

RTX 5090 优化路径需要：

- Linux x86_64，实测 Ubuntu 22.04
- NVIDIA RTX 5090 或其他 Blackwell GPU
- 与 CUDA 12.8 兼容的 NVIDIA 驱动
- CUDA Toolkit 12.8 或更高版本
- Bash、GNU Make、Python 3

Vaddro 同时支持 RTX 4090（`sm_89`）和 RTX 3090（`sm_86`），两条路径均可编译运行，但功能、性能和长时间运行验证目前不如 RTX 5090 充分。除非另有说明，本文公布的资格验证和性能数据仅适用于 RTX 5090 / `sm_120`。Makefile 还包含 `sm_75`、`sm_80` 和 `sm_90` 目标。

## 快速开始

### 1. 获取源码

~~~bash
git clone https://github.com/manifoldor/vaddro.git
cd vaddro
~~~

### 2. 在离线机生成 Seed

在负责保管 Seed 私有标量的机器上执行：

~~~bash
python3 seed.py
~~~

命令会写入 `seeds/split-seed.txt`，并显示需要交给 GPU Worker 的 128 位十六进制公钥点。

### 3. 配置 GPU 算力机

~~~bash
cp config/miner.env.example config/miner.env
~~~

编辑 `config/miner.env`，填入公钥点，并且只选择一种搜索模式：

~~~ini
PUBLIC_KEY="YOUR_128_HEX_SECP256K1_PUBLIC_POINT"

# 40 位精确模板；X/x/. 表示通配符。
PATTERN="0000000000000XXXXXXXXXXXXXXXXXXXXXXXXXXX"
SAME_CHAR_ENDS=0

# 第一个有效命中后停止。
MAX_HITS=1

WORKER_ID="worker-01"
SHARD_INDEX=0
SHARD_COUNT=1
GPU_INDEX=0
~~~

或搜索首尾相同字符：

~~~ini
PATTERN=""
SAME_CHAR_ENDS=6
~~~

### 4. 环境检查、编译和启动

~~~bash
./00_doctor.sh
./01_build.sh
./02_start_search.sh
~~~

构建脚本会识别 GPU 并选择原生架构目标；Blackwell 会编译为 `sm_120`，随后立即执行 GPU 密码学自检。

### 5. 查看状态与控制任务

~~~bash
./05_status.sh
./03_check_result.sh
./04_stop_search.sh --gpu 0
~~~

`02_start_search.sh` 会确保每张 GPU 同时只有一个 Vaddro 进程。已验证结果直接写入扁平目录，每次运行仍拥有唯一 Run ID 和独立结构化日志：

~~~text
result/
logs/<worker-id>/<run-id>/events.jsonl
~~~

主界面会显示版本、源码 Revision、Worker/Run 身份、当前 Shard、搜索模式、输出目录、GPU/驱动、内核参数、canary 周期、实时算力、累计候选、运行时间和命中进度。

## 验证并合并结果

在任意能够读取公开结果文件的机器上执行：

~~~bash
python3 verify_result.py result/split-result-0x<address>.txt
~~~

公开验证通过后，将结果复制到离线机，与 Seed 合并：

~~~bash
python3 combine.py \
  seeds/split-seed.txt \
  split-result-0x<address>.txt \
  wallets/wallet.txt
~~~

合并器会验证 Seed 公钥点、Share、Variant 和最终 EVM 地址，然后以原子方式写入权限为 `0600` 的钱包文件，并输出 EIP-55 Checksum 地址。

## 搜索模式与命中策略

| 配置 | 含义 |
| --- | --- |
| `PATTERN` | 40 位精确模板，支持十六进制字符和 `X`/`x`/`.` 通配符 |
| `SAME_CHAR_ENDS=N` | 搜索首尾 `N` 位均为同一个字符的地址（`1..8`） |
| `MAX_HITS=0` | 持续搜索，直到手动停止 |
| `MAX_HITS=1` | 第一个已验证结果后停止 |
| `MAX_HITS=N` | 累计恰好 `N` 个已验证结果后停止 |

## 项目文档

- [DEPLOYMENT_CN.md](DEPLOYMENT_CN.md) / [DEPLOYMENT.md](DEPLOYMENT.md)：中英文部署指南
- [SECURITY_CN.md](SECURITY_CN.md) / [SECURITY.md](SECURITY.md)：中英文安全策略
- [CHANGELOG.md](CHANGELOG.md)：版本变更记录

## 联系方式

普通联系：yishan（[linyishan@gmail.com](mailto:linyishan@gmail.com)）。安全漏洞请使用 [SECURITY_CN.md](SECURITY_CN.md) 中的私密报告渠道。

## Buy me a coffee

如果 Vaddro 对你有帮助，可以通过以下 EVM 地址支持项目持续开发：

~~~text
0x00000000000003eb5045a0868cdb96a9926e2328
~~~

## 许可证

[MIT](LICENSE) © 2026 yishan。
