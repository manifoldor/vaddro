# Vaddro 部署指南

[English](DEPLOYMENT.md)

Vaddro 将 Seed 私有标量留在离线端，由 Linux GPU 算力机在线搜索公开的 Search Share。当前重点优化的生产平台是 NVIDIA RTX 5090 / Blackwell `sm_120`。

## 环境要求

- Linux x86_64，已在 Ubuntu 22.04 验证
- NVIDIA RTX 5090（`sm_120`）、RTX 4090（`sm_89`）或 RTX 3090（`sm_86`）
- 与 CUDA Toolkit 兼容的 NVIDIA 驱动
- RTX 5090 使用 CUDA Toolkit 12.8+，RTX 4090 使用 11.8+，RTX 3090 使用 11.1+
- Bash、GNU Make、`flock`、`tar`、`sha256sum` 和 Python 3

RTX 4090 和 RTX 3090 已支持，但目前验证覆盖不如 RTX 5090 充分。在完成对应硬件的独立验证前，应把调优参数和长时间稳定性视为硬件相关结果。

安装前检查：

~~~bash
nvidia-smi
nvcc --version
command -v make
command -v flock
~~~

## 1. 准备发布包

在源码目录执行：

~~~bash
./scripts/package_worker.sh
cd dist
sha256sum -c SHA256SUMS
~~~

macOS 使用 `shasum -a 256 -c SHA256SUMS`。发布包包含 Worker 源码、脚本、公开文档和安全配置模板，不包含真实配置、Seed、钱包、结果、日志或已编译二进制。

## 2. 在离线机生成 Seed

只在负责保管 Seed 私有标量的离线机执行：

~~~bash
python3 seed.py
~~~

将 `seeds/split-seed.txt` 始终留在离线端，只把程序显示的 128 位十六进制公钥点复制给 GPU 算力机。

## 3. 安装 GPU 算力机

通过环境认可的部署方式，把发布包和校验文件传输到 Worker。主机名、用户名、端口和绝对安装路径均由实际环境决定，本文不提供相关示例。

根据环境设置 `TRANSFER_DIR` 和 `VADDRO_DIR`，然后校验并解压：

~~~bash
cd "$TRANSFER_DIR"
sha256sum -c SHA256SUMS

mkdir -p "$VADDRO_DIR"
tar -xzf vaddro-worker-linux-x86_64.tar.gz -C "$VADDRO_DIR"
cd "$VADDRO_DIR"
~~~

创建 Worker 配置：

~~~bash
cp config/miner.env.example config/miner.env
chmod 600 config/miner.env
~~~

编辑 `config/miner.env`，填入公钥点，并且只选择一种搜索模式：

~~~ini
PUBLIC_KEY="YOUR_128_HEX_SECP256K1_PUBLIC_POINT"

# 40 位精确模板，X/x/. 表示通配符。
PATTERN="0000000000000XXXXXXXXXXXXXXXXXXXXXXXXXXX"
SAME_CHAR_ENDS=0

MAX_HITS=1
WORKER_ID="worker-01"
SHARD_INDEX=0
SHARD_COUNT=1
GPU_INDEX=0
~~~

如需搜索首尾相同字符：

~~~ini
PATTERN=""
SAME_CHAR_ENDS=6
~~~

除非已在目标 GPU 上完成验证，否则保留 `miner.env.example` 中的默认调优参数。

## 4. 检查并编译

~~~bash
./00_doctor.sh
./01_build.sh
~~~

构建脚本会选择对应 GPU 架构，验证全部 8,160 个 secp256k1 预计算点，编译 CUDA 程序，并执行 GPU 密码学自检。

## 5. 启动、查看和停止

启动：

~~~bash
./02_start_search.sh
~~~

在另一个终端查看状态和结果：

~~~bash
./05_status.sh --gpu 0
./03_check_result.sh
~~~

正常停止：

~~~bash
./04_stop_search.sh --gpu 0
~~~

已验证命中直接写入 `result/`，运行事件写入 `logs/<worker-id>/<run-id>/events.jsonl`。如果运行期间删除了 `result/`，程序会在写入有效结果前自动重建。

## 6. 配置多台算力机

同一任务的 Worker 使用相同的 `PUBLIC_KEY` 和 `SHARD_COUNT`，但每台必须设置唯一的 `WORKER_ID` 和 `SHARD_INDEX`。下方以四台 Worker 配置为例，说明分片分配方法。

| 算力机 | `SHARD_INDEX` | `SHARD_COUNT` |
| --- | ---: | ---: |
| 算力机 1 | 0 | 4 |
| 算力机 2 | 1 | 4 |
| 算力机 3 | 2 | 4 |
| 算力机 4 | 3 | 4 |

`SHARD_COUNT` 必须是 1 到 65536 之间的 2 的幂，`SHARD_INDEX` 必须位于 `0..SHARD_COUNT-1`。同一任务期间不要修改这两个值。启动界面、状态、JSONL 日志和结果元数据都会显示当前分片。

## 7. 只更新，不启动

停止旧任务并创建回滚备份：

~~~bash
cd "$VADDRO_DIR"
./04_stop_search.sh --gpu 0

backup_dir="$BACKUP_DIR/vaddro-backup-YYYYMMDD"
mkdir -p "$backup_dir"
cp -a config cuda "$backup_dir"/
cp -a BUILD_COMMIT "$backup_dir"/ 2>/dev/null || true
~~~

上传新发布包和校验文件，然后执行：

~~~bash
cd "$TRANSFER_DIR"
sha256sum -c SHA256SUMS

cd "$VADDRO_DIR"
config_before=$(sha256sum config/miner.env | awk '{print $1}')
tar -xzf "$TRANSFER_DIR/vaddro-worker-linux-x86_64.tar.gz" -C "$VADDRO_DIR"
config_after=$(sha256sum config/miner.env | awk '{print $1}')
test "$config_before" = "$config_after"

./01_build.sh
./cuda/bin/vaddro-cuda --version
~~~

确认更新没有启动搜索：

~~~bash
test ! -f run/gpu-0.pid
if pgrep -x vaddro-cuda >/dev/null; then
  echo "ERROR: Worker 正在运行"
else
  echo "Worker 已停止，可以人工启动"
fi
~~~

原位更新会保留 `logs/`、`result/` 和 `config/miner.env`。启动始终是单独的人工操作。

## 8. 验证并合并命中

公开验证可以在任意机器执行：

~~~bash
python3 verify_result.py result/split-result-0x<address>.txt
~~~

验证通过后，把结果文件送到离线机，与 Seed 合并：

~~~bash
python3 combine.py \
  seeds/split-seed.txt \
  split-result-0x<address>.txt \
  wallets/wallet.txt
~~~

不要把 Seed 文件或最终钱包复制到 GPU 算力机。

## 常见问题

- **找不到 `nvcc`：**安装兼容的 CUDA Toolkit，并确认可以通过 `PATH` 找到 `nvcc`。
- **GPU 已有 Worker：**使用 `./05_status.sh --gpu 0` 检查，再用 `./04_stop_search.sh --gpu 0` 正常停止。
- **找不到运行状态：**确认当前安装目录和 GPU 编号正确。
- **分片错误：**在开始新任务前修正 `SHARD_INDEX` 和 `SHARD_COUNT`。
- **编译或自检失败：**不要启动 Worker；先恢复备份或修复主机环境。

安全边界和漏洞报告方式见 [SECURITY_CN.md](SECURITY_CN.md)。
