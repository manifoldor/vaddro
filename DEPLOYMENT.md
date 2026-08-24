# Vaddro Deployment Guide

[中文部署指南](DEPLOYMENT_CN.md)

Vaddro keeps the Seed private scalar offline while Linux GPU workers search public Search Shares. The optimized production target is NVIDIA RTX 5090 / Blackwell `sm_120`.

## Requirements

- Linux x86_64; Ubuntu 22.04 is tested
- NVIDIA RTX 5090 (`sm_120`), RTX 4090 (`sm_89`), or RTX 3090 (`sm_86`)
- NVIDIA driver compatible with the installed CUDA Toolkit
- CUDA Toolkit 12.8+ for RTX 5090, 11.8+ for RTX 4090, or 11.1+ for RTX 3090
- Bash, GNU Make, `flock`, `tar`, `sha256sum`, and Python 3

RTX 4090 and RTX 3090 are supported, but their validation coverage is currently less complete than RTX 5090. Treat their tuning values and long-duration stability as hardware-specific until independently qualified.

Check the host before installation:

~~~bash
nvidia-smi
nvcc --version
command -v make
command -v flock
~~~

## 1. Prepare the release archive

From a source checkout:

~~~bash
./scripts/package_worker.sh
cd dist
sha256sum -c SHA256SUMS
~~~

On macOS, use `shasum -a 256 -c SHA256SUMS` instead. The archive contains worker source, scripts, public documentation, and a safe configuration template. It does not contain an operator configuration, Seed, wallet, result, log, or compiled binary.

## 2. Generate the Seed offline

Run this only on the offline machine that will retain the Seed private scalar:

~~~bash
python3 seed.py
~~~

Keep `seeds/split-seed.txt` offline. Copy only the displayed 128-hex-character public point to the GPU workers.

## 3. Install a GPU worker

Transfer the archive and checksum to the worker using your approved deployment method. Hostnames, usernames, ports, and absolute installation paths are intentionally environment-specific and omitted here.

Set `TRANSFER_DIR` and `VADDRO_DIR` for your environment, then verify and extract:

~~~bash
cd "$TRANSFER_DIR"
sha256sum -c SHA256SUMS

mkdir -p "$VADDRO_DIR"
tar -xzf vaddro-worker-linux-x86_64.tar.gz -C "$VADDRO_DIR"
cd "$VADDRO_DIR"
~~~

Create the worker configuration:

~~~bash
cp config/miner.env.example config/miner.env
chmod 600 config/miner.env
~~~

Edit `config/miner.env`. Set the public point and choose exactly one search mode:

~~~ini
PUBLIC_KEY="YOUR_128_HEX_SECP256K1_PUBLIC_POINT"

# Exact 40-nibble pattern; X/x/. are wildcards.
PATTERN="0000000000000XXXXXXXXXXXXXXXXXXXXXXXXXXX"
SAME_CHAR_ENDS=0

MAX_HITS=1
WORKER_ID="worker-01"
SHARD_INDEX=0
SHARD_COUNT=1
GPU_INDEX=0
~~~

For same-character prefix/suffix mode:

~~~ini
PATTERN=""
SAME_CHAR_ENDS=6
~~~

Keep the tuning values from `miner.env.example` unless you have qualified different values on the target GPU.

## 4. Diagnose and build

~~~bash
./00_doctor.sh
./01_build.sh
~~~

The build script selects the GPU architecture, verifies all 8,160 precomputed secp256k1 points, compiles the CUDA engine, and runs the GPU cryptographic self-test.

## 5. Start, monitor, and stop

~~~bash
./02_start_search.sh
~~~

In another terminal:

~~~bash
./05_status.sh --gpu 0
./03_check_result.sh
~~~

Stop the worker cleanly:

~~~bash
./04_stop_search.sh --gpu 0
~~~

Verified hits are written directly under `result/`. Runtime events are stored under `logs/<worker-id>/<run-id>/events.jsonl`. If `result/` is removed while the worker is running, it is recreated before a verified result is written.

## 6. Configure multiple workers

Workers on the same task use the same `PUBLIC_KEY` and `SHARD_COUNT`, but each needs a unique `WORKER_ID` and `SHARD_INDEX`. The four-worker layout below is only an example of shard assignment.

| Worker | `SHARD_INDEX` | `SHARD_COUNT` |
| --- | ---: | ---: |
| worker-01 | 0 | 4 |
| worker-02 | 1 | 4 |
| worker-03 | 2 | 4 |
| worker-04 | 3 | 4 |

`SHARD_COUNT` must be a power of two from 1 through 65536. `SHARD_INDEX` must be in `0..SHARD_COUNT-1`. Keep both stable for the lifetime of a task. The active shard appears in the startup screen, state, JSONL log, status output, and result metadata.

## 7. Update without starting

Stop the existing worker and create a rollback backup:

~~~bash
cd "$VADDRO_DIR"
./04_stop_search.sh --gpu 0

backup_dir="$BACKUP_DIR/vaddro-backup-YYYYMMDD"
mkdir -p "$backup_dir"
cp -a config cuda "$backup_dir"/
cp -a BUILD_COMMIT "$backup_dir"/ 2>/dev/null || true
~~~

Upload the new archive and checksum, then run:

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

Confirm that the update did not start a search process:

~~~bash
test ! -f run/gpu-0.pid
if pgrep -x vaddro-cuda >/dev/null; then
  echo "ERROR: worker is running"
else
  echo "Worker is stopped and ready"
fi
~~~

`logs/`, `result/`, and `config/miner.env` are preserved during an in-place update. Starting remains a separate operator action.

## 8. Verify and combine a hit

Public verification can run on any machine:

~~~bash
python3 verify_result.py result/split-result-0x<address>.txt
~~~

Move the verified result to the offline machine, then combine it with the Seed:

~~~bash
python3 combine.py \
  seeds/split-seed.txt \
  split-result-0x<address>.txt \
  wallets/wallet.txt
~~~

Do not copy the Seed file or final wallet to a GPU worker.

## Troubleshooting

- **`nvcc` not found:** install a compatible CUDA Toolkit and ensure `nvcc` is available through `PATH`.
- **GPU already has a worker:** inspect it with `./05_status.sh --gpu 0`; stop it with `./04_stop_search.sh --gpu 0`.
- **No state found:** confirm the correct installation directory and GPU index.
- **Wrong shard:** correct `SHARD_INDEX` and `SHARD_COUNT` before starting a new run.
- **Build or self-test fails:** do not start the worker; restore the backup or fix the host environment first.

For the trust model and incident reporting, see [SECURITY.md](SECURITY.md).
