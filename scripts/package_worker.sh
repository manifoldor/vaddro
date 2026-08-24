#!/usr/bin/env bash
# ==============================================================================
# Vaddro - Worker Release Packager
# Generates clean, production-ready tarballs for deploying 算力机 (workers).
# Strictly excludes offline seed files, private keys, wallets, and build artifacts.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$PROJECT_ROOT/dist"
BUILD_TEMP="$DIST_DIR/vaddro-worker"
VERSION=$(tr -d '[:space:]' < "$PROJECT_ROOT/VERSION")
ARCHIVE_NAME="vaddro-worker-linux-x86_64.tar.gz"
CHECKSUM_FILE="$DIST_DIR/SHA256SUMS"

echo "=================================================="
echo "📦 Building Vaddro Deployment Package ($VERSION)"
echo "=================================================="

mkdir -p "$DIST_DIR"
rm -rf "$BUILD_TEMP" "$DIST_DIR/$ARCHIVE_NAME" "$CHECKSUM_FILE"
mkdir -p "$BUILD_TEMP"

# 1. Copy operational scripts and public root documentation
cp "$PROJECT_ROOT"/00_doctor.sh "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/01_build.sh "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/02_start_search.sh "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/03_check_result.sh "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/04_stop_search.sh "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/05_status.sh "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/benchmark.sh "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/README.md "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/README_CN.md "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/DEPLOYMENT.md "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/DEPLOYMENT_CN.md "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/SECURITY.md "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/SECURITY_CN.md "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/CHANGELOG.md "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/LICENSE "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/VERSION "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/combine.py "$BUILD_TEMP/"
cp "$PROJECT_ROOT"/verify_result.py "$BUILD_TEMP/"

# 2. Copy CUDA sources
mkdir -p "$BUILD_TEMP/cuda"
cp "$PROJECT_ROOT"/cuda/*.cu "$BUILD_TEMP/cuda/" 2>/dev/null || true
cp "$PROJECT_ROOT"/cuda/*.cuh "$BUILD_TEMP/cuda/" 2>/dev/null || true
cp "$PROJECT_ROOT"/cuda/*.hpp "$BUILD_TEMP/cuda/" 2>/dev/null || true
cp "$PROJECT_ROOT"/cuda/Makefile "$BUILD_TEMP/cuda/"
cp "$PROJECT_ROOT"/cuda/precomp.cu.sha256 "$BUILD_TEMP/cuda/"

# 3. Copy the safe config template; real operator config is never packaged.
mkdir -p "$BUILD_TEMP/config"
cp "$PROJECT_ROOT"/config/miner.env.example "$BUILD_TEMP/config/"
mkdir -p "$BUILD_TEMP/scripts"
cp "$PROJECT_ROOT"/scripts/config_loader.sh "$BUILD_TEMP/scripts/"
cp "$PROJECT_ROOT"/scripts/verify_precomp.py "$BUILD_TEMP/scripts/"

# Record the reviewed source revision without requiring git on the GPU worker.
if [[ -n "${VADDRO_BUILD_COMMIT:-}" ]]; then
    BUILD_COMMIT="$VADDRO_BUILD_COMMIT"
elif command -v git >/dev/null 2>&1; then
    BUILD_COMMIT=$(git -C "$PROJECT_ROOT" rev-parse --verify HEAD 2>/dev/null || true)
else
    BUILD_COMMIT=""
fi
BUILD_COMMIT="${BUILD_COMMIT:-local-build}"
printf '%s\n' "$BUILD_COMMIT" > "$BUILD_TEMP/BUILD_COMMIT"

# 4. Create empty runtime directories
mkdir -p "$BUILD_TEMP/result" "$BUILD_TEMP/logs" "$BUILD_TEMP/run"

# 5. Clean macOS metadata and set execution permissions
find "$BUILD_TEMP" -name "._*" -delete 2>/dev/null || true
find "$BUILD_TEMP" -name ".DS_Store" -delete 2>/dev/null || true
chmod +x "$BUILD_TEMP"/*.sh

# 6. Build a deterministic tarball with normalized metadata.
python3 "$PROJECT_ROOT/scripts/create_reproducible_tar.py" "$BUILD_TEMP" "$DIST_DIR/$ARCHIVE_NAME"
rm -rf "$BUILD_TEMP"

if tar -tzf "$DIST_DIR/$ARCHIVE_NAME" | grep -E '^\./?(docs(/|$)|DESIGN\.md$|ARTICLE_CN\.md$|README\.txt$|RELEASE_CHECKLIST\.md$|VALIDATION\.md$|CONTRIBUTING\.md$|CODE_OF_CONDUCT\.md$|CITATION\.cff$|\.github/pull_request_template\.md$)' >/dev/null; then
    echo "ERROR: internal documentation found in worker archive." >&2
    exit 1
fi

ARCHIVE_SIZE=$(ls -lh "$DIST_DIR/$ARCHIVE_NAME" | awk '{print $5}')
if command -v shasum &> /dev/null; then
    SHA256_SUM=$(shasum -a 256 "$DIST_DIR/$ARCHIVE_NAME" | awk '{print $1}')
elif command -v sha256sum &> /dev/null; then
    SHA256_SUM=$(sha256sum "$DIST_DIR/$ARCHIVE_NAME" | awk '{print $1}')
else
    echo "ERROR: could not compute SHA-256 checksum." >&2
    exit 1
fi
printf '%s  %s\n' "$SHA256_SUM" "$ARCHIVE_NAME" > "$CHECKSUM_FILE"

echo "✓ Vaddro Deployment bundle generated successfully!"
echo "--------------------------------------------------"
echo "📁 Path:   dist/$ARCHIVE_NAME"
echo "📏 Size:   $ARCHIVE_SIZE"
echo "🔑 SHA256: $SHA256_SUM"
echo "🧾 Manifest: dist/SHA256SUMS"
echo "=================================================="
