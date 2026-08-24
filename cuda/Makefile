NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++14 -Xptxas -O3 --threads 4
VERSION ?= $(shell cat ../VERSION 2>/dev/null || echo 1.0.0)
VERSION_DEFINE = -DVADDRO_VERSION='"$(VERSION)"'
GLV_X_REUSE ?= 1
GLV_X_REUSE_DEFINE = -DVADDRO_REUSE_GLV_X=$(GLV_X_REUSE)
INV_ADDITION_CHAIN ?= 0
INV_ADDITION_CHAIN_DEFINE = -DVADDRO_INV_ADDITION_CHAIN=$(INV_ADDITION_CHAIN)
SPLIT_VARIANT_HASH ?= 0
SPLIT_VARIANT_HASH_DEFINE = -DVADDRO_SPLIT_VARIANT_HASH=$(SPLIT_VARIANT_HASH)
AB_CXXFLAGS ?= $(CXXFLAGS) -lineinfo -Xptxas=-v

# Target architectures (Blackwell sm_120, Hopper sm_90, Ada sm_89, Ampere sm_86/80, Turing sm_75)
# If ARCH is specified (e.g. ARCH=sm_120), compile for that target only for 10x faster builds.
ifdef ARCH
GENCODES ?= -gencode arch=compute_$(subst sm_,,$(ARCH)),code=$(ARCH)
else
GENCODES ?= -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_89,code=sm_89 \
            -gencode arch=compute_90,code=sm_90 \
            -gencode arch=compute_120,code=sm_120 \
            -gencode arch=compute_120,code=compute_120
endif

BINDIR = bin
TARGET = $(BINDIR)/vaddro-cuda
AB_TARGET_ON = $(BINDIR)/vaddro-cuda-glv-reuse-on
AB_TARGET_OFF = $(BINDIR)/vaddro-cuda-glv-reuse-off
INV_AB_TARGET_ON = $(BINDIR)/vaddro-cuda-inv-chain-on
INV_AB_TARGET_OFF = $(BINDIR)/vaddro-cuda-inv-chain-off
SPLIT_AB_TARGET_ON = $(BINDIR)/vaddro-cuda-split-hash-on
SPLIT_AB_TARGET_OFF = $(BINDIR)/vaddro-cuda-split-hash-off

SRCS = main.cu precomp.cu
HEADERS = secp256k1_ptx.cuh keccak_ptx.cuh cuda_dispatcher.cuh precomp.hpp result_io.hpp search_shard.hpp

.PHONY: all ab-build ab-self-test inv-ab-build inv-ab-self-test split-ab-build split-ab-self-test clean self-test sanitizer sanitizer-build

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS) | $(BINDIR)
	$(NVCC) $(CXXFLAGS) $(VERSION_DEFINE) $(GLV_X_REUSE_DEFINE) $(INV_ADDITION_CHAIN_DEFINE) $(SPLIT_VARIANT_HASH_DEFINE) $(GENCODES) $(SRCS) -o $@

# Build both production-search variants from identical sources and compiler
# flags for controlled RTX 5090 A/B measurement.
ab-build: $(AB_TARGET_ON) $(AB_TARGET_OFF)

$(AB_TARGET_ON): $(SRCS) $(HEADERS) | $(BINDIR)
	$(NVCC) $(AB_CXXFLAGS) $(VERSION_DEFINE) -DVADDRO_REUSE_GLV_X=1 $(INV_ADDITION_CHAIN_DEFINE) $(SPLIT_VARIANT_HASH_DEFINE) $(GENCODES) $(SRCS) -o $@

$(AB_TARGET_OFF): $(SRCS) $(HEADERS) | $(BINDIR)
	$(NVCC) $(AB_CXXFLAGS) $(VERSION_DEFINE) -DVADDRO_REUSE_GLV_X=0 $(INV_ADDITION_CHAIN_DEFINE) $(SPLIT_VARIANT_HASH_DEFINE) $(GENCODES) $(SRCS) -o $@

ab-self-test: ab-build
	./$(AB_TARGET_ON) --self-test
	./$(AB_TARGET_OFF) --self-test

# Build fixed-chain and binary-inverse variants from identical sources and
# flags. RTX 5090 qualification found no end-to-end search gain, so the fixed
# chain remains opt-in even though its isolated inverse kernel is faster.
inv-ab-build: $(INV_AB_TARGET_ON) $(INV_AB_TARGET_OFF)

$(INV_AB_TARGET_ON): $(SRCS) $(HEADERS) | $(BINDIR)
	$(NVCC) $(AB_CXXFLAGS) $(VERSION_DEFINE) $(GLV_X_REUSE_DEFINE) -DVADDRO_INV_ADDITION_CHAIN=1 $(SPLIT_VARIANT_HASH_DEFINE) $(GENCODES) $(SRCS) -o $@

$(INV_AB_TARGET_OFF): $(SRCS) $(HEADERS) | $(BINDIR)
	$(NVCC) $(AB_CXXFLAGS) $(VERSION_DEFINE) $(GLV_X_REUSE_DEFINE) -DVADDRO_INV_ADDITION_CHAIN=0 $(SPLIT_VARIANT_HASH_DEFINE) $(GENCODES) $(SRCS) -o $@

inv-ab-self-test: inv-ab-build
	./$(INV_AB_TARGET_ON) --self-test
	./$(INV_AB_TARGET_OFF) --self-test

# Build the qualified fused path and the experimental point-basis + six-warp
# hash path from identical sources and compiler flags.
split-ab-build: $(SPLIT_AB_TARGET_ON) $(SPLIT_AB_TARGET_OFF)

$(SPLIT_AB_TARGET_ON): $(SRCS) $(HEADERS) | $(BINDIR)
	$(NVCC) $(AB_CXXFLAGS) $(VERSION_DEFINE) -DVADDRO_REUSE_GLV_X=1 $(INV_ADDITION_CHAIN_DEFINE) -DVADDRO_SPLIT_VARIANT_HASH=1 $(GENCODES) $(SRCS) -o $@

$(SPLIT_AB_TARGET_OFF): $(SRCS) $(HEADERS) | $(BINDIR)
	$(NVCC) $(AB_CXXFLAGS) $(VERSION_DEFINE) -DVADDRO_REUSE_GLV_X=1 $(INV_ADDITION_CHAIN_DEFINE) -DVADDRO_SPLIT_VARIANT_HASH=0 $(GENCODES) $(SRCS) -o $@

split-ab-self-test: split-ab-build
	./$(SPLIT_AB_TARGET_ON) --self-test
	./$(SPLIT_AB_TARGET_OFF) --self-test

$(BINDIR):
	mkdir -p $@

clean:
	rm -rf $(BINDIR)

self-test: $(TARGET)
	./$(TARGET) --self-test

sanitizer-build:
	$(MAKE) clean
	$(MAKE) CXXFLAGS="$(CXXFLAGS) -lineinfo" all

sanitizer: sanitizer-build
	compute-sanitizer --tool memcheck --error-exitcode 1 ./$(TARGET) --self-test
	compute-sanitizer --tool initcheck --error-exitcode 1 ./$(TARGET) --self-test
	@san_dir=$$(mktemp -d /tmp/vaddro-sanitizer.XXXXXX); \
	trap 'rm -rf "$$san_dir"' EXIT; \
	for tool in memcheck initcheck racecheck synccheck; do \
		mkdir -p "$$san_dir/$$tool"; \
		compute-sanitizer --tool "$$tool" --error-exitcode 1 ./$(TARGET) \
			--matching 0XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX --match-score 1 --max-hits 1 \
			-z 79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8 \
			-o "$$san_dir/$$tool" -I 8 -s 1 -w 32 --canary-interval 20 || exit 1; \
	done
