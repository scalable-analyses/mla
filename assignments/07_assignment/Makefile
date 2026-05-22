MLIR_AIE_DIR ?= $(shell python3 -c "from aie.utils.config import root_path; print(root_path())")
AIE_INC      := $(MLIR_AIE_DIR)/include

PEANO_CC     = $(PEANO_INSTALL_DIR)/bin/clang++
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw
KERNEL_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -I $(AIE_INC)

.PHONY: all asm_vadd asm_matmul asm_matmul_bfp16 \
        obj_vadd obj_custom_vadd run_vadd run_custom_vadd clean

# ── Assembly targets ──────────────────────────────────────────────────────────

asm_vadd: build/vadd.s

build/vadd.s: src/vadd.cpp | build/
	$(PEANO_CC) $(KERNEL_FLAGS) -S $< -o $@

asm_matmul: build/matmul_normal.s

build/matmul_normal.s: src/matmul.cpp | build/
	$(PEANO_CC) $(KERNEL_FLAGS) -S $< -o $@

asm_matmul_bfp16: build/matmul_bfp16.s

build/matmul_bfp16.s: src/matmul.cpp | build/
	$(PEANO_CC) $(KERNEL_FLAGS) -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -S $< -o $@

# ── Object file targets ───────────────────────────────────────────────────────

obj_vadd: build/vadd.o

build/vadd.o: src/vadd.cpp | build/
	$(PEANO_CC) $(KERNEL_FLAGS) -c $< -o $@

obj_custom_vadd: build/custom_vadd.o

build/custom_vadd.o: src/custom_vadd.s | build/
	$(PEANO_CC) $(KERNEL_FLAGS) -c $< -o $@

# ── xclbin targets ────────────────────────────────────────────────────────────

build/final_vadd.xclbin: build/vadd.o src/vadd.mlir | build/
	cd build && aiecc.py --alloc-scheme=basic-sequential --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_vadd.bin \
	    --aie-generate-xclbin --xclbin-name=final_vadd.xclbin \
	    ../src/vadd.mlir

build/final_custom_vadd.xclbin: build/custom_vadd.o src/custom_vadd.mlir | build/
	cd build && aiecc.py --alloc-scheme=basic-sequential --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_custom_vadd.bin \
	    --aie-generate-xclbin --xclbin-name=final_custom_vadd.xclbin \
	    ../src/custom_vadd.mlir

# ── Run targets ───────────────────────────────────────────────────────────────

run_vadd: build/final_vadd.xclbin
	python3 src/driver.py vadd

run_custom_vadd: build/final_custom_vadd.xclbin
	python3 src/driver.py custom_vadd

# ── Utility ───────────────────────────────────────────────────────────────────

build/:
	mkdir -p build

clean:
	rm -rf build
