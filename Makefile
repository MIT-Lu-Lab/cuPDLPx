# Compiler
CC = gcc
NVCC = nvcc
LINKER = nvcc
SYSTEM_GXX := $(shell which g++)

SRC_DIR = ./cupdlpx
BUILD_DIR = ./build

# CFLAGS for C compiler (gcc)
CFLAGS = -I. -I$(CUDA_HOME)/include -fPIC -O3 -Wall -Wextra -g

# NVCCFLAGS for CUDA compiler (nvcc)
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//')
NVCCFLAGS = -I. -I$(CUDA_HOME)/include -O3 -g \
            -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH) \
            -Xcompiler -fPIC  -Xcompiler -gdwarf-4 -ccbin $(SYSTEM_GXX)

# LDFLAGS for the linker
LDFLAGS = -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart -lcusparse -lcublas -lz -lm


# Version header generation
GEN_DIR := $(BUILD_DIR)/generated
VERSION := $(shell sed -n 's/^version *= *"\(.*\)"/\1/p' pyproject.toml)
VERSION_H := $(GEN_DIR)/version.h

$(VERSION_H): $(SRC_DIR)/version.h.in pyproject.toml
	@mkdir -p $(GEN_DIR)
	sed 's/@CUPDLPX_VERSION@/$(VERSION)/g' $< > $@
	@echo "generated $@ (version $(VERSION))"

# Add include path for generated headers
CFLAGS    += -I$(GEN_DIR)
NVCCFLAGS += -I$(GEN_DIR)

# Ensure objects that include version.h depend on it
$(BUILD_DIR)/utils.o: $(VERSION_H)

# Source discovery (exclude the debug main)
C_SOURCES = $(filter-out $(SRC_DIR)/cli.c, $(wildcard $(SRC_DIR)/*.c))
CU_SOURCES = $(wildcard $(SRC_DIR)/*.cu)

C_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(C_SOURCES))
CU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CU_SOURCES))
OBJECTS = $(C_OBJECTS) $(CU_OBJECTS)

TARGET_STATIC = $(BUILD_DIR)/libcupdlpx.a
TARGET_SHARED = $(BUILD_DIR)/libcupdlpx.so

# CLI executable 
CLI_SRC := $(SRC_DIR)/cli.c
CLI_BIN := $(BUILD_DIR)/cupdlpx

.PHONY: cli
cli: $(CLI_BIN)

$(CLI_BIN): $(CLI_SRC) $(TARGET_STATIC) $(VERSION_H)
	@echo "Building CLI $(CLI_BIN)..."
	@mkdir -p $(BUILD_DIR)
	@$(LINKER) $(NVCCFLAGS) $(CLI_SRC) -o $(CLI_BIN) $(TARGET_STATIC) $(LDFLAGS)

# Tests auto-discovery
TEST_DIR := ./test
TEST_BUILD_DIR := $(BUILD_DIR)/tests

TEST_CU_SOURCES := $(wildcard $(TEST_DIR)/*.cu)
TEST_C_SOURCES := $(wildcard $(TEST_DIR)/*.c)

# Each test source becomes an executable at build/tests/<basename>
TEST_EXEC_CU := $(patsubst $(TEST_DIR)/%.cu,$(TEST_BUILD_DIR)/%,$(TEST_CU_SOURCES))
TEST_EXEC_C := $(patsubst $(TEST_DIR)/%.c,$(TEST_BUILD_DIR)/%,$(TEST_C_SOURCES))

# Phony targets
.PHONY: all clean build tests test run-tests run-test clean-tests shared install

# Default: build the static library
all: $(TARGET_STATIC)

# Archive all objects into the static library
$(TARGET_STATIC): $(OBJECTS)
	@echo "Archiving objects into $(TARGET_STATIC)..."
	@mkdir -p $(BUILD_DIR)
	@ar rcs $@ $^

# Build shared library
shared: $(OBJECTS)
	@echo "Building shared library $(TARGET_SHARED)..."
	@mkdir -p $(BUILD_DIR)
	$(NVCC) -shared -o $(TARGET_SHARED) $(OBJECTS) $(LDFLAGS) --cudart=shared

# Build the CLI executable
build: cli

# Pattern rules for objects
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	@echo "Compiling $< -> $@..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	@echo "Compiling $< -> $@..."
	@$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Build all tests discovered under test/
test: tests
tests: $(TARGET_STATIC) $(TEST_EXEC_CU) $(TEST_EXEC_C)
	@echo "All tests built under $(TEST_BUILD_DIR)/"

# Run all tests one by one
run-tests: tests
	@echo "Running all tests..."
	@set -e; \
	for t in $(TEST_EXEC_CU) $(TEST_EXEC_C); do \
	  echo "=== $$t ==="; \
	  "$$t" || exit $$?; \
	  echo; \
	done

# Run a single test by basename: make run-test name=<basename>
run-test: tests
	@if [ -z "$(name)" ]; then \
	  echo "Usage: make run-test name=<basename-of-test-file>"; exit 2; \
	fi
	@if [ -x "$(TEST_BUILD_DIR)/$(name)" ]; then \
	  echo "=== $(TEST_BUILD_DIR)/$(name) ==="; \
	  "$(TEST_BUILD_DIR)/$(name)"; \
	else \
	  echo "Executable not found: $(TEST_BUILD_DIR)/$(name)"; \
	  echo "Did you 'make tests' and is there a test source named '$(TEST_DIR)/$(name).c(u)'?"; \
	  exit 1; \
	fi

# Build rule for CUDA tests
$(TEST_BUILD_DIR)/%: $(TEST_DIR)/%.cu $(TARGET_STATIC)
	@mkdir -p $(TEST_BUILD_DIR)
	@echo "Building CUDA test $< -> $@..."
	@$(LINKER) $(NVCCFLAGS) -I$(SRC_DIR) $< -o $@ $(TARGET_STATIC) $(LDFLAGS)

# Build rule for C tests
$(TEST_BUILD_DIR)/%: $(TEST_DIR)/%.c $(TARGET_STATIC)
	@mkdir -p $(TEST_BUILD_DIR)
	@echo "Building C test $< -> $@..."
	@$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $(TEST_BUILD_DIR)/$*.o
	@$(LINKER) $(NVCCFLAGS) $(TEST_BUILD_DIR)/$*.o -o $@ $(TARGET_STATIC) $(LDFLAGS)

# Install rule for BinaryBuilder
install: all shared
	@echo "Installing to $(PREFIX)..."
	@mkdir -p $(PREFIX)/lib
	@mkdir -p $(PREFIX)/include
	@cp $(TARGET_STATIC) $(PREFIX)/lib/
	@cp $(TARGET_SHARED) $(PREFIX)/lib/
	@cp $(SRC_DIR)/*.h $(PREFIX)/include/

# Cleaning
clean-tests:
	@echo "Cleaning test executables..."
	@rm -rf $(TEST_BUILD_DIR)

clean:
	@echo "Cleaning up..."
	@rm -rf $(BUILD_DIR) $(TARGET_STATIC) $(TARGET_SHARED)
	@rm -rf $(TEST_BUILD_DIR)
