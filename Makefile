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
NVCCFLAGS = -I. -I$(CUDA_HOME)/include -O3 -g \
            -gencode arch=compute_90,code=sm_90 \
            -gencode arch=compute_80,code=sm_80 \
			-gencode arch=compute_70,code=sm_70 \
            -Xcompiler -fPIC  -Xcompiler -gdwarf-4 -ccbin $(SYSTEM_GXX)

# LDFLAGS for the linker
LDFLAGS = -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart -lcusparse -lcublas -lz -lm

# Source discovery (exclude the debug main)
C_SOURCES = $(filter-out $(SRC_DIR)/cupdlpx.c, $(wildcard $(SRC_DIR)/*.c))
CU_SOURCES = $(wildcard $(SRC_DIR)/*.cu)

C_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(C_SOURCES))
CU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CU_SOURCES))
OBJECTS = $(C_OBJECTS) $(CU_OBJECTS)

TARGET_STATIC = $(BUILD_DIR)/libcupdlpx.a
TARGET_SHARED = $(BUILD_DIR)/libcupdlpx.so

# Debug executable (optional)
DEBUG_SRC = $(SRC_DIR)/cupdlpx.c
DEBUG_EXEC = $(BUILD_DIR)/cupdlpx

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

# Build the debug executable (links the library with cupdlpx.c main)
build: $(DEBUG_EXEC)

$(DEBUG_EXEC): $(DEBUG_SRC) $(TARGET_STATIC)
	@echo "Building debug executable..."
	@$(LINKER) $(NVCCFLAGS) $(DEBUG_SRC) -o $(DEBUG_EXEC) $(TARGET_STATIC) $(LDFLAGS)

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
	@rm -rf $(BUILD_DIR) $(TARGET_STATIC) $(TARGET_SHARED) $(DEBUG_EXEC)
	@rm -rf $(TEST_BUILD_DIR)
