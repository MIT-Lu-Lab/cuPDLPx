# Compiler
CC = gcc
NVCC = nvcc
LINKER = nvcc
SYSTEM_GXX := $(shell which g++)

SRC_DIR = ./cupdlpx
BUILD_DIR = ./build

# CFLAGS for C compiler (gcc)
CFLAGS = -I. -I$(CUDA_HOME)/include -O3 -Wall -Wextra -g

# NVCCFLAGS for CUDA compiler (nvcc)
NVCCFLAGS = -I. -I$(CUDA_HOME)/include -O3 -g -gencode arch=compute_90,code=sm_90 -gencode arch=compute_80,code=sm_80 -Xcompiler -gdwarf-4 -ccbin $(SYSTEM_GXX)

# LDFLAGS for the linker
LDFLAGS = -L$(CONDA_PREFIX)/lib -L$(CUDA_HOME)/lib64 -lcudart -lcusparse -lcublas -lz -lm

# Source discovery (exclude the debug main)
C_SOURCES = $(filter-out $(SRC_DIR)/cupdlpx.c, $(wildcard $(SRC_DIR)/*.c))
CU_SOURCES = $(wildcard $(SRC_DIR)/*.cu)

C_OBJECTS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(C_SOURCES))
CU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CU_SOURCES))
OBJECTS = $(C_OBJECTS) $(CU_OBJECTS)

TARGET_LIB = $(BUILD_DIR)/libcupdlpx.a

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
.PHONY: all clean build tests test run-tests run-test clean-tests

# Default: build the static library
all: $(TARGET_LIB)

# Archive all objects into the static library
$(TARGET_LIB): $(OBJECTS)
	@echo "Archiving objects into $(TARGET_LIB)..."
	@mkdir -p $(BUILD_DIR)
	@ar rcs $@ $^

# Build the debug executable (links the library with cupdlpx.c main)
build: $(DEBUG_EXEC)

$(DEBUG_EXEC): $(DEBUG_SRC) $(TARGET_LIB)
	@echo "Building debug executable..."
	@$(LINKER) $(NVCCFLAGS) $(DEBUG_SRC) -o $(DEBUG_EXEC) $(TARGET_LIB) $(LDFLAGS)

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
tests: $(TARGET_LIB) $(TEST_EXEC_CU) $(TEST_EXEC_C)
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

# Build rule for CUDA tests: compile and link directly with nvcc
$(TEST_BUILD_DIR)/%: $(TEST_DIR)/%.cu $(TARGET_LIB)
	@mkdir -p $(TEST_BUILD_DIR)
	@echo "Building CUDA test $< -> $@..."
	@$(LINKER) $(NVCCFLAGS) -I$(SRC_DIR) $< -o $@ $(TARGET_LIB) $(LDFLAGS)

# Build rule for C tests: compile with gcc, link with nvcc to get CUDA libs
$(TEST_BUILD_DIR)/%: $(TEST_DIR)/%.c $(TARGET_LIB)
	@mkdir -p $(TEST_BUILD_DIR)
	@echo "Building C test $< -> $@..."
	@$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $(TEST_BUILD_DIR)/$*.o
	@$(LINKER) $(NVCCFLAGS) $(TEST_BUILD_DIR)/$*.o -o $@ $(TARGET_LIB) $(LDFLAGS)


# Cleaning
clean-tests:
	@echo "Cleaning test executables..."
	@rm -rf $(TEST_BUILD_DIR)

clean:
	@echo "Cleaning up..."
	@rm -rf $(BUILD_DIR) $(TARGET_LIB) $(DEBUG_EXEC)
	@rm -rf $(TEST_BUILD_DIR)