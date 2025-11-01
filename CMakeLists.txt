cmake_minimum_required(VERSION 3.20)

# Project config
project(cupdlpx LANGUAGES C CXX CUDA)

# C/C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# On Windows, map strtok_r to strtok_s for compatibility
if (WIN32)
    add_definitions(-Dstrtok_r=strtok_s)
endif()

# Optional: be explicit about CUDA standard and separable compilation
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "all")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(ZLIB REQUIRED)

# -----------------------------------------------------------------------------
# Version header generation
# -----------------------------------------------------------------------------
# Get python for versioning script
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Extract version from pyproject.toml
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
[=[
import sys
# Use stdlib tomllib if available, otherwise fall back to tomli (both accept .load on a binary file)
try:
    import tomllib as toml  # Python 3.11+
except ModuleNotFoundError:
    import tomli as toml

with open("pyproject.toml", "rb") as f:
    data = toml.load(f)

ver = data["project"]["version"]
print(ver)
]=]
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_VARIABLE CUPDLPX_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Print the version to CMake output
message(STATUS "cuPDLPx version from pyproject.toml = ${CUPDLPX_VERSION}")

# Generate version.h from version.h.in
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cupdlpx/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/version.h
  @ONLY
)

# Make the generated files available for inclusion
include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)

# -----------------------------------------------------------------------------
# Build the core native library that the Python extension links against
# -----------------------------------------------------------------------------
add_library(cupdlpx_core STATIC
  cupdlpx/interface.c
  cupdlpx/io.c
  cupdlpx/solver.cu
  cupdlpx/utils.cu
)

# Public headers for dependents (e.g., the pybind11 module)
target_include_directories(cupdlpx_core
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/cupdlpx
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated
)

# Ensure position-independent code so it can be linked into a Python extension
set_target_properties(cupdlpx_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# If you compile CUDA sources in this target, enable separable compilation
set_target_properties(cupdlpx_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link against CUDA runtime and libraries you use
target_link_libraries(cupdlpx_core PUBLIC
  CUDA::cudart
  CUDA::cublas
  CUDA::cusparse
  ZLIB::ZLIB
)

# -----------------------------------------------------------------------------
# Add the pybind11 module (lives in subdir)
# -----------------------------------------------------------------------------
add_subdirectory(python_bindings)

