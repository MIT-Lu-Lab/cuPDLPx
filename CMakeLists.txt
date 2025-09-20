cmake_minimum_required(VERSION 3.20)

# Project config
project(pycupdlpx LANGUAGES C CXX CUDA)

# C/C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional: be explicit about CUDA standard and separable compilation
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optional: choose CUDA architectures (adjust as needed)
# e.g., 80=Ampere A100, 86=RTX30, 89=Hopper H100 (SM90=90)
# set(CMAKE_CUDA_ARCHITECTURES 80;86;89)

find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# -----------------------------------------------------------------------------
# Build the core native library that the Python extension links against
# -----------------------------------------------------------------------------
add_library(cupdlpx_core STATIC
  cupdlpx/interface.c
  # cupdlpx/solver.cu   # uncomment when you add CUDA sources
  cupdlpx/utils.c
)

# Public headers for dependents (e.g., the pybind11 module)
target_include_directories(cupdlpx_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/cupdlpx
)

# Ensure position-independent code so it can be linked into a Python extension
set_target_properties(cupdlpx_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# If you compile CUDA sources in this target, enable separable compilation
# (safe to keep ON even if there are no .cu files yet)
set_target_properties(cupdlpx_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# Link against CUDA runtime and libraries you use
target_link_libraries(cupdlpx_core PUBLIC
  CUDA::cudart
  CUDA::cublas
  CUDA::cusparse
)

# (Optional) Warnings for C/C++
if(MSVC)
  target_compile_options(cupdlpx_core PRIVATE /W4)
else()
  target_compile_options(cupdlpx_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# -----------------------------------------------------------------------------
# Add the pybind11 module (lives in subdir)
# -----------------------------------------------------------------------------
add_subdirectory(python_bindings)
