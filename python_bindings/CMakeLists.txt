# python_bindings/CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

# sources
set(PYBIND_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/_core_bindings.cpp
)

# build the pybind11 module
pybind11_add_module(_cupdlpx_core MODULE ${PYBIND_SOURCES})

# include paths
target_include_directories(_cupdlpx_core
  PRIVATE
    ${PROJECT_SOURCE_DIR}/cupdlpx
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# link against the core native library and CUDA
target_link_libraries(_cupdlpx_core
  PRIVATE
    cupdlpx_core
    CUDA::cudart
)

# better error messages from pybind11
target_compile_definitions(_cupdlpx_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)

# --- Install into the Python package directory (for wheel) ---
install(TARGETS _cupdlpx_core
  LIBRARY DESTINATION cupdlpx
  RUNTIME DESTINATION cupdlpx
  ARCHIVE DESTINATION cupdlpx
)
