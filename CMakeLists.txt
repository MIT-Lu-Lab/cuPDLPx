# -----------------------------------------------------------------------------
# ROOT CMakeLists.txt for cuPDLPx (Unified Build System)
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)

# Project config
project(cupdlpx LANGUAGES C CXX CUDA)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# C/C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Set default build type to Release if not specified (Optimized builds)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Increase compatibility by compiling for all supported real and virtual architectures.
set(CMAKE_CUDA_ARCHITECTURES all)

# Global Compile flags (corresponding to CFLAGS/NVCCFLAGS)
add_compile_options(-fPIC -O3 -Wall -Wextra -g)

# Windows compatibility
if (WIN32)
    add_definitions(-Dstrtok_r=strtok_s)
endif()

# CUDA standards and RDC (Relocatable Device Code)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# CONTROL OPTIONS
# -----------------------------------------------------------------------------
include(CMakeDependentOption)

option(CUPDLPX_BUILD_STATIC_LIB "Build the cuPDLPx static library" ON)
option(CUPDLPX_BUILD_SHARED_LIB "Build the cuPDLPx shared library" ON)

# format: cmake_dependent_option(OPTION "docstring" DEFAULT_VALUE "DEPENDENCY_VARIABLE" FORCE_OFF_VALUE)
cmake_dependent_option(CUPDLPX_BUILD_PYTHON "Build the cuPDLPx Python bindings" OFF
                       "CUPDLPX_BUILD_STATIC_LIB" OFF)

cmake_dependent_option(CUPDLPX_BUILD_CLI "Build the cuPDLPx command-line executable" ON
                       "CUPDLPX_BUILD_STATIC_LIB" OFF)

cmake_dependent_option(CUPDLPX_BUILD_TESTS "Build the cuPDLPx test suite" OFF
                       "CUPDLPX_BUILD_STATIC_LIB" OFF)

# -----------------------------------------------------------------------------
# FIND DEPENDENCIES
# -----------------------------------------------------------------------------
# Core dependencies (required for Julia/Yggdrasil and Python)
find_package(CUDAToolkit REQUIRED)
find_package(ZLIB REQUIRED)

if (CUPDLPX_BUILD_PYTHON)
    # Dependencies required only for Python bindings
    find_package(pybind11 CONFIG REQUIRED)
    find_package(Python3 COMPONENTS Interpreter REQUIRED) # For versioning script and pybind11
endif()

# -----------------------------------------------------------------------------
# VERSION HEADER GENERATION (Build/generated/version.h)
# -----------------------------------------------------------------------------
if (CUPDLPX_BUILD_PYTHON)
    # Extract version from pyproject.toml (Requires Python3_EXECUTABLE)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c
[=[
import sys
try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml
import os

if not os.path.exists("pyproject.toml"):
    sys.stderr.write("Error: pyproject.toml not found in the root directory.\n")
    sys.exit(1)

with open("pyproject.toml", "rb") as f:
    data = toml.load(f)
ver = data["project"]["version"]
print(ver)
]=]
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
      OUTPUT_VARIABLE CUPDLPX_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "cuPDLPx version from pyproject.toml = ${CUPDLPX_VERSION}")
else()
    # Set a fallback version when Python is not used (suitable for Yggdrasil builds)
    set(CUPDLPX_VERSION "0.0.0-core-only")
    message(STATUS "cuPDLPx version (core only) = ${CUPDLPX_VERSION}")
endif()

# Generate version.h from version.h.in
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/version.h.in 
  ${CMAKE_CURRENT_BINARY_DIR}/generated/version.h
  @ONLY
)

# Make the generated files available for inclusion
include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)


# -----------------------------------------------------------------------------
# SOURCE DISCOVERY & TARGET DEFINITION
# -----------------------------------------------------------------------------
# Using file(GLOB) for convenience, but explicit lists are recommended for robust builds
file(GLOB C_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)
file(GLOB CU_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)
# Exclude cli.c from library builds
list(REMOVE_ITEM C_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/cli.c")

# Set common include directories for the core libraries
set(CORE_INCLUDE_DIRS
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include     # Public API headers
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/internal    # Internal implementation headers
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated   # Generated headers (version.h)
)

# Set common link libraries
set(CORE_LINK_LIBS PUBLIC
  CUDA::cudart
  CUDA::cublas
  CUDA::cusparse
  ZLIB::ZLIB
)

# -----------------------------------------------------------------------------
# 1. Core STATIC Library (cupdlpx_core)
# -----------------------------------------------------------------------------
if(CUPDLPX_BUILD_STATIC_LIB)
    add_library(cupdlpx_core STATIC
      ${C_SOURCES}
      ${CU_SOURCES}
    )
    target_include_directories(cupdlpx_core ${CORE_INCLUDE_DIRS})
    target_link_libraries(cupdlpx_core ${CORE_LINK_LIBS})

    set_target_properties(cupdlpx_core PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# -----------------------------------------------------------------------------
# 2. Shared Library (libcupdlpx.so)
# -----------------------------------------------------------------------------
if(CUPDLPX_BUILD_SHARED_LIB)
    add_library(cupdlpx_shared SHARED
      ${C_SOURCES}
      ${CU_SOURCES}
    )
    target_include_directories(cupdlpx_shared ${CORE_INCLUDE_DIRS})
    target_link_libraries(cupdlpx_shared ${CORE_LINK_LIBS})

    # Shared library must resolve device symbols as it is a final link point
    set_target_properties(cupdlpx_shared PROPERTIES
        OUTPUT_NAME "cupdlpx"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# -----------------------------------------------------------------------------
# 3. CLI Executable (cupdlpx)
# -----------------------------------------------------------------------------
if(CUPDLPX_BUILD_CLI)
    if(NOT TARGET cupdlpx_core)
        message(FATAL_ERROR "CUPDLPX_BUILD_CLI=ON requires CUPDLPX_BUILD_STATIC_LIB=ON.")
    endif()
    
    add_executable(cupdlpx_cli src/cli.c)

    target_include_directories(cupdlpx_cli PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/internal
        ${CMAKE_CURRENT_BINARY_DIR}/generated
    )

    # Link CLI to the static core library
    target_link_libraries(cupdlpx_cli PRIVATE cupdlpx_core)

    # CLI is a final executable, it must resolve device symbols
    set_target_properties(cupdlpx_cli PROPERTIES
        OUTPUT_NAME "cupdlpx"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# -----------------------------------------------------------------------------
# 4. Tests (CTest Integration)
# -----------------------------------------------------------------------------
if(CUPDLPX_BUILD_TESTS)
    if(NOT TARGET cupdlpx_core)
        message(FATAL_ERROR "CUPDLPX_BUILD_TESTS=ON requires CUPDLPX_BUILD_STATIC_LIB=ON.")
    endif()

    enable_testing()
    file(GLOB TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/test/*.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cu"
    )

    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE) 
        
        add_executable(${TEST_NAME} ${TEST_SRC})
        
        # Link tests to the core static library
        target_link_libraries(${TEST_NAME} PRIVATE cupdlpx_core)

        # Set up test includes
        target_include_directories(${TEST_NAME} 
          PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
          PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/internal
          PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated
        )
        
        # Tests are final executables, they must resolve device symbols
        set_target_properties(${TEST_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tests"
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
        
        # Register with CTest
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
        
    endforeach()
endif()

# -----------------------------------------------------------------------------
# 5. Python Bindings (Conditional)
# -----------------------------------------------------------------------------
if (CUPDLPX_BUILD_PYTHON)
    if(NOT TARGET cupdlpx_core)
        message(FATAL_ERROR "CUPDLPX_BUILD_PYTHON=ON requires CUPDLPX_BUILD_STATIC_LIB=ON.")
    endif()
    
    add_subdirectory(python_bindings)
endif()

# -----------------------------------------------------------------------------
# 6. Install Targets
# -----------------------------------------------------------------------------

if (CUPDLPX_BUILD_PYTHON)
    install(DIRECTORY include/
        DESTINATION include/
        FILES_MATCHING PATTERN "*.h"
    )

else()
    if(TARGET cupdlpx_core)
        install(TARGETS cupdlpx_core
            ARCHIVE DESTINATION lib
        )
    endif()
    
    if(TARGET cupdlpx_shared)
        install(TARGETS cupdlpx_shared
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin # 'bin' for DLLs on Windows, 'lib' for .so on Linux
        )
    endif()
    
    if(TARGET cupdlpx_cli)
        install(TARGETS cupdlpx_cli
            RUNTIME DESTINATION bin
        )
    endif()

    install(DIRECTORY include/
        DESTINATION include/
        FILES_MATCHING PATTERN "*.h"
    )
endif()