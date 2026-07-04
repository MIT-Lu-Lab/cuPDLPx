# -----------------------------------------------------------------------------
# ROOT CMakeLists.txt for cuPDLPx (Unified Build System)
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)

# HIP/ROCm support option (must be set before project() to influence language detection)
option(USE_HIP "Build with HIP for AMD GPUs" OFF)

# Project config
if(USE_HIP)
  project(cupdlpx LANGUAGES C CXX HIP)
else()
  project(cupdlpx LANGUAGES C CXX)

  # Default CUDA architectures: SASS for every current arch, PTX only for the newest one as a forward-compat fallback.
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13)
      set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90-real
          100-real 120-real 120-virtual)
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
      set(CMAKE_CUDA_ARCHITECTURES 60-real 70-real 75-real 80-real 86-real
          89-real 90-real 100-real 120-real 120-virtual)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 60-real 70-real 75-real 80-real 86-real
          89-real 90-real 90-virtual)
    endif()
  endif()

  enable_language(CUDA)
  message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

set(CUPDLPX_VERSION_MAJOR 0)
set(CUPDLPX_VERSION_MINOR 2)
set(CUPDLPX_VERSION_PATCH 9)

set(CUPDLPX_VERSION "${CUPDLPX_VERSION_MAJOR}.${CUPDLPX_VERSION_MINOR}.${CUPDLPX_VERSION_PATCH}")
add_compile_definitions(CUPDLPX_VERSION="${CUPDLPX_VERSION}")
add_compile_definitions(CUSPARSE_ENABLE_EXPERIMENTAL_API)

if (WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# C/C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(USE_HIP)
  # HIP architecture configuration
  # Default to gfx90a if not specified; can override with -DCMAKE_HIP_ARCHITECTURES=gfx1100, etc.
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
    set(CMAKE_HIP_ARCHITECTURES "gfx90a")
  endif()
endif()

# -----------------------------------------------------------------------------
# [ELEGANT DESIGN] Target-based Compile Flags
# -----------------------------------------------------------------------------
add_library(cupdlpx_compile_flags INTERFACE)

if(MSVC)
    target_compile_options(cupdlpx_compile_flags INTERFACE
        $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:/O2 /W4 /Zi>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zi>
    )
    target_compile_definitions(cupdlpx_compile_flags INTERFACE 
        _CRT_SECURE_NO_WARNINGS
        _CRT_NONSTDC_NO_DEPRECATE
        strtok_r=strtok_s
        strdup=_strdup
    )
else()
    target_compile_options(cupdlpx_compile_flags INTERFACE -O3 -Wall -Wextra -g)
    if(NOT WIN32)
        target_compile_options(cupdlpx_compile_flags INTERFACE -fPIC)
    endif()
endif()

# CUDA/HIP standards and RDC
if(USE_HIP)
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
else()
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# -----------------------------------------------------------------------------
# CONTROL OPTIONS
# -----------------------------------------------------------------------------
include(CMakeDependentOption)

option(CUPDLPX_BUILD_STATIC_LIB "Build the cuPDLPx static library" ON)
option(CUPDLPX_BUILD_SHARED_LIB "Build the cuPDLPx shared library" ON)

cmake_dependent_option(CUPDLPX_BUILD_PYTHON "Build the cuPDLPx Python bindings" OFF "CUPDLPX_BUILD_STATIC_LIB" OFF)
cmake_dependent_option(CUPDLPX_BUILD_CLI "Build the cuPDLPx command-line executable" ON "CUPDLPX_BUILD_STATIC_LIB" OFF)
cmake_dependent_option(CUPDLPX_BUILD_TESTS "Build the cuPDLPx test suite" OFF "CUPDLPX_BUILD_STATIC_LIB" OFF)

if(WIN32 AND CUPDLPX_BUILD_CLI)
    message(STATUS "Disabling CUPDLPX_BUILD_CLI on Windows (MSVC) due to getopt.h/libgen.h dependency.")
    set(CUPDLPX_BUILD_CLI OFF CACHE BOOL "Build the cuPDLPx command-line executable" FORCE)
endif()

# -----------------------------------------------------------------------------
# FIND DEPENDENCIES
# -----------------------------------------------------------------------------
if(USE_HIP)
  # Find ROCm/HIP libraries
  find_package(hip REQUIRED)
  find_package(hipblas REQUIRED)
  find_package(hipsparse REQUIRED)
  find_package(hipcub REQUIRED)
  find_package(rocprim REQUIRED)
endif()
include(FetchContent)

# 1. ZLIB Configuration
find_package(ZLIB QUIET)
if(NOT ZLIB_FOUND)
    FetchContent_Declare(
        zlib
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG        v1.3
    )
    FetchContent_MakeAvailable(zlib)
endif()

if(NOT TARGET ZLIB::ZLIB)
    if(TARGET zlibstatic)
        add_library(ZLIB::ZLIB ALIAS zlibstatic)
        target_include_directories(zlibstatic INTERFACE
            $<BUILD_INTERFACE:${zlib_SOURCE_DIR}>
            $<BUILD_INTERFACE:${zlib_BINARY_DIR}>
        )
    elseif(TARGET zlib)
        add_library(ZLIB::ZLIB ALIAS zlib)
        target_include_directories(zlib INTERFACE
            $<BUILD_INTERFACE:${zlib_SOURCE_DIR}>
            $<BUILD_INTERFACE:${zlib_BINARY_DIR}>
        )
    else()
        message(FATAL_ERROR "ZLIB target not found. Expected ZLIB::ZLIB, zlibstatic, or zlib.")
    endif()
endif()

# 2. PSLP Configuration
set(PSLP_VERSION_TAG "v0.0.8")
FetchContent_Declare(
  pslp
  GIT_REPOSITORY https://github.com/dance858/PSLP.git
  GIT_TAG        ${PSLP_VERSION_TAG}
)
FetchContent_MakeAvailable(pslp)

# --- PSLP target include/flags normalization ---
if(TARGET PSLP)
    # Expose PSLP headers via target usage requirements (no directory-wide includes).
    target_include_directories(PSLP INTERFACE 
        $<BUILD_INTERFACE:${pslp_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${pslp_SOURCE_DIR}/include/PSLP>
    )

    # Defensive flag filtering for MSVC
    if(MSVC)
        get_target_property(PSLP_OPTS PSLP COMPILE_OPTIONS)
        if(PSLP_OPTS)
            list(REMOVE_ITEM PSLP_OPTS "-Wall" "-Wextra" "-Wpedantic" "-Werror")
            set_target_properties(PSLP PROPERTIES COMPILE_OPTIONS "${PSLP_OPTS}")
        endif()
    endif()
endif()

target_compile_definitions(cupdlpx_compile_flags INTERFACE PSLP_VERSION="${PSLP_VERSION_TAG}")

# -----------------------------------------------------------------------------
# TARGET DEFINITIONS
# -----------------------------------------------------------------------------
file(GLOB C_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
file(GLOB CU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
list(REMOVE_ITEM C_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/cli.c")
if(WIN32)
  # mps_parser.c is CLI-only; exclude it on Windows where strtok_r is unavailable
  list(REMOVE_ITEM C_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/mps_parser.c")
endif()

set(CORE_INCLUDE_DIRS
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/internal
)

if(USE_HIP)
  set(CORE_LINK_LIBS
    PUBLIC cupdlpx_compile_flags
    PUBLIC hip::device
    PUBLIC roc::hipblas
    PUBLIC roc::hipsparse
    PUBLIC hip::hipcub
    PUBLIC ZLIB::ZLIB
    PUBLIC PSLP
  )
  # Mark .cu files as HIP language
  set_source_files_properties(${CU_SOURCES} PROPERTIES LANGUAGE HIP)
  # Define USE_HIP for the compat header; attach to the interface target so it
  # travels to every consumer (core, shared, cli, tests, python bindings).
  target_compile_definitions(cupdlpx_compile_flags INTERFACE USE_HIP)
else()
  set(CORE_LINK_LIBS
    PUBLIC cupdlpx_compile_flags
    PUBLIC CUDA::cudart
    PUBLIC CUDA::cublas
    PUBLIC CUDA::cusparse
    PUBLIC ZLIB::ZLIB
    PUBLIC PSLP
  )
endif()

# 1. Core STATIC Library
if(CUPDLPX_BUILD_STATIC_LIB)
    add_library(cupdlpx_core STATIC ${C_SOURCES} ${CU_SOURCES})
    target_include_directories(cupdlpx_core ${CORE_INCLUDE_DIRS})
    target_link_libraries(cupdlpx_core ${CORE_LINK_LIBS})
    set_target_properties(cupdlpx_core PROPERTIES
      POSITION_INDEPENDENT_CODE ON
    )
    if(USE_HIP)
      # HIP compiles device code whole-program by default (-fgpu-rdc off), so
      # each object is self-contained and the archive needs no separate device
      # link. Host-only consumers (the pybind module) resolve everything at the
      # normal link step. HIP_ARCHITECTURES is inherited from CMAKE_HIP_ARCHITECTURES.
      set_target_properties(cupdlpx_core PROPERTIES
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
      )
    else()
      # CUDA uses relocatable device code here, so device symbols must be
      # resolved at the archive boundary for host-only consumers to link.
      set_target_properties(cupdlpx_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
      )
    endif()
endif()

# 2. Shared Library
if(CUPDLPX_BUILD_SHARED_LIB)
    add_library(cupdlpx_shared SHARED ${C_SOURCES} ${CU_SOURCES})
    target_include_directories(cupdlpx_shared ${CORE_INCLUDE_DIRS})
    target_link_libraries(cupdlpx_shared ${CORE_LINK_LIBS})
    set_target_properties(cupdlpx_shared PROPERTIES
        OUTPUT_NAME "cupdlpx"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    if(USE_HIP)
      set_target_properties(cupdlpx_shared PROPERTIES
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
      )
    else()
      set_target_properties(cupdlpx_shared PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
      )
    endif()
endif()

# 3. CLI Executable
if(CUPDLPX_BUILD_CLI)
    if(NOT TARGET cupdlpx_core)
        message(FATAL_ERROR "CUPDLPX_BUILD_CLI=ON requires CUPDLPX_BUILD_STATIC_LIB=ON.")
    endif()

    add_executable(cupdlpx_cli src/cli.c)
    target_include_directories(cupdlpx_cli PRIVATE include internal)
    target_link_libraries(cupdlpx_cli PRIVATE cupdlpx_core)
    set_target_properties(cupdlpx_cli PROPERTIES
        OUTPUT_NAME "cupdlpx"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    if(NOT USE_HIP)
      set_target_properties(cupdlpx_cli PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
      )
    endif()
endif()

# 4. Tests
if(CUPDLPX_BUILD_TESTS)
    if(NOT TARGET cupdlpx_core)
        message(FATAL_ERROR "CUPDLPX_BUILD_TESTS=ON requires CUPDLPX_BUILD_STATIC_LIB=ON.")
    endif()

    enable_testing()
    file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cu")
    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE cupdlpx_core)
        target_include_directories(${TEST_NAME} PRIVATE include internal)
        set_target_properties(${TEST_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tests"
        )
        if(NOT USE_HIP)
          set_target_properties(${TEST_NAME} PROPERTIES
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
          )
        endif()
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
endif()

# 5. Python Bindings
if (CUPDLPX_BUILD_PYTHON)
    if(NOT TARGET cupdlpx_core)
        message(FATAL_ERROR "CUPDLPX_BUILD_PYTHON=ON requires CUPDLPX_BUILD_STATIC_LIB=ON.")
    endif()

    find_package(pybind11 CONFIG REQUIRED)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_subdirectory(python_bindings)
endif()

# 6. Install Targets
if (CUPDLPX_BUILD_PYTHON)
    install(DIRECTORY include/ DESTINATION include/ FILES_MATCHING PATTERN "*.h")
else()
    if(TARGET cupdlpx_core)
        install(TARGETS cupdlpx_core ARCHIVE DESTINATION lib)
    endif()
    if(TARGET cupdlpx_shared)
        install(TARGETS cupdlpx_shared LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
    endif()
    if(TARGET cupdlpx_cli)
        install(TARGETS cupdlpx_cli RUNTIME DESTINATION bin)
    endif()
    install(DIRECTORY include/ DESTINATION include/ FILES_MATCHING PATTERN "*.h")
endif()
