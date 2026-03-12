# -----------------------------------------------------------------------------
# ROOT CMakeLists.txt for cuPDLPx (Unified Build System)
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)

# Project config
project(cupdlpx LANGUAGES C CXX CUDA)

set(CUPDLPX_VERSION_MAJOR 0)
set(CUPDLPX_VERSION_MINOR 2)
set(CUPDLPX_VERSION_PATCH 7)

set(CUPDLPX_VERSION "${CUPDLPX_VERSION_MAJOR}.${CUPDLPX_VERSION_MINOR}.${CUPDLPX_VERSION_PATCH}")
add_compile_definitions(CUPDLPX_VERSION="${CUPDLPX_VERSION}")

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

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90)
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

# CUDA standards and RDC
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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
find_package(CUDAToolkit REQUIRED)
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

set(CORE_INCLUDE_DIRS
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/internal
)

set(CORE_LINK_LIBS 
  PUBLIC cupdlpx_compile_flags
  PUBLIC CUDA::cudart
  PUBLIC CUDA::cublas
  PUBLIC CUDA::cusparse
  PUBLIC ZLIB::ZLIB
  PUBLIC PSLP
)

# 1. Core STATIC Library
if(CUPDLPX_BUILD_STATIC_LIB)
    add_library(cupdlpx_core STATIC ${C_SOURCES} ${CU_SOURCES})
    target_include_directories(cupdlpx_core ${CORE_INCLUDE_DIRS})
    target_link_libraries(cupdlpx_core ${CORE_LINK_LIBS})
    set_target_properties(cupdlpx_core PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# 2. Shared Library
if(CUPDLPX_BUILD_SHARED_LIB)
    add_library(cupdlpx_shared SHARED ${C_SOURCES} ${CU_SOURCES})
    target_include_directories(cupdlpx_shared ${CORE_INCLUDE_DIRS})
    target_link_libraries(cupdlpx_shared ${CORE_LINK_LIBS})
    set_target_properties(cupdlpx_shared PROPERTIES
        OUTPUT_NAME "cupdlpx"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
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
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
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
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
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
