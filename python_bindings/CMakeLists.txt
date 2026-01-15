# python_bindings/CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

# sources
set(PYBIND_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/_core_bindings.cpp)

# build the pybind11 module
pybind11_add_module(_cupdlpx_core MODULE ${PYBIND_SOURCES})

target_link_libraries(_cupdlpx_core PRIVATE 
    cupdlpx_core
)

# set rpath so that the module can find the shared libraries at runtime
set_target_properties(_cupdlpx_core PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN/../lib"
  BUILD_RPATH   "$ORIGIN/../lib"
)

# better error messages from pybind11
target_compile_definitions(_cupdlpx_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)

# install into the Python package directory
install(TARGETS PSLP
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(TARGETS _cupdlpx_core
  LIBRARY DESTINATION cupdlpx
  RUNTIME DESTINATION cupdlpx
  ARCHIVE DESTINATION cupdlpx
)
