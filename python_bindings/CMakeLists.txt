# Build the pybind11 extension: pycupdlpx/_pycupdlpx_core
# NOTE: The target name becomes the Python module name unless overridden by OUTPUT_NAME.
pybind11_add_module(_pycupdlpx_core
  _core_bindings.cpp
)

# Link against your core native library
target_link_libraries(_pycupdlpx_core PRIVATE cupdlpx_core)

# Headers for the native library (adjust if your include path differs)
target_include_directories(_pycupdlpx_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../cupdlpx
)

# Put the compiled extension directly into the Python package folder
# (this eases `import pycupdlpx` during local development)
set(PY_PKG_DIR ${PROJECT_SOURCE_DIR}/python/pycupdlpx)

set_target_properties(_pycupdlpx_core PROPERTIES
  OUTPUT_NAME "_pycupdlpx_core"                 # final module filename (minus platform suffix)
  LIBRARY_OUTPUT_DIRECTORY   ${PY_PKG_DIR}      # Unix-like .so / macOS .so
  RUNTIME_OUTPUT_DIRECTORY   ${PY_PKG_DIR}      # Windows .pyd goes here
  ARCHIVE_OUTPUT_DIRECTORY   ${PY_PKG_DIR}      # static/archive artifacts if generated
)

# (Optional) Set RPATH so the extension can find adjacent shared libs at runtime.
# On Linux, "$ORIGIN" refers to the directory containing the extension.
# On macOS, "@loader_path" plays a similar role.
if(APPLE)
  set_target_properties(_pycupdlpx_core PROPERTIES
    BUILD_RPATH "@loader_path"
    INSTALL_RPATH "@loader_path"
  )
elseif(UNIX)
  set_target_properties(_pycupdlpx_core PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
  )
endif()
