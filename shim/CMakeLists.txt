cmake_minimum_required(VERSION 3.30)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(cumlrkt VERSION 0.1.0 LANGUAGES C CXX CUDA)

include(CTest)
include(GNUInstallDirs)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_definitions(CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--threads=1>)

find_package(raftrkt CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_library(CUML_LIBRARY NAMES cuml REQUIRED)
find_path(CUML_INCLUDE_DIR cuml/version_config.hpp REQUIRED)

add_library(cuml_runtime SHARED IMPORTED)
set_target_properties(cuml_runtime PROPERTIES
  IMPORTED_LOCATION "${CUML_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${CUML_INCLUDE_DIR}"
)

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  find_library(RAFTRKT_LIBRARY NAMES raftrkt REQUIRED)
  include(GoogleTest)
endif()

include(cmake/Cumlrkt.cmake)

cumlrkt_family(cluster
  SOURCES cluster/src/blobs.cpp cluster/src/cluster.cpp
  TESTS cluster/tests/cluster_test.cpp
)

cumlrkt_add_tidy_target(tidy)
