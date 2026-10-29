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

option(CUMLRKT_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(CUMLRKT_SANITIZE)
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=address,undefined;-fno-sanitize-recover=undefined;-fno-omit-frame-pointer>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=address,-fsanitize=undefined,-fno-sanitize-recover=undefined,-fno-omit-frame-pointer>")
  add_link_options(
    "$<$<LINK_LANGUAGE:C,CXX>:-fsanitize=address,undefined>"
    "$<$<LINK_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=address,-fsanitize=undefined>")
endif()

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
  SOURCES cluster/src/blobs.cpp cluster/src/cluster.cpp cluster/src/dbscan.cpp
          cluster/src/dbscan_device.cu cluster/src/hdbscan.cpp
          cluster/src/hdbscan_predict.cpp cluster/src/kmeans.cpp
          cluster/src/linkage.cpp cluster/src/linkage_device.cu
          cluster/src/model_release.cpp
          cluster/src/spectral.cpp cluster/src/spectral_device.cu
  TESTS cluster/tests/cluster_test.cpp cluster/tests/dbscan_refusal_test.cpp
        cluster/tests/dbscan_test.cpp cluster/tests/hdbscan_refusal_test.cpp
        cluster/tests/hdbscan_test.cpp cluster/tests/kmeans_test.cpp
        cluster/tests/kmeans_refusal_test.cpp cluster/tests/linkage_refusal_test.cpp
        cluster/tests/linkage_test.cpp cluster/tests/spectral_graph_refusal_test.cpp
        cluster/tests/spectral_refusal_test.cpp
        cluster/tests/spectral_test.cpp
)

cumlrkt_family(metrics
  SOURCES metrics/src/metrics.cpp metrics/src/labels.cu metrics/src/contingency.cpp
          metrics/src/contingency_device.cu metrics/src/norms.cu
          metrics/src/silhouette.cpp
          metrics/src/trustworthiness.cpp
  TESTS metrics/tests/metrics_test.cpp metrics/tests/metrics_refusal_test.cpp
)

cumlrkt_family(preprocessing NO_LIBCUML
  SOURCES preprocessing/src/preprocessing.cpp preprocessing/src/standard_scaler.cpp
          preprocessing/src/standard_scaler_device.cu
  TESTS preprocessing/tests/preprocessing_test.cpp
        preprocessing/tests/standard_scaler_test.cpp
        preprocessing/tests/standard_scaler_refusal_test.cpp
)

cumlrkt_family(decomposition
  SOURCES decomposition/src/decomposition.cpp decomposition/src/pca.cpp
          decomposition/src/pca_device.cu
  TESTS decomposition/tests/decomposition_test.cpp
        decomposition/tests/pca_test.cpp
        decomposition/tests/pca_refusal_test.cpp
        decomposition/tests/pca_sign_test.cpp
        decomposition/tests/scale_test.cpp
        decomposition/src/pca_device.cu
)

cumlrkt_family(probe TEST_ONLY
  SOURCES probe/src/probe.cpp
  TESTS probe/tests/model_test.cpp
)

cumlrkt_add_tidy_target(tidy)
