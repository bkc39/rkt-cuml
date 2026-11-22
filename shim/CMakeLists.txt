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
          cluster/src/spectral.cpp common/src/graph.cu
  TESTS cluster/tests/cluster_test.cpp cluster/tests/dbscan_refusal_test.cpp
        cluster/tests/dbscan_test.cpp cluster/tests/hdbscan_refusal_test.cpp
        cluster/tests/hdbscan_test.cpp cluster/tests/kmeans_test.cpp
        cluster/tests/kmeans_refusal_test.cpp cluster/tests/linkage_refusal_test.cpp
        cluster/tests/linkage_test.cpp cluster/tests/spectral_graph_refusal_test.cpp
        cluster/tests/spectral_refusal_test.cpp
        cluster/tests/spectral_test.cpp
)

cumlrkt_family(metrics
  SOURCES metrics/src/metrics.cpp metrics/src/contingency.cpp
          metrics/src/contingency_device.cu
          metrics/src/scores.cpp metrics/src/scores_device.cu
          metrics/src/silhouette.cpp
          metrics/src/trustworthiness.cpp
  TESTS metrics/tests/metrics_test.cpp metrics/tests/metrics_refusal_test.cpp
        metrics/tests/scores_test.cpp
)

cumlrkt_family(preprocessing NO_LIBCUML
  SOURCES preprocessing/src/preprocessing.cpp preprocessing/src/standard_scaler.cpp
          preprocessing/src/standard_scaler_device.cu
  TESTS preprocessing/tests/preprocessing_test.cpp
        preprocessing/tests/standard_scaler_test.cpp
        preprocessing/tests/standard_scaler_refusal_test.cpp
)

cumlrkt_family(decomposition
  SOURCES common/src/transpose.cu decomposition/src/decomposition.cpp
          decomposition/src/pca.cpp decomposition/src/pca_device.cu
          decomposition/src/tsvd.cpp decomposition/src/tsvd_device.cu
  TESTS decomposition/tests/decomposition_test.cpp
        decomposition/tests/pca_test.cpp
        decomposition/tests/pca_refusal_test.cpp
        decomposition/tests/pca_sign_test.cpp
        decomposition/tests/scale_test.cpp
        decomposition/tests/tsvd_test.cpp
        decomposition/tests/tsvd_refusal_test.cpp
        decomposition/src/pca_device.cu
)

# The weighted softmax compiles cuML's header-only qn solver, so its source
# must be the release of the linked libcuml.
find_path(CUML_SOURCE_DIR glm/qn/qn.cuh HINTS $ENV{CUML_SOURCE_DIR} REQUIRED)
file(STRINGS "${CUML_INCLUDE_DIR}/cuml/version_config.hpp" cuml_version_defines
  REGEX "^#define CUML_VERSION_(MAJOR|MINOR|PATCH) ")
set(cuml_version)
foreach(part MAJOR MINOR PATCH)
  string(REGEX MATCH "CUML_VERSION_${part} ([0-9]+)" _ "${cuml_version_defines}")
  set(n "${CMAKE_MATCH_1}")
  if(NOT part STREQUAL "MAJOR" AND n LESS 10)
    set(n "0${n}")
  endif()
  list(APPEND cuml_version "${n}")
endforeach()
list(JOIN cuml_version "." cuml_version)
set(cuml_source_version_file "${CUML_SOURCE_DIR}/../../VERSION")
set(cuml_source_version "none")
if(EXISTS "${cuml_source_version_file}")
  file(STRINGS "${cuml_source_version_file}" cuml_source_version LIMIT_COUNT 1)
endif()
if(NOT cuml_source_version STREQUAL cuml_version)
  message(FATAL_ERROR
    "cuML source ${CUML_SOURCE_DIR} is version ${cuml_source_version}, the "
    "linked libcuml ${cuml_version}: bump cumlSrc in flake.nix to v${cuml_version}")
endif()

cumlrkt_family(linear_model
  SOURCES linear_model/src/linear_model.cpp
          linear_model/src/glm_fit.cpp linear_model/src/linear_regression.cpp
          linear_model/src/ridge.cpp linear_model/src/ridge_staging.cpp
          linear_model/src/gram_condition.cu linear_model/src/ritz.cpp
          linear_model/src/block_gram.cu linear_model/src/conditioning.cpp
          linear_model/src/spectrum_enclosure.cpp
          linear_model/src/symmetric_eigen.cpp
          linear_model/src/fit_checks.cpp linear_model/src/predict.cpp
          linear_model/src/glm_staging.cpp
          linear_model/src/glm_staging_device.cu
          linear_model/src/residual_check.cu common/src/transpose.cu
          linear_model/src/elastic_net.cpp linear_model/src/cd_staging.cpp
          linear_model/src/cd_device.cu linear_model/src/logistic.cpp
          linear_model/src/logistic_apply.cpp linear_model/src/logistic_device.cu
          linear_model/src/logistic_apply_device.cu
          linear_model/src/logistic_resolve.cpp
          linear_model/src/logistic_solve.cpp
          linear_model/src/logistic_weights.cpp
          linear_model/src/logistic_softmax.cu
          linear_model/src/mbsgd.cpp linear_model/src/mbsgd_device.cu
          common/src/labels.cu
  TESTS linear_model/tests/glm_staging_test.cpp
        linear_model/tests/cd_staging_test.cpp
        linear_model/tests/cd_zero_fit_test.cpp
        linear_model/src/cd_staging.cpp linear_model/src/cd_device.cu
        linear_model/tests/linear_model_refusal_test.cpp
        linear_model/tests/logistic_checks_test.cpp
        linear_model/tests/logistic_rescue_test.cpp
        linear_model/src/logistic_solve.cpp
        linear_model/src/logistic_resolve.cpp
        linear_model/src/logistic_device.cu
        linear_model/src/logistic_softmax.cu
        linear_model/tests/logistic_softmax_test.cpp
        linear_model/tests/mbsgd_checks_test.cpp
        linear_model/tests/mbsgd_round_off_test.cpp
        linear_model/tests/ridge_staging_test.cpp
        linear_model/tests/gram_condition_test.cpp
        linear_model/tests/ritz_plan_test.cpp
        linear_model/tests/gram_error_test.cpp
        linear_model/tests/gram_proof_test.cpp
        linear_model/tests/column_sums_test.cu
        linear_model/src/ridge_staging.cpp
        linear_model/src/gram_condition.cu
        linear_model/src/ritz.cpp
        linear_model/src/block_gram.cu linear_model/src/conditioning.cpp
        linear_model/src/spectrum_enclosure.cpp
        linear_model/src/symmetric_eigen.cpp
        linear_model/src/glm_staging.cpp
        linear_model/src/glm_staging_device.cu
        linear_model/src/residual_check.cu
        linear_model/src/fit_checks.cpp common/src/rows.cu
)
target_link_libraries(cumlrkt_linear_model PRIVATE CUDA::cusolver CUDA::cublas)
target_include_directories(cumlrkt_linear_model SYSTEM PRIVATE ${CUML_SOURCE_DIR})
if(TARGET cumlrkt_linear_model_tests)
  target_link_libraries(cumlrkt_linear_model_tests
    PRIVATE CUDA::cusolver CUDA::cublas cuml_runtime)
  target_include_directories(cumlrkt_linear_model_tests SYSTEM PRIVATE
    ${CUML_SOURCE_DIR})
endif()

cumlrkt_family(manifold
  SOURCES common/src/transpose.cu manifold/src/manifold.cpp
          manifold/src/manifold_support.cpp manifold/src/tsne.cpp
          manifold/src/spectral_embedding.cpp manifold/src/spectral_residual.cu
          common/src/graph.cu
          common/src/largest_positive.cu manifold/src/umap.cpp manifold/src/umap_device.cu
  TESTS manifold/tests/manifold_test.cpp manifold/tests/tsne_test.cpp
        manifold/tests/tsne_refusal_test.cpp manifold/tests/umap_test.cpp
        manifold/tests/umap_spectral_test.cpp
        manifold/tests/umap_refusal_test.cpp
        manifold/tests/spectral_embedding_test.cpp
        manifold/tests/spectral_embedding_refusal_test.cpp common/src/rows.cu
        common/src/largest_positive.cu manifold/src/spectral_residual.cu
)

cumlrkt_family(svm
  SOURCES svm/src/svm.cpp svm/src/svc.cpp svm/src/svm_apply.cpp
          svm/src/gamma.cu svm/src/svm_device.cu common/src/transpose.cu
  TESTS svm/tests/ownership_test.cpp svm/tests/model_guard_test.cpp
        svm/tests/kernel_checks_test.cpp svm/tests/gamma_test.cpp
        svm/tests/svc_refusal_test.cpp svm/src/gamma.cu svm/src/svm_device.cu
        common/src/transpose.cu common/src/rows.cu
)
if(TARGET cumlrkt_svm_tests)
  target_link_libraries(cumlrkt_svm_tests PRIVATE cuml_runtime)
endif()

cumlrkt_family(probe TEST_ONLY
  SOURCES probe/src/probe.cpp
  TESTS probe/tests/model_test.cpp
)

cumlrkt_add_tidy_target(tidy)
