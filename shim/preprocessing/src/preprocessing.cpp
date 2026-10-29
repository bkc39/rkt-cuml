#include "cumlrkt/preprocessing.h"

#include "cumlrkt/common.hpp"

extern "C" {

const char* cuml_preprocessing_last_error(void) {
  return cumlrkt::last_error_slot().message.data();
}

int cuml_preprocessing_last_error_kind(void) {
  return static_cast<int>(cumlrkt::last_error_slot().kind);
}

int cuml_preprocessing_check_abi(const rr_abi_tag* loaded) {
  return cumlrkt::check_abi(loaded);
}

const char* cuml_preprocessing_libcuml_version(void) {
  return cumlrkt::libcuml_release.data();
}
}
