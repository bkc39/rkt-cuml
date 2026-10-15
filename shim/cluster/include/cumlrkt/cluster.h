#ifndef CUMLRKT_CLUSTER_H
#define CUMLRKT_CLUSTER_H

#include <stddef.h>
#include <stdint.h>

#include "cumlrkt/api.h"
#include "raftrkt/abi.h"
#include "raftrkt/array.h"
#include "raftrkt/job.h"

#ifdef __cplusplus
extern "C" {
#endif

CUMLRKT_API const char* cuml_cluster_last_error(void);
CUMLRKT_API int cuml_cluster_last_error_kind(void);
CUMLRKT_API int cuml_cluster_check_abi(const rr_abi_tag* loaded);
CUMLRKT_API const char* cuml_cluster_libcuml_version(void);

CUMLRKT_API int cuml_cluster_make_blobs(void* handle, const rr_view* out,
                                        const rr_view* labels,
                                        int32_t n_clusters, double cluster_std,
                                        int32_t shuffle, double center_box_min,
                                        double center_box_max, uint64_t seed);
CUMLRKT_API void cuml_cluster_make_blobs_job(void* args, size_t args_len,
                                             rr_job_outcome* outcome);

#ifdef __cplusplus
}
#endif

#endif
