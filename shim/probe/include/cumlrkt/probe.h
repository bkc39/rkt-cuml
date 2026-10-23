#ifndef CUMLRKT_PROBE_H
#define CUMLRKT_PROBE_H

#include <stddef.h>
#include <stdint.h>

#include "cumlrkt/api.h"
#include "raftrkt/abi.h"
#include "raftrkt/array.h"
#include "raftrkt/deposit.h"
#include "raftrkt/job.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CUML_PROBE_KIND_A 1
#define CUML_PROBE_KIND_B 2

CUMLRKT_API const char* cuml_probe_last_error(void);
CUMLRKT_API int cuml_probe_last_error_kind(void);
CUMLRKT_API int cuml_probe_check_abi(const rr_abi_tag* loaded);
CUMLRKT_API const char* cuml_probe_libcuml_version(void);

/* Deposits {anchor, int32[region_length] = 0..region_length-1} in out, and
   holds payload_bytes of opaque device memory that release frees. */
CUMLRKT_API int cuml_probe_model_create(void* handle, rr_deposit* out,
                                        int64_t payload_bytes,
                                        int32_t region_length, int32_t kind);
CUMLRKT_API void cuml_probe_model_create_job(void* args, size_t args_len,
                                             rr_job_outcome* outcome);

/* probe-a only. Waits for gate to open when gate >= 0, at most 5 s, then
   writes the payload on the stream of the handle that created the model,
   synchronises it, and answers the payload's size. */
CUMLRKT_API int cuml_probe_model_touch(void* handle, const rr_view* anchor,
                                       int32_t gate, int64_t* payload_bytes);
CUMLRKT_API void cuml_probe_model_touch_job(void* args, size_t args_len,
                                            rr_job_outcome* outcome);

CUMLRKT_API int cuml_probe_model_release(void* handle, const rr_view* anchor);
CUMLRKT_API void cuml_probe_model_release_job(void* args, size_t args_len,
                                              rr_job_outcome* outcome);

/* stale counts destructors that found the creating stream gone. */
CUMLRKT_API int cuml_probe_model_counts(int64_t* created, int64_t* released,
                                        int64_t* destroyed, int64_t* stale);

CUMLRKT_API int cuml_probe_gate_create(int32_t* gate);
CUMLRKT_API int cuml_probe_gate_open(int32_t gate);
/* How many calls are waiting at gate now. */
CUMLRKT_API int cuml_probe_gate_waiters(int32_t gate, int32_t* waiters);

#ifdef __cplusplus
}
#endif

#endif
