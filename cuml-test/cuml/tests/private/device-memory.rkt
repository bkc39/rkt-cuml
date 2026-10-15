#lang racket/base

(require (only-in ffi/unsafe
                  _bytes
                  _fun
                  _int
                  _list
                  _pointer
                  _ptr
                  _size
                  _string
                  _uint
                  _uint64
                  define-cstruct
                  ffi-lib)
         (only-in ffi/unsafe/define define-ffi-definer make-not-available)
         (only-in racket/format ~r)
         (only-in racket/os getpid)
         (only-in racket/string string-append*)
         (only-in raft/private/resource with-release))

(provide call-with-device-allocation
         process-device-bytes)

(define-ffi-definer define-cuda
                    (ffi-lib "libcuda" '("1") #:fail (lambda () #f))
                    #:default-make-fail make-not-available)

(define-ffi-definer define-nvml
                    (ffi-lib "libnvidia-ml" '("1") #:fail (lambda () #f))
                    #:default-make-fail make-not-available)

(define-cstruct _process-info
  ([pid _uint]
   [used-bytes _uint64]
   [gpu-instance _uint]
   [compute-instance _uint]))

(define nvml-insufficient-size 7)
(define nvml-value-not-available #xFFFFFFFFFFFFFFFF)

(define-cuda cu-init (_fun _uint -> _int) #:c-id cuInit)
(define-cuda cu-device
             (_fun (device : (_ptr o _int)) _int -> (status : _int) -> (values status device))
             #:c-id cuDeviceGet)
(define-cuda cu-device-uuid
             (_fun (uuid : _bytes = (make-bytes 16)) _int -> (status : _int) -> (values status uuid))
             #:c-id cuDeviceGetUuid_v2)
(define-cuda cu-primary-retain
             (_fun (context : (_ptr o _pointer)) _int -> (status : _int) -> (values status context))
             #:c-id cuDevicePrimaryCtxRetain)
(define-cuda cu-primary-release (_fun _int -> _int) #:c-id cuDevicePrimaryCtxRelease_v2)
(define-cuda cu-context-push (_fun _pointer -> _int) #:c-id cuCtxPushCurrent_v2)
(define-cuda cu-context-pop
             (_fun (context : (_ptr o _pointer)) -> (status : _int) -> status)
             #:c-id cuCtxPopCurrent_v2)
(define-cuda cu-allocate
             (_fun (address : (_ptr o _uint64)) _size -> (status : _int) -> (values status address))
             #:c-id cuMemAlloc_v2)
(define-cuda cu-free (_fun _uint64 -> _int) #:c-id cuMemFree_v2)

(define-nvml nvml-init (_fun -> _int) #:c-id nvmlInit_v2)
(define-nvml nvml-shutdown (_fun -> _int) #:c-id nvmlShutdown)
(define-nvml nvml-device-by-uuid
             (_fun _string (device : (_ptr o _pointer)) -> (status : _int) -> (values status device))
             #:c-id nvmlDeviceGetHandleByUUID)
(define-nvml nvml-compute-processes
             (_fun (device capacity)
                   ::
                   (device : _pointer)
                   (count : (_ptr io _uint) = capacity)
                   (infos : (_list o _process-info capacity))
                   -> (status : _int)
                   -> (values status
                              count
                              (for/list ([info (in-list infos)]
                                         [_ (in-range count)])
                                (cons (process-info-pid info) (process-info-used-bytes info)))))
             #:c-id nvmlDeviceGetComputeRunningProcesses_v3)

(define (cuda-ok who status)
  (unless (zero? status)
    (error who "CUDA driver error ~a" status)))

(define (nvml-ok who status)
  (unless (zero? status)
    (error who "NVML error ~a" status)))

(define (checked who call . arguments)
  (define-values (status result) (apply call arguments))
  (cuda-ok who status)
  result)

(define (device-uuid device)
  (cuda-ok 'cuInit (cu-init 0))
  (define cuda-device (checked 'cuDeviceGet cu-device device))
  (define uuid (checked 'cuDeviceGetUuid cu-device-uuid cuda-device))
  (define hex
    (string-append* (for/list ([byte (in-bytes uuid)])
                      (~r byte #:base 16 #:min-width 2 #:pad-string "0"))))
  (format "GPU-~a-~a-~a-~a-~a"
          (substring hex 0 8)
          (substring hex 8 12)
          (substring hex 12 16)
          (substring hex 16 20)
          (substring hex 20)))

(define (nvml-session)
  (nvml-ok 'nvmlInit (nvml-init)))

(define (end-nvml-session _session)
  (nvml-ok 'nvmlShutdown (nvml-shutdown)))

(define (compute-processes handle [capacity 64])
  (define-values (status count processes) (nvml-compute-processes handle capacity))
  (cond
    [(= status nvml-insufficient-size) (compute-processes handle (+ count 16))]
    [else
     (nvml-ok 'nvmlDeviceGetComputeRunningProcesses status)
     processes]))

(define (process-device-bytes #:device [device 0] #:pid [pid (getpid)])
  (define uuid (device-uuid device))
  (with-release ([_session (nvml-session) end-nvml-session])
    (define-values (status handle) (nvml-device-by-uuid uuid))
    (nvml-ok 'nvmlDeviceGetHandleByUUID status)
    (define used
      (cond
        [(assv pid (compute-processes handle))
         =>
         cdr]
        [else 0]))
    (when (= used nvml-value-not-available)
      (error 'process-device-bytes "NVML does not report per-process memory on ~a" uuid))
    used))

(define (primary-context device)
  (define-values (status context) (cu-primary-retain device))
  (cuda-ok 'cuDevicePrimaryCtxRetain status)
  context)

(define ((release-primary-context device) _context)
  (cuda-ok 'cuDevicePrimaryCtxRelease (cu-primary-release device)))

(define (push-context context)
  (cuda-ok 'cuCtxPushCurrent (cu-context-push context))
  context)

(define (pop-context _context)
  (cuda-ok 'cuCtxPopCurrent (cu-context-pop)))

(define (free-device-memory address)
  (cuda-ok 'cuMemFree (cu-free address)))

(define (call-with-device-allocation size proc #:device [device 0])
  (cuda-ok 'cuInit (cu-init 0))
  (with-release ([primary (primary-context device) (release-primary-context device)]
                 [_current (push-context primary) pop-context]
                 [_address (checked 'cuMemAlloc cu-allocate size) free-device-memory])
    (proc)))
