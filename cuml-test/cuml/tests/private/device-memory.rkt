#lang racket/base

(require (only-in ffi/unsafe _fun _int _pointer _ptr _size _uint ffi-lib)
         (only-in ffi/unsafe/define define-ffi-definer make-not-available)
         (only-in raft/private/resource with-release))

(provide device-free-bytes)

(define-ffi-definer define-cuda
                    (ffi-lib "libcuda" '("1") #:fail (lambda () #f))
                    #:default-make-fail make-not-available)

(define-cuda cu-init (_fun _uint -> _int) #:c-id cuInit)
(define-cuda cu-primary-retain
             (_fun (context : (_ptr o _pointer)) _int -> (status : _int) -> (values status context))
             #:c-id cuDevicePrimaryCtxRetain)
(define-cuda cu-primary-release (_fun _int -> _int) #:c-id cuDevicePrimaryCtxRelease_v2)
(define-cuda cu-context-push (_fun _pointer -> _int) #:c-id cuCtxPushCurrent_v2)
(define-cuda cu-context-pop
             (_fun (context : (_ptr o _pointer)) -> (status : _int) -> status)
             #:c-id cuCtxPopCurrent_v2)
(define-cuda cu-memory-info
             (_fun (free-bytes : (_ptr o _size)) (total : (_ptr o _size))
                   -> (status : _int)
                   -> (values status free-bytes))
             #:c-id cuMemGetInfo_v2)

(define (cuda-ok who status)
  (unless (zero? status)
    (error who "CUDA driver error ~a" status)))

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

(define (device-free-bytes [device 0])
  (cuda-ok 'cuInit (cu-init 0))
  (with-release ([primary (primary-context device) (release-primary-context device)]
                 [_current (push-context primary) pop-context])
    (define-values (status free-bytes) (cu-memory-info))
    (cuda-ok 'cuMemGetInfo status)
    free-bytes))
