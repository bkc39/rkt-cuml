#lang racket/base

(require (only-in ffi/unsafe
                  _bytes/nul-terminated
                  _fun
                  _int
                  _pointer
                  _string/utf-8
                  ffi-lib
                  get-ffi-obj)
         (only-in raft/unsafe raft-abi-pointer status-checker)
         (only-in "../../core.rkt" exn:fail:cuml)
         (only-in "install-native.rkt" not-staged-advice staged?))

(provide (struct-out cuml-family)
         load-family)

(struct cuml-family (library call libcuml-version))

(define (load-problem family dir loader-message)
  (if (staged? family dir)
      (format
       "libcumlrkt_~a is staged but would not load\n  found in: ~a\n  the loader said: ~a\n  check that a CUDA 13 driver is installed and LD_LIBRARY_PATH holds no other CUDA; if the library is stale, restage it: ~a"
       family
       dir
       loader-message
       not-staged-advice)
      (format "libcumlrkt_~a is not staged\n  looked in: ~a\n  ~a" family dir not-staged-advice)))

(define (load-error family dir e)
  (exn:fail:cuml
   (format "cuml/~a: ~a" family (load-problem family (simplify-path dir) (exn-message e)))
   (current-continuation-marks)
   'generic))

(define (family-entries family dir)
  (with-handlers ([exn:fail? (lambda (e) (raise (load-error family dir e)))])
    (define library (ffi-lib (build-path dir (string-append "libcumlrkt_" family))))
    (define (entry name type)
      (get-ffi-obj (format "cuml_~a_~a" family name) library type))
    (values library
            (entry "last_error" (_fun -> _bytes/nul-terminated))
            (entry "last_error_kind" (_fun -> _int))
            (entry "check_abi" (_fun _pointer -> _int))
            (entry "libcuml_version" (_fun -> _string/utf-8)))))

(define (load-family family native-libs-dir)
  (define-values (library last-error last-error-kind check-abi libcuml-version)
    (family-entries family native-libs-dir))
  (define call (status-checker last-error last-error-kind #:exn exn:fail:cuml))
  (call (string->symbol (string-append "cuml/" family)) (lambda () (check-abi (raft-abi-pointer))))
  (cuml-family library call (libcuml-version)))
