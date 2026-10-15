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

(define (library-error family dir loader-message)
  (if (staged? family dir)
      (format
       "libcumlrkt_~a is staged but would not load\n  found in: ~a\n  the loader said: ~a\n  check that a CUDA 13 driver is installed and LD_LIBRARY_PATH holds no other CUDA"
       family
       dir
       loader-message)
      (format "libcumlrkt_~a is not staged\n  looked in: ~a\n  ~a" family dir not-staged-advice)))

(define (family-library family dir)
  (with-handlers ([exn:fail? (lambda (e)
                               (raise (exn:fail:cuml (format "cuml/~a: ~a"
                                                             family
                                                             (library-error family
                                                                            (simplify-path dir)
                                                                            (exn-message e)))
                                                     (current-continuation-marks)
                                                     'generic)))])
    (ffi-lib (build-path dir (string-append "libcumlrkt_" family)))))

(define (load-family family native-libs-dir)
  (define library (family-library family native-libs-dir))
  (define (entry name type)
    (get-ffi-obj (format "cuml_~a_~a" family name) library type))
  (define call
    (status-checker (entry "last_error" (_fun -> _bytes/nul-terminated))
                    (entry "last_error_kind" (_fun -> _int))
                    #:exn exn:fail:cuml))
  (call (string->symbol (string-append "cuml/" family))
        (lambda () ((entry "check_abi" (_fun _pointer -> _int)) (raft-abi-pointer))))
  (cuml-family library call ((entry "libcuml_version" (_fun -> _string/utf-8)))))
