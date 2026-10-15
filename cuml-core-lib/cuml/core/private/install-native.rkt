#lang racket/base

(require (only-in racket/file call-with-atomic-output-file file->bytes make-directory*))

(provide native-pre-installer
         not-staged-advice
         stage-native-libs!
         staged?)

(define not-staged-advice
  "build it with nix (`nix develop` stages it), or set CUML_NATIVE_LIB_PATH to a directory whose lib/ holds it and reinstall")

(define (library-pattern family)
  (regexp (string-append "^libcumlrkt_" (regexp-quote family) "[.]")))

(define (staged? family dir)
  (and (directory-exists? dir)
       (for/or ([f (in-list (directory-list dir))])
         (regexp-match? (library-pattern family) f))))

(define (same-bytes? a b)
  (and (file-exists? b) (= (file-size a) (file-size b)) (equal? (file->bytes a) (file->bytes b))))

(define (stage-file! src dst)
  (unless (same-bytes? src dst)
    (call-with-atomic-output-file dst
                                  (lambda (out tmp)
                                    (write-bytes (file->bytes src) out)
                                    (file-or-directory-permissions tmp #o555)))))

(define (stage-native-libs! family source-dir dest-dir)
  (make-directory* dest-dir)
  (for ([f (in-list (directory-list source-dir))]
        #:when (regexp-match? (library-pattern family) f))
    (stage-file! (build-path source-dir f) (build-path dest-dir f))))

(define ((native-pre-installer family) _collections-top-path this-collection-path _user-specific?)
  (define native-libs-dir (build-path this-collection-path "native-libs"))
  (define override (getenv "CUML_NATIVE_LIB_PATH"))
  (cond
    [override (stage-native-libs! family (build-path override "lib") native-libs-dir)]
    [(staged? family native-libs-dir) (void)]
    [else
     (eprintf "cuml/~a: libcumlrkt_~a is not staged in ~a; ~a\n"
              family
              family
              native-libs-dir
              not-staged-advice)]))
