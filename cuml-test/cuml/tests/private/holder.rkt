#lang racket/base

(require (only-in compiler/find-exe find-exe)
         (only-in racket/match match-define)
         (only-in racket/os getpid))

(provide holder-program
         start-holder
         stop-holder)

(define pr-set-pdeathsig 1)
(define sigkill 9)

(define (holder-program requires body)
  `(module holder racket/base
     (require (only-in ffi/unsafe _fun _int _ulong get-ffi-obj)
              ,@requires)
     (define prctl (get-ffi-obj "prctl" #f (_fun _int _ulong _ulong _ulong _ulong -> _int)))
     (define getppid (get-ffi-obj "getppid" #f (_fun -> _int)))
     (unless (zero? (prctl ,pr-set-pdeathsig ,sigkill 0 0 0))
       (exit 3))
     (unless (= (getppid) (string->number (vector-ref (current-command-line-arguments) 0)))
       (exit 4))
     ,@body))

(define (start-holder dir program #:name [name "holder"])
  (define path (build-path dir (string-append name ".rkt")))
  (call-with-output-file path (lambda (out) (write program out)))
  (define-values (process stdout stdin _stderr)
    (parameterize ([current-directory dir])
      (subprocess #f #f (current-error-port) (find-exe) path (number->string (getpid)))))
  (list process stdout stdin))

(define (stop-holder started)
  (match-define (list process _ stdin) started)
  (close-output-port stdin)
  (unless (sync/timeout 10 process)
    (subprocess-kill process #t)
    (sync process)))
