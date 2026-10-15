#lang racket/base

(require (only-in compiler/find-exe find-exe)
         (only-in racket/file delete-directory/files make-temporary-directory)
         (only-in racket/match match-define)
         (only-in racket/port read-line-evt)
         (only-in rackunit check-equal? check-true)
         (only-in raft/private/resource with-release)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/device-memory.rkt" call-with-device-allocation process-device-bytes))

(define held-bytes (* 512 1024 1024))
(define tolerance (* 2 1024 1024))

(define holder
  `(module holder racket/base
     (require (only-in cuml/tests/private/device-memory call-with-device-allocation))
     (call-with-device-allocation ,held-bytes
                                  (lambda ()
                                    (displayln "held")
                                    (flush-output)
                                    (read-line)))))

(define (start-holder dir)
  (define path (build-path dir "holder.rkt"))
  (call-with-output-file path (lambda (out) (write holder out)))
  (define-values (process stdout stdin _stderr)
    (subprocess #f #f (current-error-port) (find-exe) path))
  (list process stdout stdin))

(define (stop-holder started)
  (match-define (list process _ stdin) started)
  (close-output-port stdin)
  (unless (sync/timeout 30 process)
    (subprocess-kill process #t)))

(define (measure-around-holder)
  (define before (process-device-bytes))
  (with-release ([dir (make-temporary-directory "cuml-device-memory-~a") delete-directory/files]
                 [started (start-holder dir) stop-holder])
    (match-define (list process stdout _) started)
    (check-equal? (sync/timeout 60 (read-line-evt stdout)) "held")
    (define holder-bytes (process-device-bytes #:pid (subprocess-pid process)))
    (define during (process-device-bytes))
    (printf "device memory: this process ~a bytes before, ~a while the holder (pid ~a) holds ~a\n"
            before
            during
            (subprocess-pid process)
            holder-bytes)
    (check-true (>= holder-bytes held-bytes) "NVML sees the holder's allocation")
    (check-true (<= (abs (- during before)) tolerance)
                "the holder's allocation is not counted here")))

(test-gpu "this process's device memory stays put while another process holds 512 MiB"
  (call-with-device-allocation (* 4 1024 1024) measure-around-holder))
