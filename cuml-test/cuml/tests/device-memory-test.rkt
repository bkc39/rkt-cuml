#lang racket/base

(require (only-in racket/file delete-directory/files file->lines file->string make-temporary-directory)
         (only-in racket/match match-define)
         (only-in racket/port read-line-evt with-output-to-string)
         ;; whole-module: define-runtime-path needs bindings only-in strips
         racket/runtime-path
         (only-in rackunit check-equal? check-false check-regexp-match check-true test-case)
         (only-in raft/private/resource with-release)
         (only-in raft/tests/private/gpu gpu-skip test-gpu test-unless-skipped)
         (only-in "private/device-memory.rkt"
                  call-with-device-allocation
                  no-process-memory-reason
                  process-device-bytes
                  process-memory-skip)
         (only-in "private/holder.rkt" holder-program start-holder stop-holder))

(define held-bytes (* 512 1024 1024))
(define tolerance (* 2 1024 1024))

(define holding
  (holder-program '((only-in cuml/tests/private/device-memory call-with-device-allocation))
                  `((call-with-device-allocation ,held-bytes
                                                 (lambda ()
                                                   (displayln "held")
                                                   (flush-output)
                                                   (read-line))))))

(define (measure-around-holder)
  (define before (process-device-bytes))
  (with-release ([dir (make-temporary-directory "cuml-device-memory-~a") delete-directory/files]
                 [started (start-holder dir holding) stop-holder])
    (match-define (list process stdout _) started)
    (check-equal? (sync/timeout 20 (read-line-evt stdout)) "held")
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

(test-unless-skipped (or gpu-skip (process-memory-skip))
                     "this process's device memory stays put while another process holds 512 MiB"
  (call-with-device-allocation (* 4 1024 1024) measure-around-holder))

(test-gpu "a device NVML does not list skips the per-process measurement"
  (check-false (process-memory-skip))
  (check-regexp-match #rx"^no per-process device memory .*MIG-00000000-0000-0000-0000-000000000000"
                      (process-memory-skip #:uuid "MIG-00000000-0000-0000-0000-000000000000")))

(define stuck-in-middle
  (holder-program
   '((only-in cuml/tests/private/holder holder-program start-holder) (only-in racket/match
                                                                              match-define))
   '((match-define (list holder stdout _)
       (start-holder (current-directory)
                     (holder-program '() '((displayln "held") (flush-output) (sleep 60)))))
     (displayln (read-line stdout))
     (writeln (subprocess-pid holder))
     (flush-output)
     (sleep 60))))

(define (process-gone? pid)
  (define stat (build-path "/proc" (number->string pid) "stat"))
  (or (not (file-exists? stat))
      (with-handlers ([exn:fail:filesystem? (lambda (_) #t)])
        (regexp-match? #rx"[)] Z " (file->string stat)))))

(define (gone-within? pid seconds)
  (for/or ([_ (in-range (* 10 seconds))])
    (sleep 1/10)
    (process-gone? pid)))

(test-unless-skipped (and (not (eq? (system-type 'os*) 'linux))
                          "the holder's parent watch is Linux-only")
                     "a holder stuck before reading dies with the process that started it"
  (with-release ([dir (make-temporary-directory "cuml-holder-~a") delete-directory/files]
                 [middle (start-holder dir stuck-in-middle #:name "middle") stop-holder])
    (match-define (list process stdout _) middle)
    (check-equal? (sync/timeout 20 (read-line-evt stdout)) "held")
    (define holder-pid (string->number (sync/timeout 20 (read-line-evt stdout))))
    (subprocess-kill process #t)
    (check-true (gone-within? holder-pid 5)
                (format "the holder (pid ~a) outlived its killed parent" holder-pid))))

(define-runtime-path allowed-skips "../../../scripts/allowed-skips.txt")

(define (skip-line reason)
  (with-output-to-string (lambda ()
                           (test-unless-skipped reason "x"
                             (void)))))

(define (allowed? line)
  (for/or ([pattern (in-list (file->lines allowed-skips))])
    (regexp-match? (pregexp pattern) line)))

(test-case "the gpu gate allows the MIG SKIP line and no other GPU or twin SKIP"
  (check-true (allowed? (skip-line (no-process-memory-reason "MIG-0123abcd-0000-0000-0000-000000000000"))))
  (check-true (allowed? (skip-line "a GPU is present")))
  (check-false (allowed? (skip-line "no GPU (the driver reports no CUDA device)")))
  (check-false (allowed? (skip-line "no Python twin (import cuml failed)"))))
