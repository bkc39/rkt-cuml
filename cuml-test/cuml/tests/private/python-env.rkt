#lang racket/base

(require (only-in json json-null read-json write-json)
         (only-in racket/dict in-dict)
         (only-in racket/file delete-directory/files make-temporary-directory)
         (only-in racket/match == match)
         (only-in racket/port open-output-nowhere)
         (only-in racket/string string-replace)
         (only-in racket/system system*)
         (only-in raft/private/resource with-release)
         (only-in raft/tests/private/gpu gpu-skip test-unless-skipped)
         ;; whole-module: its syntax classes come with it
         syntax/parse/define
         (only-in threading ~>))

(provide run-cases
         run-twin
         test-twin
         twin-params)

(define python (find-executable-path "python3"))

(define (call-with-twin-env thunk #:env [extra '()])
  (define env (environment-variables-copy (current-environment-variables)))
  (define driver (getenv "CUML_CUDA_DRIVER_PATH"))
  (when driver
    (environment-variables-set! env #"LD_LIBRARY_PATH" (string->bytes/utf-8 driver)))
  (for ([(name value) (in-dict extra)])
    (environment-variables-set! env (string->bytes/utf-8 name) (string->bytes/utf-8 value)))
  (parameterize ([current-environment-variables env])
    (thunk)))

(define (python-imports? module)
  (and python
       (call-with-twin-env #:env '(("PYTHONSAFEPATH" . "1"))
                           (lambda ()
                             (parameterize ([current-directory (find-system-path 'temp-dir)]
                                            [current-output-port (open-output-nowhere)]
                                            [current-error-port (open-output-nowhere)])
                               (system* python "-c" (format "import ~a" module)))))))

(define twin-skip ;; noqa
  (or gpu-skip
      (cond
        [(not python) "no Python twin (python3 is not on PATH)"]
        [(not (python-imports? "cuml")) "no Python twin (import cuml failed)"]
        [else #f])))

(define (run-twin script input)
  (with-release ([dir
                  (make-temporary-directory "cuml-twin-~a")
                  (lambda (dir) (delete-directory/files dir #:must-exist? #f))])
    (define in-path (build-path dir "in.json"))
    (define out-path (build-path dir "out.json"))
    (call-with-output-file in-path (lambda (out) (write-json input out)))
    (define err (open-output-string))
    (define ok?
      (call-with-twin-env
       (lambda ()
         (parameterize ([current-directory dir]
                        [current-error-port err])
           (system* python (path->string script) (path->string in-path) (path->string out-path))))))
    (unless ok?
      (error 'run-twin "~a failed:\n~a" script (get-output-string err)))
    (call-with-input-file out-path read-json)))

(define (run-cases script cases)
  (hash-ref (run-twin script (hasheq 'cases cases)) 'results))

(define-syntax-parse-rule (test-twin name:expr body:expr ...+)
  (test-unless-skipped twin-skip name body ...))

(define unbound-twin-params '(verbose output-type))

(define (twin-key key)
  (~> key symbol->string (regexp-replace* #rx"_" _ "-") string->symbol))

(define (twin-value value)
  (match value
    [(== (json-null)) #f]
    [(? string?) (string->symbol (string-replace value "_" "-"))]
    [_ value]))

(define (twin-params theirs unbound)
  (for*/hasheq ([(key value) (in-hash theirs)]
                [name (in-value (twin-key key))]
                #:unless (memq name (append unbound-twin-params unbound)))
    (values name (twin-value value))))
