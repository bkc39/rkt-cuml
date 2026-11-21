#lang racket/base

(require (only-in json json-null read-json write-json)
         (only-in racket/dict in-dict)
         (only-in racket/file delete-directory/files make-temporary-directory)
         (only-in racket/list append*)
         (only-in racket/match == match match-define)
         (only-in racket/port open-output-nowhere)
         (only-in racket/promise delay force)
         (only-in racket/string string-replace)
         (only-in racket/system system*)
         (only-in rackunit check-equal?)
         (only-in raft/private/resource with-release)
         (only-in raft/tests/private/gpu gpu-skip test-unless-skipped)
         ;; whole-module: its syntax classes come with it
         syntax/parse/define
         (only-in threading ~>))

(provide bits-of
         checked-results
         decoded
         matrix->bits
         or-null
         python-cuml-version
         python-sklearn-version
         run-twin
         test-python
         test-twin
         twin-default-params
         twin-params
         twin-results)

(define python-cuml-version "26.08.00")

(define python-sklearn-version "1.8.0")

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

(define python-skip
  (delay
    (cond
      [(not python) "no Python twin (python3 is not on PATH)"]
      [(not (python-imports? "numpy")) "no Python twin (import numpy failed)"]
      [else #f])))

(define twin-skip ;; noqa
  (delay
    (or gpu-skip
        (force python-skip)
        (and (not (python-imports? "cuml")) "no Python twin (import cuml failed)"))))

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

(define (checked-results answer #:expected-cuml [expected #f])
  (match answer
    [(hash-table ('results results) ('version (hash-table ('cuml version))))
     (check-equal? version python-cuml-version "the twin's Python cuML is not the pinned release")
     (when expected
       (check-equal? version expected "the twin's Python cuML is not the native libcuml"))
     results]
    [(hash-table ('results _))
     (error 'twin-results
            "the twin wrote no version.cuml (got ~e); write its output with twin_common.run"
            (hash-ref answer 'version #f))]
    [_ (error 'twin-results "the twin wrote no results; write its output with twin_common.run")]))

(define (twin-results script
                      requests
                      #:expected-cuml [expected #f]
                      #:fixtures [fixtures #f]
                      #:run [run run-twin])
  (define request
    (for/hasheq ([(key value) (in-dict `((cases . ,requests) (fixtures . ,fixtures)))]
                 #:when value)
      (values key value)))
  (checked-results (run script request) #:expected-cuml expected))

(define (twin-default-params script #:unbound [unbound '()] #:expected-cuml [expected #f])
  (match-define (list theirs)
    (twin-results script (list (hasheq 'op "params")) #:expected-cuml expected))
  (twin-params theirs #:unbound unbound))

(define-syntax-parse-rule (test-twin name:expr body:expr ...+)
  (test-unless-skipped (force twin-skip) name body ...))

(define-syntax-parse-rule (test-python name:expr body:expr ...+)
  (test-unless-skipped (force python-skip) name body ...))

(define unbound-twin-params '(verbose output-type))

(define (twin-key key)
  (~> key symbol->string (regexp-replace* #rx"_" _ "-") string->symbol))

(define (twin-value value)
  (match value
    [(== (json-null)) #f]
    [(? string?) (string->symbol (string-replace value "_" "-"))]
    [_ value]))

(define (twin-params theirs #:unbound [unbound '()])
  (for*/hasheq ([(key value) (in-hash theirs)]
                [name (in-value (twin-key key))]
                #:unless (memq name (append unbound-twin-params unbound)))
    (values name (twin-value value))))

(define (or-null v)
  (or v (json-null)))

(define (float-size dt)
  (if (eq? dt 'float32) 4 8))

(define (bits-of dt xs)
  (for/list ([x (in-list xs)])
    (integer-bytes->integer (real->floating-point-bytes x (float-size dt)) #f)))

(define (decoded dt bits)
  (for/list ([b (in-list bits)])
    (floating-point-bytes->real (integer->integer-bytes b (float-size dt) #f))))

(define (matrix->bits dt rows)
  (bits-of dt (append* rows)))
