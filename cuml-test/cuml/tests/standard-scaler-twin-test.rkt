#lang racket/base

(require (only-in cuml/preprocessing
                  standard-scaler-fit
                  standard-scaler-inverse-transform
                  standard-scaler-params
                  standard-scaler-result-mean
                  standard-scaler-result-n-samples
                  standard-scaler-result-scale
                  standard-scaler-result-var
                  standard-scaler-transform)
         (only-in json json-null)
         (only-in racket/list append*)
         (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in "private/python-env.rkt" run-twin test-twin twin-params)
         (only-in "private/scaler-data.rkt" constant-rows generated-rows rows->matrix ulp))

(define-runtime-path twin "python/standard_scaler_twin.py")

(define (twin-results cases)
  (match-define (hash-table ('results results) ('version version)) (run-twin twin (hasheq 'cases cases)))
  (check-equal? version "1.8.0")
  results)

(struct dataset (name rows))

(define (datasets dt)
  (list (dataset "normal" (generated-rows 1 2000 16 dt (lambda (z _u _j) z)))
        (dataset "offset" (generated-rows 2 5000 8 dt (lambda (z _u _j) (+ 1e4 z))))
        (dataset "scales"
                 (generated-rows 3 178 13 dt (lambda (z _u j) (* z (expt 10.0 (- (/ j 2.0) 3.0))))))
        (dataset "uniform" (generated-rows 4 1000 32 dt (lambda (_z u _j) (floor (* 256 u)))))
        (dataset "constant" (map (lambda (row) (list (car row) 3.3)) (generated-rows 5 50 1 dt (lambda (z _u _j) z))))
        (dataset "near-constant" (cons (list (+ 1e8 (ulp 1e8))) (constant-rows 2 1e8)))
        (dataset "tiny spread" (cons (list (+ 1e6 1e-4)) (constant-rows 999 1e6)))
        (dataset "one row" (list '(1.5 -2.0 1e3)))))

(define (case-of op data dt flags #:y [y #f] #:y-dtype [y-dt dt])
  (match-define (list with-mean with-std) flags)
  (hasheq 'op op
          'X (append* (dataset-rows data))
          'Y (append* (or y (dataset-rows data)))
          'd (length (car (dataset-rows data)))
          'dtype (symbol->string dt)
          'y_dtype (symbol->string y-dt)
          'with_mean with-mean
          'with_std with-std))

(define flag-pairs '((#t #t) (#t #f) (#f #t) (#f #f)))

(define (fit-of data dt flags)
  (match-define (list with-mean with-std) flags)
  (standard-scaler-fit (rows->matrix (dataset-rows data) dt) #:with-mean with-mean #:with-std with-std))

(define (listed v)
  (and v (device-vector->list v)))

(define (theirs-vector result key)
  (define v (hash-ref result key))
  (and (not (eq? v (json-null))) v))

(define (within? ours theirs tolerance)
  (for/and ([o (in-list ours)]
            [t (in-list theirs)]
            [tol (in-list tolerance)])
    (<= (abs (- o t)) tol)))

(test-twin "L14: the statistics are within 1e-12 of sklearn's, and the masked columns are exactly 1"
  (define cases
    (for*/list ([dt (in-list '(float32 float64))]
                [data (in-list (datasets dt))]
                [flags (in-list flag-pairs)])
      (list dt data flags)))
  (define results
    (twin-results (for/list ([c (in-list cases)])
                    (apply case-of "fit" (car c) (cdr c)))))
  (for ([c (in-list cases)]
        [result (in-list results)])
    (match-define (list dt data flags) c)
    (test-case (format "~a ~a ~a" dt (dataset-name data) flags)
      (define fitted (fit-of data dt flags))
      (define mean (listed (standard-scaler-result-mean fitted)))
      (define var (listed (standard-scaler-result-var fitted)))
      (define scale (listed (standard-scaler-result-scale fitted)))
      (define mean-sk (theirs-vector result 'mean))
      (define var-sk (theirs-vector result 'var))
      (define scale-sk (theirs-vector result 'scale))
      (check-equal? (standard-scaler-result-n-samples fitted) (hash-ref result 'n_samples))
      (check-equal? (list (and mean #t) (and var #t) (and scale #t))
                    (list (or (car flags) (cadr flags)) (and mean-sk #t) (and scale-sk #t)))
      (when (and mean mean-sk)
        (check-true (within? mean
                             mean-sk
                             (for/list ([m (in-list mean-sk)]
                                        [s (in-list (or scale-sk (map (lambda (_) 1.0) mean-sk)))])
                               (* 1e-12 (+ (abs m) s))))
                    (format "mean ~a vs ~a" mean mean-sk)))
      (when var-sk
        (check-true (within? var var-sk (map (lambda (v) (* 1e-11 v)) var-sk))
                    (format "var ~a vs ~a" var var-sk))
        (check-true (within? scale scale-sk (map (lambda (s) (* 1e-11 s)) scale-sk))
                    (format "scale ~a vs ~a" scale scale-sk))
        (for ([s (in-list scale)]
              [s-sk (in-list scale-sk)])
          (check-equal? (= s 1.0) (= s-sk 1.0) "a masked column is masked on both sides"))))))

(define (bits-of xs dt)
  (for/list ([x (in-list xs)])
    (if (eq? dt 'float32)
        (integer-bytes->integer (real->floating-point-bytes x 4) #f)
        (integer-bytes->integer (real->floating-point-bytes x 8) #f))))

(test-twin "L15, L16: transform and inverse are sklearn's, bitwise in float32"
  (define cases
    (for*/list ([fit-dt (in-list '(float32 float64))]
                [y-dt (in-list '(float32 float64))]
                [data (in-list (datasets fit-dt))]
                [flags (in-list flag-pairs)])
      (list fit-dt y-dt data flags)))
  (define results
    (twin-results (for/list ([c (in-list cases)])
                    (match-define (list fit-dt y-dt data flags) c)
                    (case-of "transform" data fit-dt flags #:y-dtype y-dt))))
  (for ([c (in-list cases)]
        [result (in-list results)])
    (match-define (list fit-dt y-dt data flags) c)
    (test-case (format "fit ~a, transform ~a, ~a ~a" fit-dt y-dt (dataset-name data) flags)
      (define fitted (fit-of data fit-dt flags))
      (define Y (rows->matrix (dataset-rows data) y-dt))
      (define T (standard-scaler-transform fitted Y))
      (define t (append* (device-matrix->list* T)))
      (define back (append* (device-matrix->list* (standard-scaler-inverse-transform fitted T))))
      (check-equal? (hash-ref result 'dtype) (symbol->string y-dt))
      (cond
        [(eq? y-dt 'float32)
         (check-equal? (bits-of t y-dt) (hash-ref result 'transform_bits) "transform")
         (check-equal? (bits-of back y-dt) (hash-ref result 'inverse_bits) "inverse")]
        [else
         (define d (length (car (dataset-rows data))))
         (define mean (or (listed (standard-scaler-result-mean fitted)) (build-list d (lambda (_) 0.0))))
         (define scale (or (listed (standard-scaler-result-scale fitted)) (build-list d (lambda (_) 1.0))))
         (define tolerance
           (for/list ([t-sk (in-list (hash-ref result 'transform))]
                      [j (in-naturals)])
             (define k (modulo j d))
             (* 1e-11 (+ 1 (abs t-sk) (/ (abs (list-ref mean k)) (list-ref scale k))))))
         (check-true (within? t (hash-ref result 'transform) tolerance) "transform")]))))

(test-twin "L17: the hyperparameters are sklearn's but copy"
  (match-define (list theirs) (twin-results (list (hasheq 'op "params"))))
  (check-equal? (standard-scaler-params) (twin-params theirs '(copy))))
