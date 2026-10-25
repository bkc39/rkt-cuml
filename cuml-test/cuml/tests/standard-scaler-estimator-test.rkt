#lang racket/base

(require (only-in cuml
                  estimator-params
                  estimator-set-params
                  fit
                  fit-transform
                  fitted?
                  inverse-transform
                  kmeans
                  standard-scaler
                  standard-scaler-model
                  standard-scaler?
                  transform)
         (only-in cuml/preprocessing
                  standard-scaler-fit
                  standard-scaler-fit-transform
                  standard-scaler-inverse-transform
                  standard-scaler-params
                  standard-scaler-result-mean
                  standard-scaler-result-scale
                  standard-scaler-result-var
                  standard-scaler-result?
                  standard-scaler-transform)
         (only-in racket/generic exn:fail:support?)
         (only-in racket/list make-list)
         (only-in rackunit check-equal? check-exn check-false check-not-equal? check-true test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/capabilities.rkt" capabilities)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/estimator-data.rkt" printed round-trips)
         (only-in "private/kmeans-data.rkt" blank-matrix)
         (only-in "private/scaler-data.rkt" bits-equal? normal-rows rows->matrix))

(define changed (hasheq 'with-mean #f 'with-std #f))

(define (statistics result)
  (for/list ([v (list (standard-scaler-result-mean result)
                      (standard-scaler-result-var result)
                      (standard-scaler-result-scale result))])
    (and v (device-vector->list v))))

(test-case "L17: the default parameters, and every keyword round-trips"
  (check-equal? (standard-scaler-params) (hasheq 'with-mean #t 'with-std #t))
  (check-equal? (estimator-params (standard-scaler)) (standard-scaler-params))
  (check-equal? (round-trips standard-scaler changed) changed)
  (check-equal? (estimator-params (standard-scaler (standard-scaler-params #:with-mean #f)
                                                   #:with-std #f))
                (standard-scaler-params #:with-mean #f #:with-std #f)))

(test-case "L17: printing, equality and set-params"
  (check-equal? (printed (standard-scaler)) (make-list 3 "#<standard-scaler>"))
  (check-equal? (format "~v" (standard-scaler #:with-mean #f)) "#<standard-scaler #:with-mean #f>")
  (check-equal? (format "~v" (standard-scaler #:with-std #f #:with-mean #f))
                "#<standard-scaler #:with-mean #f #:with-std #f>")
  (check-equal? (format "~v" (standard-scaler #:resources 'some-resources))
                "#<standard-scaler #:resources 'some-resources>")
  (check-equal? (standard-scaler #:with-mean #f) (standard-scaler #:with-mean #f))
  (check-not-equal? (standard-scaler #:with-mean #f) (standard-scaler))
  (check-equal? (equal-hash-code (standard-scaler #:with-std #f))
                (equal-hash-code (standard-scaler #:with-std #f)))
  (check-equal? (estimator-set-params (standard-scaler) #:with-mean #f)
                (standard-scaler #:with-mean #f))
  (check-cuml-error 'logic
                    "standard-scaler-params: standard-scaler has no parameter copy"
                    (lambda () (standard-scaler-params (hasheq 'copy #t))))
  (check-cuml-error 'logic
                    "estimator-set-params: standard-scaler has no parameter copy"
                    (lambda () (estimator-set-params (standard-scaler) (hasheq 'copy #t))))
  (check-false (standard-scaler? 'standard-scaler))
  (check-false (standard-scaler-model (standard-scaler))))

(test-case "L18: a standard scaler is an estimator and a transformer, not a clusterer or a predictor"
  (check-equal? (capabilities (standard-scaler) standard-scaler?) '(#t #f #f #t #f #t)))

(test-gpu "L18: inverse-transform of a transformer without one is not supported"
  (define X (rows->matrix (normal-rows 2 40 2 'float32) 'float32))
  (check-exn exn:fail:support? (lambda () (inverse-transform (fit (kmeans #:n-clusters 2) X) X))))

(test-gpu "L11, L18: the estimator is the procedures, bitwise, and fit leaves its argument unfitted"
  (for* ([dt (in-list '(float32 float64))]
         [with-mean (in-list '(#t #f))]
         [with-std (in-list '(#t #f))])
    (test-case (format "~a ~a ~a" dt with-mean with-std)
      (define X (rows->matrix (normal-rows 11 300 7 dt) dt))
      (define e (standard-scaler #:with-mean with-mean #:with-std with-std))
      (define fitted-e (fit e X '(ignored)))
      (check-equal? (list (fitted? e) (fitted? fitted-e)) '(#f #t))
      (check-true (standard-scaler-result? (standard-scaler-model fitted-e)))
      (check-equal? (format "~v" fitted-e)
                    (format "~v" (fit (standard-scaler #:with-mean with-mean #:with-std with-std) X)))
      (check-false (fitted? (estimator-set-params fitted-e (hasheq))))
      (define result (standard-scaler-fit X #:with-mean with-mean #:with-std with-std))
      (check-true (bits-equal? (statistics (standard-scaler-model fitted-e)) (statistics result)))
      (define T (device-matrix->list* (standard-scaler-transform result X)))
      (check-true (bits-equal? (device-matrix->list* (transform fitted-e X)) T))
      (check-true (bits-equal? (device-matrix->list* (fit-transform e X)) T))
      (define-values (again T2)
        (standard-scaler-fit-transform X #:with-mean with-mean #:with-std with-std))
      (check-true (bits-equal? (statistics again) (statistics result)))
      (check-true (bits-equal? (device-matrix->list* T2) T))
      (define Tm (transform fitted-e X))
      (check-true (bits-equal? (device-matrix->list* (inverse-transform fitted-e Tm))
                               (device-matrix->list* (standard-scaler-inverse-transform result
                                                                                        Tm)))))))

(test-gpu "L11: fit-transform names the generic for a fit refusal"
  (check-cuml-error 'logic
                    "fit-transform: X: expected at least 1 row, got 0"
                    (lambda () (fit-transform (standard-scaler) (blank-matrix 0 3 'float32)))))
