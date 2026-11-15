#lang racket/base

(require (only-in cuml/linear-model logistic-regression-predict-proba)
         (only-in cuml/tests/private/linear-data gap max-abs on-matrix)
         (only-in cuml/tests/private/logistic-data
                  fit-flat
                  fit-parts
                  gauged-fit
                  lr-case-name
                  lr-case-X
                  relative-gap
                  setting
                  setting-C
                  setting-fit
                  setting-penalty
                  tight-max-iter
                  tight-tol)
         (only-in racket/list append-map)
         (only-in racket/match match)
         (only-in rackunit check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in "glmnet-data.rkt"
                  bound-of
                  default-bound
                  glmnet-binomial
                  glmnet-multinomial
                  narrowed-lr-case
                  trial-case
                  trial-Cs))

(provide check-binomial
         check-multinomial)

(define (penalties C)
  (list (setting 'l2 C #f) (setting 'l1 C #f) (setting 'elasticnet C 0.5)))

(define (tight-fit c dt s fi)
  (setting-fit c dt s #:fit-intercept fi #:tol (tight-tol dt) #:max-iter tight-max-iter))

(define (probabilities-of r c dt)
  (device-matrix->list* (logistic-regression-predict-proba r (on-matrix (lr-case-X c) dt))))

(define (uncentred parts)
  (gauged-fit parts #:centre-coef? #f))

(define (check-settings t dt check)
  (define c (narrowed-lr-case (trial-case t) dt))
  (for* ([s (in-list (append-map penalties (trial-Cs t)))]
         [fi (in-list '(#t #f))])
    (test-case (format "~a ~a ~a C ~a intercept ~a"
                       dt
                       (lr-case-name c)
                       (setting-penalty s)
                       (setting-C s)
                       fi)
      (check c s fi))))

(define (check-gaps t dt c s fi tight flat reference)
  (define (gap-to parts)
    (relative-gap (flat parts) reference))
  (define gap (gap-to (fit-parts tight)))
  (check-true (<= gap (bound-of t dt)) (format "gap ~a" gap))
  (define cap (default-bound t dt))
  (when cap
    (define default-gap (gap-to (fit-parts (setting-fit c dt s #:fit-intercept fi))))
    (check-true (<= default-gap cap) (format "default-tol gap ~a" default-gap))))

(define (check-binomial t dt)
  (check-settings t
                  dt
                  (lambda (c s fi)
                    (define reference
                      (match dt
                        ['float64 (fit-flat (glmnet-binomial c s #:fit-intercept fi))]
                        ['float32 (fit-flat (fit-parts (tight-fit c 'float64 s fi)))]))
                    (check-gaps t dt c s fi (tight-fit c dt s fi) fit-flat reference))))

(define (check-multinomial t dt)
  (check-settings t
                  dt
                  (lambda (c s fi)
                    (define-values (reference probabilities)
                      (match dt
                        ['float64
                         (define-values (theirs theirs-probabilities)
                           (glmnet-multinomial c s #:fit-intercept fi))
                         (values (uncentred theirs) theirs-probabilities)]
                        ['float32
                         (define r (tight-fit c 'float64 s fi))
                         (values (uncentred (fit-parts r)) (probabilities-of r c 'float64))]))
                    (define tight (tight-fit c dt s fi))
                    (check-gaps t dt c s fi tight uncentred reference)
                    (define spread (max-abs (map gap (probabilities-of tight c dt) probabilities)))
                    (check-true (<= spread (bound-of t dt)) (format "probabilities ~a" spread)))))
