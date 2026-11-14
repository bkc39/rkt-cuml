#lang racket/base

(require (only-in cuml/linear-model logistic-regression-predict-proba)
         (only-in rackunit check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in "glmnet-data.rkt" glmnet-binomial glmnet-multinomial narrowed-lr-case)
         (only-in "linear-data.rkt" by-dtype on-matrix)
         (only-in "logistic-data.rkt"
                  default-tol-cap
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
         (only-in "real-logistic-data.rkt" labelled-case labelled-set))

(provide (struct-out trial)
         check-binomial
         check-multinomial
         labelled-lr-case)

(struct trial
        (case Cs
          float32-bound
          float64-bound
          default-capped?))

(define (penalties C)
  (list (setting 'l2 C #f) (setting 'l1 C #f) (setting 'elasticnet C 0.5)))

(define (settings Cs)
  (apply append (map penalties Cs)))

(define (bound-of t dt)
  (by-dtype dt (trial-float32-bound t) (trial-float64-bound t)))

(define (tight-fit c dt s fi)
  (setting-fit c dt s #:fit-intercept fi #:tol (tight-tol dt) #:max-iter tight-max-iter))

(define (check-default t c dt s fi gap-to)
  (when (trial-default-capped? t)
    (define gap (gap-to (fit-parts (setting-fit c dt s #:fit-intercept fi))))
    (check-true (<= gap (default-tol-cap dt)) (format "default-tol gap ~a" gap))))

(define (check-binomial t dt)
  (define c (narrowed-lr-case (trial-case t) dt))
  (for* ([s (in-list (settings (trial-Cs t)))]
         [fi (in-list '(#t #f))])
    (test-case (format "~a ~a ~a C ~a intercept ~a"
                       dt
                       (lr-case-name c)
                       (setting-penalty s)
                       (setting-C s)
                       fi)
      (define theirs (fit-flat (glmnet-binomial c s #:fit-intercept fi)))
      (define (gap-to parts)
        (relative-gap (fit-flat parts) theirs))
      (define gap (gap-to (fit-parts (tight-fit c dt s fi))))
      (check-true (<= gap (bound-of t dt)) (format "gap ~a" gap))
      (check-default t c dt s fi gap-to))))

(define (largest-difference rows other)
  (for*/fold ([m 0])
             ([(r o) (in-parallel rows other)]
              [(p q) (in-parallel r o)])
    (max m (abs (- p q)))))

(define (check-multinomial t dt)
  (define c (narrowed-lr-case (trial-case t) dt))
  (for* ([s (in-list (settings (trial-Cs t)))]
         [fi (in-list '(#t #f))])
    (test-case (format "~a ~a ~a C ~a intercept ~a"
                       dt
                       (lr-case-name c)
                       (setting-penalty s)
                       (setting-C s)
                       fi)
      (define-values (theirs probabilities) (glmnet-multinomial c s #:fit-intercept fi))
      (define (gap-to parts)
        (relative-gap (gauged-fit parts) (gauged-fit theirs)))
      (define tight (tight-fit c dt s fi))
      (define gap (gap-to (fit-parts tight)))
      (check-true (<= gap (bound-of t dt)) (format "centred gap ~a" gap))
      (check-default t c dt s fi gap-to)
      (define ours
        (device-matrix->list* (logistic-regression-predict-proba tight (on-matrix (lr-case-X c) dt))))
      (define spread (largest-difference ours probabilities))
      (check-true (<= spread (bound-of t dt)) (format "probabilities ~a" spread)))))

(define (labelled-lr-case name response excluded #:standardise? [standardise? #f])
  (labelled-case (labelled-set name response excluded #:standardise? standardise?)))
