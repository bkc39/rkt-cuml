#lang racket/base

(require (only-in cuml/linear-model logistic-regression-decision-function)
         (only-in racket/list range)
         (only-in racket/match match-define struct*)
         (only-in rackunit check-equal? check-exn check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes on-matrix)
         (only-in "private/logistic-data.rkt"
                  blob-case
                  cutoff-message
                  fit-parts
                  lr-case-name
                  lr-case-w
                  lr-case-X
                  lr-case-y
                  lr-fit
                  separable-case
                  setting
                  setting-C
                  setting-fit
                  setting-l1-ratio
                  setting-penalty
                  tight-max-iter
                  tight-tol)
         (only-in "private/logistic-oracle.rkt" residual residual-of)
         (only-in "private/logistic-twin-data.rkt" logistic-twin logistic-twin-results their-fit)
         (only-in "private/python-env.rkt" test-twin))

(define small
  (list (blob-case "binary" 64 4 2 41)
        (blob-case "binary weighted" 48 2 2 42 #:weighted? #t)
        (blob-case "three classes" 60 3 3 43)))

(define settings
  (list (setting 'l2 1.0 #f)
        (setting 'l2 0.1 #f)
        (setting 'l1 1.0 #f)
        (setting 'elasticnet 0.5 0.5)
        (setting 'none 1.0 #f)))

(struct trial (case dt setting max-iter tol))

(define (trial-name t)
  (match-define (trial c dt s max-iter tol) t)
  (format "~a ~a ~a ~a ~a ~a" (lr-case-name c) dt (setting-penalty s) (setting-C s) max-iter tol))

(define (fit-or-refusal t)
  (match-define (trial c dt s max-iter tol) t)
  (with-handlers ([(lambda (e) (and (exn:fail? e) (regexp-match? cutoff-message (exn-message e))))
                   (lambda (_) #f)])
    (setting-fit c dt s #:max-iter max-iter #:tol tol)))

(define (residual-at c parts s tol)
  (match-define (struct* lr-fit ([coef coef] [intercept intercept])) parts)
  (residual-of (lr-case-X c)
               (lr-case-y c)
               (lr-case-w c)
               coef
               intercept
               #:penalty (setting-penalty s)
               #:C (setting-C s)
               #:l1-ratio (setting-l1-ratio s)
               #:tol tol))

(test-gpu "every accepted small fit has a first-order residual within the bound"
  (define outcomes
    (for*/list ([c (in-list small)]
                [dt (in-list dtypes)]
                [s (in-list settings)]
                [max-iter (in-list '(1 3 10 1000))]
                [tol (in-list '(1e-4 1e-6))])
      (define t (trial c dt s max-iter tol))
      (cons t (fit-or-refusal t))))
  (define accepted (filter cdr outcomes))
  (for ([outcome (in-list accepted)])
    (match-define (cons t r) outcome)
    (test-case (trial-name t)
      (match-define (residual K F bound)
        (residual-at (trial-case t) (fit-parts r) (trial-setting t) (trial-tol t)))
      (check-true (<= K bound) (format "K ~a bound ~a F ~a" K bound F))))
  (check-true (< 0 (length accepted) (length outcomes))
              (format "accepted ~a of ~a" (length accepted) (length outcomes))))

(define stationary-bound 1e-8)

(test-gpu "a tight float64 l2 fit is stationary relative to the objective"
  (for* ([c (in-list small)]
         [C (in-list '(1.0 0.1))])
    (test-case (format "~a ~a" (lr-case-name c) C)
      (define s (setting 'l2 C #f))
      (define tol (tight-tol 'float64))
      (define r (setting-fit c 'float64 s #:tol tol #:max-iter tight-max-iter))
      (match-define (residual K F _) (residual-at c (fit-parts r) s tol))
      (check-true (<= K (* stationary-bound (max F 1))) (format "K ~a F ~a" K F)))))

(define hard-cutoff
  (blob-case "hard cutoff"
             2000
             10
             2
             44
             #:scales (map (lambda (j) (expt 2 (- (modulo j 7) 3))) (range 10))))

(define default-setting (setting 'l2 1.0 #f))

(test-twin "one iteration on mixed scales is refused, and Python cuML's point there is above the bound"
  (define requests
    (for/list ([dt (in-list dtypes)])
      (logistic-twin "cuml-fit" hard-cutoff dt #:max-iter 1)))
  (for ([dt (in-list dtypes)]
        [theirs (in-list (logistic-twin-results requests))])
    (test-case (symbol->string dt)
      (check-exn cutoff-message (lambda () (setting-fit hard-cutoff dt default-setting #:max-iter 1)))
      (match-define (residual K _ bound)
        (residual-at hard-cutoff (their-fit theirs dt) default-setting 1e-4))
      (check-true (> K bound) (format "K ~a bound ~a" K bound)))))

(test-gpu "separable data without a penalty is accepted, fits every row, and is within the bound"
  (define c (separable-case "separable" 200 3 45))
  (define s (setting 'none 1.0 #f))
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (define r (setting-fit c dt s))
      (match-define (residual K _ bound) (residual-at c (fit-parts r) s 1e-4))
      (check-true (<= K bound) (format "K ~a bound ~a" K bound))
      (define z
        (device-matrix->list* (logistic-regression-decision-function r (on-matrix (lr-case-X c) dt))))
      (check-equal? (map (lambda (zi) (if (> (car zi) 0) 1 0)) z) (lr-case-y c)))))
