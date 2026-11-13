#lang racket/base

(require (only-in racket/list range)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-exn check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in cuml/linear-model logistic-regression-decision-function)
         (only-in "private/linear-data.rkt" dtypes on-matrix)
         (only-in "private/linear-twin-data.rkt" decoded logistic-twin logistic-twin-results)
         (only-in "private/logistic-check.rkt" residual residual-of)
         (only-in "private/logistic-data.rkt"
                  blob-case
                  case-fit
                  fit-parts
                  lr-case-name
                  lr-case-w
                  lr-case-X
                  lr-case-y
                  separable-case)
         (only-in "private/python-env.rkt" test-twin))

(define small
  (list (blob-case "binary" 64 4 2 41)
        (blob-case "binary weighted" 48 2 2 42 #:weighted? #t)
        (blob-case "three classes" 60 3 3 43)))

(define settings '((l2 1.0 #f) (l2 0.1 #f) (l1 1.0 #f) (elasticnet 0.5 0.5) (none 1.0 #f)))

(define cutoff-message #rx"first-order residual .* above the bound")

(define (fit-or-refusal c dt penalty C ratio max-iter tol)
  (with-handlers ([(lambda (e) (and (exn:fail? e) (regexp-match? cutoff-message (exn-message e))))
                   (lambda (_) #f)])
    (case-fit c dt #:penalty penalty #:C C #:l1-ratio ratio #:max-iter max-iter #:tol tol)))

(define (residual-at c parts penalty C ratio tol)
  (match-define (list coef intercept _) parts)
  (residual-of (lr-case-X c)
               (lr-case-y c)
               (lr-case-w c)
               coef
               intercept
               #:penalty penalty
               #:C C
               #:l1-ratio ratio
               #:tol tol))

(test-gpu "every accepted small fit has a 128-bit first-order residual within the bound (H1)"
  (define-values (accepted refused)
    (for*/fold ([accepted 0]
                [refused 0])
               ([c (in-list small)]
                [dt (in-list dtypes)]
                [s (in-list settings)]
                [max-iter (in-list '(1 3 10 1000))]
                [tol (in-list '(1e-4 1e-6))])
      (match-define (list penalty C ratio) s)
      (define r (fit-or-refusal c dt penalty C ratio max-iter tol))
      (cond
        [r
         (match-define (residual K F bound) (residual-at c (fit-parts r) penalty C ratio tol))
         (test-case (format "~a ~a ~a ~a ~a ~a" (lr-case-name c) dt penalty C max-iter tol)
           (check-true (<= K bound) (format "K ~a bound ~a F ~a" K bound F)))
         (values (add1 accepted) refused)]
        [else (values accepted (add1 refused))])))
  (check-true (and (> accepted 0) (> refused 0)) (format "accepted ~a refused ~a" accepted refused)))

(test-gpu "at tol 1e-10 a float64 l2 fit is stationary to 1e-8 of the objective (H2)"
  (for* ([c (in-list small)]
         [C (in-list '(1.0 0.1))])
    (test-case (format "~a ~a" (lr-case-name c) C)
      (define r (case-fit c 'float64 #:C C #:tol 1e-10 #:max-iter 100000))
      (match-define (residual K F _) (residual-at c (fit-parts r) 'l2 C #f 1e-10))
      (check-true (<= K (* 1e-8 (max F 1))) (format "K ~a F ~a" K F)))))

(define hard-cutoff
  (blob-case "hard cutoff"
             2000
             10
             2
             44
             #:scales (map (lambda (j) (expt 2 (- (modulo j 7) 3))) (range 10))))

(test-twin "one iteration on mixed scales is refused, and Python cuML's point there is above the bound (H3)"
  (define requests
    (for/list ([dt (in-list dtypes)])
      (logistic-twin "cuml-fit" hard-cutoff dt (hasheq 'max_iter 1))))
  (for ([dt (in-list dtypes)]
        [theirs (in-list (logistic-twin-results requests))])
    (test-case (symbol->string dt)
      (check-exn cutoff-message (lambda () (case-fit hard-cutoff dt #:max-iter 1)))
      (match-define (residual K _ bound)
        (residual-at hard-cutoff
                     (list (list (decoded dt (hash-ref theirs 'coef_bits)))
                           (decoded dt (hash-ref theirs 'intercept_bits))
                           (hash-ref theirs 'n_iter))
                     'l2
                     1.0
                     #f
                     1e-4))
      (check-true (> K bound) (format "K ~a bound ~a" K bound)))))

(test-gpu "separable data without a penalty is accepted, fits every row, and is within the bound (H4)"
  (define c (separable-case "separable" 200 3 45))
  (for ([dt (in-list dtypes)])
    (test-case (symbol->string dt)
      (define r (case-fit c dt #:penalty 'none))
      (match-define (residual K _ bound) (residual-at c (fit-parts r) 'none 1.0 #f 1e-4))
      (check-true (<= K bound) (format "K ~a bound ~a" K bound))
      (define z
        (device-matrix->list* (logistic-regression-decision-function r (on-matrix (lr-case-X c) dt))))
      (check-equal? (map (lambda (zi) (if (> (car zi) 0) 1 0)) z) (lr-case-y c)))))
