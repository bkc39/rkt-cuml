#lang racket/base

(require (only-in cuml/linear-model logistic-regression-fit logistic-regression-result-coef)
         (only-in racket/list make-list)
         (only-in racket/match match-lambda)
         (only-in racket/math pi)
         (only-in rackunit check-false check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" normal-source uniform-source)
         (only-in "private/linear-data.rkt" dtypes on-matrix on-vector)
         (only-in "private/logistic-data.rkt"
                  cutoff-message
                  default-tol-cap
                  fit-flat
                  fit-parts
                  lr-case-X
                  relative-gap)
         (only-in "private/logistic-newton.rkt" newton-coef)
         (only-in "private/real-logistic-data.rkt" labelled-case labelled-labels labelled-set))

(define (offset-data n seed)
  (define normal (normal-source seed))
  (define X
    (for/list ([_ (in-range n)])
      (define a (normal))
      (list a (normal))))
  (values X
          (map (match-lambda
                 [(list a b) (if (> (+ (* 2 a) (- b) (* 0.5 (normal))) 0) 1 0)])
               X)))

(define (fit-or-message X
                        y
                        dt
                        #:fit-intercept fi
                        #:penalty [penalty 'l2]
                        #:C [C 1.0]
                        #:tol [tol 1e-4])
  (with-handlers ([exn:fail? exn-message])
    (logistic-regression-fit (on-matrix X dt)
                             (on-vector y 'int32)
                             #:fit-intercept fi
                             #:penalty penalty
                             #:C C
                             #:tol tol)))

(define (coef-of r)
  (car (device-matrix->list* (logistic-regression-result-coef r))))

(define (shifted X offset)
  (for/list ([r (in-list X)])
    (map (lambda (v) (+ v offset)) r)))

(define (check-near-oracle r X y dt #:fit-intercept fi #:lam lam)
  (define gap (relative-gap (coef-of r) (newton-coef X y #:lam lam #:fit-intercept fi)))
  (check-true (<= gap (default-tol-cap dt)) (format "gap ~a" gap)))

(define offset-rows 400)

(define (offset-fit offset dt penalty #:tol [tol 1e-4])
  (define-values (X0 y) (offset-data offset-rows 9))
  (define X (shifted X0 offset))
  (values (fit-or-message X y dt #:fit-intercept #f #:penalty penalty #:tol tol)
          (lambda (r)
            (check-near-oracle r
                               X
                               y
                               dt
                               #:fit-intercept #f
                               #:lam (if (eq? penalty 'l2)
                                         (/ 1.0 offset-rows)
                                         0.0)))))

(test-gpu "without an intercept, columns on a common offset fit near the oracle or are refused"
  (for* ([offset (in-list '(0.0 100.0 1000.0 10000.0))]
         [dt (in-list dtypes)]
         [penalty (in-list '(l2 none))])
    (test-case (format "~a ~a ~a" offset dt penalty)
      (define-values (r check-near) (offset-fit offset dt penalty))
      (cond
        [(string? r)
         (check-true (< 100.0 offset) r)
         (check-true (regexp-match? cutoff-message r) r)]
        [else (check-near r)]))))

(test-gpu "without an intercept, float64 columns on a common offset fit near the oracle at tol 1e-6"
  (for* ([offset (in-list '(1000.0 10000.0))]
         [penalty (in-list '(l2 none))])
    (test-case (format "~a ~a" offset penalty)
      (define-values (r check-near) (offset-fit offset 'float64 penalty #:tol 1e-6))
      (check-false (string? r) (and (string? r) r))
      (unless (string? r)
        (check-near r)))))

(test-gpu "iris on an offset of 1e4 without an intercept is never fitted as all zeros"
  (define iris (labelled-set 'iris "species" '()))
  (define X (shifted (map (lambda (r) (map exact->inexact r)) (lr-case-X (labelled-case iris))) 1e4))
  (for ([dt (in-list dtypes)])
    (test-case (format "~a" dt)
      (define r
        (with-handlers ([exn:fail? exn-message])
          (logistic-regression-fit (on-matrix X dt) (labelled-labels iris) #:fit-intercept #f)))
      (unless (string? r)
        (check-false
         (andmap zero? (apply append (device-matrix->list* (logistic-regression-result-coef r)))))))))

(define (scale-data m)
  (define uniform (uniform-source 11))
  (define (normal)
    (define u (- 1.0 (uniform)))
    (* (sqrt (* -2.0 (log u))) (cos (* 2.0 pi (uniform)))))
  (define X
    (for/list ([_ (in-range m)])
      (define a (normal))
      (list a (normal))))
  (values X
          (map (match-lambda
                 [(list a b) (if (> (+ a (* 0.5 b) (- (uniform) 0.5)) 0) 1 0)])
               X)))

(define (check-scaled-fit m dt e C #:refusal [refusal #f])
  (define-values (X0 y) (scale-data m))
  (define X
    (for/list ([r (in-list X0)])
      (map (lambda (v) (* v (expt 2.0 (- e)))) r)))
  (define r (fit-or-message X y dt #:fit-intercept #t #:C C))
  (cond
    [(string? r) (check-true (and refusal (regexp-match? refusal r)) r)]
    [else (check-near-oracle r X y dt #:fit-intercept #t #:lam (/ 1.0 (* C m)))]))

(define too-strong #rx"^logistic-regression-fit: C: the penalty is too strong for the float64 solver")

(test-gpu "a small X over many rows is not refused for its penalty: the bound is on l2 per row"
  (check-scaled-fit 20000 'float64 16 1.0))

(test-gpu "a penalty past the bound at few rows is refused as too strong, not as a cutoff"
  (check-scaled-fit 200 'float64 18 100.0 #:refusal too-strong))

(test-gpu "'balanced with uniform weights counts only the rows of positive weight, two classes"
  (define-values (X0 y0) (offset-data 400 5))
  (define (balanced X y w)
    (fit-flat (fit-parts (logistic-regression-fit (on-matrix X 'float64)
                                                  (on-vector y 'int32)
                                                  #:sample-weight (on-vector w 'float64)
                                                  #:class-weight 'balanced))))
  (define zero-rows '((0.5 0.5) (1.0 -1.0) (2.0 0.0) (-1.0 1.0) (0.0 3.0)))
  (define with-zeros
    (balanced (append X0 zero-rows)
              (append y0 '(0 0 0 0 1))
              (append (make-list (length y0) 0.1) (make-list (length zero-rows) 0.0))))
  (define without (balanced X0 y0 (make-list (length y0) 0.1)))
  (define gap (relative-gap with-zeros without))
  (check-true (<= gap 1e-6) (format "gap ~a" gap)))
