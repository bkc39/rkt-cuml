#lang racket/base

(require (only-in cuml/decomposition tsvd-fit tsvd-inverse-transform tsvd-transform)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in threading ~>)
         (only-in raft device-matrix->list*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/pca-data.rkt" column fixture-name fixture-rows norm)
         (only-in "private/scaler-data.rkt" rows->matrix unit-roundoff)
         (only-in "private/tsvd-data.rkt" law-fixtures popvars squared-norm svd tsvd-of))

(define (exact-product rows columns)
  (define exact-columns
    (for/list ([c (in-list columns)])
      (map inexact->exact c)))
  (for/list ([row (in-list rows)])
    (define x (map inexact->exact row))
    (for/list ([c (in-list exact-columns)])
      (for/sum ([a (in-list x)] [b (in-list c)]) (* a b)))))

(define (absolute rows)
  (for/list ([row (in-list rows)])
    (map abs row)))

(define (transpose rows)
  (apply map list rows))

(define (gamma m u)
  (/ (* m u) (- 1 (* m u))))

(define (product-problems got a columns u what)
  (define m (length (car columns)))
  (define want (exact-product a columns))
  (define scale (exact-product (absolute a) (absolute columns)))
  (define g (gamma m u))
  (for*/list ([(got-row want-row scale-row i) (in-parallel got want scale (in-naturals))]
              [(x y s j) (in-parallel got-row want-row scale-row (in-naturals))]
              #:when (> (abs (- (inexact->exact x) y)) (* g (inexact->exact s))))
    (format "~a[~a, ~a]: ~a vs ~a, bound ~a" what i j x (exact->inexact y) (* g s))))

(define (cases)
  (for*/list ([dt (in-list '(float32 float64))]
              [solver (in-list '(full jacobi))]
              [f (in-list law-fixtures)])
    (list dt solver f)))

(define (extents rows)
  (match-define (cons first-row _) rows)
  (values (length rows) (length first-row)))

(define (tall-case? c)
  (match-define (list _ _ f) c)
  (define-values (n d) (extents (fixture-rows f 'float64)))
  (<= d n))

(define (first-problems problems)
  (for/list ([p (in-list problems)]
             [_ (in-range 3)])
    p))

(test-gpu "L5, L6: transform and inverse are within gamma of the host products of the fit's own components"
  (for ([c (in-list (cases))])
    (match-define (list dt solver f) c)
    (test-case (format "~a ~a ~a" dt solver (fixture-name f))
      (define rows (fixture-rows f dt))
      (define X (rows->matrix rows dt))
      (define result (tsvd-fit X #:n-components 3 #:solver solver))
      (match-define (svd components _ _ _) (tsvd-of result))
      (define T (tsvd-transform result X))
      (define T-rows (device-matrix->list* T))
      (define I-rows (device-matrix->list* (tsvd-inverse-transform result T)))
      (define u (unit-roundoff dt))
      (check-equal? (first-problems (product-problems T-rows rows components u "T")) '() "L5")
      (check-equal? (first-problems (product-problems I-rows T-rows (transpose components) u "I"))
                    '()
                    "L6"))))

(test-gpu "L7: at k = d the round trip returns X within 8 d u (32 d u under Jacobi) of its largest row"
  (for ([c (in-list (filter tall-case? (cases)))])
    (match-define (list dt solver f) c)
    (define rows (fixture-rows f dt))
    (define-values (_n d) (extents rows))
    (test-case (format "~a ~a ~a" dt solver (fixture-name f))
      (define X (rows->matrix rows dt))
      (define result (tsvd-fit X #:n-components d #:solver solver))
      (define back (~> (tsvd-transform result X) (tsvd-inverse-transform result _) device-matrix->list*))
      (define worst
        (for*/fold ([worst 0.0])
                   ([(row back-row) (in-parallel rows back)]
                    [(x y) (in-parallel row back-row)])
          (max worst (abs (- x y)))))
      (define largest (apply max (map norm rows)))
      (define bound (* (if (eq? solver 'jacobi) 32 8) d (unit-roundoff dt) largest))
      (check-true (<= worst bound) (format "~a, bound ~a" worst bound)))))

(define (mean-square xs)
  (/ (for/sum ([x (in-list xs)]) (* x x)) (length xs)))

(test-gpu "L10, L11: the explained variance is the transform's population variance, over X's total"
  (for* ([c (in-list (cases))]
         [which (in-list '(3 full))])
    (match-define (list dt solver f) c)
    (define rows (fixture-rows f dt))
    (define-values (n d) (extents rows))
    (define k
      (if (eq? which 'full)
          (min n d)
          3))
    (test-case (format "~a ~a ~a k=~a" dt solver (fixture-name f) k)
      (define X (rows->matrix rows dt))
      (define result (tsvd-fit X #:n-components k #:solver solver))
      (match-define (svd _ variance ratio _) (tsvd-of result))
      (define T-rows (device-matrix->list* (tsvd-transform result X)))
      (define u (unit-roundoff dt))
      (define total (apply + (popvars rows)))
      (define ratio-bound (/ (* 16 u (squared-norm rows)) (* n total)))
      (for ([v (in-list variance)]
            [r (in-list ratio)]
            [want (in-list (popvars T-rows))]
            [j (in-naturals)])
        (define t (column T-rows j))
        (check-true (<= (abs (- v want)) (* 16 u (mean-square t)))
                    (format "L10: variance ~a: ~a vs ~a" j v want))
        (check-true (<= (abs (- r (/ v total))) ratio-bound)
                    (format "L10: ratio ~a: ~a vs ~a" j r (/ v total))))
      (define sum-bound (* 4 ratio-bound))
      (define sum (apply + ratio))
      (check-true (andmap (lambda (r) (>= r 0.0)) ratio) "L11: no negative ratio")
      (check-true (<= sum (+ 1 sum-bound)) (format "L11: ratios sum to ~a" sum))
      (check-true (or (< k d) (<= (abs (- sum 1)) sum-bound))
                  (format "L11: at k = d the ratios sum to ~a" sum)))))
