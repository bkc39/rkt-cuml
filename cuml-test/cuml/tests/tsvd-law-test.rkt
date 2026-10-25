#lang racket/base

(require (only-in cuml/decomposition tsvd-fit)
         (only-in racket/list drop)
         (only-in racket/match match match-define)
         (only-in rackunit check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/pca-data.rkt" first-max-entries fixture-name fixture-rows host-gram)
         (only-in "private/scaler-data.rkt" rows->matrix unit-roundoff)
         (only-in "private/tsvd-data.rkt"
                  duplicate-fixture
                  law-fixtures
                  squared-norm
                  svd
                  tie-fixture
                  tsvd-of
                  twos-fixture
                  wide-fixture
                  zeros-fixture))

(define dtypes '(float32 float64))
(define solvers '(full jacobi))

(define (extents rows)
  (match-define (cons first-row _) rows)
  (values (length rows) (length first-row)))

(define (non-increasing? xs)
  (match xs
    [(list* a (and more (list* b _))) (and (>= a b) (non-increasing? more))]
    [_ #t]))

(define (off-identity components)
  (for*/fold ([worst 0.0])
             ([(row i) (in-indexed (host-gram components))]
              [(g j) (in-indexed row)])
    (max worst (abs (- g (if (= i j) 1.0 0.0))))))

(define (orthonormal-bound solver d u)
  (* (if (eq? solver 'jacobi) 16 4) d u))

(define (all-finite? rows)
  (for*/and ([row (in-list rows)]
             [x (in-list row)])
    (rational? x)))

(define (spectrum-ok? singular)
  (and (non-increasing? singular) (andmap (lambda (s) (and (rational? s) (>= s 0.0))) singular)))

(define (fit-of f dt solver k)
  (define rows (fixture-rows f dt))
  (values rows (tsvd-of (tsvd-fit (rows->matrix rows dt) #:n-components k #:solver solver))))

(define (law-cases)
  (for*/list ([dt (in-list dtypes)]
              [solver (in-list solvers)]
              [f (in-list (append law-fixtures (list zeros-fixture twos-fixture tie-fixture)))]
              [k (in-list '(1 full))])
    (list dt solver f k)))

(test-gpu "L2, L3, L4: singular values, orthonormality and signs on every fixture"
  (for ([c (in-list (law-cases))])
    (match-define (list dt solver f which) c)
    (define-values (n d) (extents (fixture-rows f 'float64)))
    (define k
      (if (eq? which 'full)
          (min n d)
          1))
    (test-case (format "~a ~a ~a k=~a" dt solver (fixture-name f) k)
      (define-values (_ fitted) (fit-of f dt solver k))
      (match-define (svd components variance ratio singular) fitted)
      (check-true (spectrum-ok? singular) (format "L2: ~a" singular))
      (check-true (andmap rational? (append variance ratio)) "L2: finite variances")
      (check-true (all-finite? components) "L2: finite components")
      (define u (unit-roundoff dt))
      (define off (off-identity components))
      (check-true (<= off (orthonormal-bound solver d u))
                  (format "L3: |CC^T - I| = ~a, ~a d u" off (/ off (* d u))))
      (check-true (andmap positive? (first-max-entries components)) "L4"))))

(test-gpu "L15: rank-deficient fits have no NaN, and their trailing singular values are small"
  (for* ([dt (in-list dtypes)]
         [solver (in-list solvers)]
         [c (in-list (list (list duplicate-fixture 6 4) (list wide-fixture 4 4)))])
    (match-define (list f k rank) c)
    (test-case (format "~a ~a ~a" dt solver (fixture-name f))
      (define-values (rows fitted) (fit-of f dt solver k))
      (match-define (svd components variance ratio singular) fitted)
      (check-true (all-finite? (list* variance ratio singular components)))
      (define most (sqrt (* 64 (unit-roundoff dt) (squared-norm rows))))
      (for ([s (in-list (drop singular rank))])
        (check-true (<= s most) (format "trailing ~a, bound ~a" s most)))
      (check-true (<= (off-identity components) (orthonormal-bound solver 6 (unit-roundoff dt)))))))

(define (positive-zeros? xs)
  (andmap (lambda (x) (eqv? x 0.0)) xs))

(test-gpu "L16: vanishing variance gives positive zeros, never NaN"
  (for* ([dt (in-list dtypes)]
         [solver (in-list solvers)])
    (test-case (format "~a ~a" dt solver)
      (define-values (_ zeros) (fit-of zeros-fixture dt solver 2))
      (match-define (svd components variance ratio singular) zeros)
      (check-true (positive-zeros? singular) (format "singular values ~a" singular))
      (check-true (positive-zeros? variance) (format "variances ~a" variance))
      (check-true (positive-zeros? ratio) (format "ratios ~a" ratio))
      (check-true (all-finite? components))
      (check-true (<= (off-identity components) (orthonormal-bound solver 4 (unit-roundoff dt))))
      (define-values (_rows twos) (fit-of twos-fixture dt solver 2))
      (match-define (svd _ twos-variance twos-ratio _) twos)
      (check-true (positive-zeros? twos-variance) (format "twos ~a" twos-variance))
      (check-true (positive-zeros? twos-ratio) (format "twos ~a" twos-ratio)))))
