#lang racket/base

(require (only-in cuml/decomposition tsvd-fit tsvd-fit-transform tsvd-transform)
         (only-in racket/list drop take)
         (only-in racket/match match-define)
         (only-in racket/math sqr)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in threading ~>)
         (only-in "private/pca-data.rkt"
                  first-max-entries
                  fixture-name
                  fixture-rows
                  gaps
                  host-gram
                  magnitude-rows
                  non-increasing?
                  over-bound)
         (only-in "private/scaler-data.rkt" bits-equal? rows->matrix scaled-rows unit-roundoff)
         (only-in "private/tsvd-data.rkt"
                  duplicate-fixture
                  extents
                  law-fixtures
                  popvars
                  squared-norm
                  svd
                  tie-fixture
                  tsvd-of
                  twos-fixture
                  wide-fixture
                  worst-difference
                  zeros-fixture))

(define dtypes '(float32 float64))
(define solvers '(full jacobi))

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

(define (least-magnitude dt)
  (if (eq? dt 'float32)
      (expt 2 -149)
      (expt 2 -1074)))

(define (unscaled x by power)
  (exact->inexact (/ (inexact->exact x) (expt (inexact->exact by) power))))

(define (tiny-problems dt solver by)
  (define unit-X (rows->matrix magnitude-rows dt))
  (define small-X (rows->matrix (scaled-rows magnitude-rows by) dt))
  (match-define (svd c v r s) (tsvd-of (tsvd-fit unit-X #:n-components 2 #:solver solver)))
  (match-define (svd _ _ _ all-singular) (tsvd-of (tsvd-fit unit-X #:n-components 3 #:solver solver)))
  (define-values (small T*) (tsvd-fit-transform small-X #:n-components 2 #:solver solver))
  (match-define (svd c* v* r* s*) (tsvd-of small))
  (define-values (n _) (extents magnitude-rows))
  (define bound
    (* 2 (if (eq? solver 'jacobi) 256 64) (unit-roundoff dt) (squared-norm magnitude-rows)))
  (define total (apply + (popvars magnitude-rows)))
  (define (floor-at power)
    (unscaled (least-magnitude dt) by power))
  (filter
   values
   (append
    (for/list ([x (in-list c*)]
               [y (in-list c)]
               [g (in-list (~> (map sqr all-singular) gaps (take 2)))]
               [i (in-naturals)])
      (over-bound (format "component ~a" i) (worst-difference x y) 0.0 (/ bound g)))
    (for/list ([x (in-list s*)]
               [y (in-list s)]
               [i (in-naturals)])
      (over-bound (format "singular value ~a" i) (unscaled x by 1) y (+ (/ bound y) (floor-at 1))))
    (for/list ([x (in-list v*)]
               [y (in-list v)]
               [i (in-naturals)])
      (over-bound (format "variance ~a" i) (unscaled x by 2) y (+ (/ bound n) (floor-at 2))))
    (for/list ([x (in-list r*)]
               [y (in-list r)]
               [i (in-naturals)])
      (or (and (not (positive? x)) (format "ratio ~a is ~a" i x))
          (over-bound (format "ratio ~a" i) x y (/ bound (* n total)))))
    (list (and (not (bits-equal? (device-matrix->list* T*)
                                 (device-matrix->list* (tsvd-transform small small-X))))
               "fit-transform's matrix is not tsvd-transform's")))))

(test-gpu "L24: tiny data, whose squares underflow, fits as the unit data scaled"
  (for* ([scale (in-list '((float32 1e-20) (float32 1e-25) (float64 1e-160) (float64 1e-165)))]
         [solver (in-list solvers)])
    (match-define (list dt by) scale)
    (test-case (format "~a ~a ~a" dt solver by)
      (check-equal? (tiny-problems dt solver by) '()))))

(test-gpu "L13, L24: on X with subnormal entries, fit-transform's matrix is fit then transform's"
  (for ([c (in-list '((float32 full 1e-38) (float64 full 1e-308) (float32 jacobi 1e-40)))])
    (match-define (list dt solver by) c)
    (test-case (format "~a ~a ~a" dt solver by)
      (define X (rows->matrix (scaled-rows magnitude-rows by) dt))
      (define-values (_ T) (tsvd-fit-transform X #:n-components 2 #:solver solver))
      (define fitted (tsvd-fit X #:n-components 2 #:solver solver))
      (check-true (bits-equal? (device-matrix->list* T)
                               (device-matrix->list* (tsvd-transform fitted X)))))))
