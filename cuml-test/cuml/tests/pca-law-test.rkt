#lang racket/base

(require (only-in cuml/decomposition pca-fit pca-inverse-transform pca-transform)
         (only-in racket/list last take)
         (only-in racket/match match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/pca-data.rkt"
                  axis-rows
                  first-max-entries
                  fitted
                  fitted-of
                  fixture-matrix
                  fixture-name
                  fixtures
                  gaps
                  host-gram
                  norm
                  normal-fixture
                  orthonormality-bound
                  tie-rows
                  wide-fixture)
         (only-in "private/scaler-data.rkt" bits-equal? rows->matrix unit-roundoff))

(define dtypes '(float32 float64))
(define solvers '(full jacobi))

(define (permuted rows permutation)
  (for/list ([row (in-list rows)])
    (for/list ([axis (in-range (length permutation))])
      (for/first ([x (in-list row)]
                  [column-axis (in-list permutation)]
                  #:when (= column-axis axis))
        x))))

(define (permutation-matrix permutation)
  (for/list ([axis (in-range (length permutation))])
    (for/list ([column-axis (in-list permutation)])
      (if (= column-axis axis) 1.0 0.0))))

(test-gpu "L2: the axis fixture's fit, transform and inverse are exact"
  (for* ([dt (in-list dtypes)]
         [solver (in-list solvers)]
         [permutation (in-list '((0 1 2) (2 0 1) (1 2 0)))])
    (test-case (format "~a ~a ~a" dt solver permutation)
      (define rows (axis-rows 4.0 2.0 1.0 permutation))
      (define X (rows->matrix rows dt))
      (define result (pca-fit X #:solver solver))
      (match-define (fitted components variance _ _ mean) (fitted-of result))
      (check-true (bits-equal? components (permutation-matrix permutation)) "components")
      (check-true (bits-equal? mean '(0.0 0.0 0.0)) "mean")
      (check-true (bits-equal? variance '(8.0 1.0 0.25)) "explained variance")
      (define T (pca-transform result X))
      (check-true (bits-equal? (device-matrix->list* T) (permuted rows permutation)) "transform")
      (check-true (bits-equal? (device-matrix->list* (pca-inverse-transform result T)) rows)
                  "inverse"))))

(define (max-abs-off-identity gram)
  (for*/fold ([worst 0.0])
             ([(row i) (in-indexed gram)]
              [(g j) (in-indexed row)])
    (max worst (abs (- g (if (= i j) 1.0 0.0))))))

(define (non-increasing? xs)
  (match xs
    [(list* a (and more (list* b _))) (and (>= a b) (non-increasing? more))]
    [_ #t]))

(define (spectrum-problems result-lists n k m u)
  (match-define (fitted _ variance ratio singular _) result-lists)
  (define total (apply + ratio))
  (filter values
          (list (and (not (non-increasing? variance)) (format "variance increases: ~a" variance))
                (and (not (andmap (lambda (v) (>= v 0.0)) variance))
                     (format "negative variance: ~a" variance))
                (and (not (andmap (lambda (r) (>= r 0.0)) ratio)) (format "negative ratio: ~a" ratio))
                (and (> total (+ 1.0 (* k u))) (format "ratios sum to ~a" total))
                (and (= k m) (> (abs (- total 1.0)) (* k u)) (format "full ratios sum to ~a" total))
                (for/first ([s (in-list singular)]
                            [v (in-list variance)]
                            #:when (> (abs (- s (sqrt (* v (sub1 n))))) (* 2 u s)))
                  (format "singular value ~a against variance ~a" s v)))))

(test-gpu "L3, L4, L5: signs, orthonormality and the spectrum on every fixture"
  (for* ([dt (in-list dtypes)]
         [f (in-list fixtures)]
         [solver (in-list solvers)]
         [k (in-list '(#f 3))])
    (test-case (format "~a ~a ~a k=~a" dt (fixture-name f) solver k)
      (define X (fixture-matrix f dt))
      (define result (pca-fit X #:solver solver #:n-components k))
      (define lists (fitted-of result))
      (match-define (fitted components variance _ _ mean) lists)
      (define u (unit-roundoff dt))
      (define d (length mean))
      (define n (length (device-matrix->list* X)))
      (define m (min n d))
      (check-true (andmap positive? (first-max-entries components)) "L3")
      (define off (max-abs-off-identity (host-gram components)))
      (check-true (<= off (orthonormality-bound solver d u))
                  (format "L4: |CC^T - I| = ~a, ~a d u" off (/ off (* d u))))
      (check-equal? (spectrum-problems lists n (length variance) m u) '() "L5"))))

(test-gpu "L3: the tie fixture's components take the first entry of largest magnitude positive"
  (for* ([dt (in-list dtypes)]
         [solver (in-list solvers)])
    (define result (pca-fit (rows->matrix tie-rows dt) #:solver solver))
    (match-define (fitted components _ _ _ _) (fitted-of result))
    (check-true (andmap positive? (first-max-entries components))
                (format "~a ~a ~a" dt solver components))))

(test-gpu "L11: n-components at 1 and at min(n, d), with n < d, and the k-fit is the m-fit's head"
  (for* ([dt (in-list dtypes)]
         [solver (in-list solvers)]
         [f (in-list (list wide-fixture normal-fixture))])
    (test-case (format "~a ~a ~a" dt solver (fixture-name f))
      (define X (fixture-matrix f dt))
      (define u (unit-roundoff dt))
      (match-define (fitted all-components all-variance _ _ _)
        (fitted-of (pca-fit X #:solver solver)))
      (match-define (list* largest _) all-variance)
      (define m (length all-variance))
      (when (eq? f wide-fixture)
        (check-equal? m 50)
        (check-true (<= (last all-variance) (* 64 u largest))
                    (format "the centred rank is n - 1: ~a" (last all-variance))))
      (define deltas (gaps all-variance))
      (for ([k (in-list '(1 3))])
        (match-define (fitted components _ _ _ _)
          (fitted-of (pca-fit X #:solver solver #:n-components k)))
        (for ([c (in-list components)]
              [c-all (in-list (take all-components k))]
              [delta (in-list deltas)])
          (define gap (norm (map - c c-all)))
          (check-true (<= gap (/ (* 64 u largest) delta))
                      (format "k=~a: component ~a from the m-fit's, gap ~a" k gap delta)))))))
