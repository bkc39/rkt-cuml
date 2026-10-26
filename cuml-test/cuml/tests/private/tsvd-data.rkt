#lang racket/base

(require (only-in cuml/decomposition
                  tsvd-result-components
                  tsvd-result-explained-variance
                  tsvd-result-explained-variance-ratio
                  tsvd-result-singular-values)
         (only-in racket/list argmax make-list take)
         (only-in racket/match match-define)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in threading ~>)
         (only-in "pca-data.rkt" dot fixture gaps over-bound)
         (only-in "scaler-data.rkt" generated-rows transpose unit-roundoff))

(provide (struct-out exact)
         (struct-out reference)
         (struct-out svd)
         accuracy-problems
         duplicate-fixture
         exact-reference
         extents
         first-problems
         geomspace
         householder-fixtures
         host-transform
         law-fixtures
         normal-fixture
         normal-rows-of
         offset-fixture
         popvars
         sign-matched
         squared-norm
         tie-fixture
         tsvd-of
         twos-fixture
         wide-fixture
         worst-difference
         zeros-fixture)

(struct svd (components variance ratio singular) #:transparent)

(struct reference (components variance ratio singular gaps))

(struct exact (fixture spectrum components))

(define (tsvd-of result)
  (svd (device-matrix->list* (tsvd-result-components result))
       (device-vector->list (tsvd-result-explained-variance result))
       (device-vector->list (tsvd-result-explained-variance-ratio result))
       (device-vector->list (tsvd-result-singular-values result))))

(define (extents rows)
  (match-define (cons first-row _) rows)
  (values (length rows) (length first-row)))

(define (first-problems problems)
  (take problems (min 3 (length problems))))

(define (worst-difference xs ys)
  (for/fold ([worst 0.0])
            ([x (in-list xs)]
             [y (in-list ys)])
    (max worst (abs (- x y)))))

(define (normal-rows-of seed n d)
  (generated-rows seed n d 'float64 (lambda (z _u _j) z)))

(define normal-fixture (fixture "normal" (normal-rows-of 31 500 20)))

(define offset-fixture
  (fixture "offset"
           (for/list ([row (in-list (normal-rows-of 32 2000 16))])
             (for/list ([x (in-list row)])
               (+ x 3.0)))))

(define duplicate-fixture
  (fixture "duplicate columns"
           (for/list ([row (in-list (normal-rows-of 33 30 4))])
             (match-define (list a b c e) row)
             (list a b a c e a))))

(define wide-fixture (fixture "n < d" (normal-rows-of 34 4 10)))

(define zeros-fixture (fixture "zeros" (make-list 6 (make-list 4 0.0))))

(define twos-fixture (fixture "twos" (make-list 6 (make-list 4 2.0))))

(define tie-fixture (fixture "tie" '((1.0 -1.0) (-1.0 1.0) (2.0 -2.0) (0.5 -0.5))))

(define (reflector a)
  (define scale (/ 2.0 (dot a a)))
  (for/list ([ai (in-list a)]
             [i (in-naturals)])
    (for/list ([aj (in-list a)]
               [j (in-naturals)])
      (- (if (= i j) 1.0 0.0) (* scale ai aj)))))

(define (positive-largest row)
  (if (negative? (argmax abs row))
      (map - row)
      row))

(define (geomspace top d)
  (for/list ([i (in-range d)])
    (expt top (/ (- d 1 i) (- d 1.0)))))

(define (householder seed n d top)
  (match-define (list a) (normal-rows-of seed 1 n))
  (match-define (list b) (normal-rows-of (add1 seed) 1 d))
  (define spectrum (geomspace top d))
  (define vt (reflector b))
  (define scaled
    (for/list ([s (in-list spectrum)]
               [row (in-list vt)])
      (map (lambda (v) (* s v)) row)))
  (define scale (/ 2.0 (dot a a)))
  (define w
    (map (lambda (column) (for/sum ([ai (in-list a)] [x (in-list column)]) (* ai x)))
         (transpose scaled)))
  (define padded (append scaled (make-list (- n d) (make-list d 0.0))))
  (define rows
    (for/list ([ai (in-list a)]
               [row (in-list padded)])
      (map (lambda (x wj) (- x (* scale ai wj))) row w)))
  (exact (fixture (format "householder ~a" top) rows) spectrum (map positive-largest vt)))

(define householder-fixtures
  (for/list ([top (in-list '(10.0 1e2 1e3))]
             [seed (in-naturals 40)])
    (householder (* 3 seed) 200 12 top)))

(define law-fixtures
  (append (map exact-fixture householder-fixtures)
          (list normal-fixture offset-fixture duplicate-fixture wide-fixture)))

(define (host-transform rows components)
  (for/list ([row (in-list rows)])
    (for/list ([c (in-list components)])
      (dot row c))))

(define (popvar xs)
  (define exact-xs (map inexact->exact xs))
  (define n (length xs))
  (define mean (/ (apply + exact-xs) n))
  (real->double-flonum (/ (for/sum ([x (in-list exact-xs)]) (* (- x mean) (- x mean))) n)))

(define (popvars rows)
  (map popvar (transpose rows)))

(define (squared-norm rows)
  (for*/sum ([row (in-list rows)] [x (in-list row)]) (* x x)))

(define (exact-reference rows spectrum components k)
  (define variance (~> components (take k) (host-transform rows _) popvars))
  (define total (apply + (popvars rows)))
  (reference (take components k)
             variance
             (map (lambda (v) (/ v total)) variance)
             (take spectrum k)
             (~> (map (lambda (s) (* s s)) spectrum) gaps (take k))))

(define (sign-matched ref got)
  (struct-copy reference
               ref
               [components
                (map (lambda (row-ref row)
                       (if (negative? (dot row-ref row))
                           (map - row-ref)
                           row-ref))
                     (reference-components ref)
                     (svd-components got))]))

(define (accuracy-problems got ref rows dt c)
  (match-define (svd components variance ratio singular) got)
  (define u (unit-roundoff dt))
  (define n (length rows))
  (define total (apply + (popvars rows)))
  (define bound (* c u (squared-norm rows)))
  (filter values
          (append (for/list ([s (in-list singular)]
                             [s-ref (in-list (reference-singular ref))]
                             [i (in-naturals)])
                    (over-bound (format "singular value ~a squared" i) (* s s) (* s-ref s-ref) bound))
                  (for/list ([row (in-list components)]
                             [row-ref (in-list (reference-components ref))]
                             [g (in-list (reference-gaps ref))]
                             [i (in-naturals)])
                    (over-bound (format "component ~a times its gap" i)
                                (* (worst-difference row row-ref) g)
                                0.0
                                bound))
                  (for/list ([v (in-list variance)]
                             [v-ref (in-list (reference-variance ref))]
                             [i (in-naturals)])
                    (over-bound (format "explained variance ~a" i) v v-ref (/ bound n)))
                  (for/list ([r (in-list ratio)]
                             [r-ref (in-list (reference-ratio ref))]
                             [i (in-naturals)])
                    (over-bound (format "ratio ~a" i) r r-ref (/ bound (* n total)))))))
