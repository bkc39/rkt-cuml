#lang racket/base

(require (only-in cuml/decomposition
                  tsvd-result-components
                  tsvd-result-explained-variance
                  tsvd-result-explained-variance-ratio
                  tsvd-result-singular-values)
         (only-in racket/list argmax make-list take)
         (only-in racket/match match-define)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in "pca-data.rkt" column dot fixture gaps)
         (only-in "scaler-data.rkt" generated-rows unit-roundoff))

(provide (struct-out exact)
         (struct-out reference)
         (struct-out svd)
         accuracy-problems
         duplicate-fixture
         exact-reference
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
         zeros-fixture)

(struct svd (components variance ratio singular) #:transparent)

(struct reference (components variance ratio singular gaps))

(struct exact (fixture spectrum components))

(define (tsvd-of result)
  (svd (device-matrix->list* (tsvd-result-components result))
       (device-vector->list (tsvd-result-explained-variance result))
       (device-vector->list (tsvd-result-explained-variance-ratio result))
       (device-vector->list (tsvd-result-singular-values result))))

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
    (for/list ([j (in-range d)])
      (for/sum ([ai (in-list a)] [row (in-list scaled)]) (* ai (list-ref row j)))))
  (define rows
    (for/list ([ai (in-list a)]
               [i (in-naturals)])
      (for/list ([wj (in-list w)]
                 [j (in-naturals)])
        (- (if (< i d)
               (list-ref (list-ref scaled i) j)
               0.0)
           (* scale ai wj)))))
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
  (match-define (cons first-row _) rows)
  (for/list ([j (in-range (length first-row))])
    (popvar (column rows j))))

(define (squared-norm rows)
  (for*/sum ([row (in-list rows)] [x (in-list row)]) (* x x)))

(define (exact-reference rows spectrum components k)
  (define variance (popvars (host-transform rows (take components k))))
  (define total (apply + (popvars rows)))
  (reference (take components k)
             variance
             (map (lambda (v) (/ v total)) variance)
             (take spectrum k)
             (take (gaps (map (lambda (s) (* s s)) spectrum)) k)))

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

(define (over what got want bound)
  (and (> (abs (- got want)) bound) (format "~a: ~a vs ~a, bound ~a" what got want bound)))

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
                    (over (format "singular value ~a squared" i) (* s s) (* s-ref s-ref) bound))
                  (for/list ([row (in-list components)]
                             [row-ref (in-list (reference-components ref))]
                             [g (in-list (reference-gaps ref))]
                             [i (in-naturals)])
                    (define worst
                      (for/fold ([worst 0.0])
                                ([x (in-list row)]
                                 [y (in-list row-ref)])
                        (max worst (abs (- x y)))))
                    (over (format "component ~a times its gap" i) (* worst g) 0.0 bound))
                  (for/list ([v (in-list variance)]
                             [v-ref (in-list (reference-variance ref))]
                             [i (in-naturals)])
                    (over (format "explained variance ~a" i) v v-ref (/ bound n)))
                  (for/list ([r (in-list ratio)]
                             [r-ref (in-list (reference-ratio ref))]
                             [i (in-naturals)])
                    (over (format "ratio ~a" i) r r-ref (/ bound (* n total)))))))
