#lang racket/base

(require (only-in rackunit check-equal? check-pred check-true test-case)
         (only-in "private/common-data.rkt" columns-of unit-roundoff)
         (only-in "private/exact-algebra.rkt" dot float32-of)
         (only-in "private/linear-centring-data.rkt"
                  bottom-grid-fit
                  case-weights
                  centred-solve
                  check-host-fit
                  device-means
                  failed-laws
                  fails?
                  medium-kernel-mean
                  offset-grid
                  serial-mean
                  sherman-morrison-step
                  weighted-mean)
         (only-in "private/linear-data.rkt" lcase-X lcase-y oracle-lstsq solution-coef)
         (only-in "private/real-linear-data.rkt" real-lcase))

(test-case "breast-cancer float32 offset by 2^14: the exact solve of the problem centred with medium-kernel means meets T1 to T4"
  (define c (real-lcase 'breast-cancer '((offset . 14) . unweighted) 'float32))
  (check-host-fit c
                  (centred-solve c
                                 (map medium-kernel-mean (columns-of (lcase-X c)))
                                 (medium-kernel-mean (lcase-y c)))))

(test-case "breast-cancer float32 offset by 2^10, rounded c* and serial float32 means: the residual sum exceeds the r1 T4 term, and T1 to T4 hold by an empirical margin"
  (define c (real-lcase 'breast-cancer '((offset . 10) . unweighted) 'float32))
  (define X (lcase-X c))
  (define y (lcase-y c))
  (define coef
    (for/list ([cj (in-list (solution-coef (oracle-lstsq X y #f #t)))])
      (float32-of cj)))
  (define means (map serial-mean (columns-of X)))
  (define products
    (for/fold ([s 0.0])
              ([m (in-list means)]
               [cj (in-list coef)])
      (float32-of (+ s (float32-of (* m cj))))))
  (define b (inexact->exact (float32-of (- (serial-mean y) products))))
  (define exact-coef (map inexact->exact coef))
  (define weights (case-weights c))
  (define r1-term
    (* (inexact->exact (unit-roundoff 'float32))
       (apply + weights)
       (+ (abs b)
          (for/sum ([column (in-list (columns-of X))] [cj (in-list exact-coef)])
                   (abs (* (weighted-mean weights column) cj))))))
  (define residual-sum
    (for/sum ([row (in-list X)] [yi (in-list y)] [wi (in-list weights)])
             (* wi (- yi b (dot row exact-coef)))))
  (check-true (> (abs residual-sum) r1-term)
              (format "residual sum ~a of the r1 term"
                      (exact->inexact (/ (abs residual-sum) r1-term))))
  (check-host-fit c (list exact-coef b)))

(test-case "float32 columns offset by 2^16, means off by their bound along G~'s bottom eigenvector: T1 holds by C1 alone"
  (define c (offset-grid 16))
  (define exact (oracle-lstsq (lcase-X c) (lcase-y c) #f #t))
  (define-values (_epsilon fit) (bottom-grid-fit c (unit-roundoff 'float32)))
  (check-equal? (failed-laws c 'float32 fit exact #:means device-means) '())
  (check-pred (fails? "T1") (failed-laws c 'float32 fit exact #:coef-centring? #f #:means device-means)))

(test-case "float32 columns offset by 2^20, means off by their bound along G~'s bottom eigenvector: C1 is the Sherman–Morrison cap, and T1 fails with it halved"
  (define c (offset-grid 20))
  (define exact (oracle-lstsq (lcase-X c) (lcase-y c) #f #t))
  (define-values (epsilon fit) (bottom-grid-fit c (unit-roundoff 'float32)))
  (define step (sherman-morrison-step c epsilon))
  (check-true (<= 1/2 step 1) (format "the Sherman–Morrison step ~a" (exact->inexact step)))
  (check-equal? (failed-laws c 'float32 fit exact #:means device-means) '())
  (check-pred (fails? "T1") (failed-laws c 'float32 fit exact #:cap-scale 1/2 #:means device-means)))
