#lang racket/base

(require (only-in cuml/svm
                  svc-decision-function
                  svc-fit
                  svc-predict
                  svc-result-gamma
                  svc-result-n-support
                  svc-result-support)
         (only-in racket/list append*)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/common-data.rkt" unit-roundoff)
         (only-in "private/linear-data.rkt" check-law-case dtypes law-case on-matrix)
         (only-in "private/svm-data.rkt" dyadic-blobs kernels))

(define docs-X '((1.0 2.0) (2.0 1.0) (2.0 3.0) (6.0 5.0) (7.0 8.0) (8.0 6.0)))
(define docs-y '(a a a b b b))

(test-gpu "the docs example separates its two classes"
  (for ([dt (in-list dtypes)])
    (test-case (format "~a" dt)
      (define X (on-matrix docs-X dt))
      (define r (svc-fit X docs-y #:kernel 'linear))
      (check-equal? (svc-predict r X) docs-y)
      (check-equal? (map (lambda (d) (>= d 0)) (device-vector->list (svc-decision-function r X)))
                    '(#f #f #f #t #t #t))
      (check-equal? (apply + (svc-result-n-support r)) (length (device-vector->list (svc-result-support r)))))))

(define (exact-variance rows)
  (define cells (map inexact->exact (append* rows)))
  (define n (length cells))
  (define mean (/ (apply + cells) n))
  (/ (for/sum ([v (in-list cells)]) (expt (- v mean) 2)) n))

(define (shifted n d)
  (for/list ([i (in-range n)])
    (for/list ([j (in-range d)])
      (exact->inexact (+ (modulo i 8) (* 16 j) 1/4)))))

(define gamma-shapes '((40 3) (300 4) (2000 10)))

(test-gpu "G1: gamma 'scale is 1/(d·var) with var within its bound of the exact variance"
  (for* ([dt (in-list dtypes)]
         [shape (in-list gamma-shapes)]
         [arm (in-list '(blobs shifted))])
    (match-define (list n d) shape)
    (define rows
      (if (eq? arm 'blobs)
          (let-values ([(rows _labels) (dyadic-blobs n d 5)]) rows)
          (shifted n d)))
    (define-values (_rows labels) (dyadic-blobs n d 5))
    (test-case (format "~a ~a ~a" dt arm shape)
      (define r (svc-fit (on-matrix rows dt) labels #:kernel 'rbf))
      (define bound
        (+ (* (+ (* 2 (integer-length (* n d))) 1) (expt 2 -53))
           (unit-roundoff dt)
           (expt 2 -52)))
      (check-law-case "G1"
                      (abs (- (* (inexact->exact (svc-result-gamma r)) d (exact-variance rows)) 1))
                      bound))))

(test-gpu "G2: gamma 'auto is exactly 1/d"
  (for* ([dt (in-list dtypes)]
         [d (in-list '(1 3 7 10))])
    (define-values (rows labels) (dyadic-blobs 60 d 3))
    (test-case (format "~a ~a" dt d)
      (law-case "G2"
        (check-equal? (svc-result-gamma (svc-fit (on-matrix rows dt) labels #:gamma 'auto))
                      (exact->inexact (/ 1 d)))))))

(test-gpu "a numeric gamma is kept as given, for every kernel"
  (define-values (rows labels) (dyadic-blobs 60 3 3))
  (for ([kernel (in-list kernels)])
    (check-true (= (svc-result-gamma (svc-fit (on-matrix rows 'float64) labels
                                              #:kernel kernel
                                              #:gamma 0.25))
                   0.25)
                (format "~a" kernel))))

