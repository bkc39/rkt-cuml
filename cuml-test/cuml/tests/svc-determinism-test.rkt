#lang racket/base

(require (only-in cuml/svm svc-decision-function svc-fit svc-predict)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes law-case on-matrix)
         (only-in "private/svm-data.rkt" dyadic-blobs fit-parts fresh-rows kernels svm-case)
         (only-in "private/svm-oracle.rkt" check-kkt))

(define-values (big-rows big-labels) (dyadic-blobs 20000 20 9))

(test-gpu "D1: three 20000×20 fits in one process are bit-identical"
  (for* ([dt (in-list dtypes)]
         [kernel (in-list '(linear rbf))])
    (test-case (format "~a ~a" dt kernel)
      (define X (on-matrix big-rows dt))
      (match-define (list first-fit second-fit third-fit)
        (for/list ([_ (in-range 3)])
          (fit-parts (svc-fit X big-labels #:kernel kernel))))
      (law-case "D1"
        (check-equal? second-fit first-fit)
        (check-equal? third-fit first-fit)))))

(define-values (rows labels) (dyadic-blobs 300 4 4))
(define probe (fresh-rows 200 4 4))

(test-gpu "D2: a cache that holds the kernel leaves the fit bit for bit as it was"
  (for* ([dt (in-list dtypes)]
         [kernel (in-list kernels)])
    (test-case (format "~a ~a" dt kernel)
      (define X (on-matrix rows dt))
      (match-define (list middle large)
        (for/list ([cache (in-list '(1024 4096))])
          (fit-parts (svc-fit X labels #:kernel kernel #:cache-size cache))))
      (law-case "D2"
        (check-equal? large middle)))))

(test-gpu "D2: a small cache changes the path, and the fit still passes K"
  (for* ([dt (in-list dtypes)]
         [kernel (in-list kernels)]
         [cache (in-list '(1 100))])
    (test-case (format "~a ~a cache ~a" dt kernel cache)
      (define c (svm-case (format "cache ~a" cache) rows labels #f 1 kernel dt 'scale))
      (check-kkt c (fit-parts (svc-fit (on-matrix rows dt) labels #:kernel kernel #:cache-size cache))))))

(test-gpu "D3: predicting twice gives bit-identical results"
  (for* ([dt (in-list dtypes)]
         [kernel (in-list '(linear poly rbf sigmoid))])
    (test-case (format "~a ~a" dt kernel)
      (define r (svc-fit (on-matrix rows dt) labels #:kernel kernel))
      (define P (on-matrix probe dt))
      (law-case "D3"
        (check-equal? (device-vector->list (svc-decision-function r P))
                      (device-vector->list (svc-decision-function r P)))
        (check-equal? (svc-predict r P) (svc-predict r P))))))
