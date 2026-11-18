#lang racket/base

(require (only-in cuml/linear-model linear-regression-fit linear-regression-result-coef)
         (only-in rackunit check-equal? test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/blob-data.rkt" uniform-source)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/linear-data.rkt"
                  dyadic
                  lcase-X
                  lcase-y
                  normal-case
                  [on-matrix matrix]
                  [on-vector vector]))

(define (summed-column n seed [eps 0])
  (define c (normal-case n 2 seed))
  (define next (uniform-source (+ 31 seed)))
  (values (for/list ([row (in-list (lcase-X c))])
            (append row (list (+ (apply + row) (* eps (dyadic (- (next) 1/2)))))))
          (lcase-y c)))

(define (fit-with algorithm rows y)
  (linear-regression-fit (matrix rows 'float32) (vector y 'float32) #:algorithm algorithm))

(define (collinear-refusal algorithm)
  (pregexp
   (format
    "^linear-regression-fit: X: the ~a solve's Gram matrix has u[*]kappa (inf|[0-9.e+-]+) >= 2\\^-8; X may have collinear columns$"
    algorithm)))

(define solves '((auto . eig) (eig . eig) (svd . svd) (svd-jacobi . svd) (qr . qr) (svd-qr . svd-qr)))

(define nearly
  '((auto . 1/4) (eig . 1/4) (svd . 1/128) (svd-jacobi . 1/128) (qr . 1/128) (svd-qr . 1/128)))

(test-gpu "every float32 algorithm refuses exactly collinear X"
  (for* ([s (in-list solves)]
         [n (in-list '(60 1000))]
         [seed (in-range 30)])
    (define-values (rows y) (summed-column n seed))
    (test-case (format "~a ~a rows seed ~a" (car s) n seed)
      (check-cuml-error 'logic (collinear-refusal (cdr s)) (lambda () (fit-with (car s) rows y))))))

(test-gpu "every float32 algorithm fits nearly collinear X"
  (for* ([s (in-list nearly)]
         [n (in-list '(60 1000))]
         [seed (in-range 30)])
    (define-values (rows y) (summed-column n seed (cdr s)))
    (test-case (format "~a ~a rows seed ~a eps ~a" (car s) n seed (cdr s))
      (check-equal?
       (length (device-vector->list (linear-regression-result-coef (fit-with (car s) rows y))))
       3))))
