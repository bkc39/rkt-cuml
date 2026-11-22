#lang racket/base

(require (only-in "private/linear-data.rkt" dtypes)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/svm-twin-data.rkt"
                  breast-cancer
                  check-sklearn-twins
                  separable-blobs
                  trial))

(define sklearn-kernels '(linear poly rbf))

(define (trials fx)
  (for*/list ([dt (in-list dtypes)]
              [kernel (in-list sklearn-kernels)])
    (trial fx dt kernel 'scale)))

(test-twin "standardised breast-cancer decisions are near sklearn's, labels equal off the boundary"
  (check-sklearn-twins (trials (breast-cancer))))

(test-twin "separable blobs give sklearn's labels and support count"
  (check-sklearn-twins (trials (separable-blobs)) #:separable? #t))
