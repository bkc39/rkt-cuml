#lang racket/base

(require (only-in "private/python-env.rkt" test-twin)
         (only-in "private/umap-data.rkt" check-parity digits jittered tail-split))

(define configs
  (for*/list ([init (in-list '(spectral random))]
              [k (in-list '(2 3 16))])
    (hasheq 'init init 'n-components k)))

(test-twin "L6: fit equals Python cuML's bitwise on digits"
  (define-values (head tail) (tail-split (digits)))
  (check-parity head tail configs #:transform? #f))

(test-twin "L6: fit and transform equal Python cuML's bitwise on digits without ties"
  (define-values (head tail) (tail-split (jittered (digits) 64)))
  (check-parity head tail configs))
