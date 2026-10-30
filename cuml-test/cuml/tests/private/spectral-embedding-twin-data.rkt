#lang racket/base

(provide parity-configs)

(define (parity-configs [n-neighbors #f])
  (for*/list ([k (in-list '(2 8))]
              [drop (in-list '(#t #f))]
              [norm (in-list '(#t #f))])
    (define options (hasheq 'n-components k 'drop-first drop 'norm-laplacian norm))
    (if n-neighbors
        (hash-set options 'n-neighbors n-neighbors)
        options)))
