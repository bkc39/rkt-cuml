#lang racket/base

(require (only-in cuml/linear-model elastic-net-fit elastic-net-predict lasso-fit)
         (only-in "private/elastic-net-data.rkt" correlated-case)
         (only-in "private/linear-data.rkt"
                  docs-case
                  lcase-X
                  lcase-w
                  lcase-y
                  on-matrix
                  on-vector
                  scaled-case
                  scaled-columns)
         (only-in "private/process-memory.rkt" test-fit-memory))

(define (setup)
  (define c (correlated-case 2000 7 0.5 31 #:weighted? #t))
  (define narrow (scaled-columns (correlated-case 100 2 0 23) (list (expt 2 -60) (expt 2 60)) 1))
  (define wide (scaled-case docs-case #:X 100000))
  (define X (on-matrix (lcase-X c) 'float32))
  (define y (on-vector (lcase-y c) 'float32))
  (define w (on-vector (lcase-w c) 'float32))
  (define narrow-X (on-matrix (lcase-X narrow) 'float32))
  (define narrow-y (on-vector (lcase-y narrow) 'float32))
  (define wide-X (on-matrix (lcase-X wide) 'float32))
  (define wide-y (on-vector (lcase-y wide) 'float32))
  (values
   (lambda ()
     (elastic-net-predict (elastic-net-fit X y #:sample-weight w #:alpha 0.01 #:selection 'random) X))
   (list (list "elastic-net-fit: selection: expected cyclic or random"
               (lambda () (elastic-net-fit X y #:selection 'shuffled)))
         (list "elastic-net-fit: l1-ratio: expected a finite value from 0 to 1, got 1.5"
               (lambda () (elastic-net-fit X y #:l1-ratio 1.5)))
         (list #rx"^elastic-net-fit: X: column 0 is too narrow "
               (lambda () (elastic-net-fit narrow-X narrow-y #:alpha 1e-3)))
         (list #rx"^lasso-fit: X: coordinate descent stopped after 1 epochs "
               (lambda () (lasso-fit wide-X wide-y #:alpha 1e4))))))

(test-fit-memory "elastic-net" #:rounds 200 #:sample-every 20 #:setup setup)
