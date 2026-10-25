#lang racket/base

(require (only-in cuml/decomposition tsvd-params)
         (only-in racket/list append*)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in "private/pca-data.rkt" fixture-name)
         (only-in "private/python-env.rkt" test-twin twin-params)
         (only-in "private/tsvd-data.rkt" svd)
         (only-in "private/tsvd-twin-data.rkt" float-bits ours twin-case twin-cases twin-results))

(define (requests)
  (for*/list ([c (in-list twin-cases)]
              [dt (in-list '(float32 float64))]
              [solver (in-list '(full jacobi))])
    (list "transform" c dt solver)))

(test-twin "L4, L18: every field, the transform and the inverse are cuML-Python's, bit for bit"
  (define asked (requests))
  (for ([request (in-list asked)]
        [theirs (in-list (twin-results asked))])
    (match-define (list _ (and c (twin-case f k)) dt solver) request)
    (test-case (format "~a ~a ~a k=~a" (fixture-name f) dt solver k)
      (define-values (_rows fitted T I) (ours c dt solver))
      (match-define (svd components variance ratio singular) fitted)
      (for ([mine (list (append* components) singular variance ratio (append* T) (append* I))]
            [name (in-list '(components singular_values
                                        explained_variance
                                        explained_variance_ratio
                                        transform
                                        inverse))])
        (check-equal? (float-bits dt mine)
                      (hash-ref theirs (string->symbol (format "~a_bits" name)))
                      (symbol->string name))))))

(define renames (hasheq 'algorithm 'solver 'n-iter 'iterations))

(test-twin "L20: the hyperparameters are cuML-Python's but random-state, with tol the one departure"
  (match-define (cons first-case _) twin-cases)
  (match-define (list theirs) (twin-results (list (list "params" first-case 'float64 'full))))
  (define mapped
    (for/hasheq ([(key value) (in-hash (twin-params theirs '(random-state)))])
      (values (hash-ref renames key key) value)))
  (check-equal? (hash-ref mapped 'tol) 1e-7)
  (check-equal? (hash-set mapped 'tol 0.0) (tsvd-params)))
