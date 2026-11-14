#lang racket/base

(require (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes)
         (only-in "private/logistic-data.rkt" blob-case)
         (only-in "private/logistic-glmnet-data.rkt" check-multinomial labelled-lr-case trial))

(test-gpu "multinomial fits at C are glmnet's at lambda 1/(C N), centred, with its probabilities"
  (define trials
    (list (trial (blob-case "three blobs" 120 3 3 33) '(0.1 1.0) 2e-3 1e-6 #t)
          (trial (labelled-lr-case 'iris "species" '()) '(0.1) 1e-2 5e-5 #f)
          (trial (labelled-lr-case 'wine "cultivar" '() #:standardise? #t) '(0.1 1.0) 1e-2 5e-5 #t)))
  (for* ([dt (in-list dtypes)]
         [t (in-list trials)])
    (check-multinomial t dt)))
