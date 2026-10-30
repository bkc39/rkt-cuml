#lang racket/base

(require (only-in rackunit check-equal? check-false check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in raft/tests/private/probe test-pools)
         (only-in "private/cuml-error.rkt" check-cuml-error raised)
         (only-in "private/refusal-rows.rkt" check-row row-call row-id)
         (only-in "private/spectral-data.rkt" graph-upper unit-rings)
         (only-in "private/spectral-embedding-refusal-data.rkt"
                  asymmetric-rows
                  check-pool-kept
                  check-then-valid
                  eigen-refusal-rows
                  eigen-rows
                  fault
                  heavy-row-advised?
                  heavy-rows
                  unit-weight-rows))

(define (refusal-message r)
  (exn-message (raised (row-call r))))

(test-gpu "L13: fit-graph refuses each graph whose eigenvectors are wrong with its message and advice"
  (for ([r (in-list (eigen-refusal-rows))])
    (test-case (row-id r)
      (check-row 'spectral-embedding-fit-graph r))))

(test-gpu "L13: a valid call succeeds after a refused eigensolve"
  (for-each check-then-valid (eigen-refusal-rows)))

(test-gpu "L13: a unit-weight graph refusal names no remedy"
  (for ([r (in-list (append (unit-weight-rows) (eigen-rows)))])
    (check-false (regexp-match? #rx"random-state|norm-laplacian off|orders of magnitude"
                                (refusal-message r))
                 (row-id r))))

(test-gpu "L13: the heavy-weight advice needs summed weights spanning 2^8, without norm-laplacian"
  (for ([r (in-list (heavy-rows))])
    (check-equal? (regexp-match? #rx"orders of magnitude" (refusal-message r))
                  (heavy-row-advised? r)
                  (row-id r))))

(test-gpu "L13: a refused graph that is not symmetric is named so, not blamed on repeated eigenvalues"
  (for ([r (in-list (asymmetric-rows))])
    (define message (refusal-message r))
    (check-true (regexp-match? #rx"; the graph is not symmetric$" message) (row-id r))
    (check-false (regexp-match? #rx"repeats eigenvalues" message) (row-id r))))

(test-gpu "L13: a graph that is not symmetric can instead fail cuML's eigensolver, passed through"
  (check-cuml-error 'generic
                    #rx"^spectral-embedding-fit-graph: .*eigensolver couldn't converge"
                    (fault (graph-upper (unit-rings 1 30))
                           (hasheq 'norm-laplacian #f 'random-state 1))))

(test-pools "L13: an eigensolver refusal leaves the RMM pool where it was"
  (for-each check-pool-kept (eigen-refusal-rows)))
