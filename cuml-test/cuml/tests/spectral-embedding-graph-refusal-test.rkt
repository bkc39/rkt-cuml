#lang racket/base

(require (only-in racket/match match-define)
         (only-in rackunit check-true test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in raft/tests/private/probe test-pools)
         (only-in "private/refusal-rows.rkt" check-row row-id)
         (only-in "private/spectral-data.rkt" unit-path)
         (only-in "private/spectral-embedding-data.rkt"
                  check-residuals
                  embedded
                  embedded-graph
                  sign-ruled?)
         (only-in "private/spectral-embedding-refusal-data.rkt"
                  check-pool-kept
                  check-then-valid
                  graph-shape-rows
                  path)
         (only-in "private/umap-data.rkt" finite-rows?))

(define least-graphs
  (list (list (unit-path 8) (hasheq))
        (list (unit-path 4) (hasheq 'n-components 1 'drop-first #f))
        (list (unit-path 6) (hasheq 'n-components 1))
        (list path (hasheq 'n-components 3 'drop-first #f))))

(test-gpu "L13: fit-graph refuses each badly shaped graph with its message, the first refusal winning"
  (for ([r (in-list (graph-shape-rows))])
    (test-case (row-id r)
      (check-row 'spectral-embedding-fit-graph r))))

(test-gpu "L13: g2: the least samples fit"
  (for ([entry (in-list least-graphs)])
    (match-define (list g options) entry)
    (define rows (embedded (embedded-graph g options)))
    (define label (format "~a samples" (length rows)))
    (check-true (and (finite-rows? rows) (sign-ruled? rows)) label)
    (check-residuals g rows label)))

(test-gpu "L13: a valid call succeeds after a refused graph shape"
  (for-each check-then-valid (graph-shape-rows)))

(test-pools "L13: a graph shape refusal leaves the RMM pool where it was"
  (for-each check-pool-kept (graph-shape-rows)))
