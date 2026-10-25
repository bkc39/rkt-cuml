#lang racket/base

(require (only-in cuml/metrics
                  adjusted-rand-score
                  contingency-matrix
                  contingency-table
                  silhouette-samples
                  silhouette-score
                  trustworthiness)
         (only-in racket/list index-of range remove-duplicates take)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true)
         (only-in raft
                  device-matrix->list*
                  device-vector->list
                  list*->device-matrix
                  list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/metrics-data.rkt" host-ari mean))

(define truth-labels '(0 0 0 1 1 1))
(define found-labels '(1 1 -1 0 0 0))
(define X-rows '((0 0) (0 1) (1 0) (10 10) (10 11) (11 10)))
(define P-rows '((0 0 0) (0 1 3) (1 0 1) (3 2 0) (5 5 4) (5 6 1) (6 5 2) (7 7 7)))
(define E-rows '((0 0) (1 2) (1 0) (2 2) (5 5) (5 6) (7 6) (7 7)))

(define (truth)
  (list->device-vector truth-labels #:dtype 'int32))

(define (found)
  (list->device-vector found-labels #:dtype 'int64))

(define (distance a b)
  (sqrt (for/sum ([x (in-list a)] [y (in-list b)]) (* (- x y) (- x y)))))

(define (mean-distance p rows ys label)
  (mean (for/list ([q (in-list rows)]
                   [l (in-list ys)]
                   #:when (and (= l label) (not (eq? p q))))
          (distance p q))))

(define (host-samples rows ys)
  (for/list ([p (in-list rows)]
             [c (in-list ys)])
    (define a (mean-distance p rows ys c))
    (define b
      (apply min
             (for/list ([label (in-list (remove c (remove-duplicates ys)))])
               (mean-distance p rows ys label))))
    (/ (- b a) (max a b))))

(define (neighbours rows i)
  (define v (list->vector rows))
  (define p (vector-ref v i))
  (define others (remove i (range (vector-length v))))
  (sort others < #:key (lambda (j) (distance p (vector-ref v j)))))

(define (host-trustworthiness X E k)
  (define n (length X))
  (define penalty
    (for*/sum ([i (in-range n)] [j (in-list (take (neighbours E i) k))])
              (max 0 (- (add1 (index-of (neighbours X i) j)) k))))
  (- 1 (/ (* 2 penalty) (* n k (- (* 2 n) (* 3 k) 1)))))

(test-gpu "the manual: contingency-matrix counts the example's labels"
  (match-define (contingency-table counts rows columns) (contingency-matrix (truth) (found)))
  (check-equal? (list (device-vector->list rows) (device-vector->list columns)) '((0 1) (-1 0 1)))
  (check-equal? (device-matrix->list* counts) '((1 0 2) (0 3 0))))

(test-gpu "the manual: adjusted-rand-score on the example's labels"
  (check-true (<= (abs (- (adjusted-rand-score (truth) (found)) (host-ari truth-labels found-labels)))
                  1e-14))
  (check-equal? (adjusted-rand-score (truth) (truth)) 1.0))

(test-gpu "the manual: silhouette-score and silhouette-samples on the example's rows"
  (define X (list*->device-matrix X-rows #:dtype 'float64))
  (define y (list->device-vector truth-labels #:dtype 'int32))
  (define expected (host-samples X-rows truth-labels))
  (check-true (<= (abs (- (silhouette-score X y) (mean expected))) 1e-12))
  (check-true (for/and ([s (in-list (device-vector->list (silhouette-samples X y)))]
                        [e (in-list expected)])
                (<= (abs (- s e)) 1e-12)))
  (check-cuml-error 'logic
                    "silhouette-score: labels: expected at least 2 distinct labels, got 1"
                    (lambda ()
                      (silhouette-score X (list->device-vector '(0 0 0 0 0 0) #:dtype 'int32)))))

(test-gpu "the manual: trustworthiness of the example's embedding"
  (define P (list*->device-matrix P-rows #:dtype 'float32))
  (define E (list*->device-matrix E-rows #:dtype 'float32))
  (check-equal? (trustworthiness P P #:n-neighbors 2) 1.0)
  (define expected (exact->inexact (host-trustworthiness P-rows E-rows 2)))
  (check-true (<= (abs (- (trustworthiness P E #:n-neighbors 2) expected)) 1e-6)
              (format "~a against ~a" (trustworthiness P E #:n-neighbors 2) expected)))
