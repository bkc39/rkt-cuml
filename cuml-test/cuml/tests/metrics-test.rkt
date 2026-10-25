#lang racket/base

(require (only-in cuml/metrics adjusted-rand-score contingency-matrix contingency-table)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list* device-vector->list dtype shape)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/metrics-data.rkt"
                  bijection
                  host-ari
                  host-counts
                  label-pair
                  labels-of
                  monotone-maps
                  permute
                  random-permutation
                  sorted-distinct))

(define dtypes '(int32 int64))

(define (cases)
  (for/list ([seed (in-range 24)])
    (define n (+ 20 (* 13 seed)))
    (define-values (truth found)
      (label-pair n (+ 1 (modulo seed 6)) (+ 1 (modulo (* 5 seed) 7)) seed))
    (list n truth found)))

(define (table truth found #:true [dt1 'int32] #:pred [dt2 'int32])
  (match-define (contingency-table counts rows columns)
    (contingency-matrix (labels-of truth dt1) (labels-of found dt2)))
  (list counts rows columns))

(define (table-lists truth found #:true [dt1 'int32] #:pred [dt2 'int32])
  (match-define (list counts rows columns) (table truth found #:true dt1 #:pred dt2))
  (list (device-matrix->list* counts) (device-vector->list rows) (device-vector->list columns)))

(define (ari truth found #:true [dt1 'int32] #:pred [dt2 'int32])
  (adjusted-rand-score (labels-of truth dt1) (labels-of found dt2)))

(define (transpose rows)
  (if (null? rows)
      '()
      (apply map list rows)))

(test-gpu "CC1: counts are int64 k1 x k2; the label vectors are the sorted distinct values in their dtypes"
  (for* ([c (in-list (cases))]
         [dt1 (in-list dtypes)]
         [dt2 (in-list dtypes)])
    (match-define (list n truth found) c)
    (match-define (list counts rows columns) (table truth found #:true dt1 #:pred dt2))
    (check-equal? (dtype counts) 'int64)
    (check-equal? (shape counts)
                  (list (length (remove-duplicates truth)) (length (remove-duplicates found))))
    (check-equal? (dtype rows) dt1)
    (check-equal? (dtype columns) dt2)
    (check-equal? (device-vector->list rows) (sorted-distinct truth))
    (check-equal? (device-vector->list columns) (sorted-distinct found))
    (check-equal? (device-matrix->list* counts) (host-counts truth found) (format "n ~a" n))))

(test-gpu "CC2: the counts total n, and row and column sums count each label"
  (for ([c (in-list (cases))])
    (match-define (list n truth found) c)
    (match-define (list counts rows columns) (table-lists truth found))
    (check-equal? (for*/sum ([row (in-list counts)] [x (in-list row)]) x) n)
    (check-equal? (map (lambda (row) (apply + row)) counts)
                  (for/list ([t (in-list rows)])
                    (for/sum ([x (in-list truth)]) (if (= x t) 1 0))))
    (check-equal? (map (lambda (col) (apply + col)) (transpose counts))
                  (for/list ([f (in-list columns)])
                    (for/sum ([x (in-list found)]) (if (= x f) 1 0))))))

(test-gpu "CC3: swapping the arguments transposes the counts and swaps the labels"
  (for ([c (in-list (cases))])
    (match-define (list _ truth found) c)
    (match-define (list counts rows columns) (table-lists truth found))
    (check-equal? (table-lists found truth) (list (transpose counts) columns rows))))

(test-gpu "CC4: a monotone map keeps the counts and maps the labels; a bijection permutes rows"
  (for* ([c (in-list (cases))]
         [f (in-list monotone-maps)])
    (match-define (list _ truth found) c)
    (match-define (list counts rows columns) (table-lists truth found))
    (check-equal? (table-lists (map f truth) found #:true 'int64) (list counts (map f rows) columns))
    (check-equal? (table-lists truth (map f found) #:pred 'int64) (list counts rows (map f columns))))
  (for ([c (in-list (cases))])
    (match-define (list _ truth found) c)
    (match-define (list counts rows _) (table-lists truth found))
    (define f (bijection truth))
    (match-define (list mapped-counts mapped-rows _) (table-lists (map f truth) found))
    (define row-of
      (for/hash ([t (in-list rows)]
                 [count (in-list counts)])
        (values (f t) count)))
    (check-equal? mapped-counts
                  (for/list ([m (in-list mapped-rows)])
                    (hash-ref row-of m)))))

(test-gpu "CC5: permuting both vectors alike changes nothing"
  (for ([c (in-list (cases))]
        [seed (in-naturals 100)])
    (match-define (list n truth found) c)
    (define order (random-permutation n seed))
    (check-equal? (table-lists (permute truth order) (permute found order))
                  (table-lists truth found))))

(test-gpu "CC6: int32 and int64 copies of the labels, in any mix, give the same counts"
  (for ([c (in-list (cases))])
    (match-define (list _ truth found) c)
    (define expected (car (table-lists truth found)))
    (for* ([dt1 (in-list dtypes)]
           [dt2 (in-list dtypes)])
      (check-equal? (car (table-lists truth found #:true dt1 #:pred dt2)) expected))))

(test-gpu "CC7: empty labels give an empty table, one sample ((1))"
  (match-define (list counts rows columns) (table '() '()))
  (check-equal? (shape counts) '(0 0))
  (check-equal? (device-vector->list rows) '())
  (check-equal? (device-vector->list columns) '())
  (check-equal? (table-lists '(7) '(-3) #:pred 'int64) '(((1)) (7) (-3))))

(test-gpu "CA1: identical partitions score exactly 1.0 under any relabelling, one label against singletons 0.0"
  (for ([c (in-list (cases))])
    (match-define (list _ truth _) c)
    (for ([f (in-list monotone-maps)])
      (check-equal? (ari truth (map f truth) #:pred 'int64) 1.0))
    (check-equal? (ari truth (map (bijection truth) truth)) 1.0))
  (check-equal? (ari '() '()) 1.0)
  (check-equal? (ari '(4) '(-9)) 1.0)
  (check-equal? (ari '(3 3 3 3) '(1 1 1 1)) 1.0)
  (check-equal? (ari '(0 1 2 3) '(0 1 2 3)) 1.0)
  (check-equal? (ari '(5 5 5 5 5) '(0 1 2 3 4)) 0.0)
  (for ([c (in-list (cases))])
    (match-define (list _ truth found) c)
    (check-true (<= (ari truth found) 1.0))))

(test-gpu "CA2: the score is symmetric, bit for bit"
  (for ([c (in-list (cases))])
    (match-define (list _ truth found) c)
    (check-equal? (ari found truth) (ari truth found))))

(test-gpu "CA3: relabelling either vector monotonically, or both by one permutation, is bit-identical"
  (for ([c (in-list (cases))]
        [seed (in-naturals 200)])
    (match-define (list n truth found) c)
    (define expected (ari truth found))
    (for ([f (in-list monotone-maps)])
      (check-equal? (ari (map f truth) found #:true 'int64) expected)
      (check-equal? (ari truth (map f found) #:pred 'int64) expected))
    (define order (random-permutation n seed))
    (check-equal? (ari (permute truth order) (permute found order)) expected)))

(test-gpu "CA4: int32 and int64 agree bit for bit, and 2^40 labels score as dense ones"
  (for ([c (in-list (cases))])
    (match-define (list _ truth found) c)
    (define expected (ari truth found))
    (for* ([dt1 (in-list dtypes)]
           [dt2 (in-list dtypes)])
      (check-equal? (ari truth found #:true dt1 #:pred dt2) expected)))
  (check-equal? (ari (list 0 0 (expt 2 40) (expt 2 40)) '(0 0 1 1) #:true 'int64) 1.0))

(test-gpu "CA7: twenty calls give one value"
  (match-define (list _ truth found) (list-ref (cases) 7))
  (check-equal? (length (remove-duplicates (for/list ([_ (in-range 20)])
                                             (ari truth found))))
                1))

(test-gpu "CA8: the score is the Hubert-Arabie formula over the contingency counts"
  (for ([c (in-list (cases))])
    (match-define (list n truth found) c)
    (define expected (exact->inexact (host-ari truth found)))
    (test-case (format "n ~a" n)
      (check-true (<= (abs (- (ari truth found) expected)) 1e-14)
                  (format "~a against ~a" (ari truth found) expected)))))

(test-gpu "CC1, CA8: tables past raft's shared-memory sizes match a hashed host count and the host formula"
  (for ([size (in-list '((20000 101 61) (200000 3001 2003)))])
    (match-define (list n k1 k2) size)
    (define-values (truth found) (label-pair n k1 k2 (+ n k1)))
    (test-case (format "n ~a, ~a x ~a labels" n k1 k2)
      (match-define (list counts rows columns) (table-lists truth found #:pred 'int64))
      (check-equal? (list (length rows) (length columns)) (list k1 k2))
      (check-equal? counts (host-counts truth found))
      (define expected (exact->inexact (host-ari truth found)))
      (check-true (<= (abs (- (ari truth found) expected)) 1e-12)
                  (format "~a against ~a" (ari truth found) expected)))))
