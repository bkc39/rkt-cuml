#lang racket/base

(require (only-in racket/flonum flsingle)
         (only-in raft device-matrix->list* list*->device-matrix list->device-vector)
         (only-in "scaler-data.rkt" bits-equal? normal-source))

(provide (struct-out fixture)
         bits-equal?
         blobs
         embedding-rows
         far-blobs
         finite-rows?
         fixture-X
         fixture-y
         held-out-fit
         held-out-new
         overlap
         quality-floors
         relabel
         tiny)

(struct fixture (name rows labels))

(define (uniform-source seed)
  (define g (make-pseudo-random-generator))
  (parameterize ([current-pseudo-random-generator g])
    (random-seed seed))
  (lambda (lo hi)
    (parameterize ([current-pseudo-random-generator g])
      (+ lo (* (- hi lo) (random))))))

(define (grouped name seed centres per-group)
  (define next (normal-source seed))
  (define rows
    (for*/list ([centre (in-list centres)]
                [_ (in-range per-group)])
      (for/list ([c (in-list centre)])
        (flsingle (+ c (next))))))
  (define labels
    (for*/list ([k (in-range (length centres))]
                [_ (in-range per-group)])
      k))
  (fixture name rows labels))

(define (axis d i value)
  (for/list ([j (in-range d)])
    (if (= i j) value 0.0)))

(define blobs
  (let ([draw (uniform-source 50)])
    (grouped "blobs"
             51
             (for/list ([_ (in-range 5)])
               (for/list ([_ (in-range 10)])
                 (draw -10.0 10.0)))
             200)))

(define (every-fifth? i)
  (= (modulo i 5) 4))

(define (split-fixture f)
  (define indexed
    (for/list ([row (in-list (fixture-rows f))]
               [label (in-list (fixture-labels f))]
               [i (in-naturals)])
      (list i row label)))
  (define (part keep?)
    (define kept (filter (lambda (entry) (keep? (car entry))) indexed))
    (fixture (fixture-name f) (map cadr kept) (map caddr kept)))
  (values (part (lambda (i) (not (every-fifth? i)))) (part every-fifth?)))

(define-values (held-out-fit held-out-new) (split-fixture blobs))

(define far-blobs (grouped "far-blobs" 52 (list (axis 5 0 0.0) (axis 5 0 50.0) (axis 5 1 50.0)) 100))

(define overlap (grouped "overlap" 53 (list (axis 10 0 0.0) (axis 10 0 3.0)) 300))

(define tiny
  (fixture "tiny"
           '((0.0 0.0 0.0) (0.5 0.25 0.0)
                           (1.0 0.0 0.5)
                           (0.25 1.0 0.0)
                           (10.0 10.0 10.0)
                           (10.5 10.25 10.0)
                           (11.0 10.0 10.5)
                           (10.25 11.0 10.0))
           '(0 0 0 0 1 1 1 1)))

(define (fixture-X f)
  (list*->device-matrix (fixture-rows f) #:dtype 'float32))

(define (fixture-y f [dtype 'int32])
  (list->device-vector (fixture-labels f) #:dtype dtype))

(define (relabel labels)
  (for/list ([y (in-list labels)])
    (if (= y -1)
        -1
        (+ (* (expt 2 40) y) 7))))

(define (embedding-rows E)
  (device-matrix->list* E))

(define (finite-rows? rows)
  (for*/and ([row (in-list rows)]
             [x (in-list row)])
    (and (flonum? x) (< (abs x) +inf.0))))

(define quality-floors (hash))
