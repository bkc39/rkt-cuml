#lang racket/base

(require (only-in racket/list remove-duplicates)
         (only-in racket/math pi)
         (only-in raft list*->device-matrix list->device-vector))

(provide bijection
         host-counts
         label-pair
         labels
         labels-of
         monotone-maps
         normals
         overlap
         overlap-labels
         permute
         projection
         random-permutation
         sorted-distinct
         three-groups
         three-groups-labels
         grid-20)

(define (seeded seed)
  (define g (make-pseudo-random-generator))
  (parameterize ([current-pseudo-random-generator g])
    (random-seed seed))
  g)

(define (labels n k seed)
  (define g (seeded seed))
  (for/list ([_ (in-range n)])
    (random k g)))

(define (label-pair n k1 k2 seed #:mix [mix 0.3])
  (define g (seeded seed))
  (define truth
    (for/list ([_ (in-range n)])
      (random k1 g)))
  (define found
    (for/list ([t (in-list truth)])
      (if (< (random g) mix)
          (random k2 g)
          (modulo t k2))))
  (values truth found))

(define (labels-of xs dt)
  (list->device-vector xs #:dtype dt))

(define (normals count seed)
  (define g (seeded seed))
  (let loop ([left count]
             [acc '()])
    (cond
      [(<= left 0) (reverse acc)]
      [else
       (define u (- 1.0 (random g)))
       (define v (random g))
       (define r (sqrt (* -2.0 (log u))))
       (loop (- left 2) (list* (* r (sin (* 2.0 pi v))) (* r (cos (* 2.0 pi v))) acc))])))

(define (rows-of xs d)
  (let loop ([xs xs]
             [acc '()])
    (cond
      [(null? xs) (reverse acc)]
      [else
       (define-values (row rest) (split xs d))
       (loop rest (cons row acc))])))

(define (split xs d)
  (for/fold ([row '()]
             [rest xs]
             #:result (values (reverse row) rest))
            ([_ (in-range d)])
    (values (cons (car rest) row) (cdr rest))))

(define (normal-rows n d seed)
  (rows-of (normals (* n d) seed) d))

(define overlap-each 500)
(define overlap-k 4)
(define overlap-d 5)
(define overlap-gap 2.0)
(define overlap-offset 3)

(define (overlap dt #:each [each overlap-each])
  (define rows
    (for*/list ([c (in-range overlap-k)]
                [row (in-list (normal-rows each overlap-d (+ 1 c)))])
      (for/list ([x (in-list row)])
        (+ x (* overlap-gap c)))))
  (list*->device-matrix rows #:dtype dt))

(define (overlap-labels #:each [each overlap-each])
  (for*/list ([c (in-range overlap-k)]
              [_ (in-range each)])
    (+ c overlap-offset)))

(define three-groups '((0 0) (0 1) (1 0) (1 1) (6 6) (6 7) (7 6) (12 0) (12 1) (13 0)))
(define three-groups-labels '(0 0 0 0 1 1 1 2 2 2))

(define (matmul a b)
  (define columns (apply map list b))
  (for/list ([row (in-list a)])
    (for/list ([col (in-list columns)])
      (for/sum ([x (in-list row)] [y (in-list col)]) (* x y)))))

(define (projection n m d #:seed [seed 0])
  (define X (normal-rows n m seed))
  (define W (normal-rows m d (+ seed 1000)))
  (values (list*->device-matrix X #:dtype 'float32)
          (list*->device-matrix (matmul X W) #:dtype 'float32)))

(define grid-20
  (for*/list ([i (in-range 20)]
              [j (in-range 20)])
    (list i j)))

(define monotone-maps
  (list (lambda (x) (- (* 1000 x) 7))
        (lambda (x) (* x (expt 2 40)))
        (lambda (x) (* (- x 1) (expt 2 40)))
        sub1))

(define (bijection xs)
  (define distinct (sorted-distinct xs))
  (define reversed (reverse distinct))
  (define table
    (for/hash ([from (in-list distinct)]
               [to (in-list (append (cdr reversed) (list (car reversed))))])
      (values from to)))
  (lambda (x) (hash-ref table x)))

(define (sorted-distinct xs)
  (sort (remove-duplicates xs) <))

(define (host-counts truth found)
  (for/list ([t (in-list (sorted-distinct truth))])
    (for/list ([f (in-list (sorted-distinct found))])
      (for/sum ([a (in-list truth)] [b (in-list found)]) (if (and (= a t) (= b f)) 1 0)))))

(define (random-permutation n seed)
  (define g (seeded seed))
  (define keyed
    (for/list ([i (in-range n)])
      (cons (random g) i)))
  (map cdr (sort keyed < #:key car)))

(define (permute xs order)
  (define v (list->vector xs))
  (for/list ([i (in-list order)])
    (vector-ref v i)))
