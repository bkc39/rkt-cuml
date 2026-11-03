#lang racket/base

(require (only-in cuml/metrics accuracy-score r2-score)
         (only-in json json-null)
         (only-in racket/match match match-define)
         (only-in racket/math exact-floor nan?)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft list->device-vector)
         (only-in "private/blob-data.rkt" uniform-source)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/scores-twin-data.rkt" case-of on twin-results weights-of))

(define int32-palette '(-1 0 1 2 7 2147483647 -2147483648))
(define int64-palette (append int32-palette (list (expt 2 40) (- (expt 2 40)))))

(define (label-cases)
  (define next (uniform-source 861))
  (define (pick xs)
    (list-ref xs (exact-floor (* (next) (length xs)))))
  (for/list ([i (in-range 60)])
    (match-define (list at bt) (list-ref '((int32 int32) (int32 int64) (int64 int64)) (modulo i 3)))
    (define n (add1 (exact-floor (* (next) 3000))))
    (define truth
      (for/list ([_ (in-range n)])
        (pick (if (eq? at 'int32) int32-palette int64-palette))))
    (define found
      (for/list ([t (in-list truth)])
        (if (< (next) 0.7)
            t
            (pick (if (eq? bt 'int32) int32-palette int64-palette)))))
    (define w
      (and (odd? (quotient i 3))
           (for/list ([_ (in-range n)])
             (* 3.0 (next)))))
    (list i at bt truth found w)))

(test-twin "T3: accuracy equals sklearn and Python cuML unweighted, within 1e-15 weighted"
  (define cases (label-cases))
  (define results
    (twin-results (for/list ([c (in-list cases)])
                    (match-define (list _ at bt truth found w) c)
                    (case-of "accuracy" truth found w at bt))))
  (for ([c (in-list cases)]
        [theirs (in-list results)])
    (match-define (list i at bt truth found w) c)
    (match-define (hash-table ('sklearn sk) ('cuml cu)) theirs)
    (define ours
      (accuracy-score (list->device-vector truth #:dtype at)
                      (list->device-vector found #:dtype bt)
                      #:sample-weight (weights-of w)))
    (test-case (format "case ~a ~a x ~a~a" i at bt (if w " weighted" ""))
      (cond
        [w
         (check-true (<= (abs (- ours sk)) 1e-15) (format "~a against sklearn's ~a" ours sk))
         (check-true (<= (abs (- ours cu)) 1e-15) (format "~a against Python cuML's ~a" ours cu))]
        [else
         (check-equal? ours sk)
         (check-equal? ours cu)]))))

(test-twin "T4: a constant y-true scores as both references where they agree, as Python cuML at n = 1"
  (define cases
    (list (list '(2.0 2.0 2.0) '(2.0 2.0 2.0) 1.0)
          (list '(2.0 2.0 2.0) '(2.0 2.0 3.0) 0.0)
          (list '(1.0) '(1.0) 1.0)
          (list '(1.0) '(2.0) 0.0)))
  (define results
    (twin-results (for/list ([c (in-list cases)])
                    (match-define (list y p _) c)
                    (case-of "r2" y p #f 'float64))))
  (for ([c (in-list cases)]
        [theirs (in-list results)])
    (match-define (list y p expected) c)
    (match-define (hash-table ('sklearn sk) ('cuml cu)) theirs)
    (define ours (r2-score (on y 'float64) (on p 'float64)))
    (test-case (format "~a against ~a" y p)
      (check-equal? ours expected)
      (check-equal? ours cu)
      (match sk
        [(? real? v)
         #:when (> (length y) 1)
         (check-equal? ours v)]
        [v (check-true (or (eq? v (json-null)) (nan? v)) "sklearn's NaN at n = 1")]))))
