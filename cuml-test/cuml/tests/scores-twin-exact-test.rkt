#lang racket/base

(require (only-in cuml/metrics accuracy-score r2-score)
         (only-in json json-null)
         (only-in racket/match match-define)
         (only-in racket/math nan?)
         (only-in rackunit check-= check-equal? check-true test-case)
         (only-in "private/blob-data.rkt" uniform-source)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/scores-data.rkt" label-cases on vec weights-of)
         (only-in "private/scores-twin-data.rkt" case-of checked-twin-results))

(define (weighted-label-cases)
  (define next (uniform-source 862))
  (for/list ([c (in-list (label-cases #:seed 861 #:count 60 #:most 3000))])
    (match-define (list i at bt truth found) c)
    (define w
      (and (odd? (quotient i 3))
           (for/list ([_ (in-list truth)])
             (* 3.0 (next)))))
    (list i at bt truth found w)))

(test-twin "T3: accuracy equals sklearn and Python cuML unweighted, within 1e-15 weighted"
  (define cases (weighted-label-cases))
  (define results
    (checked-twin-results (for/list ([c (in-list cases)])
                            (match-define (list _ at bt truth found w) c)
                            (case-of "accuracy" truth found w #:dtype at #:pred-dtype bt))))
  (for ([c (in-list cases)]
        [theirs (in-list results)])
    (match-define (list i at bt truth found w) c)
    (match-define (hash-table ('sklearn sk) ('cuml cu)) theirs)
    (define ours (accuracy-score (vec truth at) (vec found bt) #:sample-weight (weights-of w)))
    (test-case (format "case ~a ~a x ~a~a" i at bt (if w " weighted" ""))
      (cond
        [w
         (check-= ours sk 1e-15 (format "~a against sklearn's ~a" ours sk))
         (check-= ours cu 1e-15 (format "~a against Python cuML's ~a" ours cu))]
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
    (checked-twin-results (for/list ([c (in-list cases)])
                            (match-define (list y p _) c)
                            (case-of "r2" y p #f #:dtype 'float64))))
  (for ([c (in-list cases)]
        [theirs (in-list results)])
    (match-define (list y p expected) c)
    (match-define (hash-table ('sklearn sk) ('cuml cu)) theirs)
    (define ours (r2-score (on y 'float64) (on p 'float64)))
    (test-case (format "~a against ~a" y p)
      (check-equal? ours expected)
      (check-equal? ours cu)
      (cond
        [(and (real? sk) (> (length y) 1)) (check-equal? ours sk)]
        [else (check-true (or (eq? sk (json-null)) (nan? sk)) "sklearn's NaN at n = 1")]))))
