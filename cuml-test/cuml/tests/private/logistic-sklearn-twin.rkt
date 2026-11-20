#lang racket/base

(require (only-in cuml/linear-model logistic-regression-decision-function)
         (only-in racket/match match match-define struct*)
         (only-in racket/list filter-not make-list)
         (only-in rackunit check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in "common-data.rkt" kept-rows)
         (only-in "linear-data.rkt" by-dtype dtypes on-matrix)
         (only-in "logistic-data.rkt"
                  blob-case
                  collapsed
                  decision-label
                  default-tol-cap
                  fit-parts
                  gauged-fit
                  gauged-sklearn
                  lr-case-name
                  lr-case-w
                  lr-case-X
                  lr-case-y
                  penalty-settings
                  relative-gap
                  repeated-rows
                  setting-fit
                  setting-penalty
                  tight-max-iter
                  tight-tol)
         (only-in "logistic-twin-data.rkt"
                  logistic-twin-results
                  setting-twin
                  twin-trial
                  twin-trial-case))

(provide check-sklearn-twins
         check-weighted-sklearn-twins
         optimum-gap-bound
         sklearn-trials)

(define (optimum-gap-bound dt)
  (by-dtype dt 2e-3 1e-6))

(define boundary-margin 1e-6)

(define (sklearn-trials cases settings #:dtypes [dts dtypes])
  (for*/list ([dt (in-list dts)]
              [c (in-list cases)]
              [s (in-list settings)])
    (twin-trial dt c s #t)))

(struct oracle (seen counts solver-gap))

(define (oracle-of c repeated?)
  (cond
    [repeated?
     (define counts (lr-case-w c))
     (define spread (repeated-rows c counts))
     (oracle spread counts (lambda (gap-of-fit) (gap-of-fit spread)))]
    [else (oracle c (make-list (length (lr-case-y c)) 1) (lambda (_) 0))]))

(define (centre-coef? penalty)
  (and (memq penalty '(l1 none)) #t))

(define (margin z)
  (match (sort z >)
    [(list top) (abs top)]
    [(list* top next _) (- top next)]))

(define (check-sklearn-twins rows #:repeated? [repeated? #f])
  (define oracles
    (for/list ([r (in-list rows)])
      (oracle-of (twin-trial-case r) repeated?)))
  (define (request r o)
    (match-define (struct* twin-trial ([dt dt] [setting s])) r)
    (setting-twin "sklearn-fit" (oracle-seen o) dt s #:tol (tight-tol dt)))
  (for ([r (in-list rows)]
        [o (in-list oracles)]
        [theirs (in-list (logistic-twin-results (map request rows oracles)))])
    (match-define (struct* twin-trial ([dt dt] (case c) [setting s])) r)
    (match-define (struct* oracle ([counts counts] [solver-gap solver-gap])) o)
    (define centred? (centre-coef? (setting-penalty s)))
    (define (tight-fit fitted-case)
      (setting-fit fitted-case dt s #:tol (tight-tol dt) #:max-iter tight-max-iter))
    (define (gap-to-sklearn parts)
      (relative-gap (gauged-fit parts #:centre-coef? centred?)
                    (gauged-sklearn theirs #:centre-coef? centred?)))
    (define (gap-of-fit fitted-case)
      (gap-to-sklearn (fit-parts (tight-fit fitted-case))))
    (test-case (format "~a ~a ~a" dt (lr-case-name c) (setting-penalty s))
      (define fitted (tight-fit c))
      (define gap (gap-to-sklearn (fit-parts fitted)))
      (define unweighted-gap (solver-gap gap-of-fit))
      (check-true (<= gap (+ unweighted-gap (optimum-gap-bound dt)))
                  (format "coef gap ~a, the unweighted fit's ~a" gap unweighted-gap))
      (define default-gap (gap-to-sklearn (fit-parts (setting-fit c dt s))))
      (check-true (<= default-gap (default-tol-cap dt)) (format "default-tol gap ~a" default-gap))
      (define z
        (kept-rows (device-matrix->list*
                    (logistic-regression-decision-function fitted (on-matrix (lr-case-X c) dt)))
                   counts))
      (for ([zi (in-list z)]
            [label (in-list (collapsed (hash-ref theirs 'predict) counts))]
            #:unless (< (margin zi) boundary-margin))
        (check-true (= (decision-label zi) label) "predict")))))

(define weighted-cases
  (list (blob-case "three classes weighted" 120 3 3 34 #:weighted? #t)
        (blob-case "five classes weighted" 150 3 5 35 #:weighted? #t)))

(define penalised-settings (filter-not (lambda (s) (eq? (setting-penalty s) 'none)) penalty-settings))

(define (check-weighted-sklearn-twins dt)
  (check-sklearn-twins (sklearn-trials weighted-cases penalised-settings #:dtypes (list dt))
                       #:repeated? #t))
