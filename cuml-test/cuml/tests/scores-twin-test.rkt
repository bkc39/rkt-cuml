#lang racket/base

(require (only-in cuml/metrics accuracy-score mean-squared-error r2-score)
         (only-in cuml/metrics/private/native libcuml-version)
         (only-in json json-null)
         (only-in racket/list take)
         (only-in racket/match match match-define)
         (only-in racket/math exact-floor nan?)
         racket/runtime-path
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft list*->device-matrix list->device-vector)
         (only-in "private/blob-data.rkt" uniform-source)
         (only-in "private/metrics-data.rkt" regression-cases regression-pair)
         (only-in "private/python-env.rkt" run-twin test-twin)
         (only-in "private/scaler-data.rkt" rows-of))

(define-runtime-path twin "python/metrics_twin.py")

(define (twin-results cases)
  (match-define (hash-table ('results results) ('version version))
    (run-twin twin (hasheq 'cases cases)))
  (check-equal? version libcuml-version "T5")
  results)

(define (on xs dt)
  (if (list? (car xs))
      (list*->device-matrix xs #:dtype dt)
      (list->device-vector xs #:dtype dt)))

(define (weights-of w)
  (and w (list->device-vector w #:dtype 'float64)))

(define (case-of op y p w dt [pt dt])
  (hasheq 'op op
          'y_true y
          'y_pred p
          'dtype_true (symbol->string dt)
          'dtype_pred (symbol->string pt)
          'weight (or w (json-null))))

(define (vector-cases)
  (for/list ([c (in-list (regression-cases))])
    (match-define (list i dt y p w) c)
    (list (format "case ~a ~a" i dt) dt y p w)))

(define (matrix-cases)
  (for/list ([i (in-range 10)])
    (define dt (if (even? i) 'float32 'float64))
    (define d (+ 2 (modulo i 4)))
    (define n 200)
    (define-values (y p w)
      (regression-pair (* n d)
                       (* 100.0 i)
                       0.5
                       (+ 9600 i)
                       #:dtype dt
                       #:weighted? (odd? (quotient i 2))))
    (list (format "matrix ~a ~a ~ax~a" i dt n d) dt (rows-of y d) (rows-of p d) (and w (take w n)))))

(define (relative ours theirs)
  (/ (abs (- ours theirs)) (abs theirs)))

(define (r2-relative ours theirs)
  (/ (abs (- ours theirs)) (max 1.0 (abs (- 1.0 theirs)))))

(test-twin "T1, T2: r2 and the error agree with sklearn on the float64 cast and Python cuML on the dtype"
  (define vectors (vector-cases))
  (define matrices (matrix-cases))
  (define r2-results
    (twin-results (for/list ([c (in-list vectors)])
                    (match-define (list _ dt y p w) c)
                    (case-of "r2" y p w dt))))
  (define mse-results
    (twin-results (for/list ([c (in-list (append vectors matrices))])
                    (match-define (list _ dt y p w) c)
                    (case-of "mse" y p w dt))))
  (define (check-against score cases results gap cuml-bound)
    (for/fold ([worst (hasheq)])
              ([c (in-list cases)]
               [theirs (in-list results)])
      (match-define (list name dt y p w) c)
      (match-define (hash-table ('sklearn sk) ('cuml cu)) theirs)
      (define ours (score (on y dt) (on p dt) #:sample-weight (weights-of w)))
      (define sk-gap (gap ours sk))
      (define cu-gap (gap ours cu))
      (test-case name
        (check-true (<= sk-gap 1e-14) (format "T1: ~a against sklearn's ~a" ours sk))
        (check-true (<= cu-gap (if (eq? dt 'float32) cuml-bound 1e-14))
                    (format "T2: ~a against Python cuML's ~a" ours cu)))
      (hash-update (hash-update worst (cons 'sklearn dt) (lambda (g) (max g sk-gap)) 0.0)
                   (cons 'cuml dt)
                   (lambda (g) (max g cu-gap))
                   0.0)))
  (printf "twin r2: worst gaps ~a\n" (check-against r2-score vectors r2-results r2-relative 1e-6))
  (printf "twin error: worst gaps ~a\n"
          (check-against mean-squared-error (append vectors matrices) mse-results relative 3e-7)))

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
