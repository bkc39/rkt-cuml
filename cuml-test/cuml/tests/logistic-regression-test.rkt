#lang racket/base

(require (only-in cuml
                  decision-function
                  fit
                  logistic-regression
                  logistic-regression-model
                  logistic-regression?
                  predict
                  predict-log-proba
                  predict-proba
                  score)
         (only-in cuml/linear-model
                  logistic-regression-decision-function
                  logistic-regression-fit
                  logistic-regression-predict
                  logistic-regression-predict-log-proba
                  logistic-regression-predict-proba
                  logistic-regression-result-classes
                  logistic-regression-result-coef
                  logistic-regression-result-intercept
                  logistic-regression-result-n-iter)
         (only-in racket/list make-list remove-duplicates)
         (only-in racket/match match-define match-lambda**)
         (only-in rackunit check-equal? check-pred check-true test-case)
         (only-in raft
                  device-matrix->list*
                  device-vector->list
                  device-vector?
                  dtype
                  list->device-vector
                  shape)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/common-data.rkt" on-vector unit-roundoff)
         (only-in "private/linear-data.rkt" dtypes on-matrix)
         (only-in "private/logistic-data.rkt"
                  argmax-index
                  blob-case
                  case-fit
                  correct
                  decision-label
                  fit-parts
                  lr-case-X
                  lr-case-name
                  lr-case-y))

(define binary (blob-case "binary" 120 3 2 1))
(define three (blob-case "three classes" 120 3 3 2))
(define five (blob-case "five classes" 150 4 5 3))

(define (named names ys)
  (map (lambda (v) (vector-ref names v)) ys))

(define (relabelled v)
  (+ (* 3 v) 7))

(test-gpu "the fit's shapes follow the classes, and no intercept is zeros"
  (for* ([c (in-list (list binary three five))]
         [dt (in-list dtypes)]
         [fi (in-list '(#t #f))])
    (test-case (format "~a ~a ~a" (lr-case-name c) dt fi)
      (define r (case-fit c dt #:fit-intercept fi #:max-iter 200))
      (define k (length (remove-duplicates (lr-case-y c))))
      (define k* (if (= k 2) 1 k))
      (match-define (cons row _) (lr-case-X c))
      (check-equal? (shape (logistic-regression-result-coef r)) (list k* (length row)))
      (check-equal? (length (device-vector->list (logistic-regression-result-intercept r))) k*)
      (unless fi
        (check-true (andmap zero? (device-vector->list (logistic-regression-result-intercept r)))))
      (check-true (<= 0 (logistic-regression-result-n-iter r) 200)))))

(test-gpu "classes are the sorted labels and predict gives y's kind"
  (define X (on-matrix (lr-case-X three)))
  (define strings (named #("c" "a" "b") (lr-case-y three)))
  (define r (logistic-regression-fit X strings))
  (check-equal? (logistic-regression-result-classes r) '("a" "b" "c"))
  (define predicted (logistic-regression-predict r X))
  (check-pred list? predicted)
  (check-equal? (remove* '("a" "b" "c") predicted) '())
  (define d (logistic-regression-fit X (on-vector (lr-case-y three) 'int64)))
  (check-equal? (logistic-regression-result-classes d) '(0 1 2))
  (define on-device (logistic-regression-predict d X))
  (check-pred device-vector? on-device)
  (check-equal? (dtype on-device) 'int64))

(test-gpu "an increasing relabelling gives the same fit bit for bit"
  (define X (on-matrix (lr-case-X five) 'float32))
  (define y (lr-case-y five))
  (define base (logistic-regression-fit X (list->device-vector y #:dtype 'int32)))
  (define mapped (logistic-regression-fit X (map relabelled y)))
  (define as-list (logistic-regression-fit X y))
  (define strings (map (lambda (v) (format "~a" v)) y))
  (define as-strings (logistic-regression-fit X strings))
  (define as-symbols (logistic-regression-fit X (map string->symbol strings)))
  (for ([other (in-list (list mapped as-list as-strings as-symbols))])
    (check-equal? (fit-parts other) (fit-parts base)))
  (check-equal? (logistic-regression-predict mapped X)
                (map relabelled (device-vector->list (logistic-regression-predict base X)))))

(test-gpu "predict is the decision's argmax, and predict-proba's, the first on ties"
  (for* ([c (in-list (list binary five))]
         [dt (in-list dtypes)])
    (define X (on-matrix (lr-case-X c) dt))
    (define r (case-fit c dt))
    (define predicted (device-vector->list (logistic-regression-predict r X)))
    (define decision (device-matrix->list* (logistic-regression-decision-function r X)))
    (define proba (device-matrix->list* (logistic-regression-predict-proba r X)))
    (check-equal? predicted (map decision-label decision))
    (check-equal? predicted (map argmax-index proba))))

(test-gpu "probabilities sum to 1, and log-proba is their logarithm, to rounding"
  (for* ([c (in-list (list binary five))]
         [dt (in-list dtypes)])
    (define X (on-matrix (lr-case-X c) dt))
    (define r (case-fit c dt))
    (define u (unit-roundoff dt))
    (define proba (device-matrix->list* (logistic-regression-predict-proba r X)))
    (define logs (device-matrix->list* (logistic-regression-predict-log-proba r X)))
    (define k (length (car proba)))
    (for ([p (in-list proba)]
          [l (in-list logs)])
      (check-true (<= (abs (- (apply + (map inexact->exact p)) 1))
                      (* (if (= k 2)
                             1
                             (add1 k))
                         u)))
      (for ([pc (in-list p)]
            [lc (in-list l)])
        (check-true (rational? lc))
        (check-true (<= (abs (- (exp lc) pc))
                        (* 4
                           u
                           (+ 1 (abs lc))
                           (if (= k 2)
                               (max pc (- 1 pc))
                               pc))))))))

(define (non-decreasing-by-score? pairs)
  (define sorted (sort pairs < #:key car))
  (for/and ([a (in-list sorted)]
            [b (in-list (cdr sorted))])
    (<= (cdr a) (cdr b))))

(test-gpu "probabilities rise with the score: binary across rows, multiclass within a row"
  (for* ([c (in-list (list binary five))]
         [dt (in-list dtypes)])
    (define X (on-matrix (lr-case-X c) dt))
    (define r (case-fit c dt))
    (define decision (device-matrix->list* (logistic-regression-decision-function r X)))
    (define proba (device-matrix->list* (logistic-regression-predict-proba r X)))
    (if (= 1 (length (car decision)))
        (check-true (non-decreasing-by-score? (map (match-lambda**
                                                     [((list z) (list _ p)) (cons z p)])
                                                   decision
                                                   proba)))
        (for ([z (in-list decision)]
              [p (in-list proba)])
          (check-true (non-decreasing-by-score? (map cons z p)))))))

(test-gpu "the estimator fits, predicts, scores and gives probabilities"
  (define X (on-matrix (lr-case-X three)))
  (define strings (named #("x" "y" "z") (lr-case-y three)))
  (define e (fit (logistic-regression #:C 2.0) X strings))
  (check-pred logistic-regression? e)
  (define r (logistic-regression-model e))
  (define predicted (predict e X))
  (check-equal? predicted (logistic-regression-predict r X))
  (check-true (< (abs (- (score e X strings) (/ (correct predicted strings) (length strings)))) 1e-6))
  (check-equal? (device-matrix->list* (predict-proba e X))
                (device-matrix->list* (logistic-regression-predict-proba r X)))
  (check-equal? (device-matrix->list* (predict-log-proba e X))
                (device-matrix->list* (logistic-regression-predict-log-proba r X)))
  (check-equal? (device-matrix->list* (decision-function e X))
                (device-matrix->list* (logistic-regression-decision-function r X)))
  (check-true (< (score e X (make-list (length strings) "w")) 1e-12)))
