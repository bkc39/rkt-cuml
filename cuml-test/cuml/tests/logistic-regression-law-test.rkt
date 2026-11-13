#lang racket/base

(require (only-in cuml/linear-model
                  logistic-regression-fit
                  logistic-regression-result-coef
                  logistic-regression-result-intercept)
         (only-in racket/list remove-duplicates)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/linear-data.rkt" dtypes on-matrix on-vector unit-roundoff)
         (only-in "private/logistic-data.rkt"
                  blob-case
                  case-fit
                  case-y
                  fit-parts
                  lr-case
                  lr-case-X
                  lr-case-name
                  lr-case-w
                  lr-case-y
                  with-column))

(define binary (blob-case "binary" 80 3 2 11 #:weighted? #t))
(define three (blob-case "three classes" 90 3 3 12))

(define (scaled-parts parts factor)
  (list (for/list ([row (in-list (car parts))])
          (map (lambda (v) (* v factor)) row))
        (cadr parts)
        (caddr parts)))

(test-gpu "a constant or zero column gets coef 0 and leaves the rest bitwise (E4)"
  (for* ([c (in-list (list binary three))]
         [dt (in-list dtypes)]
         [setting (in-list '((3.0 #t) (0.0 #f)))])
    (test-case (format "~a ~a ~a" (lr-case-name c) dt setting)
      (define fi (cadr setting))
      (define base (fit-parts (case-fit c dt #:fit-intercept fi)))
      (define wide (fit-parts (case-fit (with-column c (car setting)) dt #:fit-intercept fi)))
      (check-true (andmap (lambda (row) (zero? (car row))) (car wide)))
      (check-equal? (cons (map cdr (car wide)) (cdr wide)) base))))

(test-gpu "rows of weight 0 fit as deleted rows, bitwise (E5)"
  (define zeroed (for/list ([w (in-list (lr-case-w binary))] [i (in-naturals)]) (if (zero? (modulo i 3)) 0 w)))
  (define kept (for/list ([w (in-list zeroed)]) (positive? w)))
  (define (keep xs) (for/list ([x (in-list xs)] [k (in-list kept)] #:when k) x))
  (for ([dt (in-list dtypes)])
    (define with-zeros (lr-case "zeros" (lr-case-X binary) (lr-case-y binary) zeroed))
    (define deleted (lr-case "deleted" (keep (lr-case-X binary)) (keep (lr-case-y binary)) (keep zeroed)))
    (check-equal? (fit-parts (case-fit with-zeros dt)) (fit-parts (case-fit deleted dt)))))

(test-gpu "weights times 2^a fit as C times 2^a, bitwise, and leave 'none unchanged (E6)"
  (for* ([dt (in-list dtypes)]
         [a (in-list '(-20 3 40))])
    (test-case (format "~a ~a" dt a)
      (define factor (expt 2 a))
      (define scaled (lr-case "scaled" (lr-case-X binary) (lr-case-y binary)
                              (map (lambda (w) (* w factor)) (lr-case-w binary))))
      (check-equal? (fit-parts (case-fit scaled dt)) (fit-parts (case-fit binary dt #:C (exact->inexact factor))))
      (check-equal? (fit-parts (case-fit scaled dt #:penalty 'none)) (fit-parts (case-fit binary dt #:penalty 'none)))
      (define uniform (lr-case "uniform" (lr-case-X three) (lr-case-y three)
                               (map (lambda (_) factor) (lr-case-y three))))
      (check-equal? (fit-parts (case-fit uniform dt)) (fit-parts (case-fit three dt #:C (exact->inexact factor)))))))

(define (scaled-case c factor)
  (lr-case (lr-case-name c)
           (for/list ([row (in-list (lr-case-X c))])
             (map (lambda (v) (* v factor)) row))
           (lr-case-y c)
           (lr-case-w c)))

(test-gpu "X times 2^a scales coef by 2^-a bitwise when both fits are staged (E7)"
  (for* ([dt (in-list dtypes)]
         [a (in-list '(12 -12 20 -20 40 -40))]
         [penalty (in-list '(l2 l1 none))])
    (test-case (format "~a ~a ~a" dt a penalty)
      (define base (scaled-case binary (expt 2 (if (positive? a) 10 -10))))
      (define C (case penalty [(l2) (expt 4 (- a))] [(l1) (expt 2 (- a))] [else 1]))
      (define plain (fit-parts (case-fit base dt #:penalty penalty)))
      (define moved (fit-parts (case-fit (scaled-case base (expt 2 a)) dt #:penalty penalty #:C (exact->inexact C))))
      (check-equal? moved (scaled-parts plain (expt 2.0 (- a)))))))

(test-gpu "balanced class weights equal the explicit mapping W/(k W_c), bitwise (E8)"
  (define ys (lr-case-y binary))
  (define classes (sort (remove-duplicates ys) <))
  (define n (length ys))
  (define mapping
    (for/hash ([c (in-list classes)])
      (values c (exact->inexact (/ n (* (length classes) (for/sum ([y (in-list ys)]) (if (= y c) 1 0))))))))
  (define unweighted (lr-case "unweighted" (lr-case-X binary) ys #f))
  (for ([dt (in-list dtypes)])
    (check-equal? (fit-parts (case-fit unweighted dt #:class-weight 'balanced))
                  (fit-parts (case-fit unweighted dt #:class-weight mapping)))))

(test-gpu "no kept columns give the closed-form intercept within 4u (E16)"
  (define ys '(0 0 1 1 1 2 2 2 2 2))
  (for* ([dt (in-list dtypes)]
         [labels (in-list (list (map (lambda (y) (min y 1)) ys) ys))])
    (define X (map (lambda (_) '(2.5 -1.0)) labels))
    (define r (logistic-regression-fit (on-matrix X dt) (on-vector labels 'int32)))
    (define counts
      (for/list ([c (in-list (sort (remove-duplicates labels) <))])
        (for/sum ([y (in-list labels)]) (if (= y c) 1 0))))
    (define expected
      (if (= 2 (length counts))
          (list (log (/ (cadr counts) (car counts))))
          (let ([logs (map log counts)])
            (map (lambda (l) (- l (/ (apply + logs) (length logs)))) logs))))
    (define u (unit-roundoff dt))
    (check-true (andmap (lambda (row) (andmap zero? row))
                        (device-matrix->list* (logistic-regression-result-coef r))))
    (for ([b (in-list (device-vector->list (logistic-regression-result-intercept r)))]
          [e (in-list expected)])
      (check-true (<= (abs (- b e)) (* 4 u (max 1 (abs e))))))))

(test-gpu "the fit leaves X, y and the weights as they were (E17)"
  (define zeroed (for/list ([w (in-list (lr-case-w binary))] [i (in-naturals)]) (if (zero? (modulo i 4)) 0 w)))
  (define staged (scaled-case binary (expt 2 20)))
  (for ([dt (in-list dtypes)])
    (define X (on-matrix (lr-case-X staged) dt))
    (define y (case-y binary))
    (define w (on-vector zeroed dt))
    (define before (list (device-matrix->list* X) (device-vector->list y) (device-vector->list w)))
    (logistic-regression-fit X y #:sample-weight w)
    (check-equal? (list (device-matrix->list* X) (device-vector->list y) (device-vector->list w)) before)))
