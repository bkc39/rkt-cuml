#lang racket/base

(require (only-in cuml/linear-model mbsgd-classifier-fit mbsgd-regressor-fit)
         (only-in rackunit check-not-exn test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/elastic-net-data.rkt" apply-keywords)
         (only-in "private/linear-data.rkt" on-matrix on-vector)
         (only-in "private/mbsgd-data.rkt" normal-rows reg-target)
         (only-in "private/refusal-rows.rkt" check-row row))

(define count-message "expected an exact integer from 1 to 2147483647, got ~a")

(define edges
  `((alpha "expected a positive finite real, got ~a" (0 -1 +nan.0 +inf.0) (1e-300))
    (l1-ratio "expected a real in [0, 1], got ~a" (-0.5 2) (0 1))
    (epochs ,count-message (0 -1 ,(expt 2 31)) (1 2147483647))
    (batch-size ,count-message (0 -1 ,(expt 2 31)) (1 2147483647))
    (n-iter-no-change ,count-message (0 -1 ,(expt 2 31)) (1 2147483647))
    (tol "expected a nonnegative finite real, got ~a" (-1 +nan.0 +inf.0) (0 1e300))
    (eta0 "expected a positive finite real, got ~a" (0 -0.01 +nan.0) (1e-300))
    (power-t "expected a nonnegative finite real, got ~a" (-1 +nan.0) (0 2))
    (random-state "expected #f or an integer in [0, 4294967295], got ~a"
                  (-1 ,(expt 2 32) 1.5)
                  (0 4294967295))))

(define rows (normal-rows 64 3 51))
(define targets (reg-target rows 51))

(define (quick key value)
  (define k (string->keyword (symbol->string key)))
  (cond
    [(eq? key 'epochs) (list k value '#:tol 1e-3)]
    [else (list k value '#:epochs (if (eq? key 'n-iter-no-change) 20 5))]))

(define (quick-fit f key v X y)
  (apply-keywords f (apply hash (quick key v)) (list X y)))

(test-gpu "each setting refuses its range edges and accepts the values just inside"
  (define X (on-matrix rows))
  (define y (on-vector targets))
  (define-values (Xc yc) (values X (map (lambda (t) (if (> t 3) 'up 'down)) targets)))
  (for ([edge (in-list edges)])
    (define key (car edge))
    (define message (cadr edge))
    (for ([v (in-list (caddr edge))])
      (test-case (format "~a ~a" key v)
        (define text (format "~a: ~a" key (format message (format "~e" v))))
        (check-row "mbsgd-regressor-fit"
                   (row "refused" text (lambda () (quick-fit mbsgd-regressor-fit key v X y))))
        (check-row "mbsgd-classifier-fit"
                   (row "refused" text (lambda () (quick-fit mbsgd-classifier-fit key v Xc yc))))))
    (for ([v (in-list (cadddr edge))])
      (test-case (format "~a ~a accepted" key v)
        (check-not-exn (lambda () (quick-fit mbsgd-regressor-fit key v X y)))
        (check-not-exn (lambda () (quick-fit mbsgd-classifier-fit key v Xc yc)))))))
