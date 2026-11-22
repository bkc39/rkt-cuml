#lang racket/base

(require (only-in racket/list append-map make-list)
         (only-in racket/match match-define struct*)
         (only-in rackunit check-equal? test-case)
         (only-in raft device-vector->list)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in cuml/svm svc-decision-function svc-result-gamma)
         (only-in "private/linear-data.rkt" check-law-case dtypes law-case on-matrix)
         (only-in "private/svm-data.rkt"
                  case-fit
                  dyadic-blobs
                  fit-parts
                  fresh-rows
                  svm-case
                  svm-case-name
                  svm-fit
                  weighted-cases)
         (only-in "private/svm-oracle.rkt" check-kkt))

(define-values (rows labels) (dyadic-blobs 300 4 1))
(define probe (fresh-rows 200 4 1))

(define (blob-case name dt kernel #:X [X rows] #:y [y labels] #:w [w #f] #:gamma [gamma 'scale])
  (svm-case name X y w 1 kernel dt gamma))

(define (base-gamma dt)
  (svc-result-gamma (case-fit (blob-case "base" dt 'rbf))))

(define (decisions c)
  (match-define (struct* svm-case ([dtype dt])) c)
  (device-vector->list (svc-decision-function (case-fit c) (on-matrix probe dt))))

(define (largest-gap ds es)
  (for/fold ([m 0.0]) ([d (in-list ds)] [e (in-list es)])
    (max m (abs (- d e)))))

(define arms
  (for*/list ([dt (in-list dtypes)]
              [kernel (in-list '(linear poly rbf))])
    (list dt kernel)))

(test-gpu "W1: the KKT law holds with C_i = C·w_i"
  (for ([c (in-list (weighted-cases))])
    (test-case (svm-case-name c)
      (check-kkt c))))

(define dropped
  (for/list ([i (in-range 300)])
    (zero? (modulo i 5))))

(define (kept xs)
  (for/list ([x (in-list xs)] [drop? (in-list dropped)] #:unless drop?) x))

(define kept-index
  (for/vector ([i (in-range 300)] [drop? (in-list dropped)] #:unless drop?) i))

(test-gpu "W2: zero-weight rows fit as deleted rows"
  (define w (for/list ([drop? (in-list dropped)]) (if drop? 0 1)))
  (for ([arm (in-list arms)])
    (match-define (list dt kernel) arm)
    (test-case (format "~a ~a" dt kernel)
      (define g (base-gamma dt))
      (define zeroed (blob-case "zeroed" dt kernel #:w w #:gamma g))
      (define deleted
        (blob-case "deleted" dt kernel #:X (kept rows) #:y (kept labels) #:gamma g))
      (match-define (struct* svm-fit ([support zs])) (fit-parts (case-fit zeroed)))
      (match-define (struct* svm-fit ([support ds])) (fit-parts (case-fit deleted)))
      (law-case "W2 support"
        (check-equal? zs (for/list ([i (in-list ds)]) (vector-ref kept-index i))))
      (check-law-case "W2" (largest-gap (decisions zeroed) (decisions deleted)) 0.004))))

(test-gpu "W3: integer weights fit as repeated rows"
  (define counts (for/list ([i (in-range 300)]) (add1 (modulo (* 7 i) 3))))
  (define (repeated xs)
    (append-map (lambda (x k) (make-list k x)) xs counts))
  (for ([arm (in-list arms)])
    (match-define (list dt kernel) arm)
    (test-case (format "~a ~a" dt kernel)
      (define g (base-gamma dt))
      (define ds (decisions (blob-case "weighted" dt kernel #:w counts #:gamma g)))
      (define es
        (decisions
         (blob-case "repeated" dt kernel #:X (repeated rows) #:y (repeated labels) #:gamma g)))
      (check-law-case "W3" (largest-gap ds es) 0.005)
      (law-case "W3 signs"
        (for ([d (in-list ds)] [e (in-list es)] #:when (> (abs d) 0.005))
          (check-equal? (>= d 0) (>= e 0)))))))

(test-gpu "W4: all weights 1 is bit-identical to no weights"
  (for ([arm (in-list arms)])
    (match-define (list dt kernel) arm)
    (test-case (format "~a ~a" dt kernel)
      (law-case "W4"
        (check-equal? (fit-parts (case-fit (blob-case "ones" dt kernel #:w (make-list 300 1))))
                      (fit-parts (case-fit (blob-case "none" dt kernel))))))))
