#lang racket/base

(require (only-in cuml/metrics silhouette-samples silhouette-score)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list* device-vector->list dtype list*->device-matrix)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/metrics-data.rkt"
                  bijection
                  labels-of
                  mean
                  monotone-maps
                  overlap
                  overlap-labels
                  three-groups
                  three-groups-labels
                  tolerance
                  worst))

(define metrics '(euclidean sqeuclidean manhattan cosine chebyshev))
(define float-dtypes '(float32 float64))
(define each 150)
(define n (* 4 each))

(define (fixture dt)
  (values (overlap dt #:each each) (overlap-labels #:each each)))

(define (samples X ys #:dtype [ldt 'int32] #:metric [metric 'euclidean] #:chunk-size [chunk 40000])
  (device-vector->list (silhouette-samples X (labels-of ys ldt) #:metric metric #:chunk-size chunk)))

(define (score X ys #:dtype [ldt 'int32] #:metric [metric 'euclidean] #:chunk-size [chunk 40000])
  (silhouette-score X (labels-of ys ldt) #:metric metric #:chunk-size chunk))

(define (check-close what got expected bound)
  (check-true (<= (abs (- got expected)) bound)
              (format "~a: ~a against ~a, past ~a" what got expected bound)))

(test-gpu "SL1: samples are n values in X's dtype within [-1, 1], and the score is their mean"
  (for* ([dt (in-list float-dtypes)]
         [metric (in-list metrics)])
    (test-case (format "~a ~a" dt metric)
      (define-values (X ys) (fixture dt))
      (define v (silhouette-samples X (labels-of ys 'int32) #:metric metric))
      (define s (device-vector->list v))
      (check-equal? (dtype v) dt)
      (check-equal? (length s) n)
      (check-true (for/and ([x (in-list s)])
                    (<= -1.0 x 1.0)))
      (check-close "score against the mean"
                   (score X ys #:metric metric)
                   (mean s)
                   (tolerance 'mean dt)))))

(define (scaled rows factor dt)
  (list*->device-matrix (for/list ([r (in-list rows)])
                          (for/list ([x (in-list r)])
                            (* factor x)))
                        #:dtype dt))

(define (smallest-squared-norm rows)
  (for/fold ([m +inf.0]) ([r (in-list rows)])
    (min m (for/sum ([x (in-list r)]) (* x x)))))

(test-gpu "S13: under cosine, a smallest row just past the least normal squared norm is accepted"
  (for ([dt (in-list float-dtypes)]
        [least (in-list '(1.1e-19 1.5e-154))]
        [up (in-list (list (expt 2.0 63) (expt 2.0 510)))])
    (define-values (X ys) (fixture dt))
    (define rows (device-matrix->list* X))
    (define small (scaled rows (/ least (sqrt (smallest-squared-norm rows))) dt))
    (define unit (scaled (device-matrix->list* small) up dt))
    (check-true (<= (worst (samples small ys #:metric 'cosine) (samples unit ys #:metric 'cosine))
                    (tolerance 'sample dt)))))

(test-gpu "SL2: a sample alone in its cluster scores exactly 0.0"
  (for ([dt (in-list float-dtypes)])
    (define X (list*->device-matrix (append three-groups '((30 30))) #:dtype dt))
    (define s (samples X (append three-groups-labels '(3))))
    (check-equal? (list-ref s 10) 0.0)))

(test-gpu "SL3: relabelling, monotone or not, keeps the score and the samples"
  (for ([dt (in-list float-dtypes)])
    (define-values (X ys) (fixture dt))
    (define base-score (score X ys))
    (define base (samples X ys))
    (for ([f (in-list (cons (bijection ys) monotone-maps))]
          [i (in-naturals)])
      (test-case (format "~a map ~a" dt i)
        (define mapped (map f ys))
        (check-close "score" (score X mapped #:dtype 'int64) base-score (tolerance 'score dt))
        (check-true (<= (worst (samples X mapped #:dtype 'int64) base) (tolerance 'sample dt)))))))

(define small-each 50)
(define small-n (* 4 small-each))

(test-gpu "SL4: the chunk size changes nothing past the tolerance"
  (for ([dt (in-list float-dtypes)])
    (define X (overlap dt #:each small-each))
    (define ys (overlap-labels #:each small-each))
    (define base-score (score X ys))
    (define base (samples X ys))
    (for ([chunk (in-list (list 1 7 100 (sub1 small-n) small-n 40000))])
      (test-case (format "~a chunk ~a" dt chunk)
        (check-close "score" (score X ys #:chunk-size chunk) base-score (tolerance 'score dt))
        (check-true (<= (worst (samples X ys #:chunk-size chunk) base) (tolerance 'sample dt)))))))

(test-gpu "SL5: ten repeats agree within the measured spread"
  (for ([dt (in-list float-dtypes)])
    (define-values (X ys) (fixture dt))
    (define scores
      (for/list ([_ (in-range 10)])
        (score X ys)))
    (define runs
      (for/list ([_ (in-range 10)])
        (samples X ys)))
    (check-true (<= (- (apply max scores) (apply min scores)) (tolerance 'spread dt)))
    (check-true (for/and ([r (in-list runs)])
                  (<= (worst r (car runs)) (tolerance 'sample dt))))))

(test-gpu "SL9: identical rows score exactly 0.0"
  (for ([dt (in-list float-dtypes)])
    (define X (list*->device-matrix (make-list 20 '(1 1)) #:dtype dt))
    (check-equal? (samples X (append (make-list 10 0) (make-list 10 1))) (make-list 20 0.0))))

(test-gpu "SL10: int32 and int64 labels agree, and X is unchanged"
  (for ([dt (in-list float-dtypes)])
    (define-values (X ys) (fixture dt))
    (define before (device-matrix->list* X))
    (match-define (list s32 s64) (list (samples X ys) (samples X ys #:dtype 'int64)))
    (check-true (<= (worst s32 s64) (tolerance 'sample dt)))
    (check-close "score" (score X ys #:dtype 'int64) (score X ys) (tolerance 'spread dt))
    (check-equal? (device-matrix->list* X) before)))
