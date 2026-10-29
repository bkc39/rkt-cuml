#lang racket/base

(require (only-in cuml/metrics
                  adjusted-rand-score
                  contingency-matrix
                  contingency-table-counts
                  silhouette-samples
                  silhouette-score
                  trustworthiness)
         (only-in cuml/metrics/private/native libcuml-version)
         (only-in racket/match match match-define)
         racket/runtime-path
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in threading ~>)
         (only-in "private/metrics-data.rkt"
                  label-pair
                  labels-of
                  overlap
                  overlap-labels
                  projection
                  tolerance
                  worst)
         (only-in "private/python-env.rkt" run-twin test-twin))

(define-runtime-path twin "python/metrics_twin.py")

(define metrics '(euclidean sqeuclidean manhattan cosine chebyshev))
(define each 150)

(define (twin-results cases)
  (match-define (hash-table ('results results) ('version version))
    (run-twin twin (hasheq 'cases cases)))
  (check-equal? version libcuml-version)
  results)

(define degenerate
  '((() ()) ((4) (9))
            ((3 3 3 3) (1 1 1 1))
            ((0 1 2 3) (0 1 2 3))
            ((5 5 5 5 5) (0 1 2 3 4))
            ((0 1 2 3 4) (5 5 5 5 5))
            ((0 0 1 1) (1 1 0 0))
            ((0 0 1 1) (0 1 0 1))
            ((0 0 0 1 1 1) (0 1 2 3 4 5))
            ((-1 -1 0 0 1) (2 2 2 0 0))
            ((7 7 7 7 7 7 7 8) (7 7 7 7 7 7 7 7))))

(define (label-cases count)
  (for/list ([seed (in-range count)])
    (define-values (truth found)
      (label-pair (+ 2 (* 7 seed)) (+ 1 (modulo seed 9)) (+ 1 (modulo (* 3 seed) 11)) (+ 1000 seed)))
    (list truth found)))

(define (counts truth found)
  (~> (contingency-matrix (labels-of truth 'int32) (labels-of found 'int64))
      contingency-table-counts
      device-matrix->list*))

(test-twin "CC8: contingency-matrix's counts equal sklearn's contingency_matrix"
  (define cases
    (append (label-cases 40) (list '((-1 -1 0 2 2 2) (1 0 0 1 3 3)) '((5 -1 5 -1) (-2 -2 4 4)))))
  (define results
    (twin-results (for/list ([c (in-list cases)])
                    (match-define (list truth found) c)
                    (hasheq 'op "contingency" 'truth truth 'found found))))
  (for ([c (in-list cases)]
        [theirs (in-list results)]
        [i (in-naturals)])
    (match-define (list truth found) c)
    (test-case (format "case ~a" i)
      (check-equal? (hash-ref theirs 'dtype) "int64")
      (check-equal? (counts truth found) (hash-ref theirs 'counts)))))

(test-twin "CA5, CA6: adjusted-rand-score is within 1e-14 of sklearn and equals Python cuML"
  (define cases (append (label-cases 200) degenerate))
  (define results
    (twin-results (for/list ([c (in-list cases)])
                    (match-define (list truth found) c)
                    (hasheq 'op "ari"
                            'truth truth
                            'found found
                            'cuml (and (>= (length truth) 2)
                                       (andmap (lambda (x) (>= x 0)) (append truth found)))))))
  (define gaps
    (for/list ([c (in-list cases)]
               [theirs (in-list results)]
               [i (in-naturals)])
      (match-define (list truth found) c)
      (define ours (adjusted-rand-score (labels-of truth 'int32) (labels-of found 'int32)))
      (define gap (abs (- ours (hash-ref theirs 'sklearn))))
      (test-case (format "case ~a" i)
        (check-true (<= gap 1e-14) (format "~a against sklearn's ~a" ours (hash-ref theirs 'sklearn)))
        (match (hash-ref theirs 'cuml)
          [(? real? cu) (check-equal? ours cu "CA6")]
          [_ (void)]))
      gap))
  (printf "twin ari: ~a cases, worst difference from sklearn ~a\n" (length gaps) (apply max gaps)))

(define (silhouette-case X ys metric dt)
  (hasheq 'op "silhouette"
          'X (device-matrix->list* X)
          'labels ys
          'metric (symbol->string metric)
          'dtype (symbol->string dt)
          'cuml #t))

(define (ours X ys metric)
  (list (silhouette-score X (labels-of ys 'int32) #:metric metric)
        (device-vector->list (silhouette-samples X (labels-of ys 'int32) #:metric metric))))

(define (noisy-labels)
  (for/list ([y (in-list (overlap-labels #:each each))]
             [i (in-naturals)])
    (if (zero? (modulo i 7)) -1 y)))

(test-twin "SL6-SL8: silhouette agrees with sklearn on the float64 cast and with Python cuML"
  (define runs
    (append (for*/list ([dt (in-list '(float32 float64))]
                        [metric (in-list metrics)])
              (list "SL6" dt metric (overlap dt #:each each) (overlap-labels #:each each)))
            (for/list ([dt (in-list '(float32 float64))])
              (list "SL7" dt 'euclidean (overlap dt #:each each) (noisy-labels)))))
  (define results
    (twin-results (for/list ([r (in-list runs)])
                    (match-define (list _ dt metric X ys) r)
                    (silhouette-case X ys metric dt))))
  (for ([r (in-list runs)]
        [theirs (in-list results)])
    (match-define (list law dt metric X ys) r)
    (match-define (list score samples) (ours X ys metric))
    (match-define (hash-table ('score sk) ('samples sks) ('cuml_score cu) ('cuml_samples cus)) theirs)
    (define cuml-tolerance (tolerance 'cuml dt))
    (test-case (format "~a ~a ~a" law dt metric)
      (check-true (<= (abs (- score sk)) (tolerance 'sklearn-score dt))
                  (format "score ~a against sklearn's ~a" score sk))
      (check-true (<= (worst samples sks) (tolerance 'sklearn-sample dt))
                  (format "samples ~a" (worst samples sks)))
      (check-true (<= (abs (- score cu)) cuml-tolerance)
                  (format "SL8 score ~a against cuML's ~a" score cu))
      (check-true (<= (worst samples cus) (tolerance 'sample dt))
                  (format "SL8 samples ~a" (worst samples cus))))))

(test-twin "TW5, TW6: trustworthiness is within 1e-6 of sklearn and equals Python cuML"
  (define runs
    (for*/list ([n (in-list '(100 300 600 1000 2000))]
                [k (in-list '(5 12))])
      (define-values (X E) (projection n 10 2 #:seed n))
      (list n k X E)))
  (define results
    (twin-results (for/list ([r (in-list runs)])
                    (match-define (list _ k X E) r)
                    (hasheq 'op "trustworthiness"
                            'X (device-matrix->list* X)
                            'E (device-matrix->list* E)
                            'n_neighbors k))))
  (check-equal? (twin-results (list (hasheq 'op "version"))) (list libcuml-version))
  (for ([r (in-list runs)]
        [theirs (in-list results)])
    (match-define (list n k X E) r)
    (define score (trustworthiness X E #:n-neighbors k))
    (test-case (format "n ~a k ~a" n k)
      (check-true (<= (abs (- score (hash-ref theirs 'sklearn))) 1e-6)
                  (format "~a against sklearn's ~a" score (hash-ref theirs 'sklearn)))
      (check-equal? score (hash-ref theirs 'cuml) "TW6"))))
