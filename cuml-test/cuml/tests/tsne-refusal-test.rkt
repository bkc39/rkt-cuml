#lang racket/base

(require (only-in cuml fit fit-transform transform tsne)
         (only-in cuml/manifold tsne-fit tsne-fit-transform tsne-result?)
         (only-in racket/hash hash-union)
         (only-in racket/list make-list take)
         (only-in racket/match match match-define)
         (only-in rackunit check-equal? check-exn check-pred check-true fail test-case)
         (only-in raft exn:fail:raft? list->device-vector)
         (only-in raft/tests/private/collect drain-finalizers!)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in raft/tests/private/probe probe-pool-used test-pools)
         (only-in "private/cuml-error.rkt" check-cuml-error raised)
         (only-in "private/estimator-data.rkt" apply-options)
         (only-in "private/kmeans-data.rkt" blank-matrix blank-vector)
         (only-in "private/manifold-refusal-data.rkt"
                  cosine-message
                  large-message
                  large-row
                  listed-X
                  nan-message
                  nan-row
                  normless-message
                  tiny-with
                  zero-row)
         (only-in "private/refusal-rows.rkt" check-row row row-call row-id)
         (only-in "private/scaler-data.rkt" rows->matrix)
         (only-in "private/tsne-data.rkt" diverged embedded)
         (only-in "private/umap-data.rkt" finite-rows? fixture-X fixture-rows tiny))

(define small (hasheq 'perplexity 5.0))

(define (tiny-X)
  (fixture-X tiny))

(define (beyond-float name v)
  (format "~a: expected at most 3.4028234663852886e+38, got ~a" name v))

(define (least-rate-message v)
  (format "learning-rate: expected at least 1.1754943508222875e-38, got ~a" v))

(define (parameter-rows)
  (list
   (row "f1" "method: expected fft or exact" (hasheq 'method 'barnes-hut))
   (row "f2"
        "metric: expected euclidean, sqeuclidean, manhattan, chebyshev or cosine"
        (hasheq 'metric 'minkowski))
   (row "f3" "init: expected random or pca" (hasheq 'init 'spectral))
   (row "f1 before f2" "method: expected fft or exact" (hasheq 'method 'bh 'metric 'minkowski))
   (row "f4" "n-components: expected 2, got 3" (hasheq 'n-components 3))
   (row "f5" "n-neighbors: expected at least 2, got 1" (hasheq 'n-neighbors 1))
   (row "f6" "n-neighbors: expected at most 1023, got 1024" (hasheq 'n-neighbors 1024))
   (row "f7" "perplexity: expected a finite value above 1, got 1" (hasheq 'perplexity 1.0))
   (row "f7 nan" "perplexity: expected a finite value above 1, got nan" (hasheq 'perplexity +nan.0))
   (row "f7 inf" "perplexity: expected a finite value above 1, got inf" (hasheq 'perplexity +inf.0))
   (row "f7 float"
        "perplexity: expected a finite value above 1, got 1.000000001"
        (hasheq 'perplexity (+ 1.0 1e-9)))
   (row "f7 most" (beyond-float 'perplexity "1e+39") (hasheq 'perplexity 1e39))
   (row "f8"
        "early-exaggeration: expected a finite value of at least 1, got 0.5"
        (hasheq 'early-exaggeration 0.5))
   (row "f8 nan"
        "early-exaggeration: expected a finite value of at least 1, got nan"
        (hasheq 'early-exaggeration +nan.0))
   (row "f8 most" (beyond-float 'early-exaggeration "1e+39") (hasheq 'early-exaggeration 1e39))
   (row "f9" "learning-rate: expected a finite value above 0, got 0" (hasheq 'learning-rate 0.0))
   (row "f9 nan"
        "learning-rate: expected a finite value above 0, got nan"
        (hasheq 'learning-rate +nan.0))
   (row "f9 most"
        "learning-rate: expected at most 1.7014117331926443e+38, got 2e+38"
        (hasheq 'learning-rate 2e38))
   (row "f9 subnormal" (least-rate-message "1e-40") (hasheq 'learning-rate 1e-40))
   (row "f9 underflow" (least-rate-message "1e-50") (hasheq 'learning-rate 1e-50))
   (row "f10" "max-iter: expected at least 1, got 0" (hasheq 'max-iter 0))
   (row "f10 most"
        "max-iter: expected at most 2147483647, got 2147483648"
        (hasheq 'max-iter (expt 2 31)))
   (row "random-state" "random-state: expected at least 0, got -1" (hasheq 'random-state -1))))

(define every-row-equal "X: every row equals row 0; t-SNE needs two distinct rows")

(define sqeuclidean-bound (- (expt 2 62) (expt 2 38)))

(define (sqeuclidean-message got)
  (format "X: row 4: expected a squared norm of at most ~a under sqeuclidean, got ~a"
          sqeuclidean-bound
          got))

(define (row-at-norm a)
  `([(4 0) 0.0] [(4 1) ,a] [(4 2) 0.0]))

(define (antipodal a)
  `([(4 0) 0.0] [(4 1) ,a] [(4 2) 0.0] [(5 0) 0.0] [(5 1) ,(- a)] [(5 2) 0.0]))

(define (X-rows)
  (define one-row (rows->matrix '((1 2 3)) 'float32))
  (define zeros (rows->matrix (make-list 8 '(0 0 0)) 'float32))
  (define cosine (hash-set small 'metric 'cosine))
  (define sqeuclidean (hash-set small 'metric 'sqeuclidean))
  (list (row "f11" "X: expected float32, got float64" (cons (tiny-with '() 'float64) small))
        (row "f12" "X: expected rank 2, got rank 1" (cons (blank-vector 8 'float32) small))
        (row "f12 layout"
             "X: expected row-major, got col-major 8x3"
             (cons (blank-matrix 8 3 'float32 #:layout 'col-major) small))
        (row "f13" "X: expected at least 2 rows, got 1" (cons one-row small))
        (row "f14"
             "X: expected at least 2 columns, got 1"
             (cons (rows->matrix (map (lambda (r) (take r 1)) (fixture-rows tiny)) 'float32) small))
        (row "f17" "perplexity: expected at most the 8 neighbours, got 30" (cons (tiny-X) (hasheq)))
        (row "f17 neighbours"
             "perplexity: expected at most the 4 neighbours, got 5"
             (cons (tiny-X) (hasheq 'perplexity 5.0 'n-neighbors 4)))
        (row "f19" nan-message (cons (tiny-with nan-row) small))
        (row "f19 inf"
             "X: row 0, column 2: expected a finite value, got -inf"
             (cons (tiny-with '([(0 2) -inf.0])) small))
        (row "f20" large-message (cons (tiny-with large-row) small))
        (row "f20b"
             (sqeuclidean-message (expt 2 62))
             (cons (tiny-with (row-at-norm (expt 2.0 31))) sqeuclidean))
        (row "f20b exact"
             (sqeuclidean-message (expt 2 62))
             (cons (tiny-with (row-at-norm (expt 2.0 31))) (hash-set sqeuclidean 'method 'exact)))
        (row "f21" cosine-message (cons (tiny-with zero-row) cosine))
        (row "f22" every-row-equal (cons (rows->matrix (make-list 8 '(1.5 1.5 1.5)) 'float32) small))
        (row "f22 zeros" every-row-equal (cons zeros small))
        (row "f1 before f11"
             "method: expected fft or exact"
             (cons (tiny-with '() 'float64) (hash-set small 'method 'bh)))
        (row "f5 before f11"
             "n-neighbors: expected at least 2, got 1"
             (cons (tiny-with '() 'float64) (hash-set small 'n-neighbors 1)))
        (row "f13 before f17" "X: expected at least 2 rows, got 1" (cons one-row (hasheq)))
        (row "f19 before f20" nan-message (cons (tiny-with (append nan-row large-row)) small))
        (row "f20 before f20b"
             large-message
             (cons (tiny-with large-row) sqeuclidean))
        (row "f20 before f21" large-message (cons (tiny-with (append zero-row large-row)) cosine))
        (row "f21 before f22" (normless-message 0) (cons zeros cosine))))

(define (fit-calls)
  (list (cons 'tsne-fit (lambda (X options) (apply-options tsne-fit options X)))
        (cons 'tsne-fit-transform (lambda (X options) (apply-options tsne-fit-transform options X)))
        (cons 'fit (lambda (X options) (fit (apply-options tsne options) X)))
        (cons 'fit-transform (lambda (X options) (fit-transform (apply-options tsne options) X)))))

(test-gpu "L11: the fits refuse each bad hyperparameter with its message"
  (for* ([entry (in-list (fit-calls))]
         [r (in-list (parameter-rows))])
    (match-define (cons who call) entry)
    (test-case (format "~a ~a" who (row-id r))
      (check-row who r (lambda () (call (tiny-X) (hash-union small (row-call r) #:combine (lambda (_ v) v)))))))
  (for ([bad (in-list (list (lambda () (tsne-fit (tiny-X) #:n-neighbors 2.5))
                            (lambda () (tsne-fit (tiny-X) #:learning-rate 'fast))
                            (lambda () (tsne-fit (tiny-X) #:perplexity 'high))))])
    (check-exn exn:fail:contract? bad)))

(test-gpu "L11: the fits refuse each bad X with its message, the first refusal winning"
  (for* ([entry (in-list (fit-calls))]
         [r (in-list (X-rows))])
    (match-define (cons who call) entry)
    (test-case (format "~a ~a" who (row-id r))
      (match-define (cons X options) (row-call r))
      (check-row who r (lambda () (call X options))))))

(define (fit-row r)
  (match-define (cons X options) (row-call r))
  (check-row 'tsne-fit r (lambda () (apply-options tsne-fit options X))))

(test-gpu "L11: f20b: opposite rows at the largest float32 squared norm below 2^62 fit"
  (define X (tiny-with (antipodal (- (expt 2.0 31) (expt 2.0 7)))))
  (for ([method (in-list '(fft exact))])
    (define fitted (tsne-fit X #:perplexity 5.0 #:metric 'sqeuclidean #:method method))
    (check-true (finite-rows? (embedded fitted)) (format "~a" method))))

(test-pools "L11: f23: a diverged embedding is refused and freed, under every fit"
  (define device 0)
  (define X (tiny-X))
  (for ([entry (in-list (fit-calls))])
    (match-define (cons who call) entry)
    (drain-finalizers!)
    (define before (probe-pool-used device))
    (check-cuml-error 'logic
                      (diverged who)
                      (lambda () (call X (hasheq 'perplexity 5.0 'learning-rate 1e30))))
    (drain-finalizers!)
    (check-equal? (probe-pool-used device) before (format "~a" who)))
  (check-cuml-error 'logic
                    (diverged 'tsne-fit)
                    (lambda ()
                      (tsne-fit (rows->matrix (take (fixture-rows tiny) 2) 'float32)
                                #:perplexity 2.0
                                #:method 'exact))))

(test-gpu "L11: f24: a non-array X is refused by the view check, under any seed"
  (for ([seed (in-list '(0 #f))])
    (test-case (format "random-state ~a" seed)
      (match (raised (lambda () (tsne-fit listed-X #:random-state seed)))
        [(? exn:fail:raft? e)
         (check-equal? (exn-message e)
                       (format "tsne-fit: expected a device array or #f for a view, given ~e"
                               listed-X))]
        [other (fail (format "expected an exn:fail:raft, got ~e" other))]))))

(test-gpu "L11: the generics refuse with their own names"
  (define w (list->device-vector '(1 1 1 1 1 1 1 1) #:dtype 'float32))
  (check-cuml-error 'logic
                    "transform: tsne has no transform; use fit-transform"
                    (lambda () (transform (fit (tsne #:perplexity 5.0) (tiny-X)) (tiny-X))))
  (check-cuml-error 'logic
                    "fit: tsne takes no sample-weight"
                    (lambda () (fit (tsne) (tiny-X) #:sample-weight w)))
  (check-cuml-error 'logic
                    "fit-transform: tsne takes no sample-weight"
                    (lambda () (fit-transform (tsne) (tiny-X) #:sample-weight w))))

(test-gpu "L11: after a refused NaN, large, constant or cosine zero row, a valid call succeeds"
  (for ([r (in-list (X-rows))]
        #:when (member (row-id r) '("f19" "f20" "f20b" "f21" "f22")))
    (fit-row r)
    (check-pred tsne-result? (tsne-fit (tiny-X) #:perplexity 5.0) (row-id r))))

(test-pools "L11: a refusal leaves the RMM pool where it was"
  (define device 0)
  (for ([r (in-list (X-rows))])
    (drain-finalizers!)
    (define before (probe-pool-used device))
    (fit-row r)
    (check-equal? (probe-pool-used device) before (row-id r))))
