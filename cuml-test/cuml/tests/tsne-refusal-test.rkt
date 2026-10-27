#lang racket/base

(require (only-in cuml fit fit-transform transform tsne)
         (only-in cuml/manifold tsne-fit tsne-fit-transform tsne-result?)
         (only-in racket/flonum flsingle)
         (only-in racket/hash hash-union)
         (only-in racket/list make-list take)
         (only-in racket/match match match-define)
         (only-in racket/math sqr)
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
         (only-in "private/tsne-data.rkt" diverged-rx embedded)
         (only-in "private/umap-data.rkt" finite-rows? fixture-X fixture-rows tiny))

(define small (hasheq 'perplexity 5.0))

(define pool-device 0)

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
   (row "f7" "perplexity: expected a finite value above 1 in float32, got 1" (hasheq 'perplexity 1.0))
   (row "f7 nan"
        "perplexity: expected a finite value above 1 in float32, got nan"
        (hasheq 'perplexity +nan.0))
   (row "f7 inf"
        "perplexity: expected a finite value above 1 in float32, got inf"
        (hasheq 'perplexity +inf.0))
   (row "f7 float"
        "perplexity: expected a finite value above 1 in float32, got 1.000000001"
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

(define (bound-message norm most metric got)
  (format "X: row 4: expected ~a of at most ~a under ~a, got ~a" norm most metric got))

(define sqeuclidean-bound (expt 2 61))

(define (sqeuclidean-message got)
  (bound-message "a squared norm" sqeuclidean-bound 'sqeuclidean got))

(define past-sqeuclidean-y (expt 2 30))

(define past-sqeuclidean-z (+ (expt 2 30) 128))

(define past-sqeuclidean `([(4 0) 0.0] [(4 1) ,past-sqeuclidean-y] [(4 2) ,past-sqeuclidean-z]))

(define past-sqeuclidean-norm (+ (sqr past-sqeuclidean-y) (sqr past-sqeuclidean-z)))

(define manhattan-bound (* (sqrt 2.0) (expt 2.0 62)))

(define (manhattan-message got)
  (bound-message "an L1 norm" (inexact->exact manhattan-bound) 'manhattan (inexact->exact got)))

(define (antipodal x y [z 0.0])
  `([(4 0) ,x] [(4 1) ,y] [(4 2) ,z] [(5 0) ,(- x)] [(5 1) ,(- y)] [(5 2) ,(- z)]))

(define diverging-pair (antipodal 1807712000.0 1159251072.0))

(define past-manhattan (flsingle 3.5e18))

(define inside-manhattan (flsingle 2.1739693712846356e18))

(define pca-bound (expt 2.0 126))

(define (pca-message got)
  (format "X: expected a total squared distance from the mean of at most ~a under pca init, got ~a"
          pca-bound
          (exact->inexact got)))

(define (spread-rows a)
  (for/list ([i (in-range 16)])
    (list (if (even? i)
              a
              (- a))
          (* (quotient i 2) (expt 2.0 44))
          (* (modulo i 3) (expt 2.0 44)))))

(define (spread-of rows)
  (for/sum ([column (in-list (apply map list rows))])
           (define mean (/ (apply + (map inexact->exact column)) (length column)))
           (for/sum ([x (in-list column)]) (sqr (- (inexact->exact x) mean)))))

(define past-pca (spread-rows (* 5 (expt 2.0 60))))

(define inside-pca (spread-rows (- (expt 2.0 61) (expt 2.0 37))))

(define small-scale-message
  (string-append "embedding: row 0, column 0: the optimisation diverged to nan;"
                 " lower learning-rate, centre or rescale X, or raise n-neighbors"))

(define (scaled-tiny k)
  (rows->matrix (map (lambda (r) (map (lambda (x) (* x k)) r)) (fixture-rows tiny)) 'float32))

(define (offset-rows t0)
  (for/list ([i (in-range 32)])
    (list (+ t0 (* 3600.0 (modulo i 4))) (+ (modulo i 4) (* 0.01 i)) 0.0)))

(define (centred rows)
  (define means (map (lambda (c) (/ (apply + c) (length c))) (apply map list rows)))
  (map (lambda (r) (map - r means)) rows))

(define (duplicates k)
  (rows->matrix (append (make-list k '(0 0)) (make-list k '(1 1))) 'float32))

(define (line-rows largest)
  (rows->matrix (for/list ([i (in-range 8)])
                  (list (* i (/ largest 7)) 0.0))
                'float32))

(define limit-ratio 0.97)

(define (X-row id message X [options small])
  (row id message (cons X options)))

(define (X-rows)
  (define one-row (rows->matrix '((1 2 3)) 'float32))
  (define zeros (rows->matrix (make-list 8 '(0 0 0)) 'float32))
  (define cosine (hash-set small 'metric 'cosine))
  (define sqeuclidean (hash-set small 'metric 'sqeuclidean))
  (define manhattan (hash-set small 'metric 'manhattan))
  (define pca (hasheq 'perplexity 3.0 'init 'pca))
  (define normless-pca (rows->matrix (cons '(0.0 0.0 0.0) (cdr past-pca)) 'float32))
  (list
   (X-row "f11" "X: expected float32, got float64" (tiny-with '() 'float64))
   (X-row "f12" "X: expected rank 2, got rank 1" (blank-vector 8 'float32))
   (X-row "f12 layout"
          "X: expected row-major, got col-major 8x3"
          (blank-matrix 8 3 'float32 #:layout 'col-major))
   (X-row "f13" "X: expected at least 2 rows, got 1" one-row)
   (X-row "f14"
          "X: expected at least 2 columns, got 1"
          (rows->matrix (map (lambda (r) (take r 1)) (fixture-rows tiny)) 'float32))
   (X-row "f17" "perplexity: expected at most the 8 neighbours, got 30" (tiny-X) (hasheq))
   (X-row "f17 neighbours"
          "perplexity: expected at most the 4 neighbours, got 5"
          (tiny-X)
          (hasheq 'perplexity 5.0 'n-neighbors 4))
   (X-row "f19" nan-message (tiny-with nan-row))
   (X-row "f19 inf"
          "X: row 0, column 2: expected a finite value, got -inf"
          (tiny-with '([(0 2) -inf.0])))
   (X-row "f20" large-message (tiny-with large-row))
   (X-row "f20b" (sqeuclidean-message past-sqeuclidean-norm) (tiny-with past-sqeuclidean) sqeuclidean)
   (X-row "f20b exact"
          (sqeuclidean-message past-sqeuclidean-norm)
          (tiny-with past-sqeuclidean)
          (hash-set sqeuclidean 'method 'exact))
   (X-row "f20b float32 rounding"
          (sqeuclidean-message (+ (sqr 1807712000) (sqr 1159251072)))
          (tiny-with diverging-pair)
          sqeuclidean)
   (X-row "f20c"
          (manhattan-message (* 3 past-manhattan))
          (tiny-with (antipodal past-manhattan past-manhattan past-manhattan))
          manhattan)
   (X-row "f20c exact"
          (manhattan-message (* 3 past-manhattan))
          (tiny-with (antipodal past-manhattan past-manhattan past-manhattan))
          (hash-set manhattan 'method 'exact))
   (X-row "f20d" (pca-message (spread-of past-pca)) (rows->matrix past-pca 'float32) pca)
   (X-row "f20d exact"
          (pca-message (spread-of past-pca))
          (rows->matrix past-pca 'float32)
          (hash-set pca 'method 'exact))
   (X-row "f21" cosine-message (tiny-with zero-row) cosine)
   (X-row "f23 small scale" small-scale-message (scaled-tiny 1e-11) sqeuclidean)
   (X-row "f23 small scale exact"
          small-scale-message
          (scaled-tiny 1e-11)
          (hash-set sqeuclidean 'method 'exact))
   (X-row "f23 offset" small-scale-message (rows->matrix (offset-rows 1.7e9) 'float32))
   (X-row "f23 duplicates" small-scale-message (duplicates 40))
   (X-row "f23 duplicates past the n-neighbors cap"
          small-scale-message
          (duplicates 1100)
          (hash-set small 'n-neighbors 1023))
   (X-row "f23 below 2^-64"
          small-scale-message
          (line-rows (* limit-ratio (expt 2.0 -64)))
          (hash-set small 'metric 'manhattan))
   (X-row "f22" every-row-equal (rows->matrix (make-list 8 '(1.5 1.5 1.5)) 'float32))
   (X-row "f22 zeros" every-row-equal zeros)
   (X-row "f1 before f11"
          "method: expected fft or exact"
          (tiny-with '() 'float64)
          (hash-set small 'method 'bh))
   (X-row "f5 before f11"
          "n-neighbors: expected at least 2, got 1"
          (tiny-with '() 'float64)
          (hash-set small 'n-neighbors 1))
   (X-row "f13 before f17" "X: expected at least 2 rows, got 1" one-row (hasheq))
   (X-row "f19 before f20" nan-message (tiny-with (append nan-row large-row)))
   (X-row "f20 before f20b" large-message (tiny-with large-row) sqeuclidean)
   (X-row "f20 before f20c" large-message (tiny-with large-row) manhattan)
   (X-row "f20 before f20d" large-message (tiny-with large-row) (hash-set small 'init 'pca))
   (X-row "f20 before f21" large-message (tiny-with (append zero-row large-row)) cosine)
   (X-row "f21 before f22" (normless-message 0) zeros cosine)
   (X-row "f21 before f20d" (normless-message 0) normless-pca (hash-set pca 'metric 'cosine))
   (X-row "f21 before f20d exact"
          (normless-message 0)
          normless-pca
          (hash-set* pca 'metric 'cosine 'method 'exact))))

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
      (check-row who
                 r
                 (lambda ()
                   (call (tiny-X) (hash-union small (row-call r) #:combine (lambda (_ v) v)))))))
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

(test-gpu "L11: rows just inside the sqeuclidean, manhattan and pca init bounds fit"
  (for* ([entry (in-list (list (list "f20b"
                                     (tiny-with (antipodal 1278245504.0 819714240.0))
                                     (hasheq 'perplexity 5.0 'metric 'sqeuclidean))
                               (list "f20c"
                                     (tiny-with
                                      (antipodal inside-manhattan inside-manhattan inside-manhattan))
                                     (hasheq 'perplexity 5.0 'metric 'manhattan))
                               (list "f20d"
                                     (rows->matrix inside-pca 'float32)
                                     (hasheq 'perplexity 3.0 'init 'pca))))]
         [method (in-list '(fft exact))])
    (match-define (list id X options) entry)
    (define fitted (apply-options tsne-fit (hash-set options 'method method) X))
    (check-true (finite-rows? (embedded fitted)) (format "~a ~a" id method))))

(test-gpu "L11: f23's remedies fit: centred offset rows, more neighbours, a line just above 2^-64"
  (for* ([entry (in-list
                 (list (list "centred" (rows->matrix (centred (offset-rows 1.7e9)) 'float32) small)
                       (list "n-neighbors" (duplicates 40) (hash-set small 'n-neighbors 60))
                       (list "above 2^-64"
                             (line-rows (/ (expt 2.0 -64) limit-ratio))
                             (hash-set small 'metric 'manhattan))))]
         [method (in-list '(fft exact))])
    (match-define (list id X options) entry)
    (define fitted (apply-options tsne-fit (hash-set options 'method method) X))
    (check-true (finite-rows? (embedded fitted)) (format "~a ~a" id method))))

(test-pools "L11: f23: a diverged embedding is refused and freed, under every fit"
  (define X (tiny-X))
  (for ([entry (in-list (fit-calls))])
    (match-define (cons who call) entry)
    (drain-finalizers!)
    (define before (probe-pool-used pool-device))
    (check-cuml-error 'logic
                      (diverged-rx who)
                      (lambda () (call X (hasheq 'perplexity 5.0 'learning-rate 1e30))))
    (drain-finalizers!)
    (check-equal? (probe-pool-used pool-device) before (symbol->string who)))
  (check-cuml-error 'logic
                    (diverged-rx 'tsne-fit)
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

(test-gpu "L11: after a refused NaN, large, spread, constant or cosine zero row, a valid call succeeds"
  (for ([r (in-list (X-rows))]
        #:when (member (row-id r) '("f19" "f20" "f20b" "f20c" "f20d" "f21" "f22")))
    (fit-row r)
    (check-pred tsne-result? (tsne-fit (tiny-X) #:perplexity 5.0) (row-id r))))

(test-pools "L11: a refusal leaves the RMM pool where it was"
  (for ([r (in-list (X-rows))])
    (drain-finalizers!)
    (define before (probe-pool-used pool-device))
    (fit-row r)
    (check-equal? (probe-pool-used pool-device) before (row-id r))))
