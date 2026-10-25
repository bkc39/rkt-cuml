#lang racket/base

(require (only-in cuml/decomposition pca-fit pca-params pca-transform)
         (only-in racket/list append* last split-at take)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in "private/pca-data.rkt" fitted-variance fixture-name fixtures)
         (only-in "private/pca-twin-data.rkt"
                  fit-ours
                  l14-problems
                  ours-fitted
                  ours-noise
                  sample-count
                  twin-case
                  twin-ks
                  twin-results
                  twin-rows
                  unit)
         (only-in "private/python-env.rkt" test-twin twin-params)
         (only-in "private/scaler-data.rkt" rows->matrix))

(define (named name)
  (for/first ([f (in-list fixtures)]
              #:when (equal? (fixture-name f) name))
    f))

(define (check-against-cuml f)
  (define cases
    (for*/list ([dt (in-list '(float32 float64))]
                [k (in-list (twin-ks f))]
                [solver (in-list '(full jacobi))])
      (list dt k solver)))
  (define results
    (twin-results (append (for/list ([c (in-list cases)])
                            (match-define (list dt k solver) c)
                            (twin-case "sklearn" f dt k))
                          (for/list ([c (in-list cases)])
                            (match-define (list dt k solver) c)
                            (twin-case "cuml" f dt k #:solver solver)))))
  (define-values (bounds theirs) (split-at results (length cases)))
  (for ([c (in-list cases)]
        [bound (in-list bounds)]
        [cuml (in-list theirs)])
    (match-define (list dt k solver) c)
    (test-case (format "~a ~a k=~a ~a" (fixture-name f) dt k solver)
      (check-equal? (l14-problems (fit-ours f dt k #:solver solver) cuml bound dt solver) '()))))

(for ([name (in-list '("normal" "correlated" "scales" "offset" "uniform integers" "low rank"))])
  (test-twin (format "L15: ~a is cuML-Python's within the L14 bounds" name)
    (check-against-cuml (named name))))

(test-twin "L15: the departures from cuML-Python are pinned"
  (define wide (named "wide"))
  (define correlated (named "correlated"))
  (define scales (named "scales"))
  (define scales-m (last (twin-ks scales)))
  (match-define (list wide-sk wide-cuml whitened scales-cuml)
    (twin-results (list (twin-case "sklearn" wide 'float64 5)
                        (twin-case "cuml" wide 'float64 5)
                        (twin-case "cuml" correlated 'float64 3 #:whiten #t #:rows 10)
                        (twin-case "cuml" scales 'float32 scales-m))))
  (test-case "n < d: their noise is not the mean of the components left out"
    (match-define (cons largest _) (hash-ref wide-sk 'explained_variance))
    (define bound (* 64 (unit 'float64) largest))
    (define ours (ours-noise (fit-ours wide 'float64 5)))
    (check-true (<= (abs (- ours (hash-ref wide-sk 'noise_variance))) bound))
    (check-true (> (abs (- ours (hash-ref wide-cuml 'noise_variance))) bound)))
  (test-case "whitened rows: theirs scale by the call's row count"
    (define rows (twin-rows correlated 'float64))
    (define result (pca-fit (rows->matrix rows 'float64) #:n-components 3 #:whiten #t))
    (define mine (append* (device-matrix->list* (pca-transform result (rows->matrix (take rows 10) 'float64)))))
    (define factor (sqrt (/ 9.0 (sub1 (sample-count correlated)))))
    (define top (apply max (map abs mine)))
    (for ([ours (in-list mine)]
          [theirs (in-list (hash-ref whitened 'transform_rows))]
          #:when (> (abs ours) (* 1e-3 top)))
      (check-true (<= (abs (- (abs theirs) (* factor (abs ours)))) (* 1e-6 (abs theirs)))
                  (format "~a vs ~a × ~a" theirs ours factor))))
  (test-case "scales in float32: theirs go negative, ours are clamped at 0"
    (check-true (for/or ([v (in-list (hash-ref scales-cuml 'explained_variance))])
                  (< v 0.0)))
    (check-true (for/and ([v (in-list (fitted-variance (ours-fitted (fit-ours scales 'float32 scales-m))))])
                  (>= v 0.0)))))

(define renames (hasheq 'iterated-power 'iterations 'svd-solver 'solver))

(test-twin "L18: the hyperparameters are cuML-Python's but copy, with two departures"
  (match-define (list theirs) (twin-results (list (hasheq 'op "params"))))
  (define mapped
    (for/hasheq ([(key value) (in-hash (twin-params theirs '(copy)))])
      (values (hash-ref renames key key) value)))
  (check-equal? (hash-ref mapped 'tol) 1e-7)
  (check-equal? (hash-ref mapped 'solver) 'auto)
  (check-equal? (hash-set* mapped 'tol 0.0 'solver 'full) (pca-params)))
