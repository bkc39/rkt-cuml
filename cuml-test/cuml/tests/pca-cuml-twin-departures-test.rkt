#lang racket/base

(require (only-in cuml/decomposition pca-fit pca-transform)
         (only-in racket/list append* last take)
         (only-in racket/match match-define)
         (only-in rackunit check-true test-case)
         (only-in raft device-matrix->list*)
         (only-in "private/pca-data.rkt" fitted-variance)
         (only-in "private/pca-twin-data.rkt"
                  fit-ours
                  named
                  ours-fitted
                  ours-noise
                  sample-count
                  twin-case
                  twin-ks
                  twin-results
                  twin-rows)
         (only-in "private/python-env.rkt" test-twin)
         (only-in "private/scaler-data.rkt" rows->matrix unit-roundoff)
         (only-in threading ~>))

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
    (define bound (* 64 (unit-roundoff 'float64) largest))
    (define ours (ours-noise (fit-ours wide 'float64 5)))
    (check-true (<= (abs (- ours (hash-ref wide-sk 'noise_variance))) bound))
    (check-true (> (abs (- ours (hash-ref wide-cuml 'noise_variance))) bound)))
  (test-case "whitened rows: theirs scale by the call's row count"
    (define rows (twin-rows correlated 'float64))
    (define result (pca-fit (rows->matrix rows 'float64) #:n-components 3 #:whiten #t))
    (define mine
      (~> (take rows 10)
          (rows->matrix 'float64)
          (pca-transform result _)
          device-matrix->list*
          append*))
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
    (check-true
     (for/and ([v (in-list (fitted-variance (ours-fitted (fit-ours scales 'float32 scales-m))))])
       (>= v 0.0)))))
