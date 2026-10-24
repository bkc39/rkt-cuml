#lang racket/base

(require (only-in cuml native-model-release! native-model-released? with-native-models)
         (only-in cuml/cluster
                  hdbscan-all-points-membership-vectors
                  hdbscan-approximate-predict
                  hdbscan-fit
                  hdbscan-membership-vector
                  hdbscan-prediction-labels
                  hdbscan-prediction-probabilities
                  hdbscan-result-labels
                  hdbscan-result-model)
         (only-in racket/list argmax count make-list range split-at take)
         (only-in rackunit check-equal? check-exn check-true test-case)
         (only-in raft device-matrix->list* device-vector->list dtype shape)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" check-cuml-error)
         (only-in "private/dbscan-data.rkt" mat)
         (only-in "private/estimator-data.rkt" apply-options)
         (only-in "private/hdbscan-data.rkt" dataset dup fit-lists prediction->lists))

(define (labels-of result)
  (device-vector->list (hdbscan-result-labels result)))

(define (predicted result X)
  (prediction->lists (hdbscan-approximate-predict result X)))

(define (alternate rows parity)
  (for/list ([row (in-list rows)]
             [i (in-naturals)]
             #:when (= (modulo i 2) parity))
    row))

(define (row-sums-at-most-one? matrix)
  (for/and ([row (in-list matrix)])
    (<= (apply + row) (+ 1.0 1e-6))))

(define (entries-in-unit? matrix)
  (for*/and ([row (in-list matrix)]
             [x (in-list row)])
    (<= 0.0 x 1.0)))

(test-gpu "AP1: predictions are int64 labels in [-1, k) and float32 probabilities in [0, 1]"
  (define result (hdbscan-fit (mat (alternate (dataset 'b50) 0) 'float32)))
  (define k (add1 (apply max (labels-of result))))
  (define p (hdbscan-approximate-predict result (mat (alternate (dataset 'b50) 1) 'float32)))
  (check-equal? (map (lambda (a) (list (dtype a) (shape a)))
                     (list (hdbscan-prediction-labels p) (hdbscan-prediction-probabilities p)))
                '((int64 (150)) (float32 (150))))
  (check-true (andmap (lambda (l) (<= -1 l (sub1 k)))
                      (device-vector->list (hdbscan-prediction-labels p))))
  (check-true (andmap (lambda (x) (<= 0.0 x 1.0))
                      (device-vector->list (hdbscan-prediction-probabilities p)))))

(test-gpu "AP2: under eom, predicting the training rows gives their fit labels"
  (for ([c (in-list (list (list "b50" (dataset 'b50) (hasheq))
                          (list "b10" (dataset 'b10) (hasheq))
                          (list "one-blob, allow-single"
                                (dataset 'one-blob)
                                (hasheq 'allow-single-cluster #t))
                          (list "b10, epsilon 3" (dataset 'b10) (hasheq 'cluster-selection-epsilon 3.0))
                          (list "b50 x10" (dup (dataset 'b50) 10) (hasheq))))])
    (test-case (car c)
      (define X (mat (cadr c) 'float32))
      (define result (apply-options hdbscan-fit (caddr c) X))
      (check-equal? (car (predicted result X)) (labels-of result)))))

(define (noise-model)
  (hdbscan-fit (mat (dataset 'identical) 'float32)))

(test-gpu "AP3: a model without clusters predicts noise"
  (check-equal? (predicted (noise-model) (mat (take (dataset 'b50) 7) 'float32))
                (list (make-list 7 -1) (make-list 7 0.0))))

(test-gpu "AP4: predicting two halves is predicting the whole"
  (define result (hdbscan-fit (mat (alternate (dataset 'b10) 0) 'float32)))
  (define-values (front back) (split-at (alternate (dataset 'b10) 1) 75))
  (define whole (predicted result (mat (append front back) 'float32)))
  (define halves (map (lambda (rows) (predicted result (mat rows 'float32))) (list front back)))
  (check-equal? whole (map append (car halves) (cadr halves))))

(define (argmax-index row)
  (argmax (lambda (i) (list-ref row i)) (range (length row))))

(test-gpu "MV1: all-points memberships are n x k, bounded, and their argmax is the label"
  (for ([name (in-list '(b50 b10 moons))])
    (test-case (symbol->string name)
      (define rows (dataset name))
      (define result (hdbscan-fit (mat rows 'float32)))
      (define labels (labels-of result))
      (define k (add1 (apply max labels)))
      (define m (hdbscan-all-points-membership-vectors result))
      (check-equal? (list (dtype m) (shape m)) (list 'float32 (list (length rows) k)))
      (define matrix (device-matrix->list* m))
      (check-true (entries-in-unit? matrix))
      (check-true (row-sums-at-most-one? matrix))
      (check-equal? (count (lambda (row l) (and (>= l 0) (not (= (argmax-index row) l))))
                           matrix
                           labels)
                    0))))

(test-gpu "MV2: held-out memberships are m x k with row sums at most 1"
  (for ([name (in-list '(b50 b10 moons))])
    (test-case (symbol->string name)
      (define result (hdbscan-fit (mat (alternate (dataset name) 0) 'float32)))
      (define k (add1 (apply max (labels-of result))))
      (define held-out (alternate (dataset name) 1))
      (define m (hdbscan-membership-vector result (mat held-out 'float32)))
      (check-equal? (shape m) (list (length held-out) k))
      (check-true (row-sums-at-most-one? (device-matrix->list* m))))))

(test-gpu "MV3: a model without clusters has empty memberships"
  (define result (noise-model))
  (check-equal? (shape (hdbscan-all-points-membership-vectors result)) '(100 0))
  (check-equal? (shape (hdbscan-membership-vector result (mat (take (dataset 'b50) 7) 'float32))) '(7 0)))

(test-gpu "MV4: the batch size does not change the memberships"
  (define result (hdbscan-fit (mat (alternate (dataset 'moons) 0) 'float32)))
  (define held-out (mat (alternate (dataset 'moons) 1) 'float32))
  (define (both batch)
    (list (device-matrix->list* (hdbscan-all-points-membership-vectors result #:batch-size batch))
          (device-matrix->list* (hdbscan-membership-vector result held-out #:batch-size batch))))
  (define reference (both 4096))
  (for ([batch (in-list '(1 7))])
    (test-case (format "batch ~a" batch)
      (check-equal? (both batch) reference))))

(define procedures
  (list (cons 'hdbscan-approximate-predict (lambda (r X) (hdbscan-approximate-predict r X)))
        (cons 'hdbscan-membership-vector (lambda (r X) (hdbscan-membership-vector r X)))
        (cons 'hdbscan-all-points-membership-vectors
              (lambda (r _X) (hdbscan-all-points-membership-vectors r)))))

(test-gpu "NM1: a released model refuses every prediction and keeps its arrays"
  (define X (mat (dataset 'b50) 'float32))
  (define result (hdbscan-fit X))
  (define before (fit-lists result))
  (native-model-release! (hdbscan-result-model result))
  (check-true (native-model-released? (hdbscan-result-model result)))
  (for ([p (in-list procedures)])
    (test-case (symbol->string (car p))
      (check-cuml-error 'logic
                        (format "~a: model: expected a live model, got a released one" (car p))
                        (lambda () ((cdr p) result X)))))
  (check-equal? (fit-lists result) before)
  (native-model-release! (hdbscan-result-model result))
  (check-true (native-model-released? (hdbscan-result-model result))))

(test-gpu "NM1: with-native-models releases its models when its body raises"
  (define result (hdbscan-fit (mat (dataset 'b10) 'float32)))
  (check-exn #rx"^inside$"
             (lambda ()
               (with-native-models ([m (hdbscan-result-model result)])
                 (error "inside"))))
  (check-true (native-model-released? (hdbscan-result-model result))))
