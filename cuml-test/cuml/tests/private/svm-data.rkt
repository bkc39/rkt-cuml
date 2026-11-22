#lang racket/base

(require (only-in cuml/svm
                  svc-fit
                  svc-result-classes
                  svc-result-coef0
                  svc-result-degree
                  svc-result-dual-coef
                  svc-result-gamma
                  svc-result-intercept
                  svc-result-kernel
                  svc-result-n-iter
                  svc-result-n-support
                  svc-result-support
                  svc-result-support-vectors)
         (only-in racket/list first)
         (only-in racket/match match-define)
         (only-in racket/math exact-floor)
         (only-in raft device-matrix->list* device-vector->list)
         (only-in threading ~>)
         (only-in "blob-data.rkt" normal-source uniform-source)
         (only-in "common-data.rkt" on-vector)
         (only-in "linear-data.rkt" dtypes on-matrix))

(provide (struct-out svm-case)
         (struct-out svm-fit)
         case-fit
         dyadic-blobs
         encoded-labels
         fresh-rows
         kernels
         kkt-cases
         label-encodings
         svc-result->svm-fit
         weighted-cases)

(struct svm-case (name X y w C kernel dtype gamma))

(struct svm-fit (sv support dual b classes n-support n-iter kernel gamma degree coef0) #:transparent)

(define kernels '(linear poly rbf sigmoid))

(define label-encodings
  '((reals (0 1) #f) (three-seven (3 7) #f)
                     (strings ("no" "yes") #f)
                     (symbols (no yes) #f)
                     (int32 (0 1) int32)
                     (int64 (3 7) int64)
                     (float32 (0 1) float32)
                     (float64 (-1 1) float64)))

(define (encoded-labels encoding codes)
  (match-define (list _ (list lo hi) device) encoding)
  (define labels
    (for/list ([v (in-list codes)])
      (if (zero? v) lo hi)))
  (if device
      (on-vector labels device)
      labels))

(define dyadic-step (expt 2 -6))

(define (dyadic v)
  (* dyadic-step (~> v inexact->exact (/ dyadic-step) round)))

(define (dyadic-blobs n d seed #:sep [sep 3/2] #:flip [flip 1/20])
  (define next (normal-source seed))
  (define coin (uniform-source (+ seed 7919)))
  (define shift (/ sep (sqrt d)))
  (for/lists (rows labels #:result (values rows labels))
             ([_ (in-range n)])
             (define label (if (< (coin) 0.5) 0 1))
             (define row
               (for/list ([_ (in-range d)])
                 (dyadic (+ (next) (* shift (- (* 2 label) 1))))))
             (values row
                     (if (< (coin) flip)
                         (- 1 label)
                         label))))

(define (fresh-rows n d seed)
  (define-values (rows _labels) (dyadic-blobs n d (+ seed 1000003)))
  rows)

(define law-shapes '((120 3) (300 4)))
(define law-seeds '(1 2 3))
(define law-penalties '(1/10 1 10))

(define (grid-cases weights-of tag)
  (for*/list ([dt (in-list dtypes)]
              [kernel (in-list kernels)]
              [shape (in-list law-shapes)]
              [seed (in-list law-seeds)]
              [C (in-list law-penalties)])
    (match-define (list n d) shape)
    (define-values (X y) (dyadic-blobs n d (+ seed (* 31 n))))
    (svm-case (format "~a ~a ~a ~ax~a seed ~a C ~a" tag dt kernel n d seed C)
              X
              y
              (weights-of seed (length y))
              C
              kernel
              dt
              'scale)))

(define (kkt-cases)
  (grid-cases (lambda (_seed _n) #f) "kkt"))

(define dyadic-weights #(0 1/2 1 2))

(define (weighted-cases)
  (grid-cases (lambda (seed n)
                (define pick (uniform-source (+ seed 4049)))
                (for/list ([_ (in-range n)])
                  (vector-ref dyadic-weights (min 3 (exact-floor (* 4 (pick)))))))
              "weighted"))

(define (case-fit c)
  (svc-fit (on-matrix (svm-case-X c) (svm-case-dtype c))
           (svm-case-y c)
           #:sample-weight (on-vector (and (svm-case-w c) (map exact->inexact (svm-case-w c)))
                                      (svm-case-dtype c))
           #:kernel (svm-case-kernel c)
           #:gamma (svm-case-gamma c)
           #:C (exact->inexact (svm-case-C c))))

(define (svc-result->svm-fit r)
  (svm-fit (device-matrix->list* (svc-result-support-vectors r))
           (device-vector->list (svc-result-support r))
           (first (device-matrix->list* (svc-result-dual-coef r)))
           (first (device-vector->list (svc-result-intercept r)))
           (svc-result-classes r)
           (svc-result-n-support r)
           (svc-result-n-iter r)
           (svc-result-kernel r)
           (svc-result-gamma r)
           (svc-result-degree r)
           (svc-result-coef0 r)))
