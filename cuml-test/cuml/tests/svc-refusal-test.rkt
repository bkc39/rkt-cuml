#lang racket/base

(require (only-in cuml exn:fail:cuml?)
         (only-in cuml/svm
                  svc-decision-function
                  svc-fit
                  svc-predict
                  svc-result
                  svc-result?
                  svc-result-support-vectors)
         (only-in math/flonum flprev)
         (only-in racket/list make-list)
         (only-in racket/match match match-define)
         (only-in racket/string string-prefix?)
         (only-in rackunit check-equal? check-pred check-true test-case)
         (only-in raft device-matrix device-vector->list dtype list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" raised)
         (only-in "private/linear-data.rkt" dtypes on-matrix)
         (only-in "private/svm-data.rkt" dyadic-blobs))

(define-values (rows labels) (dyadic-blobs 40 3 17))

(define (scaled by)
  (for/list ([row (in-list rows)])
    (map (lambda (v) (* by v)) row)))

(define (with-cell v)
  (match rows
    [(cons (cons _ rest) more) (cons (cons v rest) more)]))

(define (weights at-zero)
  (for/list ([l (in-list labels)])
    (if (= l at-zero) 0.0 1.0)))

(define (faint at)
  (for/list ([l (in-list labels)])
    (if (= l at) 1e-45 1.0)))

(define word-labels
  (for/list ([l (in-list labels)])
    (if (= l 1) "yes" "no")))

(define negative-weights (cons -1.0 (make-list 39 1.0)))

(define heavy-weights (make-list 40 1e10))

(define float-halfway (- (expt 2.0 128) (expt 2.0 103)))

(define fit-rows
  `(#hasheq((message
             .
             "svc-fit: kernel: 'precomputed is not supported; pass X with 'linear, 'poly, 'rbf or 'sigmoid")
            (#:kernel . precomputed))
    #hasheq((message . "svc-fit: kernel: expected 'linear, 'poly, 'rbf or 'sigmoid, got 'cubic")
            (#:kernel . cubic))
    #hasheq((message . "svc-fit: degree: expected an exact integer from 0 to 2147483647, got -1")
            (#:degree . -1))
    #hasheq((message . "svc-fit: degree: expected an exact integer from 0 to 2147483647, got 2.5")
            (#:degree . 2.5))
    #hasheq((message . "svc-fit: gamma: expected 'scale, 'auto or a finite real at least 0, got -1.0")
            (#:gamma . -1.0))
    #hasheq((message . "svc-fit: gamma: expected 'scale, 'auto or a finite real at least 0, got 1000")
            (#:gamma . ,(expt 10 400)))
    #hasheq((message . "svc-fit: coef0: expected a finite real, got +nan.0") (#:coef0 . +nan.0))
    #hasheq((message . "svc-fit: coef0: expected a finite real, got 1000") (#:coef0 . ,(expt 10 400)))
    #hasheq((message . "svc-fit: C: expected a positive finite real, got 0") (#:C . 0))
    #hasheq((message . "svc-fit: C: expected a positive finite real, got -1.0") (#:C . -1.0))
    #hasheq((message . "svc-fit: C: expected a positive finite real, got +nan.0") (#:C . +nan.0))
    #hasheq((message . "svc-fit: C: expected a positive finite real, got 1000")
            (#:C . ,(expt 10 400)))
    #hasheq((message
             . "svc-fit: C: expected a positive real representable as a nonzero flonum, got 1/1000")
            (#:C . ,(expt 10 -400)))
    #hasheq((message . "svc-fit: tol: expected a positive finite real, got 0") (#:tol . 0))
    #hasheq((message . "svc-fit: tol: expected a positive finite real, got -1") (#:tol . -1))
    #hasheq((message
             . "svc-fit: tol: expected a positive real representable as a nonzero flonum, got 1/1000")
            (#:tol . ,(expt 10 -400)))
    #hasheq((message . "svc-fit: cache-size: expected a positive finite real, got -1")
            (#:cache-size . -1))
    #hasheq((message . "svc-fit: C: expected a positive finite real, got 0")
            (#:cache-size . -1)
            (#:tol . 0)
            (#:C . 0))
    #hasheq((message . "svc-fit: tol: expected a positive finite real, got 0")
            (#:cache-size . -1)
            (#:tol . 0))
    #hasheq((message . "svc-fit: max-iter: expected an exact integer from 1 to 2147483647, got 0")
            (#:max-iter . 0))
    #hasheq((message
             . "svc-fit: nochange-steps: expected an exact integer from 0 to 2147483647, got -1")
            (#:nochange-steps . -1))
    #hasheq((message . "svc-fit: sample-weight: expected a device-vector or #f, got 'w")
            (#:sample-weight . w))
    #hasheq((message . "svc-fit: y: expected 40 elements, got 2") (y . (0 1)))
    #hasheq((message . "svc-fit: y: expected two classes, got one (0)") (y . ,(make-list 40 0)))
    #hasheq((message
             . "svc-fit: y: expected two classes, got 3 (0 1 ...); multiclass svc is not supported")
            (y . ,(for/list ([i (in-range 40)])
                    (modulo i 3))))
    #hasheq((message . "svc-fit: y: expected integral class labels, got 1.5")
            (y . ,(cons 1.5 (cdr labels))))
    #hasheq((message . "svc-fit: X: row 0, column 0: expected a finite value, got nan")
            (X . ,(with-cell +nan.0)))
    #hasheq((message . "svc-fit: X: row 0, column 0: expected a finite value, got inf")
            (X . ,(with-cell +inf.0)))
    #hasheq((message . "svc-fit: gamma 'scale: X has zero variance; pass a numeric #:gamma")
            (X . ,(make-list 40 '(1.5 1.5 1.5))))
    #hasheq((message . "svc-fit: sample-weight: class 0 has no row of positive weight")
            (weight-list . ,(make-list 40 0.0)))
    #hasheq((message . "svc-fit: sample-weight: negative weight -1 at row 0")
            (weight-list . ,negative-weights))
    #hasheq((message . "svc-fit: sample-weight: class 1 has no row of positive weight")
            (weight-list . ,(weights 1)))
    #hasheq((message . "svc-fit: sample-weight: class \"yes\" has no row of positive weight")
            (y . ,word-labels)
            (weight-list . ,(weights 1)))
    #hasheq((message . "svc-fit: sample-weight: class \"no\" has no row of positive weight")
            (y . ,word-labels)
            (weight-list . ,(weights 0)))
    #hasheq((message . "svc-fit: sample-weight: row 0: expected a finite value, got nan")
            (weight-list . ,(cons +nan.0 (make-list 39 1.0))))
    #hasheq((message . "svc-fit: the kernel can overflow (bound ") (#:gamma . 1e30))
    #hasheq((message
             .
             "svc-fit: cache-size: 1e+06 MiB overflows the kernel cache's index for 40 rows; pass at most ")
            (#:cache-size . 1e6))
    #hasheq((message . "svc-fit: the fit has no support vectors; lower #:tol or raise #:max-iter")
            (#:tol . 3))))

(define float32-rows
  `(#hasheq((message . "svc-fit: C: 1e+39 overflows float32") (#:C . 1e39))
    #hasheq((message . "svc-fit: C: 1e-50 rounds to 0 in float32") (#:C . 1e-50))
    #hasheq((message . "svc-fit: tol: 1e-50 rounds to 0 in float32") (#:tol . 1e-50))
    #hasheq((message . "svc-fit: gamma: 1e+39 overflows float32") (#:gamma . 1e39))
    #hasheq((message . "svc-fit: coef0: -1e+39 overflows float32") (#:coef0 . -1e39))
    #hasheq((message . "svc-fit: gamma: expected a value finite in float32, got ")
            (X . ,(scaled 1e-20)))
    #hasheq((message . "svc-fit: sample-weight: class 1 has no row of positive weight")
            (#:C . 0.1)
            (weight-list . ,(faint 1)))
    #hasheq((message . "svc-fit: X: row 0: expected a squared norm of at most") (X . ,(scaled 1e19)))
    #hasheq((message
             .
             "svc-fit: sample-weight: row 0: C*w overflows float32; lower #:C or rescale the weights")
            (#:C . 1e30)
            (weight-list . ,heavy-weights))
    #hasheq((message . "svc-fit: C: 3.4028235677973366e+38 overflows float32") (#:C . ,float-halfway))
    #hasheq((message . "svc-fit: the kernel can overflow (bound ")
            (#:kernel . poly)
            (#:degree . 40)
            (#:coef0 . 1)
            (#:gamma . 1))))

(define float64-rows
  `(#hasheq((message
             .
             "svc-fit: sample-weight: row 0: C*w overflows float64; lower #:C or rescale the weights")
            (#:C . 1e300)
            (weight-list . ,heavy-weights))))

(define (fit-call dt row)
  (define X (on-matrix (hash-ref row 'X rows) dt))
  (define y (hash-ref row 'y labels))
  (define given
    (for/hasheq ([(key v) (in-hash row)]
                 #:when (keyword? key))
      (values key v)))
  (define kws
    (match (hash-ref row 'weight-list #f)
      [#f given]
      [w (hash-set given '#:sample-weight (list->device-vector w #:dtype dt))]))
  (define pairs (sort (hash->list kws) keyword<? #:key car))
  (lambda () (keyword-apply svc-fit (map car pairs) (map cdr pairs) (list X y))))

(define (check-refused prefix thunk)
  (define e (raised thunk))
  (check-pred exn:fail:cuml? e prefix)
  (when (exn? e)
    (check-true (string-prefix? (exn-message e) prefix) (exn-message e))))

(test-gpu "every fit refusal, in both dtypes"
  (for* ([dt (in-list dtypes)]
         [row (in-list fit-rows)])
    (define message (hash-ref row 'message))
    (test-case (format "~a ~a" dt message)
      (check-refused message (fit-call dt row)))))

(test-gpu "what narrows to 0 or infinity in float32 or float64, an overlong row and a steep poly kernel"
  (for* ([dt-rows (in-list (list (cons 'float32 float32-rows) (cons 'float64 float64-rows)))]
         [row (in-list (cdr dt-rows))])
    (define message (hash-ref row 'message))
    (test-case message
      (check-refused message (fit-call (car dt-rows) row)))))

(test-gpu "C at float32's rounding boundary is accepted below it"
  (for ([C (in-list (list (flprev float-halfway) 3.4028235e38))])
    (check-pred svc-result? (svc-fit (on-matrix rows 'float32) labels #:C C))))

(test-gpu "a fit on no rows is refused"
  (for ([dt (in-list dtypes)])
    (check-refused
     "svc-fit: X: expected at least 1 row, got 0"
     (lambda () (svc-fit (device-matrix 0 3 #:dtype dt) (list->device-vector '() #:dtype 'int32))))))

(test-gpu "tol 0 is refused at once"
  (for ([tol (in-list (list 0 (expt 10 -400)))])
    (define started (current-inexact-milliseconds))
    (check-refused "svc-fit: tol:" (fit-call 'float64 (hasheq '#:tol tol)))
    (check-true (< (- (current-inexact-milliseconds) started) 1000))))

(test-gpu "predict and decision refusals"
  (define r (svc-fit (on-matrix rows 'float64) labels))
  (define r32 (svc-fit (on-matrix rows 'float32) labels))
  (define r32-sigmoid (svc-fit (on-matrix rows 'float32) labels #:kernel 'sigmoid))
  (define forged
    `((,(struct-copy svc-result r32-sigmoid [coef0 1e39]) "coef0: expected a value finite in float32")
      (,(struct-copy svc-result r32 [gamma 1e39]) "gamma: expected a value finite in float32")
      (,(struct-copy svc-result r32 [cache-size 1e39])
       "cache-size: expected a value finite in float32"))))
  (for ([apply-svc (in-list (list svc-predict svc-decision-function))]
        [who (in-list '(svc-predict svc-decision-function))])
    (check-refused (format "~a: result: expected an svc-result, got 'r" who)
                   (lambda () (apply-svc 'r (on-matrix rows 'float64))))
    (check-refused (format "~a: X: expected a device-matrix of float64, got" who)
                   (lambda () (apply-svc r (on-matrix rows 'float32))))
    (check-refused (format "~a: X: expected 3 columns, got 2" who)
                   (lambda () (apply-svc r (on-matrix '((1.0 2.0)) 'float64))))
    (check-refused (format "~a: X: row 0, column 0: expected a finite value, got nan" who)
                   (lambda () (apply-svc r (on-matrix (with-cell +nan.0) 'float64))))
    (check-refused (format "~a: the kernel can overflow (bound " who)
                   (lambda () (apply-svc r32 (on-matrix (scaled (expt 2 11)) 'float32))))
    (check-refused (format "~a: X: row 0: expected a squared norm of at most" who)
                   (lambda () (apply-svc r32 (on-matrix (scaled 1e19) 'float32))))
    (for ([f (in-list forged)])
      (match-define (list result message) f)
      (define X (on-matrix rows (dtype (svc-result-support-vectors result))))
      (check-refused (format "~a: ~a" who message) (lambda () (apply-svc result X))))))

(test-gpu "predict narrows cache-size to a finite value and checks nothing more"
  (define r (svc-fit (on-matrix rows 'float64) labels))
  (define r32 (svc-fit (on-matrix rows 'float32) labels))
  (define accepted
    (list (struct-copy svc-result r32 [cache-size 8200.0])
          (struct-copy svc-result r [cache-size 8200.0])
          (struct-copy svc-result r [cache-size 1e39])
          (svc-fit (on-matrix rows 'float32) labels #:cache-size 1e-50)))
  (for ([result (in-list accepted)])
    (define X (on-matrix rows (dtype (svc-result-support-vectors result))))
    (check-equal? (length (svc-predict result X)) (length rows))
    (check-equal? (length (device-vector->list (svc-decision-function result X))) (length rows))))
