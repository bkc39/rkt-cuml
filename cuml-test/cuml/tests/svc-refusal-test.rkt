#lang racket/base

(require (only-in cuml exn:fail:cuml?)
         (only-in cuml/svm svc-decision-function svc-fit svc-predict)
         (only-in racket/list make-list)
         (only-in racket/match match-define)
         (only-in racket/string string-prefix?)
         (only-in rackunit check-pred check-true test-case)
         (only-in raft list->device-vector)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/cuml-error.rkt" raised)
         (only-in "private/linear-data.rkt" dtypes on-matrix)
         (only-in "private/svm-data.rkt" dyadic-blobs))

(define-values (rows labels) (dyadic-blobs 40 3 17))

(define (scaled by)
  (for/list ([row (in-list rows)])
    (map (lambda (v) (* by v)) row)))

(define (with-cell v)
  (cons (list* v (cdr (car rows))) (cdr rows)))

(define (weights at-zero)
  (for/list ([l (in-list labels)])
    (if (= l at-zero) 0.0 1.0)))

(define negative-weights (cons -1.0 (make-list 39 1.0)))

(define fit-rows
  `((()
     "svc-fit: kernel: 'precomputed is not supported yet; pass X with 'linear, 'poly, 'rbf or 'sigmoid"
     #:kernel precomputed)
    (() "svc-fit: kernel: expected 'linear, 'poly, 'rbf or 'sigmoid, got 'cubic" #:kernel cubic)
    (() "svc-fit: degree: expected an exact integer from 0 to 2147483647, got -1" #:degree -1)
    (() "svc-fit: degree: expected an exact integer from 0 to 2147483647, got 2.5" #:degree 2.5)
    (() "svc-fit: gamma: expected 'scale, 'auto or a finite real at least 0, got -1.0" #:gamma -1.0)
    (() "svc-fit: coef0: expected a finite real, got +nan.0" #:coef0 +nan.0)
    (() "svc-fit: C: expected a positive finite real, got 0" #:C 0)
    (() "svc-fit: C: expected a positive finite real, got -1.0" #:C -1.0)
    (() "svc-fit: C: expected a positive finite real, got +nan.0" #:C +nan.0)
    (() "svc-fit: tol: expected a positive finite real, got 0" #:tol 0)
    (() "svc-fit: tol: expected a positive finite real, got -1" #:tol -1)
    (() "svc-fit: cache-size: expected a positive finite real, got -1" #:cache-size -1)
    (() "svc-fit: max-iter: expected #f or an exact integer from 1 to 2147483647, got 0" #:max-iter 0)
    (()
     "svc-fit: nochange-steps: expected an exact integer from 0 to 2147483647, got -1"
     #:nochange-steps -1)
    (() "svc-fit: sample-weight: expected a device-vector or #f, got 'w" #:sample-weight w)
    ((y (0 1)) "svc-fit: y: expected 40 elements, got 2")
    ((y ,(make-list 40 0)) "svc-fit: y: expected two classes, got one (0)")
    ((y ,(for/list ([i (in-range 40)]) (modulo i 3)))
     "svc-fit: y: expected two classes, got 3 (0 1 ...); multiclass svc is not supported yet")
    ((y ,(cons 1.5 (cdr labels))) "svc-fit: y: expected integral class labels, got 1.5")
    ((X ,(with-cell +nan.0)) "svc-fit: X: row 0, column 0: expected a finite value, got nan")
    ((X ,(with-cell +inf.0)) "svc-fit: X: row 0, column 0: expected a finite value, got inf")
    ((X ,(make-list 40 '(1.5 1.5 1.5)))
     "svc-fit: gamma 'scale: X has zero variance; pass a numeric #:gamma")
    ((w ,(make-list 40 0.0)) "svc-fit: sample-weight: class 0 has no row of positive weight")
    ((w ,negative-weights) "svc-fit: sample-weight: negative weight -1 at row 0")
    ((w ,(weights 1)) "svc-fit: sample-weight: class 1 has no row of positive weight")
    ((w ,(cons +nan.0 (make-list 39 1.0)))
     "svc-fit: sample-weight: row 0: expected a finite value, got nan")
    (() "svc-fit: the kernel can overflow (bound " #:gamma 1e30)
    (() "svc-fit: the fit has no support vectors; lower #:tol or raise #:max-iter" #:tol 3)))

(define (fit-call dt given keywords)
  (define X (on-matrix (hash-ref given 'X rows) dt))
  (define y (hash-ref given 'y labels))
  (define w (hash-ref given 'w #f))
  (define kws
    (if w
        (append keywords (list '#:sample-weight (list->device-vector w #:dtype dt)))
        keywords))
  (define pairs
    (sort (let loop ([kws kws])
            (if (null? kws)
                '()
                (cons (cons (car kws) (cadr* kws)) (loop (cddr kws)))))
          keyword<?
          #:key car))
  (lambda ()
    (keyword-apply svc-fit (map car pairs) (map cdr pairs) (list X y))))

(define (cadr* kws)
  (match-define (list* _ v _) kws)
  v)

(define (check-refused prefix thunk)
  (define e (raised thunk))
  (check-pred exn:fail:cuml? e prefix)
  (when (exn? e)
    (check-true (string-prefix? (exn-message e) prefix) (exn-message e))))

(define (given-of pairs)
  (for/hasheq ([p (in-list (let loop ([ps pairs])
                             (if (null? ps)
                                 '()
                                 (cons (cons (car ps) (cadr* ps)) (loop (cddr ps))))))])
    (values (car p) (cdr p))))

(test-gpu "F: every fit refusal, in both dtypes"
  (for* ([dt (in-list dtypes)]
         [row (in-list fit-rows)])
    (match-define (list* given message keywords) row)
    (test-case (format "~a ~a" dt message)
      (check-refused message (fit-call dt (given-of given) keywords)))))

(test-gpu "F: a non-finite or overlong row and a steep poly kernel are refused in float32"
  (check-refused "svc-fit: X: row 0: expected a squared norm of at most"
                 (fit-call 'float32 (hasheq 'X (scaled 1e19)) '()))
  (check-refused "svc-fit: the kernel can overflow (bound "
                 (fit-call 'float32 (hash) '(#:kernel poly #:degree 40 #:coef0 1 #:gamma 1))))

(test-gpu "F: tol 0 is refused at once"
  (define started (current-inexact-milliseconds))
  (check-refused "svc-fit: tol:" (fit-call 'float64 (hash) '(#:tol 0)))
  (check-true (< (- (current-inexact-milliseconds) started) 1000)))

(test-gpu "F: predict and decision refusals"
  (define r (svc-fit (on-matrix rows 'float64) labels))
  (for ([apply-svc (in-list (list svc-predict svc-decision-function))]
        [who (in-list '(svc-predict svc-decision-function))])
    (check-refused (format "~a: result: expected an svc-result, got 'r" who)
                   (lambda () (apply-svc 'r (on-matrix rows 'float64))))
    (check-refused (format "~a: X: expected a device-matrix of float64, got" who)
                   (lambda () (apply-svc r (on-matrix rows 'float32))))
    (check-refused (format "~a: X: expected 3 columns, got 2" who)
                   (lambda () (apply-svc r (on-matrix '((1.0 2.0)) 'float64))))
    (check-refused (format "~a: X: row 0, column 0: expected a finite value, got nan" who)
                   (lambda () (apply-svc r (on-matrix (with-cell +nan.0) 'float64))))))
