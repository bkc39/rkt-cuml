#lang racket/base

(require (only-in ffi/unsafe _fun _uint _void get-ffi-obj)
         (only-in racket/match match-define)
         (only-in rackunit check-equal? test-case)
         (only-in raft/tests/private/gpu test-gpu)
         (only-in "private/elastic-net-data.rkt"
                  correlated-case
                  en-case-fit
                  l1-ratios
                  law-alpha
                  selections)
         (only-in "private/elastic-net-oracle.rkt" problem-of)
         (only-in "private/linear-data.rkt"
                  dtypes
                  kept-rows
                  lcase
                  lcase-name
                  lcase-w
                  lcase-X
                  lcase-y
                  scaled-case
                  zeroed))

(define srand (get-ffi-obj "srand" #f (_fun _uint -> _void)))

(define (alpha-of c ratio r)
  (law-alpha (problem-of (lcase-X c) (lcase-y c) (lcase-w c) #t) ratio r))

(define (powers dt)
  (if (eq? dt 'float32)
      '(-12 -3 5 12)
      '(-40 -3 5 40)))

(test-gpu "X and y scaled by 2^a with alpha by 4^a give the same bits, the intercept by 2^a (E2)"
  (define c (correlated-case 200 7 0.9 11 #:weighted? #t))
  (for* ([dt (in-list dtypes)]
         [r (in-list l1-ratios)]
         [s (in-list selections)])
    (define alpha (alpha-of c 0.01 r))
    (match-define (list coef b k) (en-case-fit c dt #:alpha alpha #:l1-ratio r #:selection s))
    (for ([a (in-list (powers dt))])
      (test-case (format "~a l1-ratio ~a ~a 2^~a" dt r s a)
        (define f (expt 2 a))
        (match-define (list coef+ b+ k+)
          (en-case-fit (scaled-case c #:X f #:y f)
                       dt
                       #:alpha (* alpha (expt 4 a))
                       #:l1-ratio r
                       #:selection s))
        (check-equal? coef+ coef)
        (check-equal? k+ k)
        (check-equal? b+ (* b f))))))

(test-gpu "weights scaled by 2^m give the same bits, odd m included (E3)"
  (define c (correlated-case 200 7 0.5 12 #:weighted? #t))
  (for* ([dt (in-list dtypes)]
         [fi (in-list '(#t #f))]
         [s (in-list selections)])
    (define fit0 (en-case-fit c dt #:alpha 0.01 #:fit-intercept fi #:selection s))
    (for ([m (in-list '(-5 -1 1 3))])
      (check-equal?
       (en-case-fit (scaled-case c #:w (expt 2 m)) dt #:alpha 0.01 #:fit-intercept fi #:selection s)
       fit0
       (format "~a ~a ~a 2^~a" dt fi s m)))))

(define gone '(3 50 101))

(test-gpu "zero weight on three rows is the fit with them deleted, bitwise (E4)"
  (for* ([n (in-list '(200 5000))]
         #:do [(define c (correlated-case n 7 0.5 13 #:weighted? #t))]
         [dt (in-list dtypes)]
         [fi (in-list '(#t #f))]
         [s (in-list selections)])
    (define zero-weight (lcase "zeroed" (lcase-X c) (lcase-y c) (zeroed (lcase-w c) gone)))
    (define deleted
      (lcase "deleted"
             (kept-rows (lcase-X c) gone)
             (kept-rows (lcase-y c) gone)
             (kept-rows (lcase-w c) gone)))
    (check-equal? (en-case-fit zero-weight dt #:alpha 0.01 #:fit-intercept fi #:selection s)
                  (en-case-fit deleted dt #:alpha 0.01 #:fit-intercept fi #:selection s)
                  (format "~a ~a ~a ~a" n dt fi s))))

(test-gpu "two identical fits are bitwise equal, weighted or not, at 50000 rows (E6)"
  (for* ([n (in-list '(1000 50000))]
         [weighted? (in-list '(#f #t))]
         #:do [(define c (correlated-case n 7 0.5 14 #:weighted? weighted?))]
         [fi (in-list '(#t #f))]
         [s (in-list selections)])
    (define (once)
      (en-case-fit c 'float32 #:alpha 0.001 #:fit-intercept fi #:selection s #:tol 1e-4))
    (check-equal? (once) (once) (format "~a ~a ~a ~a" (lcase-name c) n fi s))))

(test-gpu "'random ignores libc's srand (E7)"
  (define c (correlated-case 200 7 0.9 15))
  (for ([dt (in-list dtypes)])
    (define base (en-case-fit c dt #:alpha 0.001 #:selection 'random))
    (for ([seed (in-list '(0 1 12345))])
      (srand seed)
      (check-equal? (en-case-fit c dt #:alpha 0.001 #:selection 'random)
                    base
                    (format "~a srand ~a" dt seed)))))
