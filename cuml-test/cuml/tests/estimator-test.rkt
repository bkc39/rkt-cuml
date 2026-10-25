#lang racket/base

(require (only-in cuml/estimator
                  clusterer?
                  estimator-params
                  estimator-set-params
                  estimator?
                  fit
                  fit-predict
                  fit-transform
                  fitted?
                  gen:clusterer
                  gen:estimator
                  gen:predictor
                  gen:scorer
                  gen:transformer
                  inverse-transform
                  predict
                  predictor?
                  score
                  scorer?
                  transform
                  transformer?)
         (only-in racket/generic exn:fail:support?)
         (only-in racket/list make-list)
         (only-in rackunit check-equal? check-false check-regexp-match check-true test-case)
         (only-in "private/capabilities.rkt" capabilities)
         (only-in "private/cuml-error.rkt" raised))

(define fit-calls (box '()))

(define (weighted-mean xs ws)
  (/ (apply + (map * xs ws)) (apply + ws)))

(struct threshold (shift cut)
  #:transparent
  #:methods gen:estimator
  [(define (fit e X [y #f] #:sample-weight [w #f])
     (set-box! fit-calls (cons (list X y w) (unbox fit-calls)))
     (threshold (threshold-shift e)
                (+ (threshold-shift e) (weighted-mean X (or w (make-list (length X) 1))))))
   (define (fitted? e)
     (and (threshold-cut e) #t))
   (define (estimator-params e)
     (hasheq 'shift (threshold-shift e)))
   (define (estimator-set-params e params)
     (threshold (hash-ref params 'shift (threshold-shift e)) #f))]
  #:methods gen:predictor
  [(define (predict e X)
     (for/list ([x (in-list X)])
       (>= x (threshold-cut e))))]
  #:methods gen:clusterer
  []
  #:methods gen:transformer
  [(define (transform e X)
     (for/list ([x (in-list X)])
       (- x (threshold-cut e))))]
  #:methods gen:scorer
  [(define (score e X [_y #f] #:sample-weight [_w #f])
     (- (for/sum ([x (in-list X)]) (expt (- x (threshold-cut e)) 2))))])

(struct labeller ()
  #:methods gen:estimator
  [(define (fit e _X [_y #f] #:sample-weight [_w #f])
     e)
   (define (fitted? _e)
     #f)
   (define (estimator-params _e)
     (hasheq))
   (define (estimator-set-params e _params)
     e)]
  #:methods gen:clusterer
  [(define (fit-predict _e _X [_y #f] #:sample-weight [_w #f])
     'labeller-marker)])

(define bare-fits (box 0))

(struct bare-clusterer ()
  #:methods gen:estimator
  [(define (fit e _X [_y #f] #:sample-weight [_w #f])
     (set-box! bare-fits (add1 (unbox bare-fits)))
     e)
   (define (fitted? _e)
     #f)
   (define (estimator-params _e)
     (hasheq))
   (define (estimator-set-params e _params)
     e)]
  #:methods gen:clusterer
  [])

(test-case "E1: requiring cuml/estimator loads neither raft nor the FFI"
  (parameterize ([current-namespace (make-base-namespace)])
    (namespace-require 'cuml/estimator)
    (check-equal? (list (module-declared? 'raft #f) (module-declared? 'ffi/unsafe #f)) '(#f #f))))

(test-case "E2: the capability predicates"
  (check-equal? (capabilities (threshold 0 #f)) '(#t #t #t #t #t))
  (check-equal? (capabilities (labeller)) '(#t #f #t #f #f))
  (for ([v (in-list (list 3 'x (hash)))])
    (check-equal? (capabilities v) '(#f #f #f #f #f) (format "~v" v))))

(define (random-list)
  (for/list ([_ (in-range (add1 (random 20)))])
    (- (random 201) 100)))

(define (random-weights n)
  (for/list ([_ (in-range n)])
    (add1 (random 5))))

(define (calls-of thunk)
  (set-box! fit-calls '())
  (define result (thunk))
  (values result (reverse (unbox fit-calls))))

(test-case "E3: fit-predict and fit-transform are predict and transform after fit"
  (random-seed 7)
  (for ([_ (in-range 200)])
    (define X (random-list))
    (define e (threshold (- (random 21) 10) #f))
    (for* ([y (in-list (list #f (map (lambda (x) (modulo x 2)) X)))]
           [w (in-list (list #f (random-weights (length X))))])
      (define f (fit e X y #:sample-weight w))
      (define-values (labels predict-calls)
        (calls-of (lambda () (fit-predict e X y #:sample-weight w))))
      (check-equal? labels (predict f X))
      (check-equal? predict-calls (list (list X y w)))
      (define-values (moved transform-calls)
        (calls-of (lambda () (fit-transform e X y #:sample-weight w))))
      (check-equal? moved (transform f X))
      (check-equal? transform-calls (list (list X y w))))
    (define-values (_ plain-calls) (calls-of (lambda () (fit-predict e X))))
    (check-equal? plain-calls (list (list X #f #f)))))

(test-case "E4: a clusterer's own fit-predict is used"
  (check-equal? (fit-predict (labeller) '(1 2 3)) 'labeller-marker))

(test-case "E5: fit-predict without predict is not supported, and fits nothing"
  (set-box! bare-fits 0)
  (define e (raised (lambda () (fit-predict (bare-clusterer) '(1 2 3)))))
  (check-true (exn:fail:support? e))
  (check-equal? (exn-message e) "fit-predict: not implemented for #<bare-clusterer>")
  (check-equal? (unbox bare-fits) 0))

(test-case "E6: a value without the capability is a contract violation naming the generic"
  (define X '(1 2 3))
  (for ([thunk
         (list (lambda () (predict 3 X)) (lambda () (fit 3 X)) (lambda () (transform (labeller) X)))]
        [pattern (list #rx"^predict: contract violation:\n  expected: predictor\\?"
                       #rx"^fit: contract violation:\n  expected: estimator\\?"
                       #rx"^transform: contract violation:\n  expected: transformer\\?")])
    (define e (raised thunk))
    (check-true (exn:fail:contract? e))
    (check-regexp-match pattern (exn-message e))))

(define (declare-and-require form)
  (parameterize ([current-namespace (make-base-namespace)])
    (eval form)
    (eval '(require 'toy))))

(test-case "E7: a struct missing a required method is refused"
  (for ([form (list '(module toy racket/base
                       (require cuml/estimator)
                       (struct no-fit ()
                         #:methods gen:estimator
                         [(define (fitted? e)
                            #f)
                          (define (estimator-params e)
                            (hasheq))
                          (define (estimator-set-params e params)
                            e)]))
                    '(module toy racket/base
                       (require cuml/estimator)
                       (struct no-transform ()
                         #:methods gen:transformer
                         [(define (fit-transform e X [y #f] #:sample-weight [w #f])
                            X)])))]
        [prefix (list #rx"^fit: required method is not implemented"
                      #rx"^transform: required method is not implemented")])
    (define e (raised (lambda () (declare-and-require form))))
    (check-true (exn:fail? e))
    (check-regexp-match prefix (exn-message e))))

(test-case "E8: an implementer's example: fit is functional"
  (define e (threshold 1 #f))
  (define f (fit e '(1 2 3)))
  (check-equal? e (threshold 1 #f))
  (check-false (fitted? e))
  (check-true (fitted? f))
  (define before (threshold (threshold-shift f) (threshold-cut f)))
  (fit f '(10 20))
  (check-equal? f before))

(test-case "E9: an implementer's example: setting parameters gives an unfitted estimator"
  (define e (threshold 2 #f))
  (define f (fit e '(1 2 3)))
  (define cloned (estimator-set-params f (hasheq)))
  (check-false (fitted? cloned))
  (check-equal? cloned e)
  (define updated (estimator-set-params f (hasheq 'shift 5)))
  (check-false (fitted? updated))
  (check-equal? (estimator-params updated) (hash-set (estimator-params f) 'shift 5)))

(struct mean-split (shift cut)
  #:transparent
  #:methods gen:estimator
  [(define (fit e X [_y #f] #:sample-weight [_w #f])
     (mean-split (mean-split-shift e) (+ (mean-split-shift e) (/ (apply + X) (length X)))))
   (define (fitted? e)
     (and (mean-split-cut e) #t))
   (define (estimator-params e)
     (hasheq 'shift (mean-split-shift e)))
   (define (estimator-set-params e params)
     (mean-split (hash-ref params 'shift (mean-split-shift e)) #f))]
  #:methods gen:predictor
  [(define (predict e X)
     (for/list ([x (in-list X)])
       (if (< x (mean-split-cut e)) 0 1)))])

(struct shifter (by)
  #:methods gen:transformer
  [(define (transform e X)
     (map (lambda (x) (+ x (shifter-by e))) X))
   (define (inverse-transform e X)
     (map (lambda (x) (- x (shifter-by e))) X))])

(test-case "E10: inverse-transform is optional, and its fallback is not supported"
  (define e (raised (lambda () (inverse-transform (threshold 0 1) '(1 2 3)))))
  (check-true (exn:fail:support? e))
  (check-equal? (exn-message e) "inverse-transform: not implemented for (threshold 0 1)")
  (check-equal? (inverse-transform (shifter 3) (transform (shifter 3) '(1 2 3))) '(1 2 3))
  (check-regexp-match #rx"^inverse-transform: contract violation:\n  expected: transformer\\?"
                      (exn-message (raised (lambda () (inverse-transform (labeller) '(1)))))))

(test-case "G1: estimator-set-params takes keywords as a host implementation's hash"
  (define e (fit (mean-split 1 #f) '(1 2 9 10)))
  (check-equal? (estimator-set-params e #:shift 5) (estimator-set-params e (hasheq 'shift 5)))
  (check-equal? (estimator-set-params e #:shift 5) (mean-split 5 #f))
  (check-equal? (estimator-set-params e (hasheq 'shift 2) #:shift 5) (mean-split 5 #f))
  (check-equal? (estimator-set-params e) (mean-split 1 #f))
  (check-false (fitted? (estimator-set-params e)))
  (check-equal? (object-name estimator-set-params) 'estimator-set-params)
  (check-regexp-match #rx"^estimator-set-params: contract violation"
                      (exn-message (raised (lambda () (estimator-set-params 3 #:shift 1))))))

(test-case "M3: estimator-set-params names itself for changes that are not a hash"
  (define e (mean-split 1 #f))
  (for ([call (in-list (list (lambda () (estimator-set-params e 5))
                             (lambda () (estimator-set-params e 5 #:shift 1))))])
    (check-regexp-match #rx"^estimator-set-params: contract violation\n  expected: hash\\?"
                        (exn-message (raised call)))))
