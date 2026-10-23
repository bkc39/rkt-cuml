#lang racket/base

(require (for-syntax racket/base)
         ;; whole-module: it also provides syntax/parse at phase 1
         syntax/parse/define
         (only-in "../../core.rkt" exn:fail:cuml))

(provide call-with-release
         with-release)

(begin-for-syntax
  (define-syntax-class release-binding
    #:description "a [name acquire release] binding"
    (pattern [name:id acquire:expr release:expr])))

(define (refuse-reentry who)
  (raise (exn:fail:cuml (format "~a: cannot re-enter its body after its resources were released" who)
                        (current-continuation-marks)
                        'logic)))

;; The acquired value arrives already evaluated, outside the extent, so control
;; that jumps back into the body (a generator resume) cannot acquire again.
(define (call-with-release who acquired release proc)
  (define held acquired)
  (define entered? #f)
  (dynamic-wind (lambda ()
                  (when entered?
                    (refuse-reentry who))
                  (set! entered? #t))
                (lambda () (proc acquired))
                (lambda ()
                  (when held
                    (set! held #f)
                    (release acquired)))))

(define-syntax-parser with-release
  [(_ #:who who:expr () body:expr ...+)
   #'(let ()
       body ...)]
  [(_ #:who who:expr (b:release-binding more:release-binding ...) body:expr ...+)
   #'(call-with-release who
                        b.acquire
                        b.release
                        (lambda (b.name) (with-release #:who who (more ...) body ...)))])
