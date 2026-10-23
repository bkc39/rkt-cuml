#lang racket/base

(require (for-syntax racket/base)
         ;; whole-module: it also provides syntax/parse at phase 1
         syntax/parse/define
         (only-in "define-who.rkt" define/who)
         (only-in "logic-error.rkt" logic-error))

(provide call-with-release
         call-with-release/who
         with-release)

(begin-for-syntax
  (define-syntax-class release-binding
    #:description "a [name acquire release] binding"
    (pattern [name:id acquire:expr release:expr])))

(define/who (call-with-release who acquired release proc)
  (define held acquired)
  (define entered? #f)
  (dynamic-wind (lambda ()
                  (when entered?
                    (logic-error "~a: cannot re-enter its body after its resources were released"
                                 who))
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
   #'(call-with-release/who who
                            b.acquire
                            b.release
                            (lambda (b.name) (with-release #:who who (more ...) body ...)))])
