#lang racket/base

(provide fitted)

(define fitted
  (make-keyword-procedure
   (lambda (fixed-kws fixed-vals proc finish)
     (make-keyword-procedure
      (lambda (kws vals . args)
        (define arguments
          (sort (append (map cons fixed-kws fixed-vals) (map cons kws vals)) keyword<? #:key car))
        (finish (lambda () (keyword-apply proc (map car arguments) (map cdr arguments) args))))))))
