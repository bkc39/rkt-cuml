#lang racket/base

(require (only-in raft exn:fail:raft exn:fail:raft-kind exn:fail:raft?)
         (only-in raft/unsafe deposit)
         (only-in "define-who.rkt" define/who))

(provide deposit-for/who)

(define ((renamed who) e)
  (raise (exn:fail:raft (regexp-replace #rx"^deposit:" (exn-message e) (format "~a:" who))
                        (exn-continuation-marks e)
                        (exn:fail:raft-kind e))))

(define/who (deposit-for who resources)
  (with-handlers ([exn:fail:raft? (renamed who)])
    (deposit #:resources resources)))
