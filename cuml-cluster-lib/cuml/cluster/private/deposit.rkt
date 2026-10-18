#lang racket/base

(require (only-in raft exn:fail:raft exn:fail:raft-kind exn:fail:raft?)
         (only-in raft/unsafe deposit))

(provide deposit-for)

(define ((renamed who) e)
  (raise (exn:fail:raft (regexp-replace #rx"^deposit:" (exn-message e) (format "~a:" who))
                        (exn-continuation-marks e)
                        (exn:fail:raft-kind e))))

(define (deposit-for who resources)
  (with-handlers ([exn:fail:raft? (renamed who)])
    (deposit #:resources resources)))
