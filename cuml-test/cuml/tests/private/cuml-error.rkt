#lang racket/base

(require (only-in cuml exn:fail:cuml?)
         (only-in rackunit check-equal? check-pred check-regexp-match)
         (only-in raft exn:fail:raft-kind exn:fail:raft?))

(provide check-cuml-error
         raised)

(define (raised thunk)
  (with-handlers ([exn:fail? values])
    (thunk)
    #f))

(define (check-cuml-error kind message thunk)
  (define e (raised thunk))
  (check-pred exn:fail:cuml? e)
  (check-pred exn:fail:raft? e)
  (check-equal? (and (exn:fail:raft? e) (exn:fail:raft-kind e)) kind)
  (if (regexp? message)
      (check-regexp-match message
                          (if (exn? e)
                              (exn-message e)
                              ""))
      (check-equal? (and (exn? e) (exn-message e)) message)))
