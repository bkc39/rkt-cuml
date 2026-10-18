#lang racket/base

(require (only-in cuml exn:fail:cuml?)
         (only-in racket/match match)
         (only-in rackunit check-equal? check-pred check-regexp-match fail)
         (only-in raft exn:fail:raft-kind exn:fail:raft?))

(provide check-cuml-error
         raised)

(define (raised thunk)
  (with-handlers ([exn:fail? values])
    (thunk)
    #f))

(define (check-message expected actual)
  (if (regexp? expected)
      (check-regexp-match expected actual)
      (check-equal? actual expected)))

(define (check-cuml-error kind message thunk)
  (match (raised thunk)
    [(? exn:fail:cuml? e)
     (check-pred exn:fail:raft? e)
     (check-equal? (exn:fail:raft-kind e) kind)
     (check-message message (exn-message e))]
    [other (fail (format "expected an exn:fail:cuml, got ~e" other))]))
