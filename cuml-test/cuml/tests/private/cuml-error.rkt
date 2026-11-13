#lang racket/base

(require (only-in cuml exn:fail:cuml?)
         (only-in racket/match match)
         (only-in rackunit check-equal? check-pred check-regexp-match fail)
         (only-in raft exn:fail:raft-kind exn:fail:raft?))

(provide (struct-out prefixed)
         check-cuml-error
         raised)

(struct prefixed (prefix pattern))

(define (raised thunk)
  (with-handlers ([exn:fail? values])
    (thunk)
    #f))

(define (check-message expected actual)
  (match expected
    [(prefixed prefix pattern)
     (check-equal? (substring actual 0 (min (string-length prefix) (string-length actual))) prefix)
     (check-regexp-match pattern (substring actual (min (string-length prefix) (string-length actual))))]
    [(? regexp?) (check-regexp-match expected actual)]
    [_ (check-equal? actual expected)]))

(define (check-cuml-error kind message thunk)
  (match (raised thunk)
    [(? exn:fail:cuml? e)
     (check-pred exn:fail:raft? e)
     (check-equal? (exn:fail:raft-kind e) kind)
     (check-message message (exn-message e))]
    [other (fail (format "expected an exn:fail:cuml, got ~e" other))]))
