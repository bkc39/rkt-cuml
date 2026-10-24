#lang racket/base

(require (only-in raft exn:fail:raft))

(provide (struct-out exn:fail:cuml))

(struct exn:fail:cuml exn:fail:raft ())
