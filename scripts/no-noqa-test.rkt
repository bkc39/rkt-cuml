#lang racket/base

(require (only-in racket/file find-files make-parent-directory*)
         (only-in racket/path find-relative-path)
         (only-in racket/match match-define)
         racket/runtime-path
         (only-in rackunit check-equal? check-false check-regexp-match check-true test-case)
         (only-in "../lint/private/modules.rkt" with-temporary-directory)
         (only-in "private/gate-run.rkt" readable? run-gate test-unless-skipped))

(define-runtime-path repo "..")

(define-runtime-path fixtures "fixtures/no-noqa")

(define scripts '("scripts/no-noqa.sh" "scripts/grep-gate.sh"))

(define (scanned? path)
  (and (regexp-match? #rx"[.](rkt|scrbl)$" (path->string path))
       (not (regexp-match? #rx"/compiled/" (path->string path)))))

(define (scanned-files)
  (for*/list ([top (in-list (directory-list repo))]
              #:when (regexp-match? #rx"^(cuml-.*|lint|scripts|[.]fmt[.]rkt)$" (path->string top))
              [file (in-list (find-files scanned? (build-path repo top)))])
    (find-relative-path (simplify-path repo) (simplify-path file))))

(define (copy-into tree relative from)
  (define file (build-path tree relative))
  (make-parent-directory* file)
  (copy-file from file #t))

(define (call-with-repo-copy proc)
  (with-temporary-directory (root "no-noqa-~a")
                            (for ([relative (in-list (append (scanned-files) scripts))])
                              (copy-into root relative (build-path repo relative)))
                            (proc root)))

(define (gate-on root)
  (run-gate (path->string (build-path root "scripts/no-noqa.sh"))))

(test-case "a copy of the checkout passes, its suppressions all allowed"
  (match-define (list passed? output) (call-with-repo-copy gate-on))
  (check-true passed? output)
  (check-regexp-match #rx"^no-noqa: no new lint suppression" output))

(call-with-repo-copy
 (lambda (root)
   (define file (build-path root "lint/review.rkt"))
   (file-or-directory-permissions file 0)
   (test-unless-skipped (and (readable? file) "the unreadable file reads (root?)")
       "a .rkt grep cannot read fails the gate"
     (match-define (list passed? output) (gate-on root))
     (check-false passed? output)
     (check-regexp-match #rx"lint/review.rkt: Permission denied" output))))

(define rows
  '(("a facade's whole-file review mark in another library module fails"
     "review-ignore.txt"
     "cuml-lib/cuml/other.rkt"
     #rx"cuml-lib/cuml/other.rkt:2: lint suppression: teach lint/review.rkt")
    ("a whole-file review mark in a new info.rkt passes"
     "review-ignore.txt"
     "lint/sub/info.rkt"
     #f)
    ("a line lint mark in an info.rkt fails"
     "lint-ignore.txt"
     "lint/sub/info.rkt"
     #rx"lint/sub/info.rkt:2: lint suppression: teach lint/review.rkt")
    ("an allowed line mark in another file fails"
     "twin-skip.txt"
     "cuml-test/cuml/tests/private/other.rkt"
     #rx"private/other.rkt:2: lint suppression: teach lint/review.rkt")))

(for ([row (in-list rows)])
  (match-define (list name fixture target message) row)
  (test-case name
    (match-define (list passed? output)
      (call-with-repo-copy (lambda (root)
                             (copy-into root target (build-path fixtures fixture))
                             (gate-on root))))
    (check-equal? passed? (not message) output)
    (when message
      (check-regexp-match message output))))
