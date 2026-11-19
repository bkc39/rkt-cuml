#lang racket/base

(module+ main
  (require (only-in racket/port with-output-to-string)
           (only-in racket/pretty pretty-write)
           (only-in racket/string string-trim)
           (only-in racket/system system)
           (only-in "logistic-data.rkt"
                    fit-parts
                    lr-fit-coef
                    lr-fit-intercept
                    lr-fit-n-iter
                    setting-C
                    setting-l1-ratio
                    setting-penalty
                    tight-max-iter
                    tight-tol)
           (only-in "real-logistic-data.rkt"
                    labelled-set
                    raw-optima-path
                    raw-optima-settings
                    tight-fit))
  (define set (labelled-set 'breast-cancer "diagnosis" '("id")))
  (define described
    (string-trim (with-output-to-string (lambda () (system "git describe --always --dirty")))))
  (define entries
    (for/list ([s (in-list raw-optima-settings)])
      (define f (fit-parts (tight-fit set 'float64 s)))
      (printf "~a: n-iter ~a\n" s (lr-fit-n-iter f))
      (list (list (setting-penalty s) (setting-C s) (setting-l1-ratio s))
            (lr-fit-coef f)
            (lr-fit-intercept f))))
  (with-output-to-file
   raw-optima-path
   #:exists 'truncate
   (lambda ()
     (printf
      (string-append ";; Raw breast-cancer's float64 logistic optima, one per capped setting.\n"
                     ";; Written by `racket private/raw-logistic-optima.rkt` on a GPU host:\n"
                     ";; cuML's qn solver through logistic-regression-fit, dtype float64,\n"
                     ";; tol ~a, max-iter ~a,\n"
                     ";; at ~a (git describe --always --dirty).\n"
                     ";; Each entry: ((penalty C l1-ratio) coef-rows intercepts).\n")
      (tight-tol 'float64)
      tight-max-iter
      described)
     (pretty-write entries))))
