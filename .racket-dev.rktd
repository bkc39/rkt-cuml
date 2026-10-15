;; Gates for the racket-dev plugin's runner (hooks/gate.rkt) and pre-push hook.
;;
;;   racket <plugin>/hooks/gate.rkt . all      ; everything, in this order
;;   racket <plugin>/hooks/gate.rkt . push     ; the subset the pre-push hook runs
;;   racket <plugin>/hooks/gate.rkt . docs     ; one gate by name
;;
;; A command whose first word is `nix` runs as is; every other command is
;; prefixed by `shell`. nix reads the git-tracked tree, so stage new files
;; before any nix gate.

((shell "nix develop --max-jobs 1 --cores 4 --command")
 (base-ref "master")
 (gates
  (native         "nix run --max-jobs 1 --cores 4 .#copy-native-libs")
  (format         "nix fmt --max-jobs 1 --cores 4 -- --ci")
  (no-syntax-rule ("scripts/no-syntax-rule.sh" #:shell ""))
  (no-raw-malloc  ("scripts/no-raw-malloc.sh" #:shell ""))
  (gpu-suite-signals ("scripts/gpu-suite-signals.sh" #:shell ""))
  (layout         "racket scripts/check-layout.rkt")
  (pkg-deps       "raco setup -j 4 --no-docs --check-pkg-deps --unused-pkg-deps --pkgs cuml-core-lib cuml-cluster-lib cuml-lib cuml-doc cuml-test cuml")
  (review         "scripts/review.sh")
  (compile        "raco make -v cuml-lib/cuml/main.rkt cuml-doc/cuml/scribblings/cuml.scrbl scripts/check-layout.rkt")
  (test           "raco test --process --timeout 45 cuml-test lint scripts/check-layout.rkt")
  (gpu            "scripts/gpu-suite.sh")
  (resyntax       "scripts/resyntax.sh")
  (docs           "scripts/render-docs.sh .scratch/docs")
  (check          "nix flake check --max-jobs 1 --cores 4"))
 (push-gates (format no-syntax-rule no-raw-malloc gpu-suite-signals layout pkg-deps review compile test)))
