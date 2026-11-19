# raco fmt and raco review, fmt's pretty-expressive, and threading-lib (for
# scripts/check-layout.rkt), each pinned by git commit (the catalog's own
# source for the package). `.fmt.rkt` uses fmt's
# internals, so a bump of fmt or pretty-expressive is a deliberate change:
# move the rev, update the hash, and re-check `.fmt.rkt` against the new fmt.
# resyntaxSources is Resyntax at resyntaxRev with its catalog dependencies at
# the commits the devshell's install resolved: only the racket-review check
# installs it, so the copy check can expand scripts/resyntax-rules*.rkt.
{
  stdenv,
  runCommand,
  fetchFromGitHub,
  fetchFromGitLab,
  racket,
  makeWrapper,
}:

let
  resyntaxRev = "40f3497321f8590eb6a0b7c7984a9116323b7ebf";
  resyntax = fetchFromGitHub {
    owner = "jackfirth";
    repo = "resyntax";
    rev = resyntaxRev;
    hash = "sha256-qd48mLtgWfPiaTN8qzaMxE0Gp48+4OGeXkkaubb/HyQ=";
  };
  rebellion = fetchFromGitHub {
    owner = "jackfirth";
    repo = "rebellion";
    rev = "8f3fc46740918205c0a293ad58dd9a8094ef3f51";
    hash = "sha256-xp65LaSuJopJQHuXq22eT7Qe2VL/grbmV9Zxhc+g6UA=";
  };
  guard = fetchFromGitHub {
    owner = "jackfirth";
    repo = "guard";
    rev = "c86ee57d3fc13eed689102b733702ed1162da681";
    hash = "sha256-Vtcgb9LbaiBkF2ixIwXnQ78rIoou5mYDoy1R6tyz5d8=";
  };
  fancyApp = fetchFromGitHub {
    owner = "samth";
    repo = "fancy-app";
    rev = "f451852164ee67e3e122f25b4bce45001a557045";
    hash = "sha256-2DdngIyocn+CrLf4A4yO9+XJQjIxzKVpmvGiNuM7mTQ=";
  };
  brag = fetchFromGitLab {
    owner = "mbutterick";
    repo = "brag";
    rev = "30cbf95e6a717e71fb8bda6b15a7253aed36115a";
    hash = "sha256-NJctskWDoBNRdBMDklALkMAPKT4A7on8pu6X3Q6NheE=";
  };
  brParserTools = fetchFromGitLab {
    owner = "mbutterick";
    repo = "br-parser-tools";
    rev = "95b7c69cf9d660a51abf4742378b9adb7100d25a";
    hash = "sha256-and0y3rBjXwmgaEwwXzJOTgX/wCSY0uUfB3+U4JLTrk=";
  };

  resyntaxSources = runCommand "rkt-cuml-resyntax-sources" { } ''
    mkdir -p $out
    cp -r ${resyntax} $out/resyntax
    cp -r ${rebellion} $out/rebellion
    cp -r ${guard} $out/guard
    cp -r ${fancyApp} $out/fancy-app
    cp -r ${brag}/brag-lib $out/brag-lib
    cp -r ${brParserTools}/br-parser-tools-lib $out/br-parser-tools-lib
    chmod -R u+w $out
  '';

  fmt = fetchFromGitHub {
    owner = "sorawee";
    repo = "fmt";
    rev = "4e1ed68e596e656960b44a8244bb33eb4e65ec64";
    hash = "sha256-zwcjNvK2qcKfsXb2mlQQNOHI2niMMPI9wx9YeEz+XTo=";
  };
  review = fetchFromGitHub {
    owner = "Bogdanp";
    repo = "racket-review";
    rev = "ecb1968f12b485b5387f66bbd9220f191d136d84";
    hash = "sha256-odbHFCxRHvLDrt+j/3qqIbeyK3lscxqSfcSt6xh0Vl4=";
  };
  threading = fetchFromGitHub {
    owner = "lexi-lambda";
    repo = "threading";
    rev = "e79cfe551740baf9f696f452d7507a6f2ed00a05";
    hash = "sha256-9iTG6eR+WsGoIClOFQyiMXeJsI220SkNvh75lh4SslM=";
  };
  prettyExpressive = fetchFromGitHub {
    owner = "sorawee";
    repo = "pretty-expressive";
    rev = "27e7be8016b38252a19f3620bc37539100b02503";
    hash = "sha256-BdwzCrufLrYrYQm3OCIOUWMgpUQGYmmsHf7CWNYLarY=";
  };

  sources = runCommand "rkt-cuml-racket-tool-sources" { } ''
    mkdir -p $out
    cp -r ${fmt} $out/fmt
    cp -r ${review} $out/review
    cp -r ${prettyExpressive}/pretty-expressive $out/pretty-expressive
    cp -r ${prettyExpressive}/pretty-expressive-lib $out/pretty-expressive-lib
    cp -r ${threading}/threading-lib $out/threading-lib
    chmod -R u+w $out
  '';
in
stdenv.mkDerivation {
  name = "rkt-cuml-racket-tools";
  dontUnpack = true;
  nativeBuildInputs = [
    racket
    makeWrapper
  ];
  buildPhase = ''
    runHook preBuild
    export HOME=$TMPDIR/home
    export PLTUSERHOME=$out/share/racket-home
    mkdir -p "$PLTUSERHOME"
    raco pkg install --batch --copy --no-docs --deps fail --scope user ${sources}/*/
    runHook postBuild
  '';
  installPhase = ''
    runHook preInstall
    makeWrapper ${racket}/bin/raco $out/bin/raco-fmt \
      --set PLTUSERHOME $out/share/racket-home --add-flags fmt
    runHook postInstall
  '';
  passthru = { inherit sources resyntaxSources resyntaxRev; };
}
