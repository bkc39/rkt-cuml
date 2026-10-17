{
  description = "rkt-cuml - Racket bindings to NVIDIA cuML";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/07e1d92cdc0ed416cfa11ff3ca40d17e61cfba7a";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    treefmt-nix.inputs.nixpkgs.follows = "nixpkgs";
    # The shim links rkt-raft's RAPIDS set, which must be exactly the one
    # libraftrkt was built against: RAFT has no versioned namespace.
    rkt-raft.url = "github:bkc39/rkt-raft/36b118de999c298851e97b9c4199c0555fa42af4";
    rkt-raft.inputs.nixpkgs.follows = "nixpkgs";
    rkt-raft.inputs.treefmt-nix.follows = "treefmt-nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      treefmt-nix,
      rkt-raft,
    }:
    let
      system = "x86_64-linux";
      version = "0.1.0";
      minRacketVersion = "9.3";
      cudaArchitectures = "86";
      buildJobs = 4;
      capJobs = ''
        NIX_BUILD_CORES=$(( NIX_BUILD_CORES < ${toString buildJobs} ? NIX_BUILD_CORES : ${toString buildJobs} ))
        export CMAKE_BUILD_PARALLEL_LEVEL=$NIX_BUILD_CORES
      '';

      families = [ "cluster" ];
      cumlPackages = map (f: "cuml-${f}-lib") families ++ [
        "cuml-core-lib"
        "cuml-lib"
        "cuml-doc"
        "cuml-test"
        "cuml"
      ];
      cumlPackageNames = lib.concatStringsSep " " cumlPackages;

      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      inherit (pkgs) lib;
      cudaPackages = pkgs.cudaPackages_13.overrideScope (
        _: _: { inherit (pkgs.cudaPackages_13_3) libnvjitlink; }
      );
      racket = pkgs.racket;

      raft = rkt-raft.packages.${system};
      raftInputs = rkt-raft.inputs;
      inherit (raft) rapids;
      pythonEnv = raft.python-twins;
      racketTools = pkgs.callPackage ./nix/racket-tools.nix { inherit racket; };
      treefmtEval = treefmt-nix.lib.evalModule pkgs (
        import "${rkt-raft}/nix/treefmt.nix" { inherit racketTools; }
      );

      cudaDevLibs = with cudaPackages; [
        cuda_cudart
        libcublas
        libcurand
        libcusolver
        libcusparse
      ];

      shim = cudaPackages.backendStdenv.mkDerivation {
        pname = "cumlrkt";
        inherit version;
        src = ./shim;
        outputs = [
          "out"
          "tests"
        ];
        nativeBuildInputs = [
          pkgs.cmake
          pkgs.ninja
          cudaPackages.cuda_nvcc
        ];
        buildInputs = [
          rapids
          raft.shim
          raft.raft-dev
          pkgs.gtest
        ]
        ++ cudaDevLibs;
        cmakeFlags = [
          "-DBUILD_TESTING=ON"
          "-DCMAKE_CUDA_ARCHITECTURES=${cudaArchitectures}"
          "-DCUMLRKT_TESTS_PREFIX=${placeholder "tests"}"
        ];
        preBuild = capJobs;
        doCheck = true;
        checkPhase = ''
          runHook preCheck
          ctest -j1 --output-on-failure
          runHook postCheck
        '';
      };

      # An in-place write rewrites pages a live process has mapped (rktorch
      # #72); rename leaves the old inode to whoever still runs it.
      stageNativeLibs = src: ''
        _stage_failed=0
        ${lib.concatMapStrings (family: ''
          _dest="$PWD/cuml-${family}-lib/cuml/${family}/native-libs"
          mkdir -p "$_dest"
          for _f in ${src}/lib/libcumlrkt_${family}.*; do
            _b=$(basename "$_f")
            if cmp -s "$_f" "$_dest/$_b"; then
              continue
            fi
            if cp -f "$_f" "$_dest/.$_b.tmp.$$" \
               && chmod 0555 "$_dest/.$_b.tmp.$$" \
               && mv -f "$_dest/.$_b.tmp.$$" "$_dest/$_b"; then
              echo "Staged $_b from ${src}"
            else
              rm -f "$_dest/.$_b.tmp.$$"
              echo "ERROR: staging $_b failed; the existing shim stays in place" >&2
              _stage_failed=1
            fi
          done
        '') families}
      '';

      installThreading = skip: ''
        raco pkg install --batch -j 4 --copy --no-docs --deps fail --scope user ${skip} \
          ${racketTools.sources}/threading-lib
      '';

      # rkt-raft's recipe at rkt-raft's pins: raft's build-deps (polars,
      # datasets) installed offline, so raft installs with --deps fail.
      installDocsDeps = skip: ''
        raco pkg install --batch -j 4 --copy --no-docs --scope user ${skip} \
          ${raftInputs.rkt-polars.packages.${system}.racket-deps}/*/
        _polars_src="$(mktemp -d)/polars"
        cp -r ${raftInputs.rkt-polars}/polars "$_polars_src"
        chmod -R u+w "$_polars_src"
        cp ${raftInputs.rkt-polars}/polars/native-libs/candidates/linux/libcompat.so "$_polars_src/native-libs/"
        rm -rf "$_polars_src/native-libs/candidates"
        raco pkg install --batch -j 4 --copy --no-docs --scope user ${skip} --name polars "$_polars_src"
        raco pkg install --batch --deps fail --no-setup --copy --scope user ${skip} \
          --name al2-test-runner ${raftInputs.al2-test-runner-src}
        raco pkg install --batch --deps fail --no-setup --copy --scope user ${skip} \
          --name data-frame ${raftInputs.data-frame-src}
        raco pkg install --batch --deps fail --no-setup --copy --scope user ${skip} \
          --name datasets-core ${raftInputs.datasets-src}/datasets-core
        raco pkg install --batch --deps fail --no-setup --copy --scope user ${skip} \
          --name datasets ${raftInputs.datasets-src}/datasets
      '';

      # raft is linked from a writable copy of the whole rkt-raft tree: its
      # manual excerpts files outside raft/ (rkt-raft #23), and its
      # pre-installer stages libraftrkt into raft/native-libs.
      installRaft = dest: skip: ''
        export RAFT_NATIVE_LIB_PATH=${raft.shim}
        _raft_root="${dest}"
        if [ ! -d "$_raft_root/raft" ]; then
          mkdir -p "$(dirname "$_raft_root")"
          cp -r ${rkt-raft} "$_raft_root"
          chmod -R u+w "$_raft_root"
        fi
        raco pkg install --batch --deps fail --no-setup --link --scope user ${skip} \
          --name raft "$_raft_root/raft"
      '';

      installCuml = skip: ''
        raco pkg install --batch --deps fail --no-setup --link --scope user ${skip} \
          ${lib.concatMapStringsSep " " (p: "\"$PWD/${p}\"") cumlPackages}
      '';

      racketPackage = pkgs.stdenv.mkDerivation {
        pname = "rkt-cuml";
        inherit version;
        src = lib.cleanSource ./.;
        nativeBuildInputs = [ racket ];
        buildInputs = [ shim ];
        buildPhase = ''
          runHook preBuild
          export HOME=$TMPDIR/home
          export PLTUSERHOME=$TMPDIR/racket-home
          export CUML_NATIVE_LIB_PATH=${shim}
          mkdir -p $PLTUSERHOME
          ${stageNativeLibs shim}
          [ "$_stage_failed" = 0 ] || exit 1
          ${installThreading ""}
          ${installDocsDeps "--skip-installed"}
          ${installRaft "$TMPDIR/rkt-raft" ""}
          ${installCuml ""}
          raco setup -j 4 --no-docs --pkgs raft
          raco setup -j 4 --no-docs --check-pkg-deps --unused-pkg-deps --pkgs ${cumlPackageNames}
          runHook postBuild
        '';
        doCheck = true;
        checkPhase = ''
          runHook preCheck
          raco test --process --timeout 45 cuml-test
          raco make -v cuml-doc/cuml/scribblings/cuml.scrbl
          racket scripts/check-layout.rkt
          runHook postCheck
        '';
        installPhase = "touch $out";
      };

      lineCount =
        pkgs.runCommand "rkt-cuml-line-count"
          {
            src = lib.cleanSource ./.;
          }
          ''
            failed=0
            while IFS= read -r file; do
              lines=$(wc -l < "$file")
              if [ "$lines" -gt 500 ]; then
                echo "ERROR: $file has $lines lines; the limit is 500" >&2
                failed=1
              fi
            done < <(find $src -type f \( -name '*.c' -o -name '*.h' -o -name '*.hpp' \
                       -o -name '*.cpp' -o -name '*.cu' -o -name '*.rkt' -o -name '*.scrbl' \))
            [ "$failed" = 0 ] || exit 1
            touch $out
          '';

      clangTidy = shim.overrideAttrs (old: {
        pname = "cumlrkt-clang-tidy";
        outputs = [ "out" ];
        nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.clang-tools ];
        cmakeFlags = [
          "-DBUILD_TESTING=ON"
          "-DCMAKE_CUDA_ARCHITECTURES=${cudaArchitectures}"
        ];
        buildPhase = ''
          runHook preBuild
          ${capJobs}
          cmake --build . --target tidy
          runHook postBuild
        '';
        doCheck = false;
        installPhase = "touch $out";
      });

      racketReview = pkgs.stdenv.mkDerivation {
        pname = "rkt-cuml-racket-review";
        inherit version;
        src = lib.cleanSource ./.;
        nativeBuildInputs = [ racket ];
        dontConfigure = true;
        buildPhase = ''
          runHook preBuild
          export HOME=$TMPDIR/home
          export PLTUSERHOME=$TMPDIR/racket-home
          mkdir -p $PLTUSERHOME
          raco pkg install --batch -j 4 --copy --no-docs --deps fail --scope user \
            ${racketTools.sources}/*/
          raco pkg install --batch --no-docs --no-setup --deps fail --scope user \
            --link --name cuml-lint ./lint
          raco setup -j 4 --no-docs --pkgs cuml-lint
          bash scripts/review.sh
          runHook postBuild
        '';
        installPhase = "touch $out";
      };

      layout =
        pkgs.runCommand "rkt-cuml-layout"
          {
            src = lib.cleanSource ./.;
            nativeBuildInputs = [ racket ];
          }
          ''
            export HOME=$TMPDIR/home
            export PLTUSERHOME=$TMPDIR/racket-home
            ${installThreading ""}
            cp -r $src source
            chmod -R u+w source
            cd source
            raco test scripts/check-layout.rkt
            racket scripts/check-layout.rkt --deps-only --same-version \
              ${racketTools.sources}/threading-lib \
              ${raftInputs.rkt-polars.packages.${system}.racket-deps}/threading-lib
            diff -r -x compiled \
              ${racketTools.sources}/threading-lib \
              ${raftInputs.rkt-polars.packages.${system}.racket-deps}/threading-lib
            touch $out
          '';

      racketVersion =
        pkgs.runCommand "rkt-cuml-racket-version"
          {
            nativeBuildInputs = [ racket ];
          }
          ''
            have=$(racket -e '(display (version))')
            lowest=$(printf '%s\n%s\n' "$have" "${minRacketVersion}" | sort -V | head -1)
            if [ "$lowest" != "${minRacketVersion}" ]; then
              echo "ERROR: rkt-cuml needs Racket >= ${minRacketVersion}; nixpkgs pins $have" >&2
              exit 1
            fi
            touch $out
          '';

      grepGate =
        name: script:
        pkgs.runCommand "rkt-cuml-${name}"
          {
            src = lib.cleanSource ./.;
          }
          ''
            cd $src
            ${pkgs.bash}/bin/bash scripts/${script}
            touch $out
          '';

      resyntaxSource = "https://github.com/jackfirth/resyntax.git#40f3497321f8590eb6a0b7c7984a9116323b7ebf";

      # PLTUSERHOME lives outside $PWD (raco refuses a --link target inside a
      # collects dir), one directory per checkout so worktrees never share
      # installed packages, and below it one home per rkt-raft and tool pin:
      # every install skips what is installed, so a pin bump needs a fresh
      # home. Entering the shell deletes everything else under the checkout's
      # directory: old pins' homes, and homes from before this layout. A shell
      # nested in one with another pin keeps its parent's home, and ids that
      # are not 12 hex digits (a failed hash) delete nothing.
      racketHomeFor = raftSource: toolsSource: ''
        _cache_root="''${XDG_CACHE_HOME:-$HOME/.cache}/rkt-cuml-devshell"
        _checkout_id=$(printf '%s' "$PWD" | sha256sum | cut -c1-12)
        _pins_id=$(printf '%s\n%s' "${raftSource}" "${toolsSource}" | sha256sum | cut -c1-12)
        _checkout_homes="$_cache_root/$_checkout_id"
        _parent_home="''${PLTUSERHOME:-}"
        export PLTUSERHOME="$_checkout_homes/$_pins_id"
        _hex12='^[0-9a-f]{12}$'
        if ! [[ "$_checkout_id" =~ $_hex12 && "$_pins_id" =~ $_hex12 ]]; then
          echo "warning: could not hash the checkout or pins ('$_checkout_id', '$_pins_id'); not removing stale homes" >&2
        elif [ "''${_parent_home%/*}" = "$_checkout_homes" ] && [ "$_parent_home" != "$PLTUSERHOME" ]; then
          echo "Nested in a shell using $_parent_home; not removing this checkout's other homes" >&2
        elif [ -d "$_checkout_homes" ]; then
          find "$_checkout_homes" -mindepth 1 -maxdepth 1 ! -name "$_pins_id" \
            -printf "Removing this checkout's stale home entry %p\n" -exec rm -rf -- {} +
        fi
        mkdir -p "$PLTUSERHOME"
      '';
      racketHome = racketHomeFor rkt-raft racketTools.sources;

      racketHomeKeys = pkgs.runCommand "rkt-cuml-racket-home-keys" { } ''
        export HOME=$TMPDIR/home
        export XDG_CACHE_HOME=$TMPDIR/cache
        root=$XDG_CACHE_HOME/rkt-cuml-devshell
        mkdir -p checkout-a checkout-b "$XDG_CACHE_HOME/other-tool/0123456789ab"
        home_in() (
          cd "$1"
          eval "$2" >&2
          printf '%s' "$PLTUSERHOME"
        )
        fail() { echo "$1" >&2; exit 1; }
        key_a=${lib.escapeShellArg (racketHomeFor "/nix/store/a-rkt-raft" "/nix/store/a-tools")}
        key_raft=${lib.escapeShellArg (racketHomeFor "/nix/store/b-rkt-raft" "/nix/store/a-tools")}
        key_tools=${lib.escapeShellArg (racketHomeFor "/nix/store/a-rkt-raft" "/nix/store/b-tools")}

        pin_a=$(home_in checkout-a "$key_a")
        touch "$pin_a/stamp"
        again=$(home_in checkout-a "$key_a")
        [ "$pin_a" = "$again" ] || fail "PLTUSERHOME is not stable: $pin_a, $again"
        [ -f "$pin_a/stamp" ] || fail "re-entering a home emptied it"
        other=$(home_in checkout-b "$key_a")
        [ "$pin_a" != "$other" ] || fail "two checkouts share PLTUSERHOME $pin_a"
        mkdir -p "$(dirname "$pin_a")/.local/share/racket"
        touch "$(dirname "$pin_a")/.cuml-installed-0"

        raft_b=$(home_in checkout-a "$key_raft")
        [ "$pin_a" != "$raft_b" ] || fail "a rkt-raft pin bump keeps PLTUSERHOME $pin_a"
        [ ! -e "$pin_a" ] || fail "a rkt-raft pin bump leaves the old home $pin_a"
        tools_b=$(home_in checkout-a "$key_tools")
        [ "$raft_b" != "$tools_b" ] && [ "$pin_a" != "$tools_b" ] \
          || fail "a tool pin bump keeps PLTUSERHOME $tools_b"
        [ ! -e "$raft_b" ] || fail "a tool pin bump leaves the old home $raft_b"

        [ "$(dirname "$pin_a")" = "$root/$(basename "$(dirname "$pin_a")")" ] \
          || fail "homes are not <checkout>/<pins> under $root: $pin_a"
        [ "$(dirname "$pin_a")" = "$(dirname "$tools_b")" ] || fail "one checkout's homes are not siblings"
        [ -d "$other" ] || fail "another checkout's home was deleted: $other"
        [ "$(ls -A "$(dirname "$pin_a")")" = "$(basename "$tools_b")" ] \
          || fail "stale entries survive beside $tools_b: $(ls -A "$(dirname "$pin_a")")"
        [ -d "$XDG_CACHE_HOME/other-tool/0123456789ab" ] || fail "a directory outside $root was deleted"

        nested=$(PLTUSERHOME="$tools_b" home_in checkout-a "$key_a")
        [ "$nested" = "$pin_a" ] || fail "a nested shell's home is $nested, not $pin_a"
        [ -d "$tools_b" ] || fail "a nested shell deleted its parent shell's home $tools_b"

        mkdir -p failing-hash
        printf '#!${pkgs.runtimeShell}\nexit 1\n' > failing-hash/sha256sum
        chmod +x failing-hash/sha256sum
        unhashed=$(
          set +e +o pipefail
          PATH="$PWD/failing-hash:$PATH" home_in checkout-a "$key_a"
        )
        [ -d "$other" ] || fail "a failed hash deleted another checkout's home $other (home $unhashed)"
        [ -d "$tools_b" ] && [ -d "$pin_a" ] || fail "a failed hash deleted this checkout's homes"
        echo "PLTUSERHOME per checkout and pins, old pins removed: $pin_a -> $raft_b -> $tools_b; $other"
        touch $out
      '';

      provisionRacket = ''
        export CUML_NATIVE_LIB_PATH=${shim}
        ${stageNativeLibs shim}
        if [ "$_stage_failed" != 0 ]; then
          echo "  *** a cuML shim was NOT staged; cuml will load a stale shim or none." >&2
        fi
        _info_hash=$(printf '%s\n' ${shim} \
                     | cat - ${lib.concatMapStringsSep " " (p: "${p}/info.rkt") cumlPackages} \
                     | sha256sum | cut -c1-16)
        _pkg_stamp="$PLTUSERHOME/.cuml-installed-$_info_hash"
        if [ ! -f "$_pkg_stamp" ]; then
          echo "Installing raft and the cuml packages into $PLTUSERHOME (link mode)"
          rm -f "$PLTUSERHOME"/.cuml-installed-* 2>/dev/null || true
          (
            set -e
            ${installThreading "--skip-installed"}
            ${installDocsDeps "--skip-installed"}
            ${installRaft "$PLTUSERHOME/${baseNameOf (toString rkt-raft)}" "--skip-installed"}
            ${installCuml "--skip-installed"}
            RAFT_KMEANS_CANARY=${raft.kmeans-canary}/lib/libkmeans_canary.so \
              raco setup -j 4 --pkgs raft
            raco setup -j 4 --no-docs --pkgs ${cumlPackageNames}
          )
          _setup_rc=$?
          if [ "$_setup_rc" = 0 ]; then
            touch "$_pkg_stamp"
          else
            echo "cuml setup FAILED; not stamped, so the next shell entry retries" >&2
          fi
        fi
        export RAFT_NATIVE_LIB_PATH=${raft.shim}
      '';

      provisionTools = ''
        _tools_id=$(printf '%s' "${racketTools.sources}" | sha256sum | cut -c1-16)
        _tools_stamp="$PLTUSERHOME/.tools-installed-$_tools_id"
        if [ ! -f "$_tools_stamp" ]; then
          echo "Installing raco fmt, raco review and cuml-lint into $PLTUSERHOME"
          rm -f "$PLTUSERHOME"/.tools-installed-* 2>/dev/null || true
          if raco pkg install --batch -j 4 --copy --no-docs --scope user --skip-installed \
               ${racketTools.sources}/*/ \
             && raco pkg install --batch -j 4 --no-docs --scope user --skip-installed \
                  --link --name cuml-lint "$PWD/lint"; then
            touch "$_tools_stamp"
          else
            echo "tool setup FAILED; not stamped, so the next shell entry retries" >&2
          fi
        fi
      '';

      provisionResyntax = ''
        _lint_id=$(printf '%s' "${resyntaxSource}" | sha256sum | cut -c1-16)
        _lint_stamp="$PLTUSERHOME/.resyntax-installed-$_lint_id"
        if [ ! -f "$_lint_stamp" ]; then
          echo "Installing Resyntax into $PLTUSERHOME"
          rm -f "$PLTUSERHOME"/.resyntax-installed-* 2>/dev/null || true
          if raco pkg install --batch -j 4 --auto --no-docs --scope user \
               --skip-installed "${resyntaxSource}" \
             && raco pkg update --batch --auto --update-deps --no-docs \
                  --scope user "${resyntaxSource}"; then
            touch "$_lint_stamp"
          else
            echo "Resyntax setup FAILED; not stamped, so the next shell entry retries" >&2
          fi
        fi
        export PATH="$(racket -e '(require setup/dirs)(display (path->string (find-user-console-bin-dir)))'):$PATH"
      '';

      # The host's ~/.bashrc puts CUDA 11.7 on LD_LIBRARY_PATH, where it would
      # shadow the CUDA 13 libraries. The host has no /run/opengl-driver, so
      # the driver's own libraries are linked into .cuda-driver/ instead.
      gpuHook = ''
        _filtered=""
        IFS=: read -ra _entries <<< "''${LD_LIBRARY_PATH:-}"
        for _e in "''${_entries[@]}"; do
          case "$_e" in
            ""|/usr/local/cuda*) ;;
            *) _filtered="''${_filtered:+$_filtered:}$_e" ;;
          esac
        done
        _drv_farm="$PWD/.cuda-driver"
        rm -rf "$_drv_farm"; mkdir -p "$_drv_farm"
        for _l in libcuda.so.1 libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1; do
          _p=$(/sbin/ldconfig -p 2>/dev/null | grep -F "$_l (libc6,x86-64" \
                 | grep -oE '/[^ ]+' | head -1)
          if [ -n "$_p" ]; then
            ln -sf "$_p" "$_drv_farm/$_l"
          else
            echo "WARNING: $_l not found via ldconfig; GPU tests will skip" >&2
          fi
        done
        export LD_LIBRARY_PATH="$_drv_farm''${_filtered:+:$_filtered}"
        export CUML_CUDA_DRIVER_PATH="$_drv_farm"
        export CUML_SHIM_TESTS="${shim.tests}/bin"
        export RAFT_SHIM_PROBE="${raft.shim.tests}/lib/libraftrkt_probe.so"
        export CUML_RAPIDS_PREFIX="${rapids}"
        export CMAKE_BUILD_PARALLEL_LEVEL=${toString buildJobs}
      '';

      inCheckout = body: ''
        if [ -f cuml-core-lib/info.rkt ]; then
          ${body}
        else
          echo "rkt-cuml: enter the shell from the checkout's root; nothing was provisioned" >&2
        fi
      '';
    in
    {
      packages.${system} = {
        default = racketPackage;
        racket = racketPackage;
        inherit shim;
        copy-native-libs = pkgs.writeShellApplication {
          name = "copy-native-libs";
          runtimeInputs = [
            pkgs.coreutils
            pkgs.diffutils
          ];
          text = ''
            if [ ! -f cuml-core-lib/info.rkt ]; then
              echo "copy-native-libs: run from the root of an rkt-cuml checkout" >&2
              exit 1
            fi
            ${stageNativeLibs shim}
            [ "$_stage_failed" = 0 ] || exit 1
            ls -la ${lib.concatMapStringsSep " " (f: "\"$PWD/cuml-${f}-lib/cuml/${f}/native-libs\"") families}
          '';
        };
      };

      apps.${system}.copy-native-libs = {
        type = "app";
        program = "${self.packages.${system}.copy-native-libs}/bin/copy-native-libs";
        meta.description = "Stage the nix-built cuML shims into each family's native-libs";
      };

      checks.${system} = {
        inherit shim layout;
        racket-home-keys = racketHomeKeys;
        racket = racketPackage;
        racket-review = racketReview;
        formatting = treefmtEval.config.build.check self;
        clang-tidy = clangTidy;
        line-count = lineCount;
        racket-version = racketVersion;
        no-syntax-rule = grepGate "no-syntax-rule" "no-syntax-rule.sh";
        no-raw-malloc = grepGate "no-raw-malloc" "no-raw-malloc.sh";
        gpu-suite-signals = grepGate "gpu-suite-signals" "gpu-suite-signals.sh";
      };

      devShells.${system} = {
        default = pkgs.mkShell {
          packages = [
            racket
            pythonEnv
            rapids
            raft.shim
            raft.raft-dev
            pkgs.binutils
            pkgs.clang-tools
            pkgs.cmake
            pkgs.gtest
            pkgs.ninja
            cudaPackages.cuda_nvcc
            cudaPackages.cuda_sanitizer_api
            treefmtEval.config.build.wrapper
          ]
          ++ cudaDevLibs;
          shellHook = inCheckout (
            gpuHook + racketHome + provisionRacket + provisionTools + provisionResyntax
          );
        };

        ci = pkgs.mkShell {
          packages = [ racket ];
          shellHook = inCheckout (racketHome + provisionTools + provisionResyntax);
        };
      };

      formatter.${system} = treefmtEval.config.build.wrapper;
    };
}
