{
  description = "ROLL: Reinforcement Learning from Logical Feedback for Large Language Models";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    systems = [
      "aarch64-darwin"
      "x86_64-linux"
      "aarch64-linux"
    ];

    forAllSystems = f:
      nixpkgs.lib.genAttrs systems (system: f system);

    devPkgs = pkgs:
      with pkgs; [
        # repo
        bash
        pre-commit

        # TE, flash-attn
        ninja
        cmake

        # lang/nix
        alejandra
        nixd

        # lang/python
        python312
        uv
        ruff
      ];

    mkShell = pkgs:
      pkgs.mkShell {
        packages = devPkgs pkgs;

        env =
          {}
          // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
              stdenv.cc.cc
              glibc
            ]);
            CPPFLAGS = "-I${pkgs.cudatoolkit}/include -I${pkgs.cudaPackages.cudnn.dev}/include";
            CUDNN_PATH = pkgs.cudaPackages.cudnn.dev;
            CUDA_PATH = pkgs.cudatoolkit;
          };
      };
  in {
    devShells = forAllSystems (system: {
      default = mkShell (import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      });
    });
  };
}
