# Cross-Term Greeks

Make-based C++ risk core and LaTeX writeup for MBS PnL attribution with
second-order and cross-factor Greeks.

## Build and Test

```sh
make test
```

This builds the C++ test binaries under `cpp/build/` and runs all
`cpp/tests/test_*.cpp` programs with strict compiler warnings enabled.

## Formatting and Editor Diagnostics

```sh
make format
make format-check
```

The repository includes `.clang-format`, `compile_flags.txt`, and
`cpp/compile_flags.txt` so clangd can resolve `mbs/...` includes without
spurious import errors. The Makefile formatting targets mirror the NYU
binomial-pricing project workflow.

## PCA/SVD Factor Inputs

The reusable PCA path is under `cpp/include/decomposition/SVD/`:

- `DenseMatrix` owns observation-by-input data with RAII value semantics.
- `SVDInput` adds optional row and input labels.
- `SVDDecomposer` computes singular values/vectors.
- `PCATransformer` centers the input matrix and runs PCA through SVD.
- `PCAResult::project(...)` maps a new raw move into component space.
- `PCAResult::factor_basis(...)` converts PCA loadings into an `mbs::factors`
  basis that can be used by factor-volga or Hessian workflows.

The module is input-agnostic: volatility buckets, rate buckets, spread buckets,
or any other numeric panel can use the same observation-by-feature matrix.

For Python research workflows, use the wheel package under `src/cross_term_greeks`.
It uses NumPy/LAPACK for PCA/SVD and keeps this project focused on factor naming,
move projection, finite-difference Greeks, and rate-PC x vol-PC interaction PnL.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy wheel
.venv/bin/python -m pip install -e . --no-build-isolation
make python-test
make wheel
```

The built wheel is written to `dist/`.

## Build the Writeup

```sh
make latex
```

The LaTeX build uses `latexmk -pdf -shell-escape` from `write_up/Makefile`
because the paper uses `minted`.

## Clean

```sh
make clean
make latex-clean
```

## C++ Layout

```text
cpp/
  include/mbs/
    attribution/   Generic attribution request/result and calculators
    factors/       Factor keys, moves, and bases
    numerics/      Finite-difference helpers
    scenarios/     Scenario prices and lookup lattice
  src/
  tests/
```

The attribution layer is math-engine oriented:

- `ScalarSecondOrderCalculator` handles volga, CC convexity, OAS convexity, and
  other one-factor second-order components.
- `CrossSecondOrderCalculator` handles rate-vol, CC-OAS, vol-OAS, and other
  pairwise cross terms.
- `MatrixHessianCalculator` handles multi-factor Hessian input by producing all
  scalar diagonal terms and all pairwise cross terms, then aggregating their
  explained PnL.
