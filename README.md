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

- `Matrix` owns observation-by-input data with RAII value semantics.
- `SVDInput` adds optional row and input labels.
- `SVDDecomposer` computes singular values/vectors.
- `PCATransformer` centers the input matrix and runs PCA through SVD.
- `PCAResult::project(...)` maps a new raw move into component space.
- `PCAResult::factor_basis(...)` converts PCA loadings into an `mbs::factors`
  basis that can be used by factor-volga or Hessian workflows.
- `PCATransformer` can solve through the existing SVD path or through direct
  covariance-matrix eigen reduction with `PCAOptions::solver =
  PCASolver::Covariance`.

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

Python `fit_pca(..., solver="svd")` keeps the default SVD behavior. Use
`fit_pca(..., solver="covariance")` to diagonalize the empirical covariance
matrix directly, which is often the most explicit representation for fixed
income bucket histories where off-diagonal covariance terms matter. The
covariance solver requires demeaned columns: leave `center=True`, or pass
pre-demeaned observations when using `center=False`.

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
    rates/         Curve, bootstrap, calibration, and rate-vol proxy helpers
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

## Rates and Calibration Helpers

The C++ rates layer is deliberately calculator-oriented: the calling app owns
market-data loading, curve conventions, identifiers, persistence, and scenario
orchestration. The library supplies deterministic math objects that can be fed
by that app:

- `numerics::Interpolator` supports linear, log-linear, flat-forward/log-DF,
  and monotone cubic interpolation.
- `rates::DiscountCurve` stores discount factors and derives zero and
  continuously-compounded forward rates.
- `rates::bootstrap_discount_curve(...)` bootstraps simple deposit, ZCB, and
  par-swap instruments into a discount curve.
- `rates::calibrate_bootstrap_curve(...)` returns model-vs-market quote errors
  and RMSE for a bootstrapped curve.
- `rates::derive_swaption_proxy_vol(...)` derives an annuity-weighted
  swaption-vol proxy from forward-rate vol buckets and a correlation matrix.
- `rates::calibrate_lmm_sofr_proxy(...)` provides a lightweight LMMSOFR-style
  proxy calibration over forward-rate volatility buckets.

This is the mathematical framework needed for calculator integration, but it is
not a full production LMM simulator or global optimizer. If the app needs true
multi-curve SOFR conventions, collateral/forecast curve separation, day-count
calendars, SABR/SVI smiles, Bermudan exercise, or full swaption-surface
calibration, those can be layered on top of these interfaces.
