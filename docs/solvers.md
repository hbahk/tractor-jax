# Choosing a solver

Forced photometry is linear in flux: with source positions and shapes fixed, the
model is $\mathbf{m} = A\mathbf{f}$, where each column of $A$ is one source's
unit-flux template. Every solver here minimizes the same weighted residual

$$\chi^2(\mathbf{f}) = (\mathbf{d} - A\mathbf{f})^\top W (\mathbf{d} - A\mathbf{f}),
\qquad W = \mathrm{diag}(\text{invvar}),$$

and differs only in **how it regularizes the normal equations**
$A^\top W A\,\mathbf{f} = A^\top W \mathbf{d}$. That choice matters because in a
crowded field $A^\top W A$ is ill-conditioned: the data constrain a blend's
*total* flux tightly but its *split* barely at all.

Pick with `optimize_fluxes(..., solver=...)`, or call the solver directly (they
are all designed to be `vmap`-ed over images).

## `linear` — plain weighted least squares

$$(A^\top W A + \mathrm{diag}(\rho_j))\,\mathbf{f} = A^\top W \mathbf{d},
\qquad \rho_j = \texttt{rcond}\cdot (A^\top W A)_{jj}$$

The ridge is **Jacobi-scaled**: each source's regularization depends only on its
own template norm, so the solution is invariant to masked padding and to how many
other sources happen to be co-fit. `rcond` defaults to $10^{-12}$ — numerically
stabilizing, statistically negligible.

Use it for sparse fields and shallow catalogs. In a crowded field it is
*unbiased but high-variance*: nothing stops the degenerate flux split from
wandering, and the reported variances honestly say so (they blow up).

```python
optimize_fluxes(tractor, solver="linear", return_variances=True)
```

## `eigfloor` — eigenvalue floor on the correlation matrix

Solves in Jacobi-normalized coordinates $\beta_j = \sqrt{(A^\top W A)_{jj}}\,f_j$,
where the Gram becomes a **correlation matrix** with unit diagonal,

$$\hat G = D^{-1} (A^\top W A) D^{-1}, \qquad D = \mathrm{diag}\big(\sqrt{(A^\top W A)_{jj}}\big),$$

then clamps its spectrum from below at $\texttt{floor}\cdot\lambda_\max(\hat G)$
before inverting (Tikhonov in the eigenbasis).

The normalization is essential, not cosmetic. On the **raw** $A^\top W A$ the
largest eigenvalue is dominated by whichever column has the largest norm —
usually the constant background column ($\sim n_\text{pix}\,w$) or a bright
galaxy — so `floor * λ_max` would exceed most source eigenvalues and shrink
*every* flux toward zero (measured at $-50$ to $-99\%$ bias at high S/N before
normalization). After normalization $\lambda_\max(\hat G) \le n$ regardless of
units, background, or depth, and `floor` is a pure correlation-degeneracy
threshold.

The result: well-constrained sources (eigenvalue $\sim 1$) are solved *exactly*,
while only the genuinely degenerate directions — the anti-correlated flux splits
of blended groups — get damped. It is **symmetric and sign-free**: no
non-negativity clip, no selection, so faint sources keep their negative
excursions and carry no rectification or selection-conditioning bias.

```python
optimize_fluxes(tractor, solver="eigfloor", eig_floor=1e-2)
```

This is the recommended default for **blind** photometry — measuring everything
in a catalog, including sources far below detection, with calibrated errors.

## `eigfloor_prior` — eigfloor plus Gaussian flux priors

$$(A^\top W A + \Lambda)\,\mathbf{f} = A^\top W \mathbf{d} + \Lambda \mathbf{f}_\text{prior},
\qquad \Lambda = \mathrm{diag}(1/\sigma_{\text{prior},j}^2)$$

with the eigenvalue floor applied to the *regularized* normalized Gram (i.e.
after adding $\Lambda$). Per source:

- $\lambda_j = 0$ → **protected**: exactly `eigfloor`, unbiased. With
  $\Lambda = 0$ everywhere the output is bit-identical to `solve_fluxes_eigfloor`.
- $\lambda_j > 0$ → **nuisance**: ridged toward an externally predicted flux
  $f_{\text{prior},j}$ (typically $\sigma_\text{prior} \sim 0.5\text{–}1 \times
  f_\text{prior}$, e.g. from a multi-band SED fit).

Use it when you must fit a full catalog but have external information about the
faint sources you are not reporting. It costs essentially nothing on the GPU —
the prior terms ride along in the same solve.

:::{note}
The prior arrays are per-source and depend on your catalog, so this solver is
driven through the batched interface rather than a single `optimize_fluxes`
keyword — build $\Lambda$ and $\mathbf{f}_\text{prior}$ with
{func}`~tractor_jax.jax.batching.prior_arrays_from_slots` and pass them to a
solver from {func}`~tractor_jax.jax.batching.make_batched_solver`, or call
{func}`~tractor_jax.jax.optimizer.solve_fluxes_eigfloor_prior` directly.
:::

## `lasso` — L1-regularized with selection and debiasing

$$\min_{\mathbf{f}} \; \tfrac{1}{2}(\mathbf{d}-A\mathbf{f})^\top W (\mathbf{d}-A\mathbf{f})
\; + \; \alpha \sum_j w_j |f_j|$$

solved with FISTA, non-negative by default, followed by a **debiasing refit** on
the selected support. The penalty is in **S/N units**: `alpha="auto"` uses the
universal-threshold rule $\alpha = \sqrt{2\ln p}$ per solve, which is unitless
and depth-invariant. Sources with `weights == 0` are **protected** — never
penalized, always in the support.

`debias_signfree` controls which coordinates skip the non-negativity clip during
the refit: `"none"` (default), `"protected"`, or `"all"`. Setting it to
`"protected"` lets a faint protected target keep a negative flux, which removes
the rectification bias for reported targets while keeping selection for the
nuisances.

```python
optimize_fluxes(tractor, solver="lasso",
                penalty={"alpha": "auto", "nonneg": True},
                debias=True, debias_signfree="protected")
```

Use it for **targeted** photometry of specific bright sources in crowded fields.
Do not use it blind: selection zeroes the faint end, and the posteriors of a
selected-then-refit estimator under-cover.

## `cg` — Newton-CG

Matrix-free conjugate gradients with a Fisher-diagonal preconditioner
(`use_preconditioner`, `precond_eps`). It never forms $A^\top W A$, so it is the
fallback when the number of free fluxes per image is too large for a dense
$p \times p$ solve. For the source counts the dense solvers handle comfortably,
prefer them — they are faster and give exact variances.

## Side by side

The {doc}`worked_example` fits one scene with three of these solvers. The
summary of what it finds:

| situation | `linear` | `eigfloor` | `lasso` |
|---|---|---|---|
| isolated, well-measured source | identical | identical | identical |
| degenerate blend (0.4 px apart) | split wanders | split damped | split wanders |
| the blend's **total** flux | correct | correct | correct |
| faint source (S/N ≈ 0.4) | signed noise | signed noise | **zeroed** |

When the problem is well-conditioned, all of these agree exactly — regularizers
only act where the data are ambiguous. Choose based on what you want to happen in
the ambiguous cases.

## Practical notes

- **Precision.** Fluxes and especially variances are calibration-grade only in
  float64. Enable it before any array is created:
  `jax.config.update("jax_enable_x64", True)` (or
  `JaxOptimizer(enable_x64=True)`). Expect roughly 3.5× the runtime of float32 on
  a data-center GPU — see {doc}`performance`.
- **Variances.** `return_variances=True` gives $\mathrm{diag}$ of the inverse
  regularized normal matrix. Dead slots — an all-zero template from shape
  padding or a fully-masked source — are pinned to flux 0 with infinite variance,
  never NaN.
- **Backgrounds.** `fit_background=True` adds a constant column per image, solved
  jointly and reported alongside the fluxes.
