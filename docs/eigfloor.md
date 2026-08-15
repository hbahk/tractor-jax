# Understanding `eigfloor`

One sentence: **eigenvalues measure how much the data know about a direction in
flux space, solving means dividing by them, and dividing by a direction the data
know nothing about turns noise into an answer. The floor is the instruction to
leave those directions alone.**

This page derives that from the normal equations, shows why the Jacobi
normalization turns the floor into a pure correlation threshold, and is honest
about what the error bars do and do not contain. {doc}`solvers` is the short
version — which estimator to pick; {doc}`worked_example` runs the numbers quoted
here on an actual scene.

## 1. It is only a system of linear equations

Forced photometry gives every source $j$ a **template** $T_j$ — the image it
would make at unit flux, at its fixed position and shape — and models the data
as $\mathbf{m} = \sum_j f_j T_j$. With inverse-variance weights $w_i$,

$$\chi^2(\mathbf{f}) = \sum_i w_i \Big( d_i - \sum_j f_j T_{ji} \Big)^2$$

is quadratic in $\mathbf{f}$, so setting its gradient to zero gives
$G\mathbf{f} = \mathbf{b}$ with

$$G_{jk} = \sum_i w_i T_{ji} T_{ki} = (A^\top W A)_{jk}, \qquad
b_j = \sum_i w_i T_{ji} d_i .$$

The two kinds of entry mean different things:

- $G_{jj} = \lVert T_j \rVert_W^2$ — how strongly source $j$'s template is
  *gripped* by the data. It does not depend on how bright the source is, only on
  the PSF size and the depth of the exposure. An isolated source's error bar is
  $\sigma_j = 1/\sqrt{G_{jj}}$.
- $G_{jk}$ — the overlap of templates $j$ and $k$. Zero if they do not touch.

If $G$ were diagonal, every source could be solved on its own and there would be
nothing to discuss. **Every difficulty lives in the off-diagonal entries — that
is, in blends.**

## 2. Eigenvalues are directional information

$G$ is symmetric positive semi-definite, so it has orthogonal eigenvectors,
$G v_k = \lambda_k v_k$. An eigenvector is a *direction in flux space*: a
combination of source fluxes, for example "raise source 1 and lower source 2".
Moving along it costs

$$\chi^2(\mathbf{f} + \epsilon v_k) - \chi^2(\mathbf{f}) = \lambda_k \epsilon^2 ,$$

so $\lambda_k$ *is* the stiffness of the data along that direction. Large
$\lambda$: the data pin that combination down. $\lambda \approx 0$: the data have
no opinion about it at all. Since $\Delta\chi^2 = 1$ marks $1\sigma$, the error
along mode $k$ is $\sigma_k = 1/\sqrt{\lambda_k}$ — $G$ is the Fisher information
matrix and its eigenvalues are the information per direction.

Solving means dividing by them:

$$\mathbf{f} = G^{-1}\mathbf{b} = \sum_k \frac{v_k^\top \mathbf{b}}{\lambda_k}\, v_k .$$

With $\mathbf{d} = A\mathbf{f}_\text{true} + \mathbf{n}$ and
$\mathrm{Cov}(\mathbf{n}) = W^{-1}$ we get $\mathrm{Cov}(\mathbf{b}) = G$, so the
noise on $v_k^\top \mathbf{b}$ is $\sqrt{\lambda_k}$ and the noise on the
*answer* is $\sqrt{\lambda_k}/\lambda_k = 1/\sqrt{\lambda_k}$. At
$\lambda = 10^{-3}$ that is a factor 32; at $10^{-6}$, a factor 1000. **A single
small eigenvalue can wreck the whole solution vector**, which is why a plain
least-squares catalog can have a perfectly healthy median residual and a long
disaster tail.

Blends produce exactly those modes. Push $+\epsilon$ into source 1 and
$-\epsilon$ into source 2:

$$\Delta \mathbf{m} = \epsilon\,(T_1 - T_2), \qquad
\Delta\chi^2 = \epsilon^2 \lVert T_1 - T_2 \rVert_W^2 .$$

If the two sit almost on top of each other, $T_1 \approx T_2$, the cost is
almost nothing and $\lambda \approx 0$. **For a fully overlapping pair the data
measure the sum and not the split.** Ordinary least squares still reports a
split — it manufactures one out of the noise by dividing by $\lambda \approx 0$.

## 3. Jacobi normalization: the coordinates are S/N

"$\lambda$ is small" is meaningless as written, because $\lambda$ carries units:
it scales with flux units, exposure time and the number of pixels. A threshold
like "$\lambda < 0.01$" would mean something different in every image.

The fix is to divide each source by its own error bar. With
$D_j = \sqrt{G_{jj}} = 1/\sigma_j$, the normalized coordinate is

$$\hat f_j = D_j f_j = f_j / \sigma_j = \textbf{the source's S/N} .$$

That is the sentence that makes everything else fall out. In these coordinates
the Gram becomes a **correlation matrix**

$$\hat G = D^{-1} G D^{-1}, \qquad
\hat G_{jj} = 1, \qquad \hat G_{jk} = \rho_{jk} \in [0, 1]$$

($\rho_{jk} \ge 0$ because PSF templates are non-negative). Its trace is $n$, so
the mean eigenvalue is exactly 1, an isolated source *is* $\lambda = 1$, and
$0 \le \lambda \le n$ regardless of units. An eigenvalue now reads as
"information relative to one isolated source", and **a floor placed on the
spectrum of $\hat G$ acts in significance space, not in flux space.**

Skipping the normalization is not a small mistake. On the raw $G$ the largest
eigenvalue belongs to whichever column has the biggest norm — usually the
constant background column, whose self-weight is $\sim n_\text{pix} w$ against a
source's $\sim A_\text{eff} w$. For a 1000-pixel cutout and a 10-pixel effective
PSF area that is $\lambda_\max / \lambda_\text{source} \sim 100$, so
$\texttt{floor}\cdot\lambda_\max$ lands right on top of the source eigenvalues
and halves *every* flux; on larger cutouts it goes to $-99\%$. Because $G_{jj}$
is independent of brightness, the fractional bias is the same at all fluxes and
therefore looks enormous in units of $\sigma$ at the bright end — which is
exactly the $-50$ to $-99\%$ high-S/N bias measured before the normalization was
added.

## 4. The floor is a filter factor

Take the eigendecomposition of the normalized Gram, $\hat G = V \Lambda V^\top$,
set $\lambda_f = \texttt{floor}\cdot\lambda_\max$, and replace $\Lambda$ by
$\max(\Lambda, \lambda_f)$ before inverting. Written per mode, the estimator is
the OLS answer scaled mode by mode,

$$\hat{\mathbf{f}} = \sum_k \varphi_k \frac{v_k^\top \hat{\mathbf{b}}}{\lambda_k} v_k,
\qquad \boxed{\varphi_k = \min\!\left(1, \frac{\lambda_k}{\lambda_f}\right)}$$

and that is the entire method. Modes with $\lambda_k \ge \lambda_f$ get
$\varphi = 1$ *exactly* — bit-identical to no regularization at all. Modes far
below the floor get pulled toward zero.

$\varphi_k$ is the standard **filter factor** of regularization theory, and the
estimators differ only in the shape of that one curve:

| method | $\varphi_k$ | consequence |
|---|---|---|
| OLS (`linear`) | $1$ always | noise explodes as $\lambda \to 0$: the disaster tail |
| TSVD | $1$ or $0$ | discontinuous; an arbitrary hard cut |
| scalar ridge | $\lambda_k/(\lambda_k+\tau)$ | never reaches 1 — shrinks *well-measured* sources too |
| `eigfloor` | $\min(1, \lambda_k/\lambda_f)$ | transparent above the floor, smooth below |
| `lasso` | not a filter | selection: it sets sources to exactly 0 |

`eigfloor` is the hinge: TSVD's transparency where the data are informative,
Tikhonov's continuity where they are not. It is not a new regularizer, it is a
choice of filter factor.

The scalar-ridge row is not hypothetical. With $\tau = 10^{-2}$ a well-measured
mode at $\lambda \approx 1$ gets $\varphi = 1/1.01 = 0.990$, a 1% shrinkage; in
S/N coordinates the bias is $(1-\varphi)\times \mathrm{S/N}$, so at a S/N of 22
that is $0.0099 \times 22 = 0.22\sigma$ — the $-0.22\sigma$ bias measured for a
scalar ridge at S/N 10–30, straight out of the formula. `eigfloor` gives
$\varphi = 1$ exactly on the same mode, hence $|\text{median}\,\chi| \le 0.1\sigma$.
Two measurements, one equation, opposite ends.

:::{admonition} The same thing said as a prior
:class: note

Adding $\tau_k$ to mode $k$ is equivalent to
$\chi^2_\text{eff} = \chi^2 + \tau_k (v_k^\top \hat{\mathbf{f}})^2$, i.e. a
Gaussian prior $\mathcal{N}(0, 1/\tau_k)$ on that flux combination. Because the
coordinates are S/N, the prior width is in sigmas. With
$\texttt{floor} = 10^{-2}$ and $\lambda_\max \approx 2$, a deeply degenerate mode
gets $\tau \approx \lambda_f \approx 0.02$, i.e.
$\sigma_\text{prior} = 1/\sqrt{0.02} \approx 7$: *"the S/N difference between two
sources that overlap completely is probably within $\pm 7$."* Every mode with
$\lambda \ge \lambda_f$ gets $\tau = 0$ — a flat prior. `eigfloor` is the MAP
estimator that puts a weak prior on the unmeasured directions **only**.
:::

## 5. A blend of two, end to end

For a pair, $\hat G = \begin{pmatrix} 1 & \rho \\ \rho & 1\end{pmatrix}$ and the
eigenvectors are fixed by symmetry:

| mode | eigenvector | $\lambda$ | floored? | result |
|---|---|---|---|---|
| sum | $(1,1)/\sqrt2$ | $1+\rho \to 2$ | never | total flux untouched |
| split | $(1,-1)/\sqrt2$ | $1-\rho \to 0$ | yes | only the split is shrunk |

The floor engages on the split mode when $1 - \rho < \texttt{floor}\,(1+\rho)$,
that is

$$\rho > \frac{1 - \texttt{floor}}{1 + \texttt{floor}}
\;\;\xrightarrow{\;\texttt{floor}=10^{-2}\;}\;\; \rho > 0.980 .$$

**That is the physical meaning of `eig_floor=1e-2`: only pairs whose templates
overlap by more than 98% are touched at all.** It is an operating point with a
statement attached, not a knob.

A worked case. Take $\rho = 0.999$ and true S/N $(100, 10)$. Then
$\lambda_+ = 1.999$, $\lambda_- = 0.001$, $\lambda_f = 0.02$, so
$\varphi_+ = 1$ and $\varphi_- = 0.05$. The sum component
$(100+10)/\sqrt2 = 77.8$ passes through untouched; the split component
$(100-10)/\sqrt2 = 63.6$ is multiplied by $0.05$ to $3.2$. Rotating back gives
$(57.2,\ 52.8)$: the S/N *difference* has been shrunk 20-fold and the total is
preserved. The estimator is saying *"the two of them together are 110σ, and I
cannot tell you whether the split is 100:10 or 57:53"* — which is precisely what
the data know.

On the real scene of the {doc}`worked_example` the numbers are milder and the
mechanism identical: $\rho = 0.9893$, so $\lambda_+ = 1.9893$,
$\lambda_- = 0.0107$, $\lambda_f = 0.0199$ and $\varphi_- = 0.537$. Decomposing
the actual solver output into the two modes gives a sum component ratio of
$1.0001$ and a split component ratio of $0.5333$ against the predicted
$\varphi_- = 0.5366$.

## 6. Error bars, honestly

Three separate things are easy to conflate.

**What is reported.** `return_variances=True` returns
$\mathrm{diag}(\hat G_f^{-1})$, the diagonal of the inverse *floored* Gram. That
is the **posterior** variance under the prior of §4, and it correctly blows up in
the degenerate directions: in the worked example each member of the blend gets a
reported $\sigma$ that is $6.86\times$ (`linear`) or $5.02\times$ (`eigfloor`)
what the same source would get if it were isolated — and $6.86$ is exactly
$1/\sqrt{1-\rho^2}$ at $\rho = 0.9893$.

**What actually scatters.** The sampling covariance of the estimator is the
sandwich $\hat G_f^{-1}\hat G \hat G_f^{-1}$, which in mode $k$ is
$\lambda_k/\lambda_{f,k}^2 = \varphi_k \cdot (1/\lambda_{f,k})$. So

$$\text{true variance} = \text{reported variance} \times \varphi_k :$$

on floored modes the reported $\sigma$ is **conservative** by $1/\sqrt{\varphi_k}$.
Measured on the worked example over 120 noise realizations: the blend's fainter
member has reported $\sigma = 0.472$ against an actual scatter of $0.342$, a
ratio of $1.38$, against $1/\sqrt{0.537} = 1.37$ predicted. On the isolated
sources the ratio is $1.06$ — calibrated, as it must be, since $\varphi = 1$
there.

**What is *not* in the error bar: the shrinkage bias.** Take the $\rho = 0.999$
example again. The reported variance of source 1 is
$0.5/1.999 + 0.5/0.02 = 25.25$, i.e. $\sigma \approx 5.0$ in S/N units, so the
estimate $57.2$ against a truth of $100$ sits at

$$\chi_1 = (57.2 - 100)/5.0 \approx -8.5\sigma .$$

The variance grew *and* the pull is still $-8.5\sigma$, because
$\mathrm{diag}(\hat G_f^{-1})$ encodes "I do not know the split" but not "I have
pulled the split toward zero". On the worked example the same effect is a benign
$+0.66\sigma$ at $\varphi_- = 0.54$.

This has a structural consequence. $\lambda_f$ is derived from
$\lambda_\max(\hat G)$ — from *the data's own* spectrum — so the prior is
data-dependent and there is no frequentist coverage guarantee to be had from the
formula. That is the reason a survey pipeline built on `eigfloor` needs an
empirical, per-S/N error calibration downstream. It is a derived conclusion, not
an ad-hoc patch — and a mild one: measured end-to-end on SPHEREx production
photometry, the per-S/N factors come out at $c \approx 0.8$–$1.05$, i.e. the
reported errors were already nearly honest, erring conservative exactly where the
sandwich says they should.

## 7. Two caveats worth stating

**$\lambda_\max$ is global, so the floor couples to neighbourhood crowding.**
For an isolated pair $\lambda_\max = 1+\rho \le 2$ and the engagement threshold
is $\rho > 0.980$. Drop one tight triple into the same image and
$\lambda_\max \approx 3$, which loosens the threshold for *every* pair in that
image to $\rho > 0.970$. This is the price of scale invariance. In practice it is
harmless — $\lambda_f \ll 1$ either way, so well-measured modes are still exactly
transparent — but it is a real coupling and worth knowing about when comparing
fields of different crowding.

**Blend shrinkage can be chromatic.** Across a multi-band or spectral data set,
if $\rho_{ij}$ is the same in every channel then $\varphi_-$ is the same in every
channel: the shrinkage is achromatic, the individual fluxes of a blend are wrong
but its *colours* survive, and photo-$z$ tolerates it. When the PSF varies with
wavelength $\rho_{ij}(\lambda)$ varies too, the shrinkage becomes
wavelength-dependent, and a blended pair can carry a residual colour term. The
diagnostic is the residual slope $d\langle r\rangle/d\ln\lambda$ of blended
versus isolated targets. On SPHEREx production photometry this differential came
out **null** — $-0.2 \pm 1.5\,\%/\ln\lambda$, i.e. achromatic to
$<3.2\,\%/\ln\lambda$ (95%) even for sub-pixel pairs — because the PSF's
chromaticity is weak (FWHM $\propto \lambda^{0.13}$), so the expected effect sits
below the measurement floor. The same test convicted the alternative: *cutting*
faint neighbours instead of modelling them absorbs their light into the targets
with a strongly one-signed chromatic slope ($+1.3$ to $+13\,\%/\ln\lambda$,
brighter to fainter). Selection destroys colours; shrinkage does not. For an
instrument with a strongly chromatic PSF the null must be re-derived, not
assumed.

## Next steps

- {doc}`solvers` — the estimator menu and when to use each.
- {doc}`worked_example` — every number on this page, on a scene you can run.
