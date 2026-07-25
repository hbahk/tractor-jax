# Installation

## Requirements

- Python ≥ 3.11
- [JAX](https://docs.jax.dev/) — `pip install jax` gives a CPU-only build; for
  GPU acceleration install a CUDA wheel, e.g. `pip install "jax[cuda12]"`.
- numpy, scipy, astropy, photutils (installed automatically)

No C extensions are built and `astrometry.net` is not required.

## From source

```bash
git clone https://github.com/hbahk/tractor-jax.git
pip install -e tractor-jax
```

For development (tests, plotting, the worked example):

```bash
pip install -e "tractor-jax[dev]"
```

To build this documentation locally:

```bash
pip install -e "tractor-jax[docs]"
sphinx-build -b html tractor-jax/docs tractor-jax/docs/_build/html
```

## GPU or CPU?

CPU is the right choice for development, testing, and running the
{doc}`worked_example` — everything works, in seconds, on a laptop.

It is **not** a production target. The engine's speed comes from a GPU +
batching co-design, and on CPU the same code is slower than the classic
Tractor. If you have no GPU, use [The Tractor](https://github.com/dstndstn/tractor)
for production runs; see {doc}`performance` for the measured comparison.

## Verifying the install

```python
import tractor_jax
print(tractor_jax.__version__)

import jax
print(jax.devices())   # shows GPU devices if a CUDA build is installed
```

Then run the test suite, which is CPU-only and takes about a minute:

```bash
cd tractor-jax
JAX_PLATFORMS=cpu pytest tests/ -q
```

## Sharing a GPU

JAX preallocates most of the card's memory by default. On a shared machine:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
```

## Precision

JAX defaults to float32, which is also the validated production default here —
fluxes *and* their reported errors are calibrated in fp32 ({doc}`performance`).

Enable float64 when you need exactness under padding/batching choices or are
working on faint, near-degenerate fluxes. It must be set **before any array is
created**:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Budget ~2.5× the runtime on a card whose FP64 rate is heavily reduced (e.g. an
L40S); A100/H100-class cards largely remove the penalty.
