# Installation

## Requirements

- Python ≥ 3.9
- [JAX](https://docs.jax.dev/) — the default `pip install jax` gives a
  CPU-only build; for GPU acceleration install the CUDA wheel, e.g.
  `pip install "jax[cuda12]"`.
- numpy, scipy, astropy, photutils (installed automatically)

## From source

```bash
git clone https://github.com/hbahk/tractor-jax.git
pip install -e tractor-jax
```

For development (tests, plotting):

```bash
pip install -e "tractor-jax[dev]"
```

To build this documentation locally:

```bash
pip install -e "tractor-jax[docs]"
sphinx-build -b html tractor-jax/docs tractor-jax/docs/_build/html
```

## Verifying the install

```python
import tractor_jax
print(tractor_jax.__version__)

import jax
print(jax.devices())  # shows GPU devices if a CUDA build is installed
```
