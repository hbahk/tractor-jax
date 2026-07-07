# Sphinx configuration for tractor-jax documentation.

from tractor_jax.version import __version__

project = "tractor-jax"
author = "Hyeonguk Bahk"
copyright = "2026, Hyeonguk Bahk"
version = str(__version__)
release = str(__version__)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

# -- napoleon (NumPy-style docstrings) ---------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

# -- intersphinx --------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "photutils": ("https://photutils.readthedocs.io/en/stable/", None),
}

# -- MyST ---------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "dollarmath"]

# -- HTML output ---------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"tractor-jax v{release}"
html_theme_options = {
    "github_url": "https://github.com/hbahk/tractor-jax",
    "show_toc_level": 2,
}
html_static_path = ["_static"]
