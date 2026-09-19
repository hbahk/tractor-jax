# Sphinx configuration for TractorJAX documentation.

from tractor_jax import __version__

project = "TractorJAX"
author = "Hyeonguk Bahk"
copyright = "2026, Hyeonguk Bahk"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
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
html_theme = "shibuya"
html_title = f"TractorJAX v{release}"
html_theme_options = {
    "github_url": "https://github.com/hbahk/tractor-jax",
    "accent_color": "lime",
    "globaltoc_expand_depth": 1,
    "toctree_collapse": False,
    # The artwork's cube seams are near-white, which glares on a dark
    # background; the dark variant swaps them for a dark seam.
    "light_logo": "_static/tractorjax-logo.svg",
    "dark_logo": "_static/tractorjax-logo-dark.svg",
    "nav_links": [
        {"title": "Quickstart", "url": "quickstart"},
        {"title": "API", "url": "api"},
        {"title": "SPHEREx layer", "url": "https://tractorjax-spherex.readthedocs.io/", "external": True},
    ],
}
html_favicon = "_static/favicon.svg"
html_static_path = ["_static"]
# custom.css: show only the light or the dark wordmark on the index page
# (Shibuya has no `only-light` / `only-dark` rule of its own).
html_css_files = ["custom.css"]

html_extra_path = ["googlee20a25095441ea75.html"]