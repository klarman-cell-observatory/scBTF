# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import pathlib
import sys

print(pathlib.Path(__file__).parents[2].resolve().as_posix())
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())


# -- Project information -----------------------------------------------------

project = 'scbtf'
copyright = '2023, Broad Institute'
author = 'Daniel Chafamo'

# The full version, including alpha/beta/rc tags
release = '0.1.5'


# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = "4.3"  # Nicer param docs

extensions = [
    'myst_parser',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'IPython.sphinxext.ipython_console_highlighting',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx_design'
]

# nbsphinx specific settings
source_suffix = [".rst", ".md"]

exclude_patterns = ['_build', '**.ipynb_checkpoints']
nbsphinx_execute = "never"

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]


# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = "bysource"
bibtex_reference_style = "author_year"
napoleon_google_docstring = True  # for pytorch lightning
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
typehints_defaults = "braces"
todo_include_todos = False
numpydoc_show_class_members = False
annotate_defaults = True  # scanpydoc option, look into why we need this
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]

# The master toctree document.
master_doc = "index"

# intersphinx_mapping = {
#     "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
#     "ipython": ("https://ipython.readthedocs.io/en/stable/", None),
#     "matplotlib": ("https://matplotlib.org/", None),
#     "numpy": ("https://numpy.org/doc/stable/", None),
#     "pandas": ("https://pandas.pydata.org/docs/", None),
#     "python": ("https://docs.python.org/3", None),
#     "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
#     "sklearn": ("https://scikit-learn.org/stable/", None),
#     "torch": ("https://pytorch.org/docs/master/", None),
#     "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
#     "pyro": ("http://docs.pyro.ai/en/stable/", None),
# }

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"
pygments_dark_style = "native"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo' # furo

# Set link name generated in the top bar.
html_title = "scBTF"
html_logo = "_static/logo.png"

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#003262",
        "color-brand-content": "#003262",
        "admonition-font-size": "var(--font-size-normal)",
        "admonition-title-font-size": "var(--font-size-normal)",
        "code-font-size": "var(--font-size--small)",
    },
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["override.css"]
html_show_sphinx = False

# -- Nbsphinx prolog -------------------------------------------

