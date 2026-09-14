import os
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SENTRA'
copyright = '2026, Barkhausen Institute gGmbH, Dresden'
author = 'Trustworthy Data Processing Group, Barkhausen Institute gGmbH, Dresden'
release = 'V0.00.002'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinxcontrib.plantuml',
    'autoapi'
]

#master_doc = "sphinx-doc-generation/index"

if os.name == 'nt':
    plantuml = 'java -jar "C:\\Program Files\\PlantUML\\plantuml-1.2024.7.jar"'

plantuml_output_format="svg_img"
plantuml_latex_output_format="pdf"
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',
                    '*.md']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css',
]

autoapi_type = "python"
autoapi_dirs = [
    "../../sentra-node/python/node/ml_training",
    "../../sentra-node/python/client/sentra_client",
]
autoapi_root = "_autoapi"
autoapi_add_toctree_entry = False

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown'
}

def generate_index_in_root_directory(app, exception):
    index_doc_path = os.path.join(app.outdir, 'index.html')
    with open(index_doc_path, "w") as index_file:
        index_file.write('<meta http-equiv="refresh" content="0; url=sphinx-doc-generation/index.html">')


#def setup(app):
#    app.connect('build-finished', generate_index_in_root_directory)

