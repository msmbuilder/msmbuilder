import os

from sphinx.util.compat import Directive
from docutils import nodes
from docutils.parsers.rst import directives
from IPython.nbconvert import HTMLExporter, PythonExporter
from IPython.nbformat.current import read, write

from runipy.notebook_runner import NotebookRunner


class NotebookDirective(Directive):
    """Insert an evaluated notebook into a document

    This uses runipy and nbconvert to transform a path
    to an unevaluated notebook into html suitable for embedding
    in a Sphinx document.

    It also adds a python version of the script as well as the
    evaluated and unevaluated notebooks.
    """
    required_arguments = 1
    optional_arguments = 1
    option_spec = {'skip_exceptions': directives.flag}

    def run(self):
        # Check if raw html is supported
        if not self.state.document.settings.raw_enabled:
            raise self.warning('"%s" directive disabled.' % self.name)
        # Get optional argument as boolean
        skip_exceptions = 'skip_exceptions' in self.options

        # Get path to notebook
        nb_filename = os.path.join(self.arguments[0])
        nb_path = os.path.join('..', nb_filename)

        # Load notebook
        with open(nb_path) as nb_file:
            notebook = read(nb_file, 'json')

        # Make output filenames and directory
        python_fn = nb_filename.replace('.ipynb', '.py')
        python_fn = os.path.join('_build', python_fn)
        uneval_fn = os.path.join('_build', nb_filename)
        eval_fn = nb_filename.replace('.ipynb', '_evaluated.ipynb')
        eval_fn = os.path.join('_build', eval_fn)
        try:
            os.makedirs(os.path.dirname(uneval_fn))
        except OSError:
            pass

        # Save python script
        python = nb_to_python(notebook)
        with open(python_fn, 'w') as python_f:
            python_f.write(python)

        # Save unevaluated Notebook
        with open(uneval_fn, 'w') as uneval_f:
            write(notebook, uneval_f, 'json')

        # Run notebook
        runner = NotebookRunner(notebook)
        runner.run_notebook(skip_exceptions=skip_exceptions)

        # Save evaluated notebook
        with open(eval_fn, 'w') as eval_f:
            write(runner.nb, eval_f, 'json')

        # Add links
        rst_file = self.state_machine.document.attributes['source']
        link_rst = ' '.join(['(',
                            formatted_link(nb_filename),
                            formatted_link(eval_fn),
                            formatted_link(python_fn),
                            ')'])
        self.state_machine.insert_input([link_rst], rst_file)

        # Create html notebook node
        html = nb_to_html(runner.nb)
        nb_node = notebook_node('', html, format='html')

        # add dependency
        self.state.document.settings.record_dependencies.add(nb_path)

        return [nb_node]


class notebook_node(nodes.raw):
    pass


def nb_to_python(notebook):
    """Convert notebook to python script"""
    exporter = PythonExporter()
    output, resources = exporter.from_notebook_node(notebook)
    return output


def nb_to_html(notebook):
    """Convert a notebook to html.

    nbconvert either produces "basic" output which is not styled
    enough or "full" output which conflicts with the sphinx css.
    """
    exporter = HTMLExporter(template_file='full')
    output, resources = exporter.from_notebook_node(notebook)

    # Get <head> and <body> sections
    header = output.split('<head>', 1)[1].split('</head>', 1)[0]
    body = output.split('<body>', 1)[1].split('</body>', 1)[0]

    # http://imgur.com/eR9bMRH
    header = header.replace('<style', '<style scoped="scoped"')
    header = header.replace(
        'body {\n  overflow: visible;\n  padding: 8px;\n}\n', '')

    # Filter out styles that conflict with the sphinx theme.
    filter_strings = [
        'navbar',
        'body{',
        'alert{',
        'uneditable-input{',
        'collapse{',
    ]
    filter_strings.extend(['h%s{' % (i + 1) for i in range(6)])

    header_lines = filter(lambda x: not any([s in x for s in filter_strings]),
                          header.split('\n'))
    header = '\n'.join(header_lines)

    # concatenate raw html lines
    lines = ['<div class="ipynotebook">', header, body, '</div>']
    return '\n'.join(lines)


def formatted_link(path):
    return "`{fn} <{fn}>`_".format(fn=os.path.basename(path))


def visit_notebook_node(self, node):
    self.visit_raw(node)


def depart_notebook_node(self, node):
    self.depart_raw(node)


def setup(app):
    app.add_node(notebook_node, html=(visit_notebook_node,
                                      depart_notebook_node))

    app.add_directive('notebook', NotebookDirective)

