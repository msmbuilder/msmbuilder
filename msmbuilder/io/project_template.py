# Author: Matthew Harrigan <matthew.harrigan@outlook.com>
# Contributors:
# Copyright (c) 2016, Stanford University
# All rights reserved.


import os
import re
from datetime import datetime

import yaml
from jinja2 import Environment, PackageLoader

from .io import backup, chmod_plus_x


def get_layout():
    """Specify a hierarchy of our templates."""
    tica_msm = TemplateDir(
        'tica',
        [
            'tica/tica.py',
            'tica/tica-plot.py',
            'tica/tica-sample-coordinate.py',
            'tica/tica-sample-coordinate-plot.py',
        ],
        [
            TemplateDir(
                'cluster',
                [
                    'cluster/cluster.py',
                    'cluster/cluster-plot.py',
                    'cluster/sample-clusters.py',
                    'cluster/sample-clusters-plot.py',
                ],
                [
                    TemplateDir(
                        'msm',
                        [
                            'msm/timescales.py',
                            'msm/timescales-plot.py',
                            'msm/microstate.py',
                            'msm/microstate-plot.py',
                            'msm/microstate-traj.py',
                        ],
                        [],
                    )
                ]
            )
        ]
    )
    layout = TemplateDir(
        '',
        [
            '0-test-install.py',
            '1-get-example-data.py',
            'README.md',
        ],
        [
            TemplateDir(
                'analysis',
                [
                    'analysis/gather-metadata.py',
                    'analysis/gather-metadata-plot.py',
                ],
                [
                    TemplateDir(
                        'rmsd',
                        [
                            'rmsd/rmsd.py',
                            'rmsd/rmsd-plot.py',
                        ],
                        [],
                    ),
                    TemplateDir(
                        'landmarks',
                        [
                            'landmarks/find-landmarks.py',
                            'landmarks/featurize.py',
                            'landmarks/featurize-plot.py',
                        ],
                        [tica_msm],
                    ),
                    TemplateDir(
                        'dihedrals',
                        [
                            'dihedrals/featurize.py',
                            'dihedrals/featurize-plot.py',
                        ],
                        [tica_msm],
                    )
                ]
            )
        ]
    )
    return layout


class TemplateProject(object):
    """A class to be used for wrapping on the command line.

    Parameters
    ----------
    root : str
        Start from this directory. The default ("") means start from the top.
        Include all sub-steps.
    step : str
        If specified, generate templates for this step and then stop. Do not
        include sub steps.
    ipynb : bool
        Render scripts as IPython/Jupyter notebooks instead.
    display : bool
        Render scripts assuming a connected display and xdg-open.
    """

    def __init__(self, root='', step=None, ipynb=False, display=False):
        if step is not None:
            find_root = step
            limit = 1
        else:
            find_root = root
            limit = None
        self.layout = get_layout().find(find_root, limit)
        if self.layout is None:
            raise ValueError("Could not find TemplateDir named {}".format(root))

        self.template_kwargs = {
            'ipynb': ipynb,
            'use_agg': (not display) and (not ipynb),
            'use_xdgopen': display and (not ipynb),
        }
        self.template_dir_kwargs = {}

    def do(self):
        """Render the templates"""
        self.layout.render(self.template_dir_kwargs, self.template_kwargs)


class MetadataPackageLoader(PackageLoader):
    meta = {}

    def get_source(self, environment, template):
        source, filename, uptodate = super(MetadataPackageLoader, self) \
            .get_source(environment, template)

        beg_str = "Meta\n----\n"
        end_str = "\n\"\"\"\n"
        beg = source.find(beg_str)
        if beg == -1:
            self.meta[filename] = {}
            return source, filename, uptodate

        end = source[beg:].find(end_str) + beg

        self.meta[filename] = yaml.load(source[beg + len(beg_str):end])
        remove_meta = source[:beg] + source[end:]
        return remove_meta, filename, uptodate


ENV = Environment(
    loader=MetadataPackageLoader('msmbuilder', 'project_templates'),
    line_statement_prefix="# ?"
)


class Template(object):
    """Render a template file

    Parameters
    ----------
    template_fn : str
        Template filename.
    """

    def __init__(self, template_fn):
        self.template_fn = template_fn
        self.template = ENV.get_template(template_fn)
        self.meta = ENV.loader.meta[self.template.filename]

    def get_write_function(self, ipynb):
        fext = self.template_fn.split('.')[-1]
        if fext == 'py':
            if ipynb:
                return self.write_ipython
            else:
                return self.write_python
        elif fext == 'sh':
            return self.write_shell
        else:
            return self.write_generic

    def get_header(self):
        return '\n'.join([
            "msmbuilder autogenerated template version 2",
            'created {}'.format(datetime.now().isoformat()),
            "please cite msmbuilder in any publications"
        ])

    def write_ipython(self, templ_fn, rendered):
        import nbformat
        from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
        templ_ipynb_fn = templ_fn.replace('.py', '.ipynb')

        cell_texts = [templ_ipynb_fn] + re.split(r'## (.*)\n', rendered)
        cells = []
        for heading, content in zip(cell_texts[:-1:2], cell_texts[1::2]):
            cells += [new_markdown_cell("## " + heading.strip()),
                      new_code_cell(content.strip())]
        nb = new_notebook(
            cells=cells,
            metadata={'kernelspec': {
                'name': 'python3',
                'display_name': 'Python 3'
            }})
        backup(templ_ipynb_fn)
        with open(templ_ipynb_fn, 'w') as f:
            nbformat.write(nb, f)

    def write_python(self, templ_fn, rendered):
        backup(templ_fn)
        with open(templ_fn, 'w') as f:
            f.write(rendered)

    def write_shell(self, templ_fn, rendered):
        backup(templ_fn)
        with open(templ_fn, 'w') as f:
            f.write(rendered)
        chmod_plus_x(templ_fn)

    def write_generic(self, templ_fn, rendered):
        backup(templ_fn)
        with open(templ_fn, 'w') as f:
            f.write(rendered)

    def render(self, ipynb, use_agg, use_xdgopen):
        rendered = self.template.render(
            header=self.get_header(),
            date=datetime.now().isoformat(),
            ipynb=ipynb,
            use_agg=use_agg,
            use_xdgopen=use_xdgopen,
        )
        write_func = self.get_write_function(ipynb)
        write_func(os.path.basename(self.template_fn), rendered)


class TemplateDir(object):
    """Represents a template directory and manages dependency symlinks

    Templates can specify "dependencies", i.e. files from parent
    directories that are required. This class handles creating symlinks
    to those files.
    """

    def __init__(self, name, files, subdirs):
        self.name = name
        self.files = files
        self.subdirs = subdirs

    def render_files(self, template_kwargs):
        depends = set()
        for fn in self.files:
            templ = Template(fn)
            if 'depends' in templ.meta:
                depends.update(templ.meta['depends'])
            templ.render(**template_kwargs)
        return depends

    def render(self, template_dir_kwargs, template_kwargs):
        depends = self.render_files(template_kwargs)
        for dep in depends:
            bn = os.path.basename(dep)
            if not os.path.exists(bn):
                os.symlink("../{}".format(dep), bn)
        for subdir in self.subdirs:
            backup(subdir.name)
            os.mkdir(subdir.name)
            pwd = os.path.abspath('.')
            os.chdir(subdir.name)
            subdir.render(template_dir_kwargs, template_kwargs)
            os.chdir(pwd)

    def find(self, name, limit=None):
        """Find the named TemplateDir in the hierarchy"""
        if name == self.name:
            if limit is not None:
                assert limit == 1
                self.subdirs = []
            return self
        for subdir in self.subdirs:
            res = subdir.find(name, limit)
            if res is not None:
                return res
        return None
