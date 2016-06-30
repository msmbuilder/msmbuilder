# Author: Matthew Harrigan <matthew.harrigan@outlook.com>
# Contributors:
# Copyright (c) 2016, Stanford University
# All rights reserved.


import os
import re
from collections import defaultdict
from datetime import datetime

import nbformat
import yaml
from jinja2 import Environment, PackageLoader
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from .io import backup, chmod_plus_x


def get_layout():
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
                    'tica/cluster/cluster.py',
                    'tica/cluster/cluster-plot.py',
                    'tica/cluster/sample-clusters.py',
                    'tica/cluster/sample-clusters-plot.py',
                ],
                [
                    TemplateDir(
                        'msm',
                        [
                            'tica/cluster/msm/msm-1-timescales.py',
                            'tica/cluster/msm/msm-1-timescales-plot.py',
                            'tica/cluster/msm/msm-2-microstate.py',
                            'tica/cluster/msm/msm-2-microstate-plot.py',
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
                            'analysis/rmsd/rmsd.py',
                            'analysis/rmsd/rmsd-plot.py',
                        ],
                        [],
                    ),
                    TemplateDir(
                        'landmarks',
                        [
                            'analysis/landmarks/featurize.py',
                            'analysis/landmarks/featurize-plot.py',
                        ],
                        [tica_msm],
                    ),
                    TemplateDir(
                        'dihedrals',
                        [
                            'analysis/dihedrals/featurize.py',
                            'analysis/dihedrals/featurize-plot.py',
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
    ipynb : bool
        Render scripts as IPython/Jupyter notebooks instead
    display : bool
        Render scripts assuming a connected display and xdg-open
    """

    def __init__(self, root='', ipynb=False, display=False):
        self.layout = get_layout().find(root)
        if self.layout is None:
            raise ValueError("Could not find TemplateDir named {}".format(root))

        self.template_kwargs = {
            'ipynb': ipynb,
            'use_agg': (not display) and (not ipynb),
            'use_xdgopen': display and (not ipynb),
        }
        self.template_dir_kwargs = {}

    def do(self):
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

    def find(self, name):
        """Find the named TemplateDir in the heirarchy"""
        if name == self.name:
            return self
        for subdir in self.subdirs:
            res = subdir.find(name)
            if res is not None:
                return res
        return None
