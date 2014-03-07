'''Plot the contents of a .jsonlines file
'''
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

from __future__ import print_function, division

import re
import sys
import os
import json
import webbrowser
import numpy as np
import pandas as pd
from flask import Flask, send_from_directory, request

from mixtape.utils import iterobjects
from mixtape.cmdline import Command, argument, FlagAction

__all__ = ['PlotGHMM']

class PlotGHMM(Command):
    name = 'plot-ghmm'
    description = __doc__
    filename = argument('filename', metavar='JSONLINES_FILE',
                     help='Path to .jsonlines file')
    open_browser = argument('--open-browser', action=FlagAction, default=True,
         help='''Open the browser''')
    port = argument('--port', default=8080, type=int, help='Port to serve on. default=8080')

    def __init__(self, args):
        self.open_browser = args.open_browser
        self.port = args.port
        self.models = list(iterobjects(args.filename))
        self.models_df = pd.DataFrame(self.models)
    
    def start(self):
        # index_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
        root = '/home/rmcgibbo/projects/mixtape/Mixtape/commands/templates/'


        app = Flask(__name__)

        @app.route("/")
        def index():
            return send_from_directory(root, 'index.html')

        @app.route('/novus-nvd3-0c03b27/<path:path>')
        def static_proxy(path):
            return send_from_directory(os.path.join(root, 'novus-nvd3-0c03b27'), path)

        @app.route('/models')
        def models():
            data = []
            group = request.args.get('groupby')
            for n, items in self.models_df.groupby(group):
                items = [e.to_dict() for _, e in items.iterrows()]
#                for item in items:
#                    item['size'] = np.random.rand()
                data.append({'key': '%s=%s' % (group, n), 'values': items})
            return json.dumps(data)

        # webbrowser.open('http://127.0.0.1:%d/index.html' % self.port)
        app.run(port=self.port, debug=True)



