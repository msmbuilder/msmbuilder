# Author: Kyle A. Beauchamp <kyleabeauchamp@gmail.com>
# Contributors: Robert McGibbon <rmcgibbo@gmail.com>
# Copyright (c) 2014, Stanford University and the Authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#   Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function, division, absolute_import

import glob
import itertools
import numpy as np
import mdtraj as md
import msmbuilder as msmb

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class Directory(object):
    def __init__(self, pdb_filename, trj_path, extension="h5", stride=1, chunk=1000):
        """A container for a directory of Trajectory objects.

        Parameters
        ----------
        pdb_filename : string
            Path to a single PDB object that will provide topology information
        trj_path : string
            Path to a directory containing trajectory objects
        stride : int, default=1, optional
            Only read every stride-th frame.        
        chunk : int, default=1000, optional
            Break trajectories into chunks of this size.
        """
        self.trj0 = md.load(pdb_filename)
        self.trj_path = trj_path
        self.filenames = sorted(glob.glob(trj_path + "/*.%s" % extension), key=msmb.utils.keynat)
        self._raw_trajectories = None
        self.stride = stride
        self.chunk = chunk        
        
    def _cache(self):
        """Load first chunks of trajectories into memory."""  # Probably better ways of handling this logic, but the current layout is pretty readable IMHO.
        
        if self.chunk == 0 and self._raw_trajectories is None:  # Load the actual trajectories, once.
            self._raw_trajectories = [md.load(filename, stride=self.stride) for filename in self.filenames]
        
        if self.chunk == 0:
            self.trajectories = itertools.chain(([trj] for trj in self._raw_trajectories))
        else:
            self.trajectories = [md.iterload(filename, stride=self.stride, chunk=self.chunk) for filename in self.filenames]

    def __iter__(self):
        self._cache()
        return self
    
    def next(self):  # Need both forms of next for py2 py3 compatibility
        return self.__next__()
    
    def __next__(self):
        for trj_chunks in self.trajectories:
            for chunk in trj_chunks:
                return chunk
        raise StopIteration
