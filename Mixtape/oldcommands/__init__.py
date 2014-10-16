# Author:  Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

# Mixtape is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Mixtape. If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division, absolute_import
import os as _os

_exclude = ['__init__.py']
__all__ = []

for _fn in _os.listdir(_os.path.dirname(_os.path.abspath(__file__))):
    if _fn in _exclude or _fn.endswith('.pyc'):
        continue
    _name = 'mixtape.commands.' + _fn.split('.py')[0]
    _module = __import__(_name, globals(), locals())
    _items = getattr(_module, '__all__', [])
    __all__.extend(_items)
    for _item in _items:
        exec('%s = _module.%s' % (_item, _item), globals(), locals())
