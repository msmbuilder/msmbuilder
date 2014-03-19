from __future__ import print_function, absolute_import
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
