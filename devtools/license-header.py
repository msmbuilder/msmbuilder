import os
import sys
import re

OLD_LICENSE_START = '''# Redistribution and use in source and binary forms'''
OLD_LICENSE_END = '''# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.'''
COPYRIGHT = '''# Author:
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

'''

NEW_LICENSE = '''# Mixtape is free software: you can redistribute it and/or modify
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
# License along with Mixtape. If not, see <http://www.gnu.org/licenses/>.'''

def replace(filename):
    with open(filename) as f:
        text = f.read()
    start = text.find(OLD_LICENSE_START)
    end = text.find(OLD_LICENSE_END)
    if text.find('Copyright (c)') == -1:
        license = COPYRIGHT + NEW_LICENSE
    else:
        license = NEW_LICENSE

    if (start != -1) and (end != -1):
        # old license found
        newtext = text[:start] + license + text[end+len(OLD_LICENSE_END):]
    else:
        newtext = license + text

    with open(filename, 'w') as f:
        f.write(newtext)

def main():
    for filename in sys.argv[1:]:
        replace(filename)
        print filename

if __name__ == '__main__':
    main()
