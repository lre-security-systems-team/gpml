#!/usr/bin/env python

import glob
import os

"""
Generate the code documentation, using pydoc.
"""

for dirname in ['gpml']:
    flist = glob.glob('%s/*.py' % dirname)
    for fname in flist:
        if '__init__' not in fname:
            os.system('pydoc -w %s' % fname)
            bname = os.path.splitext(os.path.basename(fname))[0]  # eg 'smartbox'
            os.system('mv %s.html %s.%s.html' % (bname, dirname, bname))
    os.system('pydoc -w %s' % dirname)
os.system('mv *.html doc')
