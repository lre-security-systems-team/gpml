#!/usr/bin/env python

import glob
import os

"""
Generate the code documentation, using pydoc.
"""

for dirname in ['gpml']:
    flist = glob.glob('%s/*/*.py' % dirname)
    for fname in flist:
        if '__init__' not in fname:
            try: #Windows
                split = str.split(os.path.splitext(fname)[0],'\\')

                bname = str(split[0]) + '.' + str(split[1]) + '.' + split[2]  # eg 'smartbox'
                os.system('pydoc -w %s' % bname)
                os.system('pydoc3 -w %s' % str(split[0]) + '.' + str(split[1]))
                # os.system('mv %s.html %s.%s.html' % (bname, dirname, bname))
            except: #Linux
                os.system('pydoc3 -w %s' % fname)
                bname= os.path.splitext(os.path.basename(fname))[0]  # eg 'smartbox'
                split = str.split(os.path.splitext(fname)[0],'/')
                lname = str(split[0]) + '.' + str(split[1]) + '.' + split[2]
                os.system('mv %s.html %s.html' % (bname, lname))
                os.system('pydoc3 -w %s' % str(split[0]) + '.' + str(split[1]))
    os.system('pydoc3 -w %s' % dirname)
os.system('mv *.html doc')
