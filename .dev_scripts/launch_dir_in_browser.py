#!/usr/bin/env python

import os
import sys
import webbrowser

try:
    from urllib import pathname2url
except:  # noqa E722
    from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
