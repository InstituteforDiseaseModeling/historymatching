#!/bin/bash
nosetests --with-coverage --cover-package=hm2 --cover-html --cover-html-dir=test_coverage -s .
