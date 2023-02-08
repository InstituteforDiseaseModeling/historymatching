#!/usr/bin/env python

# This script should be executed from the root directory of the repo
import glob
import webbrowser
import argparse
import os
from livereload import Server, shell  # noqa: I900

PACKAGE_NAME = 'history_matching'

parser = argparse.ArgumentParser("Serves and autobuilds the docs")
parser.add_argument('--port', type=int, default=8000, help="Port to serve docs on.")
args = parser.parse_args()

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(root_dir)
server = Server()
dirs_to_watch = set()
# Find all the directories within docs to watch
for filepath in glob.glob("**/*.rst", recursive=True):
    if f"{PACKAGE_NAME}_" not in filepath:
        d = os.path.dirname(filepath)
        if d not in dirs_to_watch:
            server.watch(f'{d}/*.rst', shell('make html', cwd='docs'), delay=5, ignore=lambda x: f"{PACKAGE_NAME}_" in x)
            dirs_to_watch.add(d)
server.watch('docs/__static/*', shell('make html', cwd='docs'), delay=5)
server.watch('docs/images/*', shell('make html', cwd='docs'), delay=5)
server.watch('docs/*.txt', shell('make html', cwd='docs'), delay=5)
server.watch('*.py', shell('make html', cwd='docs'), delay=5)
webbrowser.open("http://localhost:8000")
server.serve(root='docs/_build/html', port=args.port)
