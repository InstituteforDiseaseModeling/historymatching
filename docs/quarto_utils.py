"""Pre- and post-render helpers for the Quarto docs build.

Run automatically by Quarto via the ``pre-render``/``post-render`` keys in
``_quarto.yml``; they can also be run by hand:

    python quarto_utils.py pre
    python quarto_utils.py post

Pre-render:

1. Write ``_variables.yml`` so the page footer shows the version being documented.
2. Run ``quartodoc build`` to regenerate ``api/*.qmd`` from the package docstrings.
3. Run ``quartodoc interlinks`` to download the ``_inv/`` inventories used to link
   references like ``pandas.DataFrame`` out to the upstream docs.
4. Convert quartodoc's ``objects.json`` into a Sphinx-compatible ``objects.inv``,
   so other projects can resolve historymatching references via intersphinx.

Post-render:

Restore ``tutorials/*.ipynb`` from the snapshot taken during pre-render. The
tutorials are committed without outputs, but Quarto executes notebooks *in place*
and writes the results (plus the local kernel version) back into the source files,
so without this step every render leaves the git tree dirty.
"""

import json
import pathlib
import shutil
import subprocess
import sys

DOCS = pathlib.Path(__file__).parent
VARIABLES = DOCS / '_variables.yml'
OBJECTS_JSON = DOCS / 'objects.json'
OBJECTS_INV = DOCS / 'objects.inv'
NOTEBOOK_DIRS = ['tutorials']
SNAPSHOT = DOCS / '.nb-snapshot'  # pristine copies of the notebooks, restored post-render


def run(*args):
    """Run a Python module, echoing the command first."""
    print(f'\n> {" ".join(args)}\n')
    subprocess.run([sys.executable, '-m', *args], check=True, cwd=DOCS)


def notebooks():
    """Every tutorial notebook, in a stable order."""
    return [p for folder in NOTEBOOK_DIRS for p in sorted((DOCS / folder).glob('*.ipynb'))]


def update_version():
    """Refresh _variables.yml from the installed package version."""
    import historymatching as hm
    VARIABLES.write_text(f"version: {hm.__version__}\nversiondate: '{hm.__versiondate__}'\n")
    print(f'Docs version set to {hm.__version__} ({hm.__versiondate__})')


def build_api_docs():
    """Regenerate the API reference pages from docstrings."""
    run('quartodoc', 'build')
    run('quartodoc', 'interlinks')


def build_objects_inv():
    """Convert quartodoc's objects.json into a Sphinx-compatible objects.inv."""
    import sphobjinv as soi

    if not OBJECTS_JSON.exists():
        print(f'No {OBJECTS_JSON.name}; skipping objects.inv')
        return

    import historymatching as hm

    data = json.loads(OBJECTS_JSON.read_text())
    inv = soi.Inventory()
    inv.project = data.get('project', 'historymatching')
    inv.version = str(data.get('version', hm.__version__))
    for item in data['items']:
        inv.objects.append(soi.DataObjStr(
            name=item['name'],
            domain=item['domain'],
            role=item['role'],
            priority=str(item.get('priority', '1')),
            uri=item['uri'],
            dispname=item.get('dispname', '-') or '-',
        ))
    OBJECTS_INV.write_bytes(soi.compress(inv.data_file()))
    print(f'Wrote {len(inv.objects)} entries to {OBJECTS_INV.name}')


def snapshot_notebooks():
    """Stash pristine copies of the tutorials before Quarto executes them."""
    if SNAPSHOT.exists():
        shutil.rmtree(SNAPSHOT)
    for path in notebooks():
        dest = SNAPSHOT / path.relative_to(DOCS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    print(f'Snapshotted {len(notebooks())} notebooks to {SNAPSHOT.name}/')


def restore_notebooks():
    """Put the pristine tutorials back, undoing Quarto's in-place execution."""
    if not SNAPSHOT.exists():
        print(f'No {SNAPSHOT.name}/ to restore from; skipping')
        return
    restored = []
    for src in sorted(SNAPSHOT.rglob('*.ipynb')):
        dest = DOCS / src.relative_to(SNAPSHOT)
        if not dest.exists() or dest.read_bytes() != src.read_bytes():
            shutil.copy2(src, dest)
            restored.append(dest.name)
    shutil.rmtree(SNAPSHOT)
    print(f'Restored: {", ".join(restored)}' if restored else 'Notebooks already pristine')


def pre_render():
    update_version()
    snapshot_notebooks()
    build_api_docs()
    build_objects_inv()


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'pre'
    if action == 'pre':
        pre_render()
    elif action == 'post':
        restore_notebooks()
    else:
        raise SystemExit(f'Unknown action {action!r}; expected "pre" or "post"')
