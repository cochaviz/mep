import sys
import subprocess
import os
import re

from nbconvert import PythonExporter
import nbformat

import scalene
import typer

def notebook_to_python(input_notebook, output_script=None, cells=None, ipython=False):
    with open(input_notebook, 'r', encoding='utf-8') as nb_file:
        nb_contents = nb_file.read()

    notebook = nbformat.reads(nb_contents, as_version=4)
    exporter = PythonExporter()

    if cells:
        notebook.cells = notebook.cells[cells[0]:cells[1]]

    python_script, _ = exporter.from_notebook_node(notebook)

    if not output_script:
        output_script = ".".join(input_notebook.split(".")[:-1]) + "_profiled.py"

    with open(output_script, 'w', encoding='utf-8') as py_file:
        if not ipython:
            pattern = re.compile(r'.*get_ipython.*\n', re.MULTILINE)
            python_script = pattern.sub('', python_script)
        py_file.write(python_script)

    return output_script

def profile(notebook, 
        cells=None, 
        scalene_args=None, 
        cleanup: bool=True
        as_ipython: bool=False
    ):
    if cells:
        try:
            start, end = map(int, cells.split("-"))
        except ValueError:
            start = int(cells)
            end = None
        finally:
            cells = (start, end)
    if scalene_args:
        scalene_args = scalene_args.split()

    conv_script = notebook_to_python(notebook, cells=cells, ipython=as_ipython)

    # fix encoding error when using cuda
    my_env = os.environ.copy()
    my_env["LC_ALL"] = "POSIX"
    subprocess.run(["scalene", *scalene_args, conv_script], env=my_env)

    if cleanup:
        os.remove(conv_script)


if __name__=="__main__":
    typer.run(profile)