import sys
import subprocess

from nbconvert import PythonExporter
import nbformat

import scalene
import typer

APP_NAME="scalene_notebook"
app = typer.Typer()

def notebook_to_python(input_notebook, output_script=None, cells=None):
    with open(input_notebook, 'r', encoding='utf-8') as nb_file:
        nb_contents = nb_file.read()

    notebook = nbformat.reads(nb_contents, as_version=4)
    exporter = PythonExporter()

    if cells:
        notebook.cells = notebook.cells[cells[0]:cells[1]]

    python_script, _ = exporter.from_notebook_node(notebook)

    if not output_script:
        output_script = input_notebook.split(".")[:-1] + "_profiled.py"

    with open(output_script, 'w', encoding='utf-8') as py_file:
        py_file.write(python_script)

    return output_script

@command
def profile(notebook, cells=None, scalene_args=[], cleanup=True):
    if cells:
        start, end = map(int, cells.split("-"))
        cells = (start, end)

    conv_script = notebook_to_python(notebook_to_python, cells=cells)
    subprocess.run(["scalene", *scalene_args, conv_script])

    if cleanup:
        os.remove(conv_script)


if __name__=="__main__":
    app(prog_name=APP_NAME)