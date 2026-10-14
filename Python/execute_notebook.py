"""Execute the portfolio notebook with this interpreter, without a global kernel."""

import os
import sys

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

from .analysis import ROOT


def execute():
    path = ROOT / "notebooks" / "01_titanic_python.ipynb"
    notebook = nbformat.read(path, as_version=4)
    connection = ROOT / "outputs" / "python" / ".notebook-kernel.json"
    connection.parent.mkdir(parents=True, exist_ok=True)
    manager = KernelManager(kernel_name="python3", connection_file=str(connection))
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
    ]
    # Windows kernel launch must inherit the project root, not a user's notebook directory.
    client = NotebookClient(
        notebook, timeout=600, km=manager,
        resources={"metadata": {"path": str(ROOT)}},
    )
    try:
        client.execute(cwd=str(ROOT), env={**os.environ, "MPLBACKEND": "Agg"})
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
        connection.unlink(missing_ok=True)
    nbformat.write(notebook, path)
    print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} cells: {path}")


if __name__ == "__main__":
    execute()
