"""Structural regression test for inference FFT backend setup."""

import ast
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "bin"
    / "inference"
    / "pycbc_inference"
)


def test_fft_backend_is_configured_inside_scheme_context():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_cli"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "fft"
    ]
    assert len(calls) == 1

    ancestor = calls[0]
    while ancestor in parents and not isinstance(ancestor, ast.With):
        ancestor = parents[ancestor]

    assert isinstance(ancestor, ast.With)
    assert len(ancestor.items) == 1
    context = ancestor.items[0].context_expr
    assert isinstance(context, ast.Name)
    assert context.id == "ctx"
