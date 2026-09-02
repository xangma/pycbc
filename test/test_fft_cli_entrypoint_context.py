"""Source contracts for FFT setup in scheme-aware command-line tools."""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    ("bin/pycbc_inspiral", "ctx", 2),
    ("bin/pycbc_live", "ctx", 1),
    ("bin/inference/pycbc_inference", "ctx", 1),
    ("bin/pycbc_optimize_snr", "scheme_context", 1),
)
WISDOM_ENTRYPOINTS = (
    (
        "bin/pycbc_inspiral",
        "ctx",
        ("strain_segments.fourier_segments", "MatchedFilterControl"),
    ),
    (
        "bin/pycbc_live",
        "ctx",
        ("LiveBatchMatchedFilter", "StrainBuffer.from_cli"),
    ),
)


def _parse(relative_path):
    path = ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _calls(tree, name):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _dotted_name(node.func) == name
    ]


def _parent_map(tree):
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _ancestors(node, parents):
    while node in parents:
        node = parents[node]
        yield node


def _with_uses_name(node, name):
    return isinstance(node, ast.With) and any(
        isinstance(item.context_expr, ast.Name)
        and item.context_expr.id == name
        for item in node.items
    )


def _assigns_scheme_context(node, name):
    return (
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and _dotted_name(node.value.func) == "scheme.from_cli"
    )


def _mentions_fft_backends(node):
    return any(
        isinstance(child, ast.Attribute) and child.attr == "fft_backends"
        for child in ast.walk(node)
    )


def _is_rank_one_test(node):
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return False
    operands = (node.left, *node.comparators)
    return (
        isinstance(node.ops[0], ast.Eq)
        and any(_dotted_name(item) == "evnt.rank" for item in operands)
        and any(
            isinstance(item, ast.Constant) and item.value == 1
            for item in operands
        )
    )


@pytest.mark.parametrize(
    ("relative_path", "context_name", "expected_calls"), ENTRYPOINTS
)
def test_fft_from_cli_runs_inside_selected_scheme(
    relative_path, context_name, expected_calls
):
    tree = _parse(relative_path)
    parents = _parent_map(tree)
    calls = _calls(tree, "fft.from_cli")
    assignments = [
        node
        for node in ast.walk(tree)
        if _assigns_scheme_context(node, context_name)
    ]

    assert len(assignments) == 1
    assert len(calls) == expected_calls
    inside_calls = []
    for call in calls:
        contexts = [
            node
            for node in _ancestors(call, parents)
            if _with_uses_name(node, context_name)
        ]
        if contexts:
            assert len(contexts) == 1
            inside_calls.append(call)
            assert assignments[0].lineno < contexts[0].lineno <= call.lineno
    assert len(inside_calls) == 1

    if relative_path == "bin/pycbc_inspiral":
        outside_calls = [call for call in calls if call not in inside_calls]
        strain_setup = _calls(tree, "strain.from_cli")
        assert len(outside_calls) == len(strain_setup) == 1
        assert outside_calls[0].lineno < strain_setup[0].lineno


@pytest.mark.parametrize(
    ("relative_path", "context_name", "work_calls"), WISDOM_ENTRYPOINTS
)
def test_wisdom_lifecycle_is_ordered_and_backend_independent(
    relative_path, context_name, work_calls
):
    tree = _parse(relative_path)
    parents = _parent_map(tree)
    configure = [
        call
        for call in _calls(tree, "fft.from_cli")
        if any(
            _with_uses_name(node, context_name)
            for node in _ancestors(call, parents)
        )
    ]
    imports = _calls(tree, "fft.import_wisdom_from_cli")
    exports = _calls(tree, "fft.export_wisdom_from_cli")

    assert len(configure) == len(imports) == len(exports) == 1
    assert configure[0].lineno < imports[0].lineno
    active_contexts = [
        node
        for node in _ancestors(configure[0], parents)
        if _with_uses_name(node, context_name)
    ]
    assert len(active_contexts) == 1
    assert active_contexts[0] in set(_ancestors(imports[0], parents))
    for work_call in work_calls:
        calls = _calls(tree, work_call)
        assert calls, f"missing plan/work marker {work_call}"
        assert imports[0].lineno < min(call.lineno for call in calls)
        assert max(call.lineno for call in calls) < exports[0].lineno
        assert all(
            active_contexts[0] in set(_ancestors(call, parents))
            for call in calls
        )

    if relative_path == "bin/pycbc_live":
        # Configuration failures must propagate as command failures instead of
        # being caught by the optional-wisdom/planning error handler.
        assert not any(
            isinstance(node, ast.Try)
            for node in _ancestors(configure[0], parents)
        )

        wisdom_tries = [
            node
            for node in _ancestors(imports[0], parents)
            if isinstance(node, ast.Try)
        ]
        assert len(wisdom_tries) == 1
        nonzero_exits = [
            node
            for handler in wisdom_tries[0].handlers
            for node in ast.walk(handler)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and _dotted_name(node.exc.func) == "SystemExit"
            and len(node.exc.args) == 1
            and isinstance(node.exc.args[0], ast.Constant)
            and node.exc.args[0].value == 1
        ]
        assert len(nonzero_exits) == 1

        export_guards = [
            node
            for node in _ancestors(exports[0], parents)
            if isinstance(node, ast.If) and _is_rank_one_test(node.test)
        ]
        assert len(export_guards) == 1

    wisdom_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and "wisdom" in (_dotted_name(node.func) or "")
    ]
    assert {
        _dotted_name(node.func) for node in wisdom_calls
    } == {
        "fft.import_wisdom_from_cli",
        "fft.export_wisdom_from_cli",
    }

    for call in imports + exports:
        assert not any(
            isinstance(node, ast.If) and _mentions_fft_backends(node.test)
            for node in _ancestors(call, parents)
        )

    list_string_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and _mentions_fft_backends(node)
        and any(
            isinstance(child, ast.Constant) and child.value == "fftw"
            for child in ast.walk(node)
        )
    ]
    assert list_string_guards == []
