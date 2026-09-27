"""Structural and synthetic tests for the pristine inspiral observer."""

import ast
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

from tools.observe_pycbc_inspiral import _instrument, _save_evidence, _save_work
from pycbc.benchmark import _jax_chisq_mode


ROOT = Path(__file__).parents[1]
BASELINE = "40e94792b3"


def _upstream_source():
    return subprocess.check_output(
        ["git", "show", "%s:bin/pycbc_inspiral" % BASELINE],
        cwd=ROOT, text=True)


def test_upstream_shape_has_observation_boundaries():
    source = _upstream_source()
    tree = _instrument(source, "bin/pycbc_inspiral")
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "__pycbc_observer"]
    assert sorted(node.func.attr for node in calls) == ["evidence", "work"]
    compile(tree, 'pycbc_inspiral', 'exec')

    class RemoveObserver(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == '__pycbc_observer'):
                return None
            return node

    assert ast.dump(RemoveObserver().visit(tree)) == ast.dump(ast.parse(source))


def test_boundary_validation_is_fail_closed():
    source = _upstream_source().replace(
        "psd.associate_psds_to_segments(",
        "psd.associate_psds_to_segments(", 1)
    source += "\npsd.associate_psds_to_segments(opt, segments, gwstrain, flen, delta_f, flow)\n"
    with pytest.raises(RuntimeError, match="association boundary"):
        _instrument(source, "synthetic.py")

    with pytest.raises(RuntimeError, match="final event output"):
        _instrument(_upstream_source() + "\nevent_mgr.write_events(opt.output)\n",
                    "synthetic.py")
    with pytest.raises(RuntimeError, match='namespace'):
        _instrument(_upstream_source() + '\n__pycbc_observer = None\n', 'synthetic.py')


def test_writers_capture_actual_dtypes(tmp_path, monkeypatch):
    strain = SimpleNamespace(
        numpy=lambda: numpy.array([1, 2], dtype=numpy.float32),
        start_time=10, delta_t=0.5)
    segment = SimpleNamespace(
        numpy=lambda: numpy.array([1 + 2j], dtype=numpy.complex64),
        psd=SimpleNamespace(numpy=lambda: numpy.array([3.], dtype=numpy.float64)),
        delta_f=0.25, epoch=11,
        analyze=slice(2, 5), seg_slice=slice(1, 6), cumulative_index=7,
        dtype=numpy.dtype("complex64"))
    evidence = tmp_path / "science.hdf"
    work = tmp_path / "work.json"
    monkeypatch.setenv("PYCBC_BENCHMARK_EVIDENCE", str(evidence))
    monkeypatch.setenv("PYCBC_BENCHMARK_WORK", str(work))
    monkeypatch.setenv('PYCBC_BENCHMARK_SCIENCE_CONFIG', '{"sample_rate":2048}')
    _save_evidence(strain, [segment])
    _save_work([segment], 4, 2.)
    import h5py
    with h5py.File(evidence) as output:
        assert set(output.attrs) == {"science_config", "strain_epoch",
                                     "strain_delta_t"}
        assert output["conditioned_strain"].dtype == numpy.dtype("float32")
        assert output["segments/0/strain"].dtype == numpy.dtype("complex64")
        assert output["segments/0/psd"].dtype == numpy.dtype("float64")
    assert __import__("json").loads(work.read_text())["completed_templates"] == 4
    assert __import__('json').loads(work.read_text())['signal_dtypes'] == ['complex64']

    # Compare the complete schema and contents to the candidate's own writer.
    tree = ast.parse((ROOT / 'pycbc/benchmark.py').read_text())
    writer = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'save_inspiral_evidence')
    namespace = {'os': os, '_jax_chisq_mode': _jax_chisq_mode}
    exec(compile(ast.Module(body=[writer], type_ignores=[]), 'writer', 'exec'), namespace)
    candidate = tmp_path / 'candidate.hdf'
    monkeypatch.setenv('PYCBC_BENCHMARK_EVIDENCE', str(candidate))
    namespace['save_inspiral_evidence'](strain, [segment])
    from tools.benchmark_science import read_hdf, exact
    a, aa = read_hdf(evidence)
    b, ba = read_hdf(candidate)
    for left, right in ((a, b), (aa, ba)):
        assert left.keys() == right.keys()
        assert all(exact(left[key], right[key]) for key in left)
