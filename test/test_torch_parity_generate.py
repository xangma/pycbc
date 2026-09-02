import importlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATE = ROOT / "tools" / "torch_parity" / "generate.py"


def _load_generate():
    spec = importlib.util.spec_from_file_location(
        "torch_parity_generate", GENERATE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lalsimulation_blocker_records_caught_import_attempts(tmp_path):
    generate = _load_generate()
    record_path = tmp_path / "attempts.json"
    blocker = generate._BlockLalsimulation(record_path)

    for module_name in ("lalsimulation", "lalsimulation.gwsignal"):
        try:
            blocker.find_spec(module_name)
        except ModuleNotFoundError:
            pass

    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["attempt_count"] == 2
    assert [event["module"] for event in payload["attempts"]] == [
        "lalsimulation",
        "lalsimulation.gwsignal",
    ]
    assert all(event["stack"] for event in payload["attempts"])


def test_lalsimulation_blocker_ignores_other_imports(tmp_path):
    generate = _load_generate()
    record_path = tmp_path / "attempts.json"
    blocker = generate._BlockLalsimulation(record_path)

    assert blocker.find_spec("lal") is None
    assert blocker.attempts == []
    assert not record_path.exists()


def test_caught_attempt_still_fails_torch_cell(monkeypatch, tmp_path):
    generate = _load_generate()
    record_path = tmp_path / "attempts.json"
    blocker = generate._BlockLalsimulation(record_path)
    for module_name in tuple(sys.modules):
        if module_name == "lalsimulation" or module_name.startswith(
            "lalsimulation."
        ):
            monkeypatch.delitem(sys.modules, module_name)

    sys.meta_path.insert(0, blocker)
    try:
        try:
            importlib.import_module("lalsimulation")
        except ModuleNotFoundError:
            pass
    finally:
        sys.meta_path.remove(blocker)

    assert blocker.attempts[0]["module"] == "lalsimulation"
    with pytest.raises(RuntimeError, match="forbidden lalsimulation"):
        generate._enforce_lalsimulation_gate(blocker)


def test_unblocked_and_clean_cells_pass_gate(tmp_path):
    generate = _load_generate()
    clean_blocker = generate._BlockLalsimulation(tmp_path / "attempts.json")

    generate._enforce_lalsimulation_gate(None)
    generate._enforce_lalsimulation_gate(clean_blocker)
