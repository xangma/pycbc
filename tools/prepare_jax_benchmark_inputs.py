#!/usr/bin/env python3
"""Prepare the pinned input banks used by the JAX benchmark campaigns.

The expensive conversion and waveform-compression commands are deliberately
run only from :func:`main`.  In particular, importing this module never
downloads data or invokes a reference checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Sequence
from urllib.request import urlretrieve

import h5py
import numpy as np


DEFAULT_REFERENCE_REVISION = "40e94792b3edf59f39b18b65102b28a4f74433a7"
DEFAULT_BANK_URL = (
    "https://raw.githubusercontent.com/gwastro/pycbc-config/"
    "710dbfd3590bd93d7679d7822da59fcb6b6fac0f/O2/bank/"
    "H1L1-HYPERBANK_SEOBNRv4v2_VARFLOW_THORNE-1163174417-604800.xml.gz"
)
DEFAULT_SIZES = (32, 1536, 3072, 6144)
SELECTION_SEED = 20260919
PHYSICAL_FIELDS = ("mass1", "mass2", "spin1z", "spin2z")
FIXED_THREAD_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
REFERENCE_RUNTIME_DEPENDENCIES = (
    "pycbc", "numpy", "h5py", "scipy", "lalsuite", "igwn-ligolw",
)
_RUNTIME_PROBE = r'''
import importlib.metadata
import json
import platform
import sys

names = %r
dependencies = {}
for name in names:
    try:
        dependencies[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        dependencies[name] = None
installed = {}
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    if name:
        installed[name] = distribution.version
print(json.dumps({
    "python": {
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    },
    "platform": platform.platform(),
    "dependencies": dependencies,
    "installed_distributions": dict(sorted(installed.items(), key=lambda item: item[0].lower())),
}))
''' % (REFERENCE_RUNTIME_DEPENDENCIES,)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not exist or is not a regular file: {path}")
    return path


def _verify_hash(path: Path, expected: str, label: str) -> str:
    expected = expected.lower()
    _validate_hash_string(expected, label)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _validate_hash_string(value: str, label: str) -> str:
    value = value.lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256")
    return value


def _resolve_python(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    located = shutil.which(value)
    if located:
        return str(Path(located).resolve())
    raise ValueError(f"Python executable not found: {value}")


def _reference_runtime(python: str) -> dict:
    """Capture provenance from the exact interpreter used for reference tools."""
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        result = subprocess.run(
            [python, "-c", _RUNTIME_PROBE], check=True, capture_output=True,
            text=True, env=environment,
        )
        runtime = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not capture runtime from {python}: {exc}") from exc
    if not isinstance(runtime, dict) or not runtime.get("python"):
        raise ValueError(f"runtime probe returned invalid metadata from {python}")
    return runtime


def _selection_runtime() -> dict:
    """Capture the interpreter and libraries performing deterministic selection."""
    return {
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": platform.platform(aliased=True, terse=True),
        "dependencies": {"numpy": np.__version__, "h5py": h5py.__version__},
    }


def _git_output(reference_source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(reference_source), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_reference(reference_source: Path, revision: str) -> dict:
    """Require a clean checkout at exactly *revision*."""
    reference_source = reference_source.expanduser().resolve()
    if not reference_source.is_dir():
        raise ValueError(f"reference source is not a directory: {reference_source}")
    try:
        head = _git_output(reference_source, "rev-parse", "HEAD")
        dirty = _git_output(reference_source, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"could not inspect reference checkout {reference_source}: {exc}") from exc
    if head != revision:
        raise ValueError(
            f"reference checkout revision mismatch: expected {revision}, got {head}"
        )
    if dirty:
        raise ValueError(f"reference checkout is not clean: {reference_source}")
    for relative in ("bin/bank/pycbc_coinc_bank2hdf", "bin/pycbc_compress_bank"):
        if not (reference_source / relative).is_file():
            raise ValueError(f"reference checkout is missing {relative}")
    return {"source": str(reference_source), "revision": head, "clean": True}


def _physical_key(values: Sequence[object], index: int) -> tuple:
    key = []
    for values_for_field in values:
        value = np.asarray(values_for_field[index]).item()
        if isinstance(value, np.generic):
            value = value.item()
        key.append(value)
    return tuple(key)


def select_rows(bank_path: Path, sizes: Iterable[int], seed: int = SELECTION_SEED) -> dict:
    """Select nested, deterministic, physically unique source rows.

    The mass and aligned-spin cuts intentionally match the retained campaign
    preparation script.  Rows are sorted only when written to each subset;
    the random order itself is recorded through those source row indices.
    """
    sizes = tuple(sizes)
    maximum = max(sizes)
    with h5py.File(bank_path, "r") as source:
        missing = [name for name in PHYSICAL_FIELDS if name not in source]
        if missing:
            raise ValueError(f"converted bank is missing physical fields: {', '.join(missing)}")
        arrays = [np.asarray(source[name][:]) for name in PHYSICAL_FIELDS]
        count = len(arrays[0])
        if any(len(array) != count for array in arrays[1:]):
            raise ValueError("converted bank physical fields have different row counts")
        mass1, mass2, spin1z, spin2z = arrays
        chirp = (mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2
        eligible = np.flatnonzero(
            (mass1 <= 10)
            & (mass2 <= 3)
            & (chirp >= 1.2)
            & (np.abs(spin1z) <= 0.2)
            & (np.abs(spin2z) <= 0.2)
        )
        rng = np.random.default_rng(seed)
        chosen = []
        seen = set()
        for index in rng.permutation(eligible):
            index = int(index)
            key = _physical_key(arrays, index)
            if key in seen:
                continue
            seen.add(key)
            chosen.append(index)
            if len(chosen) == maximum:
                break
        if len(chosen) < maximum:
            raise ValueError(
                f"insufficient distinct bank rows after cuts: need {maximum}, "
                f"found {len(chosen)} (eligible rows: {len(eligible)})"
            )
    return {str(size): sorted(chosen[:size]) for size in sizes}


def _copy_rows(source_path: Path, output_path: Path, indices: Sequence[int]) -> None:
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "x") as output:
        for name, value in source.items():
            if isinstance(value, h5py.Dataset):
                output.create_dataset(name, data=value[indices])
            else:
                source.copy(name, output)
        for key, value in source.attrs.items():
            output.attrs[key] = value


def _validate_compressed_output(path: Path, expected_rows: int) -> None:
    """Reject compressor outputs whose workers silently skipped templates."""
    with h5py.File(path, "r") as compressed:
        if "template_hash" not in compressed:
            raise ValueError(f"compressed bank is missing template_hash: {path}")
        actual_rows = len(compressed["template_hash"])
        if actual_rows != expected_rows:
            raise ValueError(
                f"compressed bank has {actual_rows} template rows; "
                f"expected {expected_rows}: {path}"
            )
        waveforms = compressed.get("compressed_waveforms")
        if waveforms is None or len(waveforms) != expected_rows:
            actual_waveforms = 0 if waveforms is None else len(waveforms)
            raise ValueError(
                f"compressed bank stores {actual_waveforms} waveforms; "
                f"expected {expected_rows}: {path}"
            )


def _command_lists(
    python: str, reference_source: Path, bank_xml: Path, full_bank: Path,
    output_dir: Path, sizes: Sequence[int],
) -> tuple[list[str], dict[str, list[str]]]:
    conversion = [
        python,
        str(reference_source / "bin/bank/pycbc_coinc_bank2hdf"),
        "--bank-file", str(bank_xml),
        "--output-file", str(full_bank),
    ]
    compression = {}
    for size in sizes:
        compression[str(size)] = [
            python,
            str(reference_source / "bin/pycbc_compress_bank"),
            "--bank-file", str(output_dir / f"o2-subset-{size}.hdf"),
            "--output", str(output_dir / f"o2-compressed-{size}.hdf"),
            "--approximant", "IMRPhenomD",
            "--psd-model", "aLIGOZeroDetHighPower",
            "--sample-rate", "2048",
            "--segment-length", "512",
            "--low-frequency-cutoff", "30",
            "--precision", "single",
            "--interpolation", "inline_linear",
            "--compression-algorithm", "mchirp",
            "--nprocesses", "1",
            "--tolerance", "0.001",
        ]
    return conversion, compression


def _reference_environment(reference_source: Path) -> dict[str, str]:
    """Return the small, recorded environment used by reference tools."""
    environment = dict(FIXED_THREAD_ENVIRONMENT)
    environment["PYTHONPATH"] = str(reference_source)
    return environment


def _download_bank(destination: Path, url: str) -> Path:
    if destination.exists():
        return _require_file(destination, "downloaded bank XML")
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, str(destination))
    return _require_file(destination, "downloaded bank XML")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used for the pristine reference tools")
    parser.add_argument("--reference-source", required=True, type=Path)
    parser.add_argument("--reference-revision", default=DEFAULT_REFERENCE_REVISION)
    parser.add_argument("--frame-file", required=True, type=Path)
    parser.add_argument("--frame-sha256", required=True)
    parser.add_argument("--bank-xml", type=Path,
                        help="Existing XML bank; omitted banks are downloaded from the pinned URL")
    parser.add_argument("--bank-xml-sha256", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--plan", action="store_true",
                        help="Print commands without downloading or running anything")
    return parser


def prepare(args: argparse.Namespace) -> dict:
    """Prepare inputs from parsed arguments and return the written manifest."""
    output_dir = args.output_dir.expanduser().resolve()
    reference_root = args.reference_source.expanduser().resolve()
    try:
        output_dir.relative_to(reference_root)
    except ValueError:
        pass
    else:
        raise ValueError("output directory must be outside the reference checkout")
    sizes = tuple(args.sizes)
    if not sizes or any(size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("sizes must be positive and unique")
    if tuple(sorted(sizes)) != sizes:
        raise ValueError("sizes must be in ascending order")
    python = _resolve_python(args.python)

    frame = _require_file(args.frame_file, "frame file")
    frame_hash = _verify_hash(frame, args.frame_sha256, "frame file")
    reference = validate_reference(reference_root, args.reference_revision)

    output_paths = [output_dir / f"o2-subset-{size}.hdf" for size in sizes]
    output_paths.extend(output_dir / f"o2-compressed-{size}.hdf" for size in sizes)
    output_paths.append(output_dir / "manifest.json")
    if not args.plan:
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError("refusing to overwrite existing outputs: " + ", ".join(existing))

    if args.bank_xml is None:
        bank_xml = output_dir / Path(DEFAULT_BANK_URL).name
        if not args.plan:
            bank_xml = _download_bank(bank_xml, DEFAULT_BANK_URL)
    else:
        bank_xml = _require_file(args.bank_xml, "bank XML")
    if args.plan and not bank_xml.is_file():
        bank_hash = _validate_hash_string(args.bank_xml_sha256, "bank XML")
    else:
        bank_hash = _verify_hash(bank_xml, args.bank_xml_sha256, "bank XML")

    full_bank = output_dir / ".o2-full.hdf" if args.plan else None
    temporary_full = None
    if not args.plan:
        reference_runtime = _reference_runtime(python)
        selection_runtime = _selection_runtime()
        descriptor, temporary_full = tempfile.mkstemp(prefix=".o2-full-", suffix=".hdf", dir=output_dir)
        os.close(descriptor)
        os.unlink(temporary_full)
        full_bank = Path(temporary_full)
    conversion, compression = _command_lists(
        python, reference_root, bank_xml,
        full_bank, output_dir, sizes,
    )
    if args.plan:
        return {"plan": True, "commands": {"conversion": conversion, "compression": compression},
                "environment": _reference_environment(reference_root)}

    environment = _reference_environment(reference_root)
    try:
        subprocess.run(conversion, check=True, cwd=str(reference_root), env={**os.environ, **environment})
        validate_reference(reference_root, args.reference_revision)
        full_bank = _require_file(full_bank, "converted bank")
        row_indices = select_rows(full_bank, sizes)
        banks = {}
        for size in sizes:
            subset = output_dir / f"o2-subset-{size}.hdf"
            compressed = output_dir / f"o2-compressed-{size}.hdf"
            _copy_rows(full_bank, subset, row_indices[str(size)])
            subprocess.run(compression[str(size)], check=True, cwd=str(reference_root),
                           env={**os.environ, **environment})
            validate_reference(reference_root, args.reference_revision)
            _require_file(compressed, "compressed bank")
            _validate_compressed_output(compressed, len(row_indices[str(size)]))
            banks[str(size)] = {
                "subset_path": str(subset),
                "compressed_path": str(compressed),
                "source_rows": row_indices[str(size)],
                "subset_sha256": sha256(subset),
                "compressed_sha256": sha256(compressed),
                "conversion_command": conversion,
                "compression_command": compression[str(size)],
            }
        manifest = {
            "schema_version": 1,
            "kind": "jax_benchmark_inputs",
            "reproducibility": "Generated HDF bytes are recorded; byte-for-byte reproduction is not guaranteed.",
            "reference": reference,
            "reference_source": reference["source"],
            "reference_revision": reference["revision"],
            "runtime": reference_runtime,
            "selection_runtime": selection_runtime,
            "execution_environment": environment,
            "inputs": {
                "frame_file": str(frame),
                "frame_sha256": frame_hash,
                "bank_xml": str(bank_xml),
                "bank_xml_sha256": bank_hash,
            },
            "options": {
                "sizes": list(sizes),
                "python": python,
                "approximant": "IMRPhenomD",
                "psd_model": "aLIGOZeroDetHighPower",
                "sample_rate": 2048,
                "segment_length": 512,
                "low_frequency_cutoff": 30,
                "precision": "single",
                "interpolation": "inline_linear",
                "compression_algorithm": "mchirp",
                "nprocesses": 1,
                "tolerance": 0.001,
            },
            "selection": {
                "seed": SELECTION_SEED,
                "physical_fields": list(PHYSICAL_FIELDS),
                "eligible_rows": None,
                "mass_cuts": "mass1<=10, mass2<=3, chirp mass>=1.2, abs aligned spins<=0.2",
            },
            "banks": banks,
        }
        # Keep eligibility explicit in the receipt without retaining the full bank.
        with h5py.File(full_bank, "r") as source:
            mass1 = source["mass1"][:]
            mass2 = source["mass2"][:]
            chirp = (mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2
            manifest["selection"]["eligible_rows"] = int(np.count_nonzero(
                (mass1 <= 10) & (mass2 <= 3) & (chirp >= 1.2)
                & (np.abs(source["spin1z"][:]) <= 0.2)
                & (np.abs(source["spin2z"][:]) <= 0.2)
            ))
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest
    finally:
        if temporary_full:
            try:
                Path(temporary_full).unlink()
            except FileNotFoundError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = prepare(args)
    except (ValueError, FileExistsError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
