#!/usr/bin/env python3
"""Prepare a reproducible 96-template workload, not a coverage/search bank.

Run in the PyCBC environment used for compression and filtering:
  python prepare-bank.py --output-dir OUTPUT --plan-only
  python prepare-bank.py --output-dir OUTPUT

Both modes evaluate the duration estimator, without generating waveforms.
Plan mode prints the complete metadata and writes no outputs. Execution writes
bank.hdf, bank-pilot.hdf and bank-metadata.json, refusing existing outputs.
Compression is a separate pycbc_compress_bank invocation.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys


APPROXIMANT = "IMRPhenomD"
FLOW = 30.0
SAMPLE_RATE = 4096
AXES = ("mass1", "mass2", "spin1z", "spin2z")
BOUNDS = {
    "BNS": ((1.2, 2.0), (1.2, 2.0), (-0.05, 0.05), (-0.05, 0.05)),
    "NSBH": ((3.0, 8.0), (1.2, 2.0), (-0.5, 0.5), (-0.05, 0.05)),
}
NUMERIC_COLUMNS = (
    "mass1", "mass2", "inclination", "spin1x", "spin1y", "spin1z",
    "spin2x", "spin2y", "spin2z", "f_lower", "f_final", "template_duration",
)
OUTPUT_NAMES = ("bank.hdf", "bank-pilot.hdf", "bank-metadata.json")


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_rows():
    """Boundary cases plus midpoint strata in four transformed coordinates."""
    rows = []

    def add(population, kind, values):
        params = dict(zip(AXES, values))
        params.update(
            inclination=0.0, spin1x=0.0, spin1y=0.0, spin2x=0.0,
            spin2y=0.0, approximant=APPROXIMANT, f_lower=FLOW,
            f_final=SAMPLE_RATE / 2.0,
        )
        row = {
            "row_index": len(rows), "source_id": f"{population}-{len(rows):03d}",
            "population": population, "selection": kind, "parameters": params,
        }
        rows.append(row)

    # The three ordered-mass triangle vertices, with spin endpoints. Equal-mass
    # spin exchanges are physically equivalent, so retain only one ordering.
    for m1, m2 in ((1.2, 1.2), (2.0, 1.2), (2.0, 2.0)):
        for s1, s2 in itertools.product((-0.05, 0.05), repeat=2):
            if m1 == m2 and s1 > s2:
                continue
            add("BNS", "boundary", (m1, m2, s1, s2))

    def strata(population, count):
        # Each multiplier is coprime to both counts (54 and 16), giving exactly
        # one midpoint in every marginal stratum, with no random-number state.
        for i in range(count):
            u = [((a * i + b) % count + 0.5) / count
                 for a, b in zip((1, 5, 7, 11), (0, 3, 7, 11))]
            values = [lo + (hi - lo) * x
                      for (lo, hi), x in zip(BOUNDS[population], u)]
            if population == "BNS":
                # This area coordinate maps the unit square to the ordered
                # mass triangle; stratification is in u/v, not m1/m2 directly.
                values[0] = 1.2 + 0.8 * math.sqrt(u[0])
                values[1] = 1.2 + (values[0] - 1.2) * u[1]
            add(population, "interior-stratum", values)

    strata("BNS", 54)
    for values in itertools.product(*BOUNDS["NSBH"]):
        add("NSBH", "boundary", values)
    strata("NSBH", 16)

    signatures = set()
    for row in rows:
        params = row["parameters"]
        values = tuple(params[name] for name in AXES)
        if values in signatures:
            raise RuntimeError("Duplicate parameter row")
        signatures.add(values)
        if not all(lo <= value <= hi for value, (lo, hi)
                   in zip(values, BOUNDS[row["population"]])):
            raise RuntimeError("Parameter outside the declared bounds")
        if params["mass1"] < params["mass2"]:
            raise RuntimeError("Mass ordering violated")
    if [sum(r["population"] == p for r in rows) for p in BOUNDS] != [64, 32]:
        raise RuntimeError("Incorrect population counts")
    return rows


def pilot_indices(rows):
    """Four per population: duration extremes, then two separated boundaries."""
    chosen = []
    for population in BOUNDS:
        group = [r for r in rows if r["population"] == population]
        ordered = sorted(group, key=lambda r: (r["parameters"]["template_duration"],
                                               r["row_index"]))
        longest = max(group, key=lambda r: (r["parameters"]["template_duration"],
                                           -r["row_index"]))
        shortest = next(row for row in ordered if row != longest)
        selected = [longest, shortest]

        def separation(row):
            distances = []
            for other in selected:
                distances.append(sum(
                    ((row["parameters"][axis] - other["parameters"][axis])
                     / (hi - lo)) ** 2
                    for axis, (lo, hi) in zip(AXES, BOUNDS[population])))
            return min(distances), -row["row_index"]

        while len(selected) < 4:
            candidates = [r for r in group if r["selection"] == "boundary"
                          and r not in selected]
            selected.append(max(candidates, key=separation))
        chosen.extend(r["row_index"] for r in selected)
    return sorted(chosen)


def provenance(duration_function):
    modules = {}
    for name in ("pycbc", "pycbc.waveform.bank", "pycbc.waveform.waveform",
                 "pycbc.pnutils", "lal", "lalsimulation"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        modules[name] = {"path": str(path), "sha256": sha256(path),
                         "version": str(getattr(module, "__version__", "unknown"))}
    # Include loaded LAL Python extension binaries as well as Python wrappers.
    for name, module in list(sys.modules.items()):
        if name.startswith(("lalsimulation._", "lal._")):
            filename = getattr(module, "__file__", None)
            if filename and Path(filename).is_file():
                path = Path(filename).resolve()
                modules[name] = {"path": str(path), "sha256": sha256(path)}
    versions = {}
    for package in ("PyCBC", "numpy", "h5py", "lalsuite"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "distribution metadata unavailable"
    implementation = inspect.getsource(duration_function)
    return {
        "python_executable": sys.executable, "python_version": sys.version,
        "platform": platform.platform(), "byteorder": sys.byteorder,
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "modules": modules, "distributions": versions,
        "duration_dispatch": {
            "qualified_name": f"{duration_function.__module__}.{duration_function.__name__}",
            "source": inspect.getsourcefile(duration_function),
            "source_sha256": hashlib.sha256(implementation.encode()).hexdigest(),
        },
        "environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "MKL_THREADING_LAYER",
            "MKL_DYNAMIC", "OPENBLAS_NUM_THREADS", "PYCBC_WAVEFORM", "PYTHONPATH")},
    }


def prepare(output_dir):
    import h5py
    import numpy as np
    from pycbc.waveform import get_waveform_filter_length_in_time
    from pycbc.waveform.bank import TemplateBank
    from pycbc.waveform.waveform import _filter_time_lengths

    rows = make_rows()
    selection_digest = hashlib.sha256(encode(rows).encode()).hexdigest()
    for row in rows:
        duration = get_waveform_filter_length_in_time(**row["parameters"])
        if duration is None or not math.isfinite(float(duration)) or duration <= 0:
            raise RuntimeError(f"Invalid duration for {row['source_id']}: {duration}")
        row["parameters"]["template_duration"] = float(duration)
    longest = max(rows, key=lambda r: r["parameters"]["template_duration"])
    maximum = longest["parameters"]["template_duration"]
    compression_length = 1 << (math.ceil(2 * maximum) - 1).bit_length()

    # Let TemplateBank create hashes from its canonical fields and dtypes.
    # This in-memory HDF has no filesystem backing. Source/population metadata
    # never becomes a bank column, because readers treat datasets as parameters.
    memory = h5py.File("benchmark-input.hdf", "w", driver="core", backing_store=False)
    try:
        for name in NUMERIC_COLUMNS:
            memory[name] = np.asarray([r["parameters"][name] for r in rows], dtype=np.float64)
        memory.create_dataset("approximant", data=np.asarray([APPROXIMANT] * len(rows),
                                                             dtype=h5py.string_dtype()))
        memory.attrs["parameters"] = list(NUMERIC_COLUMNS) + ["approximant"]
        bank = TemplateBank(file_handler=memory, approximant=APPROXIMANT)
        hashes = [int(value) for value in bank.table["template_hash"]]
        if len(set(hashes)) != len(rows):
            raise RuntimeError("Template hash collision")
        for row, value in zip(rows, hashes):
            row["template_hash"] = value
        pilot = pilot_indices(rows)
        metadata = {
            "schema_version": 1, "kind": "deterministic benchmark template set",
            "scope": [
                "No minimal-match placement or astrophysical population weights.",
                "IMRPhenomD has no neutron-star tidal or disruption physics here.",
                "Masses are detector-frame solar masses; spin components are dimensionless.",
                "Longest means maximum estimator result among the selected 96 rows.",
                "Duration estimates are not a verification of FFT edge containment.",
                "Compression and PSD-weighted waveform validation are separate steps.",
            ],
            "selection": {
                "sha256": selection_digest, "population_counts": {"BNS": 64, "NSBH": 32},
                "bounds": dict(zip(BOUNDS, (dict(zip(AXES, v)) for v in BOUNDS.values()))),
                "method": "BNS:10 boundary+54 interior; NSBH:16 boundary+16 interior. "
                          "Coprime midpoint permutations in four transformed coordinates.",
            },
            "sample_rate_hz": SAMPLE_RATE, "f_lower_hz": FLOW,
            "approximant": APPROXIMANT, "longest": {
                "source_id": longest["source_id"], "row_index": longest["row_index"],
                "template_hash": longest["template_hash"], "duration_seconds": maximum,
            },
            "compression": {"performed": False, "segment_length_seconds": compression_length,
                            "delta_f_hz": 1.0 / compression_length,
                            "selection_rule": "smallest power of two >= 2 * longest duration"},
            "pilot": {
                "row_indices_in_main_bank": pilot,
                "source_ids": [rows[i]["source_id"] for i in pilot],
                "template_hashes": [hashes[i] for i in pilot],
                "selection": "Four per population: longest and shortest duration, then two "
                             "boundary cases maximizing minimum normalized parameter distance.",
            },
            "outputs": {name: {"path": str(output_dir / name)} for name in OUTPUT_NAMES},
            "provenance": provenance(_filter_time_lengths[APPROXIMANT]), "templates": rows,
        }
        return bank, memory, metadata
    except BaseException:
        memory.close()
        raise


def write_outputs(bank, metadata, output_dir):
    import h5py
    import numpy as np
    from pycbc.waveform.bank import TemplateBank

    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    original = bank.table
    try:
        for name, indices in (
                ("bank.hdf", list(range(len(original)))),
                ("bank-pilot.hdf", metadata["pilot"]["row_indices_in_main_bank"])):
            path = output_dir / name
            bank.table = original[indices]
            with h5py.File(path, "x") as stream:
                created.append(path)
                bank.write_to_hdf(str(path), file_handler=stream,
                                  write_compressed_waveforms=False)
            with h5py.File(path, "r") as stream:
                restored = TemplateBank(file_handler=stream, approximant=APPROXIMANT)
                if set(stream.keys()) != set(NUMERIC_COLUMNS) | {"approximant", "template_hash"}:
                    raise RuntimeError("Unexpected HDF datasets")
                for column in list(NUMERIC_COLUMNS) + ["template_hash", "approximant"]:
                    if not np.array_equal(restored.table[column], bank.table[column]):
                        raise RuntimeError(f"Readback differs for {name}:{column}")
            metadata["outputs"][name].update(sha256=sha256(path), rows=len(indices))
        path = output_dir / "bank-metadata.json"
        with path.open("x", encoding="utf-8") as stream:
            created.append(path)
            stream.write(encode(metadata))
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise
    finally:
        bank.table = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true",
                        help="Compute durations and print JSON; write no outputs.")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if not args.plan_only:
        for name in OUTPUT_NAMES:
            if (output_dir / name).exists():
                parser.error(f"Refusing existing output: {output_dir / name}")
    bank, memory, metadata = prepare(output_dir)
    try:
        if args.plan_only:
            print(encode(metadata), end="")
        else:
            write_outputs(bank, metadata, output_dir)
            print(encode({key: metadata[key] for key in
                          ("outputs", "longest", "compression", "pilot")}), end="")
    finally:
        memory.close()


if __name__ == "__main__":
    main()
