# Independent fresh-benchmark verification

Run from this directory with a Python environment providing NumPy and h5py:

```sh
python -B verify_evidence.py ../remote --repo /path/to/pycbc --output verification.json
```

The repository must contain the measured Git objects. The verifier reads the
archive and repository, imports only the checksum-pinned trigger comparator, and
writes the requested verification JSON outside the archive. It does not import
PyCBC, run workloads, contact remote hosts, or modify the scientific results.

Checks cover all four qualifications and 16 timed processes: command/source and
input receipts, imported module and native/version hashes, original frozen build
receipts and native source equivalence, one-thread runtime settings and affinity,
the complete PSD arrays, every trigger identity and numerical comparison, exact
CPU scientific datasets, balanced chronological order, preservation of every
sample, two elapsed-time records, and recomputed medians/ranges/rates/ratios.
Host CPU utilization is recomputed from raw counters. GPU and CPU observations
are reported as shared-host samples, without implying resource reservation.

The large external input files and Linux native binaries are referenced by hash;
their bytes are not archived here. Conditioned strain is represented by its
full-array hash and schema. These limits are explicit in the verification JSON.
