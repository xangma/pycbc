# Torch documentation cleanup, 2026-09-08

[Cleaned guides and source mapping](torch-docs-cleanup/README.md). Documentation only; scientific evidence below is unchanged.

---

# Torch compatibility fixed, 2026-09-08

Current results: [Torch parity fix](torch-parity-fix/README.md). All four routes produce 1988 triggers and pass the unchanged scientific gates; original CPU remains unchanged. No performance claim.

---

# Original CPU restoration, 2026-09-08

Current results are in [original-cpu-restoration](original-cpu-restoration/README.md): original CPU preserved; Torch scientific comparison fails; no current speedup claim. The CPU precision proposal below is withdrawn from the Torch stack.

---

# Standalone CPU precision validation, 2026-09-08

Start with the [CPU correction review](cpu-review.md) and [runtime cost](cpu-cost-report.md). The four-commit code branch is `66789ac4a7468094b0cc3ca1498a1de67e0311f6`, based on frozen original `40e94792b3edf59f39b18b65102b28a4f74433a7`.

This immutable evidence snapshot contains the local test/lint records, independent numerical/bin analysis, a bounded captured-input injection probe, standalone Linux qualification, and eight CPU timing trials. `SHA256SUMS.json` pins every delivered file. `cpu-evidence-download-verification.json` records 97 remote/local hash matches. Native binaries and captured multi-gigabyte input arrays are not included; their hashes, provenance and remote locations remain in the records. Analysis scripts use the recorded source/input locations.

Earlier reports and failed launcher attempts are preserved verbatim as dated evidence; their then-current remaining-work statements are superseded by cpu-review.md. The obsolete initial numerical-input audit script is not included: numerical-acquisition/numerical-input-audit.py is the corrected, executed extended-precision reference, pinned by its completed receipt and results.

The [earlier frozen baseline archive](https://github.com/xangma/pycbc/tree/bc88a36a225f9b89559e0480e66fac828ee3dd77/baseline-final-20260908) contains the original/combined executable comparisons. Scientific inputs and commands here are frozen to that workload. Source changes intentionally alter trigger ranking; none of this constitutes FAR, population-sensitivity or production-search acceptance validation.
