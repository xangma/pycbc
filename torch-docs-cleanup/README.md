# Torch documentation cleanup

Removed fix implementation narrative, dated qualification counts and superseded campaign history from ten Torch guides. Usage, numerical policies, benchmark definitions and test commands remain. Scientific evidence remains in [the preceding archive](../torch-parity-fix/README.md).

Strict Sphinx built all 19 scoped Torch and dependency pages with zero warnings. The source hashes match cleaned main eb8fef9ed1d06378b59cae8439fd40af63827575. All non-documentation files in every restacked PR are byte-identical to their tested heads and previous publication; no scientific workload was rerun. The manifest records each mapping. Optional native CPU documentation retains its existing feature-specific delta, reviewed separately from the main render.

The build imports the previously qualified runtime 88878b1c38c952e63002b812058a0c7316123f70. Its production tree matches cleaned main; documentation differs. The build receipt records the pre-commit source HEAD, and source hashes identify the exact rendered documents. Rendered HTML is archived without copied static assets; use the source and build script to reproduce the themed pages.
