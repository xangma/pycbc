# CPU copy diagnostic protocol erratum

The frozen CPU-COPY-PROTOCOL.md says “three-seed/four-scale/three-pattern”.
The executed, unchanged matrix uses four seeds (7, 91, 812, 20260906),
three patterns (dense, banded, impulse), and three scales (1e-12, 1, 1e12).
This gives 36 cases per qualification, 144 per worker, and 432 across three
workers. The strict FFTW L2/max-absolute budgets and bitwise complex128 MKL
comparison were unchanged. All recorded cases equal the prior stage diagnostic.

Frozen inputs and acquired records are preserved unchanged. This erratum
corrects prose only; it does not introduce a new matrix or rerun.
