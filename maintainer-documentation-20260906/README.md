# Maintainer documentation validation

The published documentation candidate is `2685db817fe0fdb77de2e0196fbdfd66d4faf713`.
It changes documentation and its standalone renderer; runtime and test files
are byte-identical to the prior publication. The comparison data is pinned to
archive commit `485cb3c7243c1b49a02faa9ee4c02abb154c79e6`.

`docs-validation.json` and `sphinx.log` record a warning-as-error Sphinx build of
all eleven Torch pages, one image and one downloadable manifest. External
documents are verified and stubbed; this is not a full-site build.
`cleanup-validation.json` records deterministic figure reproduction and rejected
invalid renderer inputs. `restack-proof.json` records preserved downstream PR
hunks and byte-identical runtime/test diffs. The PR17 document blob IDs change
because its base and head both inherit the new results link; an initial strict
whole-patch comparison rejected this harmless metadata change before the hunks
and runtime/test diffs were checked separately.

The renderer and CI-selected F401 checks pass. The full F401 invocation retains
pre-existing public-reexport and test_schemes findings in
`full-f401-preexisting.log`. Qlty excludes `docs/**` and `tools/**` in this
repository; no additional Qlty-covered files changed. Existing unit results
retain their original source identities; the numerical implementation is
unchanged by this cleanup.
