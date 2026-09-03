# IMRPhenomD reference data

`LALSimIMRPhenomD.h` supplies the quasi-normal-mode tables used by the
Torch-scheme IMRPhenomD implementation. It is copied from the GPL-licensed
LALSuite repository at commit
`53acc0c2cd7ea2d3c83fc71a7bd88035c2861d6c` and has SHA-256 digest
`46d2104d89846c16d6e1b8020354adb2fe6940b43c6226abb3070cf0cb117b23`.

Bundling the table makes installed wheels and source distributions independent
of a separate LALSuite source checkout. Runtime waveform data such as SEOBNR
ROM files remain external and are located through `LAL_DATA_PATH`.
