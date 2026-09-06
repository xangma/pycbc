# TaylorF2 Triton: public batch calls

Revision `6829dc9bcba07ca7d3da44de7589cc4e9fb84da5` · 2026-09-06 · 20/20 route/workload groups qualified from 60/60 worker files.

CUDA gate off and gate on use the same candidate revision. Warm throughput is the median of three worker estimates. Parentheses show their observed minimum–maximum, not a confidence interval.

Hardware: NVIDIA GeForce RTX 4090 on `len`; Torch 2.13.0+cu130, Triton 3.7.1.

## 4,097 bins · Δf = 0.25 Hz

| Batch | Route | Qualified workers | Waveforms/s (range) | On/off ratio | Cold first call, ms (range) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Torch CUDA, off | 3/3 | 81.7 (81.6–81.7) | 1.000× | 397 (397–399) |
| 1 | Triton CUDA, on | 3/3 | 93.1 (92.9–93.2) | 1.139× | 1,994 (1,994–1,996) |
| 8 | Torch CUDA, off | 3/3 | 640 (636–647) | 1.000× | 399 (398–399) |
| 8 | Triton CUDA, on | 3/3 | 734 (716–738) | 1.148× | 1,995 (1,994–1,997) |
| 32 | Torch CUDA, off | 3/3 | 2,537 (2,522–2,575) | 1.000× | 399 (398–400) |
| 32 | Triton CUDA, on | 3/3 | 2,970 (2,958–2,983) | 1.170× | 1,996 (1,995–2,092) |
| 128 | Torch CUDA, off | 3/3 | 10,211 (10,130–10,342) | 1.000× | 402 (401–402) |
| 128 | Triton CUDA, on | 3/3 | 11,412 (11,274–11,549) | 1.118× | 1,997 (1,994–1,997) |
| 512 | Torch CUDA, off | 3/3 | 33,435 (33,112–33,999) | 1.000× | 402 (401–403) |
| 512 | Triton CUDA, on | 3/3 | 40,714 (40,606–41,263) | 1.218× | 1,998 (1,997–1,999) |

## 32,769 bins · Δf = 0.03125 Hz

| Batch | Route | Qualified workers | Waveforms/s (range) | On/off ratio | Cold first call, ms (range) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Torch CUDA, off | 3/3 | 80.3 (79.6–80.5) | 1.000× | 399 (397–399) |
| 1 | Triton CUDA, on | 3/3 | 91.2 (91.1–92) | 1.136× | 2,089 (1,993–2,096) |
| 8 | Torch CUDA, off | 3/3 | 651 (648–652) | 1.000× | 399 (398–401) |
| 8 | Triton CUDA, on | 3/3 | 743 (726–744) | 1.141× | 1,998 (1,996–2,003) |
| 32 | Torch CUDA, off | 3/3 | 2,546 (2,531–2,572) | 1.000× | 401 (400–404) |
| 32 | Triton CUDA, on | 3/3 | 2,823 (2,811–2,845) | 1.109× | 1,999 (1,995–2,001) |
| 128 | Torch CUDA, off | 3/3 | 5,871 (5,844–5,895) | 1.000× | 408 (406–409) |
| 128 | Triton CUDA, on | 3/3 | 9,495 (9,356–9,612) | 1.617× | 2,001 (1,996–2,002) |
| 512 | Torch CUDA, off | 3/3 | 8,164 (8,164–8,175) | 1.000× | 448 (447–451) |
| 512 | Triton CUDA, on | 3/3 | 24,264 (24,260–24,465) | 2.972× | 2,018 (2,010–2,033) |

## Qualification

The renderer verifies all 60 raw worker records and independently recomputes the five-group worker medians and campaign estimates. It checks candidate and harness hashes, clean source, module origins, fixed inputs, runtime identity, output dtype/support metadata, sample duration, actual launcher and kernel launches, and both links of the scalar/LAL reference chain.

Actual-versus-gate-off and gate-off-versus-native-scalar require pointwise relative and relative-L2 errors ≤2e-10. Native scalar versus LAL requires relative L2 ≤1e-11. Direct actual-versus-LAL errors are additionally retained in the raw records. All comparisons require finite values, exact zero support, and exact metadata.

## Measurement scope

- Complete public get_fd_waveform_batch call, including validation, host-list conversion, coefficient work, allocation and output wrapping; host output copies are excluded.
- Complex128 hplus and hcross, one host thread, no autograd; eight distinct deterministic BNS mass pairs repeat in larger batches.
- All bins and both polarizations of every row are checked. Exact repeated rows reuse their complete scalar reference; no rows or frequency bins are sampled.
- Warm timing uses five synchronized groups of at least 50 ms per worker. Three process replicates support an observed range, not a confidence interval or tail-latency claim.
- Cold time is the first complete public call with a fresh per-worker Triton cache. It includes any compilation plus lazy initialization and is not isolated compiler time; the CUDA driver cache is not cleared.
- The measured candidate's gate-off route is the same-revision control. These results do not compare different commits or establish a whole-search or inference speedup.
- Performance cells qualify only after exact-source, dispatch, timing and waveform checks. Failed and incomplete cells remain listed, without performance comparisons.
