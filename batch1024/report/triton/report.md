# TaylorF2 Triton: public batch calls

Revision `4885b64560e9f39b740e85b6a976898869dd360e` · 2026-09-06 · 24/24 route/workload groups qualified from 72/72 worker files.

CUDA gate off and gate on use the same candidate revision. Warm throughput is the median of three worker estimates. Parentheses show their observed minimum–maximum, not a confidence interval.

Hardware: NVIDIA GeForce RTX 4090 on `len`; Torch 2.13.0+cu130, Triton 3.7.1.

## 4,097 bins · Δf = 0.25 Hz

| Batch | Route | Qualified workers | Waveforms/s (range) | On/off ratio | Cold first call, ms (range) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Torch CUDA, off | 3/3 | 80.9 (79.7–81.4) | 1.000× | 398 (397–399) |
| 1 | Triton CUDA, on | 3/3 | 92.5 (90.9–93.4) | 1.143× | 2,098 (1,994–2,115) |
| 8 | Torch CUDA, off | 3/3 | 636 (635–641) | 1.000× | 406 (397–408) |
| 8 | Triton CUDA, on | 3/3 | 720 (718–727) | 1.131× | 2,094 (1,994–2,100) |
| 32 | Torch CUDA, off | 3/3 | 2,561 (2,540–2,582) | 1.000× | 401 (399–402) |
| 32 | Triton CUDA, on | 3/3 | 2,946 (2,942–2,968) | 1.150× | 2,004 (1,997–2,096) |
| 128 | Torch CUDA, off | 3/3 | 10,104 (10,061–10,187) | 1.000× | 398 (393–403) |
| 128 | Triton CUDA, on | 3/3 | 11,298 (11,293–11,420) | 1.118× | 2,004 (1,999–2,005) |
| 512 | Torch CUDA, off | 3/3 | 33,735 (33,656–33,794) | 1.000× | 400 (400–404) |
| 512 | Triton CUDA, on | 3/3 | 41,340 (40,913–41,599) | 1.225× | 2,095 (2,008–2,098) |
| 1024 | Torch CUDA, off | 3/3 | 46,283 (45,958–46,300) | 1.000× | 405 (404–406) |
| 1024 | Triton CUDA, on | 3/3 | 73,522 (73,201–74,002) | 1.589× | 2,104 (2,097–2,114) |

## 32,769 bins · Δf = 0.03125 Hz

| Batch | Route | Qualified workers | Waveforms/s (range) | On/off ratio | Cold first call, ms (range) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Torch CUDA, off | 3/3 | 79.7 (79.2–80.8) | 1.000× | 399 (396–403) |
| 1 | Triton CUDA, on | 3/3 | 92.3 (91.5–92.4) | 1.159× | 1,999 (1,998–2,008) |
| 8 | Torch CUDA, off | 3/3 | 648 (646–648) | 1.000× | 398 (397–400) |
| 8 | Triton CUDA, on | 3/3 | 737 (735–747) | 1.138× | 2,005 (1,999–2,011) |
| 32 | Torch CUDA, off | 3/3 | 2,539 (2,511–2,572) | 1.000× | 400 (399–401) |
| 32 | Triton CUDA, on | 3/3 | 2,802 (2,723–2,818) | 1.104× | 1,996 (1,995–1,996) |
| 128 | Torch CUDA, off | 3/3 | 5,865 (5,862–5,884) | 1.000× | 406 (406–407) |
| 128 | Triton CUDA, on | 3/3 | 9,511 (9,431–9,575) | 1.622× | 2,016 (1,998–2,105) |
| 512 | Torch CUDA, off | 3/3 | 8,165 (8,157–8,174) | 1.000× | 448 (446–448) |
| 512 | Triton CUDA, on | 3/3 | 24,239 (24,070–24,389) | 2.969× | 2,007 (2,005–2,017) |
| 1024 | Torch CUDA, off | 3/3 | 8,950 (8,935–8,952) | 1.000× | 503 (501–505) |
| 1024 | Triton CUDA, on | 3/3 | 32,371 (32,051–33,126) | 3.617× | 2,121 (2,115–2,128) |

## Qualification

The renderer verifies all 72 raw worker records and independently recomputes the five-group worker medians and campaign estimates. It checks candidate and harness hashes, clean source, module origins, fixed inputs, runtime identity, output dtype/support metadata, sample duration, actual launcher and kernel launches, and both links of the scalar/LAL reference chain.

Actual-versus-gate-off and gate-off-versus-native-scalar require pointwise relative and relative-L2 errors ≤2e-10. Native scalar versus LAL requires relative L2 ≤1e-11. Direct actual-versus-LAL errors are additionally retained in the raw records. All comparisons require finite values, exact zero support, and exact metadata.

## Measurement scope

- Complete public get_fd_waveform_batch call, including validation, host-list conversion, coefficient work, allocation and output wrapping; host output copies are excluded.
- Complex128 hplus and hcross, one host thread, no autograd; eight distinct deterministic BNS mass pairs repeat in larger batches.
- All bins and both polarizations of every row are checked. Exact repeated rows reuse their complete scalar reference; no rows or frequency bins are sampled.
- Warm timing uses five synchronized groups of at least 50 ms per worker. Three process replicates support an observed range, not a confidence interval or tail-latency claim.
- Cold time is the first complete public call with a fresh per-worker Triton cache. It includes any compilation plus lazy initialization and is not isolated compiler time; the CUDA driver cache is not cleared.
- The measured candidate's gate-off route is the same-revision control. These results do not compare different commits or establish a whole-search or inference speedup.
- Performance cells qualify only after exact-source, dispatch, timing and waveform checks. Failed and incomplete cells remain listed, without performance comparisons.
