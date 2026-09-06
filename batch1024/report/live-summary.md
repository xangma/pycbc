Fresh live benchmark results (2026-09-06).

Throughput is the median of three independent worker medians, each from five warm iterations. Ranges show the minimum and maximum worker median, not confidence intervals. One unit is one template evaluated against one data block; waveform generation is excluded. Ratios divide the displayed median throughputs and have no uncertainty bars.

Main revision: `4885b64560e9f39b740e85b6a976898869dd360e`. Optional CPU revision: `d544420232428225c214a4be84fbe1262a6d307b`.

Each route is compared with standard CPU from the same source head, thread count, and batch size. Rows failing the recorded parity or provenance requirements have no performance claim. All measured routes, including ratios below one, remain in the tables.

| Head | CPU threads | Batch | Route | Throughput / s | Worker range / s | Ratio to same-head CPU | Qualification |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| main | 1 | 1 | Standard CPU (same-head control) | 1,131.1 | 1,084.1–1,143.8 | 1.000× | reference finite output |
| main | 1 | 1 | Torch CPU (defaults) | 466.9 | 452.4–471.3 | 0.413× | passed |
| main | 1 | 1 | Torch CPU (native requested)* | 462.2 | 455.4–466.0 | 0.409× | passed |
| main | 1 | 1 | Torch CUDA (defaults) | 2,175.2 | 2,152.9–2,179.1 | 1.923× | passed |
| main | 1 | 1 | Torch CUDA (native requested) | 2,127.0 | 2,116.3–2,134.1 | 1.881× | passed |
| main | 1 | 8 | Standard CPU (same-head control) | 1,184.1 | 1,180.8–1,184.9 | 1.000× | reference finite output |
| main | 1 | 8 | Torch CPU (defaults) | 292.1 | 291.8–292.9 | 0.247× | passed |
| main | 1 | 8 | Torch CPU (native requested)* | 485.2 | 484.7–486.9 | 0.410× | passed |
| main | 1 | 8 | Torch CUDA (defaults) | 13,554.8 | 13,507.5–13,850.8 | 11.447× | passed |
| main | 1 | 8 | Torch CUDA (native requested) | 11,668.0 | 11,648.9–11,805.2 | 9.854× | passed |
| main | 1 | 32 | Standard CPU (same-head control) | 1,189.9 | 1,183.3–1,190.9 | 1.000× | reference finite output |
| main | 1 | 32 | Torch CPU (defaults) | 297.1 | 297.0–297.2 | 0.250× | passed |
| main | 1 | 32 | Torch CPU (native requested)* | 493.7 | 493.1–495.0 | 0.415× | passed |
| main | 1 | 32 | Torch CUDA (defaults) | 41,244.9 | 40,973.2–41,541.3 | 34.662× | passed |
| main | 1 | 32 | Torch CUDA (native requested) | 27,140.6 | 26,864.9–27,163.5 | 22.809× | passed |
| main | 1 | 128 | Standard CPU (same-head control) | 1,190.9 | 1,187.3–1,192.0 | 1.000× | reference finite output |
| main | 1 | 128 | Torch CPU (defaults) | 212.4 | 211.1–213.3 | 0.178× | passed |
| main | 1 | 128 | Torch CPU (native requested)* | 331.2 | 330.9–332.8 | 0.278× | passed |
| main | 1 | 128 | Torch CUDA (defaults) | 48,852.6 | 48,751.3–49,226.5 | 41.023× | passed |
| main | 1 | 128 | Torch CUDA (native requested) | 20,962.4 | 20,305.4–21,013.6 | 17.603× | passed |
| main | 1 | 512 | Standard CPU (same-head control) | 1,199.9 | 1,173.4–1,205.3 | 1.000× | reference finite output |
| main | 1 | 512 | Torch CPU (defaults) | 211.2 | 209.8–211.9 | 0.176× | passed |
| main | 1 | 512 | Torch CPU (native requested)* | 321.5 | 307.5–323.7 | 0.268× | passed |
| main | 1 | 512 | Torch CUDA (defaults) | 52,570.9 | 52,550.5–52,580.1 | 43.811× | passed |
| main | 1 | 512 | Torch CUDA (native requested) | 8,772.8 | 8,749.6–8,787.4 | 7.311× | passed |
| main | 1 | 1024 | Standard CPU (same-head control) | 1,193.3 | 1,155.0–1,195.5 | 1.000× | reference finite output |
| main | 1 | 1024 | Torch CPU (defaults) | 207.9 | 207.7–208.2 | 0.174× | passed |
| main | 1 | 1024 | Torch CPU (native requested)* | 287.3 | 273.4–304.0 | 0.241× | passed |
| main | 1 | 1024 | Torch CUDA (defaults) | 52,984.4 | 52,725.0–53,004.5 | 44.402× | passed |
| main | 1 | 1024 | Torch CUDA (native requested) | 4,948.2 | 4,943.2–4,951.7 | 4.147× | passed |
| main | 4 | 1 | Standard CPU (same-head control) | 1,965.6 | 1,961.8–1,978.6 | 1.000× | reference finite output |
| main | 4 | 1 | Torch CPU (defaults) | 851.6 | 835.2–870.8 | 0.433× | passed |
| main | 4 | 1 | Torch CPU (native requested)* | 932.5 | 919.4–932.6 | 0.474× | passed |
| main | 4 | 8 | Standard CPU (same-head control) | 2,784.8 | 2,780.7–2,841.2 | 1.000× | reference finite output |
| main | 4 | 8 | Torch CPU (defaults) | 718.7 | 715.5–722.8 | 0.258× | passed |
| main | 4 | 8 | Torch CPU (native requested)* | 892.9 | 880.7–907.5 | 0.321× | passed |
| main | 4 | 32 | Standard CPU (same-head control) | 2,724.5 | 2,722.0–2,733.8 | 1.000× | reference finite output |
| main | 4 | 32 | Torch CPU (defaults) | 751.8 | 748.8–752.2 | 0.276× | passed |
| main | 4 | 32 | Torch CPU (native requested)* | 1,162.0 | 1,156.7–1,166.8 | 0.427× | passed |
| main | 4 | 128 | Standard CPU (same-head control) | 2,863.5 | 2,857.4–2,878.3 | 1.000× | reference finite output |
| main | 4 | 128 | Torch CPU (defaults) | 542.4 | 540.3–544.1 | 0.189× | passed |
| main | 4 | 128 | Torch CPU (native requested)* | 897.1 | 894.3–897.4 | 0.313× | passed |
| main | 4 | 512 | Standard CPU (same-head control) | 2,935.4 | 2,931.0–2,936.8 | 1.000× | reference finite output |
| main | 4 | 512 | Torch CPU (defaults) | 553.3 | 551.9–554.6 | 0.189× | passed |
| main | 4 | 512 | Torch CPU (native requested)* | 898.1 | 893.4–898.3 | 0.306× | passed |
| main | 4 | 1024 | Standard CPU (same-head control) | 2,937.9 | 2,855.4–2,949.6 | 1.000× | reference finite output |
| main | 4 | 1024 | Torch CPU (defaults) | 546.2 | 545.9–550.2 | 0.186× | passed |
| main | 4 | 1024 | Torch CPU (native requested)* | 855.6 | 854.2–855.8 | 0.291× | passed |
| cpu | 1 | 1 | Standard CPU (same-head control) | 1,130.2 | 1,068.7–1,133.9 | 1.000× | reference finite output |
| cpu | 1 | 1 | Torch CPU (defaults) | 495.3 | 494.5–497.6 | 0.438× | passed |
| cpu | 1 | 1 | Torch CPU (native requested)* | 942.9 | 930.6–950.6 | 0.834× | passed |
| cpu | 1 | 8 | Standard CPU (same-head control) | 1,176.0 | 1,172.0–1,178.9 | 1.000× | reference finite output |
| cpu | 1 | 8 | Torch CPU (defaults) | 292.3 | 291.8–293.0 | 0.249× | passed |
| cpu | 1 | 8 | Torch CPU (native requested)* | 964.6 | 964.3–965.9 | 0.820× | passed |
| cpu | 1 | 32 | Standard CPU (same-head control) | 1,189.4 | 1,184.7–1,190.4 | 1.000× | reference finite output |
| cpu | 1 | 32 | Torch CPU (defaults) | 296.6 | 296.4–296.9 | 0.249× | passed |
| cpu | 1 | 32 | Torch CPU (native requested)* | 943.5 | 929.9–944.1 | 0.793× | passed |
| cpu | 1 | 128 | Standard CPU (same-head control) | 1,200.7 | 1,197.7–1,200.9 | 1.000× | reference finite output |
| cpu | 1 | 128 | Torch CPU (defaults) | 211.2 | 211.1–211.2 | 0.176× | passed |
| cpu | 1 | 128 | Torch CPU (native requested)* | 940.7 | 935.4–942.6 | 0.783× | passed |
| cpu | 1 | 512 | Standard CPU (same-head control) | 1,201.0 | 1,198.9–1,203.8 | 1.000× | reference finite output |
| cpu | 1 | 512 | Torch CPU (defaults) | 211.9 | 211.7–213.9 | 0.176× | passed |
| cpu | 1 | 512 | Torch CPU (native requested)* | 896.5 | 893.0–896.5 | 0.746× | passed |
| cpu | 1 | 1024 | Standard CPU (same-head control) | 1,196.2 | 1,191.0–1,202.5 | 1.000× | reference finite output |
| cpu | 1 | 1024 | Torch CPU (defaults) | 206.9 | 206.9–207.9 | 0.173× | passed |
| cpu | 1 | 1024 | Torch CPU (native requested)* | 760.8 | 610.5–830.9 | 0.636× | passed |
| cpu | 4 | 1 | Standard CPU (same-head control) | 1,945.3 | 1,931.4–1,945.5 | 1.000× | reference finite output |
| cpu | 4 | 1 | Torch CPU (defaults) | 905.1 | 886.6–937.0 | 0.465× | passed |
| cpu | 4 | 1 | Torch CPU (native requested)* | 1,562.7 | 1,517.1–1,570.8 | 0.803× | passed |
| cpu | 4 | 8 | Standard CPU (same-head control) | 2,801.4 | 2,753.9–2,803.5 | 1.000× | reference finite output |
| cpu | 4 | 8 | Torch CPU (defaults) | 714.8 | 712.5–717.7 | 0.255× | passed |
| cpu | 4 | 8 | Torch CPU (native requested)* | 1,374.6 | 1,337.0–1,406.0 | 0.491× | passed |
| cpu | 4 | 32 | Standard CPU (same-head control) | 2,723.6 | 2,716.1–2,747.6 | 1.000× | reference finite output |
| cpu | 4 | 32 | Torch CPU (defaults) | 745.1 | 744.8–747.1 | 0.274× | passed |
| cpu | 4 | 32 | Torch CPU (native requested)* | 1,716.2 | 1,711.4–1,739.3 | 0.630× | passed |
| cpu | 4 | 128 | Standard CPU (same-head control) | 2,860.8 | 2,858.0–2,869.4 | 1.000× | reference finite output |
| cpu | 4 | 128 | Torch CPU (defaults) | 540.2 | 539.8–541.4 | 0.189× | passed |
| cpu | 4 | 128 | Torch CPU (native requested)* | 2,245.0 | 2,215.9–2,274.5 | 0.785× | passed |
| cpu | 4 | 512 | Standard CPU (same-head control) | 2,923.3 | 2,923.0–2,923.4 | 1.000× | reference finite output |
| cpu | 4 | 512 | Torch CPU (defaults) | 551.5 | 551.4–553.5 | 0.189× | passed |
| cpu | 4 | 512 | Torch CPU (native requested)* | 2,096.6 | 2,091.1–2,096.6 | 0.717× | passed |
| cpu | 4 | 1024 | Standard CPU (same-head control) | 2,818.3 | 2,773.9–2,931.3 | 1.000× | reference finite output |
| cpu | 4 | 1024 | Torch CPU (defaults) | 548.5 | 546.5–548.8 | 0.195× | passed |
| cpu | 4 | 1024 | Torch CPU (native requested)* | 1,832.7 | 1,832.5–1,835.0 | 0.650× | passed |

Optional CPU / main ratios use independently measured cells. Native requested CPU additionally enables native batch peaks on the optional head; this measures both the follow-up and its supplied route configuration.

| CPU threads | Batch | Route | Optional CPU / main | Configuration difference |
| ---: | ---: | --- | ---: | --- |
| 1 | 1 | Standard CPU (same-head control) | 0.999× | None requested by the route |
| 1 | 1 | Torch CPU (defaults) | 1.061× | None requested by the route |
| 1 | 1 | Torch CPU (native requested)* | 2.040× | optional CPU additionally requests native batch peaks |
| 1 | 8 | Standard CPU (same-head control) | 0.993× | None requested by the route |
| 1 | 8 | Torch CPU (defaults) | 1.000× | None requested by the route |
| 1 | 8 | Torch CPU (native requested)* | 1.988× | optional CPU additionally requests native batch peaks |
| 1 | 32 | Standard CPU (same-head control) | 1.000× | None requested by the route |
| 1 | 32 | Torch CPU (defaults) | 0.998× | None requested by the route |
| 1 | 32 | Torch CPU (native requested)* | 1.911× | optional CPU additionally requests native batch peaks |
| 1 | 128 | Standard CPU (same-head control) | 1.008× | None requested by the route |
| 1 | 128 | Torch CPU (defaults) | 0.994× | None requested by the route |
| 1 | 128 | Torch CPU (native requested)* | 2.840× | optional CPU additionally requests native batch peaks |
| 1 | 512 | Standard CPU (same-head control) | 1.001× | None requested by the route |
| 1 | 512 | Torch CPU (defaults) | 1.003× | None requested by the route |
| 1 | 512 | Torch CPU (native requested)* | 2.789× | optional CPU additionally requests native batch peaks |
| 1 | 1024 | Standard CPU (same-head control) | 1.002× | None requested by the route |
| 1 | 1024 | Torch CPU (defaults) | 0.995× | None requested by the route |
| 1 | 1024 | Torch CPU (native requested)* | 2.648× | optional CPU additionally requests native batch peaks |
| 4 | 1 | Standard CPU (same-head control) | 0.990× | None requested by the route |
| 4 | 1 | Torch CPU (defaults) | 1.063× | None requested by the route |
| 4 | 1 | Torch CPU (native requested)* | 1.676× | optional CPU additionally requests native batch peaks |
| 4 | 8 | Standard CPU (same-head control) | 1.006× | None requested by the route |
| 4 | 8 | Torch CPU (defaults) | 0.995× | None requested by the route |
| 4 | 8 | Torch CPU (native requested)* | 1.539× | optional CPU additionally requests native batch peaks |
| 4 | 32 | Standard CPU (same-head control) | 1.000× | None requested by the route |
| 4 | 32 | Torch CPU (defaults) | 0.991× | None requested by the route |
| 4 | 32 | Torch CPU (native requested)* | 1.477× | optional CPU additionally requests native batch peaks |
| 4 | 128 | Standard CPU (same-head control) | 0.999× | None requested by the route |
| 4 | 128 | Torch CPU (defaults) | 0.996× | None requested by the route |
| 4 | 128 | Torch CPU (native requested)* | 2.503× | optional CPU additionally requests native batch peaks |
| 4 | 512 | Standard CPU (same-head control) | 0.996× | None requested by the route |
| 4 | 512 | Torch CPU (defaults) | 0.997× | None requested by the route |
| 4 | 512 | Torch CPU (native requested)* | 2.334× | optional CPU additionally requests native batch peaks |
| 4 | 1024 | Standard CPU (same-head control) | 0.959× | None requested by the route |
| 4 | 1024 | Torch CPU (defaults) | 1.004× | None requested by the route |
| 4 | 1024 | Torch CPU (native requested)* | 2.142× | optional CPU additionally requests native batch peaks |

Measurement limits:

- Synthetic LiveBatchMatchedFilter.process_data library workload; not full CLI or application throughput.
- Waveform generation, PSD estimation, bank loading, frame I/O, startup, and workflow scheduling are excluded.
- Fixed complex64 strain/templates/output and float32 PSD; bank size equals batch size (1, 8, 32, 128, 512, or 1024); FFT length 131072.
- Chi-square is disabled and sine-Gaussian post-processing is stubbed in this harness.
- Parity is the harness's final trigger comparison and aggregate output-L2 check, not pointwise output equivalence.
- Three workers with five warm iterations each do not support population confidence intervals or tail-latency claims.
- Native requested labels describe configuration; admission and fallback require the separate untimed dispatch probes.
- Main native CPU requests correlation and FFTW batching. Optional CPU additionally requests native batch peaks; this comparison includes that configuration change.
- CUDA on-device peak helper is disabled in these routes; timings do not validate the separately reported asynchronous host-copy race.
- Cross-head ratios compare independently run campaign cells, not paired samples; each route is qualified against its own head's standard CPU control.

Source files (SHA-256):

- `main-t1.json`: `c1b834e3f26094ac7446d41c297734b0609fcac16482afb5a9d76c303011cc94`; 2026-09-06T12:18:41.379792+00:00 to 2026-09-06T12:47:46.000404+00:00; host `len`.
- `main-t4.json`: `d41b4973362f0cdaef8a42572f00593c96e8d4948d08b79593c660b20c3c2d22`; 2026-09-06T13:08:03.749390+00:00 to 2026-09-06T13:20:38.198915+00:00; host `len`.
- `cpu-t1.json`: `22293627dbbd025963ec9e76e52390441e0207144cffae96c7f2b663fa1b1f75`; 2026-09-06T12:47:48.215594+00:00 to 2026-09-06T13:08:01.618995+00:00; host `len`.
- `cpu-t4.json`: `723a4d398201b82af6c8d246d556bca78b2384d9b5693ef720663c1e6f448180`; 2026-09-06T13:20:40.350070+00:00 to 2026-09-06T13:32:02.531534+00:00; host `len`.
