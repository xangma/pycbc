Fresh live benchmark results (2026-09-06).

Throughput is the median of three independent worker medians, each from five warm iterations. Ranges show the minimum and maximum worker median, not confidence intervals. One unit is one template evaluated against one data block; waveform generation is excluded. Ratios divide the displayed median throughputs and have no uncertainty bars.

Main revision: `607bce53ead14f12af32552a5b2441d3bc667267`. Optional CPU revision: `1a2ebea088d9e0a31cbb22c19ad24f96ffea2b7c`.

Each route is compared with standard CPU from the same source head, thread count, and batch size. Rows failing the recorded parity or provenance requirements have no performance claim. All measured routes, including ratios below one, remain in the tables.

| Head | CPU threads | Batch | Route | Throughput / s | Worker range / s | Ratio to same-head CPU | Qualification |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| main | 1 | 1 | Standard CPU (same-head control) | 1,136.9 | 1,131.1–1,139.4 | 1.000× | reference finite output |
| main | 1 | 1 | Torch CPU (defaults) | 469.9 | 468.9–470.4 | 0.413× | passed |
| main | 1 | 1 | Torch CPU (native requested)* | 458.2 | 449.3–468.1 | 0.403× | passed |
| main | 1 | 1 | Torch CUDA (defaults) | 2,169.7 | 2,169.6–2,184.6 | 1.908× | passed |
| main | 1 | 1 | Torch CUDA (native requested) | 2,154.0 | 2,131.5–2,161.5 | 1.895× | passed |
| main | 1 | 8 | Standard CPU (same-head control) | 1,185.1 | 1,184.7–1,186.7 | 1.000× | reference finite output |
| main | 1 | 8 | Torch CPU (defaults) | 292.3 | 292.0–294.9 | 0.247× | passed |
| main | 1 | 8 | Torch CPU (native requested)* | 484.7 | 484.6–487.2 | 0.409× | passed |
| main | 1 | 8 | Torch CUDA (defaults) | 13,664.2 | 13,570.7–13,665.6 | 11.530× | passed |
| main | 1 | 8 | Torch CUDA (native requested) | 11,656.8 | 11,625.5–11,778.6 | 9.836× | passed |
| main | 1 | 32 | Standard CPU (same-head control) | 1,191.5 | 1,191.4–1,196.8 | 1.000× | reference finite output |
| main | 1 | 32 | Torch CPU (defaults) | 296.3 | 295.9–297.3 | 0.249× | passed |
| main | 1 | 32 | Torch CPU (native requested)* | 495.9 | 493.8–496.4 | 0.416× | passed |
| main | 1 | 32 | Torch CUDA (defaults) | 41,316.5 | 41,038.7–41,384.7 | 34.675× | passed |
| main | 1 | 32 | Torch CUDA (native requested) | 26,862.2 | 26,794.5–27,065.5 | 22.544× | passed |
| main | 4 | 1 | Standard CPU (same-head control) | 1,975.2 | 1,973.7–1,983.7 | 1.000× | reference finite output |
| main | 4 | 1 | Torch CPU (defaults) | 858.7 | 835.0–862.4 | 0.435× | passed |
| main | 4 | 1 | Torch CPU (native requested)* | 927.1 | 884.4–941.4 | 0.469× | passed |
| main | 4 | 8 | Standard CPU (same-head control) | 2,819.0 | 2,799.8–2,850.1 | 1.000× | reference finite output |
| main | 4 | 8 | Torch CPU (defaults) | 719.7 | 716.5–723.3 | 0.255× | passed |
| main | 4 | 8 | Torch CPU (native requested)* | 888.3 | 827.1–939.2 | 0.315× | passed |
| main | 4 | 32 | Standard CPU (same-head control) | 2,727.7 | 2,724.0–2,735.7 | 1.000× | reference finite output |
| main | 4 | 32 | Torch CPU (defaults) | 750.2 | 748.9–750.6 | 0.275× | passed |
| main | 4 | 32 | Torch CPU (native requested)* | 1,159.6 | 1,143.0–1,163.3 | 0.425× | passed |
| cpu | 1 | 1 | Standard CPU (same-head control) | 1,132.7 | 1,130.9–1,134.0 | 1.000× | reference finite output |
| cpu | 1 | 1 | Torch CPU (defaults) | 499.2 | 492.5–500.2 | 0.441× | passed |
| cpu | 1 | 1 | Torch CPU (native requested)* | 947.1 | 940.9–950.3 | 0.836× | passed |
| cpu | 1 | 8 | Standard CPU (same-head control) | 1,181.6 | 1,180.4–1,184.0 | 1.000× | reference finite output |
| cpu | 1 | 8 | Torch CPU (defaults) | 293.6 | 291.9–293.7 | 0.248× | passed |
| cpu | 1 | 8 | Torch CPU (native requested)* | 960.1 | 958.6–964.0 | 0.813× | passed |
| cpu | 1 | 32 | Standard CPU (same-head control) | 1,190.1 | 1,186.3–1,193.4 | 1.000× | reference finite output |
| cpu | 1 | 32 | Torch CPU (defaults) | 297.0 | 297.0–297.4 | 0.250× | passed |
| cpu | 1 | 32 | Torch CPU (native requested)* | 941.0 | 938.6–947.6 | 0.791× | passed |
| cpu | 4 | 1 | Standard CPU (same-head control) | 1,946.0 | 1,920.4–1,957.8 | 1.000× | reference finite output |
| cpu | 4 | 1 | Torch CPU (defaults) | 915.8 | 910.9–933.4 | 0.471× | passed |
| cpu | 4 | 1 | Torch CPU (native requested)* | 1,558.7 | 1,543.4–1,571.7 | 0.801× | passed |
| cpu | 4 | 8 | Standard CPU (same-head control) | 2,783.3 | 2,777.0–2,823.7 | 1.000× | reference finite output |
| cpu | 4 | 8 | Torch CPU (defaults) | 715.9 | 713.0–716.5 | 0.257× | passed |
| cpu | 4 | 8 | Torch CPU (native requested)* | 1,337.2 | 1,327.9–1,348.5 | 0.480× | passed |
| cpu | 4 | 32 | Standard CPU (same-head control) | 2,719.6 | 2,719.5–2,723.9 | 1.000× | reference finite output |
| cpu | 4 | 32 | Torch CPU (defaults) | 749.9 | 749.4–750.0 | 0.276× | passed |
| cpu | 4 | 32 | Torch CPU (native requested)* | 1,686.1 | 1,681.4–1,727.2 | 0.620× | passed |

Optional CPU / main ratios use independently measured cells. Native requested CPU additionally enables native batch peaks on the optional head; this measures both the follow-up and its supplied route configuration.

| CPU threads | Batch | Route | Optional CPU / main | Configuration difference |
| ---: | ---: | --- | ---: | --- |
| 1 | 1 | Standard CPU (same-head control) | 0.996× | None requested by the route |
| 1 | 1 | Torch CPU (defaults) | 1.062× | None requested by the route |
| 1 | 1 | Torch CPU (native requested)* | 2.067× | optional CPU additionally requests native batch peaks |
| 1 | 8 | Standard CPU (same-head control) | 0.997× | None requested by the route |
| 1 | 8 | Torch CPU (defaults) | 1.004× | None requested by the route |
| 1 | 8 | Torch CPU (native requested)* | 1.981× | optional CPU additionally requests native batch peaks |
| 1 | 32 | Standard CPU (same-head control) | 0.999× | None requested by the route |
| 1 | 32 | Torch CPU (defaults) | 1.002× | None requested by the route |
| 1 | 32 | Torch CPU (native requested)* | 1.898× | optional CPU additionally requests native batch peaks |
| 4 | 1 | Standard CPU (same-head control) | 0.985× | None requested by the route |
| 4 | 1 | Torch CPU (defaults) | 1.066× | None requested by the route |
| 4 | 1 | Torch CPU (native requested)* | 1.681× | optional CPU additionally requests native batch peaks |
| 4 | 8 | Standard CPU (same-head control) | 0.987× | None requested by the route |
| 4 | 8 | Torch CPU (defaults) | 0.995× | None requested by the route |
| 4 | 8 | Torch CPU (native requested)* | 1.505× | optional CPU additionally requests native batch peaks |
| 4 | 32 | Standard CPU (same-head control) | 0.997× | None requested by the route |
| 4 | 32 | Torch CPU (defaults) | 1.000× | None requested by the route |
| 4 | 32 | Torch CPU (native requested)* | 1.454× | optional CPU additionally requests native batch peaks |

Measurement limits:

- Synthetic LiveBatchMatchedFilter.process_data library workload; not full CLI or application throughput.
- Waveform generation, PSD estimation, bank loading, frame I/O, startup, and workflow scheduling are excluded.
- Fixed complex64 strain/templates/output and float32 PSD; bank size equals batch size (1, 8, or 32); FFT length 131072.
- Chi-square is disabled and sine-Gaussian post-processing is stubbed in this harness.
- Parity is the harness's final trigger comparison and aggregate output-L2 check, not pointwise output equivalence.
- Three workers with five warm iterations each do not support population confidence intervals or tail-latency claims.
- Native requested labels describe configuration; admission and fallback require the separate untimed dispatch probes.
- Main native CPU requests correlation and FFTW batching. Optional CPU additionally requests native batch peaks; this comparison includes that configuration change.
- CUDA on-device peak helper is disabled in these routes; timings do not validate the separately reported asynchronous host-copy race.
- Cross-head ratios compare independently run campaign cells, not paired samples; each route is qualified against its own head's standard CPU control.

Source files (SHA-256):

- `main-t1.json`: `a87c2df38568b883b9e720818445b1a35a6d6e6b5f151736911cf82c6ec70b93`; 2026-09-06T09:01:56.229493+00:00 to 2026-09-06T09:05:57.311396+00:00; host `len`.
- `main-t4.json`: `95a0f48d5c1151cbb3f6537aace17a795a98c9c995dc0cc29e77606f8d3b0431`; 2026-09-06T09:08:25.371294+00:00 to 2026-09-06T09:10:41.622625+00:00; host `len`.
- `cpu-t1.json`: `e9559962bd70dcfb6f53c6ef5b03f756d3474c7d541df19df89e2cbd70efebc6`; 2026-09-06T09:05:57.908203+00:00 to 2026-09-06T09:08:24.736540+00:00; host `len`.
- `cpu-t4.json`: `ec3f52739421995e5043122976ed421bb91558560712e8d3d055066430a21e2a`; 2026-09-06T09:10:42.267808+00:00 to 2026-09-06T09:13:00.270369+00:00; host `len`.
