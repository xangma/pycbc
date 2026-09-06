# Paired Torch performance comparison

Before: `dfd42bf76766cadca0eecf609a1eaeac73534676`. After: `97bf1614f3afe53a6e2edb4d8c7dff79e9661782`.

Campaign completed 2026-09-06T15:47:33.542233+00:00 on len; CUDA device: NVIDIA GeForce RTX 4090. All 252 live workers (including 36 standard CPU controls), 108 waveform workers and 108 live parity records are present and validated.

Three separate worker processes per source, route, thread count and batch. Throughput is the median of their three warm throughput medians. Ranges are min–max across workers, not confidence intervals. The paired ratio is the median of candidate/baseline worker ratios matched by replicate; it need not equal the ratio of the two aggregate medians.

Sequential workers pinned to CPU cores 8–11; before/after order reverses for replicate 2. Timing excludes profiling and dispatch instrumentation. Live workers use three warm iterations after one warmup; waveform workers use five calibrated timed blocks after two warmups. CUDA calls synchronize at timing boundaries. These three replicates characterize run variation, not independent scientific workloads.

## Live filtering capacity at real time

![Live filtering capacity at real time before and after](live-before-after.png)

Parentheses show the range across three workers. Paired after/before values above 1 indicate higher capacity or generation rate.

| Route | Batch | Unit | Before | After | Paired after/before |
|:--|--:|:--|--:|--:|--:|
| Torch CPU · 1 thread | 1 | templates/core at real time | 25,620.0 (25,369.7–25,754.5) | 38,097.4 (37,224.8–38,369.2) | 1.487 (1.467–1.490) |
| Torch CPU · 1 thread | 8 | templates/core at real time | 16,392.7 (16,383.0–16,444.9) | 20,687.9 (20,590.7–20,759.2) | 1.258 (1.257–1.266) |
| Torch CPU · 1 thread | 32 | templates/core at real time | 16,606.1 (16,416.8–16,662.7) | 21,057.1 (20,993.6–21,066.7) | 1.268 (1.260–1.283) |
| Torch CPU · 1 thread | 128 | templates/core at real time | 11,860.0 (11,831.0–11,884.0) | 18,551.0 (18,488.5–18,591.7) | 1.564 (1.563–1.564) |
| Torch CPU · 1 thread | 512 | templates/core at real time | 11,817.6 (11,811.5–11,855.3) | 18,549.7 (18,544.7–18,584.8) | 1.570 (1.565–1.573) |
| Torch CPU · 1 thread | 1024 | templates/core at real time | 11,819.7 (11,630.7–11,856.2) | 18,403.7 (18,399.5–18,552.8) | 1.557 (1.552–1.595) |
| Torch CPU · 4 threads | 1 | templates/core at real time | 11,671.9 (11,515.9–11,811.3) | 14,395.5 (14,229.3–14,656.7) | 1.236 (1.219–1.256) |
| Torch CPU · 4 threads | 8 | templates/core at real time | 10,035.7 (10,020.6–10,037.9) | 11,658.2 (11,642.4–11,746.4) | 1.162 (1.162–1.170) |
| Torch CPU · 4 threads | 32 | templates/core at real time | 10,487.9 (10,485.8–10,508.0) | 11,954.5 (11,892.3–11,969.7) | 1.140 (1.132–1.141) |
| Torch CPU · 4 threads | 128 | templates/core at real time | 7,591.2 (7,570.8–7,602.1) | 10,762.5 (10,721.1–10,803.0) | 1.422 (1.410–1.423) |
| Torch CPU · 4 threads | 512 | templates/core at real time | 7,764.5 (7,758.6–7,777.4) | 10,979.8 (10,978.6–10,981.6) | 1.414 (1.412–1.415) |
| Torch CPU · 4 threads | 1024 | templates/core at real time | 7,827.7 (7,818.4–7,831.4) | 10,967.3 (10,939.7–10,979.4) | 1.401 (1.399–1.402) |
| Torch CUDA · 1 thread | 1 | templates/GPU at real time | 120,578.8 (119,081.9–120,643.9) | 115,652.9 (114,427.3–116,066.0) | 0.959 (0.948–0.975) |
| Torch CUDA · 1 thread | 8 | templates/GPU at real time | 749,473.9 (746,812.3–761,471.4) | 728,223.6 (723,721.5–743,517.5) | 0.966 (0.956–0.996) |
| Torch CUDA · 1 thread | 32 | templates/GPU at real time | 2,298,638.3 (2,277,604.2–2,309,412.3) | 2,287,623.6 (2,240,933.7–2,311,720.2) | 0.991 (0.984–1.006) |
| Torch CUDA · 1 thread | 128 | templates/GPU at real time | 2,742,214.6 (2,616,437.5–2,770,674.1) | 2,725,105.7 (2,723,551.9–2,761,240.2) | 1.007 (0.983–1.042) |
| Torch CUDA · 1 thread | 512 | templates/GPU at real time | 2,942,934.0 (2,930,361.6–2,947,597.6) | 2,938,440.4 (2,932,327.3–2,954,534.2) | 1.001 (0.998–1.002) |
| Torch CUDA · 1 thread | 1024 | templates/GPU at real time | 2,965,339.6 (2,951,922.3–2,968,097.3) | 2,956,722.7 (2,948,978.7–2,988,596.7) | 1.002 (0.994–1.007) |
| Torch CPU · 1 thread · native enabled | 1 | templates/core at real time | 25,784.9 (25,493.0–25,785.1) | 37,456.0 (37,411.3–38,561.0) | 1.468 (1.453–1.495) |
| Torch CPU · 1 thread · native enabled | 8 | templates/core at real time | 27,253.3 (27,218.5–27,266.2) | 42,064.5 (41,694.8–42,149.1) | 1.543 (1.532–1.547) |
| Torch CPU · 1 thread · native enabled | 32 | templates/core at real time | 27,646.8 (27,372.7–27,685.7) | 30,539.8 (30,505.4–30,781.0) | 1.113 (1.103–1.114) |
| Torch CPU · 1 thread · native enabled | 128 | templates/core at real time | 18,600.4 (18,576.2–18,630.7) | 43,209.6 (32,832.4–43,353.9) | 2.326 (1.762–2.331) |
| Torch CPU · 1 thread · native enabled | 512 | templates/core at real time | 18,127.4 (18,052.8–18,136.9) | 33,778.2 (33,679.5–33,855.9) | 1.867 (1.858–1.871) |
| Torch CPU · 1 thread · native enabled | 1024 | templates/core at real time | 17,748.6 (17,744.5–17,820.4) | 34,682.4 (33,890.2–34,860.3) | 1.955 (1.909–1.956) |
| Torch CPU · 4 threads · native enabled | 1 | templates/core at real time | 12,647.8 (12,358.8–12,718.4) | 14,776.7 (14,658.7–15,129.6) | 1.162 (1.159–1.224) |
| Torch CPU · 4 threads · native enabled | 8 | templates/core at real time | 12,902.1 (12,883.3–13,461.0) | 16,319.2 (15,416.4–17,171.9) | 1.267 (1.145–1.331) |
| Torch CPU · 4 threads · native enabled | 32 | templates/core at real time | 16,199.1 (16,015.4–16,362.5) | 15,283.4 (15,078.6–15,342.5) | 0.942 (0.934–0.947) |
| Torch CPU · 4 threads · native enabled | 128 | templates/core at real time | 12,478.4 (12,371.1–12,583.3) | 25,328.9 (19,165.8–25,527.3) | 2.030 (1.523–2.063) |
| Torch CPU · 4 threads · native enabled | 512 | templates/core at real time | 12,547.7 (12,544.1–12,599.5) | 20,686.6 (20,611.1–20,688.8) | 1.643 (1.642–1.649) |
| Torch CPU · 4 threads · native enabled | 1024 | templates/core at real time | 11,942.8 (11,902.7–11,947.7) | 20,974.8 (20,927.8–21,045.5) | 1.756 (1.752–1.768) |
| Torch CUDA · 1 thread · native enabled | 1 | templates/GPU at real time | 117,016.1 (115,665.7–118,002.6) | 113,102.8 (111,842.3–113,831.4) | 0.958 (0.956–0.984) |
| Torch CUDA · 1 thread · native enabled | 8 | templates/GPU at real time | 643,704.1 (642,323.7–650,793.6) | 653,666.6 (652,785.6–659,768.2) | 1.018 (1.003–1.025) |
| Torch CUDA · 1 thread · native enabled | 32 | templates/GPU at real time | 1,494,216.2 (1,400,792.3–1,509,139.0) | 1,789,577.5 (1,767,121.3–1,798,792.2) | 1.204 (1.171–1.278) |
| Torch CUDA · 1 thread · native enabled | 128 | templates/GPU at real time | 1,175,811.4 (1,171,916.7–1,182,439.1) | 2,047,704.2 (2,043,300.0–2,050,645.9) | 1.744 (1.732–1.744) |
| Torch CUDA · 1 thread · native enabled | 512 | templates/GPU at real time | 490,941.7 (490,608.7–492,026.9) | 2,138,968.4 (2,132,888.3–2,157,802.9) | 4.347 (4.347–4.395) |
| Torch CUDA · 1 thread · native enabled | 1024 | templates/GPU at real time | 277,782.9 (277,133.5–278,305.4) | 2,162,886.4 (2,157,512.7–2,167,719.6) | 7.804 (7.752–7.804) |

Synthetic seeded templates and injected strain; N=131072, three strain blocks per iteration, complex64 inputs/outputs, SNR threshold 5.5. The public LiveBatchMatchedFilter.process_data API is timed with chi-square and sine-Gaussian vetoes disabled. Frame I/O, PSD estimation, bank loading, waveform generation and CLI startup are excluded. The raw unit templates/s counts one template filtered against one strain block. Native labels mean opt-in routing is enabled; admission can select a fallback for a given batch.

Real-time filtering capacity is derived from the measured template–block rate: multiply by 56 seconds of valid strain per block, then divide CPU routes by their configured 1- or 4-core budget. CUDA capacity uses one GPU and includes the timed host work. The CPU divisor is the configured thread count, not measured CPU utilization; affinity CPUs 8–11 are distinct physical cores. This is synthetic matched-filter component capacity with vetoes and I/O excluded, not complete production search capacity. Raw template–block rates and all unchanged ratios remain in JSON. The live harness's legacy throughput_wps_summary unit says waveforms/second, but its numerator is batch size times block count: template–block evaluations, not generated waveforms. That raw label is preserved; standalone waveform generation has a separate waveforms/second metric. The 56 s basis, pinned source trees, full file hashes and source excerpts are recorded in search_capacity_basis.

Each baseline and candidate is checked against the corresponding standard CPU control: trigger counts and template IDs, end times within 1e-4 s, SNR and wrapped phase differences below 1e-3, relative sigma-squared and aggregate output-norm differences below 1e-3. These checks do not establish pointwise equivalence of every matched-filter output sample.

## Standalone TaylorF2 waveform generation

![Standalone TaylorF2 waveform generation before and after](waveform-before-after.png)

Parentheses show the range across three workers. Paired after/before values above 1 indicate higher capacity or generation rate.

| Route | Batch | Unit | Before | After | Paired after/before |
|:--|--:|:--|--:|--:|--:|
| Torch CPU · 1 thread | 1 | waveforms/s | 224.5 (223.3–224.7) | 255.2 (255.0–256.0) | 1.136 (1.136–1.147) |
| Torch CPU · 1 thread | 8 | waveforms/s | 897.5 (871.0–909.2) | 880.5 (829.9–889.4) | 0.968 (0.925–1.021) |
| Torch CPU · 1 thread | 32 | waveforms/s | 830.6 (830.5–830.6) | 899.8 (897.7–903.8) | 1.083 (1.081–1.088) |
| Torch CPU · 1 thread | 128 | waveforms/s | 649.5 (645.8–686.9) | 847.5 (847.4–913.7) | 1.312 (1.305–1.330) |
| Torch CPU · 1 thread | 512 | waveforms/s | 663.7 (638.3–676.7) | 767.1 (764.9–768.9) | 1.152 (1.136–1.202) |
| Torch CPU · 1 thread | 1024 | waveforms/s | 465.1 (436.4–465.7) | 539.8 (538.9–540.5) | 1.162 (1.157–1.237) |
| Torch CPU · 4 threads | 1 | waveforms/s | 224.1 (220.7–226.9) | 253.1 (251.2–256.6) | 1.129 (1.107–1.163) |
| Torch CPU · 4 threads | 8 | waveforms/s | 886.7 (882.2–972.5) | 990.5 (989.9–999.9) | 1.122 (1.018–1.128) |
| Torch CPU · 4 threads | 32 | waveforms/s | 2,084.1 (2,081.5–2,124.4) | 2,307.5 (2,289.6–2,325.6) | 1.109 (1.078–1.116) |
| Torch CPU · 4 threads | 128 | waveforms/s | 1,698.8 (1,694.3–1,763.0) | 2,491.2 (2,485.6–2,900.4) | 1.467 (1.410–1.712) |
| Torch CPU · 4 threads | 512 | waveforms/s | 1,686.1 (1,611.2–1,704.7) | 2,232.7 (2,219.2–2,240.4) | 1.324 (1.314–1.377) |
| Torch CPU · 4 threads | 1024 | waveforms/s | 1,140.1 (1,066.6–1,148.4) | 1,426.4 (1,424.9–1,435.8) | 1.259 (1.242–1.336) |
| Torch CUDA · 1 thread | 1 | waveforms/s | 81.4 (80.2–82.4) | 92.6 (91.3–93.2) | 1.132 (1.122–1.154) |
| Torch CUDA · 1 thread | 8 | waveforms/s | 638.8 (636.3–643.8) | 719.7 (717.8–727.0) | 1.127 (1.115–1.142) |
| Torch CUDA · 1 thread | 32 | waveforms/s | 2,561.6 (2,546.1–2,564.6) | 2,900.8 (2,880.7–2,913.0) | 1.132 (1.123–1.144) |
| Torch CUDA · 1 thread | 128 | waveforms/s | 10,222.2 (10,119.1–10,350.4) | 11,601.2 (11,451.9–11,705.3) | 1.132 (1.131–1.135) |
| Torch CUDA · 1 thread | 512 | waveforms/s | 33,679.6 (33,443.9–33,861.2) | 39,762.7 (38,646.8–40,011.2) | 1.181 (1.156–1.182) |
| Torch CUDA · 1 thread | 1024 | waveforms/s | 46,165.8 (45,804.9–46,432.6) | 59,643.6 (59,093.6–59,717.8) | 1.286 (1.280–1.302) |

Standalone waveform generation, measured in waveforms/second; this is not search capacity. Complete public get_fd_waveform_batch TaylorF2 calls, both polarizations, complex128 and 4097 frequency bins (delta_f=0.25 Hz, 20–1024 Hz). Host parameter conversion is included; output stays on the requested device and host copies are excluded. Triton and autograd are disabled. Every row and both polarizations pass native-scalar pointwise relative error <=2e-10 and native-scalar versus LAL CPU relative L2 <1e-11, with finite values, exact zero support, frequency spacing and epoch checks. Direct batch/LAL errors are reported without an additional numerical tolerance.

## Standard CPU controls

These baseline controls anchor live parity; the campaign did not time candidate standard CPU controls.

| Configured cores | Batch | Templates/core at real time, median (range) |
|--:|--:|--:|
| 1 | 1 | 62,513.9 (62,495.0–62,869.4) |
| 1 | 8 | 66,354.3 (66,198.2–66,365.4) |
| 1 | 32 | 65,766.7 (63,471.8–66,863.9) |
| 1 | 128 | 66,969.4 (66,435.1–67,245.1) |
| 1 | 512 | 67,528.3 (67,479.0–67,630.7) |
| 1 | 1024 | 67,354.9 (66,997.0–67,435.4) |
| 4 | 1 | 26,649.3 (25,207.4–27,379.3) |
| 4 | 8 | 38,937.5 (38,541.4–39,272.2) |
| 4 | 32 | 37,830.7 (37,281.9–38,335.9) |
| 4 | 128 | 40,296.9 (40,124.7–40,359.5) |
| 4 | 512 | 40,416.3 (40,280.9–41,063.7) |
| 4 | 1024 | 41,101.8 (40,815.2–41,138.8) |

All source identities, worker aggregates and paired ratios are in [report.json](report.json). [input-manifest.json](input-manifest.json) records SHA-256 hashes of every input. The waveform worker hash is checked against the local worker script; the orchestration script hash is a local provenance record. Live parity was also recomputed from the recorded triggers and norms.
