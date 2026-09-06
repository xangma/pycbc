# Paired diagnostic profiles

Diagnostic attribution only: instrumented durations are not throughput evidence, speedup estimates, or replicated timing measurements.

Each pstats file covers one warmed public call. Each Torch trace covers a separate warmed public call; do not add measurements across these calls.

cProfile self time excludes profiled callees; cumulative time includes callees. Nested cumulative times must not be added. CUDA CPU times can include launch or synchronization costs and are not GPU kernel durations.

Operator counts come from every complete cpu_op event in trace.json, not the truncated operators.txt table. Their inclusive CPU durations can overlap through nesting and must not be summed as wall time.

Chrome trace event durations are microseconds; displayTimeUnit is a viewer preference. Tables below use milliseconds for cProfile costs.

Absent functions have zero observed calls in that captured call; this does not establish that the implementation or capability is absent.

Waveform timing.json supplies workload metadata only; its timing samples and derived throughput are deliberately excluded from this report.

Baseline provenance is the supplied source-environment snapshot and embedded profile paths; no contemporaneous baseline command receipt or baseline script-hash receipt is available. Candidate command receipts and output/script hashes are validated against supplied source identity.

All values below are baseline → candidate. All 8 live and 4 waveform pairs passed validation.

Host: `len`. Source identities:

| Source | Commit | Tree |
|---|---|---|
| baseline | `dfd42bf76766cadca0eecf609a1eaeac73534676` | `b0d5ef6e3b39bfe246b4c416528839256edadf1f` |
| candidate | `97bf1614f3afe53a6e2edb4d8c7dff79e9661782` | `d9819f6f6ebf2998ac967178778cc1911757e455` |

## Live overlap validation and peak extraction

| Route | Batch | Disjoint validation cumulative ms | Span-comparison calls | Peak extraction cumulative ms |
|---|---:|---:|---:|---:|
| torch_cpu | 32 | 0 → 0 (unobserved) | 0 → 0 (unobserved) | 35.394 → 12.959 |
| torch_cpu | 1024 | 0 → 0 (unobserved) | 0 → 0 (unobserved) | 2079.986 → 401.874 |
| torch_cpu_native | 32 | 1.103 → 0.113 | 1,552 → 32 | 35.609 → 20.578 |
| torch_cpu_native | 1024 | 1013.982 → 2.746 | 1,573,376 → 1,024 | 2085.843 → 770.811 |
| torch_cuda | 32 | 0 → 0 (unobserved) | 0 → 0 (unobserved) | 0.221 → 0.265 |
| torch_cuda | 1024 | 0 → 0 (unobserved) | 0 → 0 (unobserved) | 10.158 → 10.137 |
| torch_cuda_native | 32 | 1.167 → 0.160 | 1,552 → 32 | 0.275 → 0.320 |
| torch_cuda_native | 1024 | 1055.894 → 2.776 | 1,573,376 → 1,024 | 11.043 → 11.185 |

## TaylorF2 phase evaluation and tensor arithmetic

Counts cover the whole public waveform call, including parameter and reference-frequency operations. The exact input shapes below distinguish these from frequency-grid operations.

| Device | Batch | Phase cumulative ms | Phase calls | log calls | addcmul calls | mul calls | add calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 32 | 5.495 → 2.179 | 2 → 2 | 6 → 6 | 30 → 30 | 371 → 309 | 160 → 100 |
| cpu | 1024 | 821.699 → 492.455 | 2 → 2 | 6 → 6 | 30 → 30 | 371 → 309 | 160 → 100 |
| cuda | 1 | 2.148 → 0.679 | 2 → 2 | 6 → 6 | 30 → 30 | 371 → 309 | 160 → 100 |
| cuda | 1024 | 2.131 → 0.763 | 2 → 2 | 6 → 6 | 30 → 30 | 371 → 309 | 160 → 100 |

A logarithm or Horner-call reduction is supported only if its observed count falls. Fewer multiply/add events with unchanged log/addcmul counts instead indicate reduced surrounding arithmetic. Operation counts alone do not establish numerical equivalence or causal throughput improvement.

## profile-live-torch_cpu-b32-t1

| Function | Calls | Self ms | Cumulative ms |
|---|---:|---:|---:|
| `_process_batch` | 2 → 2 | 0.256 → 0.259 | 108.818 → 85.894 |
| `_torch_batch_peak_magnitudes` | 1 → 1 | 0.015 → 0.014 | 0.097 → 0.094 |
| `_torch_batch_peak_values` | 1 → 1 | 0.136 → 1.857 | 35.394 → 12.959 |
| `process_data` | 1 → 1 | 0.011 → 0.011 | 109.052 → 86.114 |
| `batch_correlate_execute` | 1 → 1 | 0.117 → 0.103 | 5.551 → 5.624 |

Profile input paths and SHA256 hashes:

| Side | File | SHA256 |
|---|---|---|
| baseline | [python.pstats](<profiles-baseline/torch_cpu-b32/python.pstats>) | `854d3af3e3504fbb6084b3a344e57584dad3532cfad28ae3bc1d849e32dbeec7` |
| baseline | [python.txt](<profiles-baseline/torch_cpu-b32/python.txt>) | `2fdefa169a766a62df16c9a7d7fb5fc301b90506f965a7347180b30cf4dcf97f` |
| baseline | [operators.txt](<profiles-baseline/torch_cpu-b32/operators.txt>) | `1424be00937dcbdf99a9055fe7698fd420066011e28d614d98386a8320f72b10` |
| baseline | [trace.json](<profiles-baseline/torch_cpu-b32/trace.json>) | `5c3d771ec389e049d2eaeeea22f2255ce883fe18aca6d0057b91ca9a57f782af` |
| baseline | [status.json](<profiles-baseline/torch_cpu-b32/status.json>) | `0fd7f9cab5b535579b9f4e28f130a9532b5ffc755a5199b31060883e6051dfd2` |
| candidate | [python.pstats](<postchecks/profiles/profile-live-torch_cpu-b32-t1/python.pstats>) | `b6fb6caa38aa4654e52fb340833b34b2e6989a4214ce01eb95559563c8cc9e4d` |
| candidate | [python.txt](<postchecks/profiles/profile-live-torch_cpu-b32-t1/python.txt>) | `88443112b80e82e246f8c9b6069562185cdf5bc693b22ac04d0abf39ef1af7ae` |
| candidate | [operators.txt](<postchecks/profiles/profile-live-torch_cpu-b32-t1/operators.txt>) | `adf5bc6d1b145867973dcbad76c8ca83f16caf1cda3b482abb42898700bf8889` |
| candidate | [trace.json](<postchecks/profiles/profile-live-torch_cpu-b32-t1/trace.json>) | `722bc78ca5e0f4cd9125896c396ac06917ad9fe9db2fda0f9c42108a6fd5e56b` |
| candidate | [status.json](<postchecks/profiles/profile-live-torch_cpu-b32-t1/status.json>) | `0fd7f9cab5b535579b9f4e28f130a9532b5ffc755a5199b31060883e6051dfd2` |
| candidate | [files-sha256.json](<postchecks/profiles/profile-live-torch_cpu-b32-t1/files-sha256.json>) | `cb602c607c5e7d9497c4a78c4595ef7d07d940516d21d184ceab04c94e63986c` |
| candidate | [worker.log](<postchecks/profiles/profile-live-torch_cpu-b32-t1/worker.log>) | `491f518cd531139db95d6f1f27e4039c0375b07ef5f6cad78b9bbd2ec70887e6` |

## profile-live-torch_cpu-b1024-t1

| Function | Calls | Self ms | Cumulative ms |
|---|---:|---:|---:|
| `_process_batch` | 2 → 2 | 61.770 → 0.594 | 4860.145 → 3102.716 |
| `_torch_batch_peak_magnitudes` | 1 → 1 | 0.019 → 0.016 | 1.569 → 1.593 |
| `_torch_batch_peak_values` | 1 → 1 | 124.864 → 51.831 | 2079.986 → 401.874 |
| `process_data` | 1 → 1 | 0.012 → 0.011 | 4860.371 → 3102.929 |
| `batch_correlate_execute` | 1 → 1 | 0.914 → 0.898 | 560.236 → 548.013 |

Profile input paths and SHA256 hashes:

| Side | File | SHA256 |
|---|---|---|
| baseline | [python.pstats](<profiles-baseline/torch_cpu-b1024/python.pstats>) | `e4116ab1fe662840d0f288139f117672d218bb0afffbd829430cf8826da2ecc7` |
| baseline | [python.txt](<profiles-baseline/torch_cpu-b1024/python.txt>) | `c3e8ab8d9bebfc4200ca776a5cf1d7da52882ad5cde87dab7cc5fdae1ebed608` |
| baseline | [operators.txt](<profiles-baseline/torch_cpu-b1024/operators.txt>) | `7ea88767e5628185034ebbc47a2285ffcb8f94284e012ba08c60f0efdbc5eb94` |
| baseline | [trace.json](<profiles-baseline/torch_cpu-b1024/trace.json>) | `d0a189781de83bc569c29c92bac16d07168dfea9dc304c954d4aaece8877d36b` |
| baseline | [status.json](<profiles-baseline/torch_cpu-b1024/status.json>) | `0fd7f9cab5b535579b9f4e28f130a9532b5ffc755a5199b31060883e6051dfd2` |
| candidate | [python.pstats](<postchecks/profiles/profile-live-torch_cpu-b1024-t1/python.pstats>) | `842e6c6cd6d94683b31dbb2d43a2d4e3613ef786cdfba0f10d9f4c39af4ef179` |
| candidate | [python.txt](<postchecks/profiles/profile-live-torch_cpu-b1024-t1/python.txt>) | `f8ad831855596c31ca8e960f7e304c4a760f7c9b293bde5b230836001db66df4` |
| candidate | [operators.txt](<postchecks/profiles/profile-live-torch_cpu-b1024-t1/operators.txt>) | `670960997e883651355109bb782c27274f43054c798a89478034dd875bbc82ec` |
| candidate | [trace.json](<postchecks/profiles/profile-live-torch_cpu-b1024-t1/trace.json>) | `83ae829d771000f15b84a97d54a901b90167cf85293756f4ed079df23bcb7206` |
| candidate | [status.json](<postchecks/profiles/profile-live-torch_cpu-b1024-t1/status.json>) | `0fd7f9cab5b535579b9f4e28f130a9532b5ffc755a5199b31060883e6051dfd2` |
| candidate | [files-sha256.json](<postchecks/profiles/profile-live-torch_cpu-b1024-t1/files-sha256.json>) | `ab1f62b932e9e9c33deb703cffa904f93c6221bfb34596558d7a37ef0d50327e` |
| candidate | [worker.log](<postchecks/profiles/profile-live-torch_cpu-b1024-t1/worker.log>) | `4fe34bc7bb66a6cbf015c11b047e089acdaaafe5d847e8dce7b66ba6406b6e3c` |

## profile-live-torch_cpu_native-b32-t1

| Function | Calls | Self ms | Cumulative ms |
|---|---:|---:|---:|
| `_process_batch` | 2 → 2 | 0.248 → 1.850 | 66.592 → 52.079 |
| `_torch_batch_peak_magnitudes` | 1 → 1 | 0.014 → 0.014 | 0.098 → 0.100 |
| `_torch_batch_peak_values` | 1 → 1 | 0.141 → 9.499 | 35.609 → 20.578 |
| `process_data` | 1 → 1 | 0.010 → 0.011 | 66.813 → 52.300 |
| `_batch_outputs_are_disjoint` | 1 → 1 | 0.040 → 0.029 | 1.103 → 0.113 |
| `_batch_tensor_contract` | 65 → 65 | 0.276 → 0.261 | 0.568 → 0.563 |
| `_has_autograd_state` | 65 → 65 | 0.073 → 0.071 | 0.150 → 0.148 |
| `_logical_storage_span` | 65 → 65 | 0.043 → 0.048 | 0.065 → 0.071 |
| `_same_array_tensors` | 2 → 2 | 0.057 → 0.057 | 0.085 → 0.083 |
| `_spans_overlap` | 1,552 → 32 | 0.325 → 0.013 | 0.325 → 0.013 |
| `batch_correlate_execute` | 1 → 1 | 0.009 → 0.010 | 3.931 → 2.902 |

Profile input paths and SHA256 hashes:

| Side | File | SHA256 |
|---|---|---|
| baseline | [python.pstats](<profiles-baseline/torch_cpu_native-b32/python.pstats>) | `69e1793a47e1fc25d84ec319c02cf736171d2e3e4f7f4bb6284be425b5d67edd` |
| baseline | [python.txt](<profiles-baseline/torch_cpu_native-b32/python.txt>) | `436b3ff02f6fc4fdd0c52feacbfeb0d0a26d5c0e885ac22b0579f2b5e8f978b7` |
| baseline | [operators.txt](<profiles-baseline/torch_cpu_native-b32/operators.txt>) | `703fa9fad25e3b286814c10e4546ad0721b87811bfbc01d319e50aebc304228b` |
| baseline | [trace.json](<profiles-baseline/torch_cpu_native-b32/trace.json>) | `1131b9f9e5605ba3fd5945f6a2e3b83e083ab6e0ef4f8c5a209076abb7767211` |
| baseline | [status.json](<profiles-baseline/torch_cpu_native-b32/status.json>) | `0fd7f9cab5b535579b9f4e28f130a9532b5ffc755a5199b31060883e6051dfd2` |
| candidate | [python.pstats](<postchecks/profiles/profile-live-torch_cpu_native-b32-t1/python.pstats>) | `ea74516f550fe393c20ff2e786eadf88e8d5ef1e8c072d3f89ac2ccc6fd717bc` |
| candidate | [python.txt](<postchecks/profiles/profile-live-torch_cpu_native-b32-t1/python.txt>) | `d0fb68533e95b8921e6d90ebfc4ac7587c62e74f6d7ec3d493743b3c5b7fe926` |
| candidate | [operators.txt](<postchecks/profiles/profile-live-torch_cpu_native-b32-t1/operators.txt>) | `6e9d6af536d7bacc1953d4f3b78be96e41d34e0d4ee6857018680fe9fbe3a00b` |
| candidate | [trace.json](<postchecks/profiles/profile-live-torch_cpu_native-b32-t1/trace.json>) | `078ff6d36092357aafffd0ba555f201cd44341846ed325fac6bc38bf9c78b29a` |
| candidate | [status.json](<postchecks/profiles/profile-live-torch_cpu_native-b32-t1/status.json>) | `0fd7f9cab5b535579b9f4e28f130a9532b5ffc755a5199b31060883e6051dfd2` |
| candidate | [files-sha256.json](<postchecks/profiles/profile-live-torch_cpu_native-b32-t1/files-sha256.json>) | `91e6c8d4ea7dbd1e083b31ae487e77a7b36e55a9d046f212d8015fdc6a790869` |
| candidate | [worker.log](<postchecks/profiles/profile-live-torch_cpu_native-b32-t1/worker.log>) | `1a597fe85100da6f02f3599e5adb80a13b94c26c7d28a8528f1b5a93f9e63c52` |

## profile-live-torch_cpu_native-b1024-t1

| Function | Calls | Self ms | Cumulative ms |
|---|---:|---:|---:|
| `_process_batch` | 2 → 2 | 60.653 → 2.692 | 4088.200 → 1704.581 |
| `_torch_batch_peak_magnitudes` | 1 → 1 | 0.017 → 0.014 | 1.587 → 1.579 |
| `_torch_batch_peak_values` | 1 → 1 | 124.057 → 258.795 | 2085.843 → 770.811 |
| `process_data` | 1 → 1 | 0.011 → 0.011 | 4088.423 → 1704.806 |
| `_batch_outputs_are_disjoint` | 1 → 1 | 2.045 → 0.637 | 1013.982 → 2.746 |
| `_batch_tensor_contract` | 2,049 → 2,049 | 7.567 → 7.464 | 15.881 → 15.579 |
| `_has_autograd_state` | 2,049 → 2,049 | 1.823 → 1.811 | 4.011 → 3.927 |
| `_logical_storage_span` | 2,049 → 2,049 | 1.146 → 1.148 | 1.804 → 1.826 |
| `_same_array_tensors` | 2 → 2 | 1.151 → 1.154 | 1.664 → 1.592 |
| `_spans_overlap` | 1,573,376 → 1,024 | 317.535 → 0.264 | 317.535 → 0.264 |
| `batch_correlate_execute` | 1 → 1 | 0.009 → 0.009 | 1093.457 → 82.239 |

Profile input paths and SHA256 hashes:

| Side | File | SHA256 |
|---|---|---|
| baseline | [python.pstats](<profiles-baseline/torch_cpu_native-b1024/python.pstats>) | `cb2f50734f789dfc449fe87a5514344a398bdfa527280c81c6d785d33492bc0c` |
| baseline | [python.txt](<profiles-baseline/torch_cpu_native-b1024/python.txt>) | `f68ad4f9ea6542e9b824fc2ec3911031e314e79a2b61af7702085589f2617ae1` |
| baseline | [operators.txt](<profiles-baseline/torch_cpu_native-b1024/operators.txt>) | `75c8f7428beb6fa42c721b6b423b5f914658ad3692a186e0f3c385634fdecbb5` |
| baseline | [trace.json](<profiles-baseline/torch_cpu_native-b1024/trace.json>) | `13fa4e42f7a77cb26e25e7c88b9c4f39f02601883bc61a8dfa75225d40119c23` |
| baseline | [status.json](<profiles-baseline/torch_cpu_native-b1024/status.json>) | `0fd7f9cab5b535579b9f4e28f130a9532b5ffc755a5199b31060883e6051dfd2` |
| candidate | [python.pstats](<postchecks/profiles/profile-live-torch_cpu_native-b1024-t1/python.pstats>) | `e04897005f92ebdd939ae1cba0645bb0379a675c957069cc3abbbeb7779a11ca` |
| candidate | [python.txt](<postchecks/profiles/profile-live-torch_cpu_native-b1024-t1/python.txt>) | `39b2ff972bd8fb60996bed9e1f36990b0ba005c8dd318e43b67e7dbad5543bf3` |
| candidate | [operators.txt](<postchecks/profiles/profile-live-torch_cpu_native-b1024-t1/operators.txt>) | `43435f9211ab496380805f9f6ea3727b88d86d78beb08b7bdca973ed7ef2b25e` |
| candidate | [trace.json](<postchecks/profiles/profile-live-torch_cpu_native-b1024-t1/trace.json>) | `9b7e478fab0108793b933568a93a7ea5790aa8dea1e480bdf23abbc8f91ccd94` |
| candidate | [status.json](<postchecks/profiles/profile-live-torch_cpu_native-b1024-t1/status.json>) | `0fd7f9cab5b535579b9f4e28f130a9532b5ffc755a5199b31060883e6051dfd2` |
| candidate | [files-sha256.json](<postchecks/profiles/profile-live-torch_cpu_native-b1024-t1/files-sha256.json>) | `90183e84912fe0b51d26358356f441fa6137732fd74f0b49bab02c0ca20c4613` |
| candidate | [worker.log](<postchecks/profiles/profile-live-torch_cpu_native-b1024-t1/worker.log>) | `b0b807376175d1ac9b40bb699dbfddc780d1ea8e8018cc4172595db1578f48c2` |

## profile-live-torch_cuda-b32-t1

| Function | Calls | Self ms | Cumulative ms |
|---|---:|---:|---:|
| `_process_batch` | 2 → 2 | 0.132 → 0.152 | 1.275 → 1.328 |
| `_torch_batch_peak_magnitudes` | 1 → 1 | 0.006 → 0.007 | 0.081 → 0.072 |
| `_torch_batch_peak_values` | 1 → 1 | 0.077 → 0.103 | 0.221 → 0.265 |
| `_try_torch_cuda_native_batch_peak_values` | 1 → 1 | 0.002 → 0.002 | 0.007 → 0.006 |
| `process_data` | 1 → 1 | 0.009 → 0.009 | 1.456 → 1.516 |
| `batch_correlate_execute` | 1 → 1 | 0.066 → 0.065 | 0.567 → 0.546 |

Profile input paths and SHA256 hashes:

| Side | File | SHA256 |
|---|---|---|
| baseline | [python.pstats](<profiles-baseline/torch_cuda-b32/python.pstats>) | `1ce48a003d21ace463b7633042db0308e58c99ef34884140f11172f6a0cca194` |
| baseline | [python.txt](<profiles-baseline/torch_cuda-b32/python.txt>) | `d71d0607c9530de780a665a22ed8e548543914580cb6ca7e4d9894f34707da2f` |
| baseline | [operators.txt](<profiles-baseline/torch_cuda-b32/operators.txt>) | `08bf3afd0a80cba532a59ed43c4ee56fa5516fd46193b6c5b16cef0d90dc41a3` |
| baseline | [trace.json](<profiles-baseline/torch_cuda-b32/trace.json>) | `2a4986c6924c5d8db5c88d6ca1ec0007c34083f7e83be8c9c9c7024b99c93b86` |
| baseline | [status.json](<profiles-baseline/torch_cuda-b32/status.json>) | `0fd7f9cab5b535579b9f4e28f130a9532b5ffc755a5199b31060883e6051dfd2` |
| candidate | [python.pstats](<postchecks/profiles/profile-live-torch_cuda-b32-t1/python.pstats>) | `0d21f5813a8a815b5397989d84d9c97f5556b6ed6c0b64d560cc9a9c29f7fe8c` |
| candidate | [python.txt](<postchecks/profiles/profile-live-torch_cuda-b32-t1/python.txt>) | `f662c478f03d9ec369853a10a9c42edf382140c86ee099fa4e76a960618bfe50` |
| candidate | [operators.txt](<postchecks/profiles/profile-live-torch_cuda-b32-t1/operators.txt>) | `ffc2d8b13c053fcda04e22877fdb6226968ee6f72eaa6fc4aa3afa8d69208743` |
| candidate | [trace.json](<postchecks/profiles/profile-live-torch_cuda-b32-t1/trace.json>) | `d36e5663a8a67c1baec4ecc4c3b2eafc3118173fea5d8e864a807a3a4fd407ff` |
| candidate | [status.json](<postchecks/profiles/profile-live-torch_cuda-b32-t1/status.json>) | `0fd7f9cab5b535579b9f4e28f130a9532b5ffc755a5199b31060883e6051dfd2` |
| candidate | [files-sha256.json](<postchecks/profiles/profile-live-torch_cuda-b32-t1/files-sha256.json>) | `c0e2cc3069f65a8991685617e0335c617820ae456c0f9677c93820814ccc793e` |
| candidate | [worker.log](<postchecks/profiles/profile-live-torch_cuda-b32-t1/worker.log>) | `a13fa5b97e65c093771e62af44873df1525fb3a663f9a90bfac0cb472843b634` |

## profile-live-torch_cuda-b1024-t1

| Function | Calls | Self ms | Cumulative ms |
|---|---:|---:|---:|
| `_process_batch` | 2 → 2 | 0.258 → 0.262 | 25.549 → 25.777 |
| `_torch_batch_peak_magnitudes` | 1 → 1 | 0.007 → 0.007 | 1.517 → 1.596 |
| `_torch_batch_peak_values` | 1 → 1 | 0.070 → 0.093 | 10.158 → 10.137 |
| `_try_torch_cuda_native_batch_peak_values` | 1 → 1 | 0.003 → 0.003 | 0.007 → 0.007 |
| `process_data` | 1 → 1 | 0.009 → 0.009 | 25.729 → 25.960 |
| `batch_correlate_execute` | 1 → 1 | 0.735 → 0.735 | 11.767 → 11.902 |

Profile input paths and SHA256 hashes:

| Side | File | SHA256 |
|---|---|---|
| baseline | [python.pstats](<profiles-baseline/torch_cuda-b1024/python.pstats>) | `9a483134621328391decc1a3437341baefa324086f9f12c8282a8438bfd3a6bb` |
| baseline | [python.txt](<profiles-baseline/torch_cuda-b1024/python.txt>) | `d5d2ae33e990c994b68ec97f20333e54a867e054f7f120883776fe190a3943cb` |
| baseline | [operators.txt](<profiles-baseline/torch_cuda-b1024/operators.txt>) | `f6625cbe6bd79e3d4190eb822c79373bfb62bbeb1a10ae8d167395bc6a31d229` |
| baseline | [trace.json](<profiles-baseline/torch_cuda-b1024/trace.json>) | `be796bbe3799c2c21a20ee3393ac1d238adb123ce5de4242de80a920af9c5781` |
| baseline | [status.json](<profiles-baseline/torch_cuda-b1024/status.json>) | `0fd7f9cab5b535579b9f4e28f130a9532b5ffc755a5199b31060883e6051dfd2` |
| candidate | [python.pstats](<postchecks/profiles/profile-live-torch_cuda-b1024-t1/python.pstats>) | `773cb8970e0e368d07cdb96cad2902e5caafda9ef7d0f67b92a8d1711e79a4ed` |
| candidate | [python.txt](<postchecks/profiles/profile-live-torch_cuda-b1024-t1/python.txt>) | `4ff6fdb3e79cae8b9a1f5a22b2db880d7c849ae82176434f9ff6ed569fed99d0` |
| candidate | [operators.txt](<postchecks/profiles/profile-live-torch_cuda-b1024-t1/operators.txt>) | `a11ed7126445b07c74f3ff11dc1e5616082846304b22cee6c315977717be4a84` |
| candidate | [trace.json](<postchecks/profiles/profile-live-torch_cuda-b1024-t1/trace.json>) | `38872266e90e2740f48731612680e280115b1fe5e160494325e77dce166583bc` |
| candidate | [status.json](<postchecks/profiles/profile-live-torch_cuda-b1024-t1/status.json>) | `0fd7f9cab5b535579b9f4e28f130a9532b5ffc755a5199b31060883e6051dfd2` |
| candidate | [files-sha256.json](<postchecks/profiles/profile-live-torch_cuda-b1024-t1/files-sha256.json>) | `c5ead74e1ae94827849f2d2d712dfee1830477f4bd055e37104e7113deb819e5` |
| candidate | [worker.log](<postchecks/profiles/profile-live-torch_cuda-b1024-t1/worker.log>) | `0abed4f313d58ffa43e2e704a38073b0cc77a9cd95c28d1995a346ab71a8e7a0` |

## profile-live-torch_cuda_native-b32-t1

| Function | Calls | Self ms | Cumulative ms |
|---|---:|---:|---:|
| `_process_batch` | 2 → 2 | 0.126 → 0.171 | 2.926 → 2.310 |
| `_torch_batch_peak_magnitudes` | 1 → 1 | 0.006 → 0.007 | 0.072 → 0.094 |
| `_torch_batch_peak_values` | 1 → 1 | 0.014 → 0.014 | 0.275 → 0.320 |
| `_try_torch_cuda_native_batch_peak_values` | 1 → 1 | 0.041 → 0.044 | 0.256 → 0.291 |
| `process_data` | 1 → 1 | 0.009 → 0.010 | 3.109 → 2.544 |
| `_batch_outputs_are_disjoint` | 1 → 1 | 0.039 → 0.043 | 1.167 → 0.160 |
| `_cuda_batch_tensor_contract` | 66 → 66 | 0.240 → 0.251 | 0.503 → 0.526 |
| `_has_autograd_state` | 66 → 66 | 0.059 → 0.064 | 0.134 → 0.143 |
| `_logical_storage_span` | 65 → 65 | 0.041 → 0.074 | 0.063 → 0.110 |
| `_same_array_tensors` | 2 → 2 | 0.040 → 0.041 | 0.053 → 0.055 |
| `_spans_overlap` | 1,552 → 32 | 0.392 → 0.009 | 0.392 → 0.009 |
| `batch_correlate_execute` | 1 → 1 | 0.006 → 0.007 | 2.167 → 1.358 |
| `standard_peak_tensor` | 1 → 1 | 0.046 → 0.062 | 0.138 → 0.165 |

Profile input paths and SHA256 hashes:

| Side | File | SHA256 |
|---|---|---|
| baseline | [python.pstats](<profiles-baseline/torch_cuda_native-b32/python.pstats>) | `b5d63d27ad407c2aed77d4f735d000bd73a9a2cd901f41eadd0f7560be2cbf9e` |
| baseline | [python.txt](<profiles-baseline/torch_cuda_native-b32/python.txt>) | `6f8b993728aa6da43f9a915070f7e67284d0f6de5a8d773dc8a27643c723ded6` |
| baseline | [operators.txt](<profiles-baseline/torch_cuda_native-b32/operators.txt>) | `1ef340649bc93b4d1eb0ccc4176bb84599ae857e4f12e9b1a126bc888ae048d2` |
| baseline | [trace.json](<profiles-baseline/torch_cuda_native-b32/trace.json>) | `9d919a69972ff7ecc39b666a25b90232573e8cd51cd74a7132eeeb95f6fa5ee2` |
| baseline | [status.json](<profiles-baseline/torch_cuda_native-b32/status.json>) | `0fd7f9cab5b535579b9f4e28f130a9532b5ffc755a5199b31060883e6051dfd2` |
| candidate | [python.pstats](<postchecks/profiles/profile-live-torch_cuda_native-b32-t1/python.pstats>) | `9322f9b68a476f1b1f66960d67fe1cb5a0598cdeaee0e11c91424078cfdeaed2` |
| candidate | [python.txt](<postchecks/profiles/profile-live-torch_cuda_native-b32-t1/python.txt>) | `118ff73fbcf890c0cec65957db25d780513c6edb8552e4d884da67c22e1fabd8` |
| candidate | [operators.txt](<postchecks/profiles/profile-live-torch_cuda_native-b32-t1/operators.txt>) | `2be6e2b2bd93381fc001f8adcdc860188c12a7239c96dcd936a9f49664e01439` |
| candidate | [trace.json](<postchecks/profiles/profile-live-torch_cuda_native-b32-t1/trace.json>) | `712fca891e15e3cf52a83788cfcfdd5f7a1bb696b7032750998a9860ebfcac13` |
| candidate | [status.json](<postchecks/profiles/profile-live-torch_cuda_native-b32-t1/status.json>) | `0fd7f9cab5b535579b9f4e28f130a9532b5ffc755a5199b31060883e6051dfd2` |
| candidate | [files-sha256.json](<postchecks/profiles/profile-live-torch_cuda_native-b32-t1/files-sha256.json>) | `f8923660bda3c3adf2a52108aadf7ea6e5572877cd2e1f95fdba1c6cfc0888bd` |
| candidate | [worker.log](<postchecks/profiles/profile-live-torch_cuda_native-b32-t1/worker.log>) | `7e205af8b968eb06dba9371fce01861ad7dabf824dfa0d583725cedb159e3a43` |

## profile-live-torch_cuda_native-b1024-t1

| Function | Calls | Self ms | Cumulative ms |
|---|---:|---:|---:|
| `_process_batch` | 2 → 2 | 0.355 → 0.273 | 1095.594 → 41.870 |
| `_torch_batch_peak_magnitudes` | 1 → 1 | 0.008 → 0.007 | 1.587 → 1.548 |
| `_torch_batch_peak_values` | 1 → 1 | 0.016 → 0.015 | 11.043 → 11.185 |
| `_try_torch_cuda_native_batch_peak_values` | 1 → 1 | 0.047 → 0.042 | 11.022 → 11.164 |
| `process_data` | 1 → 1 | 0.011 → 0.010 | 1095.798 → 42.056 |
| `_batch_outputs_are_disjoint` | 1 → 1 | 2.024 → 0.645 | 1055.894 → 2.776 |
| `_cuda_batch_tensor_contract` | 2,050 → 2,050 | 6.779 → 6.597 | 14.414 → 14.170 |
| `_has_autograd_state` | 2,050 → 2,050 | 1.766 → 1.664 | 3.917 → 3.826 |
| `_logical_storage_span` | 2,049 → 2,049 | 1.137 → 1.159 | 1.786 → 1.850 |
| `_same_array_tensors` | 2 → 2 | 0.985 → 0.961 | 1.344 → 1.283 |
| `_spans_overlap` | 1,573,376 → 1,024 | 387.795 → 0.185 | 387.795 → 0.185 |
| `batch_correlate_execute` | 1 → 1 | 0.007 → 0.007 | 1080.677 → 27.001 |
| `standard_peak_tensor` | 1 → 1 | 0.056 → 0.047 | 0.179 → 0.142 |

Profile input paths and SHA256 hashes:

| Side | File | SHA256 |
|---|---|---|
| baseline | [python.pstats](<profiles-baseline/torch_cuda_native-b1024/python.pstats>) | `153e9f51ef09df4bccdd312c2467276462b9c2bbb403499cba00b492924363b4` |
| baseline | [python.txt](<profiles-baseline/torch_cuda_native-b1024/python.txt>) | `7c0e0d7af2bbd330a4932f31e440f89bddeb99f150757b1b8edc181a97f4660c` |
| baseline | [operators.txt](<profiles-baseline/torch_cuda_native-b1024/operators.txt>) | `faa02adb335d22f724fda23c8de9f454ae0de59147e7f63c9cdb14ecae8933b0` |
| baseline | [trace.json](<profiles-baseline/torch_cuda_native-b1024/trace.json>) | `6b70003a68390a1dff95db848025150afadae147dcd9db3138cadc9ae19651b3` |
| baseline | [status.json](<profiles-baseline/torch_cuda_native-b1024/status.json>) | `0fd7f9cab5b535579b9f4e28f130a9532b5ffc755a5199b31060883e6051dfd2` |
| candidate | [python.pstats](<postchecks/profiles/profile-live-torch_cuda_native-b1024-t1/python.pstats>) | `d8ef97c7a225a72f89812aa1ccaad6030c82f837a74137b865a7e3d34639b75b` |
| candidate | [python.txt](<postchecks/profiles/profile-live-torch_cuda_native-b1024-t1/python.txt>) | `b1e356d9c6936987ea5b4418e9f261bd2693359462c230bc4235de7ac1507293` |
| candidate | [operators.txt](<postchecks/profiles/profile-live-torch_cuda_native-b1024-t1/operators.txt>) | `e034ad79d36997de882c4d5746f43dd77c40d9265c9ff01a327ce25c5a402e4f` |
| candidate | [trace.json](<postchecks/profiles/profile-live-torch_cuda_native-b1024-t1/trace.json>) | `a2f92dbeec256049f22917e6cb31766a70862f62ddbe56c18e6f4e55e75d3fa9` |
| candidate | [status.json](<postchecks/profiles/profile-live-torch_cuda_native-b1024-t1/status.json>) | `0fd7f9cab5b535579b9f4e28f130a9532b5ffc755a5199b31060883e6051dfd2` |
| candidate | [files-sha256.json](<postchecks/profiles/profile-live-torch_cuda_native-b1024-t1/files-sha256.json>) | `4406423bc55614d1e5d5279e0deb285c7f120aae7b15f8691ba98e452b17fc11` |
| candidate | [worker.log](<postchecks/profiles/profile-live-torch_cuda_native-b1024-t1/worker.log>) | `9168d36df0dd7008132992d334118df638ecc025ad17523ba2c1a8462f3b6405` |

## profile-waveform-cpu-b32-t1

| Function | Calls | Self ms | Cumulative ms |
|---|---:|---:|---:|
| `_batch_validate` | 6 → 6 | 0.060 → 0.058 | 0.396 → 0.386 |
| `_evaluate_phase_polynomial` | 2 → 2 | 3.853 → 0.594 | 5.495 → 2.179 |
| `taylorf2_aligned_phasing` | 1 → 1 | 0.169 → 0.155 | 1.635 → 1.591 |
| `taylorf2_fd_batch` | 1 → 1 | 2.730 → 4.219 | 21.250 → 19.322 |

| Operator | Input dimensions | Calls |
|---|---|---:|
| `aten::add` | `[[32,1],[32,1],[]]` | 32 → 2 |
| `aten::add` | `[[32,1],[32,4017],[]]` | 16 → 2 |
| `aten::add` | `[[32,4017],[1,4017],[]]` | 1 → 1 |
| `aten::add` | `[[32,4017],[32,4017],[]]` | 17 → 1 |
| `aten::add` | `[[32],[32],[]]` | 51 → 51 |
| `aten::add` | `[[32],[],[]]` | 43 → 43 |
| `aten::addcmul` | `[[32,1],[32,1],[32,1],[]]` | 15 → 15 |
| `aten::addcmul` | `[[32,1],[32,1],[32,4017],[]]` | 0 → 1 |
| `aten::addcmul` | `[[32,1],[32,4017],[32,4017],[]]` | 0 → 12 |
| `aten::addcmul` | `[[32,4017],[32,4017],[32,4017],[]]` | 15 → 2 |
| `aten::log` | `[[32,1]]` | 1 → 1 |
| `aten::log` | `[[32,4017]]` | 1 → 1 |
| `aten::log` | `[[32]]` | 4 → 4 |
| `aten::mul` | `[[1,4017],[]]` | 2 → 2 |
| `aten::mul` | `[[16,32],[32]]` | 3 → 3 |
| `aten::mul` | `[[32,1],[1,4017]]` | 1 → 1 |
| `aten::mul` | `[[32,1],[32,1]]` | 36 → 5 |
| `aten::mul` | `[[32,1],[32,4017]]` | 37 → 7 |
| `aten::mul` | `[[32,1],[]]` | 1 → 1 |
| `aten::mul` | `[[32,4017],[32,1]]` | 2 → 2 |
| `aten::mul` | `[[32,4017],[32,4017]]` | 5 → 4 |
| `aten::mul` | `[[32,4017],[]]` | 1 → 1 |
| `aten::mul` | `[[32],[32]]` | 172 → 172 |
| `aten::mul` | `[[32],[]]` | 111 → 111 |

Profile input paths and SHA256 hashes:

| Side | File | SHA256 |
|---|---|---|
| baseline | [python.pstats](<wave-profiles-baseline/cpu-b32/python.pstats>) | `ddaa1c303420b06e7102fd8cc9db0feffb3d4295657522f7a69aed57c024582b` |
| baseline | [python.txt](<wave-profiles-baseline/cpu-b32/python.txt>) | `b1cc614fd5a3a524b2b252682893e66de41f287aed22dc8fd617794dec19cbfc` |
| baseline | [operators.txt](<wave-profiles-baseline/cpu-b32/operators.txt>) | `0d0334f102c4c782e6164be0861e401eaf9c0b19d04113aaa1b0c9d6a50e21c5` |
| baseline | [trace.json](<wave-profiles-baseline/cpu-b32/trace.json>) | `9894080848d13fd87ef5b002c9349b79bf33c395ada4de94e1df7979a51f2e5c` |
| baseline | [timing.json](<wave-profiles-baseline/cpu-b32/timing.json>) | `a13d52ad752c2b87cf5908e857eef167e07bd6e1a42ce0e50748da178c856f25` |
| candidate | [python.pstats](<postchecks/profiles/profile-waveform-cpu-b32-t1/python.pstats>) | `32769381bb16acecba594e5f92e75029c4f8e5f67f75e6f91ab525bd4391b5d9` |
| candidate | [python.txt](<postchecks/profiles/profile-waveform-cpu-b32-t1/python.txt>) | `a43245fd03f5b31122f1e37119e8ade36a05f825a7e20b3f2f553e035ef229e4` |
| candidate | [operators.txt](<postchecks/profiles/profile-waveform-cpu-b32-t1/operators.txt>) | `a8ae511a50c2ae7f7e3775d94659d909f58b28ea81a4b7f9e779df35cab06191` |
| candidate | [trace.json](<postchecks/profiles/profile-waveform-cpu-b32-t1/trace.json>) | `b0cd609a6309b34848ce45547c7e355af779d0e347a5f5f45577e555d2e63ebf` |
| candidate | [timing.json](<postchecks/profiles/profile-waveform-cpu-b32-t1/timing.json>) | `891e4304993a9045a3d61abde100ab1eeb61ab9754e60617073c80e894cb9b86` |
| candidate | [files-sha256.json](<postchecks/profiles/profile-waveform-cpu-b32-t1/files-sha256.json>) | `7983402a6ac7b9786cbd768d163ccf1ef430480d2f690fb50b9df89726088811` |

## profile-waveform-cpu-b1024-t1

| Function | Calls | Self ms | Cumulative ms |
|---|---:|---:|---:|
| `_batch_validate` | 6 → 6 | 0.067 → 0.063 | 0.444 → 0.440 |
| `_evaluate_phase_polynomial` | 2 → 2 | 712.520 → 191.599 | 821.699 → 492.455 |
| `taylorf2_aligned_phasing` | 1 → 1 | 0.207 → 0.197 | 1.998 → 1.951 |
| `taylorf2_fd_batch` | 1 → 1 | 695.528 → 692.926 | 2226.056 → 1849.769 |

| Operator | Input dimensions | Calls |
|---|---|---:|
| `aten::add` | `[[1024,1],[1024,1],[]]` | 32 → 2 |
| `aten::add` | `[[1024,1],[1024,4017],[]]` | 16 → 2 |
| `aten::add` | `[[1024,4017],[1,4017],[]]` | 1 → 1 |
| `aten::add` | `[[1024,4017],[1024,4017],[]]` | 17 → 1 |
| `aten::add` | `[[1024],[1024],[]]` | 51 → 51 |
| `aten::add` | `[[1024],[],[]]` | 43 → 43 |
| `aten::addcmul` | `[[1024,1],[1024,1],[1024,1],[]]` | 15 → 15 |
| `aten::addcmul` | `[[1024,1],[1024,1],[1024,4017],[]]` | 0 → 1 |
| `aten::addcmul` | `[[1024,1],[1024,4017],[1024,4017],[]]` | 0 → 12 |
| `aten::addcmul` | `[[1024,4017],[1024,4017],[1024,4017],[]]` | 15 → 2 |
| `aten::log` | `[[1024,1]]` | 1 → 1 |
| `aten::log` | `[[1024,4017]]` | 1 → 1 |
| `aten::log` | `[[1024]]` | 4 → 4 |
| `aten::mul` | `[[1,4017],[]]` | 2 → 2 |
| `aten::mul` | `[[1024,1],[1,4017]]` | 1 → 1 |
| `aten::mul` | `[[1024,1],[1024,1]]` | 36 → 5 |
| `aten::mul` | `[[1024,1],[1024,4017]]` | 37 → 7 |
| `aten::mul` | `[[1024,1],[]]` | 1 → 1 |
| `aten::mul` | `[[1024,4017],[1024,1]]` | 2 → 2 |
| `aten::mul` | `[[1024,4017],[1024,4017]]` | 5 → 4 |
| `aten::mul` | `[[1024,4017],[]]` | 1 → 1 |
| `aten::mul` | `[[1024],[1024]]` | 172 → 172 |
| `aten::mul` | `[[1024],[]]` | 111 → 111 |
| `aten::mul` | `[[16,1024],[1024]]` | 3 → 3 |

Profile input paths and SHA256 hashes:

| Side | File | SHA256 |
|---|---|---|
| baseline | [python.pstats](<wave-profiles-baseline/cpu-b1024/python.pstats>) | `5a7b6970ef6a40e18064b2c3e1f5e4e75c39b67c2058a233e0306befc27b23d4` |
| baseline | [python.txt](<wave-profiles-baseline/cpu-b1024/python.txt>) | `89d7bfeb5aed7e7542404bc875f558f4522bdd8b6b747dc1b04d402104b7a700` |
| baseline | [operators.txt](<wave-profiles-baseline/cpu-b1024/operators.txt>) | `0fe12c8e8877c3f5a7a39ca243dda0d5f99b09c0148b888bf8083e9afcb8ea51` |
| baseline | [trace.json](<wave-profiles-baseline/cpu-b1024/trace.json>) | `73f28baedf5c4e4a9b2edc3cdc1bed41e706c50714cffef03aa471796cd7874e` |
| baseline | [timing.json](<wave-profiles-baseline/cpu-b1024/timing.json>) | `18d56f9375e46688131c578305ce7da0c5966cdc3e96f0d14288feb4ee020cc4` |
| candidate | [python.pstats](<postchecks/profiles/profile-waveform-cpu-b1024-t1/python.pstats>) | `7d070d3642e695e20294c0d39392df8661b157701db6e24d8274e4e7c13fb4aa` |
| candidate | [python.txt](<postchecks/profiles/profile-waveform-cpu-b1024-t1/python.txt>) | `49ddb5e38285d15cdd0817da0aa580667caa170af09f357d8d76f30a2717ef02` |
| candidate | [operators.txt](<postchecks/profiles/profile-waveform-cpu-b1024-t1/operators.txt>) | `fa942b2017d7bee86d13b02ff278c65f80d4efbbdcee019f22a1435e05fd8f2e` |
| candidate | [trace.json](<postchecks/profiles/profile-waveform-cpu-b1024-t1/trace.json>) | `d3c1844b757d4bc5b362ad4890418c0ab6d0f54039ea740d0c9bbc9aa96ebc88` |
| candidate | [timing.json](<postchecks/profiles/profile-waveform-cpu-b1024-t1/timing.json>) | `76dffb2014134df3499e5645af2b0ae4f70b21195a9705ca12595759bdcddd71` |
| candidate | [files-sha256.json](<postchecks/profiles/profile-waveform-cpu-b1024-t1/files-sha256.json>) | `f6336b98448120e2eb92b9fd9491eaf39259feb11ee5bb023443eae90bbd72bd` |

## profile-waveform-cuda-b1-t1

| Function | Calls | Self ms | Cumulative ms |
|---|---:|---:|---:|
| `_batch_validate` | 6 → 6 | 0.111 → 0.099 | 1.245 → 1.236 |
| `_evaluate_phase_polynomial` | 2 → 2 | 1.809 → 0.348 | 2.148 → 0.679 |
| `taylorf2_aligned_phasing` | 1 → 1 | 0.391 → 0.384 | 4.172 → 4.148 |
| `taylorf2_fd_batch` | 1 → 1 | 1.348 → 1.346 | 13.013 → 11.511 |

| Operator | Input dimensions | Calls |
|---|---|---:|
| `aten::add` | `[[1,1],[1,1],[]]` | 32 → 2 |
| `aten::add` | `[[1,1],[1,4017],[]]` | 16 → 2 |
| `aten::add` | `[[1,4017],[1,4017],[]]` | 18 → 2 |
| `aten::add` | `[[1],[1],[]]` | 51 → 51 |
| `aten::add` | `[[1],[],[]]` | 43 → 43 |
| `aten::addcmul` | `[[1,1],[1,1],[1,1],[]]` | 15 → 15 |
| `aten::addcmul` | `[[1,1],[1,1],[1,4017],[]]` | 0 → 1 |
| `aten::addcmul` | `[[1,1],[1,4017],[1,4017],[]]` | 0 → 12 |
| `aten::addcmul` | `[[1,4017],[1,4017],[1,4017],[]]` | 15 → 2 |
| `aten::log` | `[[1,1]]` | 1 → 1 |
| `aten::log` | `[[1,4017]]` | 1 → 1 |
| `aten::log` | `[[1]]` | 4 → 4 |
| `aten::mul` | `[[1,1],[1,1]]` | 36 → 5 |
| `aten::mul` | `[[1,1],[1,4017]]` | 38 → 8 |
| `aten::mul` | `[[1,1],[]]` | 1 → 1 |
| `aten::mul` | `[[1,4017],[1,1]]` | 2 → 2 |
| `aten::mul` | `[[1,4017],[1,4017]]` | 5 → 4 |
| `aten::mul` | `[[1,4017],[]]` | 3 → 3 |
| `aten::mul` | `[[16,1],[1]]` | 3 → 3 |
| `aten::mul` | `[[1],[1]]` | 172 → 172 |
| `aten::mul` | `[[1],[]]` | 111 → 111 |

Profile input paths and SHA256 hashes:

| Side | File | SHA256 |
|---|---|---|
| baseline | [python.pstats](<wave-profiles-baseline/cuda-b1/python.pstats>) | `89ca4c066cc2a7ab089203d0723a0716853adc930b7508766e32d64b7d469ec4` |
| baseline | [python.txt](<wave-profiles-baseline/cuda-b1/python.txt>) | `f3dac273d80b5f8361a67a9a96837f51b0beaf7fcc39b516afef2a199e3a733d` |
| baseline | [operators.txt](<wave-profiles-baseline/cuda-b1/operators.txt>) | `6bb1ae24e8b91d28b53295e1e20671376ba76b62af6d53848df2190f6a3709d9` |
| baseline | [trace.json](<wave-profiles-baseline/cuda-b1/trace.json>) | `743c68cd32da63cab9fc5f429e7d28e1ab86a85ca2a14932996226070093f4b3` |
| baseline | [timing.json](<wave-profiles-baseline/cuda-b1/timing.json>) | `dd20473531feb6f9d4715eaf25742a96eff95c51dbac938fd884219f0bbd3cb5` |
| candidate | [python.pstats](<postchecks/profiles/profile-waveform-cuda-b1-t1/python.pstats>) | `6eafff5e9247cae3306fdec04dc246dd50bb0da4442b675c41f46d0e1b03c109` |
| candidate | [python.txt](<postchecks/profiles/profile-waveform-cuda-b1-t1/python.txt>) | `1345bba120f64321882bd60bb3caf9db82c22ed60bfafcf8ecba3ff3dbbf03d0` |
| candidate | [operators.txt](<postchecks/profiles/profile-waveform-cuda-b1-t1/operators.txt>) | `ab70b43200f4f0ea5717a2687bab601dba706ed087182f2aee60e38afa341f7e` |
| candidate | [trace.json](<postchecks/profiles/profile-waveform-cuda-b1-t1/trace.json>) | `9393fc31306247da7f54bdc375f99a37e256a3168d11ea6a297711abe99a0db3` |
| candidate | [timing.json](<postchecks/profiles/profile-waveform-cuda-b1-t1/timing.json>) | `f2f97219f564806791e9ab5aa49cae38280da542f5e3b913817cf75659fd6744` |
| candidate | [files-sha256.json](<postchecks/profiles/profile-waveform-cuda-b1-t1/files-sha256.json>) | `5fa0d0a22494a25953943d5ddfba428ae45cadb20def4888b64cb4fd83d54c2e` |

## profile-waveform-cuda-b1024-t1

| Function | Calls | Self ms | Cumulative ms |
|---|---:|---:|---:|
| `_batch_validate` | 6 → 6 | 0.104 → 0.108 | 1.277 → 1.342 |
| `_evaluate_phase_polynomial` | 2 → 2 | 1.803 → 0.410 | 2.131 → 0.763 |
| `taylorf2_aligned_phasing` | 1 → 1 | 0.407 → 0.425 | 4.298 → 4.414 |
| `taylorf2_fd_batch` | 1 → 1 | 1.425 → 1.488 | 13.476 → 12.629 |

| Operator | Input dimensions | Calls |
|---|---|---:|
| `aten::add` | `[[1024,1],[1024,1],[]]` | 32 → 2 |
| `aten::add` | `[[1024,1],[1024,4017],[]]` | 16 → 2 |
| `aten::add` | `[[1024,4017],[1,4017],[]]` | 1 → 1 |
| `aten::add` | `[[1024,4017],[1024,4017],[]]` | 17 → 1 |
| `aten::add` | `[[1024],[1024],[]]` | 51 → 51 |
| `aten::add` | `[[1024],[],[]]` | 43 → 43 |
| `aten::addcmul` | `[[1024,1],[1024,1],[1024,1],[]]` | 15 → 15 |
| `aten::addcmul` | `[[1024,1],[1024,1],[1024,4017],[]]` | 0 → 1 |
| `aten::addcmul` | `[[1024,1],[1024,4017],[1024,4017],[]]` | 0 → 12 |
| `aten::addcmul` | `[[1024,4017],[1024,4017],[1024,4017],[]]` | 15 → 2 |
| `aten::log` | `[[1024,1]]` | 1 → 1 |
| `aten::log` | `[[1024,4017]]` | 1 → 1 |
| `aten::log` | `[[1024]]` | 4 → 4 |
| `aten::mul` | `[[1,4017],[]]` | 2 → 2 |
| `aten::mul` | `[[1024,1],[1,4017]]` | 1 → 1 |
| `aten::mul` | `[[1024,1],[1024,1]]` | 36 → 5 |
| `aten::mul` | `[[1024,1],[1024,4017]]` | 37 → 7 |
| `aten::mul` | `[[1024,1],[]]` | 1 → 1 |
| `aten::mul` | `[[1024,4017],[1024,1]]` | 2 → 2 |
| `aten::mul` | `[[1024,4017],[1024,4017]]` | 5 → 4 |
| `aten::mul` | `[[1024,4017],[]]` | 1 → 1 |
| `aten::mul` | `[[1024],[1024]]` | 172 → 172 |
| `aten::mul` | `[[1024],[]]` | 111 → 111 |
| `aten::mul` | `[[16,1024],[1024]]` | 3 → 3 |

Profile input paths and SHA256 hashes:

| Side | File | SHA256 |
|---|---|---|
| baseline | [python.pstats](<wave-profiles-baseline/cuda-b1024/python.pstats>) | `d2260aecdb22edc071ac2db90580b7a7e375e74333c9ec6c7961c2860f074a97` |
| baseline | [python.txt](<wave-profiles-baseline/cuda-b1024/python.txt>) | `26c2af50d200268e41d1bee384c4e9d9637b45a1e4fbfcd56231aaf1aec45bf0` |
| baseline | [operators.txt](<wave-profiles-baseline/cuda-b1024/operators.txt>) | `f584a1d8e803bdfd753299cd75daf63ede0c56935b2f3c7040387cae6a9aa3d4` |
| baseline | [trace.json](<wave-profiles-baseline/cuda-b1024/trace.json>) | `9eb6e962541e6f0a6acd2bf77c419251836cd985d84274f1a4a655305469808a` |
| baseline | [timing.json](<wave-profiles-baseline/cuda-b1024/timing.json>) | `c05d2a596994b8690d8cf4ff36ec128f758c85f734e97cada5b50e7a73807abf` |
| candidate | [python.pstats](<postchecks/profiles/profile-waveform-cuda-b1024-t1/python.pstats>) | `8c517f28f2ee52bf6cfda603666f06de4e771650de17d8b5417d0f29208cd853` |
| candidate | [python.txt](<postchecks/profiles/profile-waveform-cuda-b1024-t1/python.txt>) | `a238142623ca53d5c9d3afc341ad0fafb180d4f557d46d5cc75962544aa0935d` |
| candidate | [operators.txt](<postchecks/profiles/profile-waveform-cuda-b1024-t1/operators.txt>) | `8868c766d86b41a8d9f64da0ab8c106bdd1f5369e995f740c04ac4ef40c325cd` |
| candidate | [trace.json](<postchecks/profiles/profile-waveform-cuda-b1024-t1/trace.json>) | `d1d755cbb7a926ceb78860bd7cbc70ce6b77002c900cb2dacf98963259528238` |
| candidate | [timing.json](<postchecks/profiles/profile-waveform-cuda-b1024-t1/timing.json>) | `27cdd27e9f7dade7ffd51e8df4a345573508c78bbd0412b8d9f86f2bc6a2701b` |
| candidate | [files-sha256.json](<postchecks/profiles/profile-waveform-cuda-b1024-t1/files-sha256.json>) | `780b6fb557a20a5160b724f4b6e3d61805446b2bdb4e9dcdfdb4005df478949f` |

## Source and profiling-script hashes

Exact source paths and line numbers for each observed focused function, candidate command receipts, all operator shape/type groups, and input hashes are retained in profile-summary.json.

| Source | Path | SHA256 |
|---|---|---|
| baseline | `pycbc/filter/matchedfilter.py` | `8d40acb0d70547bac98464a165b775d4943d822de21e03e17bbde7fff518cc2e` |
| baseline | `pycbc/filter/matchedfilter_torch.py` | `2db0475c19872675647f5c3b3711a61ba2210c1599ecb67c5eee8b5c910d7350` |
| baseline | `pycbc/waveform/taylorf2_torch.py` | `9361103c2db535b1b65040bb4af82a686c5d5d9db173cc79ee54b64bb78bed9e` |
| baseline | `test/test_live_batch_torch_peaks.py` | `712f2b9730a9b0dbcf934bdf638c644d086be089ac0e7f7d442a747f04ac7957` |
| candidate | `pycbc/filter/matchedfilter.py` | `b6d98c6cd9d3e1502b3a80f3512a8437b46691d8d31aa26aab193d754e103720` |
| candidate | `pycbc/filter/matchedfilter_torch.py` | `a7e5a56cd5255fbf0f7b88663f6ec32bdbd31e4757f00b93a8b7d2c9fa5ed533` |
| candidate | `pycbc/waveform/taylorf2_torch.py` | `46137f45803da55bb1601e05a1cb26f7bc83d4879d8ee7823b3e3bc4ef85a648` |
| candidate | `test/test_live_batch_torch_peaks.py` | `60c57afe670a848fb7de17f2ec681a92400ceea1a1db26577b39dec765ec85cb` |
| candidate | `test/test_torch_batch_overlap_scaling.py` | `26cfac4f7ca97f2dbc902b42930cdc44d5a4a9713cabf9c0456ab14ecb50efb3` |
| candidate | `test/waveform/test_taylorf2_phase_evaluation.py` | `b36d53b6269057f6dfaa8b988f1a0135ef09671b809c73b9a237c3b0618d47d2` |
| supplied script | [profile-live.py](<profile-live.py>) | `2023c5f9d91f3bfe9eeb34dae3fa734e3094be0ec67495cba06673404a01750b` |
| supplied script | [profile-waveform.py](<profile-waveform.py>) | `05dbfd48551e3f87bfee0ada622f715257edd78bbfcae88510219fce132f7e6d` |
