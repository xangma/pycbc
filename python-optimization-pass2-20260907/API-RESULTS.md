# Pass-two API result and cutoff assessment

Six serial fresh workers in ABBAAB order, three per role; seven samples per cell. CPU 8, one thread, shared len. Both precisions use identical input and output bytes across all workers. The measurements below are public squared_norm calls, not standalone executable wall time.

| dtype | size | stride | baseline median [min, max] µs | candidate median [min, max] µs | B/C ratio |
|---|---:|---:|---:|---:|---:|
| complex128 | 16 | 1 | 17.9053 [17.5664, 18.4847] | 18.1510 [17.8124, 18.2131] | 0.986466 |
| complex128 | 512 | 1 | 35.1138 [34.9069, 35.5907] | 35.2917 [34.8853, 35.4986] | 0.994960 |
| complex128 | 4095 | 1 | 152.5934 [152.1290, 153.2237] | 153.5473 [152.3608, 154.9975] | 0.993787 |
| complex128 | 4096 | 1 | 35.2410 [34.9582, 35.5110] | 34.1298 [33.6479, 34.6157] | 1.032557 |
| complex128 | 4096 | 3 | 38.8102 [38.5363, 39.3057] | 37.8760 [37.3555, 38.2395] | 1.024665 |
| complex128 | 8193 | 1 | 42.2879 [41.8714, 42.8291] | 40.9999 [40.5589, 41.1899] | 1.031416 |
| complex128 | 131072 | 1 | 233.3304 [231.8521, 234.3921] | 210.6949 [208.2879, 211.8189] | 1.107433 |
| complex128 | 131072 | 3 | 340.0619 [332.3152, 345.7898] | 319.7724 [308.0024, 319.8322] | 1.063450 |
| complex128 | 1048577 | 1 | 3437.0620 [3386.0819, 3488.7860] | 3119.9153 [3093.6168, 3140.4887] | 1.101652 |
| complex128 | 1048577 | 3 | 6110.0380 [5963.4028, 6125.1300] | 5768.7930 [5737.1890, 5769.1574] | 1.059154 |
| complex64 | 16 | 1 | 17.4754 [17.4722, 18.1273] | 17.7976 [17.5506, 18.0975] | 0.981893 |
| complex64 | 512 | 1 | 22.3726 [22.0650, 22.7003] | 22.3159 [22.1518, 22.5945] | 1.002541 |
| complex64 | 4095 | 1 | 48.7007 [48.6356, 49.3137] | 48.9756 [48.7328, 49.0615] | 0.994386 |
| complex64 | 4096 | 1 | 33.7619 [33.3180, 34.7651] | 32.9552 [32.5829, 33.3449] | 1.024480 |
| complex64 | 4096 | 3 | 34.8017 [34.4073, 35.4670] | 34.2242 [34.0019, 34.3197] | 1.016874 |
| complex64 | 8193 | 1 | 39.4318 [38.8295, 39.7174] | 38.1536 [37.6893, 38.9917] | 1.033501 |
| complex64 | 131072 | 1 | 196.4998 [195.9278, 196.6821] | 188.4767 [187.3381, 190.9311] | 1.042568 |
| complex64 | 131072 | 3 | 211.1500 [207.2944, 212.6514] | 203.1874 [201.9304, 205.2439] | 1.039188 |
| complex64 | 1048577 | 1 | 1804.3294 [1795.4014, 1808.6176] | 1630.8624 [1617.9648, 1676.4814] | 1.106365 |
| complex64 | 1048577 | 3 | 3467.8978 [3463.8890, 3486.5945] | 3321.1670 [3314.1171, 3330.8051] | 1.044180 |

Assessment: retain the existing 4096 cutoff. All 14 changed-path cells improve in median (1.0169–1.1074x). All six unchanged-path cells have overlapping ranges and ratios 0.9819–1.0025x. No evidence supports changing the cutoff or expanding the scope. The fresh profile attributes ~1.47 s of 118.07 s instrumented full wall time to this operation, so executable benefit is expected to be small. Proceeding to the already specified eight fresh full-executable workers is reasonable, subject to the coordinating task’s review.

API summary SHA256: 8b3efc0e981dee8ca36cdedb88c274d70d3f9287afba5014680b310ba11de20a. API status SHA256: e0c83e8d346a0aa739ead9b24e742dd9456cfbc31380312754eb033890c36764. Independent audit SHA256: 50f02caa1495f7ea5afdb724285b73e15756621d181b85a2cecce9507c12f62a. All owned API groups are inactive; shared lock independently reacquired; source/native/input/helper pins unchanged. Local transfer hashes match all 18 worker evidence files and eight stage receipts. Linux control tests: 143 passed, zero skipped.
