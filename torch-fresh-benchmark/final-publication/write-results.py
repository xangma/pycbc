"""Insert independently verified fresh results into the performance guide."""
import json
from pathlib import Path
import subprocess

O = Path(__file__).resolve().parent
F = O.parent / 'torch-fresh-benchmark-20260908'
W = Path('/private/tmp/pycbc-torch-benchmark-summary-20260908')
s = json.loads((F / 'remote/summary.json').read_text())
v = json.loads((F / 'independent-verification/verification.json').read_text())
e = json.loads((F / 'evidence-publication.json').read_text())
assert s['scientific_gates_pass'] and s['cpu_preserved'] and s['timing_outputs_pass'] and v['status'] == 'pass'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W)
labels = {'original-cpu': 'Original CPU', 'proposed-cpu': 'Candidate CPU',
          'torch-cpu': 'Torch CPU', 'torch-cuda': 'Torch CUDA'}
rows = []
baseline = s['arms']['original-cpu']['median_seconds']
for arm, label in labels.items():
    r = s['arms'][arm]
    assert len(r['samples_seconds']) == 4
    rows.append(f'''   * - {label}
     - {r['median_seconds']:.2f}
     - {r['min_seconds']:.2f}–{r['max_seconds']:.2f}
     - {baseline / r['median_seconds']:.2f}×
     - {r['template_seconds_per_wall_second']:,.0f}''')
text = f'''Fresh complete-executable results
---------------------------------

Measured on **2026-09-08**, using four fresh unprofiled processes per route
in rotating order after scientific qualification. The workload contains
**384 compressed templates, five segments and 1904 unique H1 seconds**
(731,136 template-seconds; 1920 template/segment pairs).

.. list-table:: Complete-executable wall times
   :header-rows: 1
   :widths: 22 16 20 20 22

   * - Route
     - Median (s)
     - Range (s)
     - Original CPU / route
     - Template-seconds / wall second
{chr(10).join(rows)}

Each process was pinned to logical CPU 8 of an AMD Ryzen Threadripper PRO
3995WX, with numerical thread pools fixed to one. The Torch routes also set
intra/inter-op counts to one. CUDA used an NVIDIA GeForce RTX 4090 with
graphs disabled. The host and
GPU were shared and unreserved. Ranges show the four observed samples; these
results do not establish sustained or full-machine capacity.

The timing boundary runs from checked-process launch through exit, including
startup, input preparation, filtering, vetoes, HDF output and runtime
verification. Qualification instrumentation and parent-side comparisons are
outside this boundary. The rate divides 731,136 template-seconds by the
median wall time; the ratio divides the original CPU median by each route's
median.

All four routes produced **1988 triggers** and passed all five frozen
cross-route trigger and complete-PSD comparisons. Original and candidate
CPU scientific data and PSDs were byte-identical. Every timed output also
passed comparison with its route's fresh qualification.

Measured sources: original CPU
``{s['source_commits']['original']}`` and candidate main
``{s['source_commits']['proposed']}``. The optional FFT and
native CPU optimization branches are outside this measurement. See the
`immutable benchmark evidence <{e['url']}>`_
for every sample, command, input hash, environment record and independent
verification, and :ref:`torch-reference-campaign` for the workload and gates.

'''
p = W / 'docs/torch_performance.rst'
old = p.read_text()
assert 'Fresh complete-executable results' not in old
marker = 'Choose the measurement boundary\n'
assert old.count(marker) == 1
p.write_text(old.replace(marker, text + marker))
print('Inserted four fresh benchmark rows with immutable evidence link')
