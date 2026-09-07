"""Render the frozen accuracy audit, explicitly separate from performance."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

root = Path(__file__).resolve().parent
data = json.loads((root / 'normalized-error-audit.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for route, label, color in [('torch_cpu', 'Torch CPU', '#2563eb'),
                            ('torch_cuda', 'Torch CUDA', '#c2410c')]:
    cells = [c for c in data['cells'] if c['route'] == route]
    batches = [c['batch'] for c in cells]
    axes[0].plot(batches, [c['raw_policy_failures'] for c in cells],
                 'o-', color=color, label=label, linewidth=2)
    axes[1].plot(batches, [c['max_complex_snr_error'] for c in cells],
                 'o-', color=color, label=label, linewidth=2)
for ax in axes:
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 8, 32, 128, 512, 1024])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('Templates per execution batch')
    ax.grid(axis='y', alpha=.18)
axes[0].set_title('Original raw gate: failed samples', loc='left', weight='bold')
axes[0].set_ylabel('Failures / 402,653,184 complex samples')
axes[0].set_ylim(0, 630)
axes[0].legend(frameon=False)
axes[1].set_title('Raw errors expressed in SNR units', loc='left', weight='bold')
axes[1].set_ylabel('Largest raw error × common SNR normalization')
axes[1].set_yscale('log')
axes[1].set_ylim(1e-6, 2.5e-3)
axes[1].axhline(1e-3, color='#475569', linestyle='--', linewidth=1.2)
axes[1].text(1, 1.15e-3, 'Existing trigger SNR tolerance (context)', color='#475569', fontsize=9)
fig.suptitle('Current batch sweep: accuracy diagnosis', x=.07, ha='left', weight='bold', fontsize=16)
fig.text(.07, .02, 'Fixed bank: 1,024 templates · N=131,072 · 3 blocks · CPU/MKL reference · source 9578a71047\n'
         'Common input-derived normalization; route-specific normalization excluded. Original qualification remains failed.',
         fontsize=9, color='#475569')
fig.tight_layout(rect=(.02, .12, 1, .94))
fig.savefig(root / 'current-batch-accuracy.png', dpi=170)
fig.savefig(root / 'current-batch-accuracy.svg')
