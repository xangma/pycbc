"""Observe an unmodified CPU search; compare only double working arithmetic.

Usage: python observe_arithmetic.py pycbc_inspiral ARGS
Writes under IMPACT_OUTPUT. The search itself always receives upstream values.
"""
import inspect
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np
from pycbc.events import eventmgr, ranking
from pycbc.vetoes import chisq
from cpu_chisq_review import kernels, reference_power

root = Path(os.environ['IMPACT_OUTPUT'])
root.mkdir(exist_ok=True)
mods = kernels(('upstream', 'arithmetic'))
original_values = chisq.SingleDetPowerChisq.values
original_points = chisq.power_chisq_at_points_from_precomputed
original_cut = eventmgr.EventManager.newsnr_threshold
context = {}
lookup = {}
records = []
captures = 0


def values(self, corr, snrv, norm, psd, indices, template):
    caller = inspect.currentframe().f_back
    stilde = caller.f_locals['stilde']
    context.update(template_hash=int(template.params.template_hash),
                   segment=int(caller.f_locals['s_num']),
                   time_offset=int(stilde.cumulative_index -
                                   stilde.analyze.start))
    return original_values(self, corr, snrv, norm, psd, indices, template)


def points(corr, snr, norm, bins, indices):
    global captures
    result = original_points(corr, snr, norm, bins, indices)
    p = len(bins) - 1
    snr = np.asarray(snr)
    snr_power = (snr.conj() * snr).real
    powers = {label: mod.shift_sum(corr, indices, bins)
              for label, mod in mods.items()}
    chi = {label: (power*p-snr_power)*(norm**2.0)
           for label, power in powers.items()}
    np.testing.assert_array_equal(chi['upstream'], result)
    rho = abs(snr * norm)
    scores = {label: ranking.newsnr(rho, c/(2*p-2))
              for label, c in chi.items()}
    flips = ((scores['upstream'] >= 5) != (scores['arithmetic'] >= 5))
    entry = dict(context, call=len(records),
                 indices=np.asarray(indices).tolist(),
                 bins=np.asarray(bins).tolist(), ntime=len(corr),
                 norm=float(norm),
                 rho=rho.tolist(),
                 upstream=chi['upstream'].tolist(),
                 arithmetic=chi['arithmetic'].tolist(),
                 time_indices=(np.asarray(indices) +
                               context['time_offset']).tolist())
    if flips.any():
        reference = ((p*reference_power(corr.numpy(), indices, bins)-snr_power)
                     * norm**2.0)
        entry['reference'] = reference.tolist()
        entry['reference_newsnr'] = ranking.newsnr(
            rho, reference/(2*p-2)).tolist()
        if captures < 10:
            filename = f'changed-call-{len(records):05d}.npz'
            np.savez(root/filename, corr=corr.numpy(), snr=snr, norm=norm,
                     bins=bins, indices=indices)
            entry['capture'] = filename
            captures += 1
        print('CROSSING', json.dumps(entry), flush=True)
    for j, sample in enumerate(entry['time_indices']):
        lookup[(context['template_hash'], sample)] = (len(records), j)
    records.append(entry)
    with (root/'point-calls.jsonl').open('a') as stream:
        stream.write(json.dumps(entry)+'\n')
    if len(records) % 250 == 0:
        print('PROGRESS', len(records), 'point calls', flush=True)
    return result


def cut(self, threshold):
    assert threshold == 5
    before = self.events.copy()
    patched = before.copy()
    rows = []
    for i, event in enumerate(before):
        template = self.template_params[int(event['template_id'])]['tmplt']
        key = (int(template.template_hash), int(event['time_index']))
        call, j = lookup[key]
        record = records[call]
        assert event['chisq'] == record['upstream'][j]
        patched['chisq'][i] = record['arithmetic'][j]
        rows.append(dict(template_hash=key[0], time_index=key[1], call=call,
                         point=j, rho=float(abs(event['snr'])),
                         upstream=float(event['chisq']),
                         arithmetic=float(patched['chisq'][i]),
                         dof=int(event['chisq_dof'])))
    # Use the actual public selection method twice with identical event inputs.
    saved_array, saved_size = self._events, self._events_size
    self._events, self._events_size = patched.copy(), len(patched)
    original_cut(self, threshold)
    patched_keys = {(int(e['template_id']), int(e['time_index']))
                    for e in self.events}
    self._events, self._events_size = saved_array, saved_size
    original_cut(self, threshold)
    upstream_keys = {(int(e['template_id']), int(e['time_index']))
                     for e in self.events}
    for row, event in zip(rows, before):
        key = (int(event['template_id']), int(event['time_index']))
        row['upstream_pass'] = key in upstream_keys
        row['arithmetic_pass'] = key in patched_keys
    summary = dict(before_cut=len(before), upstream_kept=len(upstream_keys),
                   arithmetic_kept=len(patched_keys),
                   changed=len(upstream_keys ^ patched_keys),
                   total_point_calls=len(records),
                   total_point_samples=sum(len(r['indices']) for r in records),
                   threshold=threshold, rows=rows)
    (root/'selection.json').write_text(json.dumps(summary, indent=2)+'\n')
    np.savez(root/'events-before-cut.npz', upstream=before, arithmetic=patched)
    print('SELECTION', json.dumps({k: v for k, v in summary.items()
                                  if k != 'rows'}), flush=True)


chisq.SingleDetPowerChisq.values = values
chisq.power_chisq_at_points_from_precomputed = points
eventmgr.EventManager.newsnr_threshold = cut
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
