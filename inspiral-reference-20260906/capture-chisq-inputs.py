#!/usr/bin/env python3
"""Untimed, observational replay of one frozen full inspiral invocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--template-hash', required=True, type=int)
    parser.add_argument('--output-name', required=True)
    args = parser.parse_args()
    assert all(Path(x).name == x for x in (args.case, args.output_name))
    root = Path(__file__).resolve().parent
    receipt_path = root / 'runs' / args.case / 'receipt.json'
    frozen = json.loads(receipt_path.read_text())
    assert frozen['state'] == 'complete' and frozen['returncode'] == 0
    source = Path(frozen['source'])
    assert frozen['source_info']['commit'] == '968bcd558117262af0d603710b054174659adb51'
    assert subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True) == ''
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == frozen['source_info']['commit']
    out = root / args.output_name
    out.mkdir(exist_ok=False)
    cfg = json.loads((root / 'config.json').read_text())
    os.environ.update(frozen['environment'])
    os.sched_setaffinity(0, {cfg['core']})
    sys.path.insert(0, str(source))
    inputs = dict(frozen['input_sha256'])
    inputs.update({str(p): digest(p) for p in [Path(__file__).resolve(), receipt_path, source / 'pycbc/vetoes/chisq.py', source / 'pycbc/vetoes/chisq_torch.py', source / 'pycbc/scheme.py']})
    assert all(digest(p) == h for p, h in inputs.items())
    cli = list(frozen['executable_cli'])
    cli[cli.index('--output') + 1] = str(out / 'triggers.hdf')
    record = dict(state='running', started_utc=utc(), pid=os.getpid(), hostname=os.uname().nodename, cwd=str(root), source_info=frozen['source_info'], input_sha256=inputs, executable_cli=cli, template_hash=args.template_hash, captures=[])

    def save():
        temporary = out / 'status.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(out / 'status.json')

    save()
    started = time.perf_counter()
    try:
        import numpy as np
        from pycbc.filter import matchedfilter
        from pycbc.strain import strain
        from pycbc.vetoes import chisq
        from pycbc.waveform import bank
        context = {}
        segment_init = strain.StrainSegments.__init__

        def observed_segment_init(obj, *a, **kw):
            segment_init(obj, *a, **kw)
            data = np.array(obj.strain.numpy(), copy=True)
            path = out / 'conditioned-strain.npy'
            np.save(path, data, allow_pickle=False)
            record['conditioned_strain'] = dict(path=str(path), sha256=digest(path), data_sha256=hashlib.sha256(data.tobytes()).hexdigest(), start_time=float(obj.strain.start_time), delta_t=float(obj.strain.delta_t))
            save()

        strain.StrainSegments.__init__ = observed_segment_init
        getitem = bank.FilterBank.__getitem__

        def observed_getitem(obj, index):
            result = getitem(obj, index)
            context['template_hash'] = int(obj.table.template_hash[index])
            context['template_index'] = int(index)
            return result

        bank.FilterBank.__getitem__ = observed_getitem
        controller_init = matchedfilter.MatchedFilterControl.__init__

        def observed_init(obj, *a, **kw):
            controller_init(obj, *a, **kw)
            selected = obj.matched_filter_and_cluster

            def observed_filter(segnum, *fa, **fk):
                seg = obj.segments[segnum]
                context['segment'] = dict(number=int(segnum), epoch=float(seg._epoch), analyze_start=int(seg.analyze.start), analyze_stop=int(seg.analyze.stop), cumulative_index=int(seg.cumulative_index), filter_bin_start=int(obj.kmin), filter_bin_stop=int(obj.kmax))
                return selected(segnum, *fa, **fk)

            obj.matched_filter_and_cluster = observed_filter

        matchedfilter.MatchedFilterControl.__init__ = observed_init
        values = chisq.SingleDetPowerChisq.values

        def array(value):
            return np.array(value.numpy() if hasattr(value, 'numpy') else value, copy=True)

        def observed_values(obj, corr, snrv, snr_norm, psd, indices, template):
            result = values(obj, corr, snrv, snr_norm, psd, indices, template)
            if context.get('template_hash') == args.template_hash:
                row = dict(context, snr_norm=float(snr_norm), snr_threshold=obj.snr_threshold, arrays={})
                serial = len(record['captures'])
                for name, value in dict(corr=corr, snrv=snrv, indices=indices, bins=obj.cached_chisq_bins(template, psd), chisq=result[0], template=template, psd=psd).items():
                    data = array(value)
                    path = out / f'capture-{serial}-{name}.npy'
                    np.save(path, data, allow_pickle=False)
                    row['arrays'][name] = dict(path=str(path), sha256=digest(path), dtype=str(data.dtype), shape=list(data.shape))
                record['captures'].append(row)
                save()
            return result

        chisq.SingleDetPowerChisq.values = observed_values
        sys.argv = cli
        code = 0
        try:
            runpy.run_path(cli[0], run_name='__main__')
        except SystemExit as exc:
            code = 0 if exc.code is None else exc.code
        assert code == 0, code
        assert len(record['captures']) == 5, len(record['captures'])
        record.update(state='complete', returncode=0)
    except BaseException:
        record.update(state='failed', returncode=1, traceback=traceback.format_exc())
        raise
    finally:
        record.update(finished_utc=utc(), elapsed_wall_seconds=time.perf_counter() - started, input_sha256_after={p: digest(p) for p in inputs}, source_status_after=subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True))
        if record['input_sha256_after'] != inputs or record['source_status_after']:
            record['state'] = 'invalid-input-mutation'
        save()
    return 0 if record['state'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
