#!/usr/bin/env python3
"""Strict scientific comparison for executable benchmark receipts.

Missing evidence fails closed; trigger order is immaterial, identity is exact.
"""
from collections import defaultdict
import json
from pathlib import Path

import h5py
import numpy as np

PERF = {f'H1/search/{s}' for s in (
    'filter_rate_per_core', 'run_time', 'setup_time_fraction', 'templates_per_core')}
POLICY = dict(rtol=5e-4, atol=1e-5, sigmasq_rtol=1e-5,
              phase_atol_radians=5e-4, psd_rtol=1e-3, psd_atol=0.0,
              template_duration_rtol=1e-3, template_duration_atol=0.1,
              chisq_rtol=1e-2, chisq_atol=1.0,
              strain_rtol=1e-4, strain_atol=1e-3)
JAX_CHISQ_MODES = {'cpu-compatible', 'direct-phase'}


def read_hdf(path):
    datasets, attrs = {}, {}
    with h5py.File(path, 'r') as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = np.asarray(obj[()])
            for key, value in obj.attrs.items():
                attrs[f'{name}@{key}'] = np.asarray(value)
        visit('/', f)
        f.visititems(visit)
    return datasets, attrs


def exact(a, b):
    return a.shape == b.shape and bool(np.array_equal(a, b))


def scalar(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    return value.decode(errors='replace') if isinstance(value, bytes) else value


def metrics(a, b, field, exact_required=False):
    result = dict(reference_shape=list(a.shape), candidate_shape=list(b.shape),
                  reference_dtype=str(a.dtype), candidate_dtype=str(b.dtype))
    result['exact'] = exact(a, b)
    if a.shape != b.shape:
        return dict(result, passed=False, reason='shape mismatch')
    if a.dtype.kind not in 'biufc' or b.dtype.kind not in 'biufc':
        return dict(result, passed=result['exact'], rule='exact')
    # Exact comparison precedes float conversion, preserving integer hashes.
    finite = np.isfinite(a) & np.isfinite(b)
    result['nonfinite_elements'] = int((~finite).sum())
    x, y = a.astype(np.complex128 if np.iscomplexobj(a) or np.iscomplexobj(b)
                    else np.float64), b.astype(np.complex128 if np.iscomplexobj(a)
                    or np.iscomplexobj(b) else np.float64)
    delta = np.abs(y - x)
    if a.dtype.kind in 'biu' and b.dtype.kind in 'biu':
        # Object arithmetic avoids precision loss or unsigned wrap in hash metrics.
        delta = np.asarray(np.abs(b.astype(object) - a.astype(object)), dtype=np.float64)
    circular = 'phase' in field.rsplit('/', 1)[-1] and not exact_required
    if circular:
        delta = np.abs(np.angle(np.exp(1j * (y - x))))
    rtol = POLICY['sigmasq_rtol'] if 'sigmasq' in field else POLICY['rtol']
    atol = POLICY['atol']
    if 'template_duration' in field:
        rtol = POLICY['template_duration_rtol']
        atol = POLICY['template_duration_atol']
    elif 'chisq' in field and not field.endswith(('_dof', 'bank_chisq')):
        rtol = POLICY['chisq_rtol']
        atol = POLICY['chisq_atol']
    elif 'strain' in field:
        rtol = POLICY['strain_rtol']
        atol = POLICY['strain_atol']
    if circular:
        rtol, atol = 0.0, POLICY['phase_atol_radians']
    if 'psd' in field.lower() and '/search/' in field:
        rtol, atol = POLICY['psd_rtol'], POLICY['psd_atol']
    if exact_required:
        ok = (a == b) & finite
        rule = 'exact'
    else:
        ok = (delta <= atol + rtol * np.abs(x)) & finite
        rule = 'circular phase' if circular else 'absolute + relative to reference'
    valid_delta = delta[finite]
    relmask = finite & (np.abs(x) > 0)
    result.update(passed=bool(np.all(ok)), rule=rule,
                  rtol=0.0 if exact_required else rtol,
                  atol=0.0 if exact_required else atol,
                  failed_elements=int((~ok).sum()),
                  max_absolute_difference=float(valid_delta.max()) if valid_delta.size else None,
                  max_relative_difference=float((delta[relmask] / np.abs(x[relmask])).max())
                      if relmask.any() else None,
                  zero_reference_nonzero_candidate=int((finite & (x == 0) & (y != 0)).sum()))
    return result


def identities(data, rate, origin):
    hashes, times = data['H1/template_hash'], data['H1/end_time']
    if hashes.ndim != 1 or times.shape != hashes.shape or not np.isfinite(times).all():
        raise ValueError('Invalid trigger identity arrays')
    samples_float = (times.astype(np.float64) - origin) * rate
    samples = np.rint(samples_float).astype(np.int64)
    residual = np.abs(samples_float - samples)
    groups = defaultdict(list)
    for i, (h, sample) in enumerate(zip(hashes, samples)):
        groups[(scalar(h), int(sample))].append(i)
    return groups, dict(max_sample_grid_residual=float(residual.max()) if len(residual) else 0.0,
                       off_grid_count=int((residual > 0.001).sum()),
                       duplicate_keys=sum(len(v) > 1 for v in groups.values()),
                       ambiguous_trigger_count=sum(len(v) for v in groups.values() if len(v) > 1))


def attrs_comparison(a, b, provenance=()):
    result = {}
    for key in sorted(set(a) | set(b)):
        present = key in a and key in b
        same = present and exact(a[key], b[key])
        result[key] = dict(passed=same)
        if key in provenance and present:
            # Source revisions necessarily differ between upstream and JAX.
            # The campaign validates their provenance separately; retain both
            # values here without treating a version string as a calculation.
            result[key] = dict(
                passed=True, exact=same, rule='recorded source provenance',
                reference=np.asarray(a[key]).astype(str).tolist(),
                candidate=np.asarray(b[key]).astype(str).tolist())
    return result


def _declared_jax_mode(config, attrs):
    """Read and validate the chi-square mode from config and HDF metadata."""
    config_mode = None
    config_obj = None
    config_error = None
    if config is not None:
        try:
            config_obj = json.loads(str(scalar(config)))
            if not isinstance(config_obj, dict):
                raise ValueError('science_config must be a JSON object')
            config_modes = [config_obj.get(key) for key in
                            ('--jax-chisq-mode', 'jax_chisq_mode')]
            config_modes = [mode for mode in config_modes if mode is not None]
            if any(mode not in JAX_CHISQ_MODES for mode in config_modes):
                raise ValueError(f'unknown jax-chisq-mode {config_modes!r}')
            if len(set(config_modes)) > 1:
                raise ValueError('jax-chisq-mode aliases disagree in config')
            config_mode = config_modes[0] if config_modes else None
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            config_error = str(exc)
    attr_mode = None
    attr_error = None
    if '/@jax_chisq_mode' in attrs:
        attr_mode = scalar(attrs['/@jax_chisq_mode'])
        if attr_mode not in JAX_CHISQ_MODES:
            attr_error = f'unknown jax_chisq_mode {attr_mode!r}'
    if (config_mode is not None and attr_mode is not None and
            config_mode != attr_mode):
        config_error = 'jax_chisq_mode disagrees between config and HDF metadata'
    return config_obj, config_mode, attr_mode, config_error or attr_error


def _configuration_attributes(xa, ya):
    """Compare science config while treating omitted CPU mode as compatible."""
    result = {}
    xraw = xa.get('/@science_config')
    yraw = ya.get('/@science_config')
    xobj, xcfg, xattr, xerr = _declared_jax_mode(xraw, xa)
    yobj, ycfg, yattr, yerr = _declared_jax_mode(yraw, ya)
    config_same = xobj is not None and yobj is not None and not xerr and not yerr
    if config_same:
        xnorm, ynorm = dict(xobj), dict(yobj)
        for normalized in (xnorm, ynorm):
            normalized.pop('--jax-chisq-mode', None)
            normalized.pop('jax_chisq_mode', None)
            normalized.pop('--jax-highpass-mode', None)
            normalized.pop('jax_highpass_mode', None)
        xnorm['--jax-chisq-mode'] = xcfg or 'cpu-compatible'
        ynorm['--jax-chisq-mode'] = ycfg or 'cpu-compatible'
        config_same = xnorm == ynorm
    result['/@science_config'] = dict(
        passed=config_same,
        reference_declared=scalar(xraw) if xraw is not None else None,
        candidate_declared=scalar(yraw) if yraw is not None else None,
        reference_mode=xcfg,
        candidate_mode=ycfg,
        reason=xerr or yerr or (None if config_same else 'configuration mismatch'))
    xmode = xattr or xcfg or 'cpu-compatible'
    ymode = yattr or ycfg or 'cpu-compatible'
    result['/@jax_chisq_mode'] = dict(
        passed=not (xerr or yerr) and xmode == ymode,
        reference_declared=xattr,
        candidate_declared=yattr,
        reference_mode=xmode,
        candidate_mode=ymode,
        reason=None if xmode == ymode else 'jax_chisq_mode mismatch')
    return result


def _live_attributes(attrs):
    """Normalize only executable location and campaign-owned CLI controls."""
    attrs = dict(attrs)
    key = '/@command_line'
    if key not in attrs:
        return attrs
    argv = np.asarray(attrs[key])
    if argv.ndim != 1 or not argv.size:
        return attrs
    args = [str(scalar(value)) for value in argv]
    normalized = [Path(args[0]).name]
    controls = {'--processing-scheme', '--output-path', '--benchmark-evidence-path',
                '--replay-rate'}
    index = 1
    while index < len(args):
        token = args[index]
        if token == '--replay-clock':
            index += 1
            continue
        if token == '--jax-chisq-mode' and index + 1 < len(args):
            mode = args[index + 1]
            if mode == 'cpu-compatible':
                index += 2
                continue
            normalized.extend((token, mode))
            index += 2
            continue
        if token.startswith('--jax-chisq-mode='):
            mode = token.split('=', 1)[1]
            if mode == 'cpu-compatible':
                index += 1
                continue
        if token in controls and index + 1 < len(args):
            index += 2
            continue
        if token.split('=', 1)[0] in controls and '=' in token:
            index += 1
            continue
        normalized.append(token)
        index += 1
    attrs[key] = np.asarray(normalized)
    return attrs


def _compare_detector(reference, candidate, rate=2048, detector="H1", live=False):
    # Normalize one detector so matching stays identical for every instrument.
    a, aa = read_hdf(reference)
    b, ba = read_hdf(candidate)
    if live:
        aa, ba = _live_attributes(aa), _live_attributes(ba)
    provenance = {'/@pycbc_version'} if live else set()
    prefix = detector + '/'
    a = {'H1/' + k[len(prefix):]: v for k, v in a.items() if k.startswith(prefix)}
    b = {'H1/' + k[len(prefix):]: v for k, v in b.items() if k.startswith(prefix)}
    required = {'H1/' + name for name in ('template_hash', 'end_time', 'snr',
                'chisq', 'chisq_dof', 'sigmasq', 'coa_phase')}
    missing = sorted(required - (a.keys() & b.keys()))
    if missing:
        # A buffer-fill block is explicitly declared by the writer. A partial
        # trigger schema, a live block, or an absent declaration still fails.
        inactive = (live and 'H1/gates' in a and 'H1/gates' in b and
                    (a.keys() | b.keys()) <= {'H1/gates', 'H1/psd'} and
                    not (required & (a.keys() | b.keys())) and
                    '/@num_live_detectors' in aa and '/@num_live_detectors' in ba and
                    exact(aa['/@num_live_detectors'], np.asarray(0)) and
                    exact(ba['/@num_live_detectors'], np.asarray(0)))
        if inactive:
            fields = {key: dict(passed=key in a and key in b and exact(a[key], b[key]))
                      for key in sorted(a.keys() | b.keys())}
            attrs = attrs_comparison(aa, ba, provenance)
            passed = all(v['passed'] for v in fields.values()) and all(
                v['passed'] for v in attrs.values())
            return dict(observed_trigger_and_metadata_pass=passed,
                        inactive_buffer_block=True, reference_triggers=0,
                        candidate_triggers=0, fields=fields, attributes=attrs)
        return dict(observed_trigger_and_metadata_pass=False, missing_fields=missing)
    origin = int(min(float(a['H1/end_time'].min()) if a['H1/end_time'].size else 0,
                     float(b['H1/end_time'].min()) if b['H1/end_time'].size else 0))
    ag, am = identities(a, rate, origin)
    bg, bm = identities(b, rate, origin)
    keys = sorted(k for k in ag.keys() & bg.keys() if len(ag[k]) == len(bg[k]) == 1)
    ai = np.asarray([ag[k][0] for k in keys], dtype=int)
    bi = np.asarray([bg[k][0] for k in keys], dtype=int)
    unmatched = lambda groups, other: [
        dict(template_hash=k[0], sample=k[1], end_time=origin + k[1] / rate,
             occurrences=len(v), reason='absent' if k not in other else 'ambiguous duplicate identity')
        for k, v in sorted(groups.items()) if k not in other or len(v) != 1 or len(other[k]) != 1]
    n, m = len(a['H1/end_time']), len(b['H1/end_time'])
    fields = {}
    for key in sorted((a.keys() | b.keys()) - PERF):
        if key not in a or key not in b:
            fields[key] = dict(passed=False, reason='missing dataset',
                               missing_from='reference' if key not in a else 'candidate')
            continue
        x, y = a[key], b[key]
        auxiliary = live and (key.endswith('/psd') or key.endswith('/gates') or key.endswith('/loudest'))
        trigger = key.startswith('H1/') and key.count('/') == 1 and not auxiliary
        if trigger:
            if x.ndim < 1 or y.ndim < 1 or x.shape[0] != n or y.shape[0] != m:
                fields[key] = dict(passed=False, reason='trigger dataset length mismatch')
                continue
            x, y = x[ai], y[bi]
        exact_required = (not trigger or key.endswith(('_dof', '/template_hash', '/end_time'))
                          or x.dtype.kind in 'biu')
        if live and key.endswith('/psd') and x.shape == y.shape:
            same_inf = np.isinf(x) & (x == y)
            x, y = x.copy(), y.copy()
            x[same_inf], y[same_inf] = 0, 0
            kmin = int(30.0 / (rate / ((len(x) - 1) * 2)))
            x[:kmin], y[:kmin] = 0, 0
            with np.errstate(invalid='ignore'):
                ok = np.isclose(x, y, rtol=POLICY['psd_rtol'], atol=0)
            fields[key] = dict(passed=bool(ok.all()), failed_elements=int((~ok).sum()))
        else:
            fields[key] = metrics(x, y, key, exact_required)
        fields[key]['matched_trigger_field'] = trigger
    attrs = attrs_comparison(aa, ba, provenance)
    identity_pass = n == m == len(keys) and not am['off_grid_count'] and not bm['off_grid_count']
    observed = identity_pass and all(v['passed'] for v in fields.values()) and all(v['passed'] for v in attrs.values())
    return dict(reference=str(reference), candidate=str(candidate), reference_triggers=n,
                candidate_triggers=m, matched_triggers=len(keys), reference_unmatched=n-len(keys),
                candidate_unmatched=m-len(keys), reference_identity=am, candidate_identity=bm,
                reference_unmatched_identities=unmatched(ag, bg), candidate_unmatched_identities=unmatched(bg, ag),
                sample_rate=rate, sample_origin=origin, identity_pass=identity_pass,
                fields=fields, attributes=attrs, observed_trigger_and_metadata_pass=observed,
                full_psd_gate='unavailable: complete PSD arrays are not saved in these trigger outputs',
                configuration_gate='HDF metadata checked exactly; full CLI configuration is not stored in trigger HDF',
                full_scientific_qualification='incomplete (PSD/configuration evidence required)' if observed
                    else 'failed (trigger/metadata differences); PSD/configuration evidence also required')



def compare_scientific_hdf(reference, candidate, sample_rate=2048,
                           evidence_reference=None, evidence_candidate=None,
                           _live=False):
    """Compare all trigger fields, metadata, full PSD/strain and geometry.

    Evidence HDF files contain conditioned_strain, segments/*/{strain,psd},
    segment geometry attributes and a normalized science_config JSON attribute.
    Without evidence, observed_trigger_and_metadata_pass remains useful, but
    passed is false. Timed repeats can reuse qualification for the identical arm.
    """
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError('sample_rate must be positive')
    a, _ = read_hdf(reference)
    b, _ = read_hdf(candidate)
    # Include every detector namespace, even when one arm omitted its
    # end_time dataset.  Otherwise an incomplete extra detector could be
    # silently ignored by the identity gate.
    detectors = sorted({prefix for key in a.keys() | b.keys()
                        for prefix in [key.split('/')[0]]
                        if '/' in key and prefix[:1].isalpha()
                        and prefix[1:].isdigit()})
    results = {ifo: _compare_detector(reference, candidate, sample_rate, ifo, live=_live)
               for ifo in detectors}
    observed = bool(results) and all(
        r['observed_trigger_and_metadata_pass'] for r in results.values())
    inactive = bool(results) and all(r.get('inactive_buffer_block', False)
                                     for r in results.values())
    missing = []
    evidence = {}
    if evidence_reference is None or evidence_candidate is None:
        missing = ['full_psd', 'conditioned_strain', 'segment_geometry',
                   'science_configuration']
    else:
        x, xa = read_hdf(evidence_reference)
        y, ya = read_hdf(evidence_candidate)
        for gate, present in (
            ('conditioned_strain', inactive or ('conditioned_strain' in x and 'conditioned_strain' in y)),
            ('full_psd', inactive or (any(k.endswith('/psd') for k in x) and any(k.endswith('/psd') for k in y))),
            ('segment_geometry', any('@analyze_start' in k for k in xa) and any('@analyze_start' in k for k in ya)),
            ('science_configuration', '/@science_config' in xa and '/@science_config' in ya),
        ):
            if not present:
                missing.append(gate)
        for key in sorted(x.keys() | y.keys()):
            if key not in x or key not in y:
                evidence[key] = dict(passed=False, reason='missing evidence dataset')
                continue
            # PSDs deliberately use +inf outside the analysis passband.
            xx, yy = x[key], y[key]
            if key.endswith('/psd') and xx.shape == yy.shape:
                same_inf = np.isinf(xx) & (xx == yy)
                xx, yy = xx.copy(), yy.copy()
                xx[same_inf], yy[same_inf] = 0, 0
                cfg_str = scalar(xa.get('/@science_config') or ya.get('/@science_config'))
                flow = 30.0
                if cfg_str:
                    try:
                        flow = float(json.loads(str(cfg_str)).get('--low-frequency-cutoff', 30.0))
                    except Exception:
                        pass
                prefix = key.rsplit('/', 1)[0]
                delta_f = float(scalar(xa.get(f'{prefix}@delta_f') or ya.get(f'{prefix}@delta_f') or 1.0))
                kmin = int(flow / delta_f)
                xx[:kmin], yy[:kmin] = 0, 0
                with np.errstate(invalid='ignore'):
                    ok = np.isclose(xx, yy, rtol=POLICY['psd_rtol'], atol=0)
                evidence[key] = dict(passed=bool(ok.all()), failed_elements=int((~ok).sum()))
            else:
                exact_req = not (key.endswith('/strain') or key == 'conditioned_strain')
                evidence[key] = metrics(xx, yy, key, exact_required=exact_req)
        attrs = attrs_comparison(xa, ya)
        attrs.pop('/@science_config', None)
        attrs.pop('/@jax_chisq_mode', None)
        attrs.pop('/@jax_highpass_mode', None)
        attrs.update(_configuration_attributes(xa, ya))
        evidence.update(attrs)
    return dict(passed=observed and not missing and all(v['passed'] for v in evidence.values()),
                scope='triggers, all scientific fields and metadata, full PSDs, strain, geometry, configuration',
                observed_trigger_and_metadata_pass=observed, missing_gates=missing,
                detectors=results, evidence=evidence, policy=POLICY,
                evidence_reference=(str(evidence_reference)
                                    if evidence_reference is not None else None),
                evidence_candidate=(str(evidence_candidate)
                                    if evidence_candidate is not None else None))


def compare_live_scientific_hdf(reference, candidate, sample_rate=2048,
                                evidence_reference=None, evidence_candidate=None):
    """Apply the common gates to the live block schema, including buffer fill.

    Saved PSDs and gates are auxiliary arrays, not rows of the trigger table.
    Buffer-fill blocks must agree on their explicit inactive declaration and
    all available data/metadata; they contribute no completed search work.
    """
    return compare_scientific_hdf(reference, candidate, sample_rate,
                                  evidence_reference, evidence_candidate, _live=True)
