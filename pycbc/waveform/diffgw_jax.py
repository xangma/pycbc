# Copyright (C) 2026 PyCBC contributors
#
# This program is free software under the GNU General Public License,
# version 3 or (at your option) any later version.
"""diffGW generation and original-implementation validation for JAX banks.

The runtime allowlist admits aligned-spin models natively supported by diffgw,
including TaylorF2, IMRPhenomD, and IMRPhenomXAS with standard 3.5PN phase
conventions and zero reference frequency. Synthesis always uses float64;
storage may use complex64 or complex128. Unsupported requests raise; original
waveform synthesis is an explicit validation option. Compressed waveforms keep their separate decompression path.
"""

import importlib.util
import math
import types

import numpy as np

from pycbc.types import Array, FrequencySeries, TimeSeries
from pycbc.types.array_jax import JAXArrayData, _reference_enabled, to_jax
from pycbc.waveform.waveform import default_args

# Bank bookkeeping is not a waveform option. Unknown explicit options are
# deliberately rejected, even if an upstream generator silently ignores them.
_BOOKKEEPING = {'template_hash', 'template_duration', 'duration', 'return_hc', 'dtype', 'sample_points'}
_PHYSICAL = {'mass1', 'mass2', 'spin1z', 'spin2z', 'coa_phase', 'inclination',
             'distance'}
_GEOMETRY = {'approximant', 'f_lower', 'f_final', 'delta_f', 'delta_t'}
_ORDERS = {'phase_order': (-1, 7), 'spin_order': (-1, 7),
           'amplitude_order': (-1, 0), 'tidal_order': (-1,)}

_SUPPORTED_APPROXIMANTS = ('TaylorF2', 'IMRPhenomD', 'IMRPhenomXAS')


def available_approximants(domain, requested_scheme):
    """List the actual selected provider without importing diffGW."""
    if 'waveform' in requested_scheme.jax_reference_operations:
        from pycbc.waveform import waveform, waveform_modes
        registries = {
            'fd': waveform.cpu_fd, 'td': waveform.cpu_td,
            'filter': waveform._inspiral_fd_filters,
            'fd_modes': waveform_modes._mode_waveform_fd,
            'td_modes': waveform_modes._mode_waveform_td,
        }
        names = list(registries[domain])
        return sorted(names) if domain.endswith('_modes') else names
    if domain in ('fd', 'filter') and importlib.util.find_spec('diffgw') is not None:
        return list(_SUPPORTED_APPROXIMANTS)
    return []


def _is_provider_enabled(bank):
    return getattr(bank, 'enable_diffgw', None) is not False


def _reason(bank, params, device):
    if not _is_provider_enabled(bank):
        return 'JAX generation requires diffGW; use the waveform reference selector for original synthesis'
    if importlib.util.find_spec('diffgw') is None:
        return "diffGW is required; install 'diffgw[jax]'"
    dev_type = str(device).split(':')[0].lower()
    if dev_type not in ('cpu', 'cuda', 'gpu', 'tpu', 'jax'):
        return 'float64 diffgw generation requires CPU or CUDA'
    if (getattr(bank, 'has_compressed_waveforms', False)
            and getattr(bank, 'enable_compressed_waveforms', False)):
        return 'compressed waveform generation takes precedence'
    approx = params.get('approximant')
    if approx not in _SUPPORTED_APPROXIMANTS:
        return f'approximant {approx} is outside the runtime allowlist'
    for key in ('mass1', 'mass2'):
        if params[key] <= 0:
            return f'{key} must be positive'
    for key in ('spin1z', 'spin2z'):
        if not abs(params[key]) <= 1:
            return f'{key} must be in [-1, 1]'
    for key in (_PHYSICAL | _GEOMETRY) - {'approximant'}:
        if params.get(key) is not None and not math.isfinite(params[key]):
            return f'{key} must be finite'
    if not 0 < params['f_lower'] < params['f_final']:
        return 'f_lower must be positive and below f_final'
    for key, value in params.items():
        if key in _PHYSICAL | _GEOMETRY | _BOOKKEEPING:
            continue
        if key in _ORDERS:
            if value not in _ORDERS[key]:
                return f'{key}={value!r} is outside the runtime allowlist'
        elif key in ('lambda1', 'lambda2') and value in (None, 0):
            continue
        elif key == 'mode_array':
            if value is not None:
                return 'explicit mode_array requires reference generation'
        elif key == 'taper':
            # The inspiral CLI passes None even when tapering is disabled.
            # get_waveform_filter applies a taper only for a non-None value.
            if value is not None:
                return 'taper requests require reference generation'
        elif key not in default_args:
            return f'option {key} is outside the runtime allowlist'
        elif value != default_args[key]:
            return f'non-default {key} requires reference generation'
    return None


def _original_waveform(function, parameters, **kwargs):
    import jax
    from pycbc import scheme
    from pycbc.waveform import waveform
    from pycbc.reference_jax import cpu_reference
    parameters = dict(parameters)
    if 'sample_points' in parameters:
        parameters['sample_points'] = np.asarray(parameters['sample_points'])
    # These original generators only call LAL with host parameters and wrap
    # its results. Run them in the caller's native-library environment; a
    # subprocess can round differently after a different library load order.
    pure_generators = {
        'get_fd_waveform': (waveform.cpu_fd, waveform._lalsim_fd_waveform),
        'get_fd_waveform_sequence': (waveform.fd_sequence,
                                     waveform._lalsim_fd_sequence),
        'get_waveform_filter': (waveform.cpu_fd, waveform._lalsim_fd_waveform),
        'get_two_pol_waveform_filter': (waveform.cpu_fd, waveform._lalsim_fd_waveform),
    }
    if function in pure_generators:
        registry, generator = pure_generators[function]
        approximant = parameters.get('approximant')
        specialized_filter = (function == 'get_waveform_filter' and
                              approximant in waveform._inspiral_fd_filters)
        if registry.get(approximant) is generator and not specialized_filter:
            original = getattr(waveform, function)
            namespace = original.__globals__.copy()
            # Rebind only this function's read-only dispatch lookup. The real
            # manager remains JAX; its ordinary Series constructors own the
            # returned samples on the selected device. Parameter checking and
            # cutoff resolution execute the unchanged original function body.
            # A plain sentinel has no Scheme destructor or shared state. The
            # cloned availability helpers use only these local CPU registries.
            state = object()
            namespace['_scheme'] = types.SimpleNamespace(
                mgr=types.SimpleNamespace(state=state), JAXScheme=scheme.JAXScheme)
            namespace.update(fd_wav={type(state): waveform.cpu_fd},
                             td_wav={type(state): waveform.cpu_td},
                             filter_wav={type(state): waveform._inspiral_fd_filters})
            for name in ('fd_approximants', 'td_approximants', 'filter_approximants'):
                helper = getattr(waveform, name)
                namespace[name] = types.FunctionType(
                    helper.__code__, namespace, helper.__name__,
                    helper.__defaults__, helper.__closure__)
            native = types.FunctionType(original.__code__, namespace,
                                        original.__name__, original.__defaults__,
                                        original.__closure__)
            if function in ('get_waveform_filter', 'get_two_pol_waveform_filter'):
                # These FD-only branches copy/resize the native samples; no
                # CPU FFT, specialized filter or TD fallback is executed.
                from pycbc.types import zeros
                output = zeros(kwargs['length'], dtype=kwargs['dtype'])
                if function == 'get_waveform_filter':
                    return (native(output, **parameters),)
                cross = zeros(kwargs['length'], dtype=kwargs['dtype'])
                return native(output, cross, None, **parameters)
            return native(**parameters)
    records = cpu_reference('waveform', function=function,
                            parameters=parameters, **kwargs)
    def wrap(record):
        if isinstance(record, dict):
            return {key: wrap(value) for key, value in record.items()}
        if len(record) != 5 or not isinstance(record[1], str):
            return tuple(wrap(value) for value in record)
        values, kind, spacing, epoch, metadata = record
        data = JAXArrayData(jax.device_put(values, scheme.mgr.state.jax_device))
        if kind == 'frequency':
            series = FrequencySeries(data, delta_f=spacing, epoch=epoch, copy=False)
        elif kind == 'time':
            series = TimeSeries(data, delta_t=spacing, epoch=epoch, copy=False)
        else:
            series = Array(data, copy=False)
        for name, value in metadata.items():
            setattr(series, name, value)
        return series
    return wrap(records)


def _native_polarizations(parameters, frequencies):
    import diffgw
    import jax.numpy as jnp
    kwargs = {name: parameters[name] for name in _PHYSICAL - {'coa_phase'}}
    app = parameters['approximant']
    # diffGW TaylorF2's positive amplitude differs by a sign from PyCBC's
    # convention; shifting the orbital phase by pi/2 supplies that sign.
    kwargs['phic'] = parameters['coa_phase'] + (np.pi / 2 if app == 'TaylorF2' else 0)
    if app == 'TaylorF2':
        kwargs['f_isco_cutoff'] = False
    else:
        kwargs['f_lower'] = parameters['f_lower']
        # Original Phenom APIs resolve f_ref=0 to the lower frequency;
        # diffGW's None instead selects its coalescence-phase convention.
        kwargs['f_ref'] = parameters.get('f_ref') or parameters['f_lower']
    return diffgw.get_fd_waveform(app, backend='jax', sample_frequencies=frequencies,
                                  dtype=jnp.float64, **kwargs)


def _resolve_parameters(template, kwargs):
    from pycbc.waveform.waveform import props, get_waveform_end_frequency
    p = props(template, **kwargs)
    final_function = p.pop('f_final_func', '')
    if final_function:
        from pycbc.pnutils import named_frequency_cutoffs
        p['f_final'] = named_frequency_cutoffs[final_function](p)
    if not p.get('f_final'):
        p['f_final'] = get_waveform_end_frequency(**p)
    return p


def generate_waveform(function, template=None, **kwargs):
    """Bridge PyCBC FD/sequence APIs; CPU synthesis is explicitly selected."""
    import jax.numpy as jnp
    from types import SimpleNamespace
    from pycbc.waveform.waveform import props, check_args
    from pycbc.waveform import parameters
    from pycbc.types.array_jax import _ensure_x64
    _ensure_x64()
    if _reference_enabled('waveform'):
        return _original_waveform(function, props(template, **kwargs))
    if function == 'get_td_waveform':
        raise NotImplementedError('diffGW time-domain bridging is not supported; '
                                  'select waveform reference validation')
    p = _resolve_parameters(template, kwargs)
    if function == 'get_fd_waveform_sequence':
        p['delta_f'], p['f_lower'] = -1, -1
    check_args(p, parameters.fd_required if function == 'get_fd_waveform'
               else parameters.fd_required)
    if function == 'get_fd_waveform_sequence':
        frequencies = to_jax(p['sample_points'])
        if frequencies.ndim != 1 or not len(frequencies):
            raise ValueError('sample_points must be a nonempty one-dimensional sequence')
        p['f_lower'], p['f_final'] = 1., max(2., float(jnp.max(frequencies)))
    else:
        if p['delta_f'] <= 0:
            raise ValueError('delta_f must be positive')
        frequencies = jnp.arange(int(p['f_final'] / p['delta_f']) + 1,
                                 dtype=jnp.float64) * p['delta_f']
    reason = _reason(SimpleNamespace(enable_diffgw=True), p, 'jax')
    if reason:
        raise ValueError('JAX waveform generation does not support request: ' + reason)
    if function == 'get_fd_waveform_sequence' and p['approximant'] != 'TaylorF2':
        # The original sequence API resolves f_ref=0 to its first sample.
        p['f_ref'] = float(frequencies[0])
    hp, hc = _native_polarizations(p, frequencies)
    dtype = p.get('dtype', np.complex128)
    if np.dtype(dtype) not in (np.dtype('complex64'), np.dtype('complex128')):
        raise TypeError('waveform storage dtype must be complex64 or complex128')
    if function == 'get_fd_waveform_sequence':
        return tuple(Array(JAXArrayData(values.astype(dtype)), copy=False)
                     for values in (hp, hc))
    mask = ((frequencies >= p['f_lower']) & (frequencies <= p['f_final']))
    return tuple(FrequencySeries(JAXArrayData(jnp.where(mask, values, 0).astype(dtype)),
                                delta_f=p['delta_f'], epoch=-1 / p['delta_f'], copy=False)
                 for values in (hp, hc))


def generate_filter(out, template=None, **kwargs):
    """Fill the caller's storage; original CPU output is opt-in validation."""
    from pycbc.waveform.waveform import props, get_waveform_filter_length_in_time
    p = props(template, **kwargs)
    if _reference_enabled('waveform'):
        series, = _original_waveform('get_waveform_filter', p,
                                    length=len(out), dtype=out.dtype)
        out[:] = series
        series._data = out._data
        return series
    p = _resolve_parameters(template, kwargs)
    p['dtype'] = out.dtype
    p['f_final'] = min(p.get('f_final') or (len(out) - 1) * p['delta_f'],
                       (len(out) - 1) * p['delta_f'])
    hp, _ = generate_waveform('get_fd_waveform', **p)
    hp.resize(len(out))
    out[:] = hp
    hp._data = out._data
    hp.chirp_length = hp.length_in_time = get_waveform_filter_length_in_time(**p)
    return hp


def generate_modes(function, template=None, **kwargs):
    """Reject unsupported mode bridging; native modes are validation-only."""
    from pycbc.waveform.waveform import props
    if _reference_enabled('waveform'):
        return _original_waveform(function, props(template, **kwargs))
    raise NotImplementedError('diffGW separate-mode bridging is not supported; '
                              'select waveform reference validation')


def generate_two_pol_filter(outplus, outcross, template=None, **kwargs):
    """Fill both polarization buffers using the selected waveform provider."""
    from pycbc.waveform.waveform import props, get_waveform_filter_length_in_time
    inclination_alias = (not hasattr(template, 'inclination') and
                         'inclination' not in kwargs and hasattr(template, 'alpha3'))
    if inclination_alias:
        kwargs = dict(kwargs, inclination=template.alpha3)
    p = props(template, **kwargs)
    if inclination_alias:
        p.pop('alpha3', None)
    if _reference_enabled('waveform'):
        hp, hc = _original_waveform('get_two_pol_waveform_filter', p,
                                  length=len(outplus), dtype=outplus.dtype)
    else:
        p['dtype'] = outplus.dtype
        hp, hc = generate_waveform('get_fd_waveform', **p)
        hp.resize(len(outplus))
        hc.resize(len(outcross))
        hp.chirp_length = hc.chirp_length = get_waveform_filter_length_in_time(**p)
        hp.length_in_time = hc.length_in_time = hp.chirp_length
    for output, series in ((outplus, hp), (outcross, hc)):
        output[:] = series
        series._data = output._data
    return hp, hc
