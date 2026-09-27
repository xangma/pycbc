"""Check CLI batching and conditioning boundaries without frame data."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'bin' / 'pycbc_inspiral'


def run_setup(jax, device, batch_size):
    tree = ast.parse(SOURCE.read_text())
    start = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'is_jax'
                         for t in node.targets))
    end = next(i for i in range(start, len(tree.body))
               if isinstance(tree.body[i], ast.FunctionDef))
    context = next(node for node in tree.body if isinstance(node, ast.With)
                   and isinstance(node.items[0].context_expr, ast.Name)
                   and node.items[0].context_expr.id == 'ctx')
    prepare_start = next(i for i, node in enumerate(tree.body)
                         if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'flow'
                                 for t in node.targets))
    prepare_end = tree.body.index(context)
    context_end = next(i for i, node in enumerate(context.body)
                       if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'out_types'
                               for t in node.targets))
    events = []

    def check_preparation_scheme():
        if jax:
            assert ctx.active

    class LegacyScheme:
        device_spec = device
        active = False

        def __enter__(self):
            self.active = True

        def __exit__(self, *args):
            self.active = False

    class JAXScheme(LegacyScheme):
        pass

    ctx = JAXScheme() if jax else LegacyScheme()
    strain_data = object()
    frequency_segments = [object()]

    def load(*args, **kwargs):
        check_preparation_scheme()
        events.append(('load', ctx.active))
        return strain_data

    def segment(options, data):
        check_preparation_scheme()
        assert data is strain_data
        events.append(('segment', ctx.active))
        return SimpleNamespace(freq_len=65, time_len=128, delta_f=0.5,
                               fourier_segments=fourier)

    def fourier():
        check_preparation_scheme()
        events.append(('fourier', ctx.active))
        return frequency_segments

    def associate(options, segments, data, flen, delta_f, flow, **kwargs):
        check_preparation_scheme()
        assert segments is frequency_segments
        assert data is strain_data
        assert (flen, delta_f, flow) == (65, 0.5, 30)
        assert kwargs == dict(dyn_range_factor=1, precision='single')
        events.append(('psd', ctx.active))

    def error(message):
        raise ValueError(message)

    scope = dict(stage_event=lambda *args, **kwargs: None,
                 events=SimpleNamespace(EventManager=object),
                 save_inspiral_evidence=lambda *args: None,
                 ctx=ctx, scheme=SimpleNamespace(JAXScheme=JAXScheme),
                 nullcontext=nullcontext,
                 opt=SimpleNamespace(batch_size=batch_size,
                                     multiprocessing_nprocesses=None,
                                     low_frequency_cutoff=30,
                                     fft_backends=['jax'] if jax else ['mkl']),
                 parser=SimpleNamespace(error=error), DYN_RANGE_FAC=1,
                 inj_filter_rejector=object(),
                 logging=SimpleNamespace(info=lambda *args: None),
                 psd=SimpleNamespace(associate_psds_to_segments=associate),
                 strain=SimpleNamespace(from_cli=load,
                                        StrainSegments=SimpleNamespace(
                                            from_cli=segment)))
    exec(compile(ast.Module(body=tree.body[start:end], type_ignores=[]),
                 str(SOURCE), 'exec'), scope)
    preparation = tree.body[prepare_start:prepare_end]
    scoped_preparation = ast.With(items=context.items,
                                  body=context.body[:context_end])
    module = ast.fix_missing_locations(ast.Module(
        body=preparation + [scoped_preparation],
        type_ignores=[]))
    exec(compile(module, str(SOURCE), 'exec'), scope)
    return scope['batch_size'], scope['use_batching'], events


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('batch_size', [None, 1])
def test_legacy_conditioning_stays_outside_context(device, batch_size):
    assert run_setup(False, device, batch_size) == (
        1, False, [('load', False), ('segment', False),
                   ('fourier', True), ('psd', True)])


@pytest.mark.parametrize('device,requested,expected', [
    ('cpu', None, 16), ('cuda:0', None, 64), ('cpu', 8, 8), ('cpu', 1, 1)])
def test_jax_conditioning_and_batch_size(device, requested, expected):
    assert run_setup(True, device, requested) == (
        expected, expected > 1, [('load', True), ('segment', True),
                                 ('fourier', True), ('psd', True)])


def test_legacy_batching_is_rejected():
    with pytest.raises(ValueError, match='JAX processing scheme'):
        run_setup(False, 'cuda:0', 64)
