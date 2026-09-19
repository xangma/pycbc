"""Check CLI batching and conditioning boundaries without loading frame data."""
import ast
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
    condition = next(node for node in context.body
                     if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Name)
                     and node.test.id == 'is_jax')
    events = []

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

    def load(*args, **kwargs):
        events.append(('load', ctx.active))
        return object()

    def segment(*args):
        events.append(('segment', ctx.active))
        return object()

    def error(message):
        raise ValueError(message)

    scope = dict(ctx=ctx, scheme=SimpleNamespace(JAXScheme=JAXScheme),
                 opt=SimpleNamespace(batch_size=batch_size),
                 parser=SimpleNamespace(error=error), DYN_RANGE_FAC=1,
                 inj_filter_rejector=object(),
                 strain=SimpleNamespace(from_cli=load,
                     StrainSegments=SimpleNamespace(from_cli=segment)))
    exec(compile(ast.Module(body=tree.body[start:end], type_ignores=[]),
                 str(SOURCE), 'exec'), scope)
    with ctx:
        exec(compile(ast.Module(body=[condition], type_ignores=[]),
                     str(SOURCE), 'exec'), scope)
    return scope['batch_size'], scope['use_batching'], events


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('batch_size', [None, 1])
def test_legacy_conditioning_stays_outside_context(device, batch_size):
    assert run_setup(False, device, batch_size) == (
        1, False, [('load', False), ('segment', False)])


@pytest.mark.parametrize('device,requested,expected', [
    ('cpu', None, 16), ('cuda:0', None, 64), ('cpu', 8, 8), ('cpu', 1, 1)])
def test_jax_conditioning_and_batch_size(device, requested, expected):
    assert run_setup(True, device, requested) == (
        expected, expected > 1, [('load', True), ('segment', True)])


def test_legacy_batching_is_rejected():
    with pytest.raises(ValueError, match='JAX processing scheme'):
        run_setup(False, 'cuda:0', 64)
