"""Exercise CLI warm-up dispatch without loading the search's frame fixture."""
import ast
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("has_candidates", [False, True])
def test_warmup_uses_production_correlations(monkeypatch, has_candidates):
    source = Path(__file__).resolve().parents[1] / "bin" / "pycbc_inspiral_jax"
    tree = ast.parse(source.read_text())
    warmup = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Try) and any(
            isinstance(child, ast.Constant)
            and child.value == "Warming up JAX JIT compilation during setup"
            for child in ast.walk(node)
        )
    )
    # Execute the body directly so an exception cannot be hidden by the CLI's
    # optional warm-up error handler.
    module = ast.Module(body=warmup.body, type_ignores=[])
    calls = []
    tensor = object()
    corr = SimpleNamespace(_batch_tensor=tensor)
    results = [(None, 1, corr if has_candidates else None, [], [])] * 2
    templates = [SimpleNamespace(f_lower=20), SimpleNamespace(f_lower=30)]
    psd = object()
    edges = {20: [0, 8, 16], 30: [2, 6, 10, 16]}
    cached_templates = []

    def cached_chisq_bins(template, template_psd):
        assert template_psd is psd
        cached_templates.append(template)
        bins = edges[template.f_lower]
        template._bin_cache = {id(psd): bins}
        return bins

    def cache_batch_power_chisq_bins_jax(veto, batch, template_psd):
        return [veto.cached_chisq_bins(template, template_psd)
                for template in batch]

    class JAXScheme:
        pass

    def fake_module(name, **members):
        value = ModuleType(name)
        value.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, value)

    fake_module("pycbc", scheme=SimpleNamespace(
        mgr=SimpleNamespace(state=JAXScheme()), JAXScheme=JAXScheme))
    fake_module("pycbc.filter.matchedfilter_jax",
                live_template_norms_jax=lambda *args, **kwargs: [2.0, 3.0])
    fake_module("pycbc.vetoes.chisq_jax",
                batch_power_chisq_jax=lambda *args, **kw: calls.append((args, kw)),
                cache_batch_power_chisq_bins_jax=
                cache_batch_power_chisq_bins_jax)
    fake_module("pycbc.opt", LimitedSizeDict=lambda **kw: {})

    class Bank:
        def __len__(self):
            return 2

        def get_batch(self, indices):
            assert indices == [0, 1]
            return templates

    scope = dict(
        logging=logging, batch_size=16, bank=Bank(), cluster_window=4,
        segments=[SimpleNamespace(psd=psd, _epoch=0,
                                  analyze=SimpleNamespace(start=12))],
        matched_filter=SimpleNamespace(
            batched_matched_filter_and_cluster=lambda *args, **kw: results),
        power_chisq=SimpleNamespace(do=True, num_bins=2, snr_threshold=5.5,
                                   cached_chisq_bins=cached_chisq_bins),
        flow=30.0, tnum_start=0,
    )
    exec(compile(module, str(source), "exec"), scope)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is (tensor if has_candidates else None)
    assert args[1] is results
    assert args[2] is templates
    assert args[3] is psd
    assert args[4] == 12
    assert kwargs["snr_threshold"] == 5.5
    assert cached_templates == templates
    for template in templates:
        assert template._bin_cache[id(psd)] is edges[template.f_lower]
