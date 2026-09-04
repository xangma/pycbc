"""Focused tests for Torch-aware inference model primitives."""

import pytest


torch = pytest.importorskip("torch")

from pycbc.inference.models import base as model_base  # noqa: E402
from pycbc.inference.models.base_data import BaseDataModel  # noqa: E402
from pycbc.inference.models.data_utils import check_for_nans  # noqa: E402


class _UserTensor(torch.Tensor):
    """A tensor subclass declared outside the torch package."""


class _Prior:
    variable_args = ("x",)

    def __init__(self, invalid=False):
        self.invalid = invalid

    @staticmethod
    def apply_boundary_conditions(**params):
        return params

    def __call__(self, *, x):
        if self.invalid:
            return x.new_full((), torch.nan)
        return -0.5 * x.square()


class _Model(model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        self.likelihood_calls = 0
        super().__init__(*args, **kwargs)

    def _loglikelihood(self):
        self.likelihood_calls += 1
        return 2.0 * self.current_params["x"]


class _DataModel(BaseDataModel):
    def __init__(self, *args, **kwargs):
        self.loglr_calls = 0
        super().__init__(*args, **kwargs)

    def _loglikelihood(self):
        return self._loglr() + self._lognl()

    def _loglr(self):
        self.loglr_calls += 1
        return 3.0 * self.current_params["x"]

    def _lognl(self):
        return self.current_params["x"].new_tensor(-1.0)


@pytest.mark.parametrize("tensor_subclass", [False, True])
def test_model_statistics_stay_on_device_and_keep_gradients(
        monkeypatch, tensor_subclass):
    x = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    value = x.as_subclass(_UserTensor) if tensor_subclass else x
    finite = _Model(("x",), prior=_Prior())
    no_prior = _Model(("x",))
    invalid = _Model(("x",), prior=_Prior(invalid=True))
    data_model = _DataModel(("x",), data={}, prior=_Prior())
    invalid_data = _DataModel(
        ("x",), data={}, prior=_Prior(invalid=True)
    )
    for model in (finite, no_prior, invalid, data_model, invalid_data):
        model.update(x=value)

    def reject_numpy(*_args, **_kwargs):
        raise AssertionError("model statistic evaluation left Torch")

    with monkeypatch.context() as patch:
        patch.setattr(model_base.numpy, "isnan", reject_numpy)
        finite_posterior = finite.logposterior
        no_prior_posterior = no_prior.logposterior
        invalid_posterior = invalid.logposterior
        data_logplr = data_model.logplr
        invalid_data_logplr = invalid_data.logplr

    assert all(
        isinstance(value, torch.Tensor)
        for value in (finite_posterior, no_prior_posterior, data_logplr)
    )
    assert invalid.likelihood_calls == 0
    assert invalid_data.loglr_calls == 0
    assert torch.isneginf(invalid_posterior)
    assert torch.isneginf(invalid_data_logplr)

    (finite_posterior + no_prior_posterior + data_logplr).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad)

    # Public statistics retain the established host-scalar contract.
    (public_prior,) = finite.get_current_stats(["logprior"])
    assert isinstance(public_prior, float)


class _TorchSeries:
    backend = "torch"

    def __init__(self, values):
        self.backend_array = values

    def numpy(self):
        raise AssertionError("Torch NaN validation copied data to NumPy")


def test_nan_validation_uses_the_device_tensor():
    finite = _TorchSeries(torch.tensor([0.0, 1.0]))
    contains_nan = _TorchSeries(torch.tensor([0.0, torch.nan]))

    check_for_nans({"H1": finite})
    with pytest.raises(ValueError, match="NaN found in strain from L1"):
        check_for_nans({"L1": contains_nan})


@pytest.mark.parametrize("tensor_subclass", [False, True])
def test_scalar_helpers_use_public_backend_storage(
        monkeypatch, tensor_subclass):
    x = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    value = (
        x.as_subclass(_UserTensor) if tensor_subclass else _TorchSeries(x)
    )
    statistic = model_base._replace_nan_with_neginf(value)
    assert not model_base._is_neginf_scalar(value)
    assert model_base._is_neginf_scalar(
        _TorchSeries(x.new_full((), -torch.inf))
    )
    prior = model_base._NoPrior()(x=value)
    torch.testing.assert_close(prior, x.new_zeros(()))

    sampling = model_base.SamplingTransforms((), (), (), [])
    monkeypatch.setattr(
        model_base.transforms, "compute_jacobian", lambda *a, **kw: value
    )
    logjacobian = sampling.logjacobian(x=value)
    torch.testing.assert_close(logjacobian, x.log())
    (statistic + logjacobian).backward()
    torch.testing.assert_close(x.grad, x.new_tensor(5.0))
