import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.types import TimeSeries  # noqa: E402
from pycbc.types.array_torch import TorchArrayData  # noqa: E402
from pycbc.waveform import supernovae  # noqa: E402
from pycbc.waveform.generator import (  # noqa: E402
    TDomainSupernovaeGenerator,
    get_td_generator,
)


@pytest.fixture(autouse=True)
def preserve_scheme_and_caches():
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    supernovae._pc_dict.clear()
    supernovae._torch_pc_dict.clear()
    try:
        yield
    finally:
        supernovae._pc_dict.clear()
        supernovae._torch_pc_dict.clear()
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


@pytest.fixture(params=("cpu", "cuda", "mps"))
def torch_device(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device unavailable")
    return device


@pytest.fixture
def pc_files(tmp_path):
    components = np.array(
        [
            [0.2, -0.4, 0.6, -0.8, 1.0],
            [1.5, 1.0, 0.5, 0.0, -0.5],
            [-0.3, 0.7, -1.1, 1.3, -1.7],
        ],
        dtype=np.float64,
    )
    first = tmp_path / "principal_components.hdf"
    second = tmp_path / "other_principal_components.hdf"
    for filename, values in (
        (first, components),
        (second, components + 10.0),
    ):
        with h5py.File(filename, "w") as pc_file:
            pc_file.create_dataset("principal_components", data=values)
    return first, second, components


def _activate_scheme(state):
    _scheme.Scheme._single = None
    _scheme.mgr.state = state


def _parameters(filename, no_of_pcs):
    return {
        "principal_components_file": filename,
        "no_of_pcs": no_of_pcs,
        "distance": 12.5,
        "delta_t": 1 / 4096,
    }


def test_principal_component_cache_is_keyed_by_filename(pc_files):
    first, second, components = pc_files
    _activate_scheme(_scheme.CPUScheme())
    coefficients = np.array([0.4, -0.2, 0.7])

    first_plus, _ = supernovae.get_corecollapse_bounce(
        **_parameters(first, 3), coefficients_array=coefficients
    )
    second_plus, _ = supernovae.get_corecollapse_bounce(
        **_parameters(second, 3), coefficients_array=coefficients
    )

    conversion = 3.08567758128e22
    np.testing.assert_allclose(
        first_plus.numpy(), coefficients @ components / (12.5 * conversion)
    )
    np.testing.assert_allclose(
        second_plus.numpy(),
        coefficients @ (components + 10.0) / (12.5 * conversion),
    )
    assert len(supernovae._pc_dict) == 2


def test_corecollapse_validates_coefficient_count(pc_files):
    first, _, _ = pc_files
    _activate_scheme(_scheme.CPUScheme())

    with pytest.raises(ValueError, match="Expected 2.*got 1"):
        supernovae.get_corecollapse_bounce(
            **_parameters(first, 2), coefficients_array=[0.4]
        )


def test_corecollapse_bounce_runs_on_torch_device_without_host_transfer(
    monkeypatch, torch_device, pc_files
):
    first, _, _ = pc_files
    coefficients = np.array([0.4, -0.2, 0.7])

    _activate_scheme(_scheme.CPUScheme())
    references = []
    for no_of_pcs in (2, 3):
        plus, cross = supernovae.get_corecollapse_bounce(
            **_parameters(first, no_of_pcs),
            coefficients_array=coefficients,
        )
        references.append((plus.numpy().copy(), cross.numpy().copy()))

    _activate_scheme(_scheme.TorchScheme(torch_device))
    dtype = torch.float32 if torch_device == "mps" else torch.float64
    tensor_coefficients = torch.as_tensor(
        coefficients, dtype=dtype, device=torch_device
    )

    def reject_host_transfer(_self):
        raise AssertionError("core-collapse waveform copied data to the host")

    def reject_numpy_vector_operation(*_args, **_kwargs):
        raise AssertionError("core-collapse reconstruction used NumPy")

    with monkeypatch.context() as patch:
        patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
        patch.setattr(supernovae.numpy, "dot", reject_numpy_vector_operation)
        patch.setattr(
            supernovae.numpy, "zeros", reject_numpy_vector_operation
        )
        first_result = supernovae.get_corecollapse_bounce(
            **_parameters(first, 2),
            coefficients_array=tensor_coefficients,
        )
        second_result = supernovae.get_corecollapse_bounce(
            **_parameters(first, 3),
            coeff_2=coefficients[2],
            coeff_0=coefficients[0],
            coeff_1=coefficients[1],
        )

    tolerance = 3e-6 if torch_device == "mps" else 2e-14
    for result, reference in zip(
        (first_result, second_result), references
    ):
        for series, expected in zip(result, reference):
            assert isinstance(series, TimeSeries)
            assert series._data.tensor.device.type == torch_device
            assert series._data.tensor.dtype == dtype
            assert series.delta_t == 1 / 4096
            np.testing.assert_allclose(
                series._data.tensor.detach().cpu().numpy(),
                expected,
                rtol=tolerance,
                atol=1e-32,
            )

    assert len(supernovae._torch_pc_dict) == 2
    assert get_td_generator("CoreCollapseBounce") is TDomainSupernovaeGenerator
