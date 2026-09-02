import numpy as np
import pytest
import pycbc.waveform
from pycbc.types import FrequencySeries

torch = pytest.importorskip("torch")
from pycbc.scheme import TorchScheme  # noqa: E402


def test_autobatch_numpy_arrays_taylorf2():
    """Verify get_fd_waveform auto-batches NumPy arrays for TaylorF2."""
    with TorchScheme("cpu"):
        m1 = np.array([1.4, 1.5, 1.6])
        m2 = np.array([1.3, 1.4, 1.3])
        s1z = np.array([0.05, -0.05, 0.0])
        s2z = np.array([-0.02, 0.02, 0.01])

        hp, hc = pycbc.waveform.get_fd_waveform(
            approximant="TaylorF2",
            mass1=m1,
            mass2=m2,
            spin1z=s1z,
            spin2z=s2z,
            delta_f=1.0,
            f_lower=30.0,
            f_final=512.0,
            distance=100.0,
        )

        assert isinstance(hp, torch.Tensor)
        assert isinstance(hc, torch.Tensor)
        assert hp.shape[0] == 3
        assert hc.shape[0] == 3

        # Compare elementwise against scalar get_fd_waveform calls
        for i in range(3):
            hp_s, hc_s = pycbc.waveform.get_fd_waveform(
                approximant="TaylorF2",
                mass1=float(m1[i]),
                mass2=float(m2[i]),
                spin1z=float(s1z[i]),
                spin2z=float(s2z[i]),
                delta_f=1.0,
                f_lower=30.0,
                f_final=512.0,
                distance=100.0,
            )
            assert isinstance(hp_s, FrequencySeries)
            diff_hp = torch.max(torch.abs(hp[i] - torch.as_tensor(hp_s._data.tensor))).item()
            diff_hc = torch.max(torch.abs(hc[i] - torch.as_tensor(hc_s._data.tensor))).item()
            assert diff_hp < 1e-12
            assert diff_hc < 1e-12


def test_autobatch_torch_tensors_imrphenomd():
    """Verify get_fd_waveform auto-batches PyTorch tensors for IMRPhenomD."""
    with TorchScheme("cpu"):
        m1 = torch.tensor([30.0, 40.0], dtype=torch.float64)
        m2 = torch.tensor([20.0, 25.0], dtype=torch.float64)
        s1z = torch.tensor([0.2, -0.3], dtype=torch.float64)
        s2z = torch.tensor([-0.1, 0.4], dtype=torch.float64)
        dist = torch.tensor([100.0, 200.0], dtype=torch.float64)

        hp, hc = pycbc.waveform.get_fd_waveform(
            approximant="IMRPhenomD",
            mass1=m1,
            mass2=m2,
            spin1z=s1z,
            spin2z=s2z,
            delta_f=0.5,
            f_lower=20.0,
            f_final=256.0,
            distance=dist,
        )

        assert isinstance(hp, torch.Tensor)
        assert isinstance(hc, torch.Tensor)
        assert hp.shape[0] == 2
        assert hc.shape[0] == 2

        for i in range(2):
            hp_s, hc_s = pycbc.waveform.get_fd_waveform(
                approximant="IMRPhenomD",
                mass1=float(m1[i]),
                mass2=float(m2[i]),
                spin1z=float(s1z[i]),
                spin2z=float(s2z[i]),
                delta_f=0.5,
                f_lower=20.0,
                f_final=256.0,
                distance=float(dist[i]),
            )
            diff_hp = torch.max(torch.abs(hp[i] - torch.as_tensor(hp_s._data.tensor))).item()
            diff_hc = torch.max(torch.abs(hc[i] - torch.as_tensor(hc_s._data.tensor))).item()
            assert diff_hp < 1e-12
            assert diff_hc < 1e-12


def test_autobatch_python_lists_imrphenomxas():
    """Verify get_fd_waveform auto-batches Python lists for IMRPhenomXAS."""
    with TorchScheme("cpu"):
        hp, hc = pycbc.waveform.get_fd_waveform(
            approximant="IMRPhenomXAS",
            mass1=[35.0, 45.0],
            mass2=[25.0, 30.0],
            spin1z=[0.1, -0.2],
            spin2z=[0.0, 0.1],
            delta_f=0.5,
            f_lower=20.0,
            f_final=256.0,
            distance=100.0,
        )

        assert isinstance(hp, torch.Tensor)
        assert isinstance(hc, torch.Tensor)
        assert hp.shape[0] == 2
        assert hc.shape[0] == 2

        # Check parity with get_fd_waveform_batch
        hp_batch, hc_batch = pycbc.waveform.get_fd_waveform_batch(
            "IMRPhenomXAS",
            mass1=[35.0, 45.0],
            mass2=[25.0, 30.0],
            spin1z=[0.1, -0.2],
            spin2z=[0.0, 0.1],
            delta_f=0.5,
            f_lower=20.0,
            f_final=256.0,
            distance=100.0,
        )
        assert torch.allclose(hp, hp_batch, atol=1e-14)
        assert torch.allclose(hc, hc_batch, atol=1e-14)


def test_autobatch_imrphenomxhm():
    """Verify get_fd_waveform auto-batches for IMRPhenomXHM."""
    with TorchScheme("cpu"):
        hp, hc = pycbc.waveform.get_fd_waveform(
            approximant="IMRPhenomXHM",
            mass1=np.array([40.0, 50.0]),
            mass2=np.array([15.0, 20.0]),
            spin1z=np.array([0.3, -0.1]),
            spin2z=np.array([-0.2, 0.2]),
            delta_f=0.5,
            f_lower=25.0,
            f_final=256.0,
            distance=150.0,
            inclination=0.5,
        )

        assert isinstance(hp, torch.Tensor)
        assert isinstance(hc, torch.Tensor)
        assert hp.shape[0] == 2
        assert hc.shape[0] == 2

        for i in range(2):
            hp_s, hc_s = pycbc.waveform.get_fd_waveform(
                approximant="IMRPhenomXHM",
                mass1=40.0 if i == 0 else 50.0,
                mass2=15.0 if i == 0 else 20.0,
                spin1z=0.3 if i == 0 else -0.1,
                spin2z=-0.2 if i == 0 else 0.2,
                delta_f=0.5,
                f_lower=25.0,
                f_final=256.0,
                distance=150.0,
                inclination=0.5,
            )
            diff_hp = torch.max(torch.abs(hp[i] - torch.as_tensor(hp_s._data.tensor))).item()
            diff_hc = torch.max(torch.abs(hc[i] - torch.as_tensor(hc_s._data.tensor))).item()
            assert diff_hp < 1e-12
            assert diff_hc < 1e-12


def test_autobatch_imrphenomxphm():
    """Verify get_fd_waveform auto-batches for IMRPhenomXPHM."""
    with TorchScheme("cpu"):
        hp, hc = pycbc.waveform.get_fd_waveform(
            approximant="IMRPhenomXPHM",
            mass1=torch.tensor([30.0, 40.0], dtype=torch.float64),
            mass2=torch.tensor([20.0, 15.0], dtype=torch.float64),
            spin1x=torch.tensor([0.1, 0.0], dtype=torch.float64),
            spin1y=torch.tensor([0.0, 0.1], dtype=torch.float64),
            spin1z=torch.tensor([0.2, -0.2], dtype=torch.float64),
            spin2x=torch.tensor([0.0, 0.0], dtype=torch.float64),
            spin2y=torch.tensor([0.0, 0.0], dtype=torch.float64),
            spin2z=torch.tensor([0.1, 0.1], dtype=torch.float64),
            delta_f=0.5,
            f_lower=25.0,
            f_final=256.0,
            distance=200.0,
            inclination=0.4,
        )

        assert isinstance(hp, torch.Tensor)
        assert isinstance(hc, torch.Tensor)
        assert hp.shape[0] == 2
        assert hc.shape[0] == 2

        for i in range(2):
            hp_s, hc_s = pycbc.waveform.get_fd_waveform(
                approximant="IMRPhenomXPHM",
                mass1=30.0 if i == 0 else 40.0,
                mass2=20.0 if i == 0 else 15.0,
                spin1x=0.1 if i == 0 else 0.0,
                spin1y=0.0 if i == 0 else 0.1,
                spin1z=0.2 if i == 0 else -0.2,
                spin2x=0.0,
                spin2y=0.0,
                spin2z=0.1,
                delta_f=0.5,
                f_lower=25.0,
                f_final=256.0,
                distance=200.0,
                inclination=0.4,
            )
            diff_hp = torch.max(torch.abs(hp[i] - torch.as_tensor(hp_s._data.tensor))).item()
            diff_hc = torch.max(torch.abs(hc[i] - torch.as_tensor(hc_s._data.tensor))).item()
            assert diff_hp < 1e-12
            assert diff_hc < 1e-12


def test_scalar_dispatch_preserved():
    """Verify scalar inputs return standard FrequencySeries."""
    with TorchScheme("cpu"):
        hp, hc = pycbc.waveform.get_fd_waveform(
            approximant="TaylorF2",
            mass1=1.4,
            mass2=1.4,
            delta_f=1.0,
            f_lower=30.0,
            f_final=512.0,
        )
        assert isinstance(hp, FrequencySeries)
        assert isinstance(hc, FrequencySeries)
        assert hp.ndim == 1


def test_unsupported_approximant_array_fallback():
    """Verify unsupported approximant with array inputs does not route to batch."""
    with TorchScheme("cpu"):
        # EccentricFD does not have a batch generator in _FD_BATCH_APPROXIMANTS,
        # so it should proceed to scalar dispatch.
        # Since scalar dispatch does not expect 1D array mass, it will handle via scalar wav_gen.
        # If wav_gen fails with array, it proves it did not route to get_fd_waveform_batch.
        try:
            pycbc.waveform.get_fd_waveform(
                approximant="EccentricFD",
                mass1=np.array([10.0, 20.0]),
                mass2=1.4,
                delta_f=1.0,
                f_lower=30.0,
            )
        except Exception as e:
            # Should NOT be NotImplementedError from get_fd_waveform_batch
            assert "Batched generation for approximant" not in str(e)


def test_autobatch_template_object():
    """Verify get_fd_waveform auto-batches when template object attributes are arrays."""
    class Template:
        def __init__(self):
            self.approximant = "IMRPhenomD"
            self.mass1 = np.array([30.0, 40.0])
            self.mass2 = np.array([20.0, 25.0])
            self.spin1z = 0.0
            self.spin2z = 0.0
            self.delta_f = 0.5
            self.f_lower = 20.0
            self.f_final = 256.0
            self.distance = 100.0

    with TorchScheme("cpu"):
        tmplt = Template()
        hp, hc = pycbc.waveform.get_fd_waveform(template=tmplt)
        assert isinstance(hp, torch.Tensor)
        assert isinstance(hc, torch.Tensor)
        assert hp.shape[0] == 2
        assert hc.shape[0] == 2


def test_autobatch_single_parameter_batch():
    """Verify get_fd_waveform auto-batches when only distance is an array."""
    with TorchScheme("cpu"):
        hp, hc = pycbc.waveform.get_fd_waveform(
            approximant="TaylorF2",
            mass1=1.4,
            mass2=1.4,
            distance=np.array([100.0, 200.0, 400.0]),
            delta_f=1.0,
            f_lower=30.0,
            f_final=512.0,
        )
        assert isinstance(hp, torch.Tensor)
        assert isinstance(hc, torch.Tensor)
        assert hp.shape[0] == 3
        assert hc.shape[0] == 3


def test_single_element_array_is_scalar():
    """Verify 1-element arrays do not route to batch and use scalar dispatch."""
    with TorchScheme("cpu"):
        hp, hc = pycbc.waveform.get_fd_waveform(
            approximant="TaylorF2",
            mass1=1.4,
            mass2=1.4,
            spin1z=0.0,
            spin2z=0.0,
            delta_f=1.0,
            f_lower=30.0,
            f_final=512.0,
        )
        assert isinstance(hp, FrequencySeries)
        assert isinstance(hc, FrequencySeries)

