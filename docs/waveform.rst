###################################################
Waveforms
###################################################

=====================================
What waveforms can I generate?
=====================================

.. literalinclude:: ../examples/waveform/what_waveform.py
.. command-output:: python ../examples/waveform/what_waveform.py


=====================================
Plotting Time Domain Waveforms 
=====================================

.. plot:: ../examples/waveform/plot_waveform.py
   :include-source:

==============================================
Generating one waveform in multiple detectors
==============================================

.. plot:: ../examples/waveform/plot_detwaveform.py
   :include-source:


===============================================
Selecting which modes to include
===============================================
Gravitational waves can be decomposed into a set
of modes. Some approximants only calculate the dominant
l=2, m=2 mode, while others included higher-order modes. These
often, but not always, include 'HM' in the name. The modes
present in the output polarizations can be selected for these waveforms
as demonstrated below. By default, all modes that the waveform model
supports are typically generated.

.. plot:: ../examples/waveform/higher_modes.py
   :include-source:

=======================================
Calculating the match between waveforms
=======================================

.. literalinclude:: ../examples/waveform/match_waveform.py
.. command-output:: python ../examples/waveform/match_waveform.py

================================================
Plotting a TD and FD waveform together in the TD
================================================

.. plot:: ../examples/waveform/plot_fd_td.py
   :include-source:
   
================================================
Plotting GW phase and amplitude of TD waveform
================================================

.. plot:: ../examples/waveform/plot_phase.py
   :include-source:

================================================
Plotting frequency evolution of TD waveform
================================================

.. plot:: ../examples/waveform/plot_freq.py
   :include-source:
   
=====================================
Adding a custom waveform
=====================================

You can also add your own custom waveform and make it available through 
the waveform interface standard. You can directly call the code as below
or if you include it in a python package, :ref:`PyCBC can directly detect it! <waveform_plugin>`

.. plot:: ../examples/waveform/add_waveform.py
   :include-source:
   

===========================================
Torch-native waveform ports (torch scheme)
===========================================
PyCBC currently registers Torch-native frequency-domain and arbitrary-frequency
sequence interfaces for five TaylorF2-family approximants.

The registered approximants are ``TaylorF2``, ``TaylorF2NLTides``,
``TaylorF2RedSpin``, ``TaylorF2RedSpinTidal``, and ``TaylorF2Ecc``.

Regular-grid rows marked ``LAL reference`` are compared with the existing LAL
implementation. Sequence rows marked ``native extension`` have no equivalent
LAL public interface. Those extensions accept their documented arbitrary-
frequency contract and are validated against analytic or regular-grid behavior
rather than a nonexistent LAL sequence result.

The global ``PYCBC_TORCH_NATIVE_PORTS`` switch and the per-approximant component
flags can override native selection. A per-component setting takes precedence.
Setting a component flag to ``0`` forces the established fallback where one
exists; it does not create a fallback for a native-only sequence interface.

Under ``TorchScheme``, supported regular-grid and sequence results use Torch-
backed PyCBC series on the selected device. Scalar coefficient preparation and
public Python control flow do not imply end-to-end autograd. Device-specific
predicates remain authoritative, including accuracy guards for devices that do
not provide the required double-precision kernels.

TaylorF2 also provides an explicit native batch interface. It is deliberately
separate from the scalar dispatcher, so vector inputs cannot change
``get_fd_waveform`` return types::

    with pycbc.scheme.TorchScheme("cpu"):
        batch = pycbc.waveform.get_fd_waveform_batch(
            "TaylorF2",
            mass1=[1.4, 1.5], mass2=1.3,
            f_lower=[20.0, 24.0], f_final=128.0, delta_f=1.0,
        )

``batch.hplus`` and ``batch.hcross`` are padded two-dimensional Torch tensors.
``batch.first_bins`` and ``batch.end_bins`` give each row's exact non-zero
frequency support; ``batch.delta_f`` and ``batch.epoch`` describe the common
grid. Scalars and length-one vectors broadcast to the common batch size, while
inconsistent vector lengths are rejected. Focused validation is provided by
``test/waveform/test_taylorf2_batch.py``.
