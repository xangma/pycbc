###################################################
Generating Noise
###################################################

=====================================
Generating time domain Gaussian noise
=====================================

When :class:`pycbc.scheme.TorchScheme` is active and the input PSD is
Torch-backed, both frequency- and time-domain Gaussian noise generation run
without the ``lal`` or ``lalsimulation`` Python modules. Together with a
Torch-native analytical PSD, this
allows :func:`pycbc.noise.noise_from_string` to remain on the selected Torch
CPU or CUDA device.  The reproducible coloring path in
``pycbc.noise.reproduceable`` has the same independence when used under a
Torch scheme.  The legacy NumPy/CPU noise path continues to use
``lalsimulation`` and reports a dependency error when it is unavailable.

.. plot:: ../examples/noise/timeseries.py
   :include-source:
