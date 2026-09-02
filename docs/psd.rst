###################################################
Handling PSDs
###################################################

=====================================
Reading / Saving a PSD from a file
=====================================

If you have a PSD in a two column, space separated, (frequency strain), you can
load this into PyCBC.

.. plot:: ../examples/psd/read.py
   :include-source:

.. _Analytic PSDs from lalsimulation:

==============================================
Generating an Analytic PSD from lalsimulation
==============================================

A certain number of PSDs are built into lalsimulation, which you'll be able
to access through PyCBC. Below we show how to see which ones are available, 
and demonstrate how to generate one.

.. plot:: ../examples/psd/analytic.py
   :include-source:

The PSDs from lalsimulation are computed at the required frequencies by
interpolating a fixed set of samples; if the required frequencies fall
outside of the range of the known samples no warnings will be raised,
and (meaningless) extrapolated values will be returned.

Therefore, users should check the validity range of the PSD they are 
using within lalsimulation, lest they get incorrect results.

========================================
Built-in analytical PSDs under Torch
========================================

When :class:`pycbc.scheme.TorchScheme` is active, ``flat_unity`` is generated
directly on the selected Torch device.  The direct LISA, TianQin, and Taiji
TDI ``XYZ``, ``AE``, and ``T`` analytical models are also evaluated and
interpolated on-device for TDI 1.5 and 2.0.  The analytical SciRD LISA,
TianQin, and Taiji sensitivity curves, their standalone Galactic-confusion
fits, and the corresponding combined instrument-plus-confusion curves are
likewise evaluated on-device.  For LISA, this includes both SciRD and
semi-analytical response curves, response-transformed ``XYZ`` PSDs,
confusion-only ``XYZ`` PSDs, and combined ``AE`` PSDs.  The small tabulated
LISA response is downloaded and cached on the host, then copied once to the
selected device; interpolation and PSD assembly remain in Torch.  TianQin and
Taiji confusion-only ``XYZ`` PSDs and combined ``AE`` PSDs are also
Torch-native.  These space-detector models use ``float64`` and therefore
support Torch CPU and CUDA devices; MPS is not supported because its
``float32`` range underflows the physical PSD values.

The analytical advanced-LIGO quantum and thermal ground-detector family is
evaluated directly on Torch CPU and CUDA devices.  This includes the quantum
and combined ``NoSRMLowPower``, ``NoSRMHighPower``, ``ZeroDetLowPower``,
``ZeroDetHighPower``, ``NSNSOpt``, ``BHBH20Deg``, and ``HighFrequency``
configurations, plus ``aLIGOThermal``.  These ports preserve LALSimulation's
DC, Nyquist, and cutoff layout and use the LALSuite 7.26.1 analytical constants
as their reference baseline.  PyCBC maintains a static registry for the
Torch-native ground-detector models, so they remain discoverable and usable
under Torch CPU or CUDA even when the ``lalsimulation`` Python module is not
installed.  Outside a supported Torch-native path, these models retain their
LALSimulation implementation and require that dependency.  Versioned table
models also require their data files to remain resolvable. With core LAL
installed, ``lal.FileResolvePath`` remains authoritative; without it, the
Torch path searches the current directory, ``LAL_DATA_PATH``, and tables
shipped in the ``lalapps`` package.

The initial- and enhanced-LIGO family ``iLIGOSRD``, ``iLIGOSeismic``,
``iLIGOThermal``, ``iLIGOShot``, ``eLIGOShot``, ``iLIGOModel``, and
``eLIGOModel`` is also evaluated directly on Torch CPU and CUDA devices.  The
native expressions preserve LALSimulation's component composition, seismic
branch, discovery rules, and output layout, using LALSuite 7.26.1 as the
reference baseline.

The data-free LALSimulation detector fits ``Virgo``, ``GEO``, ``GEOHF``,
``TAMA``, ``KAGRA``, and ``AdvVirgo`` are likewise evaluated directly on
Torch CPU and CUDA devices.  Their native expressions use the LALSuite 7.26.1
fits as the reference baseline and preserve the same discovery and output
layout rules.

The 52 installed LALSimulation wrappers backed by versioned detector tables
from ``LIGO-T0900288``, ``LIGO-P1200087``, ``LIGO-P1600143``,
``LIGO-T1600593``, ``LIGO-T1800042``, ``LIGO-T1800044``, and
``LIGO-T1800545`` are also supported on Torch CPU and CUDA devices.  These
include the observing-scenario, design, GWINC, Cosmic Explorer, Einstein
Telescope, Virgo, and KAGRA curves, together with LALSimulation's deprecated
``T1800044`` and ``T1800545`` aliases.  Their versioned data files are
resolved and parsed on the host on each call. Core LAL's resolver remains
authoritative when installed; otherwise the current directory,
``LAL_DATA_PATH``, and ``lalapps`` package data are searched. The samples are then
copied to the selected device for LAL-compatible log-ASD interpolation,
extrapolation, cutoff handling, and output construction.
Only finite, positive, strictly increasing exact two-column tables take this
native path; noncanonical resolver overrides retain LALSimulation's file
reader.

PSD text and XML files are necessarily parsed on the host.  Under
``TorchScheme``, their tabulated samples are then copied once to the selected
Torch CPU or CUDA device, where log interpolation, cutoff zeroing, and output
construction are performed.  Torch MPS is not supported for these readers
because the existing reader contract returns ``float64`` PSDs.  Models
exposed by an unrecognized LALSimulation version continue to use their
existing LAL implementation.

====================================
Estimating the PSD of a time series
====================================

.. plot:: ../examples/psd/estimate.py
   :include-source:
