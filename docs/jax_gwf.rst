.. _jax-gwf:

JAX GWF replay
==============

``GWFReplaySource`` reads sample-aligned spans from a supported subset of
local version-8/9 GWF files into JAX ``TimeSeries`` objects. The bounded
``JAXReplayFrameReader`` uses this source when the layout is supported and
otherwise uses the existing frame reader. Ordinary CPU frame reading retains
its original implementation.

Device and decompression
------------------------

Select the device with ``JAXScheme``. Raw and zero-suppressed numeric payloads
are reconstructed on that device. On a CUDA device, zlib payloads prefer the
optional ``cuda-zlib`` backend by default. A CPU device, an unavailable CUDA
backend or an RFC 1952 gzip wrapper uses bounded host decompression followed
by transfer to the selected device. File I/O, container parsing and GWF CRC
validation run on the host.

.. code-block:: python

   from pycbc import scheme
   from pycbc.frame.gwf_replay_jax import GWFReplaySource

   with scheme.JAXScheme("cuda:0"):
       source = GWFReplaySource("H-STRAIN-1000000000-16.gwf", "H1:STRAIN", 4096)
       strain = source.read(1000000004, 4)

For an explicit host-decompression comparison, pass ``cuda_codec="host"`` to
``GWFReplaySource`` or set ``PYCBC_GWF_CUDA_DECOMPRESS=host``. The default value
is ``deflate``. This option controls compressed-vector decompression; raw and
zero-suppressed payloads continue to use JAX. Decoding expands the selected
vector before slicing the requested interval, so memory use depends on the
vector size as well as the returned span.

Optional CUDA codec
-------------------

The separately maintained
`cuda-zlib package <https://github.com/xangma/cuda-zlib>`__ provides the native
JAX FFI codec. Version ``0.1.0a1`` requires Python 3.12 or later, Linux, an
NVIDIA GPU, JAX/JAXlib 0.11.2 or later, and a compatible CUDA toolkit 12 or
later containing ``nvcc``. Install from source; there is no prebuilt native
wheel or PyPI release:

.. code-block:: console

   python -m pip install \
       'cuda-zlib[cuda12] @ git+https://github.com/xangma/cuda-zlib.git'

The ``cuda12`` extra installs JAX's CUDA runtime; install the toolkit compiler
separately. With an existing compatible GPU JAX installation, omit the extra.
The native library builds on first use and is cached. Codec calls return
independently owned JAX buffers and validate the stream before returning.
The codec limits input and decoded output to 256 MiB each. See the package
guide for workspace limits, compiler and cache configuration.

Supported inputs and errors
---------------------------

Direct replay requires local ``*-GPS-DURATION.gwf`` files with contiguous,
non-overlapping coverage of the requested span. The admitted layout requires:

* One ``FrameH`` per file, with integer GPS time and duration matching the
  filename; residual GPS nanoseconds must be zero.
* One matching channel linked through ``FrameH`` to ``FrRawData`` /
  ``FrAdcData``, or to a time-series ``FrProcData`` (type 1). The parent channel
  identifies the data; the ``FrVect`` name need not match the channel name.
* Zero channel time offset and vector origin, one unlinked one-dimensional
  vector covering the entire frame, and sample spacing matching the stream.
  ADC channels also require zero validity flags and a matching sample rate.

The vector must use a dtype supported by ``TimeSeries``:
float32, float64, complex64, complex128, int32, uint32 or the platform integer
dtype. Exact 64-bit element decoding requires JAX X64, enabled by default in
``JAXScheme``.

Raw and zero-suppressed vectors are decoded with JAX. Ordinary zlib and gzip
vectors are supported through the decompression paths above. Differential
gzip, Zstandard, differential Zstandard, strings, validity metadata and other
unproved layouts require the compatibility reader. Cache/LCF sources,
unsupported format versions, scalar layouts and checksum schemes also use
that reader. A nonzero version-9 ``FrEndOfFile.chkSumTOC`` requires the
compatibility reader because the independent TOC checksum is not implemented.

``GWFReplaySource`` reports these compatibility cases as
``GWFReplayUnsupported``; ``JAXReplayFrameReader`` catches that exception and
uses the original reader. Malformed admitted structures, truncation,
supported CRC mismatches, invalid compressed streams and codec capacity
errors propagate instead of silently changing readers. Preset dictionaries
raise an error. Unexpected CUDA/XLA runtime errors also propagate.

Exact sample preservation
-------------------------

The :download:`GWF comparison notebook <../examples/jax/jax_gwf.ipynb>` generates
small temporary frames with the public writer and compares direct replay with
``read_frame(check_integrity=True)`` under ``CPUScheme``. It asserts equal
shape, dtype, epoch, sample interval and bytes for full spans, fractional
sample-aligned starts and reads across contiguous files. Float32 and float64
examples include signed zeros, subnormals and quiet NaN payloads.

Raw decoding assembles integer words and bitcasts them to their sample dtype.
Zero suppression reconstructs integer differences with modular integer
addition before the same bitcast. These operations introduce no floating-point
approximation; the notebook also proves both payload paths independently.
Lossless frame decoding does not establish equivalence of later conditioning
or filtering calculations.

Format references are the
`version-8 GWF specification <https://dcc.ligo.org/public/0000/T970130/002/T970130-v2.pdf>`__,
`version-9 GWF specification <https://dcc.ligo.org/public/0000/T970130/004/T970130-v4.pdf>`__,
`RFC 1950 <https://www.rfc-editor.org/rfc/rfc1950.html>`__ and
`RFC 1952 <https://www.rfc-editor.org/rfc/rfc1952.html>`__.
