JAX ground-detector geometry
============================

``Detector`` accepts raw JAX arrays for tensor antenna patterns, geocentric
delays, arrival times, finite-arm response, and effective distance. Scalar and
NumPy calls retain their original implementation. ``NetworkGeometry`` shares
the polarization basis when its detectors have the same built-in methods and
clock settings; otherwise it calls each detector's public methods. It reads
current detector responses and locations rather than retaining stale copies.

The first raw JAX argument determines placement, including mixed-device
arguments. Geometry uses a common floating-point precision; scalar metadata
and physical constants can promote results. Use float64 for absolute GPS-time
arrays: float32 values near modern GPS epochs cannot retain subsecond spacing.
Widening an already rounded array cannot recover those times. JAX schemes
enable double precision by default.

For example::

    import jax.numpy as jnp
    from pycbc.detector import Detector, NetworkGeometry
    from pycbc.scheme import JAXScheme

    with JAXScheme("cpu"):
        detector = Detector("H1")
        times = jnp.array([1126259462.0, 1126259462.125], dtype=jnp.float64)
        ra = jnp.array([0.3, 0.7])
        plus, cross, delay = detector.antenna_pattern_and_time_delay(
            ra, -0.2, 0.1, times)
        network = NetworkGeometry(["H1", "L1", "V1"])
        responses = network.antenna_pattern_and_time_delay(ra, -0.2, 0.1, times)

Select ``"cuda"`` for an available CUDA device. Default computations support
JAX differentiation away from response singularities; original validation
controls require concrete inputs.
The JAX backend supports tensor polarizations and real floating-point angles.
A fixed Python frequency of zero selects the real static response. A traced
scalar frequency uses the complex finite-arm branch and must be nonzero; this
avoids changing the compiled result dtype with a runtime value.
Time grids require a GMST reference time for the built-in on-device clock.
With ``reference_time=None``, a scalar time can still use the original accurate
clock calculation; the GMST validation control also supports original array
clock calculations.

The combined method evaluates antenna factors and delay at its supplied
``t_gps``. It does not substitute a detector arrival time. Call
``Detector.arrival_time`` first when a model defines its response at arrival.

Static-response arrays retain the original dot/broadcast axes wherever that
native calculation is supported, including its unusual higher-dimensional
conventions. Additional JAX broadcast grids use the component axis. Original
validation for those additional shapes, and for frequency grids, replays the
original scalar method once per grid point. The notebook distinguishes these
API extensions from floating-point differences.

Independent original controls
-----------------------------

Select ``detector`` for all original geometry operations, or individual names
through ``JAXScheme(reference_operations=...)``:

* ``detector.Detector.gmst_estimate``
* ``detector.Detector.antenna_pattern``
* ``detector.Detector.time_delay_from_location``
* ``detector.single_arm_frequency_response``
* ``detector.Detector.effective_distance_scale``
* ``detector.Detector.effective_distance``
* ``detector.Detector.arrival_time``

Combined and network operations honor the antenna, delay, and GMST controls
independently. The distance-scale control runs the original scaling body with
the already computed antenna factors; selecting it together with the antenna
control restores the original complete effective distance. The existing
``divide`` control independently restores its final division. Validation
preserves original result dtype and bytes on the input device, requires enough
precision for those results, and propagates original errors.

.. toctree::
   :maxdepth: 1

   jax_detector_numerical_differences
