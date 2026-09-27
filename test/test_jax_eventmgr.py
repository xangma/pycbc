"""Parity tests for the device-resident inspiral event manager."""

from types import SimpleNamespace
import h5py

import numpy as np
import jax.numpy as jnp
from pycbc import scheme

from pycbc.events.eventmgr import findchirp_cluster_over_window
from pycbc.events.eventmgr_jax import (
    JAXEventManager,
    findchirp_cluster_over_window_jax,
)
from pycbc.events.eventmgr import EventManager


def _options(**values):
    defaults = dict(chisq_bins=True, chisq_threshold=0, chisq_delta=0,
                    newsnr_threshold=0, keep_loudest_interval=0,
                    keep_loudest_num=1, keep_loudest_stat="snr",
                    keep_loudest_log_chirp_window=None, injection_window=0,
                    sample_rate=1)
    defaults.update(values)
    return SimpleNamespace(**defaults)


def test_findchirp_float_times_and_strict_window_match_cpu():
    times = np.array([0.1, 1.1, 2.1, 4.2])
    values = np.array([1., 4., 2., 3.])
    expected = findchirp_cluster_over_window(times, values, 1)
    actual = findchirp_cluster_over_window_jax(
        jnp.asarray(times), jnp.asarray(values), 1)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_active_jax_scheme_dispatches_host_inputs_to_jax():
    with scheme.JAXScheme():
        result = findchirp_cluster_over_window(
            np.array([0., 2.]), np.array([1., 2.]), 1)
    assert type(result).__module__.startswith(("jax", "jaxlib"))


def test_jax_event_lifecycle_template_ids_and_thresholds():
    manager = JAXEventManager(
        _options(newsnr_threshold=4),
        ["time_index", "snr", "chisq", "chisq_dof"],
        [float, complex, float, float],
    )
    manager.new_template()
    manager.add_template_events(
        ["time_index", "snr", "chisq", "chisq_dof"],
        [jnp.array([1., 2.]), jnp.array([3.+0j, 8.+0j]),
         jnp.ones(2), jnp.ones(2)],
    )
    manager.finalize_template_events()
    manager.consolidate_events(manager.opt)
    np.testing.assert_array_equal(np.asarray(manager.events["template_id"]), [0])
    np.testing.assert_allclose(np.asarray(manager.events["snr"]), [8.+0j])


def test_jax_manager_clusters_exactly_and_batches_global_appends():
    manager = JAXEventManager(
        _options(), ["time_index", "snr"], [int, complex])
    expected_times = []
    for template_id in range(3):
        times = np.array([0, 1, 2, 5, 6], dtype=np.int32)
        snr = np.array([1, 4, 2, 3, 5], dtype=np.complex64)
        expected = findchirp_cluster_over_window(times, snr, 2)
        expected_times.extend(times[expected])
        manager.new_template()
        manager.add_template_events(
            ["time_index", "snr"], [jnp.asarray(times), jnp.asarray(snr)])
        manager.cluster_template_events("time_index", "snr", 2)
        manager.finalize_template_events()

    assert len(manager._pending_event_chunks) == 3
    assert manager._events["template_id"].size == 0
    np.testing.assert_array_equal(
        np.asarray(manager.events["time_index"]), expected_times)
    assert not manager._pending_event_chunks


def test_jax_host_snapshot_preserves_columns():
    manager = JAXEventManager(
        _options(), ["time_index", "snr"], [float, complex])
    manager.new_template()
    manager.add_template_events(
        ["time_index", "snr"], [jnp.array([2.]), jnp.array([4.+1j])])
    manager.finalize_template_events()
    snapshot = manager._host_structured_events()
    assert snapshot.dtype.names == ("template_id", "time_index", "snr")
    assert snapshot["template_id"].tolist() == [0]
    np.testing.assert_allclose(snapshot["snr"], [4.+1j])


def _hdf_options():
    return SimpleNamespace(
        channel_name="H1:STRAIN", sample_rate=4, gps_start_time=100,
        gps_end_time=110, trig_start_time=None, trig_end_time=None,
        segment_start_pad=0, segment_end_pad=0, autochi_number_points=2,
        autochi_onesided=None, autochi_two_phase=False,
        autochi_max_valued_dof=None, psdvar_segment=None,
    )


def _populate(manager):
    manager.new_template(tmplt=SimpleNamespace(
        template_hash=b"template-0", template_duration=1.5,
        sigmasq=4.0))
    values = {"time_index": jnp.array([3.]), "snr": jnp.array([4.+1j]),
              "chisq": jnp.array([1.]), "chisq_dof": jnp.array([4.]),
              "bank_chisq": jnp.array([2.]), "bank_chisq_dof": jnp.array([3.]),
              "cont_chisq": jnp.array([1.]), "sigmasq": jnp.array([4.])}
    manager_names = [name for name, _ in manager.event_dtype]
    names = [name for name in values if name in manager_names]
    manager.add_template_events(names, [values[name] for name in names])
    manager.finalize_template_events()


def test_jax_write_events_matches_cpu(tmp_path):
    opt = _hdf_options()
    columns = ["time_index", "snr", "chisq", "chisq_dof", "bank_chisq",
               "bank_chisq_dof", "cont_chisq", "sigmasq"]
    types = [int, complex, float, int, float, int, float, float]
    cpu = EventManager(opt, columns, types)
    jax_manager = JAXEventManager(opt, columns, types)
    _populate(cpu)
    _populate(jax_manager)
    cpu_path, jax_path = tmp_path / "cpu.hdf", tmp_path / "jax.hdf"
    cpu.write_events(str(cpu_path))
    jax_manager.write_events(str(jax_path))
    with h5py.File(cpu_path, "r") as left, h5py.File(jax_path, "r") as right:
        datasets = []
        left.visititems(lambda name, obj: datasets.append(name)
                        if isinstance(obj, h5py.Dataset) else None)
        right_datasets = []
        right.visititems(lambda name, obj: right_datasets.append(name)
                         if isinstance(obj, h5py.Dataset) else None)
        assert set(datasets) == set(right_datasets)
        for name in datasets:
            left_data, right_data = left[name][:], right[name][:]
            assert left_data.dtype == right_data.dtype
            if left_data.dtype.kind in "SUO":
                np.testing.assert_array_equal(left_data, right_data)
            else:
                np.testing.assert_allclose(left_data, right_data)


def test_jax_hdf_parity_two_templates_empty_gating_and_performance(tmp_path):
    opt = _hdf_options()
    opt = SimpleNamespace(**vars(opt),)
    columns = ["time_index", "snr", "chisq", "chisq_dof", "bank_chisq",
               "bank_chisq_dof", "cont_chisq", "sigmasq", "sg_chisq"]
    types = [int, complex, float, int, float, int, float, float, float]
    kwargs = dict(gating_info={"file": [], "auto": []})
    cpu = EventManager(opt, columns, types, **kwargs)
    jax_manager = JAXEventManager(opt, columns, types, **kwargs)
    for manager in (cpu, jax_manager):
        manager.new_template(tmplt=SimpleNamespace(
            template_hash=b"a", template_duration=1., sigmasq=2.))
        manager.add_template_events(
                columns,
                [jnp.array([9, 2]), jnp.array([2.+0j, 8.+0j]),
                     jnp.ones(2), jnp.ones(2), jnp.ones(2), jnp.ones(2),
                     jnp.ones(2), jnp.ones(2), jnp.ones(2)])
        manager.finalize_template_events()
        manager.new_template(tmplt=SimpleNamespace(
            template_hash=b"b", template_duration=2., sigmasq=3.))
        manager.add_template_events(
                columns,
                [jnp.array([1]), jnp.array([5.+0j]), jnp.ones(1),
                     jnp.ones(1), jnp.ones(1), jnp.ones(1), jnp.ones(1),
                     jnp.ones(1), jnp.ones(1)])
        manager.finalize_template_events()
        manager.save_performance(1, 2, 3, 4., 1.)
    left_path, right_path = tmp_path / "cpu2.hdf", tmp_path / "jax2.hdf"
    cpu.write_events(str(left_path))
    jax_manager.write_events(str(right_path))
    with h5py.File(left_path) as left, h5py.File(right_path) as right:
        ld, rd = [], []
        left.visititems(lambda n, o: ld.append(n) if isinstance(o, h5py.Dataset) else None)
        right.visititems(lambda n, o: rd.append(n) if isinstance(o, h5py.Dataset) else None)
        assert set(ld) == set(rd)
        for name in ld:
            assert left[name].dtype == right[name].dtype
            if left[name].dtype.kind in "SUO":
                np.testing.assert_array_equal(left[name][:], right[name][:])
            else:
                np.testing.assert_allclose(left[name][:], right[name][:])


def test_jax_hdf_parity_precessing_columns(tmp_path):
    opt = _hdf_options()
    columns = ["time_index", "snr", "chisq", "bank_chisq",
               "bank_chisq_dof", "cont_chisq", "u_vals", "coa_phase",
               "hplus_cross_corr"]
    types = [int, complex, float, float, int, float, complex, float, complex]
    kwargs = dict(gating_info={"file": []})
    cpu = EventManager(opt, columns, types, **kwargs)
    jax_manager = JAXEventManager(opt, columns, types, **kwargs)
    values = [jnp.array([2]), jnp.array([3. + 4j]), jnp.array([1.]),
              jnp.array([2.]), jnp.array([3]), jnp.array([1.]),
              jnp.array([1. + 2j]), jnp.array([.25]),
              jnp.array([4. + 5j])]
    for manager in (cpu, jax_manager):
        manager.new_template(tmplt=SimpleNamespace(
            template_hash=b"p", template_duration=1.),
            sigmasq_plus=7., sigmasq_cross=5.)
        manager.add_template_events(columns, values)
        manager.finalize_template_events()
    left_path, right_path = tmp_path / "cpu-pre.hdf", tmp_path / "jax-pre.hdf"
    cpu.write_events(str(left_path))
    jax_manager.write_events(str(right_path))
    with h5py.File(left_path) as left, h5py.File(right_path) as right:
        names = []
        left.visititems(lambda n, o: names.append(n)
                        if isinstance(o, h5py.Dataset) else None)
        right_names = []
        right.visititems(lambda n, o: right_names.append(n)
                         if isinstance(o, h5py.Dataset) else None)
        assert set(names) == set(right_names)
        for name in names:
            assert left[name].dtype == right[name].dtype
            if left[name].dtype.kind in "SUO":
                np.testing.assert_array_equal(left[name][:], right[name][:])
            else:
                np.testing.assert_allclose(left[name][:], right[name][:])


def test_jax_hdf_parity_empty_events_performance_and_gates(tmp_path):
    opt = _hdf_options()
    columns = ["time_index", "snr", "chisq", "chisq_dof"]
    types = [int, complex, float, int]
    kwargs = {"gating_info": {"file": [], "auto": []}}
    cpu = EventManager(opt, columns, types, **kwargs)
    jax_manager = JAXEventManager(opt, columns, types, **kwargs)
    for manager in (cpu, jax_manager):
        manager.save_performance(1, 2, 3, 4., 1.)
    left_path = tmp_path / "cpu-empty.hdf"
    right_path = tmp_path / "jax-empty.hdf"
    cpu.write_events(str(left_path))
    jax_manager.write_events(str(right_path))
    with h5py.File(left_path) as left, h5py.File(right_path) as right:
        left_names = []
        right_names = []
        left.visititems(lambda name, obj: left_names.append(name)
                        if isinstance(obj, h5py.Dataset) else None)
        right.visititems(lambda name, obj: right_names.append(name)
                         if isinstance(obj, h5py.Dataset) else None)
        assert left_names == right_names
        for name in left_names:
            assert left[name].dtype == right[name].dtype
            np.testing.assert_array_equal(left[name][:], right[name][:])


def test_jax_checkpoint_roundtrip(tmp_path):
    opt = _hdf_options()
    columns = ["time_index", "snr", "chisq", "chisq_dof"]
    manager = JAXEventManager(opt, columns, [int, complex, float, int])
    _populate(manager)
    path = tmp_path / "state.hdf"
    manager.save_state(7, str(path))
    next_template, restored = JAXEventManager.restore_state(str(path))
    assert next_template == 8
    for name in manager.events:
        np.testing.assert_allclose(np.asarray(manager.events[name]),
                                   np.asarray(restored.events[name]))
    restored.new_template()
    restored.add_template_events(
        columns, [jnp.array([3]), jnp.array([5. + 0j]),
                  jnp.array([1.]), jnp.array([2])])
    assert int(np.asarray(restored.template_events["template_id"])[0]) == 1
