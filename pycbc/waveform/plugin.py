""" Utilities for handling waveform plugins
"""

from importlib.metadata import entry_points


def add_custom_waveform(approximant, function, domain,
                        sequence=False, has_det_response=False,
                        force=False,):
    """ Make custom waveform available to pycbc

    Parameters
    ----------
    approximant : str
        The name of the waveform
    function : function
        The function to generate the waveform
    domain : str
        Either 'frequency' or 'time' to indicate the domain of the waveform.
    sequence : bool, False
        Function evaluates waveform at only chosen points (instead of a
        equal-spaced grid).
    has_det_response : bool, False
        Check if waveform generator has built-in detector response.
    """
    from pycbc.waveform.waveform import (cpu_fd, cpu_td, fd_sequence,
                                         fd_det, fd_det_sequence,
                                         td_fd_waveform_transform)

    used = RuntimeError("Can't load plugin waveform {}, the name is"
                        " already in use.".format(approximant))

    if domain == 'time':
        if not force and (approximant in cpu_td):
            raise used
        cpu_td[approximant] = function
        td_fd_waveform_transform(approximant)
    elif domain == 'frequency':
        if sequence:
            if not has_det_response:
                if not force and (approximant in fd_sequence):
                    raise used
                fd_sequence[approximant] = function
            else:
                if not force and (approximant in fd_det_sequence):
                    raise used
                fd_det_sequence[approximant] = function
        else:
            if not has_det_response:
                if not force and (approximant in cpu_fd):
                    raise used
                cpu_fd[approximant] = function
            else:
                if not force and (approximant in fd_det):
                    raise used
                fd_det[approximant] = function
    else:
        raise ValueError("Invalid domain ({}), should be "
                         "'time' or 'frequency'".format(domain))


def add_length_estimator(approximant, function):
    """ Add length estimator for an approximant

    Parameters
    ----------
    approximant : str
        Name of approximant
    function : function
        A function which takes kwargs and returns the waveform length
    """
    from pycbc.waveform.waveform import _filter_time_lengths
    if approximant in _filter_time_lengths:
        raise RuntimeError("Can't load length estimator {}, the name is"
                           " already in use.".format(approximant))
    _filter_time_lengths[approximant] = function

    from pycbc.waveform.waveform import td_fd_waveform_transform
    td_fd_waveform_transform(approximant)


def add_end_frequency_estimator(approximant, function):
    """ Add end frequency estimator for an approximant

    Parameters
    ----------
    approximant : str
        Name of approximant
    function : function
        A function which takes kwargs and returns the waveform end frequency
    """
    from pycbc.waveform.waveform import _filter_ends
    if approximant in _filter_ends:
        raise RuntimeError("Can't load freqeuncy estimator {}, the name is"
                           " already in use.".format(approximant))

    _filter_ends[approximant] = function


class _LazyPlugin:
    """Wrapper that defers plugin.load() until called or inspected."""

    def __init__(self, plugin):
        self._plugin = plugin
        self._loaded = None

    def _get_target(self):
        if self._loaded is None:
            self._loaded = self._plugin.load()
        return self._loaded

    def __call__(self, *args, **kwargs):
        return self._get_target()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._get_target(), name)


def retrieve_waveform_plugins():
    """ Process external waveform plugins
    """

    # Check for fd waveforms (no detector response)
    for plugin in entry_points(group='pycbc.waveform.fd'):
        add_custom_waveform(plugin.name, _LazyPlugin(plugin), 'frequency')

    # Check for fd waveforms (has detector response)
    for plugin in entry_points(group='pycbc.waveform.fd_det'):
        add_custom_waveform(plugin.name, _LazyPlugin(plugin), 'frequency',
                            has_det_response=True)

    # Check for fd sequence waveforms (no detector response)
    for plugin in entry_points(group='pycbc.waveform.fd_sequence'):
        add_custom_waveform(plugin.name, _LazyPlugin(plugin), 'frequency',
                            sequence=True)

    # Check for fd sequence waveforms (has detector response)
    for plugin in entry_points(group='pycbc.waveform.fd_det_sequence'):
        add_custom_waveform(plugin.name, _LazyPlugin(plugin), 'frequency',
                            sequence=True, has_det_response=True)

    # Check for td waveforms
    for plugin in entry_points(group='pycbc.waveform.td'):
        add_custom_waveform(plugin.name, _LazyPlugin(plugin), 'time')

    # Check for waveform length estimates
    for plugin in entry_points(group='pycbc.waveform.length'):
        add_length_estimator(plugin.name, _LazyPlugin(plugin))

    # Check for waveform end frequency estimates
    for plugin in entry_points(group='pycbc.waveform.end_freq'):
        add_end_frequency_estimator(plugin.name, _LazyPlugin(plugin))
