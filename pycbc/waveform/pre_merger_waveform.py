from functools import cache
from scipy import signal

import pycbc.fft
import pycbc.noise
import pycbc.strain
import pycbc.waveform


@cache
def get_window(window_length):
    if window_length:
        return signal.windows.hann(window_length * 2 + 1)[:window_length]
    else:
        return None


def apply_pre_merger_kernel(
    f_series,
    whitening_psd,
    window,
    window_length,
    nfz,
    nctf,
    uid,
    copy_output=False,
):
    """Helper function to apply the pre-merger kernel.
    
    Parameters
    ----------
    f_series : pycbc.types.FrequencySeries
        Frequency series to apply the kernel to.
    whitening_psd : pycbc.types.FrequencySeries
        PSD for whitening the data in the frequency-domain.
    window : numpy.ndarray
        Window array.
    window_length : int
        Pre-computed length of the window in samples.
    nefz : int
        Number of forward zeroes.
    nctf : int
        Number of samples to zero at the end of the data.
    uid : int
        UID for computing the iFFTs.

    Returns
    -------
    pycbc.types.TimeSeries
        Whitened time series.
    """
    # Whiten data
    f_series.data[:] = f_series.data[:] * (whitening_psd.data[:]).conj()

    # TD to FD to apply zeroes
    tout_ww = pycbc.strain.strain.execute_cached_ifft(
        f_series,
        copy_output=copy_output,
        uid=uid,
    )
    # Zero initial data
    tout_ww.data[:nfz] = 0
    if window is not None:
        # Apply window
        tout_ww.data[nfz:nfz+window_length] *= window
    # Zero data from cutoff
    tout_ww.data[-nctf:] = 0
    return tout_ww


def generate_data_lisa_pre_merger(
    waveform_params,
    psds_for_datagen,
    sample_rate,
    seed=137,
    zero_noise=False,
    no_signal=False,
    duration=None,
):
    """Generate pre-merger LISA data.

    UIDs used for FFTs: 4235(0), 4236(0)
    
    Parameters
    ----------
    waveform_params : dict
        Dictionary of waveform parameters
    psds_for_datagen : dict
        PSDs for data generation.
    sample_rate : float
        Sampling rate in Hz. 
    seed : int
        Random seed used for generating the noise.
    zero_noise : bool
        If true, the noise will be set to zero.
    no_signal : bool
        If true, the signal will not be added to data and only noise will
        be returned.
    duration : float, optional
        If specified, the waveform will be truncated to match the specified
        duration.

    Returns
    -------
    Dict[str: pycbc.types.TimeSeries]
        Dictionary containing the time-domain data for each channel.
    """
    # Generate injection
    outs = pycbc.waveform.get_fd_det_waveform(
        ifos=['LISA_A','LISA_E','LISA_T'],
        **waveform_params
    )

    # Shift waveform so the merger is not at the end of the data
    outs['LISA_A'] = outs['LISA_A'].cyclic_time_shift(-waveform_params['additional_end_data'])
    outs['LISA_E'] = outs['LISA_E'].cyclic_time_shift(-waveform_params['additional_end_data'])

    # FS waveform to TD
    tout_A = outs['LISA_A'].to_timeseries()
    tout_E = outs['LISA_E'].to_timeseries()

    # Generate TD noise from the original PSDs
    strain_w_A = pycbc.noise.noise_from_psd(
        len(tout_A),
        tout_A.delta_t,
        psds_for_datagen['LISA_A'],
        seed=seed,
    )
    strain_w_E = pycbc.noise.noise_from_psd(
        len(tout_E),
        tout_E.delta_t,
        psds_for_datagen['LISA_E'],
        seed=seed + 1,
    )

    # We need to make sure the noise times match the signal
    strain_w_A._epoch = tout_A._epoch
    strain_w_E._epoch = tout_E._epoch

    # If zero noise, set noise to zero
    if zero_noise:
        strain_w_A *= 0.0
        strain_w_E *= 0.0

    # Only add signal if no_signal=False
    if not no_signal:
        strain_w_A[:] += tout_A[:]
        strain_w_E[:] += tout_E[:]

    # If duration is specified, discard the extra data
    if duration is not None:
        if duration > tout_A.duration:
            raise RuntimeError(
                "Specified duration is longer than the generated waveform"
            )
        nkeep = int(duration * sample_rate)
        # New start time will be nkeep sample time
        new_epoch = strain_w_A.sample_times[-nkeep]
        strain_w_A = pycbc.types.TimeSeries(
            strain_w_A.data[-nkeep:],
            delta_t=strain_w_A.delta_t,
        )
        strain_w_E = pycbc.types.TimeSeries(
            strain_w_E.data[-nkeep:],
            delta_t=strain_w_E.delta_t,
        )
        # Set the start time so that the GPS time is still correct
        strain_w_A.start_time = new_epoch
        strain_w_E.start_time = new_epoch
    
    return {
        "LISA_A": strain_w_A,
        "LISA_E": strain_w_E,
    }


def pre_process_data_lisa_pre_merger(
    data,
    sample_rate,
    psds_for_whitening,
    window_length,
    cutoff_time,
    forward_zeroes=0,
):
    """Pre-process the pre-merger data.

    The data is truncated, windowed and whitened.

    data : dict
        Dictionary containing time-domain data. 
    sample_rate : float
        Sampling rate in Hz. 
    psds_for_whitening : dict
        PSDs for whitening.
    window_length : int
        Length of the hann window use to taper the start of the data.
    cutoff_time : float
        Time (in seconds) from the end of the waveform to cutoff.
    forward_zeroes : float
        Number of samples to set to zero at the start of the waveform. If used,
        the window will be applied starting after the zeroes.

    Returns
    -------
    Dict[str: pycbc.types.TimeSeries]
        Dictionary containing the time-domain data for each channel.
    """
    window = get_window(window_length)

    # Number of samples to zero
    nctf = int(cutoff_time * sample_rate)

    # Apply pre-merger kernel to both channels
    # Function needs frequency series
    strain_ww = {}
    strain_ww["LISA_A"] = apply_pre_merger_kernel(
        data["LISA_A"].to_frequencyseries(),
        whitening_psd=psds_for_whitening["LISA_A"],
        window=window,
        window_length=window_length,
        nfz=forward_zeroes,
        nctf=nctf,
        uid=4235,
        copy_output=True,
    )
    strain_ww["LISA_E"] = apply_pre_merger_kernel(
        data["LISA_E"].to_frequencyseries(),
        whitening_psd=psds_for_whitening["LISA_E"],
        window=window,
        window_length=window_length,
        nfz=forward_zeroes,
        nctf=nctf,
        uid=4236,
        copy_output=True,
    )
    return strain_ww


def generate_waveform_lisa_pre_merger(
    waveform_params,
    psds_for_whitening,
    sample_rate,
    window_length,
    cutoff_time,
    forward_zeroes=0,
):
    """Generate a pre-merger LISA waveform.

    UIDs used for FFTs: 1234(0), 1235(0), 1236(0), 1237(0) 
    
    Parameters
    ----------
    waveform_params: dict
        A dictionary of waveform parameters that will be passed to the waveform
        generator.
    psds_for_whitening: dict[str: FrequencySeries]
        Power spectral denisities for whitening in the frequency-domain.
    sample_rate : float
        Sampling rate.
    window_length : int
        Length (in samples) of time-domain window applied to the start of the
        waveform.
    cutoff_time: float
        Time (in seconds) from the end of the waveform to cutoff.
    forward_zeroes : int
        Number of samples to set to zero at the start of the waveform. If used,
        the window will be applied starting after the zeroes.
    """
    window = get_window(window_length)
    nctf = int((cutoff_time + waveform_params.get("cutoff_deltat", 0.0)) * sample_rate)

    outs = pycbc.waveform.get_fd_det_waveform(
        ifos=['LISA_A','LISA_E'], **waveform_params
    )

    # Apply pre-merger kernel
    tout_A_ww = apply_pre_merger_kernel(
        outs["LISA_A"],
        whitening_psd=psds_for_whitening["LISA_A"],
        window=window,
        window_length=window_length,
        nfz=forward_zeroes,
        nctf=nctf,
        uid=1235,
    )
    tout_E_ww = apply_pre_merger_kernel(
        outs["LISA_E"],
        whitening_psd=psds_for_whitening["LISA_E"],
        window=window,
        window_length=window_length,
        nfz=forward_zeroes,
        nctf=nctf,
        uid=12350,
    )
    
    # Back to FD for search/inference
    fouts_ww = {}
    fouts_ww["LISA_A"] = pycbc.strain.strain.execute_cached_fft(
        tout_A_ww,
        copy_output=False,
        uid=1236,
    )
    fouts_ww["LISA_E"] = pycbc.strain.strain.execute_cached_fft(
        tout_E_ww,
        copy_output=False,
        uid=12360,
    )
    return fouts_ww
