# Copyright (C) 2022 Gareth Cabourn Davies
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

"""
This module contains functions for reading in command line options and
applying cuts to triggers or templates in the offline search
"""
from pycbc.types.backend import (
    backend_array, wrap_backend_array,
)
import logging
import copy
import numpy as np
from pycbc.events import ranking
from pycbc.io import hdf
from pycbc.tmpltbank import bank_conversions as bank_conv
from pycbc.io import get_chisq_from_file_choice
from pycbc.types import Array

# Only used to check isinstance:
from pycbc.io.hdf import ReadByTemplate

logger = logging.getLogger('pycbc.events.cuts')

# sngl_rank_keys are the allowed names of reweighted SNR functions
sngl_rank_keys = ranking.sngls_ranking_function_dict.keys()

trigger_param_choices = list(sngl_rank_keys)
trigger_param_choices += [cc + '_chisq' for cc in hdf.chisq_choices]
trigger_param_choices += ['end_time', 'psd_var_val', 'sigmasq',
                          'sigma_multiple']

template_fit_param_choices = ['fit_by_fit_coeff', 'smoothed_fit_coeff',
                              'fit_by_count_above_thresh',
                              'smoothed_fit_count_above_thresh',
                              'fit_by_count_in_template',
                              'smoothed_fit_count_in_template']
template_param_choices = bank_conv.conversion_options + \
                             template_fit_param_choices

# What are the inequalities associated with the cuts?
# 'upper' means upper limit, and so requires value < threshold
# to keep a trigger
ineq_functions = {
    'upper': np.less,
    'lower': np.greater,
    'upper_inc': np.less_equal,
    'lower_inc': np.greater_equal
}
ineq_choices = list(ineq_functions.keys())


def _torch_cut_tensor(value):
    """Return existing Torch storage without importing the optional backend."""
    return backend_array(value, "torch")


def _torch_trigger_cut_state(triggers):
    """Create device-resident trigger indices for Torch-backed inputs."""
    snr = triggers["snr"]
    reference = _torch_cut_tensor(snr)
    if reference is None:
        return None

    import torch

    if reference.ndim != 1:
        raise ValueError("trigger columns must be one-dimensional")
    indices = torch.arange(
        reference.shape[0], dtype=torch.int64, device=reference.device
    )
    return reference, indices, isinstance(snr, Array)


def _torch_cut_vector(value, reference, expected_length):
    """Place a numeric cut vector beside the reference Torch tensor."""
    import torch

    tensor = _torch_cut_tensor(value)
    if tensor is None:
        host = value.numpy() if isinstance(value, Array) else value
        try:
            tensor = torch.as_tensor(host, device=reference.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError(
                "trigger cut values must be numeric vectors"
            ) from exc
    elif tensor.device != reference.device:
        raise ValueError("trigger columns must use one Torch device")

    if tensor.ndim != 1 or tensor.is_complex() or tensor.dtype == torch.bool:
        raise TypeError(
            "trigger cut values must be one-dimensional numeric vectors"
        )
    if tensor.shape[0] != expected_length:
        raise ValueError("trigger columns must have matching lengths")
    return tensor


def _torch_cut_mask(value, threshold, cut_function):
    """Apply a parsed cut without invoking NumPy's host comparison path."""
    import torch

    if cut_function is np.less:
        mask = torch.lt(value, threshold)
    elif cut_function is np.greater:
        mask = torch.gt(value, threshold)
    elif cut_function is np.less_equal:
        mask = torch.le(value, threshold)
    elif cut_function is np.greater_equal:
        mask = torch.ge(value, threshold)
    else:
        mask = cut_function(value, threshold)

    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError(
            "Torch trigger cut functions must return a boolean tensor"
        )
    if mask.shape != value.shape:
        raise ValueError("trigger cut masks must match the selected values")
    return mask


def _wrap_torch_cut_indices(indices, wrap_array):
    """Preserve the public PyCBC Array container on the Torch path."""
    if not wrap_array:
        return indices


    return Array(wrap_backend_array(indices), copy=False)


def insert_cuts_option_group(parser):
    """
    Add options to the parser for cuts to the templates/triggers
    """
    parser.add_argument('--trigger-cuts', nargs='+',
                        help="Cuts to apply to the triggers, supplied as "
                             "PARAMETER:VALUE:LIMIT, where, PARAMETER is the "
                             "parameter to be cut, VALUE is the value at "
                             "which it is cut, and LIMIT is one of '"
                             + "', '".join(ineq_choices) +
                             "' to indicate the inequality needed. "
                             "PARAMETER is one of:'"
                             + "', '".join(trigger_param_choices) +
                             "'. For example snr:6:LOWER removes triggers "
                             "with matched filter SNR < 6")
    parser.add_argument('--template-cuts', nargs='+',
                        help="Cuts to apply to the triggers, supplied as "
                             "PARAMETER:VALUE:LIMIT. Format is the same as in "
                             "--trigger-cuts. PARAMETER can be one of '"
                             + "', '".join(template_param_choices) + "'.")


def convert_inputstr(inputstr, choices):
    """
    Convert the inputstr into a dictionary keyed on parameter
    with a tuple of the function to be used in the cut, and
    the float to compare to.
    Do input checks
    """
    try:
        cut_param, cut_value_str, cut_limit = inputstr.split(':')
    except ValueError as value_e:
        logger.warning("ERROR: Cut string format not correct, please "
                       "supply as PARAMETER:VALUE:LIMIT")
        raise value_e

    if cut_param.lower() not in choices:
        raise NotImplementedError("Cut parameter " + cut_param.lower() + " "
                                  "not recognised, choose from "
                                  + ", ".join(choices))
    if cut_limit.lower() not in ineq_choices:
        raise NotImplementedError("Cut inequality " + cut_limit.lower() + " "
                                  "not recognised, choose from "
                                  + ", ".join(ineq_choices))

    try:
        cut_value = float(cut_value_str)
    except ValueError as value_e:
        logger.warning("ERROR: Cut value must be convertible into a float, "
                       "got '%s'.", cut_value_str)
        raise value_e

    return {(cut_param, ineq_functions[cut_limit]): cut_value}


def check_update_cuts(cut_dict, new_cut):
    """
    Update a cuts dictionary, but check whether the cut exists already,
    warn and only apply the strictest cuts


    Parameters
    ----------
    cut_dict: dictionary
        Dictionary containing the cuts to be checked, will be updated

    new_cut: single-entry dictionary
        dictionary to define the new cut which is being considered to add
    """
    new_cut_key = list(new_cut.keys())[0]
    if new_cut_key in cut_dict:
        # The cut has already been called
        logger.warning("WARNING: Cut parameter %s and function %s have "
                       "already been used. Utilising the strictest cut.",
                       new_cut_key[0], new_cut_key[1].__name__)
        # Extract the function and work out which is strictest
        cut_function = new_cut_key[1]
        value_new = list(new_cut.values())[0]
        value_old = cut_dict[new_cut_key]
        if cut_function(value_new, value_old):
            # The new threshold would survive the cut of the
            # old threshold, therefore the new threshold is stricter
            # - update it
            logger.warning("WARNING: New threshold of %.3f is "
                           "stricter than old threshold %.3f, "
                           "using cut at %.3f.",
                           value_new, value_old, value_new)
            cut_dict.update(new_cut)
        else:
            # New cut would not make a difference, ignore it
            logger.warning("WARNING: New threshold of %.3f is less "
                           "strict than old threshold %.3f, using "
                           "cut at %.3f.",
                           value_new, value_old, value_old)
    else:
        # This is a new cut - add it
        cut_dict.update(new_cut)


def ingest_cuts_option_group(args):
    """
    Return dictionaries for trigger and template cuts.
    """
    # Deal with the case where no cuts are supplied:
    if not args.trigger_cuts and not args.template_cuts:
        return {}, {}

    # Deal with the case where one set of cuts is supplied
    # but not the other
    trigger_cut_strs = args.trigger_cuts or []
    template_cut_strs = args.template_cuts or []

    # Handle trigger cuts
    trigger_cut_dict = {}
    for inputstr in trigger_cut_strs:
        new_trigger_cut = convert_inputstr(inputstr, trigger_param_choices)
        check_update_cuts(trigger_cut_dict, new_trigger_cut)

    # Handle template cuts
    template_cut_dict = {}
    for inputstr in template_cut_strs:
        new_template_cut = convert_inputstr(inputstr, template_param_choices)
        check_update_cuts(template_cut_dict, new_template_cut)

    return trigger_cut_dict, template_cut_dict


def sigma_multiple_cut_thresh(template_ids, statistic,
                              cut_thresh, ifo):
    """
    Apply cuts based on a multiple of the median sigma value for the template

    Parameters
    ----------

    template_ids:
        template_id values for each of the triggers to be considered,
        this will be used to associate a sigma threshold for each trigger
    statistic:
        A PyCBC ranking statistic instance. Used to get the median_sigma
        value for the cuts. If fits_by_tid does not exist for the specified
        ifo (where median_sigma lives), an error will be raised.
    ifo:
        The IFO for which we want to read median_sigma
    cut_thresh: int or float
        The multiple of median_sigma to compare triggers to

    Returns
    -------
    idx_out: numpy array
        An array of the indices of triggers which meet the criteria
        set by the dictionary
    """
    statistic_classname = statistic.__class__.__name__
    if not hasattr(statistic, 'fits_by_tid'):
        raise ValueError("Cut parameter 'sigma_muliple' cannot "
                         "be used when the ranking statistic " +
                         statistic_classname + " does not use "
                         "template fitting.")
    tid_med_sigma = statistic.fits_by_tid[ifo]['median_sigma']
    return cut_thresh * tid_med_sigma[template_ids]


def apply_trigger_cuts(triggers, trigger_cut_dict, statistic=None):
    """
    Fetch/Calculate the parameter for triggers, and then
    apply the cuts defined in template_cut_dict

    Parameters
    ----------
    triggers: ReadByTemplate object or dictionary
        The triggers in this particular template. This
        must have the correct datasets required to calculate
        the values we cut on.

    trigger_cut_dict: dictionary
        Dictionary with tuples of (parameter, cut_function)
        as keys, cut_thresholds as values
        made using ingest_cuts_option_group function


    Returns
    -------
    idx_out: numpy.ndarray, torch.Tensor, or pycbc.types.Array
        An array of the indices which meet the criteria
        set by the dictionary. Torch-backed trigger dictionaries return
        indices on the same device; PyCBC Array inputs retain that container.
    """
    torch_state = _torch_trigger_cut_state(triggers)
    if torch_state is None:
        idx_out = np.arange(len(triggers['snr']))
        torch_reference = None
        wrap_torch_result = False
    else:
        torch_reference, idx_out, wrap_torch_result = torch_state

    # Loop through the different cuts, and apply them
    for parameter_cut_function, cut_thresh in trigger_cut_dict.items():
        # The function and threshold are stored as a tuple so unpack it
        parameter, cut_function = parameter_cut_function

        # What kind of parameter is it?
        if parameter.endswith('_chisq'):
            # parameter is a chisq-type thing
            chisq_choice = parameter.split('_')[0]
            # Currently calculated for all triggers - this seems inefficient
            value = get_chisq_from_file_choice(triggers, chisq_choice)
            # Apply any previous cuts to the value for comparison
            if torch_reference is None:
                value = value[idx_out]
        elif parameter == "sigma_multiple":
            if isinstance(triggers, ReadByTemplate):
                value = np.sqrt(triggers['sigmasq'][idx_out])
                # Get a cut threshold value, this will be different
                # depending on the template ID, so we rewrite cut_thresh
                # as a value for each trigger, numpy comparison functions
                # allow this
                cut_thresh = sigma_multiple_cut_thresh(triggers.template_num,
                                                       statistic,
                                                       cut_thresh,
                                                       triggers.ifo)
            else:
                err_msg = "Cuts on 'sigma_multiple' are only implemented for "
                err_msg += "triggers in a ReadByTemplate format. This code "
                err_msg += f"uses a {type(triggers).__name__} format."
                raise NotImplementedError(err_msg)
        elif ((not hasattr(triggers, "file") and parameter in triggers)
                or (hasattr(triggers, "file")
                    and parameter in triggers.file[triggers.ifo])):
            # parameter can be read direct from the trigger dictionary / file
            value = triggers[parameter]
            # Apply any previous cuts to the value for comparison
            if torch_reference is None:
                value = value[idx_out]
        elif parameter in sngl_rank_keys:
            # parameter is a newsnr-type thing
            # Currently calculated for all triggers - this seems inefficient
            value = ranking.get_sngls_ranking_from_trigs(triggers, parameter)
            # Apply any previous cuts to the value for comparison
            if torch_reference is None:
                value = value[idx_out]
        else:
            raise NotImplementedError("Parameter '" + parameter + "' not "
                                      "recognised. Input sanitisation means "
                                      "this shouldn't have happened?!")

        if torch_reference is None:
            idx_out = idx_out[cut_function(value, cut_thresh)]
        else:
            value = _torch_cut_vector(
                value, torch_reference, len(triggers["snr"])
            )
            selected = value[idx_out]
            idx_out = idx_out[
                _torch_cut_mask(selected, cut_thresh, cut_function)
            ]

    if torch_reference is not None:
        return _wrap_torch_cut_indices(idx_out, wrap_torch_result)
    return idx_out


def apply_template_fit_cut(statistic, ifos, parameter_cut_function, cut_thresh,
                           template_ids):
    """
    Apply cuts to template fit parameters, these have a few more checks
    needed, so we separate out from apply_template_cuts defined later

    Parameters
    ----------
    statistic:
        A PyCBC ranking statistic instance. Used for the template fit
        cuts. If fits_by_tid does not exist for each ifo, then
        template fit cuts will be skipped. If a fit cut has been specified
        and fits_by_tid does not exist for all ifos, an error will be raised.

    ifos: list of strings
        List of IFOS used in this findtrigs instance.
        Templates must pass cuts in all IFOs.

    parameter_cut_function: tuple
        First entry: Which parameter is being used for the cut?
        Second entry: Cut function

    cut_thresh: float or int
        Cut threshold to the parameter according to the cut function

    template_ids: numpy array
        Array of template_ids which have passed previous cuts


    Returns
    -------
    tids_out: numpy array
        Array of template_ids which have passed this cut
    """
    parameter, cut_function = parameter_cut_function
    statistic_classname = statistic.__class__.__name__

    # We can only apply template fit cuts if template fits have been done
    if not hasattr(statistic, 'fits_by_tid'):
        raise ValueError("Cut parameter " + parameter + " cannot "
                         "be used when the ranking statistic " +
                         statistic_classname + " does not use "
                         "template fitting.")

    # Is the parameter actually in the fits dictionary?
    if parameter not in statistic.fits_by_tid[ifos[0]]:
        # Shouldn't get here due to input sanitisation
        raise ValueError("Cut parameter " + parameter + " not "
                         "available in fits file.")

    # Template IDs array to cut down in each IFO
    tids_out = copy.copy(template_ids)
    # Need to apply this cut to all IFOs
    for ifo in ifos:
        fits_dict = statistic.fits_by_tid[ifo]
        values = fits_dict[parameter][tids_out]
        # Only keep templates which pass this cut
        tids_out = tids_out[cut_function(values, cut_thresh)]

    return tids_out


def apply_template_cuts(bank, template_cut_dict, template_ids=None,
                        statistic=None, ifos=None):
    """
    Fetch/calculate the parameter for the templates, possibly already
    preselected by template_ids, and then apply the cuts defined
    in template_cut_dict
    As this is used to select templates for use in findtrigs codes,
    we remove anything which does not pass

    Parameters
    ----------
    bank: h5py File object, or a dictionary
        Must contain the usual template bank datasets

    template_cut_dict: dictionary
        Dictionary with tuples of (parameter, cut_function)
        as keys, cut_thresholds as values
        made using ingest_cuts_option_group function

    Optional Parameters
    -------------------
    template_ids: list of indices
        Indices of templates to consider within the bank, useful if
        templates have already been down-selected

    statistic:
        A PyCBC ranking statistic instance. Used for the template fit
        cuts. If fits_by_tid does not exist for each ifo, then
        template fit cuts will be skipped. If a fit cut has been specified
        and fits_by_tid does not exist for all ifos, an error will be raised.
        If not supplied, no template fit cuts will be attempted.

    ifos: list of strings
        List of IFOS used in this findtrigs instance.
        Templates must pass cuts in all IFOs. This is important
        e.g. for template fit parameter cuts.


    Returns
    -------
    tids_out: numpy array
        Array of template_ids which have passed all cuts
    """
    # Get the initial list of templates:
    tids_out = np.arange(bank['mass1'].size) \
        if template_ids is None else template_ids[:]

    if (statistic is None) ^ (ifos is None):
        raise NotImplementedError("Either both or neither of statistic and "
                                  "ifos must be supplied.")

    if not template_cut_dict:
        # No cuts are defined in the dictionary: just return the
        # list of all tids
        return tids_out

    # Loop through the different cuts, and apply them
    for parameter_cut_function, cut_thresh in template_cut_dict.items():
        # The function and threshold are stored as a tuple so unpack it
        parameter, cut_function = parameter_cut_function

        if parameter in bank_conv.conversion_options:
            # Calculate the parameter values using the bank property helper
            values = bank_conv.get_bank_property(parameter, bank, tids_out)
            # Only keep templates which pass this cut
            tids_out = tids_out[cut_function(values, cut_thresh)]
        elif parameter in template_fit_param_choices:
            if statistic and ifos:
                tids_out = apply_template_fit_cut(statistic,
                                                  ifos,
                                                  parameter_cut_function,
                                                  cut_thresh,
                                                  tids_out)
        else:
            raise ValueError("Cut parameter " + parameter + " not recognised."
                             " This shouldn't happen with input sanitisation")

    return tids_out
