from importlib import import_module

from .gate import add_gate_option_group, gates_from_cli
from .gate import (
    apply_gates_to_td,
    apply_gates_to_fd,
    gate_data,
    psd_gates_from_cli,
)

_STRAIN_EXPORTS = {
    "detect_loud_glitches",
    "from_cli",
    "from_cli_single_ifo",
    "from_cli_multi_ifos",
    "insert_strain_option_group",
    "insert_strain_option_group_multi_ifo",
    "verify_strain_options",
    "verify_strain_options_multi_ifo",
    "StrainSegments",
    "StrainBuffer",
}
_RECALIBRATE_EXPORTS = {"CubicSpline", "PhysicalModel"}
_SUBMODULE_EXPORTS = {"calibration", "gate", "lines", "recalibrate", "strain"}


def __getattr__(name):
    """Load frame- and detector-facing strain features only when requested."""
    if name in _STRAIN_EXPORTS:
        value = getattr(import_module(".strain", __name__), name)
    elif name in _RECALIBRATE_EXPORTS:
        value = getattr(import_module(".recalibrate", __name__), name)
    elif name in _SUBMODULE_EXPORTS:
        value = import_module(f".{name}", __name__)
    elif name == "models":
        recalibrate = import_module(".recalibrate", __name__)
        value = {
            recalibrate.CubicSpline.name: recalibrate.CubicSpline,
            recalibrate.PhysicalModel.name: recalibrate.PhysicalModel,
        }
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def read_model_from_config(cp, ifo, section="calibration"):
    """Returns an instance of the calibration model specified in the
    given configuration file.

    Parameters
    ----------
    cp : WorflowConfigParser
        An open config file to read.
    ifo : string
        The detector (H1, L1)  whose model will be loaded.
    section : {"calibration", string}
        Section name from which to retrieve the model.

    Returns
    -------
    instance
        An instance of the calibration model class.
    """
    model = cp.get_opt_tag(section, "{}_model".format(ifo.lower()), None)
    model_classes = __getattr__("models")
    recalibrator = model_classes[model].from_config(
        cp, ifo.lower(), section
    )

    return recalibrator


__all__ = [
    "CubicSpline",
    "PhysicalModel",
    "detect_loud_glitches",
    "from_cli",
    "from_cli_single_ifo",
    "from_cli_multi_ifos",
    "insert_strain_option_group",
    "insert_strain_option_group_multi_ifo",
    "verify_strain_options",
    "verify_strain_options_multi_ifo",
    "gate_data",
    "StrainSegments",
    "StrainBuffer",
    "add_gate_option_group",
    "gates_from_cli",
    "apply_gates_to_td",
    "apply_gates_to_fd",
    "psd_gates_from_cli",
    "models",
    "read_model_from_config",
]
