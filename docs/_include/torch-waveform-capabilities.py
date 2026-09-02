#!/usr/bin/env python

# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.

"""Build the Torch-native waveform capability table from its registry."""

from pycbc.waveform.torch_waveform_registry import (
    render_torch_waveform_capabilities,
)


with open("torch-waveform-capabilities.rst", "w", encoding="utf-8") as output:
    output.write(render_torch_waveform_capabilities())
