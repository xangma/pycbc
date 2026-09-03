# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Optional Triton kernels used by Torch-native waveform models.

Each model module owns its availability check so importing this package does
not import Torch, Triton, or an unrelated waveform implementation.
"""
