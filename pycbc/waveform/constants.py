"""Physical constants shared by Torch-native waveform ports.

The values mirror lalsimulation so independently ported waveform models do
not drift through rounded local copies.
"""

import math


_PI = math.pi
_EULER_GAMMA = 0.577215664901532860606512090082402431
_MSUN_SI = 1.988409870698050731911960804878414216e30
_PC_SI = 3.085677581491367278913937957796471611e16
_MRSUN_SI = 1.476625038050124729627979840144936351e3
_MTSUN_SI = 4.925490947641266978197229498498379006e-6


__all__ = [
    "_PI",
    "_EULER_GAMMA",
    "_MSUN_SI",
    "_PC_SI",
    "_MRSUN_SI",
    "_MTSUN_SI",
]
