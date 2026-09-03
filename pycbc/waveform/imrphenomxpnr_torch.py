# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

"""Torch-native parameter fits shared by IMRPhenomXO4a and XPNR.

This module contains the effective single-spin map, calibration tapers, and
ringdown opening-angle fit used by the PNR Euler-angle prescription.  The
operations are batchable and remain on the input Torch device.  Waveform
assembly is intentionally kept in the model modules that consume these fits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import torch

from pycbc.waveform._cubic_spline_torch import (
    _natural_cubic_coeff,
    _spline_derivative,
    _spline_eval,
)
from pycbc.waveform.imrphenomx_utils_torch import (
    MTSUN,
    final_spin_2017,
    get_remnant_fMs,
    precessing_final_spin_2017,
    qnm_fdamp_21,
)
from pycbc.waveform.imrphenomx_spintaylor_torch import (
    SpinTaylorAngleSpline,
    SpinTaylorJFrame,
    SpinTaylorTrajectory,
    build_spintaylor_angle_spline,
    spintaylor_alpha_imr,
    spintaylor_beta_imr,
    spintaylor_inspiral_cosbeta,
    spintaylor_j_frame,
    spintaylor_j_frame_angles,
    spintaylor_t4_time_trajectory,
)
from pycbc.waveform.imrphenomxp_msa_torch import (
    PNRSourceFrame,
    build_msa_state,
    msa_angles,
    remap_source_frame_parameters_pnr,
)


_ALPHA_A1_COEFFICIENTS = (
    (
        (1.04459978e00, -3.06075403e00, 2.95704570e00, 7.48684859e00, -4.31919865e00),
        (-7.01729932e00, 2.36356947e01, -8.83233850e00, -5.53590259e01, 1.49383286e01),
        (1.62369754e01, -5.16103690e01, 7.51590132e00, 1.18370165e02, -1.37666457e01),
        (-1.11475857e01, 3.35645858e01, 0.0, -7.70839731e01, 0.0),
    ),
    (
        (-7.12384997e00, 3.04210473e01, -4.56974441e01, -8.29725789e01, 6.53043804e01),
        (6.38502724e01, -2.19868327e02, 1.84907979e02, 6.00452453e02, -2.80558563e02),
        (-1.60606542e02, 4.77451870e02, -2.63672148e02, -1.27586539e03, 3.98164795e02),
        (1.15484744e02, -3.09155358e02, 1.21504291e02, 8.37135478e02, -1.62233651e02),
    ),
    (
        (2.54285862e01, -6.89847827e01, 1.40382881e02, 2.09407834e02, -1.87954067e02),
        (-1.73426975e02, 4.86279790e02, -6.61752262e02, -1.50438712e03, 9.25254367e02),
        (4.15347868e02, -1.07048007e03, 1.12523251e03, 3.22299321e03, -1.52732767e03),
        (-2.95068093e02, 7.03127824e02, -6.36159668e02, -2.14803436e03, 7.75401797e02),
    ),
    (
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
)

_ALPHA_A2_COEFFICIENTS = (
    (
        (7.08347660e-01, 0.0, 1.51680145e00, -1.12976630e00, -8.96225587e-01),
        (-4.78891038e00, 0.0, 7.95447722e00, 5.14517166e00, -2.24073158e01),
        (1.09348438e01, 0.0, -2.66171290e01, -6.02467237e00, 6.64680541e01),
        (-7.71924820e00, 0.0, 1.92361544e01, 0.0, -4.84350012e01),
    ),
    (
        (-5.51390083e00, -4.14264606e00, -2.48290270e01, 4.27226700e01, 9.78335223e00),
        (4.61263476e01, 4.04546173e01, -1.34011079e02, -2.56156198e02, 4.23612394e02),
        (-1.11931113e02, -8.80207636e01, 4.22254858e02, 4.50060096e02, -1.18188078e03),
        (8.41956834e01, 5.56575471e01, -2.86460064e02, -2.14207360e02, 8.26652394e02),
    ),
    (
        (1.81136200e01, 2.07574049e01, 7.77696260e01, -3.40634217e02, 4.91231205e01),
        (-1.24497496e02, -1.85643669e02, 1.04563374e03, 2.10619339e03, -2.95316488e03),
        (2.90326939e02, 3.77825566e02, -2.78531037e03, -3.86114313e03, 7.56334211e03),
        (-2.35039929e02, -2.17428672e02, 1.77987293e03, 2.04687886e03, -5.06952341e03),
    ),
    (
        (0.0, 0.0, 0.0, 7.38662593e02, -2.87089460e02),
        (0.0, 0.0, -2.71250495e03, -4.57681784e03, 6.59296751e03),
        (0.0, 5.54407725e01, 6.59102127e03, 8.55100850e03, -1.60298657e04),
        (6.41488129e01, -9.27544052e01, -4.16235846e03, -4.76870795e03, 1.04968429e04),
    ),
)

_ALPHA_A3_COEFFICIENTS = (
    (
        (2.37641854e-02, 8.53850677e-01, 2.26634422e-01, -1.68372588e00, 0.0),
        (2.29535857e-01, -6.63002642e00, -2.26132428e00, 1.24225169e01, 0.0),
        (-3.69774894e-01, 1.43739502e01, 4.86077798e00, -2.59086482e01, 0.0),
        (0.0, -9.30697076e00, -2.79807179e00, 1.59684471e01, 0.0),
    ),
    (
        (0.0, -1.36518140e01, -4.60399072e00, 2.94176176e01, 0.0),
        (-5.75632659e00, 1.05782962e02, 4.57920306e01, -2.15952616e02, 0.0),
        (1.02674822e01, -2.27904811e02, -9.95024478e01, 4.48148907e02, 0.0),
        (-2.15345685e00, 1.46405699e02, 5.86426400e01, -2.74864857e02, 0.0),
    ),
    (
        (-1.05830284e00, 6.56968525e01, 2.84160912e01, -1.58222595e02, 0.0),
        (4.22174178e01, -5.05493224e02, -2.82597366e02, 1.15097184e03, 0.0),
        (-8.04678865e01, 1.07807205e03, 6.18415907e02, -2.37206023e03, 0.0),
        (2.63933567e01, -6.84842578e02, -3.68281254e02, 1.44651605e03, 0.0),
    ),
    (
        (3.83732885e00, -9.76185045e01, -5.44153111e01, 2.68829791e02, 0.0),
        (-9.21827041e01, 7.44071961e02, 5.41525413e02, -1.93561667e03, 0.0),
        (1.82667964e02, -1.56625497e03, -1.18882155e03, 3.96215587e03, 0.0),
        (-7.18225585e01, 9.81227739e02, 7.10130827e02, -2.40403332e03, 0.0),
    ),
)

_ALPHA_A4_COEFFICIENTS = (
    (
        (
            1.59708156e-01,
            -3.13330481e-01,
            3.82163433e-01,
            6.14498512e-01,
            -3.91386611e-01,
        ),
        (3.68521318e-01, 2.65522323e00, -2.23540184e00, -4.37061988e00, 2.01077124e00),
        (-8.32553332e-01, -6.02094408e00, 4.86548073e00, 8.36610850e00, -2.56728295e00),
        (8.05210158e-01, 4.06905683e00, -3.59268661e00, -4.39013590e00, 5.98214021e-01),
    ),
    (
        (1.76157925e00, 4.37672477e00, -3.65397391e00, -9.43917739e00, 2.79796061e00),
        (-8.10802054e00, -3.83223991e01, 2.30718712e01, 6.83013158e01, -1.66542213e01),
        (1.87499491e01, 9.08302756e01, -5.62222338e01, -1.34263772e02, 1.75463323e01),
        (-1.63620472e01, -6.32085162e01, 4.63243515e01, 7.33928698e01, 0.0),
    ),
    (
        (-1.01776826e01, -2.06957701e01, 8.82898695e00, 4.68191983e01, 0.0),
        (5.43489123e01, 1.91078478e02, -8.10382470e01, -3.45628809e02, 3.77841307e01),
        (-1.27304367e02, -4.65340116e02, 2.43493364e02, 6.96748172e02, -4.14889950e01),
        (1.07160127e02, 3.33094494e02, -2.25400241e02, -3.93084818e02, -2.62824389e00),
    ),
    (
        (2.05624058e01, 3.29280099e01, 0.0, -7.52635673e01, -2.10483740e01),
        (-1.10660415e02, -3.19269499e02, 8.40211777e01, 5.64442664e02, 0.0),
        (2.60872767e02, 7.88656773e02, -3.46838253e02, -1.15687118e03, 1.30514173e01),
        (-2.17001178e02, -5.76084011e02, 3.60191260e02, 6.62943701e02, 0.0),
    ),
)


_BETA_B0_COEFFICIENTS = (
    (
        (1.66666664, 0.0, -1.56973167, -0.731228312, 3.01647084),
        (0.0499601117, 1.63860030, 0.0, 0.0, -4.93485987),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        (-17.6204300, 0.0, 12.5605160, 5.10396051, -23.3247834),
        (0.0, -30.7243357, 0.0, 0.0, 81.2576632),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        (94.2765416, 0.0, 0.0, 0.0, 0.0),
        (0.0, 168.442510, -4.98474167, 0.0, -419.372351),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        (-190.285349, 0.0, -89.5949243, -35.4218713, 169.656947),
        (0.0, -287.986635, 0.0, 0.0, 711.235264),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
)

_BETA_B1_COEFFICIENTS = (
    (
        (9.90840259, -33.2454515, -28.6081100, 25.1567951, 0.0),
        (8.90772876, 151.402126, 44.1611143, -49.2342757, 0.0),
        (-11.1362664, -129.998137, -38.2595920, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        (-191.489844, 710.589710, 641.633616, -644.370516, 0.0),
        (-43.3793801, -3236.10605, -1305.28975, 1778.60699, 0.0),
        (66.2671100, 2775.17195, 1291.79889, -865.826585, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        (1261.43166, -4739.28061, -4121.20327, 4800.89044, 0.0),
        (0.0, 21443.2888, 9005.14520, -15126.6329, 0.0),
        (0.0, -18286.4908, -9404.56062, 9492.93573, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        (-2624.96323, 9852.62924, 8112.82583, -10693.4934, 0.0),
        (0.0, -44345.8441, -18042.8855, 36086.4889, 0.0),
        (-188.093090, 37589.0504, 19423.3528, -24837.9649, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
)

_BETA_B2_COEFFICIENTS = (
    (
        (16.8444475, -9.96099698, -76.8630145, -42.1556808, 73.3370180),
        (3.49258945, -74.4266403, 339.005879, 529.979449, -604.337081),
        (0.0, 489.740754, -552.234827, -1321.26338, 1200.63914),
        (-8.31987801, -470.597369, 262.076484, 886.094957, -725.121024),
    ),
    (
        (-258.211935, 333.569187, 1380.61489, 458.035577, -1076.48973),
        (0.0, 637.900482, -5605.52556, -8017.36623, 9319.54814),
        (0.0, -8376.54765, 8187.95918, 22079.4318, -18297.2414),
        (39.8198351, 8668.82648, -3038.68645, -15693.2408, 10922.5254),
    ),
    (
        (1630.32394, -2652.57994, -7658.99068, -1152.54729, 4938.39236),
        (0.0, -1243.25251, 28371.3129, 39874.9465, -45444.2750),
        (0.0, 49206.8135, -36788.9222, -123091.363, 87385.8633),
        (0.0, -53190.9605, 8867.69471, 92402.6509, -51064.5576),
    ),
    (
        (-3328.26175, 5921.87248, 13511.1259, 0.0, -7076.09947),
        (-253.504341, 0.0, -44371.5296, -66121.9185, 69238.1252),
        (0.0, -96618.5024, 47228.6307, 227703.658, -127186.811),
        (0.0, 106346.795, 0.0, -178622.185, 70742.6805),
    ),
)

_BETA_B3_COEFFICIENTS = (
    (
        (-38.4565012, -372.563299, 415.226702, 1006.90394, -839.744987),
        (0.0, 3460.28868, -2136.68892, -8664.65191, 6597.82412),
        (0.0, -8980.93407, 2774.56494, 19615.8493, -12193.6587),
        (27.1298988, 6572.89583, -916.538843, -13098.7997, 7039.85911),
    ),
    (
        (641.285217, 6434.39871, -6124.30709, -16930.1376, 12133.1375),
        (333.545476, -61820.6463, 25987.5058, 149340.484, -96562.5710),
        (-931.951414, 165382.217, -18577.8489, -345011.414, 168522.923),
        (248.942253, -123069.045, -6074.27694, 234066.117, -92907.3962),
    ),
    (
        (-3711.86534, -36704.8647, 26840.4803, 92979.0279, -53733.3198),
        (-4438.80950, 365054.516, -74419.8342, -843552.293, 433017.698),
        (9759.83575, -1000298.02, -77927.3838, 1990510.43, -681841.816),
        (-3863.13127, 753226.934, 165966.971, -1370708.58, 339981.734),
    ),
    (
        (7198.81427, 69218.7556, -35239.7747, -167861.147, 72443.8477),
        (11705.8924, -706752.764, 0.0, 1561423.69, -580532.954),
        (-23099.6347, 1968590.70, 491882.690, -3755766.40, 725646.477),
        (8963.50149, -1493115.58, -556575.555, 2618763.68, -257044.129),
    ),
)

_BETA_B4_COEFFICIENTS = (
    (
        (4.43516177, 6009.23095, 0.0, -8874.03368, 0.0),
        (0.0, -43762.1347, 462.648713, 64111.6102, 0.0),
        (0.0, 90777.5710, 0.0, -134626.196, 0.0),
        (-291.806956, -56098.0213, 0.0, 84382.9555, 0.0),
    ),
    (
        (0.0, -119641.275, 0.0, 180891.623, 0.0),
        (6766.53146, 869092.181, -9106.00697, -1307564.39, 0.0),
        (-17121.7123, -1799674.54, 523.377635, 2746465.43, 0.0),
        (17819.7625, 1112321.75, 0.0, -1724143.23, 0.0),
    ),
    (
        (5226.73374, 736732.009, -1558.62480, -1133361.65, 0.0),
        (-80934.6012, -5343319.17, 66063.3209, 8192021.60, 0.0),
        (197663.741, 11048688.5, -28035.6801, -17189743.6, 0.0),
        (-173210.655, -6829273.41, 17094.3961, 10789880.4, 0.0),
    ),
    (
        (-16306.3512, -1420584.59, 0.0, 2214987.43, 0.0),
        (201426.348, 10291679.2, -100924.025, -16005455.8, 0.0),
        (-484862.499, -21254874.4, 0.0, 33540165.8, 0.0),
        (404834.917, 13136881.3, 0.0, -21037195.1, 0.0),
    ),
)

_BETA_B5_COEFFICIENTS = (
    (
        (0.310386302, 0.0600813481, -0.353776520, -0.886045114, 0.0),
        (0.0, 1.25742267, 0.384777116, 3.01828258, 0.0),
        (0.0, -2.21073099, 0.0, -2.04284938, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        (-1.88945371, -0.726051110, 9.06175997, 14.6766342, 0.0),
        (4.86280161, -26.6243788, -20.5433977, -42.7192221, 0.0),
        (-4.39245048, 44.5640600, 15.1720256, 22.2127532, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        (14.0364230, 0.0, -60.4082590, -72.5620325, 0.0),
        (-41.0909266, 177.965640, 160.792365, 175.894855, 0.0),
        (40.8774229, -275.528934, -139.837130, -54.3148597, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        (-27.5834452, 6.53438283, 120.524698, 113.506432, 0.0),
        (86.1828779, -359.365491, -343.549992, -220.083234, 0.0),
        (-92.4276057, 523.784793, 320.340101, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
)


_BETA_BF_COEFFICIENTS = (
    (
        (3.09601897, 1.34032610, -1.45826218, -0.937928603, 0.0),
        (0.0, -3.88910127, 3.71319679, -0.550316593, 0.0),
        (-0.378818904, 3.44727678, -3.74449485, 0.821928673, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        (-36.3088945, -13.8476321, 26.6368579, 6.74798082, 0.0),
        (0.0, 18.9698391, -15.8854735, -1.11744859, 0.0),
        (3.45434991, -17.3678802, 15.8282455, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        (203.218040, 63.6061063, -167.985530, -8.74477395, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (-7.62767762, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        (-410.212724, -131.527994, 343.405885, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ),
)


# Indexed as fit, eta power, spin-magnitude power, and cos(theta) power.
_COPRECESSING_FIT_COEFFICIENTS = (
    (  # mu1
        (
            (-2.98890174e-01, -2.69235962e-01, -1.66981400e00, -3.25953234e-01),
            (3.10731588e00, 2.35767086e00, 0.0, 2.43498635e00),
            (-5.27239994e00, -2.85191772e00, 3.69910553e00, -3.34353185e00),
        ),
        (
            (2.83580591e00, 4.58765298e00, 3.80240537e01, 0.0),
            (-4.58961732e01, -5.90965234e01, -3.06134124e01, -6.71291161e00),
            (8.53735433e01, 7.77424585e01, -3.86558965e01, 1.89942289e01),
        ),
        (
            (-6.65783904e00, -1.69564326e01, -2.49773688e02, 0.0),
            (2.28592256e02, 3.65316048e02, 2.91303571e02, -3.78457492e01),
            (-4.48771815e02, -5.04214664e02, 1.10210016e02, 0.0),
        ),
        (
            (0.0, 0.0, 5.19092710e02, 4.49179407e01),
            (-3.71962074e02, -6.11685145e02, -7.50921355e02, 0.0),
            (7.62503928e02, 9.08677395e02, 0.0, 0.0),
        ),
    ),
    (  # mu2
        (
            (-2.02502540e-01, 0.0, -3.87863737e00, -1.94647765e00),
            (0.0, 0.0, 9.12534447e00, -2.87950209e00),
            (-1.58131236e00, -8.00671915e00, -1.06817659e01, 2.70715432e00),
        ),
        (
            (0.0, 0.0, 5.78604124e01, 4.08691562e01),
            (8.72364229e00, 6.55310845e00, -6.43240440e01, 2.10132916e01),
            (5.33069685e00, 1.06442533e02, 1.03449590e02, 0.0),
        ),
        (
            (0.0, 0.0, -3.16185296e02, -2.76924732e02),
            (-5.05936693e01, 0.0, 0.0, -9.28493309e01),
            (0.0, -5.79811830e02, -2.05802266e02, 0.0),
        ),
        (
            (8.51436158e00, -2.49087968e01, 6.51115537e02, 6.47099541e02),
            (7.52332356e01, 0.0, 2.54661923e02, 0.0),
            (0.0, 1.03222489e03, 0.0, 0.0),
        ),
    ),
    (  # mu3
        (
            (-3.17858924e-03, 2.48066282e-02, -1.23872940e-02, -9.08508144e-03),
            (6.43824401e-02, -1.14919973e-01, -4.01788734e-02, 7.84548415e-02),
            (0.0, 1.51718799e-01, 0.0, -1.20970655e-01),
        ),
        (
            (6.54067059e-03, -1.50482854e-01, 1.61029465e-01, 0.0),
            (-8.51847106e-01, 5.63051639e-01, 7.87294341e-01, -1.88632810e-01),
            (0.0, -9.10606320e-01, -1.42496492e-01, 5.00625025e-01),
        ),
        (
            (0.0, 1.94815013e-01, -9.60114345e-01, 0.0),
            (4.35501780e00, 0.0, -3.90246382e00, -5.29498004e-01),
            (0.0, 1.14108488e00, 3.60304651e-01, 0.0),
        ),
        (
            (0.0, 0.0, 1.67682276e00, 5.44663689e-01),
            (-7.78742176e00, -1.45121681e00, 6.67574591e00, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        ),
    ),
    (  # nu0
        (
            (-1.40050910e02, 1.21045131e02, 3.75573898e02, 3.56880662e02),
            (9.69656479e02, -2.62095295e03, -2.80301545e03, 1.13064791e03),
            (-3.80191737e02, 3.76489144e03, 2.34502372e03, -1.09191647e03),
        ),
        (
            (1.28604150e03, 0.0, -3.28538542e03, -8.93916723e03),
            (-8.58972371e03, 3.46510614e04, 2.98001687e04, -3.75871238e03),
            (0.0, -5.40526831e04, -2.41699850e04, 5.11072226e03),
        ),
        (
            (-3.02176869e03, 0.0, 0.0, 4.96525314e04),
            (1.91665811e04, -1.76931761e05, -7.36180264e04, -7.99273344e03),
            (2.54661712e04, 2.78587059e05, 5.84802389e04, 0.0),
        ),
        (
            (0.0, -1.27661553e04, 2.65990668e04, -7.11798894e04),
            (0.0, 3.35360241e05, 0.0, 0.0),
            (-7.73072478e04, -4.99111316e05, 0.0, 0.0),
        ),
    ),
    (  # nu4
        (
            (4.49481835e-03, 1.59179379e-02, -7.33656551e-03, 0.0),
            (-3.16028942e-02, -7.76768528e-02, 7.64861112e-02, -7.93716551e-03),
            (3.25729551e-02, 4.84795811e-02, -8.33709907e-02, 3.41917884e-02),
        ),
        (
            (-4.67316682e-02, -3.27118716e-01, 0.0, 0.0),
            (3.77496559e-01, 1.74016014e00, -7.44879716e-01, -9.13086579e-02),
            (-3.45847253e-01, -1.42151090e00, 8.28154534e-01, -1.52767109e-01),
        ),
        (
            (1.08635205e-01, 1.90217409e00, 5.13747474e-01, 2.22117601e-01),
            (-1.32662456e00, -1.06852751e01, 1.80163775e00, 5.87741796e-01),
            (8.34552086e-01, 9.46433414e00, -2.00129470e00, 0.0),
        ),
        (
            (0.0, -3.35542027e00, -1.67075105e00, -1.01537065e00),
            (1.38785202e00, 1.97711659e01, 0.0, 0.0),
            (0.0, -1.81767526e01, 0.0, 0.0),
        ),
    ),
    (  # nu5
        (
            (-8.24852329e-03, -2.68106259e-02, -3.95509060e-02, 4.13508242e-03),
            (0.0, 8.87404766e-02, 1.28777123e-01, -3.96980662e-02),
            (-6.96270582e-02, -8.52495198e-02, -5.33540972e-02, 4.02874363e-02),
        ),
        (
            (1.29541487e-01, 3.96908249e-01, 6.66123113e-01, 0.0),
            (-4.57402308e-01, -8.83292400e-01, -1.67674467e00, 4.25544610e-01),
            (1.34613777e00, 3.45438117e-01, 2.93065248e-01, -1.29990564e-01),
        ),
        (
            (-7.66209616e-01, -1.97688956e00, -4.21512938e00, -6.00764557e-01),
            (3.52527019e00, 3.39853108e00, 9.57710907e00, -1.31746981e00),
            (-8.95748509e00, 0.0, 0.0, 0.0),
        ),
        (
            (1.33051669e00, 3.19730879e00, 9.11903434e00, 2.55062055e00),
            (-6.34717777e00, -5.03020035e00, -2.11785354e01, 0.0),
            (1.82938901e01, 0.0, 0.0, 0.0),
        ),
    ),
    (  # nu6
        (
            (1.60753917e-02, 3.69589365e-02, 0.0, 4.67460809e-02),
            (-8.14163374e-02, -3.53871924e-01, -3.23664263e-02, 0.0),
            (1.57642062e-01, 4.49846052e-01, 4.68080513e-02, -7.10443312e-02),
        ),
        (
            (-3.14932520e-01, -5.43366127e-01, -4.05623855e-02, -9.03957684e-01),
            (1.65256072e00, 5.78266831e00, 4.06667657e-01, 5.63722799e-01),
            (-2.77552262e00, -7.23117535e00, -6.79662526e-01, 2.39385816e-01),
        ),
        (
            (1.91773389e00, 3.18104872e00, 0.0, 4.62060734e00),
            (-1.01369676e01, -3.29265427e01, 0.0, -2.12229389e00),
            (1.57596950e01, 3.96497438e01, 1.81706494e00, 0.0),
        ),
        (
            (-3.81823248e00, -6.37440535e00, 2.92509266e-01, -7.10237285e00),
            (1.97689861e01, 6.16173322e01, -3.62928100e00, 0.0),
            (-2.89973575e01, -7.13125243e01, 0.0, 0.0),
        ),
    ),
    (  # zeta1
        (
            (-6.64596933e-06, -2.65487354e-05, 2.93046524e-05, -4.06577752e-05),
            (9.14075603e-05, 2.78657414e-04, -2.40358466e-04, 2.97287666e-05),
            (-7.04782574e-05, -3.28088748e-04, 1.70628764e-04, 0.0),
        ),
        (
            (7.26570370e-05, 6.13445304e-04, -1.71570550e-04, 7.19295439e-04),
            (-1.13031091e-03, -5.75897400e-03, 2.69942014e-03, -6.18248914e-04),
            (6.89104675e-04, 6.89186183e-03, -1.33029865e-03, 8.47582112e-05),
        ),
        (
            (-1.62592535e-04, -4.33992656e-03, 0.0, -3.59470348e-03),
            (4.77274706e-03, 3.68151682e-02, -1.14024958e-02, 1.68471071e-03),
            (-1.58004432e-03, -4.28423981e-02, 2.67905006e-03, 0.0),
        ),
        (
            (0.0, 9.27082591e-03, 1.14545659e-03, 5.76871905e-03),
            (-7.10577068e-03, -7.28752610e-02, 1.73091684e-02, 0.0),
            (0.0, 8.18766554e-02, 0.0, 0.0),
        ),
    ),
    (  # zeta2
        (
            (1.21917535e00, 5.64084304e00, -4.07600123e00, 3.34325806e00),
            (-1.93349374e01, -3.45695270e01, 4.19166290e01, -1.54037927e01),
            (1.91050318e01, 3.69257896e01, -3.57570544e01, 1.53555516e01),
        ),
        (
            (0.0, -1.35818573e02, 0.0, -1.82452234e01),
            (2.03450770e02, 8.18248325e02, -4.35345536e02, 9.20983561e01),
            (-1.83835664e02, -8.86211351e02, 3.47168913e02, -8.26009443e01),
        ),
        (
            (-7.13247650e01, 8.83374450e02, 2.99182150e02, 0.0),
            (-6.59883962e02, -5.26052382e03, 1.27778712e03, -3.64839255e01),
            (4.36853250e02, 5.67490957e03, -8.06783834e02, 0.0),
        ),
        (
            (2.33819330e02, -1.69758341e03, -9.23263240e02, 0.0),
            (5.76375582e02, 1.00342283e04, -8.79147687e02, 0.0),
            (0.0, -1.07373527e04, 0.0, 0.0),
        ),
    ),
)


@dataclass(frozen=True)
class PNRSingleSpin:
    """Effective single-spin quantities used by the PNR calibration."""

    mass_ratio: torch.Tensor
    symmetric_mass_ratio: torch.Tensor
    magnitude: torch.Tensor
    cosine: torch.Tensor
    antisymmetric_magnitude: torch.Tensor
    antisymmetric_angle: torch.Tensor
    final_cosine: torch.Tensor


@dataclass(frozen=True)
class PNRAlphaParameters:
    """Frequency-domain coefficients for the tuned PNR alpha angle."""

    a1: torch.Tensor
    a2: torch.Tensor
    a3: torch.Tensor
    a4: torch.Tensor
    mf_lower: torch.Tensor
    mf_upper: torch.Tensor
    interp0: torch.Tensor
    interp1: torch.Tensor
    interp2: torch.Tensor
    interp3: torch.Tensor
    mr_offset: torch.Tensor


@dataclass(frozen=True)
class PNRBetaParameters:
    """Merger-ringdown coefficients and connection data for PNR beta."""

    b0: torch.Tensor
    b1: torch.Tensor
    b2: torch.Tensor
    b3: torch.Tensor
    b4: torch.Tensor
    b5: torch.Tensor
    mf_lower: torch.Tensor
    mf_upper: torch.Tensor
    beta_lower: torch.Tensor
    beta_upper: torch.Tensor
    derivative_lower: torch.Tensor
    derivative_upper: torch.Tensor
    rescale1: torch.Tensor
    rescale2: torch.Tensor


@dataclass(frozen=True)
class PNRSpinTaylorBetaParameters:
    """PNR beta data connected to numerical SpinTaylor angles."""

    merger: PNRBetaParameters
    mf_interpolation_start: torch.Tensor
    interp0: torch.Tensor
    interp1: torch.Tensor
    interp2: torch.Tensor
    interp3: torch.Tensor


@dataclass(frozen=True)
class PNRSpinTaylorRemnant:
    """Evolved-spin remnant data used by numerical version-330 angles."""

    final_spin: torch.Tensor
    radiated_energy: torch.Tensor
    final_mass: torch.Tensor
    ringdown_frequency: torch.Tensor
    damping_frequency: torch.Tensor
    damping_difference: torch.Tensor


@dataclass(frozen=True)
class PNRSpinTaylorIntegration:
    """Frequency controls for the version-330 SpinTaylor trajectory."""

    starting_frequency: torch.Tensor
    trajectory_minimum_frequency: torch.Tensor
    interpolation_delta_f: torch.Tensor
    integration_buffer: torch.Tensor


@dataclass(frozen=True)
class PNRSpinTaylorAngleModel:
    """Complete scalar setup for version-330 PNR angle evaluation."""

    mass1: float
    mass2: float
    spin1: tuple[float, float, float]
    spin2: tuple[float, float, float]
    inclination: float
    reference_frequency: float
    total_mass_seconds: float
    integration: PNRSpinTaylorIntegration
    trajectory: SpinTaylorTrajectory
    frame: SpinTaylorJFrame
    remnant: PNRSpinTaylorRemnant
    single_spin: PNRSingleSpin
    msa_state: dict
    single_spin_msa_state: dict | None
    angles: SpinTaylorAngleSpline
    alpha_parameters: PNRAlphaParameters
    beta_parameters: PNRSpinTaylorBetaParameters
    alpha_offset: torch.Tensor


@dataclass(frozen=True)
class PNRSpinTaylorAngles:
    """Uniform-frequency version-330 PNR angles and reference data."""

    frequencies: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    gamma: torch.Tensor
    alpha_reference: torch.Tensor
    beta_reference: torch.Tensor
    gamma_reference: torch.Tensor
    source_frame: PNRSourceFrame
    model: PNRSpinTaylorAngleModel


@dataclass(frozen=True)
class PNRCoprecessingFits:
    """Calibrated `(2, 2)` co-precessing waveform-deviation fits."""

    mu1: torch.Tensor
    mu2: torch.Tensor
    mu3: torch.Tensor
    nu0: torch.Tensor
    nu4: torch.Tensor
    nu5: torch.Tensor
    nu6: torch.Tensor
    zeta1: torch.Tensor
    zeta2: torch.Tensor


@dataclass(frozen=True)
class PNRCoprecessingDeviations:
    """Tuned co-precessing fits and their common waveform scale."""

    strength: torch.Tensor
    fits: PNRCoprecessingFits


def _as_common_tensors(*values):
    reference = next(
        (value for value in values if isinstance(value, torch.Tensor)),
        None,
    )
    if reference is None:
        device = torch.device("cpu")
        dtype = torch.float64
    else:
        device = reference.device
        dtype = reference.dtype if reference.dtype.is_floating_point else torch.float64
    return tuple(torch.as_tensor(value, device=device, dtype=dtype) for value in values)


def _safe_ratio(numerator, denominator, *, threshold=0.0):
    safe_denominator = torch.where(
        torch.abs(denominator) > threshold,
        denominator,
        torch.ones_like(denominator),
    )
    return torch.where(
        torch.abs(denominator) > threshold,
        numerator / safe_denominator,
        torch.zeros_like(numerator),
    )


def pnr_higher_mode_transition_frequencies(
    mprime,
    pnr_mf_low,
    pnr_mf_high,
    mf_ring_22,
    mf_ring_lm,
):
    """Return LAL's transition frequencies for mapped higher-mode angles.

    The PNR angles are calibrated for the co-precessing ``(2, 2)`` mode.
    Higher modes use inspiral ``2 f / m'`` scaling below the first transition
    and a ringdown-frequency shift above the second one.  This helper returns
    the two geometric frequencies joining those regions.
    """

    if (
        not isinstance(mprime, Integral)
        or isinstance(mprime, bool)
        or mprime <= 0
    ):
        raise ValueError("PNR higher-mode mprime must be a positive integer")
    pnr_mf_low, pnr_mf_high, mf_ring_22, mf_ring_lm = _as_common_tensors(
        pnr_mf_low,
        pnr_mf_high,
        mf_ring_22,
        mf_ring_lm,
    )
    ringdown_difference = mf_ring_lm - mf_ring_22
    lower = 0.65 * pnr_mf_low * int(mprime) / 2.0
    upper = 1.1 * (pnr_mf_high + ringdown_difference)
    invalid_upper = (upper < 0.0) | (
        (ringdown_difference < 0.0) & (upper < 0.5 * pnr_mf_high)
    )
    return lower, torch.where(invalid_upper, pnr_mf_high, upper)


def pnr_higher_mode_frequency_map(
    mf,
    ell,
    mprime,
    mf_lower,
    mf_upper,
    mf_ring_22,
    mf_ring_lm,
    *,
    inspiral_only=False,
):
    """Map a higher-mode frequency onto the tuned ``(2, 2)`` PNR angles."""

    for name, value in (("ell", ell), ("mprime", mprime)):
        if (
            not isinstance(value, Integral)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(
                f"PNR higher-mode {name} must be a positive integer"
            )
    mf, mf_lower, mf_upper, mf_ring_22, mf_ring_lm = _as_common_tensors(
        mf,
        mf_lower,
        mf_upper,
        mf_ring_22,
        mf_ring_lm,
    )
    if (int(ell), int(mprime)) == (2, 2):
        return mf

    inspiral = 2.0 * mf / int(mprime)
    if inspiral_only:
        return inspiral
    if bool(torch.any(mf_lower >= mf_upper)):
        raise ValueError(
            "PNR higher-mode transition frequencies must be strictly ordered"
        )

    ringdown_difference = mf_ring_lm - mf_ring_22
    slope = (
        mf_upper
        - ringdown_difference
        - 2.0 * mf_lower / int(mprime)
    ) / (mf_upper - mf_lower)
    transition = slope * (mf - mf_lower) + 2.0 * mf_lower / int(mprime)
    ringdown = mf - ringdown_difference
    return torch.where(
        mf <= mf_lower,
        inspiral,
        torch.where(mf > mf_upper, ringdown, transition),
    )


def _pnr_effective_precession_spin(values):
    """Return LAL's source-frame effective precession spin ``chi_p``."""

    mass1, mass2, *spin_components = values
    spin1 = spin_components[:3]
    spin2 = spin_components[3:]
    swap = mass2 > mass1
    larger_mass = torch.where(swap, mass2, mass1)
    smaller_mass = torch.where(swap, mass1, mass2)
    larger_spin = tuple(
        torch.where(swap, second, first) for first, second in zip(spin1, spin2)
    )
    smaller_spin = tuple(
        torch.where(swap, first, second) for first, second in zip(spin1, spin2)
    )
    mass_ratio = larger_mass / smaller_mass
    inverse_ratio = 1.0 / mass_ratio
    spin_weight1 = 2.0 + 1.5 * inverse_ratio
    spin_weight2 = 2.0 + 1.5 * mass_ratio
    return torch.maximum(
        torch.hypot(larger_spin[0], larger_spin[1]),
        spin_weight2
        * inverse_ratio.square()
        * torch.hypot(smaller_spin[0], smaller_spin[1])
        / spin_weight1,
    )


def _pnr_single_spin_mapping(
    values,
    *,
    effective_precession_spin=None,
    use_evolved_perpendicular=False,
):
    """Build PNR's effective spin from common-device source tensors."""

    mass1, mass2, *spin_components = values
    spin1 = spin_components[:3]
    spin2 = spin_components[3:]
    swap = mass2 > mass1
    larger_mass = torch.where(swap, mass2, mass1)
    smaller_mass = torch.where(swap, mass1, mass2)
    larger_spin = tuple(
        torch.where(swap, second, first) for first, second in zip(spin1, spin2)
    )
    smaller_spin = tuple(
        torch.where(swap, first, second) for first, second in zip(spin1, spin2)
    )

    mass_ratio = larger_mass / smaller_mass
    inverse_ratio = 1.0 / mass_ratio
    total_mass = larger_mass + smaller_mass
    symmetric_mass_ratio = larger_mass * smaller_mass / (total_mass * total_mass)
    if effective_precession_spin is None:
        chi_p = _pnr_effective_precession_spin(values)
    else:
        chi_p = torch.as_tensor(
            effective_precession_spin,
            dtype=larger_mass.dtype,
            device=larger_mass.device,
        )

    parallel = larger_spin[2] + inverse_ratio * smaller_spin[2]
    inverse_ratio2 = inverse_ratio * inverse_ratio
    symmetric_perp = torch.hypot(
        larger_spin[0] + inverse_ratio2 * smaller_spin[0],
        larger_spin[1] + inverse_ratio2 * smaller_spin[1],
    )
    antisymmetric_perp = torch.hypot(
        larger_spin[0] - inverse_ratio2 * smaller_spin[0],
        larger_spin[1] - inverse_ratio2 * smaller_spin[1],
    )
    transition_phase = math.pi * (mass_ratio - 1.0)
    chi_p_weight = torch.sin(transition_phase) ** 2
    vector_weight = torch.cos(transition_phase) ** 2
    calibrated = mass_ratio <= 1.5
    if use_evolved_perpendicular:
        # Version 330 uses the evolved symmetric in-plane spin at every q.
        perpendicular = symmetric_perp
    else:
        perpendicular = torch.where(
            calibrated,
            chi_p_weight * chi_p + vector_weight * symmetric_perp,
            chi_p,
        )
    antisymmetric_perpendicular = torch.where(
        calibrated,
        chi_p_weight * chi_p + vector_weight * antisymmetric_perp,
        chi_p,
    )

    magnitude = torch.hypot(parallel, perpendicular)
    cosine = _safe_ratio(parallel, magnitude, threshold=1.0e-6)
    antisymmetric_magnitude = torch.hypot(parallel, antisymmetric_perpendicular)
    antisymmetric_cosine = torch.clamp(
        _safe_ratio(
            parallel,
            antisymmetric_magnitude,
            threshold=1.0e-6,
        ),
        -1.0,
        1.0,
    )
    antisymmetric_angle = torch.where(
        antisymmetric_magnitude >= 1.0e-6,
        torch.acos(antisymmetric_cosine),
        torch.zeros_like(antisymmetric_magnitude),
    )

    aligned_final_spin = final_spin_2017(
        symmetric_mass_ratio,
        magnitude * cosine,
        torch.zeros_like(magnitude),
    )
    larger_mass_fraction = mass_ratio / (1.0 + mass_ratio)
    final_perpendicular = (
        larger_mass_fraction
        * larger_mass_fraction
        * magnitude
        * torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0))
    )
    final_magnitude = torch.hypot(aligned_final_spin, final_perpendicular)
    final_cosine = _safe_ratio(
        aligned_final_spin,
        final_magnitude,
        threshold=1.0e-6,
    )
    return PNRSingleSpin(
        mass_ratio=mass_ratio,
        symmetric_mass_ratio=symmetric_mass_ratio,
        magnitude=magnitude,
        cosine=cosine,
        antisymmetric_magnitude=antisymmetric_magnitude,
        antisymmetric_angle=antisymmetric_angle,
        final_cosine=final_cosine,
    )


def pnr_single_spin_mapping(
    mass1,
    mass2,
    spin1x,
    spin1y,
    spin1z,
    spin2x,
    spin2y,
    spin2z,
):
    """Map generic two-spin parameters to LAL's PNR effective spin."""

    return _pnr_single_spin_mapping(
        _as_common_tensors(
            mass1,
            mass2,
            spin1x,
            spin1y,
            spin1z,
            spin2x,
            spin2y,
            spin2z,
        )
    )


def pnr_spintaylor_evolved_spins(trajectory, mass1, mass2):
    """Return version-330 endpoint spins in the final orbital frame.

    SpinTaylor stores component spins normalized by total mass.  LAL recovers
    the dimensionless component spins at the final accepted state and rotates
    them so that the evolved orbital direction is the positive z axis before
    constructing the PNR single-spin fit parameters.
    """

    state = trajectory.state
    if state.ndim != 2 or state.shape[0] == 0 or state.shape[1] != 14:
        raise ValueError("SpinTaylor evolved-spin mapping requires a trajectory")
    if not state.dtype.is_floating_point:
        raise ValueError("SpinTaylor trajectory state must be floating point")

    mass1 = torch.as_tensor(mass1, dtype=state.dtype, device=state.device)
    mass2 = torch.as_tensor(mass2, dtype=state.dtype, device=state.device)
    if mass1.numel() != 1 or mass2.numel() != 1:
        raise ValueError("SpinTaylor evolved-spin masses must be scalar")
    mass1 = mass1.reshape(())
    mass2 = mass2.reshape(())
    masses = torch.stack((mass1, mass2))
    if (
        not bool(torch.all(torch.isfinite(masses)).detach().cpu())
        or float(torch.minimum(mass1, mass2).detach().cpu()) <= 0.0
    ):
        raise ValueError("SpinTaylor evolved-spin masses must be finite and positive")

    endpoint = state[-1]
    lnhat = endpoint[2:5]
    lnorm = torch.linalg.vector_norm(lnhat)
    if not bool(torch.isfinite(lnorm).detach().cpu()) or float(
        lnorm.detach().cpu()
    ) <= 0.0:
        raise ValueError("SpinTaylor final orbital direction must be finite and nonzero")

    total_mass = mass1 + mass2
    spin1 = endpoint[5:8] / (mass1 / total_mass).square()
    spin2 = endpoint[8:11] / (mass2 / total_mass).square()
    azimuth = torch.atan2(lnhat[1], lnhat[0])
    polar = torch.acos(torch.clamp(lnhat[2] / lnorm, -1.0, 1.0))

    def rotate_to_orbital_frame(spin):
        cosine = torch.cos(azimuth)
        sine = torch.sin(azimuth)
        x = spin[0] * cosine + spin[1] * sine
        y = -spin[0] * sine + spin[1] * cosine
        z = spin[2]
        cosine = torch.cos(polar)
        sine = torch.sin(polar)
        return torch.stack(
            (x * cosine - z * sine, y, x * sine + z * cosine)
        )

    return rotate_to_orbital_frame(spin1), rotate_to_orbital_frame(spin2)


def pnr_spintaylor_single_spin_mapping(
    trajectory,
    mass1,
    mass2,
    spin1x,
    spin1y,
    spin1z,
    spin2x,
    spin2y,
    spin2z,
):
    """Map a version-330 SpinTaylor endpoint to PNR's effective spin."""

    initial_values = _as_common_tensors(
        mass1,
        mass2,
        spin1x,
        spin1y,
        spin1z,
        spin2x,
        spin2y,
        spin2z,
    )
    initial_chi_p = _pnr_effective_precession_spin(initial_values)
    evolved_spin1, evolved_spin2 = pnr_spintaylor_evolved_spins(
        trajectory,
        mass1,
        mass2,
    )
    return _pnr_single_spin_mapping(
        _as_common_tensors(
            mass1,
            mass2,
            *evolved_spin1,
            *evolved_spin2,
        ),
        effective_precession_spin=initial_chi_p,
        use_evolved_perpendicular=True,
    )


def _cosine_taper(argument):
    return torch.where(
        argument > 1.0,
        torch.zeros_like(argument),
        torch.where(
            argument > 0.0,
            0.5 * torch.cos(math.pi * argument) + 0.5,
            torch.ones_like(argument),
        ),
    )


def pnr_angles_window(mass_ratio, spin_magnitude):
    """Return the PNR-angle calibration taper."""

    mass_ratio, spin_magnitude = _as_common_tensors(mass_ratio, spin_magnitude)
    return _cosine_taper((mass_ratio - 8.5) / 3.5) * _cosine_taper(
        (spin_magnitude - 0.85) / 0.35
    )


def pnr_coprecessing_window(mass_ratio):
    """Return the mass-ratio taper for tuned co-precessing deviations."""

    (mass_ratio,) = _as_common_tensors(mass_ratio)
    return _cosine_taper((mass_ratio - 10.0) / 10.0)


def pnr_coprecessing_fits(theta, symmetric_mass_ratio, spin_magnitude):
    """Evaluate the tuned `(2, 2)` co-precessing deviation fits."""

    theta, symmetric_mass_ratio, spin_magnitude = _as_common_tensors(
        theta,
        symmetric_mass_ratio,
        spin_magnitude,
    )
    cosine = torch.cos(theta)
    eta_powers = torch.stack(
        (
            torch.ones_like(symmetric_mass_ratio),
            symmetric_mass_ratio,
            symmetric_mass_ratio**2,
            symmetric_mass_ratio**3,
        ),
        dim=-1,
    )
    spin_powers = torch.stack(
        (
            torch.ones_like(spin_magnitude),
            spin_magnitude,
            spin_magnitude**2,
        ),
        dim=-1,
    )
    cosine_powers = torch.stack(
        (
            torch.ones_like(cosine),
            cosine,
            cosine**2,
            cosine**3,
        ),
        dim=-1,
    )
    coefficients = symmetric_mass_ratio.new_tensor(
        _COPRECESSING_FIT_COEFFICIENTS
    )
    values = torch.einsum(
        "...i,...j,...k,nijk->...n",
        eta_powers,
        spin_powers,
        cosine_powers,
        coefficients,
    )
    return PNRCoprecessingFits(*values.unbind(dim=-1))


def build_pnr_coprecessing_deviations(single_spin, *, prec_version):
    """Build LAL-compatible tuned co-precessing deviation parameters."""

    eta = single_spin.symmetric_mass_ratio
    spin_magnitude = single_spin.magnitude
    if prec_version == 330:
        fitted_eta = torch.where(
            eta >= 0.09876,
            eta,
            0.09876 - (0.09876 - eta) * 0.1641,
        )
        fitted_spin = torch.where(
            spin_magnitude <= 0.8,
            spin_magnitude,
            0.8 + (spin_magnitude - 0.8) / 12.0,
        )
        fitted_spin = torch.clamp(fitted_spin, min=0.2)
    else:
        fitted_eta = torch.clamp(eta, min=0.09876)
        fitted_spin = torch.clamp(spin_magnitude, min=0.2, max=0.8)

    cosine = torch.clamp(single_spin.cosine, -1.0, 1.0)
    sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0))
    fits = pnr_coprecessing_fits(
        torch.acos(cosine),
        fitted_eta,
        fitted_spin,
    )
    strength = (
        pnr_coprecessing_window(single_spin.mass_ratio)
        * spin_magnitude
        * sine
    )
    return PNRCoprecessingDeviations(strength=strength, fits=fits)


def _evaluate_coefficient_array(coefficients, eta, chi, cosine):
    coefficients = eta.new_tensor(coefficients)
    eta_powers = torch.stack((torch.ones_like(eta), eta, eta**2, eta**3), dim=-1)
    chi_powers = torch.stack((torch.ones_like(chi), chi, chi**2, chi**3), dim=-1)
    cosine_powers = torch.stack(
        (
            torch.ones_like(cosine),
            cosine,
            cosine**2,
            cosine**3,
            cosine**4,
        ),
        dim=-1,
    )
    return torch.einsum(
        "...i,...j,...k,ijk->...",
        eta_powers,
        chi_powers,
        cosine_powers,
        coefficients,
    )


def _pnr_fit_inputs(single_spin, prec_version):
    eta = single_spin.symmetric_mass_ratio
    chi = single_spin.magnitude
    if prec_version == 330:
        eta = torch.clamp(eta, min=0.09)
        spin_boundary = 0.80 - 0.20 * torch.exp(
            -((single_spin.mass_ratio - 6.0) / 1.5) ** 8
        )
        chi = torch.minimum(chi, spin_boundary)
    return eta, chi


def _pnr_alpha_fit_coefficients(single_spin, prec_version):
    eta, chi = _pnr_fit_inputs(single_spin, prec_version)
    cosine = single_spin.cosine
    sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0))

    alpha1 = _evaluate_coefficient_array(
        _ALPHA_A1_COEFFICIENTS,
        eta,
        chi,
        cosine,
    )
    alpha2 = _evaluate_coefficient_array(
        _ALPHA_A2_COEFFICIENTS,
        eta,
        chi,
        cosine,
    )
    alpha3 = _evaluate_coefficient_array(
        _ALPHA_A3_COEFFICIENTS,
        eta,
        chi,
        cosine,
    )
    alpha4 = _evaluate_coefficient_array(
        _ALPHA_A4_COEFFICIENTS,
        eta,
        chi,
        cosine,
    )

    a1 = torch.abs(chi * sine * alpha1 * alpha1)
    a2 = -chi * sine * alpha2 * alpha2
    a2 = torch.minimum(a2, torch.zeros_like(a2))
    a3 = torch.clamp(torch.abs(alpha3 * alpha3), min=1.0e-5)
    a2 = torch.maximum(a2, -(math.pi**2) * torch.sqrt(a3))
    return a1, a2, a3, alpha4 * alpha4


def _pn_alpha(mf, msa_state):
    velocity = torch.pow(math.pi * mf, 1.0 / 3.0)
    return msa_angles(velocity, msa_state)[0]


def _mr_alpha(mf, a1, a2, a3, a4):
    return -(a1 / mf + a2 * torch.sqrt(a3) / (a3 + (mf - a4) * (mf - a4)))


def _alpha_interpolation_coefficients(mf1, mf2, alpha1, alpha2, dalpha1, dalpha2):
    difference = mf1 - mf2
    denominator = (mf2 - mf1) ** 3
    mf1_boundary = mf1 * dalpha1 + alpha1
    mf2_boundary = mf2 * dalpha2 + alpha2

    interp0 = (
        2.0 * (mf1 * alpha1 - mf2 * alpha2)
        - difference * (mf1 * dalpha1 + mf2 * dalpha2 + alpha1 + alpha2)
    ) / denominator
    interp1 = (
        3.0 * (mf1 + mf2) * (mf2 * alpha2 - mf1 * alpha1)
        + difference
        * ((mf1 + 2.0 * mf2) * mf1_boundary + (2.0 * mf1 + mf2) * mf2_boundary)
    ) / denominator
    interp2 = (
        6.0 * mf1 * mf2 * (mf1 * alpha1 - mf2 * alpha2)
        - difference
        * (
            mf2 * (2.0 * mf1 + mf2) * mf1_boundary
            + mf1 * (mf1 + 2.0 * mf2) * mf2_boundary
        )
    ) / denominator
    interp3 = (
        mf1 * mf2 * mf2 * (mf2 - 3.0 * mf1) * alpha1
        - mf1 * mf1 * mf2 * (mf1 - 3.0 * mf2) * alpha2
        + mf1 * mf2 * difference * (mf2 * mf1_boundary + mf1 * mf2_boundary)
    ) / denominator
    return tuple(
        torch.where(torch.isnan(value), torch.zeros_like(value), value)
        for value in (interp0, interp1, interp2, interp3)
    )


def _build_pnr_alpha_parameters(
    single_spin,
    total_mass_seconds,
    pn_alpha,
    *,
    prec_version,
    mf_min_integration=None,
):
    a1, a2, a3, a4 = _pnr_alpha_fit_coefficients(single_spin, prec_version)
    total_mass_seconds = torch.as_tensor(
        total_mass_seconds,
        device=a1.device,
        dtype=a1.dtype,
    )
    derivative_step = a1.new_tensor(0.0005)
    mf_upper = a4 / 3.0
    mf_lower = (3.0 / 3.5) * mf_upper
    disabled = mf_upper < 2.0 * total_mass_seconds
    if mf_min_integration is not None:
        mf_min_integration = torch.as_tensor(
            mf_min_integration,
            device=a1.device,
            dtype=a1.dtype,
        )
        disabled = disabled | (
            mf_lower - 2.0 * derivative_step < mf_min_integration
        )
    mf_lower = torch.where(disabled, mf_lower.new_tensor(100.0), mf_lower)
    mf_upper = torch.where(disabled, mf_upper.new_tensor(100.0), mf_upper)

    alpha_lower = pn_alpha(mf_lower)
    alpha_upper = pn_alpha(mf_upper)
    derivative_lower = (
        pn_alpha(mf_lower + derivative_step)
        - pn_alpha(mf_lower - derivative_step)
    ) / (2.0 * derivative_step)
    derivative_upper = (
        _mr_alpha(mf_upper + derivative_step, a1, a2, a3, a4)
        - _mr_alpha(mf_upper - derivative_step, a1, a2, a3, a4)
    ) / (2.0 * derivative_step)
    interp0, interp1, interp2, interp3 = _alpha_interpolation_coefficients(
        mf_lower,
        mf_upper,
        alpha_lower,
        alpha_upper,
        derivative_lower,
        derivative_upper,
    )
    mr_offset = alpha_upper - _mr_alpha(mf_upper, a1, a2, a3, a4)
    return PNRAlphaParameters(
        a1=a1,
        a2=a2,
        a3=a3,
        a4=a4,
        mf_lower=mf_lower,
        mf_upper=mf_upper,
        interp0=interp0,
        interp1=interp1,
        interp2=interp2,
        interp3=interp3,
        mr_offset=mr_offset,
    )


def build_pnr_alpha_parameters(single_spin, msa_state, total_mass_seconds):
    """Build the tuned-alpha connection data for the MSA-223 prescription."""

    return _build_pnr_alpha_parameters(
        single_spin,
        total_mass_seconds,
        lambda mf: _pn_alpha(mf, msa_state),
        prec_version=223,
    )


def build_pnr_spintaylor_alpha_parameters(
    single_spin,
    angles,
    total_mass_seconds,
    *,
    alpha_offset=0.0,
):
    """Build tuned-alpha connection data from numerical SpinTaylor angles."""

    return _build_pnr_alpha_parameters(
        single_spin,
        total_mass_seconds,
        lambda mf: spintaylor_alpha_imr(mf, angles, offset=alpha_offset),
        prec_version=330,
        mf_min_integration=angles.mf[0],
    )


def _pnr_alpha(mf, parameters, single_spin, pn_alpha):
    mf = torch.as_tensor(
        mf,
        device=parameters.a1.device,
        dtype=parameters.a1.dtype,
    )
    inspiral_alpha = pn_alpha(mf)
    mr_alpha = (
        _mr_alpha(
            mf,
            parameters.a1,
            parameters.a2,
            parameters.a3,
            parameters.a4,
        )
        + parameters.mr_offset
    )
    intermediate_alpha = (
        parameters.interp0 * mf * mf
        + parameters.interp1 * mf
        + parameters.interp2
        + parameters.interp3 / mf
    )
    tuned_alpha = torch.where(
        mf <= parameters.mf_lower,
        inspiral_alpha,
        torch.where(mf >= parameters.mf_upper, mr_alpha, intermediate_alpha),
    )
    window = pnr_angles_window(single_spin.mass_ratio, single_spin.magnitude)
    return window * tuned_alpha + (1.0 - window) * inspiral_alpha


def pnr_alpha(mf, parameters, single_spin, msa_state):
    """Evaluate the calibrated MSA-223 PNR alpha angle."""

    return _pnr_alpha(
        mf,
        parameters,
        single_spin,
        lambda frequencies: _pn_alpha(frequencies, msa_state),
    )


def pnr_spintaylor_alpha(
    mf,
    parameters,
    single_spin,
    angles,
    *,
    alpha_offset=0.0,
):
    """Evaluate calibrated PNR alpha from numerical SpinTaylor angles."""

    return _pnr_alpha(
        mf,
        parameters,
        single_spin,
        lambda frequencies: spintaylor_alpha_imr(
            frequencies,
            angles,
            offset=alpha_offset,
        ),
    )


def _pnr_beta_fit_coefficients(single_spin, prec_version):
    eta, chi = _pnr_fit_inputs(single_spin, prec_version)
    cosine = single_spin.cosine
    sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0))

    polynomial0 = _evaluate_coefficient_array(
        _BETA_B0_COEFFICIENTS,
        eta,
        chi,
        cosine,
    )
    polynomial1 = _evaluate_coefficient_array(
        _BETA_B1_COEFFICIENTS,
        eta,
        chi,
        cosine,
    )
    polynomial2 = _evaluate_coefficient_array(
        _BETA_B2_COEFFICIENTS,
        eta,
        chi,
        cosine,
    )
    polynomial3 = _evaluate_coefficient_array(
        _BETA_B3_COEFFICIENTS,
        eta,
        chi,
        cosine,
    )
    polynomial4 = _evaluate_coefficient_array(
        _BETA_B4_COEFFICIENTS,
        eta,
        chi,
        cosine,
    )
    polynomial5 = _evaluate_coefficient_array(
        _BETA_B5_COEFFICIENTS,
        eta,
        chi,
        cosine,
    )

    b0 = torch.acos(torch.clamp(single_spin.final_cosine, -1.0, 1.0))
    b0 -= chi * sine * polynomial0
    b1 = chi * sine * torch.exp(polynomial1)
    b2 = -chi * sine * torch.exp(polynomial2)
    b3 = b2 * polynomial3
    b4 = torch.clamp(polynomial4 * polynomial4, min=175.0)
    b5 = -(polynomial5 * polynomial5)
    return b0, b1, b2, b3, b4, b5


def pnr_mr_beta(mf, parameters):
    """Evaluate the PNR merger-ringdown beta ansatz."""

    mf = torch.as_tensor(
        mf,
        device=parameters.b0.device,
        dtype=parameters.b0.dtype,
    )
    shifted = mf + parameters.b5
    numerator = parameters.b1 + parameters.b2 * mf + parameters.b3 * mf * mf
    return parameters.b0 + numerator / (1.0 + parameters.b4 * shifted * shifted)


def _pnr_mr_beta_derivative(mf, parameters):
    b1 = parameters.b1
    b2 = parameters.b2
    b3 = parameters.b3
    b4 = parameters.b4
    b5 = parameters.b5
    numerator = (
        (2.0 * b3 * b4 * b5 - b2 * b4) * mf * mf
        + (2.0 * b3 - 2.0 * b1 * b4 + 2.0 * b3 * b4 * b5 * b5) * mf
        + b2
        - 2.0 * b1 * b4 * b5
        + b2 * b4 * b5 * b5
    )
    return numerator / (1.0 + b4 * (mf + b5) ** 2) ** 2


def _pnr_mr_beta_second_derivative(mf, parameters):
    b1 = parameters.b1
    b2 = parameters.b2
    b3 = parameters.b3
    b4 = parameters.b4
    b5 = parameters.b5
    a = b2 * b4 * b4 - 2.0 * b3 * b4 * b4 * b5
    b = -3.0 * b3 * b4 + 3.0 * b1 * b4 * b4 - 3.0 * b3 * b4 * b4 * b5 * b5
    c = -3.0 * b2 * b4 + 6.0 * b1 * b4 * b4 * b5 - 3.0 * b2 * b4 * b4 * b5 * b5
    d = (
        b3
        - b1 * b4
        - 2.0 * b2 * b4 * b5
        + 2.0 * b3 * b4 * b5 * b5
        + 3.0 * b1 * b4 * b4 * b5 * b5
        - 2.0 * b2 * b4 * b4 * b5**3
        + b3 * b4 * b4 * b5**4
    )
    return 2.0 * (a * mf**3 + b * mf * mf + c * mf + d) / (
        1.0 + b4 * (b5 + mf) ** 2
    ) ** 3


def _pnr_mr_beta_third_derivative(mf, parameters):
    b1 = parameters.b1
    b2 = parameters.b2
    b3 = parameters.b3
    b4 = parameters.b4
    b5 = parameters.b5
    a = -b2 * b4 * b4 + 2.0 * b3 * b4 * b4 * b5
    b = 4.0 * b3 * b4 - 4.0 * b1 * b4 * b4 + 4.0 * b3 * b4 * b4 * b5 * b5
    c = 6.0 * b2 * b4 - 12.0 * b1 * b4 * b4 * b5 + 6.0 * b2 * b4 * b4 * b5 * b5
    d = (
        -4.0 * b3
        + 4.0 * b1 * b4
        + 8.0 * b2 * b4 * b5
        - 8.0 * b3 * b4 * b5 * b5
        - 12.0 * b1 * b4 * b4 * b5 * b5
        + 8.0 * b2 * b4 * b4 * b5**3
        - 4.0 * b3 * b4 * b4 * b5**4
    )
    e = (
        -b2
        - 2.0 * b3 * b5
        + 4.0 * b1 * b4 * b5
        + 2.0 * b2 * b4 * b5 * b5
        - 4.0 * b3 * b4 * b5**3
        - 4.0 * b1 * b4 * b4 * b5**3
        + 3.0 * b2 * b4 * b4 * b5**4
        - 2.0 * b3 * b4 * b4 * b5**5
    )
    return 6.0 * b4 * (a * mf**4 + b * mf**3 + c * mf * mf + d * mf + e) / (
        1.0 + b4 * (b5 + mf) ** 2
    ) ** 4


def _real_cuberoot(value):
    return torch.sign(value) * torch.abs(value).pow(1.0 / 3.0)


def _pnr_beta_inflection_frequency(parameters):
    b1 = parameters.b1
    b2 = parameters.b2
    b3 = parameters.b3
    b4 = parameters.b4
    b5 = parameters.b5
    a = 2.0 * (b2 * b4 * b4 - 2.0 * b3 * b4 * b4 * b5)
    b = 6.0 * (-b3 * b4 + b1 * b4 * b4 - b3 * b4 * b4 * b5 * b5)
    c = 6.0 * (-b2 * b4 + 2.0 * b1 * b4 * b4 * b5 - b2 * b4 * b4 * b5 * b5)
    d = 2.0 * (
        b3
        - b1 * b4
        - 2.0 * b2 * b4 * b5
        + 2.0 * b3 * b4 * b5 * b5
        + 3.0 * b1 * b4 * b4 * b5 * b5
        - 2.0 * b2 * b4 * b4 * b5**3
        + b3 * b4 * b4 * b5**4
    )

    safe_b = torch.where(torch.abs(b) > 0.0, b, torch.ones_like(b))
    discriminant = c * c - 4.0 * b * d
    quadratic_delta = torch.sqrt(torch.clamp(discriminant, min=0.0))
    quadratic_plus = (-c + quadratic_delta) / (2.0 * safe_b)
    quadratic_minus = (-c - quadratic_delta) / (2.0 * safe_b)
    quadratic_root = torch.zeros_like(a)
    quadratic_root = torch.where(
        _pnr_mr_beta_derivative(quadratic_plus, parameters) < 0.0,
        quadratic_plus,
        quadratic_root,
    )
    quadratic_root = torch.where(
        _pnr_mr_beta_derivative(quadratic_minus, parameters) < 0.0,
        quadratic_minus,
        quadratic_root,
    )

    safe_a = torch.where(torch.abs(a) > 0.0, a, torch.ones_like(a))
    depressed_p = (3.0 * a * c - b * b) / (3.0 * safe_a * safe_a)
    depressed_q = (
        2.0 * b**3 - 9.0 * a * b * c + 27.0 * a * a * d
    ) / (27.0 * safe_a**3)
    cubic_discriminant = (depressed_q / 2.0) ** 2 + (depressed_p / 3.0) ** 3
    discriminant_root = torch.sqrt(torch.clamp(cubic_discriminant, min=0.0))
    single_root = (
        _real_cuberoot(-depressed_q / 2.0 + discriminant_root)
        + _real_cuberoot(-depressed_q / 2.0 - discriminant_root)
        - b / (3.0 * safe_a)
    )

    safe_p = torch.where(
        torch.abs(depressed_p) > 0.0,
        depressed_p,
        -torch.ones_like(depressed_p),
    )
    acos_argument = (3.0 * depressed_q / (2.0 * safe_p)) * torch.sqrt(
        torch.clamp(-3.0 / safe_p, min=0.0)
    )
    phi = torch.acos(torch.clamp(acos_argument, -1.0, 1.0))
    amplitude = 2.0 * torch.sqrt(torch.clamp(-depressed_p / 3.0, min=0.0))
    shift = b / (3.0 * safe_a)
    root0 = amplitude * torch.cos(phi / 3.0) - shift
    root1 = amplitude * torch.cos((phi - 2.0 * math.pi) / 3.0) - shift
    root2 = amplitude * torch.cos((phi - 4.0 * math.pi) / 3.0) - shift
    selected_three_roots = torch.where(
        a < 0.0,
        root1,
        torch.where(
            b / (3.0 * safe_a) > b5 / 2.0 - 2141.0 / 90988.0,
            root0,
            root2,
        ),
    )
    cubic_root = torch.where(
        cubic_discriminant > 0.0,
        single_root,
        selected_three_roots,
    )
    degenerate_root = torch.where(
        torch.abs(b) < 2.0e-10,
        -d / c,
        quadratic_root,
    )
    return torch.where(torch.abs(a) < 1.0e-10, degenerate_root, cubic_root)


def _pnr_beta_connection_frequencies(parameters):
    b1 = parameters.b1
    b2 = parameters.b2
    b3 = parameters.b3
    b4 = parameters.b4
    b5 = parameters.b5
    inflection = _pnr_beta_inflection_frequency(parameters)
    derivative_at_inflection = _pnr_mr_beta_derivative(inflection, parameters)

    common = b3 - b1 * b4 + b3 * b4 * b5 * b5
    root_term = (
        b4
        * (b2 - 2.0 * b3 * b5)
        * (b2 - 2.0 * b1 * b4 * b5 + b2 * b4 * b5 * b5)
        + common * common
    )
    root = torch.sqrt(root_term)
    denominator = b4 * (b2 - 2.0 * b3 * b5)
    mf_plus = (common + root) / denominator
    mf_minus = (common - root) / denominator
    plus_is_minimum = _pnr_mr_beta_second_derivative(mf_plus, parameters) > 0.0
    mf_minimum = torch.where(plus_is_minimum, mf_plus, mf_minus)
    mf_maximum = torch.where(plus_is_minimum, mf_minus, mf_plus)

    second_at_maximum = _pnr_mr_beta_second_derivative(mf_maximum, parameters)
    third_at_maximum = _pnr_mr_beta_third_derivative(mf_maximum, parameters)
    sign = torch.where(
        derivative_at_inflection > 0.0,
        torch.ones_like(derivative_at_inflection),
        -torch.ones_like(derivative_at_inflection),
    )
    chosen_derivative = sign * derivative_at_inflection * derivative_at_inflection / 400.0
    delta_root = torch.sqrt(
        second_at_maximum * second_at_maximum
        + 2.0 * third_at_maximum * chosen_derivative
    )
    delta1 = (-second_at_maximum + delta_root) / third_at_maximum
    delta2 = (-second_at_maximum - delta_root) / third_at_maximum
    delta = torch.where(
        delta1 > 0.0,
        torch.where(delta2 > 0.0, torch.minimum(delta1, delta2), delta1),
        delta2,
    )

    nominal_lower = torch.where(
        inflection >= 0.06,
        inflection - 0.03,
        3.0 * inflection / 5.0,
    )
    turnover_case = (mf_minimum > mf_maximum) | (inflection > mf_maximum)
    turnover_lower = torch.where(
        mf_maximum >= nominal_lower,
        mf_maximum + delta,
        nominal_lower,
    )
    minimum_lower = torch.where(
        mf_minimum > 0.06,
        mf_minimum - 0.03,
        3.0 * mf_minimum / 5.0,
    )
    mf_lower = torch.where(turnover_case, turnover_lower, minimum_lower)
    mf_upper = torch.where(
        mf_minimum > inflection,
        mf_minimum,
        mf_minimum.new_tensor(100.0),
    )
    disabled = (mf_lower < 0.0) | torch.isnan(mf_lower)
    mf_lower = torch.where(disabled, mf_lower.new_tensor(100.0), mf_lower)
    mf_upper = torch.where(disabled, mf_upper.new_tensor(100.0), mf_upper)
    return mf_lower, mf_upper


def build_pnr_beta_merger_parameters(single_spin, *, prec_version=223):
    """Build the PNR beta merger-ringdown ansatz and its frequencies."""

    b0, b1, b2, b3, b4, b5 = _pnr_beta_fit_coefficients(
        single_spin,
        prec_version,
    )
    zero = torch.zeros_like(b0)
    provisional = PNRBetaParameters(
        b0=b0,
        b1=b1,
        b2=b2,
        b3=b3,
        b4=b4,
        b5=b5,
        mf_lower=zero,
        mf_upper=zero,
        beta_lower=zero,
        beta_upper=zero,
        derivative_lower=zero,
        derivative_upper=zero,
        rescale1=zero,
        rescale2=zero,
    )
    mf_lower, mf_upper = _pnr_beta_connection_frequencies(provisional)
    return PNRBetaParameters(
        b0=b0,
        b1=b1,
        b2=b2,
        b3=b3,
        b4=b4,
        b5=b5,
        mf_lower=mf_lower,
        mf_upper=mf_upper,
        beta_lower=zero,
        beta_upper=zero,
        derivative_lower=zero,
        derivative_upper=zero,
        rescale1=zero,
        rescale2=zero,
    )


def _msa_has_two_spin(msa_state):
    spin1_norm = math.sqrt(
        msa_state["chi1x"] ** 2
        + msa_state["chi1y"] ** 2
        + msa_state["chi1z"] ** 2
    )
    spin2_norm = math.sqrt(
        msa_state["chi2x"] ** 2
        + msa_state["chi2y"] ** 2
        + msa_state["chi2z"] ** 2
    )
    return spin1_norm != 0.0 and spin2_norm >= 1.0e-3


def _scalar_float(value, name):
    value = torch.as_tensor(value)
    if value.numel() != 1:
        raise ValueError(f"{name} must be scalar when constructing an MSA state")
    return float(value.detach().cpu())


def build_pnr_single_spin_msa_state(single_spin, msa_state):
    """Build LAL's effective single-spin MSA-223 state when it is needed."""

    if not _msa_has_two_spin(msa_state):
        return None

    magnitude = _scalar_float(single_spin.magnitude, "single-spin magnitude")
    cosine = min(
        1.0,
        max(-1.0, _scalar_float(single_spin.cosine, "single-spin cosine")),
    )
    spin1 = (
        magnitude * math.sqrt(max(1.0 - cosine * cosine, 0.0)),
        0.0,
        magnitude * cosine,
    )
    f_ref = msa_state["v_ref"] ** 3 / (math.pi * msa_state["m_sec"])
    return build_msa_state(
        msa_state["m1"],
        msa_state["m2"],
        spin1,
        (0.0, 0.0, 0.0),
        msa_state["m_sec"],
        f_ref,
    )


def build_pnr_spintaylor_msa_state(msa_state):
    """Return an MSA state carrying LAL's version-330 3PN ``L`` fit.

    Version 330 temporarily initializes the MSA-223 dynamics needed for the
    two-spin taper, but retains the SpinTaylor 3PN orbital-angular-momentum
    coefficients used by the PNR waveform-angle map.  Keep those two roles
    separate by replacing only the cached ``L`` coefficients.
    """

    values = dict(msa_state)
    eta = values["eta"]
    delta = values["delta"]
    chi1_l = values["chi1z"]
    chi2_l = values["chi2z"]
    values.update(
        {
            "L0": 1.0,
            "L1": 0.0,
            "L2": 1.5 + eta / 6.0,
            "L3": (
                5.0
                * (
                    chi1_l * (-2.0 - 2.0 * delta + eta)
                    + chi2_l * (-2.0 + 2.0 * delta + eta)
                )
                / 6.0
            ),
            "L4": (81.0 + (-57.0 + eta) * eta) / 24.0,
            "L5": (
                -7.0
                * (
                    chi1_l
                    * (
                        72.0
                        + delta * (72.0 - 31.0 * eta)
                        + eta * (-121.0 + 2.0 * eta)
                    )
                    + chi2_l
                    * (
                        72.0
                        + eta * (-121.0 + 2.0 * eta)
                        + delta * (-72.0 + 31.0 * eta)
                    )
                )
                / 144.0
            ),
            "L6": (
                10935.0
                + eta
                * (
                    -62001.0
                    + eta * (1674.0 + 7.0 * eta)
                    + 2214.0 * math.pi**2
                )
            )
            / 1296.0,
            "L7": 0.0,
            "L8": 0.0,
            "L8L": 0.0,
        }
    )
    return values


def _lpn_orbital_angular_momentum(velocity, msa_state):
    velocity2 = velocity * velocity
    velocity4 = velocity2 * velocity2
    velocity6 = velocity4 * velocity2
    velocity8 = velocity4 * velocity4
    return (
        msa_state["eta"]
        / velocity
        * (
            msa_state["L0"]
            + msa_state["L1"] * velocity
            + msa_state["L2"] * velocity2
            + msa_state["L3"] * velocity2 * velocity
            + msa_state["L4"] * velocity4
            + msa_state["L5"] * velocity4 * velocity
            + msa_state["L6"] * velocity6
            + msa_state["L7"] * velocity6 * velocity
            + msa_state["L8"] * velocity8
            + msa_state["L8L"] * velocity8 * torch.log(velocity2)
        )
    )


def pnr_spintaylor_interpolation_delta_f(
    f_min,
    msa_state,
    *,
    output_delta_f=0.0,
    error_tolerance=0.01,
):
    """Return LAL's adaptive interpolation spacing for PNR angles in Hz."""

    f_min, output_delta_f, error_tolerance = _as_common_tensors(
        f_min,
        output_delta_f,
        error_tolerance,
    )
    if any(value.numel() != 1 for value in (f_min, output_delta_f, error_tolerance)):
        raise ValueError("PNR interpolation controls must be scalar")
    f_min, output_delta_f, error_tolerance = (
        value.reshape(()) for value in (f_min, output_delta_f, error_tolerance)
    )
    controls = torch.stack((f_min, output_delta_f, error_tolerance))
    if not bool(torch.all(torch.isfinite(controls)).detach().cpu()):
        raise ValueError("PNR interpolation controls must be finite")
    if float(f_min.detach().cpu()) <= 0.0:
        raise ValueError("PNR interpolation minimum frequency must be positive")
    if float(output_delta_f.detach().cpu()) < 0.0:
        raise ValueError("PNR output frequency spacing cannot be negative")
    if float(error_tolerance.detach().cpu()) <= 0.0:
        raise ValueError("PNR interpolation tolerance must be positive")

    aligned = msa_state["chi1_perp"] == 0.0 and msa_state["chi2_perp"] == 0.0
    if aligned:
        return torch.where(
            output_delta_f != 0.0,
            output_delta_f,
            output_delta_f.new_tensor(0.1),
        )

    total_mass_seconds = f_min.new_tensor(msa_state["m_sec"])
    eta = f_min.new_tensor(msa_state["eta"])
    mf = f_min * total_mass_seconds
    eta_term = torch.sqrt(torch.clamp(1.0 - 4.0 * eta, min=0.0))
    numerator = (
        3.0
        * math.pi
        * mf.pow(5)
        * error_tolerance
        * (1.0 + eta_term)
    )
    denominator = 7.0 + 13.0 * eta_term
    delta_f_alpha = (
        4.0 * math.sqrt(2.0 / 5.0) * torch.pow(numerator / denominator, 0.25)
    ) / total_mass_seconds

    if not _msa_has_two_spin(msa_state):
        return torch.clamp(delta_f_alpha, min=0.01)

    velocity = torch.pow(math.pi * mf, 1.0 / 3.0)
    velocity2 = velocity.square()
    dpsi = (
        msa_state["g0"]
        * msa_state["delta_qq"]
        * math.pi
        / (4.0 * velocity2.pow(3))
        * (3.0 + 2.0 * msa_state["psi1"] * velocity + msa_state["psi2"] * velocity2)
    )
    inverse_dpsi = torch.abs(1.0 / dpsi)

    spintaylor_state = build_pnr_spintaylor_msa_state(msa_state)
    orbital_momentum = _lpn_orbital_angular_momentum(
        velocity,
        spintaylor_state,
    )
    spin1_perp = msa_state["m1_2"] * msa_state["chi1_perp"]
    spin2_perp = msa_state["m2_2"] * msa_state["chi2_perp"]
    longitudinal_momentum = orbital_momentum + msa_state["SL"]
    beta_min = torch.atan2(
        torch.abs(f_min.new_tensor(spin1_perp - spin2_perp)),
        longitudinal_momentum,
    )
    beta_max = torch.atan2(
        f_min.new_tensor(spin1_perp + spin2_perp),
        longitudinal_momentum,
    )
    sharp_alpha = (beta_min < 0.01) & (beta_min / beta_max < 0.55)
    inverse_dpsi = torch.where(sharp_alpha, inverse_dpsi / 4.0, inverse_dpsi)
    delta_f_two_spin = inverse_dpsi / (4.0 * total_mass_seconds)
    use_two_spin = (delta_f_two_spin < delta_f_alpha) & ~torch.isnan(dpsi)
    spacing = torch.where(use_two_spin, delta_f_two_spin, delta_f_alpha)
    return torch.clamp(spacing, min=0.01)


def build_pnr_spintaylor_integration(
    f_min,
    output_delta_f,
    ringdown_frequency,
    single_spin,
    msa_state,
    *,
    error_tolerance=0.01,
):
    """Build the buffered lower-frequency controls for a `(2, 2)` PNR grid.

    Frequencies are returned in Hz.  ``starting_frequency`` is LAL's buffered,
    interpolation-grid-aligned value; the time-domain SpinTaylor driver evolves
    another 0.5 Hz lower when integrating backward from the reference state.
    """

    f_min, output_delta_f, ringdown_frequency = _as_common_tensors(
        f_min,
        output_delta_f,
        ringdown_frequency,
    )
    if any(
        value.numel() != 1
        for value in (f_min, output_delta_f, ringdown_frequency)
    ):
        raise ValueError("PNR SpinTaylor integration controls must be scalar")
    f_min, output_delta_f, ringdown_frequency = (
        value.reshape(())
        for value in (f_min, output_delta_f, ringdown_frequency)
    )
    controls = torch.stack((f_min, output_delta_f, ringdown_frequency))
    if not bool(torch.all(torch.isfinite(controls)).detach().cpu()):
        raise ValueError("PNR SpinTaylor integration controls must be finite")
    if float(f_min.detach().cpu()) <= 0.0:
        raise ValueError("PNR SpinTaylor minimum frequency must be positive")
    if float(output_delta_f.detach().cpu()) < 0.0:
        raise ValueError("PNR output frequency spacing cannot be negative")
    if float(ringdown_frequency.detach().cpu()) <= 0.0:
        raise ValueError("PNR ringdown frequency must be positive")

    flow = torch.where(
        output_delta_f != 0.0,
        torch.floor(f_min / torch.where(
            output_delta_f != 0.0,
            output_delta_f,
            torch.ones_like(output_delta_f),
        ))
        * output_delta_f,
        f_min,
    )
    fmin_hm_inspiral = flow

    _, _, _, alpha4 = _pnr_alpha_fit_coefficients(single_spin, 223)
    mf_low_cut = (3.0 / 3.5) * (alpha4 / 3.0)
    mf_high_cut = build_pnr_beta_merger_parameters(
        single_spin,
        prec_version=223,
    ).mf_lower
    chi_effective = f_min.new_tensor(
        msa_state["m1"] * msa_state["chi1z"]
        + msa_state["m2"] * msa_state["chi2z"]
    )
    f_cut = torch.where(
        chi_effective > 0.99,
        f_min.new_tensor(0.33),
        f_min.new_tensor(0.3),
    )
    mf_high_cut = torch.where(
        (mf_high_cut > f_cut) | (mf_high_cut < 0.1 * ringdown_frequency),
        ringdown_frequency,
        mf_high_cut,
    )
    mf_low_cut = torch.where(
        (mf_low_cut > f_cut) | (mf_high_cut < mf_low_cut),
        mf_high_cut / 2.0,
        mf_low_cut,
    )

    total_mass_seconds = f_min.new_tensor(msa_state["m_sec"])
    flow_alpha = 0.65 * mf_low_cut / total_mass_seconds
    flow = torch.where(
        flow_alpha < flow,
        fmin_hm_inspiral / 1.5,
        fmin_hm_inspiral,
    )
    interpolation_delta_f = pnr_spintaylor_interpolation_delta_f(
        flow,
        msa_state,
        output_delta_f=output_delta_f,
        error_tolerance=error_tolerance,
    )
    integration_buffer = 1.4 * interpolation_delta_f
    buffered_flow = torch.where(
        flow - 2.0 * interpolation_delta_f < 0.0,
        flow / 2.0,
        flow - 2.0 * interpolation_delta_f,
    )
    starting_frequency = (
        torch.floor(buffered_flow / interpolation_delta_f)
        * interpolation_delta_f
    )
    if float(starting_frequency.detach().cpu()) <= 0.0:
        raise ValueError("PNR SpinTaylor starting frequency must be positive")
    return PNRSpinTaylorIntegration(
        starting_frequency=starting_frequency,
        trajectory_minimum_frequency=starting_frequency - 0.5,
        interpolation_delta_f=interpolation_delta_f,
        integration_buffer=integration_buffer,
    )


def _pnr_full_pn_beta(mf, parameters, msa_state):
    mf = torch.as_tensor(
        mf,
        device=parameters.b0.device,
        dtype=parameters.b0.dtype,
    )
    velocity = torch.pow(math.pi * mf, 1.0 / 3.0)
    return torch.acos(msa_angles(velocity, msa_state)[2])


def pnr_pn_beta(mf, parameters, msa_state, single_spin_msa_state=None):
    """Evaluate MSA-223 beta with two-spin oscillations tapered at merger."""

    mf = torch.as_tensor(
        mf,
        device=parameters.b0.device,
        dtype=parameters.b0.dtype,
    )
    full_beta = _pnr_full_pn_beta(mf, parameters, msa_state)
    if not _msa_has_two_spin(msa_state):
        return full_beta
    if single_spin_msa_state is None:
        raise ValueError("a single-spin MSA state is required for a two-spin system")

    velocity = torch.pow(math.pi * mf, 1.0 / 3.0)
    single_spin_beta = torch.acos(msa_angles(velocity, single_spin_msa_state)[2])
    envelope = torch.cos(math.pi * mf / (2.0 * parameters.mf_lower)) ** 2
    tapered_beta = single_spin_beta + (full_beta - single_spin_beta) * envelope
    return torch.where(mf <= parameters.mf_lower, tapered_beta, single_spin_beta)


def pnr_pn_waveform_beta(mf, pn_beta, msa_state):
    """Map the MSA dynamics angle to the PNR waveform opening angle."""

    mf, pn_beta = _as_common_tensors(mf, pn_beta)
    velocity = torch.pow(math.pi * mf, 1.0 / 3.0)
    reference_velocity = velocity.new_tensor(msa_state["v_ref"])
    l_ref = _lpn_orbital_angular_momentum(reference_velocity, msa_state)
    # LAL stores LRef and J0 in total-mass-normalized units but applies Mtot^2
    # to the frequency-dependent L in this wrapper. Preserve that convention.
    total_mass_solar = velocity.new_tensor(msa_state["m_sec"] / MTSUN)
    l_3pn = total_mass_solar * total_mass_solar * _lpn_orbital_angular_momentum(
        velocity,
        msa_state,
    )

    mass1_fraction = velocity.new_tensor(msa_state["m1"])
    mass2_fraction = velocity.new_tensor(msa_state["m2"])
    mass1 = mass1_fraction * total_mass_solar
    mass2 = mass2_fraction * total_mass_solar
    spin_x = (
        mass1_fraction * mass1_fraction * msa_state["chi1x"]
        + mass2_fraction * mass2_fraction * msa_state["chi2x"]
    )
    spin_y = (
        mass1_fraction * mass1_fraction * msa_state["chi1y"]
        + mass2_fraction * mass2_fraction * msa_state["chi2y"]
    )
    spin_z = (
        mass1_fraction * mass1_fraction * msa_state["chi1z"]
        + mass2_fraction * mass2_fraction * msa_state["chi2z"]
    )
    j_ref = torch.sqrt(spin_x * spin_x + spin_y * spin_y + (spin_z + l_ref) ** 2)
    chi_effective = (
        mass1_fraction * msa_state["chi1z"]
        + mass2_fraction * msa_state["chi2z"]
    )
    chi_parallel = (mass1 + mass2) * chi_effective / mass1
    chi_perpendicular = (
        (j_ref - (l_ref - l_3pn)) * torch.sin(pn_beta) / (mass1 * mass1)
    )
    chi = torch.hypot(chi_parallel, chi_perpendicular)
    cosine_theta = torch.where(
        chi > 0.0,
        chi_parallel / chi,
        torch.ones_like(chi),
    )
    theta = torch.acos(torch.clamp(cosine_theta, -1.0, 1.0))

    symmetric_mass_ratio = mass1 * mass2 / (total_mass_solar * total_mass_solar)
    mass_difference = (mass1 - mass2) / total_mass_solar
    velocity2 = velocity * velocity
    velocity3 = velocity2 * velocity
    cosine_iota = torch.cos(pn_beta)
    sine_iota = torch.sin(pn_beta)
    cosine_half_iota = torch.cos(pn_beta / 2.0)
    sine_half_iota = torch.sin(pn_beta / 2.0)
    cosine_theta = torch.cos(theta)
    sine_theta = torch.sin(theta)

    numerator0 = 84.0 * sine_iota
    numerator2 = 2.0 * (55.0 * symmetric_mass_ratio - 107.0) * sine_iota
    numerator3 = (
        -7.0
        * (5.0 * symmetric_mass_ratio + 6.0 * mass_difference + 6.0)
        * chi
        * (2.0 * cosine_iota - 1.0)
        * sine_theta
        + 56.0
        * (
            3.0 * math.pi
            - (1.0 + mass_difference - symmetric_mass_ratio) * chi * cosine_theta
        )
        * sine_iota
    )
    numerator = (
        numerator0 + numerator2 * velocity2 + numerator3 * velocity3
    ) / cosine_half_iota

    denominator0 = 84.0 * cosine_half_iota
    denominator2 = (
        2.0 * (55.0 * symmetric_mass_ratio - 107.0) * cosine_half_iota
    )
    denominator3 = (
        56.0
        * (
            3.0 * math.pi
            + (symmetric_mass_ratio - 1.0 - mass_difference)
            * chi
            * cosine_theta
        )
        * cosine_half_iota
        + 14.0
        * (6.0 + 6.0 * mass_difference + 5.0 * symmetric_mass_ratio)
        * chi
        * sine_theta
        * sine_half_iota
    )
    denominator = denominator0 + denominator2 * velocity2 + denominator3 * velocity3
    atan_denominator = 2.0 * denominator
    near_origin = (torch.abs(numerator) < 1.0e-15) & (
        torch.abs(atan_denominator) < 1.0e-15
    )
    return 2.0 * torch.where(
        near_origin,
        torch.zeros_like(numerator),
        torch.atan2(numerator, atan_denominator),
    )


def _beta_rescaling(
    mf,
    beta_lower,
    beta_upper,
    derivative_lower,
    derivative_upper,
):
    mixed_derivative = (
        beta_lower * derivative_upper - beta_upper * derivative_lower
    )
    rescale1 = -(
        -2.0 * beta_lower * (beta_upper - beta_lower)
        + mixed_derivative * mf
    ) / (beta_lower * beta_lower * mf)
    rescale2 = -(
        beta_lower * (beta_upper - beta_lower) - mixed_derivative * mf
    ) / (beta_lower * mf) ** 2
    return (
        torch.where(torch.isnan(rescale1), torch.zeros_like(rescale1), rescale1),
        torch.where(torch.isnan(rescale2), torch.zeros_like(rescale2), rescale2),
    )


def build_pnr_beta_parameters(single_spin, msa_state, single_spin_msa_state=None):
    """Build the complete calibrated PNR beta connection data."""

    parameters = build_pnr_beta_merger_parameters(single_spin)
    if parameters.mf_lower.numel() != 1:
        raise ValueError("PNR beta connection data require scalar source parameters")
    if _msa_has_two_spin(msa_state) and single_spin_msa_state is None:
        raise ValueError("a single-spin MSA state is required for a two-spin system")

    derivative_step = parameters.b0.new_tensor(0.0005)
    sample_frequencies = parameters.mf_lower + derivative_step * torch.tensor(
        (-1.0, 0.0, 1.0),
        device=parameters.b0.device,
        dtype=parameters.b0.dtype,
    )
    dynamics_beta = pnr_pn_beta(
        sample_frequencies,
        parameters,
        msa_state,
        single_spin_msa_state,
    )
    pn_waveform_beta = pnr_pn_waveform_beta(
        sample_frequencies,
        dynamics_beta,
        msa_state,
    )
    mr_beta = pnr_mr_beta(sample_frequencies, parameters)
    derivative_lower = (pn_waveform_beta[2] - pn_waveform_beta[0]) / (
        2.0 * derivative_step
    )
    derivative_upper = (mr_beta[2] - mr_beta[0]) / (2.0 * derivative_step)
    beta_lower = pn_waveform_beta[1]
    beta_upper = mr_beta[1]
    rescale1, rescale2 = _beta_rescaling(
        parameters.mf_lower,
        beta_lower,
        beta_upper,
        derivative_lower,
        derivative_upper,
    )
    return PNRBetaParameters(
        b0=parameters.b0,
        b1=parameters.b1,
        b2=parameters.b2,
        b3=parameters.b3,
        b4=parameters.b4,
        b5=parameters.b5,
        mf_lower=parameters.mf_lower,
        mf_upper=parameters.mf_upper,
        beta_lower=beta_lower,
        beta_upper=beta_upper,
        derivative_lower=derivative_lower,
        derivative_upper=derivative_upper,
        rescale1=rescale1,
        rescale2=rescale2,
    )


def _spintaylor_beta_interpolation_coefficients(
    mf1,
    mf2,
    beta1,
    beta2,
    derivative1,
    derivative2,
):
    """Return LAL's global cubic coefficients for the ST-to-PNR bridge."""

    mf1_2 = mf1 * mf1
    mf1_3 = mf1_2 * mf1
    mf2_2 = mf2 * mf2
    mf2_3 = mf2_2 * mf2
    denominator = (mf1 - mf2) ** 3

    numerator0 = (
        -beta2 * mf1_3
        + 3.0 * beta2 * mf1_2 * mf2
        + derivative2 * mf1_3 * mf2
        - 3.0 * beta1 * mf1 * mf2_2
        + derivative1 * mf1_2 * mf2_2
        - derivative2 * mf1_2 * mf2_2
        + beta1 * mf2_3
        - derivative1 * mf1 * mf2_3
    )
    numerator1 = (
        -derivative2 * mf1_3
        + 6.0 * beta1 * mf1 * mf2
        - 6.0 * beta2 * mf1 * mf2
        - 2.0 * derivative1 * mf1_2 * mf2
        - derivative2 * mf1_2 * mf2
        + derivative1 * mf1 * mf2_2
        + 2.0 * derivative2 * mf1 * mf2_2
        + derivative1 * mf2_3
    )
    numerator2 = (
        -3.0 * (beta1 - beta2) * (mf1 + mf2)
        + (derivative1 + 2.0 * derivative2) * mf1_2
        + (derivative1 - derivative2) * mf1 * mf2
        - (2.0 * derivative1 + derivative2) * mf2_2
    )
    numerator3 = 2.0 * (beta1 - beta2) - (
        derivative1 + derivative2
    ) * (mf1 - mf2)
    return tuple(
        -numerator / denominator
        for numerator in (numerator0, numerator1, numerator2, numerator3)
    )


def build_pnr_spintaylor_beta_parameters(
    single_spin,
    angles,
    msa_state,
    single_spin_msa_state=None,
):
    """Build the version-330 SpinTaylor-to-PNR beta connection data."""

    parameters = build_pnr_beta_merger_parameters(single_spin, prec_version=330)
    if parameters.mf_lower.numel() != 1:
        raise ValueError("PNR beta connection data require scalar source parameters")
    two_spin = _msa_has_two_spin(msa_state)
    if two_spin and single_spin_msa_state is None:
        raise ValueError("a single-spin MSA state is required for a two-spin system")

    derivative_step = parameters.b0.new_tensor(0.0005)
    ftrans = torch.minimum(angles.ftrans_mrd, 0.9 * parameters.mf_lower)
    mf_interpolation_start = torch.maximum(
        ftrans - derivative_step,
        angles.mf[0] + 2.0 * derivative_step,
    )
    sample_offsets = derivative_step * parameters.b0.new_tensor((-1.0, 0.0, 1.0))
    lower_frequencies = mf_interpolation_start + sample_offsets
    lower_beta = torch.acos(
        torch.clamp(spintaylor_inspiral_cosbeta(lower_frequencies, angles), -1.0, 1.0)
    )
    derivative_lower = (lower_beta[2] - lower_beta[0]) / (
        2.0 * derivative_step
    )

    upper_frequencies = parameters.mf_lower + sample_offsets
    connection_msa_state = single_spin_msa_state if two_spin else msa_state
    upper_beta = _pnr_full_pn_beta(
        upper_frequencies,
        parameters,
        connection_msa_state,
    )
    derivative_upper = (upper_beta[2] - upper_beta[0]) / (
        2.0 * derivative_step
    )
    interp0, interp1, interp2, interp3 = (
        _spintaylor_beta_interpolation_coefficients(
            mf_interpolation_start,
            parameters.mf_lower,
            lower_beta[1],
            # LAL intentionally holds beta at its SpinTaylor endpoint value.
            lower_beta[1],
            derivative_lower,
            derivative_upper,
        )
    )

    center_beta = upper_beta[1] if two_spin else lower_beta[1]
    connection_beta = torch.stack(
        (upper_beta[0], center_beta, upper_beta[2])
    )
    pn_waveform_beta = pnr_pn_waveform_beta(
        upper_frequencies,
        connection_beta,
        msa_state,
    )
    mr_beta = pnr_mr_beta(upper_frequencies, parameters)
    derivative_pn = (pn_waveform_beta[2] - pn_waveform_beta[0]) / (
        2.0 * derivative_step
    )
    derivative_mr = (mr_beta[2] - mr_beta[0]) / (2.0 * derivative_step)
    rescale1, rescale2 = _beta_rescaling(
        parameters.mf_lower,
        pn_waveform_beta[1],
        mr_beta[1],
        derivative_pn,
        derivative_mr,
    )
    merger = PNRBetaParameters(
        b0=parameters.b0,
        b1=parameters.b1,
        b2=parameters.b2,
        b3=parameters.b3,
        b4=parameters.b4,
        b5=parameters.b5,
        mf_lower=parameters.mf_lower,
        mf_upper=parameters.mf_upper,
        beta_lower=pn_waveform_beta[1],
        beta_upper=mr_beta[1],
        derivative_lower=derivative_pn,
        derivative_upper=derivative_mr,
        rescale1=rescale1,
        rescale2=rescale2,
    )
    return PNRSpinTaylorBetaParameters(
        merger=merger,
        mf_interpolation_start=mf_interpolation_start,
        interp0=interp0,
        interp1=interp1,
        interp2=interp2,
        interp3=interp3,
    )


def _spintaylor_pnr_beta_connection(mf, parameters):
    mf = torch.as_tensor(
        mf,
        device=parameters.interp0.device,
        dtype=parameters.interp0.dtype,
    )
    return (
        parameters.interp0
        + parameters.interp1 * mf
        + parameters.interp2 * mf * mf
        + parameters.interp3 * mf * mf * mf
    )


def pnr_spintaylor_beta_imr(mf, parameters, angles, *, use_mr_beta):
    """Evaluate version-330 numerical beta with either continuation policy."""

    mf = torch.as_tensor(
        mf,
        device=parameters.interp0.device,
        dtype=parameters.interp0.dtype,
    )
    if not use_mr_beta:
        return spintaylor_beta_imr(mf, angles)

    transition = (
        torch.minimum(angles.ftrans_mrd, 0.9 * parameters.merger.mf_lower)
        - mf.new_tensor(0.0005)
    )
    inspiral = torch.acos(
        torch.clamp(spintaylor_inspiral_cosbeta(mf, angles), -1.0, 1.0)
    )
    return torch.where(
        mf < transition,
        inspiral,
        _spintaylor_pnr_beta_connection(mf, parameters),
    )


def _pnr_spintaylor_dynamics_beta(
    mf,
    parameters,
    angles,
    msa_state,
    single_spin_msa_state,
    *,
    use_mr_beta,
):
    """Return numerical beta with LAL's two-spin oscillation taper."""

    full_beta = pnr_spintaylor_beta_imr(
        mf,
        parameters,
        angles,
        use_mr_beta=use_mr_beta,
    )
    if not _msa_has_two_spin(msa_state):
        return full_beta
    if single_spin_msa_state is None:
        raise ValueError("a single-spin MSA state is required for a two-spin system")

    merger = parameters.merger
    velocity = torch.pow(math.pi * mf, 1.0 / 3.0)
    single_spin_beta = torch.acos(
        msa_angles(velocity, single_spin_msa_state)[2]
    )
    envelope = torch.cos(math.pi * mf / (2.0 * merger.mf_lower)) ** 2
    tapered_beta = single_spin_beta + (full_beta - single_spin_beta) * envelope
    return torch.where(mf <= merger.mf_lower, tapered_beta, single_spin_beta)


def pnr_spintaylor_beta(
    mf,
    parameters,
    single_spin,
    angles,
    msa_state,
    single_spin_msa_state=None,
):
    """Evaluate the complete numerical SpinTaylor-330 PNR beta angle."""

    merger = parameters.merger
    mf = torch.as_tensor(
        mf,
        device=merger.b0.device,
        dtype=merger.b0.dtype,
    )

    # Outside the calibration region LAL retains the full two-spin
    # SpinTaylor evolution and its generic merger-ringdown continuation.
    full_generic_beta = pnr_spintaylor_beta_imr(
        mf,
        parameters,
        angles,
        use_mr_beta=False,
    )
    no_merger_beta = _arctan_window(
        pnr_pn_waveform_beta(mf, full_generic_beta, msa_state)
    )

    # The calibrated side uses the PNR-connected continuation. In the
    # transition window, LAL blends it with the generic continuation after
    # applying the same two-spin taper and waveform-angle mapping to each.
    tuned_dynamics_beta = _pnr_spintaylor_dynamics_beta(
        mf,
        parameters,
        angles,
        msa_state,
        single_spin_msa_state,
        use_mr_beta=True,
    )
    tuned_waveform_beta = pnr_pn_waveform_beta(
        mf,
        tuned_dynamics_beta,
        msa_state,
    )
    rescaled_beta = tuned_waveform_beta * (
        1.0 + merger.rescale1 * mf + merger.rescale2 * mf * mf
    )
    final_beta = pnr_mr_beta(merger.mf_upper, merger)
    tuned_beta = torch.where(
        mf <= merger.mf_lower,
        rescaled_beta,
        torch.where(
            mf >= merger.mf_upper,
            final_beta,
            pnr_mr_beta(mf, merger),
        ),
    )

    generic_dynamics_beta = _pnr_spintaylor_dynamics_beta(
        mf,
        parameters,
        angles,
        msa_state,
        single_spin_msa_state,
        use_mr_beta=False,
    )
    generic_waveform_beta = pnr_pn_waveform_beta(
        mf,
        generic_dynamics_beta,
        msa_state,
    )
    window = pnr_angles_window(single_spin.mass_ratio, single_spin.magnitude)
    calibrated_beta = _arctan_window(
        window * tuned_beta + (1.0 - window) * generic_waveform_beta
    )

    attach_merger = (
        (merger.mf_lower >= 0.009)
        & (merger.mf_lower != 100.0)
        & (merger.beta_upper > 0.0)
        & (merger.beta_upper <= 5.0 * (merger.b0 + 0.1))
    )
    return torch.where(
        attach_merger & (window > 0.0),
        calibrated_beta,
        no_merger_beta,
    )


def pnr_beta(mf, parameters, single_spin, msa_state, single_spin_msa_state=None):
    """Evaluate the complete MSA-223 PNR beta prescription."""

    mf = torch.as_tensor(
        mf,
        device=parameters.b0.device,
        dtype=parameters.b0.dtype,
    )
    full_pn_beta = _pnr_full_pn_beta(mf, parameters, msa_state)
    no_merger_beta = _arctan_window(
        pnr_pn_waveform_beta(mf, full_pn_beta, msa_state)
    )

    dynamics_beta = pnr_pn_beta(
        mf,
        parameters,
        msa_state,
        single_spin_msa_state,
    )
    pn_waveform_beta = pnr_pn_waveform_beta(mf, dynamics_beta, msa_state)
    rescaled_beta = pn_waveform_beta * (
        1.0 + parameters.rescale1 * mf + parameters.rescale2 * mf * mf
    )
    final_beta = pnr_mr_beta(parameters.mf_upper, parameters)
    tuned_beta = torch.where(
        mf <= parameters.mf_lower,
        rescaled_beta,
        torch.where(
            mf >= parameters.mf_upper,
            final_beta,
            pnr_mr_beta(mf, parameters),
        ),
    )
    window = pnr_angles_window(single_spin.mass_ratio, single_spin.magnitude)
    blended_beta = torch.where(
        window >= 1.0,
        tuned_beta,
        window * tuned_beta + (1.0 - window) * pn_waveform_beta,
    )
    calibrated_beta = _arctan_window(blended_beta)

    almost_antialigned = (
        (window >= 1.0)
        & (mf <= parameters.mf_lower)
        & (parameters.beta_lower < 0.01 * parameters.beta_upper)
    )
    calibrated_beta = torch.where(
        almost_antialigned,
        _arctan_window(pnr_mr_beta(parameters.mf_lower, parameters)),
        calibrated_beta,
    )
    attach_merger = (
        (parameters.mf_lower >= 0.009)
        & (parameters.mf_lower != 100.0)
        & (parameters.beta_upper > 0.0)
        & (parameters.beta_upper <= 5.0 * (parameters.b0 + 0.1))
    )
    use_calibration = attach_merger & (window > 0.0)
    return torch.where(use_calibration, calibrated_beta, no_merger_beta)


def pnr_gamma(frequencies, alpha, beta):
    """Integrate the minimal-rotation PNR gamma angle using Boole's rule."""

    frequencies, alpha, beta = _as_common_tensors(frequencies, alpha, beta)
    if frequencies.ndim != 1:
        raise ValueError("PNR gamma frequencies must be one-dimensional")
    if alpha.shape != frequencies.shape or beta.shape != frequencies.shape:
        raise ValueError("PNR alpha, beta, and frequency arrays must have equal shape")
    if frequencies.numel() < 2:
        raise ValueError("PNR gamma integration requires at least two frequencies")
    if bool(torch.any(frequencies[1:] <= frequencies[:-1])):
        raise ValueError("PNR gamma frequencies must be strictly increasing")

    angle_values = torch.stack((alpha, beta), dim=-1)
    linear, quadratic, cubic = _natural_cubic_coeff(
        frequencies,
        angle_values,
    )
    lower = frequencies[:-1]
    width = frequencies[1:] - lower
    nodes = lower[:, None] + width[:, None] * frequencies.new_tensor(
        (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    alpha_derivative = _spline_derivative(
        nodes,
        frequencies,
        linear[:, 0],
        quadratic[:, 0],
        cubic[:, 0],
    )
    beta_nodes = _spline_eval(
        nodes,
        frequencies,
        beta,
        linear[:, 1],
        quadratic[:, 1],
        cubic[:, 1],
    )
    integrand = -alpha_derivative * torch.cos(beta_nodes)
    weights = frequencies.new_tensor((7.0, 32.0, 12.0, 32.0, 7.0))
    increments = width / 90.0 * torch.sum(integrand * weights, dim=-1)
    return torch.cat(
        (
            torch.zeros(1, device=frequencies.device, dtype=frequencies.dtype),
            torch.cumsum(increments, dim=0),
        )
    )


def pnr_beta_bf_coefficient(eta, chi, cosine):
    """Return the fitted merger-ringdown offset in the PNR beta angle."""

    eta, chi, cosine = _as_common_tensors(eta, chi, cosine)
    polynomial = _evaluate_coefficient_array(
        _BETA_BF_COEFFICIENTS,
        eta,
        chi,
        cosine,
    )
    return chi * torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0)) * polynomial


def _arctan_window(beta):
    border = 0.01
    pi_by_two = 1.570796326794897
    log_ratio = 500.0 * torch.log(torch.abs(beta - pi_by_two)) - math.log(
        7.308338225719002e97
    )
    reduced_ratio = torch.exp(-torch.abs(log_ratio))
    angle = torch.where(
        log_ratio > 0.0,
        pi_by_two - torch.atan(reduced_ratio),
        torch.atan(reduced_ratio),
    )
    sign = torch.where(beta < pi_by_two, -torch.ones_like(beta), torch.ones_like(beta))
    tapered = sign * 1.569378278348018 * angle.pow(0.002) + pi_by_two
    return torch.where(
        (beta <= border) | (beta >= math.pi - border),
        tapered,
        beta,
    )


def pnr_ringdown_beta(single_spin: PNRSingleSpin):
    """Return the fitted PNR opening angle during ringdown."""

    raw_beta = torch.acos(torch.clamp(single_spin.final_cosine, -1.0, 1.0))
    raw_beta -= pnr_beta_bf_coefficient(
        single_spin.symmetric_mass_ratio,
        single_spin.magnitude,
        single_spin.cosine,
    )
    return _arctan_window(raw_beta)


def pnr_spintaylor_final_spin_model4(
    trajectory: SpinTaylorTrajectory,
    mass1,
    mass2,
    cosbeta_max,
):
    """Return LAL's evolved-spin final-spin model 4.

    The final accepted SpinTaylor state supplies the component spins and
    orbital direction.  ``cosbeta_max`` is evaluated at the selected inspiral
    endpoint, which can differ from the final trajectory frequency when LAL's
    MECO safeguard is active.
    """

    state = trajectory.state
    if state.ndim != 2 or state.shape[0] == 0 or state.shape[1] != 14:
        raise ValueError("SpinTaylor final-spin evaluation requires a trajectory")
    if not state.dtype.is_floating_point:
        raise ValueError("SpinTaylor trajectory state must be floating point")

    mass1 = torch.as_tensor(mass1, dtype=state.dtype, device=state.device)
    mass2 = torch.as_tensor(mass2, dtype=state.dtype, device=state.device)
    cosbeta_max = torch.as_tensor(
        cosbeta_max,
        dtype=state.dtype,
        device=state.device,
    )
    if any(value.numel() != 1 for value in (mass1, mass2, cosbeta_max)):
        raise ValueError("SpinTaylor final-spin inputs must be scalar")
    mass1, mass2, cosbeta_max = (
        value.reshape(()) for value in (mass1, mass2, cosbeta_max)
    )
    inputs = torch.stack((mass1, mass2, cosbeta_max))
    if not bool(torch.all(torch.isfinite(inputs)).detach().cpu()):
        raise ValueError("SpinTaylor final-spin inputs must be finite")
    if float(torch.minimum(mass1, mass2).detach().cpu()) <= 0.0:
        raise ValueError("SpinTaylor final-spin masses must be positive")

    endpoint = state[-1]
    lnhat = endpoint[2:5]
    spin1 = endpoint[5:8]
    spin2 = endpoint[8:11]
    if bool((mass2 > mass1).detach().cpu()):
        mass1, mass2 = mass2, mass1
        spin1, spin2 = spin2, spin1

    total_mass = mass1 + mass2
    fraction1 = mass1 / total_mass
    fraction2 = mass2 / total_mass
    lnorm = torch.linalg.vector_norm(lnhat)
    if not bool(torch.isfinite(lnorm).detach().cpu()) or float(
        lnorm.detach().cpu()
    ) <= 0.0:
        raise ValueError("SpinTaylor final orbital direction must be finite and nonzero")

    component_spin1 = spin1 / fraction1.square()
    component_spin2 = spin2 / fraction2.square()
    spin1_l = torch.dot(component_spin1, lnhat) / lnorm
    spin2_l = torch.dot(component_spin2, lnhat) / lnorm
    spin1_perp = fraction1.square() * (component_spin1 - lnhat * spin1_l)
    spin2_perp = fraction2.square() * (component_spin2 - lnhat * spin2_l)
    chi_perp = torch.linalg.vector_norm(spin1_perp + spin2_perp) / fraction1.square()
    magnitude = precessing_final_spin_2017(
        fraction1 * fraction2,
        spin1_l,
        spin2_l,
        chi_perp,
    )
    return torch.copysign(torch.ones_like(cosbeta_max), cosbeta_max) * magnitude


def build_pnr_spintaylor_remnant(
    trajectory: SpinTaylorTrajectory,
    mass1,
    mass2,
    chi1_l,
    chi2_l,
    cosbeta_max,
):
    """Build the evolved-spin remnant and QNM data for version 330."""

    final_spin = pnr_spintaylor_final_spin_model4(
        trajectory,
        mass1,
        mass2,
        cosbeta_max,
    )
    mass1 = torch.as_tensor(mass1, dtype=final_spin.dtype, device=final_spin.device)
    mass2 = torch.as_tensor(mass2, dtype=final_spin.dtype, device=final_spin.device)
    chi1_l = torch.as_tensor(chi1_l, dtype=final_spin.dtype, device=final_spin.device)
    chi2_l = torch.as_tensor(chi2_l, dtype=final_spin.dtype, device=final_spin.device)
    if any(value.numel() != 1 for value in (mass1, mass2, chi1_l, chi2_l)):
        raise ValueError("SpinTaylor remnant source parameters must be scalar")
    mass1, mass2, chi1_l, chi2_l = (
        value.reshape(()) for value in (mass1, mass2, chi1_l, chi2_l)
    )
    if bool((mass2 > mass1).detach().cpu()):
        mass1, mass2 = mass2, mass1
        chi1_l, chi2_l = chi2_l, chi1_l

    remnant = get_remnant_fMs(
        mass1,
        mass2,
        chi1_l,
        chi2_l,
        final_spin=final_spin,
    )
    radiated_energy = torch.as_tensor(
        remnant.radiated_energy,
        dtype=final_spin.dtype,
        device=final_spin.device,
    )
    ringdown_frequency = torch.as_tensor(
        remnant.ringdown_frequency,
        dtype=final_spin.dtype,
        device=final_spin.device,
    )
    damping_frequency = torch.as_tensor(
        remnant.damping_frequency,
        dtype=final_spin.dtype,
        device=final_spin.device,
    )
    final_mass = 1.0 - radiated_energy
    damping_difference = qnm_fdamp_21(final_spin) / final_mass - damping_frequency
    return PNRSpinTaylorRemnant(
        final_spin=final_spin,
        radiated_energy=radiated_energy,
        final_mass=final_mass,
        ringdown_frequency=ringdown_frequency,
        damping_frequency=damping_frequency,
        damping_difference=damping_difference,
    )


def _pnr_spintaylor_host_source(mass1, mass2, spin1, spin2, inclination):
    """Validate and mass-order scalar source parameters on the host."""

    def scalar(value, name):
        value = torch.as_tensor(value)
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        value = float(value.detach().cpu())
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    mass1 = scalar(mass1, "mass1")
    mass2 = scalar(mass2, "mass2")
    inclination = scalar(inclination, "inclination")
    try:
        spin1 = tuple(spin1)
        spin2 = tuple(spin2)
    except TypeError as exc:
        raise ValueError("PNR SpinTaylor spins must have length three") from exc
    if len(spin1) != 3 or len(spin2) != 3:
        raise ValueError("PNR SpinTaylor spins must have length three")
    spin1 = tuple(scalar(value, f"spin1[{index}]") for index, value in enumerate(spin1))
    spin2 = tuple(scalar(value, f"spin2[{index}]") for index, value in enumerate(spin2))
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("PNR SpinTaylor masses must be positive")
    if mass2 > mass1:
        mass1, mass2 = mass2, mass1
        spin1, spin2 = spin2, spin1
    return mass1, mass2, spin1, spin2, inclination


def _pnr_spintaylor_context(values, *, dtype, device):
    """Choose the Torch context without moving scalar initialization on device."""

    reference = next(
        (value for value in values if isinstance(value, torch.Tensor)),
        None,
    )
    if device is None:
        device = reference.device if reference is not None else torch.device("cpu")
    if dtype is None:
        dtype = (
            reference.dtype
            if reference is not None and reference.dtype.is_floating_point
            else torch.float64
        )
    probe = torch.empty((), dtype=dtype, device=device)
    if not probe.dtype.is_floating_point:
        raise ValueError("PNR SpinTaylor angles require a floating-point dtype")
    return probe


def build_pnr_spintaylor_angle_model(
    mass1,
    mass2,
    spin1,
    spin2,
    inclination,
    f_min,
    output_delta_f,
    f_ref,
    *,
    dtype=None,
    device=None,
    coarse_factor=10.0,
):
    """Build LAL's numerical version-330 PNR angle prescription.

    Source setup remains scalar host work, while the SpinTaylor evolution,
    spline construction, fitted connections, and subsequent array evaluation
    use Torch on ``device``.  Frequencies passed here are in Hz.
    """

    context_values = (mass1, mass2, *spin1, *spin2, inclination, f_min, f_ref)
    context = _pnr_spintaylor_context(
        context_values,
        dtype=dtype,
        device=device,
    )
    mass1, mass2, spin1, spin2, inclination = _pnr_spintaylor_host_source(
        mass1,
        mass2,
        spin1,
        spin2,
        inclination,
    )
    f_min = _scalar_float(f_min, "minimum frequency")
    output_delta_f = _scalar_float(output_delta_f, "frequency spacing")
    f_ref = _scalar_float(f_ref, "reference frequency")
    coarse_factor = _scalar_float(coarse_factor, "SpinTaylor coarse factor")
    controls = (f_min, output_delta_f, f_ref, coarse_factor)
    if not all(math.isfinite(value) for value in controls):
        raise ValueError("PNR SpinTaylor frequency controls must be finite")
    if f_min <= 0.0 or f_ref <= 0.0:
        raise ValueError("PNR SpinTaylor frequencies must be positive")
    if output_delta_f < 0.0:
        raise ValueError("PNR SpinTaylor output spacing cannot be negative")
    if coarse_factor < 1.0:
        raise ValueError("PNR SpinTaylor coarse factor must be at least one")

    total_mass_seconds = (mass1 + mass2) * MTSUN
    msa_state = build_msa_state(
        mass1,
        mass2,
        spin1,
        spin2,
        total_mass_seconds,
        f_ref,
    )
    source = context.new_tensor((mass1, mass2, *spin1, *spin2))
    mass1_tensor, mass2_tensor = source[:2]
    spin1_tensor = source[2:5]
    spin2_tensor = source[5:8]
    initial_single_spin = pnr_single_spin_mapping(*source)
    initial_remnant = get_remnant_fMs(
        mass1_tensor,
        mass2_tensor,
        spin1_tensor[2],
        spin2_tensor[2],
    )
    integration = build_pnr_spintaylor_integration(
        context.new_tensor(f_min),
        context.new_tensor(output_delta_f),
        initial_remnant.ringdown_frequency,
        initial_single_spin,
        msa_state,
    )
    trajectory = spintaylor_t4_time_trajectory(
        integration.trajectory_minimum_frequency * total_mass_seconds,
        context.new_tensor(f_ref * total_mass_seconds),
        initial_remnant.ringdown_frequency
        + 8.0 * initial_remnant.damping_frequency,
        mass1_tensor,
        mass2_tensor,
        spin1_tensor,
        spin2_tensor,
        coarse_factor=coarse_factor,
        pnr_fine_grid=True,
    )
    frame = spintaylor_j_frame(
        f_ref * total_mass_seconds,
        mass1_tensor,
        mass2_tensor,
        spin1_tensor,
        spin2_tensor,
        inclination=inclination,
        phi_ref=0.0,
        convention=1,
    )

    _, raw_cosbeta = spintaylor_j_frame_angles(trajectory, frame)
    linear, quadratic, cubic = _natural_cubic_coeff(
        trajectory.mf,
        raw_cosbeta.unsqueeze(-1),
    )
    fmax_inspiral = trajectory.mf[-1]
    if bool(
        (
            fmax_inspiral
            > initial_remnant.ringdown_frequency
            - initial_remnant.damping_frequency
        )
        .detach()
        .cpu()
    ):
        fmax_inspiral = 1.020 * initial_remnant.meco_frequency
    cosbeta_max = _spline_eval(
        fmax_inspiral,
        trajectory.mf,
        raw_cosbeta,
        linear[:, 0],
        quadratic[:, 0],
        cubic[:, 0],
    )

    single_spin = pnr_spintaylor_single_spin_mapping(
        trajectory,
        mass1_tensor,
        mass2_tensor,
        *spin1_tensor,
        *spin2_tensor,
    )
    remnant = build_pnr_spintaylor_remnant(
        trajectory,
        mass1_tensor,
        mass2_tensor,
        spin1_tensor[2],
        spin2_tensor[2],
        cosbeta_max,
    )
    angles = build_spintaylor_angle_spline(
        trajectory,
        frame,
        fmax_inspiral,
        damping_difference=remnant.damping_difference,
        ringdown_beta=pnr_ringdown_beta(single_spin),
    )
    msa_state = build_pnr_spintaylor_msa_state(msa_state)
    single_spin_msa_state = build_pnr_single_spin_msa_state(
        single_spin,
        msa_state,
    )
    alpha_offset = -frame.alpha0
    alpha_parameters = build_pnr_spintaylor_alpha_parameters(
        single_spin,
        angles,
        total_mass_seconds,
        alpha_offset=alpha_offset,
    )
    beta_parameters = build_pnr_spintaylor_beta_parameters(
        single_spin,
        angles,
        msa_state,
        single_spin_msa_state,
    )
    return PNRSpinTaylorAngleModel(
        mass1=mass1,
        mass2=mass2,
        spin1=spin1,
        spin2=spin2,
        inclination=inclination,
        reference_frequency=f_ref,
        total_mass_seconds=total_mass_seconds,
        integration=integration,
        trajectory=trajectory,
        frame=frame,
        remnant=remnant,
        single_spin=single_spin,
        msa_state=msa_state,
        single_spin_msa_state=single_spin_msa_state,
        angles=angles,
        alpha_parameters=alpha_parameters,
        beta_parameters=beta_parameters,
        alpha_offset=alpha_offset,
    )


def _pnr_angle_at_reference(frequencies, values, reference_frequency):
    """Linearly interpolate a uniform angle array as the LAL wrapper does."""

    reference = frequencies.new_tensor(reference_frequency)
    if bool(((reference < frequencies[0]) | (reference > frequencies[-1])).cpu()):
        raise ValueError("PNR reference frequency must lie on the output interval")
    index = int(torch.searchsorted(frequencies, reference, right=True).item()) - 1
    index = min(max(index, 0), frequencies.numel() - 2)
    lower_frequency = frequencies[index]
    upper_frequency = frequencies[index + 1]
    weight = (reference - lower_frequency) / (upper_frequency - lower_frequency)
    return values[index] + weight * (values[index + 1] - values[index])


def evaluate_pnr_spintaylor_angles(model, frequencies):
    """Evaluate a version-330 model on a uniform frequency grid in Hz."""

    frequencies = torch.as_tensor(
        frequencies,
        dtype=model.alpha_offset.dtype,
        device=model.alpha_offset.device,
    )
    if frequencies.ndim != 1 or frequencies.numel() < 2:
        raise ValueError("PNR angle frequencies must contain at least two samples")
    if bool(torch.any(~torch.isfinite(frequencies)).detach().cpu()):
        raise ValueError("PNR angle frequencies must be finite")
    spacing = frequencies[1:] - frequencies[:-1]
    if bool(torch.any(spacing <= 0.0).detach().cpu()):
        raise ValueError("PNR angle frequencies must be strictly increasing")
    tolerance = 64.0 * torch.finfo(frequencies.dtype).eps
    if not bool(
        torch.allclose(
            spacing,
            spacing[0].expand_as(spacing),
            rtol=tolerance,
            atol=tolerance * max(1.0, float(spacing[0].detach().cpu())),
        )
    ):
        raise ValueError("version-330 PNR angle evaluation requires a uniform grid")

    geometric_frequencies = frequencies * model.total_mass_seconds
    alpha = pnr_spintaylor_alpha(
        geometric_frequencies,
        model.alpha_parameters,
        model.single_spin,
        model.angles,
        alpha_offset=model.alpha_offset,
    )
    beta = pnr_spintaylor_beta(
        geometric_frequencies,
        model.beta_parameters,
        model.single_spin,
        model.angles,
        model.msa_state,
        model.single_spin_msa_state,
    )
    gamma = pnr_gamma(geometric_frequencies, alpha, beta)
    raw_alpha_reference = _pnr_angle_at_reference(
        frequencies,
        alpha,
        model.reference_frequency,
    )
    beta_reference = _pnr_angle_at_reference(
        frequencies,
        beta,
        model.reference_frequency,
    )
    raw_gamma_reference = _pnr_angle_at_reference(
        frequencies,
        gamma,
        model.reference_frequency,
    )
    source_frame = remap_source_frame_parameters_pnr(
        model.mass1,
        model.mass2,
        model.reference_frequency,
        0.0,
        model.inclination,
        model.spin1,
        model.spin2,
        model.total_mass_seconds,
        float(beta_reference.detach().cpu()),
    )
    alpha_reference = raw_alpha_reference - source_frame.alpha_offset_shift
    gamma_reference = raw_gamma_reference + model.frame.epsilon0
    return PNRSpinTaylorAngles(
        frequencies=frequencies,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        alpha_reference=alpha_reference,
        beta_reference=beta_reference,
        gamma_reference=gamma_reference,
        source_frame=source_frame,
        model=model,
    )


def generate_pnr_spintaylor_angles(
    mass1,
    mass2,
    spin1,
    spin2,
    inclination,
    delta_f,
    f_min,
    f_max,
    f_ref=0.0,
    *,
    dtype=None,
    device=None,
    coarse_factor=10.0,
):
    """Generate uniform-Hz version-330 angles like LAL's public wrapper."""

    f_min = _scalar_float(f_min, "minimum frequency")
    f_max = _scalar_float(f_max, "maximum frequency")
    delta_f = _scalar_float(delta_f, "frequency spacing")
    f_ref = _scalar_float(f_ref, "reference frequency")
    if not all(math.isfinite(value) for value in (f_min, f_max, delta_f, f_ref)):
        raise ValueError("PNR angle frequency controls must be finite")
    if f_min <= 0.0 or f_max < f_min or delta_f <= 0.0:
        raise ValueError("PNR angles require 0 < f_min <= f_max and delta_f > 0")
    if f_ref == 0.0:
        f_ref = f_min
    if f_ref < f_min or f_ref > f_max:
        raise ValueError("PNR reference frequency must lie between f_min and f_max")

    context = _pnr_spintaylor_context(
        (mass1, mass2, *spin1, *spin2, inclination),
        dtype=dtype,
        device=device,
    )
    start = int(f_min / delta_f)
    stop = int(f_max / delta_f) + 1
    frequencies = torch.arange(
        start,
        stop,
        dtype=context.dtype,
        device=context.device,
    ) * delta_f
    if frequencies.numel() < 2:
        raise ValueError("PNR angle grid must contain at least two samples")
    model = build_pnr_spintaylor_angle_model(
        mass1,
        mass2,
        spin1,
        spin2,
        inclination,
        float(frequencies[0].detach().cpu()),
        delta_f,
        f_ref,
        dtype=context.dtype,
        device=context.device,
        coarse_factor=coarse_factor,
    )
    return evaluate_pnr_spintaylor_angles(model, frequencies)


def pnr_final_spin_model7(
    eta,
    chi1_l,
    chi2_l,
    chi_tot_perp,
    beta_ringdown,
):
    """Return XPNR's final-spin model 7 on the active Torch device."""

    eta, chi1_l, chi2_l, chi_tot_perp, beta_ringdown = _as_common_tensors(
        eta,
        chi1_l,
        chi2_l,
        chi_tot_perp,
        beta_ringdown,
    )
    magnitude = torch.abs(
        precessing_final_spin_2017(
            eta,
            chi1_l,
            chi2_l,
            chi_tot_perp,
        )
    )
    return torch.copysign(magnitude, torch.cos(beta_ringdown))
