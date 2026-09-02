# Copyright (C) 2014 Eliu Huerta
# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""Scalar coefficients for the Torch-native ``EccentricFD`` waveform.

These expressions are a direct transcription of LALSuite's
``LALSimInspiralOptimizedCoefficientsEccentricityFD.c``.  They are kept in a
separate module so the generated analytic fits do not obscure the device-side
waveform assembly in :mod:`pycbc.waveform.eccentricfd_torch`.
"""

import math
from dataclasses import dataclass

_EULER_GAMMA = 0.577215664901532860606512090


def _c0(eta):
    return 3.0 / (128.0 * eta)


def _c1(total_mass):
    return math.pi * total_mass


def _c2(e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p1 = pow(f0, 19.0 / 9.0)
    return p1 * (
        (-2355.0 * e2) / 1462.0
        - (2608555.0 * e4) / 444448.0
        - (1326481225.0 * e6) / 1.01334144e8
        - (6505217202575.0 * e8) / 2.77250217984e11
    )


def _c3(e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p2 = pow(f0, 19.0 / 3.0)
    return p2 * (
        (-75356125.0 * e6) / 3.326976e6 - (250408403375.0 * e8) / 1.011400704e9
    )


def _c4(e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p3 = pow(f0, 38.0 / 9.0)
    return p3 * (
        (5222765.0 * e4) / 998944.0
        + (17355248095.0 * e6) / 4.55518464e8
        + (128274289063885.0 * e8) / 8.30865678336e11
    )


def _c5(e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p4 = pow(f0, 76.0 / 9.0)
    return (4537813337273.0 * e8 * p4) / (39444627456.0)


def _c6(eta):
    return 3715.0 / 756.0 + (55.0 * eta) / 9.0


def _c7(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p1 = pow(f0, 19.0 / 9.0)
    return p1 * (
        (-583255.0 * e2) / 122808.0
        - (1938156365.0 * e4) / 1.12000896e8
        - (985575550175 * e6) / 2.5536204288e10
        - (690482340216175.0 * e8) / 9.981007847424e12
        - (8635.0 * e2 * eta) / 1462.0
        - (28694105.0 * e4 * eta) / 1.333344e6
        - (14591293475.0 * e6 * eta) / 3.04002432e8
        - (71557389228325.0 * e8 * eta) / 8.31750653952e11
    )


def _c8(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p2 = pow(f0, 19.0 / 3.0)
    return p2 * (
        (-31207858707289299381995.0 * e6) / 5.3807139303898e20
        - (103703714484322341846369385.0 * e8) / 1.635737034838499e23
        - (462027517873731215615.0 * e6 * eta) / 6.405611821892619e18
        - (1535317441894408829488645.0 * e8 * eta) / 1.9473059938553564e21
    )


def _c9(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p3 = pow(f0, 38.0 / 9.0)
    return p3 * (
        (1867778296655755.0 * e4) / 1.34516772125568e14
        + (6206627279787073865.0 * e6) / 6.133964808925901e16
        + (45873772442848006154795.0 * e8) / 1.1188351811480843e20
        + (27652168591135.0 * e4 * eta) / 1.601390144352e12
        + (91888156228341605.0 * e6 * eta) / 7.30233905824512e14
        + (679154100768947601215.0 * e8 * eta) / 1.33194664422391e18
    )


def _c10(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p4 = pow(f0, 76.0 / 9.0)
    return p4 * (
        (2222952136716664456903429135345831.0 * e8) / 7.66183921683643e30
        + (32910462320165961003953863376587.0 * e8 * eta) / 9.121237162900513e28
    )


def _c11(eta):
    return -16.0 * math.pi


def _c12(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p1 = pow(f0, 19.0 / 9.0)
    return (
        math.pi
        * p1
        * (
            (7536.0 * e2) / 731.0
            + (521711.0 * e4) / 13889.0
            + (265296245.0 * e6) / 3.166692e6
            + (1301043440515.0 * e8) / 8.664069312e9
        )
    )


def _c13(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p2 = pow(f0, 19.0 / 3.0)
    return (
        math.pi
        * p2
        * (
            (7800202806596945705.0 * e6) / 6.672512314471478e16
            + (25920073926321650577715.0 * e8) / 2.0284437435993293e19
        )
    )


def _c14(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p3 = pow(f0, 38.0 / 9.0)
    return (
        math.pi
        * p3
        * (
            (-475065859669.0 * e4) / 1.6681147337e10
            - (1578643851680087.0 * e6) / 7.606603185672e12
            - (11667906828579178421.0 * e8) / 1.3874444210665728e16
        )
    )


def _c15(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p4 = pow(f0, 76.0 / 9.0)
    return -(
        (2759558801317902042155334469457.0 * e8 * p4 * math.pi)
        / 4750644355677350183868572160.0
    )


def _c16(eta):
    return 15293365.0 / 508032.0 + (27145.0 * eta) / 504.0 + (3085.0 * eta * eta) / 72.0


def _c17(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p1 = pow(f0, 19.0 / 9.0)
    return p1 * (
        (-2401058305.0 * e2) / 2.47580928e8
        - (7978716747515.0 * e4) / 2.25793806336e11
        - (4057272307914425.0 * e6) / 5.1480987844608e13
        - (2842476030950240425.0 * e8) / 2.0121711820406784e16
        - (4261765.0 * e2 * eta) / 245616.0
        - (14161845095.0 * e4 * eta) / 2.24001792e8
        - (7201466570525.0 * e6 * eta) / 5.1072408576e10
        - (5045260598968525.0 * e8 * eta) / 1.9962015694848e13
        - (484345.0 * e2 * eta * eta) / 35088.0
        - (1609478435.0 * e4 * eta * eta) / 3.2000256e7
        - (818438915825.0 * e6 * eta * eta) / 7.296058368e9
        - (4013719013988775.0 * e8 * eta * eta) / 1.9962015694848e13
    )


def _c18(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p2 = pow(f0, 19.0 / 3.0)
    return p2 * (
        (-110474729572613560552840685.0 * e6) / 1.0847519283665837e24
        - (367107526369794861717089596255.0 * e8) / 3.297645862234414e26
        - (196087423156943883913505.0 * e6 * eta) / 1.07614278607796e21
        - (651598507150524526244577115.0 * e8 * eta) / 3.271474069676998e23
        - (22285124348468295519365.0 * e6 * eta * eta) / 1.5373468372542285e20
        - (74053468209960146010849895.0 * e8 * eta * eta) / 4.673534385252855e22
    )


def _c19(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p3 = pow(f0, 38.0 / 9.0)
    return p3 * (
        (6841716503626986565.0 * e4) / 2.711858126051451e17
        + (22735023941552476355495.0 * e6) / 1.2366073054794616e20
        + (168036723934429498871218085.0 * e8) / 2.255571725194538e23
        + (12143723404950745.0 * e4 * eta) / 2.69033544251136e14
        + (40353592874651325635.0 * e6 * eta) / 1.2267929617851802e17
        + (298257242353143912203705.0 * e8 * eta) / 2.2376703622961686e20
        + (1380121079545885.0 * e4 * eta * eta) / 3.8433363464448e13
        + (4586142347330975855.0 * e6 * eta * eta) / 1.7525613739788288e16
        + (33896614207384379043965.0 * e8 * eta * eta) / 3.1966719461373837e19
    )


def _c20(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p4 = pow(f0, 76.0 / 9.0)
    return p4 * (
        (7770499311332403684925585429305516241.0 * e8) / 1.5446267861142245e34
        + (13792268987637324946295665896844693.0 * e8 * eta) / 1.532367843367286e31
        + (1567476508633676458254637292015689.0 * e8 * eta * eta) / 2.189096919096123e30
    )


def _c21(eta):
    return math.pi * (38645.0 / 756.0 - 65.0 * eta / 9.0)


def _c22(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p1 = pow(f0, 19.0 / 9.0)
    return (
        math.pi
        * p1
        * (
            (6067265.0 * e2) / 122808.0
            + (20161521595.0 * e4) / 1.12000896e8
            + (10252373388025.0 * e6) / 2.5536204288e10
            + (7182689108386025.0 * e8) / 9.981007847424e12
            - (10205.0 * e2 * eta) / 1462.0
            - (33911215.0 * e4 * eta) / 1.333344e6
            - (17244255925.0 * e6 * eta) / 3.04002432e8
            - (84567823633475.0 * e8 * eta) / 8.31750653952e11
        )
    )


def _c23(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p2 = pow(f0, 19.0 / 3.0)
    return (
        math.pi
        * p2
        * (
            (36828693183369973116475.0 * e6) / 7.686734186271143e19
            + (122381747448338420666046425.0 * e8) / 2.3367671926264273e22
            - (433615096349678814025.0 * e6 * eta) / 6.405611821892619e18
            - (1440902965169982699005075.0 * e8 * eta) / 1.9473059938553564e21
        )
    )


def _c24(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p3 = pow(f0, 38.0 / 9.0)
    return (
        math.pi
        * p3
        * (
            (-2316846009950855.0 * e4) / 1.9216681732224e13
            - (7698879291066691165.0 * e6) / 8.762806869894144e15
            - (56903148963613058870695.0 * e8) / 1.5983359730686919e19
            + (27278171420045.0 * e4 * eta) / 1.601390144352e12
            + (90645363628809535.0 * e6 * eta) / 7.30233905824512e14
            + (669968502482700007405.0 * e8 * eta) / 1.33194664422391e18
        )
    )


def _c25(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p4 = pow(f0, 76.0 / 9.0)
    return (
        math.pi
        * p4
        * (
            (-18041156334765213110425249225007393.0 * e8) / 7.66183921683643e30
            + (2334216112662079584736091244017.0 * e8 * eta) / 7.016336279154241e27
        )
    )


def _c26(eta):
    return (
        11583231236531.0 / 4694215680.0
        - (15737765635.0 * eta) / 3.048192e6
        + (76055.0 * eta * eta) / 1728.0
        - (127825.0 * eta * eta * eta) / 1296.0
        - (6848.0 * _EULER_GAMMA) / 21.0
        - (640.0 * math.pi * math.pi) / 3.0
        + (2255.0 * eta * math.pi * math.pi) / 12.0
    )


def _c27(eta):
    return -6848.0 / 21.0


def _c28(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p1 = pow(f0, 19.0 / 9.0)
    return p1 * (
        (-537568.0 * e2) / 5117.0
        - (111646154.0 * e4) / 291669.0
        - (28386698215.0 * e6) / 3.3250266e7
        - (19887378305015.0 * e8) / 1.2996103968e10
    )


def _c29(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p2 = pow(f0, 19.0 / 3.0)
    return p2 * (
        (-657204451373573967571.0 * e6) / 7.006137930195053e17
        - (2183890391914386294238433.0 * e8) / 2.1298659307792957e20
    )


def _c30(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p3 = pow(f0, 38.0 / 9.0)
    return p3 * (
        (83880153412870.0 * e4) / 3.50304094077e11
        + (139366874895483505.0 * e6) / 7.9869333449556e13
        + (1030073825416757818915.0 * e8) / 1.4568166421199014e17
    )


def _c31(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p4 = pow(f0, 76.0 / 9.0)
    return (
        229018097854706671787019720252139.0 * e8 * p4
    ) / 49881765734612176930620007680.0


def _c32(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p1 = pow(f0, 19.0 / 9.0)
    return p1 * (
        (1578237767500487.0 * e2) / 2.28764777472e12
        + (5244484101404118301.0 * e4) / 2.08633477054464e15
        + (533376501191162085059.0 * e6) / 9.513686553683558e16
        + (373677141943502503806739.0 * e8) / 3.718492344411174e19
        - (2470829204695.0 * e2 * eta) / 1.485485568e9
        - (8210565447201485.0 * e4 * eta) / 1.354762838016e12
        - (4175170127655540575.0 * e6 * eta) / 3.08885927067648e14
        - (2925073821111304814575.0 * e8 * eta) / 1.207302709224407e17
        + (11940635.0 * e2 * eta * eta) / 842112.0
        + (39678730105.0 * e4 * eta * eta) / 7.68006144e8
        + (20177105913475.0 * e6 * eta * eta) / 1.75105400832e11
        + (98950858868368325.0 * e8 * eta * eta) / 4.79088376676352e14
        - (20068525.0 * e2 * eta * eta * eta) / 631584.0
        - (66687708575.0 * e4 * eta * eta * eta) / 5.76004608e8
        - (33911492517125.0 * e6 * eta * eta * eta) / 1.31329050624e11
        - (166305877783829875.0 * e8 * eta * eta * eta) / 3.59316282507264e14
        - (537568.0 * e2 * _EULER_GAMMA) / 5117.0
        - (111646154.0 * e4 * _EULER_GAMMA) / 291669.0
        - (28386698215.0 * e6 * _EULER_GAMMA) / 3.3250266e7
        - (19887378305015.0 * e8 * _EULER_GAMMA) / 1.2996103968e10
        - (50240.0 * e2 * math.pi * math.pi) / 731.0
        - (10434220.0 * e4 * math.pi * math.pi) / 41667.0
        - (1326481225.0 * e6 * math.pi * math.pi) / 2.375019e6
        - (6505217202575.0 * e8 * math.pi * math.pi) / 6.498051984e9
        + (354035.0 * e2 * eta * math.pi * math.pi) / 5848.0
        + (1176458305.0 * e4 * eta * math.pi * math.pi) / 5.333376e6
        + (598243032475.0 * e6 * eta * math.pi * math.pi) / 1.216009728e9
        + (2933852958361325.0 * e8 * eta * math.pi * math.pi) / 3.327002615808e12
    )


def _c33(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p2 = pow(f0, 19.0 / 3.0)
    return p2 * (
        (62534662812190508908451288422363.0 * e6) / 1.0023107818107234e28
        + (207802684524909061102783631427512249.0 * e8) / 3.047024776704599e30
        - (96662893738280943308126058155.0 * e6 * eta) / 6.508511570199502e24
        - (321210795892307574612902891249065.0 * e8 * eta) / 1.9785875173406487e27
        + (8813910165617557415555.0 * e6 * eta * eta) / 6.961570583792733e19
        + (29288623480347143291889265.0 * e8 * eta * eta) / 2.116317457472991e22
        - (785113635484365349577225.0 * e6 * eta * eta * eta) / 2.7672243070576115e21
        - (2608932610714546056645118675.0 * e8 * eta * eta * eta) / 8.412361893455138e23
        - (657204451373573967571.0 * e6 * _EULER_GAMMA) / 7.006137930195053e17
        - (2183890391914386294238433.0 * e8 * _EULER_GAMMA) / 2.1298659307792957e20
        - (30710488381942708765.0 * e6 * math.pi * math.pi) / 5.004384235853609e16
        - (102050952893195621226095.0 * e8 * math.pi * math.pi) / 1.5213328076994972e19
        + (13850430260256161653015.0 * e6 * eta * math.pi * math.pi)
        / 2.5622447287570477e19
        + (46024979754831225172968845.0 * e8 * eta * math.pi * math.pi)
        / 7.789223975421425e21
    )


def _c34(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p3 = pow(f0, 38.0 / 9.0)
    return p3 * (
        (-796520595220381729758415.0 * e4) / 5.0115138169430816e20
        - (2646837937917328487987213045.0 * e6) / 2.285250300526045e23
        - (19563030899655064495340095274735.0 * e8) / 4.168296548159506e26
        + (6168627083362586227675.0 * e4 * eta) / 1.6271148756308705e18
        + (20498347798013874034564025.0 * e6 * eta) / 7.41964383287677e20
        + (151505237861278885566710654075.0 * e8 * eta) / 1.3533430351167228e24
        - (562467383866675.0 * e4 * eta * eta) / 1.7403787229184e13
        - (1869079116588961025.0 * e6 * eta * eta) / 7.936126976507904e15
        - (13814541490402312805075.0 * e8 * eta * eta) / 1.4475495605150417e19
        + (50102713130841625.0 * e4 * eta * eta * eta) / 6.91800542360064e14
        + (166491315733786719875.0 * e6 * eta * eta * eta) / 3.154610473161892e17
        + (1230553147045766992549625.0 * e8 * eta * eta * eta) / 5.754009503047291e20
        + (83880153412870.0 * e4 * _EULER_GAMMA) / 3.50304094077e11
        + (139366874895483505.0 * e6 * _EULER_GAMMA) / 7.9869333449556e13
        + (1030073825416757818915.0 * e8 * _EULER_GAMMA) / 1.4568166421199014e17
        + (7839266674100.0 * e4 * math.pi * math.pi) / 5.0043442011e10
        + (6512470789508575.0 * e6 * math.pi * math.pi) / 5.704952389254e12
        + (48134290907325131725.0 * e8 * math.pi * math.pi) / 1.0405833157999296e16
        - (883877317504775.0 * e4 * eta * math.pi * math.pi) / 6.405560577408e12
        - (2937124326068367325.0 * e6 * eta * math.pi * math.pi) / 2.920935623298048e15
        - (21708565199203634407975.0 * e8 * eta * math.pi * math.pi)
        / 5.32778657689564e18
    )


def _c35(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p4 = pow(f0, 76.0 / 9.0)
    return p4 * (
        (-21803333020968369933785155127082363071942707.0 * e8) / 7.136175751847717e38
        + (6736884392917513798607444845703601288679.0 * e8 * eta) / 9.267760716685346e34
        - (614282187703745932737146352451999.0 * e8 * eta * eta) / 9.91289170911452e29
        + (54718202538837159478833263067719005.0 * e8 * eta * eta * eta)
        / 3.940374454373021e31
        + (229018097854706671787019720252139.0 * e8 * _EULER_GAMMA)
        / 4.988176573461218e28
        + (2140356054716884783056259067777.0 * e8 * math.pi * math.pi)
        / 7.125966533516026e26
        - (965300580677315037158372839567427.0 * e8 * eta * math.pi * math.pi)
        / 3.648494865160205e29
    )


def _c36(eta):
    return math.pi * (
        77096675.0 / 254016.0
        + (378515.0 * eta) / 1512.0
        - (74045.0 * eta * eta) / 756.0
    )


def _c37(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p1 = pow(f0, 19.0 / 9.0)
    return (
        math.pi
        * p1
        * (
            (12104177975.0 * e2) / 6.1895232e7
            + (40222183410925.0 * e4) / 5.6448451584e10
            + (20453458379485375.0 * e6) / 1.2870246961152e13
            + (14329446184895255375.0 * e8) / 5.030427955101696e15
            + (59426855 * e2 * eta) / 368424.0
            + (197475439165.0 * e4 * eta) / 3.36002688e8
            + (100418608176175.0 * e6 * eta) / 7.6608612864e10
            + (70352065412362175.0 * e8 * eta) / 2.9943023542272e13
            - (11625065.0 * e2 * eta * eta) / 184212.0
            - (38630090995.0 * e4 * eta * eta) / 1.68001344e8
            - (19643860461025.0 * e6 * eta * eta) / 3.8304306432e10
            - (13762251650419025.0 * e8 * eta * eta) / 1.4971511771136e13
        )
    )


def _c38(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p2 = pow(f0, 19.0 / 3.0)
    return (
        math.pi
        * p2
        * (
            (434593322826290843757317275.0 * e6) / 2.7118798209164593e23
            + (1444153611751764473805565304825.0 * e8) / 8.244114655586036e25
            + (2133685941573919740699595.0 * e6 * eta) / 1.61421417911694e21
            + (7090238383850135298344754185.0 * e8 * eta) / 4.907211104515498e23
            - (417391055952448085809285.0 * e6 * eta * eta) / 8.0710708955847e20
            - (1386990478929984989144254055.0 * e8 * eta * eta) / 2.453605552257749e23
        )
    )


def _c39(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p3 = pow(f0, 38.0 / 9.0)
    return (
        math.pi
        * p3
        * (
            (-28083426938595740975.0 * e4) / 6.779645315128627e16
            - (93321227716953647259925.0 * e6) / 3.091518263698654e19
            - (689746068418917003152253775.0 * e8) / 5.638929312986345e22
            - (137878817052260255.0 * e4 * eta) / 4.03550316376704e14
            - (458171309064660827365.0 * e6 * eta) / 1.8401894426777702e17
            - (3386387715003096689295295.0 * e8 * eta) / 3.3565055434442526e20
            + (26971816199185265.0 * e4 * eta * eta) / 2.01775158188352e14
            + (89627345229892635595.0 * e6 * eta * eta) / 9.200947213388851e16
            + (662444231688055412226385.0 * e8 * eta * eta) / 1.6782527717221263e20
        )
    )


def _c40(eta, e0, f0):
    e2 = e0 * e0
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e6 * e2
    p4 = pow(f0, 76.0 / 9.0)
    return (
        math.pi
        * p4
        * (
            (-1776389932392373211125413523490219335.0 * e8) / 2.2715099795797417e32
            - (8721390841557034022662273046715703.0 * e8 * eta) / 1.3520892735593702e30
            + (1706076073241722479183197515934809.0 * e8 * eta * eta)
            / 6.760446367796851e29
        )
    )


def _z1(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p5 = pow(f0, 133.0 / 18.0)
    return p5 * (
        (46031066168471.0 * c2b * e7 * plus_response) / (1.2136808448e11)
        + (46031066168471.0 * c2b * c2i * e7 * plus_response) / (1.2136808448e11)
        - (46031066168471.0 * ci * e7 * cross_response * s2b) / (6.068404224e10)
        - (8391437082143.0 * e7 * plus_response * s2i) / (3.6410425344e10)
    )


def _z2(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p6 = pow(f0, 95.0 / 18.0)
    return p6 * (
        -(717415013.0 * c2b * e5 * plus_response) / (1.3307904e7)
        - (717415013.0 * c2b * c2i * e5 * plus_response) / (1.3307904e7)
        - (11919850440995.0 * c2b * e7 * plus_response) / (2.4273616896e10)
        - (11919850440995.0 * c2b * c2i * e7 * plus_response) / (2.4273616896e10)
        + (717415013.0 * ci * e5 * cross_response * s2b) / (6.653952e6)
        + (11919850440995.0 * ci * e7 * cross_response * s2b) / (1.2136808448e10)
        + (220389695.0 * e5 * plus_response * s2i) / (6.653952e6)
        + (3661774782425.0 * e7 * plus_response * s2i) / (1.2136808448e10)
    )


def _z3(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p7 = pow(f0, 19.0 / 6.0)
    return p7 * (
        (30299.0 * c2b * e3 * plus_response) / (3648.0)
        + (30299.0 * c2b * c2i * e3 * plus_response) / (3648.0)
        + (100683577.0 * c2b * e5 * plus_response) / (2.217984e6)
        + (100683577.0 * c2b * c2i * e5 * plus_response) / (2.217984e6)
        + (384584085937.0 * c2b * e7 * plus_response) / (2.697068544e9)
        + (384584085937.0 * c2b * c2i * e7 * plus_response) / (2.697068544e9)
        - (30299.0 * ci * e3 * cross_response * s2b) / (1824.0)
        - (100683577.0 * ci * e5 * cross_response * s2b) / (1.108992e6)
        - (384584085937.0 * ci * e7 * cross_response * s2b) / (1.348534272e9)
        - (9517.0 * e3 * plus_response * s2i) / (1824.0)
        - (31624991.0 * e5 * plus_response * s2i) / (1.108992e6)
        - (120798928871.0 * e7 * plus_response * s2i) / (1.348534272e9)
    )


def _z4(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p8 = pow(f0, 19.0 / 18.0)
    return p8 * (
        -(3.0 * c2b * e0 * plus_response) / (2.0)
        - (3.0 * c2b * c2i * e0 * plus_response) / (2.0)
        - (3323.0 * c2b * e3 * plus_response) / (1216.0)
        - (3323.0 * c2b * c2i * e3 * plus_response) / (1216.0)
        - (15994231.0 * c2b * e5 * plus_response) / (4.435968e6)
        - (15994231.0 * c2b * c2i * e5 * plus_response) / (4.435968e6)
        - (105734339801.0 * c2b * e7 * plus_response) / (2.4273616896e10)
        - (105734339801.0 * c2b * c2i * e7 * plus_response) / (2.4273616896e10)
        + 3.0 * ci * e0 * cross_response * s2b
        + (3323.0 * ci * e3 * cross_response * s2b) / (608.0)
        + (15994231.0 * ci * e5 * cross_response * s2b) / (2.217984e6)
        + (105734339801.0 * ci * e7 * cross_response * s2b) / (1.2136808448e10)
        + e0 * plus_response * s2i
        + (3323.0 * e3 * plus_response * s2i) / (1824.0)
        + (15994231.0 * e5 * plus_response * s2i) / (6.653952e6)
        + (105734339801.0 * e7 * plus_response * s2i) / (3.6410425344e10)
    )


def _z5(f0, inc, bet, f_plus, f_cross):
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    return (
        2.0 * c2b * plus_response
        + 2.0 * c2b * c2i * plus_response
        - 4.0 * ci * cross_response * s2b
    )


def _z6(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p1 = pow(f0, 19.0 / 9.0)
    return p1 * (
        -(277.0 * c2b * e2 * plus_response) / (24.0)
        - (277.0 * c2b * c2i * e2 * plus_response) / (24.0)
        - (920471.0 * c2b * e4 * plus_response) / (21888.0)
        - (920471.0 * c2b * c2i * e4 * plus_response) / (21888.0)
        - (468070445.0 * c2b * e6 * plus_response) / (4.990464e6)
        - (468070445.0 * c2b * c2i * e6 * plus_response) / (4.990464e6)
        - (2295471547915.0 * c2b * e8 * plus_response) / (1.3653909504e10)
        - (2295471547915.0 * c2b * c2i * e8 * plus_response) / (1.3653909504e10)
        + e2 * plus_response * s2i
        + (3323.0 * e4 * plus_response * s2i) / (912.0)
        + (1689785.0 * e6 * plus_response * s2i) / (207936.0)
        + (8286900895.0 * e8 * plus_response * s2i) / (5.68912896e8)
        + (277.0 * ci * e2 * cross_response * s2b) / (12.0)
        + (920471.0 * ci * e4 * cross_response * s2b) / (10944.0)
        + (468070445.0 * ci * e6 * cross_response * s2b) / (2.495232e6)
        + (2295471547915.0 * ci * e8 * cross_response * s2b) / (6.826954752e9)
    )


def _z7(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p2 = pow(f0, 19.0 / 3.0)
    return p2 * (
        -(103729904239.0 * c2b * e6 * plus_response) / (1.9961856e8)
        - (103729904239.0 * c2b * c2i * e6 * plus_response) / (1.9961856e8)
        - (344694471786197.0 * c2b * e8 * plus_response) / (6.068404224e10)
        - (344694471786197.0 * c2b * c2i * e8 * plus_response) / (6.068404224e10)
        + (103729904239.0 * ci * e6 * cross_response * s2b) / (9.980928e7)
        + (344694471786197.0 * ci * e8 * cross_response * s2b) / (3.034202112e10)
        + (29064841.0 * e6 * plus_response * s2i) / (554496.0)
        + (96582466643.0 * e8 * plus_response * s2i) / (1.68566784e8)
    )


def _z8(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p3 = pow(f0, 38.0 / 9.0)
    return p3 * (
        (3254599.0 * c2b * e4 * plus_response) / (43776.0)
        + (3254599.0 * c2b * c2i * e4 * plus_response) / (43776.0)
        + (10815032477.0 * c2b * e6 * plus_response) / (1.9961856e7)
        + (10815032477.0 * c2b * c2i * e6 * plus_response) / (1.9961856e7)
        + (79934933490791.0 * c2b * e8 * plus_response) / (3.6410425344e10)
        + (79934933490791.0 * c2b * c2i * e8 * plus_response) / (3.6410425344e10)
        - (3254599.0 * ci * e4 * cross_response * s2b) / (21888.0)
        - (10815032477.0 * ci * e6 * cross_response * s2b) / (9.980928e6)
        - (79934933490791.0 * ci * e8 * cross_response * s2b) / (1.8205212672e10)
        - (3305.0 * e4 * plus_response * s2i) / (456.0)
        - (10982515.0 * e6 * plus_response * s2i) / (207936.0)
        - (81172812745.0 * e8 * plus_response * s2i) / (3.79275264e8)
    )


def _z9(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p4 = pow(f0, 76.0 / 9.0)
    return p4 * (
        (8340362777769439.0 * c2b * e8 * plus_response) / (2.18462552064e12)
        + (8340362777769439.0 * c2b * c2i * e8 * plus_response) / (2.18462552064e12)
        - (8340362777769439.0 * ci * e8 * cross_response * s2b) / (1.09231276032e12)
        - (4442498396267.0 * e8 * plus_response * s2i) / (1.137825792e10)
    )


def _z10(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p5 = pow(f0, 133.0 / 18.0)
    return p5 * (
        (-74006050878931.0 * c2b * e7 * plus_response) / (4.045602816e10)
        - (74006050878931.0 * c2b * c2i * e7 * plus_response) / (4.045602816e10)
        + (74006050878931.0 * ci * e7 * cross_response * s2b) / (2.022801408e10)
        + (2531702819.0 * e7 * plus_response * s2i) / (2.957312e7)
    )


def _z11(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p6 = pow(f0, 95.0 / 18.0)
    return p6 * (
        (1810486747.0 * c2b * e5 * plus_response) / (7.39328e6)
        + (1810486747.0 * c2b * c2i * e5 * plus_response) / (7.39328e6)
        + (6016247460281.0 * c2b * e7 * plus_response) / (2.697068544e9)
        + (6016247460281.0 * c2b * c2i * e7 * plus_response) / (2.697068544e9)
        - (1810486747.0 * ci * e5 * cross_response * s2b) / (3.69664e6)
        - (6016247460281.0 * ci * e7 * cross_response * s2b) / (1.348534272e9)
        - (50883.0 * e5 * plus_response * s2i) / (4864.0)
        - (281807015.0 * e7 * plus_response * s2i) / (2.957312e6)
    )


def _z12(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p7 = pow(f0, 19.0 / 6.0)
    return p7 * (
        -(40863.0 * c2b * e3 * plus_response) / (1216.0)
        - (40863.0 * c2b * c2i * e3 * plus_response) / (1216.0)
        - (135787749.0 * c2b * e5 * plus_response) / (739328.0)
        - (135787749.0 * c2b * c2i * e5 * plus_response) / (739328.0)
        - (518672547069.0 * c2b * e7 * plus_response) / (8.99022848e8)
        - (518672547069.0 * c2b * c2i * e7 * plus_response) / (8.99022848e8)
        + (40863.0 * ci * e3 * cross_response * s2b) / (608.0)
        + (135787749.0 * ci * e5 * cross_response * s2b) / (369664.0)
        + (518672547069.0 * ci * e7 * cross_response * s2b) / (4.49511424e8)
        + (9.0 * e3 * plus_response * s2i) / (8.0)
        + (29907.0 * e5 * plus_response * s2i) / (4864.0)
        + (114236667.0 * e7 * plus_response * s2i) / (5.914624e6)
    )


def _z13(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p8 = pow(f0, 19.0 / 18.0)
    return p8 * (
        (9.0 * c2b * e0 * plus_response) / (2.0)
        + (9.0 * c2b * c2i * e0 * plus_response) / (2.0)
        + (9969.0 * c2b * e3 * plus_response) / (1216.0)
        + (9969.0 * c2b * c2i * e3 * plus_response) / (1216.0)
        + (15994231.0 * c2b * e5 * plus_response) / (1.478656e6)
        + (15994231.0 * c2b * c2i * e5 * plus_response) / (1.478656e6)
        + (105734339801.0 * c2b * e7 * plus_response) / (8.091205632e9)
        + (105734339801.0 * c2b * c2i * e7 * plus_response) / (8.091205632e9)
        - 9.0 * ci * e0 * cross_response * s2b
        - (9969.0 * ci * e3 * cross_response * s2b) / (608.0)
        - (15994231.0 * ci * e5 * cross_response * s2b) / (739328.0)
        - (105734339801.0 * ci * e7 * cross_response * s2b) / (4.045602816e9)
    )


def _z14(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p1 = pow(f0, 19.0 / 9.0)
    return p1 * (
        8.0 * c2b * e2 * plus_response
        + 8.0 * c2b * c2i * e2 * plus_response
        + (3323.0 * c2b * e4 * plus_response) / (114.0)
        + (3323.0 * c2b * c2i * e4 * plus_response) / (114.0)
        + (1689785.0 * c2b * e6 * plus_response) / (25992.0)
        + (1689785.0 * c2b * c2i * e6 * plus_response) / (25992.0)
        + (8286900895.0 * c2b * e8 * plus_response) / (7.1114112e7)
        + (8286900895.0 * c2b * c2i * e8 * plus_response) / (7.1114112e7)
        - 16.0 * ci * e2 * cross_response * s2b
        - (3323.0 * ci * e4 * cross_response * s2b) / (57.0)
        - (1689785.0 * ci * e6 * cross_response * s2b) / (12996.0)
        - (8286900895.0 * ci * e8 * cross_response * s2b) / (3.5557056e7)
    )


def _z15(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p2 = pow(f0, 19.0 / 3.0)
    return p2 * (
        (71474367.0 * c2b * e6 * plus_response) / (115520.0)
        + (71474367.0 * c2b * c2i * e6 * plus_response) / (115520.0)
        + (237509321541.0 * c2b * e8 * plus_response) / (3.511808e7)
        + (237509321541.0 * c2b * c2i * e8 * plus_response) / (3.511808e7)
        - (71474367.0 * ci * e6 * cross_response * s2b) / (57760.0)
        - (237509321541.0 * ci * e8 * cross_response * s2b) / (1.755904e7)
        - (51793.0 * e6 * plus_response * s2i) / (3420.0)
        - (172108139.0 * e8 * plus_response * s2i) / (1.03968e6)
    )


def _z16(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p3 = pow(f0, 38.0 / 9.0)
    return p3 * (
        -(1431.0 * c2b * e4 * plus_response) / (19.0)
        - (1431.0 * c2b * c2i * e4 * plus_response) / (19.0)
        - (1585071.0 * c2b * e6 * plus_response) / (2888.0)
        - (1585071.0 * c2b * c2i * e6 * plus_response) / (2888.0)
        - (3905136831.0 * c2b * e8 * plus_response) / (1.755904e6)
        - (3905136831.0 * c2b * c2i * e8 * plus_response) / (1.755904e6)
        + (2862.0 * ci * e4 * cross_response * s2b) / (19.0)
        + (1585071.0 * ci * e6 * cross_response * s2b) / (1444.0)
        + (3905136831.0 * ci * e8 * cross_response * s2b) / (877952.0)
        + (4.0 * e4 * plus_response * s2i) / (3.0)
        + (3323.0 * e6 * plus_response * s2i) / (342.0)
        + (24560609.0 * e8 * plus_response * s2i) / (623808.0)
    )


def _z17(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p4 = pow(f0, 76.0 / 9.0)
    return p4 * (
        (-49436673769121.0 * c2b * e8 * plus_response) / (9.95597568e9)
        - (49436673769121.0 * c2b * c2i * e8 * plus_response) / (9.95597568e9)
        + (865873231.0 * e8 * plus_response * s2i) / (6.23808e6)
        + (49436673769121.0 * ci * e8 * cross_response * s2b) / (4.97798784e9)
    )


def _z18(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p5 = pow(f0, 133.0 / 18.0)
    return p5 * (
        (3053741715625.0 * c2b * e7 * plus_response) / (2.235727872e9)
        + (3053741715625.0 * c2b * c2i * e7 * plus_response) / (2.235727872e9)
        - (3053741715625.0 * ci * e7 * cross_response * s2b) / (1.117863936e9)
        - (15300625.0 * e7 * plus_response * s2i) / (700416.0)
    )


def _z19(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p6 = pow(f0, 95.0 / 18.0)
    return p6 * (
        -(13023125.0 * c2b * e5 * plus_response) / (87552.0)
        - (13023125.0 * c2b * c2i * e5 * plus_response) / (87552.0)
        - (216379221875.0 * c2b * e7 * plus_response) / (1.59694848e8)
        - (216379221875.0 * c2b * c2i * e7 * plus_response) / (1.59694848e8)
        + (13023125.0 * ci * e5 * cross_response * s2b) / (43776.0)
        + (216379221875.0 * ci * e7 * cross_response * s2b) / (7.9847424e7)
        + (625.0 * e5 * plus_response * s2i) / (384.0)
        + (10384375.0 * e7 * plus_response * s2i) / (700416.0)
    )


def _z20(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p7 = pow(f0, 19.0 / 6.0)
    return p7 * (
        (625.0 * c2b * e3 * plus_response) / (48.0)
        + (625.0 * c2b * c2i * e3 * plus_response) / (48.0)
        + (2076875.0 * c2b * e5 * plus_response) / (29184.0)
        + (2076875.0 * c2b * c2i * e5 * plus_response) / (29184.0)
        + (7933101875.0 * c2b * e7 * plus_response) / (3.5487744e7)
        + (7933101875.0 * c2b * c2i * e7 * plus_response) / (3.5487744e7)
        - (625.0 * ci * e3 * cross_response * s2b) / (24.0)
        - (2076875.0 * ci * e5 * cross_response * s2b) / (14592.0)
        - (7933101875.0 * ci * e7 * cross_response * s2b) / (1.7743872e7)
    )


def _z21(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p2 = pow(f0, 19.0 / 3.0)
    return p2 * (
        -(1656963.0 * c2b * e6 * plus_response) / (6080.0)
        - (1656963.0 * c2b * c2i * e6 * plus_response) / (6080.0)
        - (5506088049.0 * c2b * e8 * plus_response) / (1.84832e6)
        - (5506088049.0 * c2b * c2i * e8 * plus_response) / (1.84832e6)
        + (1656963.0 * ci * e6 * cross_response * s2b) / (3040.0)
        + (5506088049.0 * ci * e8 * cross_response * s2b) / (924160.0)
        + (81.0 * e6 * plus_response * s2i) / (40.0)
        + (269163.0 * e8 * plus_response * s2i) / (12160.0)
    )


def _z22(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p3 = pow(f0, 38.0 / 9.0)
    return p3 * (
        (81.0 * c2b * e4 * plus_response) / (4.0)
        + (81.0 * c2b * c2i * e4 * plus_response) / (4.0)
        + (89721.0 * c2b * e6 * plus_response) / (608.0)
        + (89721.0 * c2b * c2i * e6 * plus_response) / (608.0)
        + (221045481.0 * c2b * e8 * plus_response) / (369664.0)
        + (221045481.0 * c2b * c2i * e8 * plus_response) / (369664.0)
        - (81.0 * ci * e4 * cross_response * s2b) / (2.0)
        - (89721.0 * ci * e6 * cross_response * s2b) / (304.0)
        - (221045481.0 * ci * e8 * cross_response * s2b) / (184832.0)
    )


def _z23(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p4 = pow(f0, 76.0 / 9.0)
    return p4 * (
        (71730555921.0 * c2b * e8 * plus_response) / (2.587648e7)
        + (71730555921.0 * c2b * c2i * e8 * plus_response) / (2.587648e7)
        - (333693.0 * e8 * plus_response * s2i) / (10640.0)
        - (71730555921.0 * ci * e8 * cross_response * s2b) / (1.293824e7)
    )


def _z24(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p5 = pow(f0, 133.0 / 18.0)
    return p5 * (
        (-1109077123.0 * c2b * e7 * plus_response) / (2.33472e6)
        - (1109077123.0 * c2b * c2i * e7 * plus_response) / (2.33472e6)
        + (1109077123.0 * ci * e7 * cross_response * s2b) / (1.16736e6)
        + (117649.0 * e7 * plus_response * s2i) / (46080.0)
    )


def _z25(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p6 = pow(f0, 95.0 / 18.0)
    return p6 * (
        (117649.0 * c2b * e5 * plus_response) / (3840.0)
        + (117649.0 * c2b * c2i * e5 * plus_response) / (3840.0)
        + (390947627.0 * c2b * e7 * plus_response) / (1.400832e6)
        + (390947627.0 * c2b * c2i * e7 * plus_response) / (1.400832e6)
        - (117649.0 * ci * e5 * cross_response * s2b) / (1920.0)
        - (390947627.0 * ci * e7 * cross_response * s2b) / (700416.0)
    )


def _z26(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p2 = pow(f0, 19.0 / 3.0)
    return p2 * (
        (2048.0 * c2b * e6 * plus_response) / (45.0)
        + (2048.0 * c2b * c2i * e6 * plus_response) / (45.0)
        + (425344.0 * c2b * e8 * plus_response) / (855.0)
        + (425344.0 * c2b * c2i * e8 * plus_response) / (855.0)
        - (4096.0 * ci * e6 * cross_response * s2b) / (45.0)
        - (850688.0 * ci * e8 * cross_response * s2b) / (855.0)
    )


def _z27(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    si = math.sin(inc)
    ci = math.cos(inc)
    s2i = si * si
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p4 = pow(f0, 76.0 / 9.0)
    return p4 * (
        (-14348288.0 * c2b * e8 * plus_response) / (17955.0)
        - (14348288.0 * c2b * c2i * e8 * plus_response) / (17955.0)
        + (28696576.0 * ci * e8 * cross_response * s2b) / (17955.0)
        + (1024.0 * e8 * plus_response * s2i) / (315.0)
    )


def _z28(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p5 = pow(f0, 133.0 / 18.0)
    return p5 * (
        (4782969.0 * c2b * e7 * plus_response) / (71680.0)
        + (4782969.0 * c2b * c2i * e7 * plus_response) / (71680.0)
        - (4782969.0 * ci * e7 * cross_response * s2b) / (35840.0)
    )


def _z29(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p4 = pow(f0, 76.0 / 9.0)
    return (
        390625.0
        * e8
        * p4
        * (
            c2b * plus_response
            + c2b * c2i * plus_response
            - 2.0 * ci * cross_response * s2b
        )
    ) / (4032.0)


def _q1(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p5 = pow(f0, 133.0 / 18.0)
    return p5 * (
        (16099508821573.0 * c2b * ci * e7 * cross_response) / (2.022801408e10)
        + (16099508821573.0 * e7 * plus_response * s2b) / (4.045602816e10)
        + (16099508821573.0 * c2i * e7 * plus_response * s2b) / (4.045602816e10)
    )


def _q2(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p6 = pow(f0, 95.0 / 18.0)
    return p6 * (
        -(749695861.0 * c2b * ci * e5 * cross_response) / (6.653952e6)
        - (12456196730515.0 * c2b * ci * e7 * cross_response) / (1.2136808448e10)
        - (749695861.0 * e5 * plus_response * s2b) / (1.3307904e7)
        - (749695861.0 * c2i * e5 * plus_response * s2b) / (1.3307904e7)
        - (12456196730515.0 * e7 * plus_response * s2b) / (2.4273616896e10)
        - (12456196730515.0 * c2i * e7 * plus_response * s2b) / (2.4273616896e10)
    )


def _q3(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p7 = pow(f0, 19.0 / 6.0)
    return p7 * (
        (31363.0 * c2b * ci * e3 * cross_response) / (1824.0)
        + (104219249.0 * c2b * ci * e5 * cross_response) / (1.108992e6)
        + (398089398569.0 * c2b * ci * e7 * cross_response) / (1.348534272e9)
        + (31363.0 * e3 * plus_response * s2b) / (3648.0)
        + (31363.0 * c2i * e3 * plus_response * s2b) / (3648.0)
        + (104219249.0 * e5 * plus_response * s2b) / (2.217984e6)
        + (104219249.0 * c2i * e5 * plus_response * s2b) / (2.217984e6)
        + (398089398569.0 * e7 * plus_response * s2b) / (2.697068544e9)
        + (398089398569.0 * c2i * e7 * plus_response * s2b) / (2.697068544e9)
    )


def _q4(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p8 = pow(f0, 19.0 / 18.0)
    return p8 * (
        -3.0 * c2b * ci * e0 * cross_response
        - (3323.0 * c2b * ci * e3 * cross_response) / (608.0)
        - (15994231.0 * c2b * ci * e5 * cross_response) / (2.217984e6)
        - (105734339801.0 * c2b * ci * e7 * cross_response) / (1.2136808448e10)
        - (3.0 * e0 * plus_response * s2b) / 2.0
        - (3.0 * c2i * e0 * plus_response * s2b) / 2.0
        - (3323.0 * e3 * plus_response * s2b) / 1216.0
        - (3323.0 * c2i * e3 * plus_response * s2b) / (1216.0)
        - (15994231.0 * e5 * plus_response * s2b) / (4.435968e6)
        - (15994231.0 * c2i * e5 * plus_response * s2b) / (4.435968e6)
        - (105734339801.0 * e7 * plus_response * s2b) / (2.4273616896e10)
        - (105734339801.0 * c2i * e7 * plus_response * s2b) / (2.4273616896e10)
    )


def _q5(f0, inc, bet, f_plus, f_cross):
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    return (
        4.0 * c2b * ci * cross_response
        + 2.0 * plus_response * s2b
        + 2.0 * c2i * plus_response * s2b
    )


def _q6(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p1 = pow(f0, 19.0 / 9.0)
    return p1 * (
        -(277.0 * c2b * ci * e2 * cross_response) / (12.0)
        - (920471.0 * c2b * ci * e4 * cross_response) / (10944.0)
        - (468070445.0 * c2b * ci * e6 * cross_response) / (2.495232e6)
        - (2295471547915.0 * c2b * ci * e8 * cross_response) / (6.826954752e9)
        - (277.0 * e2 * plus_response * s2b) / (24.0)
        - (277.0 * c2i * e2 * plus_response * s2b) / (24.0)
        - (920471.0 * e4 * plus_response * s2b) / (21888.0)
        - (920471.0 * c2i * e4 * plus_response * s2b) / (21888.0)
        - (468070445.0 * e6 * plus_response * s2b) / (4.990464e6)
        - (468070445.0 * c2i * e6 * plus_response * s2b) / (4.990464e6)
        - (2295471547915.0 * e8 * plus_response * s2b) / (1.3653909504e10)
        - (2295471547915.0 * c2i * e8 * plus_response * s2b) / (1.3653909504e10)
    )


def _q7(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p2 = pow(f0, 19.0 / 3.0)
    return p2 * (
        -(104238504751.0 * c2b * ci * e6 * cross_response) / (9.980928e7)
        - (346384551287573.0 * c2b * ci * e8 * cross_response) / (3.034202112e10)
        - (104238504751.0 * e6 * plus_response * s2b) / (1.9961856e8)
        - (104238504751.0 * c2i * e6 * plus_response * s2b) / (1.9961856e8)
        - (346384551287573.0 * e8 * plus_response * s2b) / (6.068404224e10)
        - (346384551287573.0 * c2i * e8 * plus_response * s2b) / (6.068404224e10)
    )


def _q8(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p3 = pow(f0, 38.0 / 9.0)
    return p3 * (
        (3265543.0 * c2b * ci * e4 * cross_response) / (21888.0)
        + (10851399389.0 * c2b * ci * e6 * cross_response) / (9.980928e6)
        + (80203724795687.0 * c2b * ci * e8 * cross_response) / (1.8205212672e10)
        + (3265543.0 * e4 * plus_response * s2b) / (43776.0)
        + (3265543.0 * c2i * e4 * plus_response * s2b) / (43776.0)
        + (10851399389.0 * e6 * plus_response * s2b) / (1.9961856e7)
        + (10851399389.0 * c2i * e6 * plus_response * s2b) / (1.9961856e7)
        + (80203724795687.0 * e8 * plus_response * s2b) / (3.6410425344e10)
        + (80203724795687.0 * c2i * e8 * plus_response * s2b) / (3.6410425344e10)
    )


def _q9(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p4 = pow(f0, 76.0 / 9.0)
    return p4 * (
        (8388155956587871.0 * e8 * plus_response * s2b) / (2.18462552064e12)
        + (8388155956587871.0 * c2i * e8 * plus_response * s2b) / (2.18462552064e12)
        + (8388155956587871.0 * c2b * ci * e8 * cross_response) / (1.09231276032e12)
    )


def _q10(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p5 = pow(f0, 133.0 / 18.0)
    return p5 * (
        (-74123294656819.0 * c2b * ci * e7 * cross_response) / (2.022801408e10)
        - (74123294656819.0 * e7 * plus_response * s2b) / (4.045602816e10)
        - (74123294656819.0 * c2i * e7 * plus_response * s2b) / (4.045602816e10)
    )


def _q11(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p6 = pow(f0, 95.0 / 18.0)
    return p6 * (
        (1812254203.0 * c2b * ci * e5 * cross_response) / (3.69664e6)
        + (6022120716569.0 * c2b * ci * e7 * cross_response) / (1.348534272e9)
        + (1812254203.0 * e5 * plus_response * s2b) / (7.39328e6)
        + (1812254203.0 * c2i * e5 * plus_response * s2b) / (7.39328e6)
        + (6022120716569.0 * e7 * plus_response * s2b) / (2.697068544e9)
        + (6022120716569.0 * c2i * e7 * plus_response * s2b) / (2.697068544e9)
    )


def _q12(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p7 = pow(f0, 19.0 / 6.0)
    return p7 * (
        -(40863.0 * c2b * ci * e3 * cross_response) / (608.0)
        - (135787749.0 * c2b * ci * e5 * cross_response) / (369664.0)
        - (518672547069.0 * c2b * ci * e7 * cross_response) / (4.49511424e8)
        - (40863.0 * e3 * plus_response * s2b) / (1216.0)
        - (40863.0 * c2i * e3 * plus_response * s2b) / (1216.0)
        - (135787749.0 * e5 * plus_response * s2b) / (739328.0)
        - (135787749.0 * c2i * e5 * plus_response * s2b) / (739328.0)
        - (518672547069.0 * e7 * plus_response * s2b) / (8.99022848e8)
        - (518672547069.0 * c2i * e7 * plus_response * s2b) / (8.99022848e8)
    )


def _q13(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p8 = pow(f0, 19.0 / 18.0)
    return p8 * (
        9.0 * c2b * ci * e0 * cross_response
        + (9969.0 * c2b * ci * e3 * cross_response) / (608.0)
        + (15994231.0 * c2b * ci * e5 * cross_response) / (739328.0)
        + (105734339801.0 * c2b * ci * e7 * cross_response) / (4.045602816e9)
        + (9.0 * e0 * plus_response * s2b) / (2.0)
        + (9.0 * c2i * e0 * plus_response * s2b) / (2.0)
        + (9969.0 * e3 * plus_response * s2b) / (1216.0)
        + (9969.0 * c2i * e3 * plus_response * s2b) / (1216.0)
        + (15994231.0 * e5 * plus_response * s2b) / (1.478656e6)
        + (15994231.0 * c2i * e5 * plus_response * s2b) / (1.478656e6)
        + (105734339801.0 * e7 * plus_response * s2b) / (8.091205632e9)
        + (105734339801.0 * c2i * e7 * plus_response * s2b) / (8.091205632e9)
    )


def _q14(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p1 = pow(f0, 19.0 / 9.0)
    return p1 * (
        16.0 * c2b * ci * e2 * cross_response
        + (3323.0 * c2b * ci * e4 * cross_response) / (57.0)
        + (1689785.0 * c2b * ci * e6 * cross_response) / (12996.0)
        + (8286900895.0 * c2b * ci * e8 * cross_response) / (3.5557056e7)
        + 8.0 * e2 * plus_response * s2b
        + 8.0 * c2i * e2 * plus_response * s2b
        + (3323.0 * e4 * plus_response * s2b) / (114.0)
        + (3323.0 * c2i * e4 * plus_response * s2b) / (114.0)
        + (1689785.0 * e6 * plus_response * s2b) / (25992.0)
        + (1689785.0 * c2i * e6 * plus_response * s2b) / (25992.0)
        + (8286900895.0 * e8 * plus_response * s2b) / (7.1114112e7)
        + (8286900895.0 * c2i * e8 * plus_response * s2b) / (7.1114112e7)
    )


def _q15(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p2 = pow(f0, 19.0 / 3.0)
    return p2 * (
        (643523447.0 * c2b * ci * e6 * cross_response) / (519840.0)
        + (2138428414381.0 * c2b * ci * e8 * cross_response) / (1.5803136e8)
        + (643523447.0 * e6 * plus_response * s2b) / (1.03968e6)
        + (643523447.0 * c2i * e6 * plus_response * s2b) / (1.03968e6)
        + (2138428414381.0 * e8 * plus_response * s2b) / (3.1606272e8)
        + (2138428414381.0 * c2i * e8 * plus_response * s2b) / (3.1606272e8)
    )


def _q16(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p3 = pow(f0, 38.0 / 9.0)
    return p3 * (
        -(2862.0 * c2b * ci * e4 * cross_response) / (19.0)
        - (1585071.0 * c2b * ci * e6 * cross_response) / (1444.0)
        - (3905136831.0 * c2b * ci * e8 * cross_response) / (877952.0)
        - (1431.0 * e4 * plus_response * s2b) / (19.0)
        - (1431.0 * c2i * e4 * plus_response * s2b) / (19.0)
        - (1585071.0 * e6 * plus_response * s2b) / (2888.0)
        - (1585071.0 * c2i * e6 * plus_response * s2b) / (2888.0)
        - (3905136831.0 * e8 * plus_response * s2b) / (1.755904e6)
        - (3905136831.0 * c2i * e8 * plus_response * s2b) / (1.755904e6)
    )


def _q17(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p4 = pow(f0, 76.0 / 9.0)
    return p4 * (
        (-49470967683233.0 * c2b * ci * e8 * cross_response) / (4.97798784e9)
        - (49470967683233.0 * e8 * plus_response * s2b) / (9.95597568e9)
        - (49470967683233.0 * c2i * e8 * plus_response * s2b) / (9.95597568e9)
    )


def _q18(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p5 = pow(f0, 133.0 / 18.0)
    return p5 * (
        (3054326535625.0 * c2b * ci * e7 * cross_response) / (1.117863936e9)
        + (3054326535625.0 * e7 * plus_response * s2b) / (2.235727872e9)
        + (3054326535625.0 * c2i * e7 * plus_response * s2b) / (2.235727872e9)
    )


def _q19(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p6 = pow(f0, 95.0 / 18.0)
    return p6 * (
        -(13023125.0 * c2b * ci * e5 * cross_response) / (43776.0)
        - (216379221875.0 * c2b * ci * e7 * cross_response) / (7.9847424e7)
        - (13023125.0 * c2i * e5 * plus_response * s2b) / (87552.0)
        - (216379221875.0 * e7 * plus_response * s2b) / (1.59694848e8)
        - (216379221875.0 * c2i * e7 * plus_response * s2b) / (1.59694848e8)
        - (13023125.0 * e5 * plus_response * s2b) / (87552.0)
    )


def _q20(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p7 = pow(f0, 19.0 / 6.0)
    return p7 * (
        (625.0 * c2b * ci * e3 * cross_response) / (24.0)
        + (2076875.0 * c2b * ci * e5 * cross_response) / (14592.0)
        + (7933101875.0 * c2b * ci * e7 * cross_response) / (1.7743872e7)
        + (625.0 * e3 * plus_response * s2b) / (48.0)
        + (625.0 * c2i * e3 * plus_response * s2b) / (48.0)
        + (2076875.0 * e5 * plus_response * s2b) / (29184.0)
        + (2076875.0 * c2i * e5 * plus_response * s2b) / (29184.0)
        + (7933101875.0 * e7 * plus_response * s2b) / (3.5487744e7)
        + (7933101875.0 * c2i * e7 * plus_response * s2b) / (3.5487744e7)
    )


def _q21(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p2 = pow(f0, 19.0 / 3.0)
    return p2 * (
        -(1656963.0 * c2b * ci * e6 * cross_response) / (3040.0)
        - (5506088049.0 * c2b * ci * e8 * cross_response) / (924160.0)
        - (1656963.0 * e6 * plus_response * s2b) / (6080.0)
        - (1656963.0 * c2i * e6 * plus_response * s2b) / (6080.0)
        - (5506088049.0 * e8 * plus_response * s2b) / (1.84832e6)
        - (5506088049.0 * c2i * e8 * plus_response * s2b) / (1.84832e6)
    )


def _q22(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p3 = pow(f0, 38.0 / 9.0)
    return p3 * (
        (81.0 * c2b * ci * e4 * cross_response) / (2.0)
        + (89721.0 * c2b * ci * e6 * cross_response) / (304.0)
        + (221045481.0 * c2b * ci * e8 * cross_response) / (184832.0)
        + (81.0 * e4 * plus_response * s2b) / (4.0)
        + (81.0 * c2i * e4 * plus_response * s2b) / (4.0)
        + (89721.0 * e6 * plus_response * s2b) / (608.0)
        + (89721.0 * c2i * e6 * plus_response * s2b) / (608.0)
        + (221045481.0 * e8 * plus_response * s2b) / (369664.0)
        + (221045481.0 * c2i * e8 * plus_response * s2b) / (369664.0)
    )


def _q23(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p4 = pow(f0, 76.0 / 9.0)
    return p4 * (
        (71738041617.0 * c2b * ci * e8 * cross_response) / (1.293824e7)
        + (71738041617.0 * e8 * plus_response * s2b) / (2.587648e7)
        + (71738041617.0 * c2i * e8 * plus_response * s2b) / (2.587648e7)
    )


def _q24(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p5 = pow(f0, 133.0 / 18.0)
    return p5 * (
        (-1109077123.0 * c2b * ci * e7 * cross_response) / (1.16736e6)
        - (1109077123.0 * e7 * plus_response * s2b) / (2.33472e6)
        - (1109077123.0 * c2i * e7 * plus_response * s2b) / (2.33472e6)
    )


def _q25(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p6 = pow(f0, 95.0 / 18.0)
    return p6 * (
        (117649.0 * c2b * ci * e5 * cross_response) / (1920.0)
        + (390947627.0 * c2b * ci * e7 * cross_response) / (700416.0)
        + (117649.0 * e5 * plus_response * s2b) / (3840.0)
        + (117649.0 * c2i * e5 * plus_response * s2b) / (3840.0)
        + (390947627.0 * e7 * plus_response * s2b) / (1.400832e6)
        + (390947627.0 * c2i * e7 * plus_response * s2b) / (1.400832e6)
    )


def _q26(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p2 = pow(f0, 19.0 / 3.0)
    return p2 * (
        (4096.0 * c2b * ci * e6 * cross_response) / (45.0)
        + (850688.0 * c2b * ci * e8 * cross_response) / (855.0)
        + (2048.0 * e6 * plus_response * s2b) / (45.0)
        + (2048.0 * c2i * e6 * plus_response * s2b) / (45.0)
        + (425344.0 * e8 * plus_response * s2b) / (855.0)
        + (425344.0 * c2i * e8 * plus_response * s2b) / (855.0)
    )


def _q27(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p4 = pow(f0, 76.0 / 9.0)
    return p4 * (
        (-28696576.0 * c2b * ci * e8 * cross_response) / (17955.0)
        - (14348288.0 * e8 * plus_response * s2b) / (17955.0)
        - (14348288.0 * c2i * e8 * plus_response * s2b) / (17955.0)
    )


def _q28(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    c2i = ci * ci
    plus_response = f_plus
    cross_response = f_cross
    p5 = pow(f0, 133.0 / 18.0)
    return p5 * (
        (4782969.0 * c2b * ci * e7 * cross_response) / (35840.0)
        + (4782969.0 * e7 * plus_response * s2b) / (71680.0)
        + (4782969.0 * c2i * e7 * plus_response * s2b) / (71680.0)
    )


def _q29(e0, f0, inc, bet, f_plus, f_cross):
    e2 = e0 * e0
    e3 = e2 * e0
    e4 = e3 * e0
    e5 = e4 * e0
    e6 = e5 * e0
    e7 = e6 * e0
    e8 = e7 * e0
    c2b = math.cos(2.0 * bet)
    s2b = math.sin(2.0 * bet)
    ci = math.cos(inc)
    plus_response = f_plus
    cross_response = f_cross
    p4 = pow(f0, 76.0 / 9.0)
    return (
        p4
        * (
            390625.0
            * e8
            * (
                2.0 * c2b * ci * cross_response
                + plus_response * s2b
                + ci * ci * plus_response * s2b
            )
        )
        / (4032.0)
    )


@dataclass(frozen=True)
class EccentricFDCoefficients:
    """Frequency-independent phase and polarization coefficients."""

    phase: tuple[float, ...]
    zeta_real_plus: tuple[float, ...]
    zeta_real_cross: tuple[float, ...]
    zeta_imag_plus: tuple[float, ...]
    zeta_imag_cross: tuple[float, ...]


def eccentricfd_coefficients(
    total_mass,
    eta,
    eccentricity,
    f_lower,
    inclination,
    long_asc_nodes,
):
    """Construct the scalar coefficients used by ``EccentricFD``."""

    phase = (
        _c0(eta),
        _c1(total_mass),
        _c2(eccentricity, f_lower),
        _c3(eccentricity, f_lower),
        _c4(eccentricity, f_lower),
        _c5(eccentricity, f_lower),
        _c6(eta),
        _c7(eta, eccentricity, f_lower),
        _c8(eta, eccentricity, f_lower),
        _c9(eta, eccentricity, f_lower),
        _c10(eta, eccentricity, f_lower),
        _c11(eta),
        _c12(eta, eccentricity, f_lower),
        _c13(eta, eccentricity, f_lower),
        _c14(eta, eccentricity, f_lower),
        _c15(eta, eccentricity, f_lower),
        _c16(eta),
        _c17(eta, eccentricity, f_lower),
        _c18(eta, eccentricity, f_lower),
        _c19(eta, eccentricity, f_lower),
        _c20(eta, eccentricity, f_lower),
        _c21(eta),
        _c22(eta, eccentricity, f_lower),
        _c23(eta, eccentricity, f_lower),
        _c24(eta, eccentricity, f_lower),
        _c25(eta, eccentricity, f_lower),
        _c26(eta),
        _c27(eta),
        _c28(eta, eccentricity, f_lower),
        _c29(eta, eccentricity, f_lower),
        _c30(eta, eccentricity, f_lower),
        _c31(eta, eccentricity, f_lower),
        _c32(eta, eccentricity, f_lower),
        _c33(eta, eccentricity, f_lower),
        _c34(eta, eccentricity, f_lower),
        _c35(eta, eccentricity, f_lower),
        _c36(eta),
        _c37(eta, eccentricity, f_lower),
        _c38(eta, eccentricity, f_lower),
        _c39(eta, eccentricity, f_lower),
        _c40(eta, eccentricity, f_lower),
    )
    zeta_real_plus = (
        _z1(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z2(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z3(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z4(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z5(f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z6(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z7(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z8(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z9(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z10(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z11(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z12(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z13(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z14(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z15(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z16(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z17(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z18(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z19(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z20(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z21(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z22(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z23(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z24(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z25(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z26(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z27(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z28(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _z29(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
    )
    zeta_real_cross = (
        _z1(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z2(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z3(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z4(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z5(f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z6(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z7(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z8(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z9(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z10(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z11(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z12(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z13(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z14(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z15(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z16(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z17(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z18(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z19(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z20(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z21(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z22(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z23(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z24(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z25(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z26(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z27(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z28(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _z29(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
    )
    zeta_imag_plus = (
        _q1(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q2(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q3(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q4(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q5(f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q6(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q7(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q8(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q9(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q10(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q11(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q12(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q13(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q14(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q15(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q16(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q17(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q18(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q19(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q20(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q21(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q22(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q23(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q24(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q25(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q26(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q27(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q28(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
        _q29(eccentricity, f_lower, inclination, long_asc_nodes, 1.0, 0.0),
    )
    zeta_imag_cross = (
        _q1(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q2(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q3(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q4(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q5(f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q6(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q7(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q8(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q9(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q10(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q11(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q12(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q13(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q14(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q15(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q16(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q17(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q18(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q19(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q20(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q21(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q22(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q23(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q24(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q25(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q26(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q27(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q28(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
        _q29(eccentricity, f_lower, inclination, long_asc_nodes, 0.0, 1.0),
    )
    return EccentricFDCoefficients(
        phase=phase,
        zeta_real_plus=zeta_real_plus,
        zeta_real_cross=zeta_real_cross,
        zeta_imag_plus=zeta_imag_plus,
        zeta_imag_cross=zeta_imag_cross,
    )
