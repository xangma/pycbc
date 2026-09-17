#!/usr/bin/env python3
# Copyright (C) 2026
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

"""Benchmark pipeline using native diffgw provider."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.bench_torchwave_pipeline import main as run_pipeline


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not any(arg.startswith('--native-provider') for arg in argv):
        argv = ['--native-provider', 'diffgw'] + list(argv)
    return run_pipeline(argv)


if __name__ == '__main__':
    sys.exit(main())
