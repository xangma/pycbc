#!/usr/bin/env python3
# Copyright (C) 2026 PyCBC contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare search HDF outputs exactly, recording explicit exclusions."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def equal_values(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if (left.shape != right.shape or left.dtype != right.dtype or
            left.dtype.metadata != right.dtype.metadata):
        return False
    if left.dtype.names:
        return all(equal_values(left[name], right[name]) for name in left.dtype.names)
    if left.dtype.hasobject:
        return all(equal_values(a, b) if isinstance(a, np.ndarray) or isinstance(b, np.ndarray)
                   else a == b for a, b in zip(left.flat, right.flat))
    return left.tobytes() == right.tobytes()


def compare_files(reference, candidate, ignore_attributes=(), ignore_datasets=()):
    errors, excluded = [], []
    ignored_attrs, ignored_data = set(ignore_attributes), set(ignore_datasets)
    with h5py.File(reference, 'r') as left, h5py.File(candidate, 'r') as right:
        lpaths, rpaths = {}, {}
        left.visititems(lambda name, obj: lpaths.__setitem__(name, obj))
        right.visititems(lambda name, obj: rpaths.__setitem__(name, obj))
        lpaths['/'], rpaths['/'] = left, right
        for path in sorted(set(lpaths) | set(rpaths)):
            if path in ignored_data:
                excluded.append(dict(path=path, kind='dataset'))
                continue
            if path not in lpaths or path not in rpaths:
                errors.append(dict(path=path, reason='missing object'))
                continue
            a, b = lpaths[path], rpaths[path]
            if isinstance(a, h5py.Dataset) != isinstance(b, h5py.Dataset):
                errors.append(dict(path=path, reason='object type differs'))
                continue
            if isinstance(a, h5py.Dataset):
                if not equal_values(a[()], b[()]):
                    errors.append(dict(path=path, reason='dtype, shape or value bytes differ'))
                for field in ('chunks', 'compression', 'compression_opts', 'shuffle',
                              'fletcher32', 'scaleoffset', 'maxshape', 'fillvalue'):
                    if not equal_values(getattr(a, field), getattr(b, field)):
                        errors.append(dict(path=path, reason='storage ' + field + ' differs'))
            for key in sorted(set(a.attrs) | set(b.attrs)):
                if key in ignored_attrs:
                    excluded.append(dict(path=path + '@' + key, kind='attribute'))
                elif key not in a.attrs or key not in b.attrs or not equal_values(a.attrs[key], b.attrs[key]):
                    errors.append(dict(path=path + '@' + key, reason='attribute differs'))
    return dict(passed=not errors, errors=errors, excluded=excluded)


def compare_outputs(reference, candidate, **kwargs):
    reference, candidate = Path(reference), Path(candidate)
    if reference.is_file() and candidate.is_file():
        return dict(files={'output': compare_files(reference, candidate, **kwargs)})
    if not reference.is_dir() or not candidate.is_dir():
        raise ValueError('Both inputs must be HDF files or directories of HDF files')
    a = {str(p.relative_to(reference)): p for p in reference.rglob('*.hdf')}
    b = {str(p.relative_to(candidate)): p for p in candidate.rglob('*.hdf')}
    if not a or set(a) != set(b):
        raise ValueError('HDF file inventories must be nonempty and identical')
    return dict(files={name: compare_files(a[name], b[name], **kwargs) for name in sorted(a)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('candidate')
    parser.add_argument('--ignore-attribute', action='append', default=[])
    parser.add_argument('--ignore-dataset', action='append', default=[])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        result = compare_outputs(args.reference, args.candidate,
                                 ignore_attributes=args.ignore_attribute,
                                 ignore_datasets=args.ignore_dataset)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    result['passed'] = all(item['passed'] for item in result['files'].values())
    text = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
