# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Overlap admission must preserve alias safety without quadratic batch work."""

import random

import pytest

torch = pytest.importorskip("torch")

from pycbc.filter import matchedfilter_torch  # noqa: E402


class SpanTensor:
    """Expose just the address metadata consumed by the admission check."""

    def __init__(self, address, itemsize=1):
        self.address = address
        self.itemsize = itemsize

    def data_ptr(self):
        return self.address

    def element_size(self):
        return self.itemsize


def quadratic_oracle(inputs, outputs, size):
    spans = [(z.data_ptr(), z.data_ptr() + size * z.element_size()) for z in outputs]

    def overlap(left, right):
        return left[0] < right[1] and right[0] < left[1]

    for index, span in enumerate(spans):
        if any(overlap(span, other) for other in inputs):
            return False
        if any(overlap(span, other) for other in spans[index + 1 :]):
            return False
    return True


@pytest.mark.parametrize(
    "inputs, addresses, size, expected",
    [
        ([], [], 8, True),
        ([(0, 20), (0, 20), (5, 15)], [20, 28], 8, True),
        ([(100, 108), (0, 8)], [24, 8, 16], 8, True),
        ([(0, 8)], [7], 8, False),
        ([(0, 100)], [20], 8, False),
        ([(20, 21)], [16], 8, False),
        ([], [0, 0], 8, False),
        ([], [16, 0, 7], 8, False),
        ([(0, 8)], [0, 8], 0, True),
        ([(0, 8)], [4], 0, False),
        ([(4, 4)], [0], 8, False),
        ([(0, 0), (8, 8)], [0], 8, True),
        ([(0, 0)], [0, 0], 0, True),
    ],
)
def test_overlap_boundaries(inputs, addresses, size, expected):
    outputs = [SpanTensor(address) for address in addresses]
    assert quadratic_oracle(inputs, outputs, size) is expected
    assert (
        matchedfilter_torch._batch_outputs_are_disjoint(inputs, outputs, size)
        is expected
    )


def test_randomized_overlap_matches_quadratic_oracle():
    rng = random.Random(91021)
    for _ in range(1500):
        starts = [rng.randrange(-64, 128) for _ in range(rng.randrange(33))]
        inputs = [(start, start + rng.randrange(33)) for start in starts]
        outputs = [
            SpanTensor(rng.randrange(-64, 128), rng.choice((1, 2, 4, 8)))
            for _ in range(rng.randrange(33))
        ]
        rng.shuffle(inputs)
        rng.shuffle(outputs)
        size = rng.randrange(17)
        assert matchedfilter_torch._batch_outputs_are_disjoint(
            inputs, outputs, size
        ) is quadratic_oracle(inputs, outputs, size)


def test_output_storage_mutation_is_rechecked():
    storage = torch.empty(128, dtype=torch.complex64)
    source = storage[:8]
    output = storage[16:24]
    inputs = [(source.data_ptr(), source.data_ptr() + 8 * source.element_size())]
    assert matchedfilter_torch._batch_outputs_are_disjoint(inputs, [output], 8)
    output.set_(storage[4:12])
    assert not matchedfilter_torch._batch_outputs_are_disjoint(inputs, [output], 8)
    output.set_(storage[8:16])
    assert matchedfilter_torch._batch_outputs_are_disjoint(inputs, [output], 8)


class CountingAddress(int):
    comparisons = 0

    def __add__(self, other):
        return type(self)(int(self) + other)

    def __lt__(self, other):
        type(self).comparisons += 1
        return int(self) < int(other)

    def __gt__(self, other):
        type(self).comparisons += 1
        return int(self) > int(other)


def test_large_disjoint_batch_has_subquadratic_address_comparisons():
    # Count comparisons instead of asserting a machine-dependent time limit.
    # The former scan makes over a million comparisons for this B=1024 case.
    count = 1024
    inputs = [
        (CountingAddress(4 * i), CountingAddress(4 * i + 1)) for i in range(count)
    ]
    outputs = [SpanTensor(CountingAddress(4 * i + 2)) for i in reversed(range(count))]
    CountingAddress.comparisons = 0
    assert matchedfilter_torch._batch_outputs_are_disjoint(inputs, outputs, 1)
    assert CountingAddress.comparisons < 64 * count
