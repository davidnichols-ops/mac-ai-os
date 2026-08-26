"""Safe mutation/property-test planning primitives.

This module plans bounded tests. It does not execute arbitrary repository code.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class MutationCase:
    seed: bytes
    mutation: str
    expected: str


def generate_mutation_cases(seed: bytes, count: int = 32, *, random_seed: int = 0) -> list[MutationCase]:
    if count < 0 or count > 10_000:
        raise ValueError("count must be between 0 and 10000")
    rng = random.Random(random_seed)
    cases: list[MutationCase] = []
    for _ in range(count):
        data = bytearray(seed)
        mutation = rng.choice(("truncate", "flip_bit", "append_nul", "duplicate"))
        if mutation == "truncate":
            data = data[: rng.randrange(len(data) + 1)] if data else data
        elif mutation == "flip_bit" and data:
            index = rng.randrange(len(data))
            data[index] ^= 1 << rng.randrange(8)
        elif mutation == "append_nul":
            data.extend(b"\x00")
        elif mutation == "duplicate":
            data.extend(data)
        cases.append(MutationCase(bytes(data), mutation, "reject safely or preserve documented invariant"))
    return cases
