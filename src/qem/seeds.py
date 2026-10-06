"""Deterministic seed derivation. Every random number in the project comes from here."""

import hashlib


def derive_seed(*parts) -> int:
    """Deterministic 31-bit seed from any sequence of hashable parts."""
    s = "|".join(str(p) for p in parts).encode()
    return int(hashlib.sha256(s).hexdigest()[:8], 16) % (2**31 - 1)
