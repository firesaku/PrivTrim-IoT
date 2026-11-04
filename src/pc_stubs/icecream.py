"""Stub module for the Perception Compressor checkout.

The upstream file does ``from icecream import ic as print`` but only ever uses
``ic`` as a debugging print, so downgrading it to ``print`` is
behaviour-preserving and avoids installing a dependency we do not need.

``src/run_pc.py`` puts this directory first on ``sys.path``.
"""


def ic(*a, **k):
    print(*a)
