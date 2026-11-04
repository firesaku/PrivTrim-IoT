"""Stub module for the Perception Compressor checkout.

The upstream file imports ``seaborn`` but never calls ``sns.*``, so an empty
module is sufficient and avoids pulling in matplotlib on a headless node.

``src/run_pc.py`` puts this directory first on ``sys.path``.
"""
