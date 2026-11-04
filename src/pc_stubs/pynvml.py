"""Stub module for the Perception Compressor checkout.

The upstream file imports ``pynvml`` but never calls any NVML function, so these
no-op definitions are enough to satisfy the import on a machine without a GPU.

``src/run_pc.py`` puts this directory first on ``sys.path``.
"""


def nvmlInit(*a, **k):
    pass


def nvmlDeviceGetHandleByIndex(*a, **k):
    return None


def nvmlDeviceGetMemoryInfo(*a, **k):
    return None
