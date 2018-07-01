# -*- coding: utf-8 -*-

"""Top-level package for ml."""

__author__ = """Tor Erlend Fjelde"""
__email__ = 'tor.erlend95@gmail.com'
__version__ = '0.1.0'


import logging as _logging
_LOG_LEVEL = 'INFO'
_LOG_FORMAT = '%(asctime)-15s %(levelname)-9s %(module)s: %(message)s'
_logging.basicConfig(format=_LOG_FORMAT, level=getattr(_logging, _LOG_LEVEL))

_log = _logging.getLogger("ml")


# FIXME: hacky way of changing between CPU and GPU
import numpy as np


def initialize_gpu():
    _log.info("Initializing GPU...")
    import cupy
    global np
    np = cupy
