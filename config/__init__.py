"""config package helper

This wrapper makes both of the following import styles work:

    import config                      # attributes exposed at top level
    from config import config as cfg   # explicit sub-module reference

All public names defined in ``config/config.py`` are re-exported at the
package level so callers can simply write ``config.OUTPUT_DIR`` etc.
"""
from importlib import import_module as _import_module
import sys as _sys

# Import the real settings module situated in the same package.
_sub = _import_module('.config', package=__name__)

# Re-export every public attribute so ``import config`` works transparently.
globals().update({k: v for k, v in vars(_sub).items() if not k.startswith('_')})

# Also keep a reference for ``from config import config`` use-case.
config = _sub  # type: ignore

__all__ = list({*globals().keys()})
