"""PySCF-specific electronic structure interface implementations."""

from .casscf import CASSCFBackend
try:
    from .cisd import CISDBackend
except ImportError:
    CISDBackend = None

__all__ = ["CASSCFBackend"]
if CISDBackend is not None:
    __all__.append("CISDBackend")
