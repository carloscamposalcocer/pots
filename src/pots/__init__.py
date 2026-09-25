"""Single-piece, support-free, 3D-printable breathing plant pot."""
from .field import make_field
from .geometry import Pot
from .patterns import PATTERNS
from .pipeline import generate

__version__ = "0.1.0"
__all__ = ["PATTERNS", "Pot", "generate", "make_field"]


def main(argv=None):
    from .cli import main as _main
    _main(argv)
