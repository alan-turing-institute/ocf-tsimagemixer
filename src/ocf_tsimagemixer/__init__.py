from importlib.metadata import version

from .imagemixer import ImageMixer

__all__ = ("__version__",)
__version__ = version(__name__)
