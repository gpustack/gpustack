import sys

if sys.version_info < (3, 10):
    import importlib_resources as pkg_resources
else:
    import importlib.resources as pkg_resources

__all__ = ['pkg_resources']
