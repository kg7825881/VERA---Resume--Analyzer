"""Use VERA's local ``workflow.py`` module without a third-party hook.

The PyInstaller contrib hook named ``workflow`` is intended for an unrelated
PyPI package.  This project ships its own application module with that name,
so the default hook must not be applied when packaging the desktop backend.
"""
