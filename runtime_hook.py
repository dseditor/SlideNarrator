"""PyInstaller runtime hook: ensure sherpa_onnx DLLs are discoverable."""
import os
import sys


def _setup_dll_search():
    """Add DLL search directories for native extensions."""
    if not getattr(sys, 'frozen', False):
        return

    # _MEIPASS is the temp extraction directory for bundled files
    base = sys._MEIPASS

    # Add _internal/ root (where we placed sherpa_onnx DLLs) to DLL search
    if hasattr(os, 'add_dll_directory'):
        # Python 3.8+ on Windows
        os.add_dll_directory(base)

    # Also add sherpa_onnx/lib/ subdirectory
    sherpa_lib = os.path.join(base, 'sherpa_onnx', 'lib')
    if os.path.isdir(sherpa_lib):
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(sherpa_lib)

    # Prepend to PATH as fallback for older DLL search mechanisms
    os.environ['PATH'] = base + os.pathsep + os.environ.get('PATH', '')
    if os.path.isdir(sherpa_lib):
        os.environ['PATH'] = sherpa_lib + os.pathsep + os.environ['PATH']


_setup_dll_search()
