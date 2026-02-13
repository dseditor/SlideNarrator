# -*- mode: python ; coding: utf-8 -*-
"""
Breeze2VITS PyInstaller Spec File

Build directory structure:
  dist/Breeze2VITS/
    Breeze2VITS.exe      (main executable)
    _internal/           (bundled Python runtime & packages)
    models/              (user copies externally)
    ffmpeg/              (user copies externally)
    dict/                (user copies externally)
    prompts/             (user copies externally)
    text_mapping.txt     (user copies externally)
"""

import os
import sys
from pathlib import Path

block_cipher = None

# -- Paths --
PROJECT_DIR = os.path.abspath('.')
SITE_PACKAGES = next(
    p for p in sys.path
    if 'site-packages' in p and os.path.isdir(p)
)

SHERPA_ONNX_DIR = os.path.join(SITE_PACKAGES, 'sherpa_onnx')
CUSTOMTKINTER_DIR = os.path.join(SITE_PACKAGES, 'customtkinter')

# -- Analysis --
a = Analysis(
    ['main.py'],
    pathex=[PROJECT_DIR],
    binaries=[
        # sherpa_onnx DLLs copied to TOP-LEVEL _internal/ for reliable DLL search
        (os.path.join(SHERPA_ONNX_DIR, 'lib', '*.dll'), '.'),
    ],
    datas=[
        # customtkinter full package (themes, assets)
        (CUSTOMTKINTER_DIR, 'customtkinter'),
        # sherpa_onnx full package (Python modules + native extensions)
        (SHERPA_ONNX_DIR, 'sherpa_onnx'),
    ],
    hiddenimports=[
        # sherpa_onnx
        'sherpa_onnx',
        'sherpa_onnx.lib',
        'sherpa_onnx.lib._sherpa_onnx',
        # numpy
        'numpy',
        'numpy.core',
        'numpy.core._methods',
        'numpy.lib',
        'numpy.lib.format',
        # customtkinter
        'customtkinter',
        # PIL / Pillow
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        # PyMuPDF
        'fitz',
        # comtypes (for PPTX conversion)
        'comtypes',
        'comtypes.client',
        # pypinyin
        'pypinyin',
        # standard library modules used
        'winsound',
        'wave',
        'io',
        'logging',
        'threading',
        'tkinter',
        'tkinter.filedialog',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=[
        # Exclude heavy packages not needed
        'torch',
        'torchaudio',
        'torchvision',
        'gradio',
        'gradio_client',
        'transformers',
        'tensorflow',
        'keras',
        'matplotlib',
        'scipy',
        'pandas',
        'sklearn',
        'scikit-learn',
        'opencv-python',
        'cv2',
        'huggingface_hub',
        'streamlit',
        'flask',
        'django',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Breeze2VITS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,           # GUI app, no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,               # Add icon path here if desired
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Breeze2VITS',
)
