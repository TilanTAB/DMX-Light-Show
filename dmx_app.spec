# -*- mode: python ; coding: utf-8 -*-
"""
DMX Light Show — PyInstaller Multi-Entry Spec File

Produces a ONE-FOLDER distribution with four executables:
  dist/dmx_app/
    ├── dmx_app.exe              ← Main entry (FastAPI + embedded React UI)
    ├── music_light_worker.exe   ← Loopback DMX engine worker (spawned by dmx_app)
    ├── ai_show_player_worker.exe ← AI show synced playback worker (spawned by dmx_app)
    ├── youtube_analyzer_worker.exe ← YouTube + AI analysis worker
    ├── frontend/dist/           ← Pre-built React static files
    ├── profiles/                ← Lighting profiles
    ├── .env.example             ← Credentials template
    └── ... (Python runtime, DLLs, etc.)

The user must place alongside the exe:
    ├── ffmpeg.exe               ← ~200MB, too large to bundle
    ├── yt-dlp.exe               ← ~18MB, frequently updated
    ├── .env                     ← Azure credentials (user-created)
    └── shows/                   ← Created automatically at runtime

Usage:
    pip install pyinstaller
    pyinstaller dmx_app.spec
"""

import os
import sys
import glob

block_cipher = None

# ---------------------------------------------------------------------------
# Locate libusb DLL — required by pyusb for uDMX hardware access.
# PyInstaller does NOT auto-detect this because pyusb loads it via ctypes
# at runtime, not via a Python import.
# ---------------------------------------------------------------------------
libusb_dll = None
# Common locations on Windows
_search_paths = [
    # If installed via pip (libusb package)
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'libusb', '_platform', '_windows', 'x64', 'libusb-1.0.dll'),
    # If in the venv
    os.path.join('.venv', 'Lib', 'site-packages', 'libusb', '_platform', '_windows', 'x64', 'libusb-1.0.dll'),
    # System-wide
    os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'System32', 'libusb-1.0.dll'),
    # Logi RightSight (common on machines with Logitech webcam software)
    os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), 'Logi', 'RightSightForWebcams', 'libusb-1.0.dll'),
    # Local copy
    'libusb-1.0.dll',
]
for p in _search_paths:
    if os.path.isfile(p):
        libusb_dll = p
        print(f"[SPEC] Found libusb at: {p}")
        break

if not libusb_dll:
    print("[SPEC] WARNING: libusb-1.0.dll not found! USB DMX will not work in the packaged app.")
    print("[SPEC] Searched:", _search_paths)

# ---------------------------------------------------------------------------
# Shared hidden imports — modules that PyInstaller's static analysis misses
# because they're loaded dynamically (e.g., uvicorn uses importlib internally)
# ---------------------------------------------------------------------------
_shared_hiddenimports = [
    # Uvicorn internals (it uses importlib to load these)
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    # Multipart form parsing (FastAPI dependency)
    'multipart',
    'python_multipart',
    # dotenv for .env loading
    'dotenv',
    # numpy — MKL/OpenBLAS backends
    'numpy',
    'numpy.core._methods',
    'numpy.lib.format',
    # USB
    'usb',
    'usb.core',
    'usb.backend',
    'usb.backend.libusb1',
    # Audio
    'pyaudiowpatch',
    # Azure OpenAI / HTTP clients used by llm_designer
    'openai',
    'httpx',
    'httpx._transports',
    'httpx._transports.default',
    'anyio',
    'anyio._backends',
    'anyio._backends._asyncio',
    'sniffio',
    'certifi',
    'httpcore',
    'h11',
]

# ---------------------------------------------------------------------------
# Data files to bundle (read-only, extracted to _MEIPASS temp dir)
# ---------------------------------------------------------------------------
_shared_datas = [
    # React production build
    ('frontend/dist', 'frontend/dist'),
    # Lighting profiles (default set)
    ('profiles', 'profiles'),
    # Env template so the user knows what to create
    ('.env.example', '.'),
]

# Add libusb DLL as a binary if found
_shared_binaries = []
if libusb_dll:
    _shared_binaries.append((libusb_dll, '.'))

# ---------------------------------------------------------------------------
# ANALYSIS 1: Main app (FastAPI server + embedded UI)
# ---------------------------------------------------------------------------
a_main = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=_shared_binaries,
    datas=_shared_datas,
    hiddenimports=_shared_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'PIL', 'cv2'],
    noarchive=False,
    optimize=0,
)

# ---------------------------------------------------------------------------
# ANALYSIS 2: music_light worker (DMX engine)
# ---------------------------------------------------------------------------
a_ml = Analysis(
    ['music_light.py'],
    pathex=['.'],
    binaries=_shared_binaries,
    datas=[],
    hiddenimports=[
        'usb', 'usb.core', 'usb.backend', 'usb.backend.libusb1',
        'pyaudiowpatch', 'numpy', 'numpy.core._methods',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'PIL', 'cv2'],
    noarchive=False,
    optimize=0,
)

# ---------------------------------------------------------------------------
# ANALYSIS 3: ai_show_player worker (AI-generated show synced playback)
# ---------------------------------------------------------------------------
a_ai = Analysis(
    ['ai_show_player.py'],
    pathex=['.'],
    binaries=_shared_binaries,
    datas=[],
    hiddenimports=[
        'usb', 'usb.core', 'usb.backend', 'usb.backend.libusb1',
        'pyaudiowpatch', 'numpy', 'numpy.core._methods',
        'bisect',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'PIL', 'cv2'],
    noarchive=False,
    optimize=0,
)

# ---------------------------------------------------------------------------
# ANALYSIS 4: youtube_analyzer worker
# ---------------------------------------------------------------------------
a_yt = Analysis(
    ['youtube_analyzer.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'numpy', 'numpy.core._methods', 'numpy.lib.format',
        'llm_designer',
        'openai', 'httpx', 'httpx._transports', 'httpx._transports.default',
        'anyio', 'anyio._backends', 'anyio._backends._asyncio',
        'sniffio', 'certifi', 'httpcore', 'h11',
        'dotenv',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'PIL', 'cv2'],
    noarchive=False,
    optimize=0,
)

# ---------------------------------------------------------------------------
# MERGE: Share common modules to avoid tripling the dist size
# ---------------------------------------------------------------------------
MERGE(
    (a_main, 'dmx_app', 'dmx_app'),
    (a_ml, 'music_light_worker', 'music_light_worker'),
    (a_ai, 'ai_show_player_worker', 'ai_show_player_worker'),
    (a_yt, 'youtube_analyzer_worker', 'youtube_analyzer_worker'),
)

# ---------------------------------------------------------------------------
# PYZ archives (compressed Python bytecode)
# ---------------------------------------------------------------------------
pyz_main = PYZ(a_main.pure, a_main.zipped_data, cipher=block_cipher)
pyz_ml = PYZ(a_ml.pure, a_ml.zipped_data, cipher=block_cipher)
pyz_ai = PYZ(a_ai.pure, a_ai.zipped_data, cipher=block_cipher)
pyz_yt = PYZ(a_yt.pure, a_yt.zipped_data, cipher=block_cipher)

# ---------------------------------------------------------------------------
# EXE targets
# ---------------------------------------------------------------------------
exe_main = EXE(
    pyz_main,
    a_main.scripts,
    [],
    exclude_binaries=True,
    name='dmx_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,     # Console window for log visibility
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,         # TODO: Add .ico file for branding
)

exe_ml = EXE(
    pyz_ml,
    a_ml.scripts,
    [],
    exclude_binaries=True,
    name='music_light_worker',
    debug=False,
    strip=False,
    upx=True,
    console=True,  # Needs console for logging
)

exe_ai = EXE(
    pyz_ai,
    a_ai.scripts,
    [],
    exclude_binaries=True,
    name='ai_show_player_worker',
    debug=False,
    strip=False,
    upx=True,
    console=True,  # Needs console for logging
)

exe_yt = EXE(
    pyz_yt,
    a_yt.scripts,
    [],
    exclude_binaries=True,
    name='youtube_analyzer_worker',
    debug=False,
    strip=False,
    upx=True,
    console=True,  # Needs console for progress output
)

# ---------------------------------------------------------------------------
# COLLECT: Merge everything into a single dist/dmx_app/ folder
# ---------------------------------------------------------------------------
coll = COLLECT(
    exe_main,
    a_main.binaries,
    a_main.zipfiles,
    a_main.datas,
    exe_ml,
    a_ml.binaries,
    a_ml.zipfiles,
    a_ml.datas,
    exe_ai,
    a_ai.binaries,
    a_ai.zipfiles,
    a_ai.datas,
    exe_yt,
    a_yt.binaries,
    a_yt.zipfiles,
    a_yt.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='dmx_app',
)
