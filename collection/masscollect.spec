# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\beaul\\Documents\\GitHub\\ETVR-Data-Collection\\collection\\masscollect.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('overlay\\EyeTrackVR-Overlay.exe', 'overlay'),
        ('overlay\\openvr_api.dll', 'overlay'),
        ('overlay\\assets\\Purple_Dot.png', 'overlay/assets'),
    ],
    hiddenimports=['cv2', 'numpy', 'colorama', 'serial.tools.list_ports'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='masscollect',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
