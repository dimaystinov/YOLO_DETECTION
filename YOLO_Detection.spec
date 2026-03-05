# -*- mode: python ; coding: utf-8 -*-
# Сборка: pyinstaller YOLO_Detection.spec

import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_all

block_cipher = None

# Подключаем все модули и данные ultralytics (YOLO)
ul_datas = collect_data_files('ultralytics')
ul_hidden = collect_submodules('ultralytics')

# OpenCV (cv2) — бинарники и данные, иначе exe не найдёт модуль
cv2_datas, cv2_binaries, cv2_hidden = collect_all('cv2')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=cv2_binaries,
    datas=ul_datas + cv2_datas,
    hiddenimports=[
        'ultralytics',
        'cv2',
        'PIL',
        'PIL._tkinter_finder',
        'torch',
        'numpy',
    ] + ul_hidden + cv2_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# onedir: exe + папка с библиотеками (cv2 и др. надёжно подхватываются)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='YOLO_Detection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='YOLO_Detection',
)
