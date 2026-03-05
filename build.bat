@echo off
chcp 65001 >nul
setlocal

REM Переход в папку скрипта (корень проекта)
cd /d "%~dp0"

if not exist "main.py" (
    echo Ошибка: main.py не найден. Запускайте из корня проекта.
    pause
    exit /b 1
)

if not exist "YOLO_Detection.spec" (
    echo Ошибка: YOLO_Detection.spec не найден.
    pause
    exit /b 1
)

REM Python проекта: .venv или venv, иначе системный
set "PY=python"
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
    echo Используется: .venv\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
    echo Используется: venv\Scripts\python.exe
) else (
    echo Используется системный: python
)
echo.

echo [1/2] Установка PyInstaller...
"%PY%" -m pip install pyinstaller
if errorlevel 1 (
    echo Ошибка установки PyInstaller.
    pause
    exit /b 1
)

echo.
echo [2/2] Сборка exe...
"%PY%" -m PyInstaller --noconfirm --clean YOLO_Detection.spec
if errorlevel 1 (
    echo Ошибка сборки.
    pause
    exit /b 1
)

echo.
echo Готово. Запуск: dist\YOLO_Detection\YOLO_Detection.exe
echo Рядом с exe положи yolo26n.pt и conf.json
pause
