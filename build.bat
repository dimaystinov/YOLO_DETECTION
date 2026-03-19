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

REM Python проекта: системный питон
set "PY=python"
echo Используется системный python

echo.
echo [1/3] Установка зависимостей...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Ошибка установки зависимостей.
    pause
    exit /b 1
)

echo.
echo [2/3] Установка PyInstaller...
"%PY%" -m pip install pyinstaller
if errorlevel 1 (
    echo Ошибка установки PyInstaller.
    pause
    exit /b 1
)

echo.
echo [3/3] Сборка exe...
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
