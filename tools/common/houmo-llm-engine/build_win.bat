@echo off
chcp 65001 > /dev/null
set VSLANG=1033
setlocal enabledelayedexpansion
set PROJECT_DIR=%cd%

rem Usage:
rem   build_win.bat

:: remove build directory if exists
if exist "build" (
    rmdir /s /q build
    echo Removed existing build directory
)
mkdir "build"

:: Check and create 3rdparty directory if not exists
if not exist "3rdparty" (
    mkdir "3rdparty"
    echo Created directory: 3rdparty
)

if not exist "3rdparty\googletest" (
    "%PYTHON_DIR%\python.exe" get_3rdparty.py
)

:: Download and setup eigen3 if not exists
if not exist "3rdparty\eigen3" (
    cd /d "3rdparty"
    echo Downloading eigen-3.4.0.zip...
    powershell -Command "(New-Object System.Net.WebClient).DownloadFile('https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip', 'eigen-3.4.0.zip')"

    echo Extracting eigen-3.4.0.zip...
    powershell -Command "Expand-Archive -Path 'eigen-3.4.0.zip' -DestinationPath '.' -Force"

    echo Renaming directory...
    ren "eigen-3.4.0" "eigen3"

    echo Cleaning up...
    del /f /q "eigen-3.4.0.zip"

    cd /d ..
    echo Eigen3 setup completed
)

rem Image/audio: header-only under %HOUMO_EXAMPLES_PATH%\apis\common\hpp\

echo All dependencies are ready

set HOUMO_EXAMPLES_PATH=%HOUMO_EXAMPLES_PATH%
set BUILD_TYPE=Release
set BUILD_JOBS=%NUMBER_OF_PROCESSORS%
set BUILD_DIR=build_vs2022
set HDPL_PLATFORM=ASIC

rem c++ example
if exist "build" (rmdir /s /q build)
md build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_INSTALL_PREFIX=%PROJECT_DIR%/bin
cmake --build . --target=install --config=%BUILD_TYPE% --parallel %BUILD_JOBS% || echo ERROR && cd .. && exit /b
endlocal
pause
