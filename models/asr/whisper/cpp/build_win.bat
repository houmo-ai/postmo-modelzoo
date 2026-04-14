@echo off
setlocal enabledelayedexpansion
:: Check and create 3rdparty directory if not exists
if not exist "3rdparty" (
    mkdir "3rdparty"
    echo Created directory: 3rdparty
)

if not exist "3rdparty\tokenizers-cpp" (
    echo Downloading precompiled model...
    cd ..
    "%PYTHON_DIR%\python.exe" get_model.py --type hmm
    cd cpp
)

if not exist "include\mel_filters.h" (
    cd cpp
    "%PYTHON_DIR%\python.exe" scripts/export_mel_filters.py --model_path ../whisper-medium/ --output include/mel_filters.h
    cd ..
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

echo All dependencies are ready

set PROJECT_DIR=%cd%
set HOUMO_EXAMPLES_PATH=%PROJECT_DIR%\..\..\..\..
set BUILD_TYPE=Release
if "%1" neq "" (set BUILD_TYPE=%1)
set BUILD_DIR=build_vs2022
set HDPL_PLATFORM=ASIC

rem c++ example
if exist "build" (rmdir /s /q build)
md build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%PROJECT_DIR%/../bin
cmake --build . --target=install --config=%BUILD_TYPE% || echo ERROR && cd .. && exit /b
endlocal
pause