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

rem Image/audio/Eigen headers are under %HOUMO_EXAMPLES_PATH%\apis\common\

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
