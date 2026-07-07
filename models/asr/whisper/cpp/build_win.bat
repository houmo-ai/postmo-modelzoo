@echo off
chcp 65001 > nul
set VSLANG=1033
setlocal enabledelayedexpansion

set PROJECT_DIR=%cd%
if "%HOUMO_EXAMPLES_PATH%"=="" set HOUMO_EXAMPLES_PATH=%PROJECT_DIR%\..\..\..\..
set BUILD_TYPE=Release
set BUILD_JOBS=%NUMBER_OF_PROCESSORS%

rem c++ example
if exist "build" (
    rmdir /s /q build
    echo Removed existing build directory
)
md build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_INSTALL_PREFIX=%PROJECT_DIR%/../bin
cmake --build . --target=install --config=%BUILD_TYPE% --parallel %BUILD_JOBS% || echo ERROR && cd .. && exit /b
endlocal
