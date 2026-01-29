@echo off
setlocal enabledelayedexpansion

set PROJECT_DIR=%cd%
set BUILD_TYPE=Release
if "%1" neq "" (set BUILD_TYPE=%1)
set BUILD_DIR=build_vs2022
set HDPL_PLATFORM=ASIC
rem get test model
%PYTHON_DIR%\python.exe get_model.py

rem python example
%PYTHON_DIR%\python.exe demo.py

rem c++ example
if exist "build" (rmdir /s /q build)
md build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%PROJECT_DIR%
cmake --build . --target=install --config=%BUILD_TYPE% || echo ERROR && cd .. && exit /b

cd ..
example_cxx_qwen3.exe
endlocal
pause