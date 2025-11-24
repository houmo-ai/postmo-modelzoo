@echo off
set PROJECT_DIR=%~dp0
set PROJECT_DIR=%PROJECT_DIR:~0,-1% 
cd /d "%PROJECT_DIR%"
set BUILD_DIR=build_vs2022
set BUILD_TYPE=Release
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
pause