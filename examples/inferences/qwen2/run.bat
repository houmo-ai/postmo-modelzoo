set PROJECT_DIR=%cd%
set HOUMO_EXAMPLES_PATH=%PROJECT_DIR%\..\..
set TCIM_RUNTIME_DIR="D:\tcim_runtime_lite\output"
set PYTHON_DIR="C:\ProgramData\miniconda3"
set CMAKE_DIR="C:\Program Files\CMake\bin"
set OPENCV_DIR=D:\Tools\opencv\build
set BUILD_TYPE=RelWithDebInfo
if "%1" neq "" (set BUILD_TYPE=%1)
if defined PYTHON_PATH (set PYTHON_DIR="%PYTHON_PATH%")
if defined CMAKE_PATH (set CMAKE_DIR="%CMAKE_PATH%")
if defined OPENCV_PATH (set OPENCV_DIR="%OPENCV_PATH%")
if not defined TCIM_RUNTIME_PATH (set TCIM_RUNTIME_PATH="%TCIM_RUNTIME_DIR%")
set BUILD_DIR=build_vs2022
set PYTHONPATH=%TCIM_RUNTIME_PATH%\python;%PYTHONPATH%
set PATH=%CMAKE_DIR%;%TCIM_RUNTIME_PATH%\python\tcim_lite;%OPENCV_DIR%\x64\vc16\bin;%PATH%
set HDPL_PLATFORM=ASIC

rem get test model
%PYTHON_DIR%\python.exe get_model.py

rem python example
%PYTHON_DIR%\python.exe demo.py
