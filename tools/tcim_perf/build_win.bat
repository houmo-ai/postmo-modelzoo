set PROJECT_DIR=%cd%
set HOUMO_EXAMPLES_PATH=%PROJECT_DIR%\..\..
set TCIM_RUNTIME_DIR=D:\temp\qwen3\houmo_tcim_runtime
set CMAKE_DIR=D:\Program Files\CMake\bin
set BUILD_TYPE=Release
if "%1" neq "" (set BUILD_TYPE=%1)
if defined CMAKE_PATH (set CMAKE_DIR=%CMAKE_PATH%)
if not defined TCIM_RUNTIME_PATH (set TCIM_RUNTIME_PATH=%TCIM_RUNTIME_DIR%)
set BUILD_DIR=build_vs2022
set HDPL_PLATFORM=ASIC

rem c++ example
if exist "build" (rmdir /s /q build)
md build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%PROJECT_DIR%/../bin
cmake --build . --target=install --config=%BUILD_TYPE% || echo ERROR && cd .. && exit /b
pause