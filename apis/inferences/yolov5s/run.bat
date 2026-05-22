@echo off
set PROJECT_DIR=%~dp0
set PROJECT_DIR=%PROJECT_DIR:~0,-1%
cd /d "%PROJECT_DIR%"
set BUILD_TYPE=RelWithDebInfo
set BUILD_DIR=build_vs2022
set PATH=%OPENCV_PATH%\x64\vc16\bin;%PATH%
rem get test model
"%PYTHON_DIR%\python.exe" get_model.py

rem python example
"%PYTHON_DIR%\python.exe" yolov5s.py
rem c++ example
if exist "build" (rmdir /s /q build)
md build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="%OPENCV_PATH%" -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_INSTALL_PREFIX=%PROJECT_DIR% || echo ERROR && cd .. && exit /b
cmake --build . --target=install --config=%BUILD_TYPE% || echo ERROR && cd .. && exit /b

cd ..

.\example_yolov5s
pause