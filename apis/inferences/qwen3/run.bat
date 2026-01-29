@echo off
setlocal enabledelayedexpansion
:: Check and create 3rdparty directory if not exists
if not exist "3rdparty" (
    mkdir "3rdparty"
    echo Created directory: 3rdparty
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

:: Download and setup tokenizers-cpp if not exists
if not exist "3rdparty\tokenizers-cpp" (
    pushd "3rdparty" || (echo Error: Failed to enter 3rdparty directory! & exit /b 1)

    set "TARGET_DIR=tokenizers-cpp"
    set "DOWNLOAD_URL=!HOUMO_MODELZOO_URL!/3rdparty/qwen3-tokenizers-cpp.zip"
    set "ZIP_FILE=tokenizers-cpp.zip"

    if "!DOWNLOAD_URL!"=="" (
        echo Error: DOWNLOAD_URL is empty! Check HOUMO_MODELZOO_URL variable.
        popd
        exit /b 1
    )

    echo Downloading tokenizers-cpp.zip from: !DOWNLOAD_URL!
    PowerShell -Command "(New-Object System.Net.WebClient).DownloadFile('!DOWNLOAD_URL!', '!ZIP_FILE!')"

    if not exist "!ZIP_FILE!" (
        echo Error: Failed to download "!ZIP_FILE!"!
        popd
        exit /b 1
    )

    echo Extracting !ZIP_FILE!...
    tar -xf "!ZIP_FILE!" || PowerShell -Command "Expand-Archive -Path '!ZIP_FILE!' -DestinationPath '.' -Force"

    popd

    if not exist "3rdparty\tokenizers-cpp" (
        echo Error: Failed to extract tokenizers-cpp!
        del /f /q "3rdparty\!ZIP_FILE!"
        exit /b 1
    )

    del /f /q "3rdparty\!ZIP_FILE!"
    echo Tokenizers-cpp setup completed
)

echo All dependencies are ready

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