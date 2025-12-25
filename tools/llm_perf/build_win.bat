setlocal enabledelayedexpansion
set "TARGET_DIR=3rdparty\eigen3"
set "DOWNLOAD_URL=!HOUMO_MODELZOO_URL!\models\qwen3\3rdparty.zip"
set "ZIP_FILE=3rdparty.zip"
if not exist "%TARGET_DIR%" (
    echo Directory not found: %TARGET_DIR%
    echo Starting to download 3rdparty.zip...
    PowerShell -Command "(New-Object System.Net.WebClient).DownloadFile('%DOWNLOAD_URL%', '%ZIP_FILE%')"
    if not exist "%ZIP_FILE%" (
        echo Error: Failed to download 3rdparty.zip!
        exit /b 1
    )
    echo Starting to extract 3rdparty.zip...
    tar -xf "%ZIP_FILE%"
    if not exist "%TARGET_DIR%" (
        echo Error: Failed to extract 3rdparty.zip!
        exit /b 1
    )
    echo Cleaning up temporary files...
    del /f /q "%ZIP_FILE%"
    echo 3rdparty dependencies installed successfully!
) else (
    echo Directory %TARGET_DIR% already exists, skipping download and extraction
)

set PROJECT_DIR=%cd%
set HOUMO_EXAMPLES_PATH=%PROJECT_DIR%\..\..
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
pause
endlocal