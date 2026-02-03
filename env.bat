@echo off

if defined PYTHON_PATH (set PYTHON_DIR="%PYTHON_PATH%")
:: Check if number of arguments exceeds 1
set argC=0
for %%x in (%*) do set /a argC+=1
if %argC% gtr 1 (
    echo Error: Only 0 arguments or 1 '--set'/'--reset'/'--help'/'-h' argument allowed
    echo Usage:
    echo   %0 --set
    echo   %0 --reset
    exit /b 1
)

:: Check if argument is --reset (if provided)
if %argC% equ 1 (
    if "%1"=="--reset" (
        echo ===============Reset Win Envs===============
    ) else if "%1"=="--set" (
        echo ===============Set Win Envs===============
    ) else if "%1"=="--help" (
        echo 'env.bat --set' set environments
        echo 'env.bat --reset' reset environments
        exit /b 1
    ) else if "%1"=="-h" (
        echo 'env.bat --set' set environments
        echo 'env.bat --reset' reset environments
        exit /b 1
    ) else (
        echo Error: Invalid argument '%1'
        echo 'env.bat --set' set environments
        echo 'env.bat --reset' reset environments
        exit /b 1
    )
)

:: Execute corresponding python command
if %argC% equ 0 (
    echo 'env.bat --set' set environments
    echo 'env.bat --reset' reset environments
    exit /b 1
) else (
    if defined PYTHON_DIR (
        "%PYTHON_DIR%\python.exe" tools\win_envs\set_environs.py "%1"
    ) else (
        python tools\win_envs\set_environs.py "%1"
    )
    exit /b 1
)

pause