@echo off

:: Check if number of arguments exceeds 1
set argC=0
for %%x in (%*) do set /a argC+=1
if %argC% gtr 1 (
    echo Error: Only 0 arguments or 1 '--reset'/'--clear' argument allowed
    echo Usage:
    echo   %0
    echo   %0 --reset
    echo   %0 --clear
    exit /b 1
)

:: Check if argument is --reset (if provided)
if %argC% equ 1 (
    if "%1"=="--reset" (
        echo ===============Reset Win Envs===============
    ) else if "%1"=="--clear" (
        echo ===============Clear Win Envs===============
    ) else if "%1"=="--help" (
        echo 'env.bat ' set environments
        echo 'env.bat --clear' clear environments 
        echo 'env.bat --reset' reset environments
        exit /b 1
    ) else if "%1"=="-h" (
        echo 'env.bat ' set environments
        echo 'env.bat --clear' clear environments 
        echo 'env.bat --reset' reset environments
        exit /b 1
    ) else (
        echo Error: Invalid argument '%1'
        echo Only '--reset' is supported
        exit /b 1
    )
)

:: Execute corresponding python command
if %argC% equ 0 (
    python tools\win_envs\set_environs.py
) else (
    python tools\win_envs\set_environs.py "%1"
)