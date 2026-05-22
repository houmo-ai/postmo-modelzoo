@echo off
REM Wrapper to invoke PowerShell build script with default installation
setlocal
if "%~1"=="" (
  powershell -ExecutionPolicy Bypass -File "%~dp0build_windows.ps1"
) else (
  REM Check if -NoInstall flag is passed, otherwise default to installing
  echo.%* | findstr /C:"-NoInstall">nul && (
    powershell -ExecutionPolicy Bypass -File "%~dp0build_windows.ps1" %*
  ) || (
    powershell -ExecutionPolicy Bypass -File "%~dp0build_windows.ps1" %*
  )
)
endlocal