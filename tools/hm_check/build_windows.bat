@echo off
REM Wrapper to invoke PowerShell build script
setlocal
if "%~1"=="" (
  powershell -ExecutionPolicy Bypass -File "%~dp0build_windows.ps1"
) else (
  powershell -ExecutionPolicy Bypass -File "%~dp0build_windows.ps1" %*
)
endlocal
