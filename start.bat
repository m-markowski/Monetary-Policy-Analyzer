@echo off
setlocal DisableDelayedExpansion
title Monetary Policy Analyzer
pushd "%~dp0"
if errorlevel 1 goto directory_error

set "powershell=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if defined PROCESSOR_ARCHITEW6432 set "powershell=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%powershell%" goto powershell_error

"%powershell%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "scripts\start.ps1" %*
set "exit_code=%errorlevel%"
popd
echo.
if "%exit_code%"=="0" (
    echo Finished. Run start.bat again whenever you want to open the application.
) else (
    echo The launcher stopped with exit code %exit_code%. Review the messages above.
)
echo Press any key to close this window.
pause >nul
exit /b %exit_code%

:powershell_error
echo ERROR: Windows PowerShell is unavailable. It is required to start the local installation.
echo On a managed computer, contact your administrator rather than changing security settings.
popd
echo Press any key to close this window.
pause >nul
exit /b 1

:directory_error
echo ERROR: Could not open the project directory. Use a writable folder on a local drive.
echo Press any key to close this window.
pause >nul
exit /b 1
