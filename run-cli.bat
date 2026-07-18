@echo off
setlocal

set "EXE=zig-out\bin\ggufy.exe"

if not exist "%EXE%" (
  echo Error: "%EXE%" not found.
  echo Build first with: compile.bat cli
  exit /b 1
)

"%EXE%" %*
exit /b %ERRORLEVEL%
