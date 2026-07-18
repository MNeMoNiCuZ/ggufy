@echo off
setlocal

set "EXE=zig-out\bin\ggufy-gui.exe"

if not exist "%EXE%" (
  echo Error: "%EXE%" not found.
  echo Build first with: compile.bat gui
  exit /b 1
)

start "" "%EXE%"
exit /b 0
