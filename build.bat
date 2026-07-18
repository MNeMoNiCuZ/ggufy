@echo off
setlocal

set "TARGET=%~1"
set "OPTIMIZE=%~2"

if "%TARGET%"=="" set "TARGET=all"
if "%OPTIMIZE%"=="" set "OPTIMIZE=ReleaseFast"

if /I not "%TARGET%"=="all" if /I not "%TARGET%"=="cli" if /I not "%TARGET%"=="gui" (
  echo Invalid target: %TARGET%
  echo Usage: build.bat [all^|cli^|gui] [Debug^|ReleaseSafe^|ReleaseFast^|ReleaseSmall]
  goto :end
)

if /I not "%OPTIMIZE%"=="Debug" if /I not "%OPTIMIZE%"=="ReleaseSafe" if /I not "%OPTIMIZE%"=="ReleaseFast" if /I not "%OPTIMIZE%"=="ReleaseSmall" (
  echo Invalid optimize mode: %OPTIMIZE%
  echo Usage: build.bat [all^|cli^|gui] [Debug^|ReleaseSafe^|ReleaseFast^|ReleaseSmall]
  goto :end
)

where git >nul 2>nul
if errorlevel 1 (
  echo Error: git is not in PATH.
  goto :end
)

where zig >nul 2>nul
if errorlevel 1 (
  echo Error: zig is not in PATH.
  goto :end
)

for /f "delims=" %%V in ('zig version') do set "ZIG_VERSION=%%V"
echo Detected Zig version: %ZIG_VERSION%
echo %ZIG_VERSION% | findstr /b "0.16." >nul
if errorlevel 1 (
  echo Error: this project currently expects Zig 0.16.x ^(build.zig.zon minimum is 0.16.0^).
  echo Installed: %ZIG_VERSION%
  echo Please install Zig 0.16.0 and run build.bat again.
  goto :end
)

echo Initializing submodules...
git submodule update --init --recursive
if errorlevel 1 (
  echo Error: failed to initialize submodules.
  goto :end
)

if /I "%TARGET%"=="all" (
  echo Running: zig build -Doptimize=%OPTIMIZE%
  zig build -Doptimize=%OPTIMIZE%
) else (
  echo Running: zig build %TARGET% -Doptimize=%OPTIMIZE%
  zig build %TARGET% -Doptimize=%OPTIMIZE%
)

if errorlevel 1 (
  echo Build failed.
  goto :end
)

echo.
echo Build complete.
echo Binaries are in .\zig-out\bin\

:end
pause
