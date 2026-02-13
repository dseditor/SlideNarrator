@echo off
REM ============================================================
REM  SlideNarrator Build Script
REM  Builds the project into a standalone executable package
REM ============================================================

setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
set "DIST_DIR=%PROJECT_DIR%dist\SlideNarrator"
set "SPEC_FILE=%PROJECT_DIR%SlideNarrator.spec"

echo ============================================================
echo  SlideNarrator Build Script
echo ============================================================
echo.

REM -- Step 1: Clean previous build --
echo [1/4] Cleaning previous build artifacts...
if exist "%PROJECT_DIR%build" (
    rmdir /s /q "%PROJECT_DIR%build"
)
if exist "%DIST_DIR%" (
    rmdir /s /q "%DIST_DIR%"
)
echo       Done.
echo.

REM -- Step 2: Run PyInstaller --
echo [2/4] Running PyInstaller...
echo       This may take several minutes...
echo.
python -m PyInstaller "%SPEC_FILE%" --noconfirm --clean
if %errorlevel% neq 0 (
    echo.
    echo *** ERROR: PyInstaller build failed! ***
    pause
    exit /b 1
)
echo.
echo       PyInstaller completed successfully.
echo.

REM -- Step 3: Copy external resources using Python (more reliable) --
echo [3/4] Copying external resources to output directory...
python -c "import shutil,os;s=r'%PROJECT_DIR%';d=r'%DIST_DIR%';[(shutil.copytree(os.path.join(s,x),os.path.join(d,x)) if not os.path.exists(os.path.join(d,x)) else None) or print(f'  Copied {x}/') for x in ['models','ffmpeg','dict','prompts'] if os.path.isdir(os.path.join(s,x))];f='text_mapping.txt';shutil.copy2(os.path.join(s,f),os.path.join(d,f)) if os.path.exists(os.path.join(s,f)) else None;print(f'  Copied {f}')"
if %errorlevel% neq 0 (
    echo       WARNING: Resource copy had errors, check manually.
)
echo       Done.
echo.

REM -- Step 4: Summary --
echo [4/4] Build Summary
echo ============================================================
echo.
echo   Output directory: %DIST_DIR%
echo.
echo   Directory structure:
echo     SlideNarrator/
echo       SlideNarrator.exe      (main executable)
echo       _internal/           (Python runtime)
echo       models/              (TTS model files)
echo       ffmpeg/              (ffmpeg.exe)
echo       dict/                (dictionary files)
echo       prompts/             (AI prompt templates)
echo       text_mapping.txt     (text conversion rules)
echo.
echo ============================================================
echo  BUILD COMPLETE
echo ============================================================
echo.
echo  You can copy the entire "dist\SlideNarrator" folder
echo  to another machine to run the application.
echo.
pause
