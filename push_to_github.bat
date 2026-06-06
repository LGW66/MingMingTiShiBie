@echo off
echo ========================================
echo Git Push Script for MingMingTiShiBie
echo ========================================
echo.

cd /d "%~dp0"

echo Current directory: %cd%
echo.

echo Checking git status...
git status

echo.
echo ========================================
echo The following files will be pushed:
echo ========================================
git log --oneline -5

echo.
echo ========================================
echo To push to GitHub, run:
echo git push -u origin main --force
echo.
echo Note: You may need to enter your GitHub credentials.
echo ========================================
echo.

pause
