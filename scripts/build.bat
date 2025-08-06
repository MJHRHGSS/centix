@echo off
cd /d "%~dp0"
cd ..
if not exist out (
    mkdir out
)
make
out\cx %*
