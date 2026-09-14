@echo off
cd /d "%~dp0"
docker build -f Dockerfile.ml-benchmark -t sentra-ml-benchmark:latest ..\..
