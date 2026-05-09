@echo off
setlocal
set "ROOT=%~dp0"
start "" python -m http.server 8080 --directory "%ROOT%"
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8080/report_presentation_standalone.html"
