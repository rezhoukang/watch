@echo off
chcp 65001 >nul
title 本地视觉训练平台
cd /d "%~dp0"

echo ============================================
echo   本地视觉训练平台 - 图片上传
echo ============================================
echo.

REM 检查 venv 是否存在
if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境 .venv，请先运行:
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo 正在启动服务，浏览器将自动打开 http://127.0.0.1:8000
echo 关闭本窗口即停止服务
echo.
start "" http://127.0.0.1:8000
.venv\Scripts\python.exe app.py

pause
