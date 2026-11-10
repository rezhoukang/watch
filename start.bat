@echo off
chcp 65001 >nul
title 坐姿视觉识别平台
cd /d "%~dp0"

echo ============================================
echo   坐姿视觉识别平台 - 图片上传
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

REM 释放 8000 端口：上次的服务窗口若是被直接关掉/崩溃，进程可能还占着端口
echo 检查并释放 8000 端口...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /c:":8000 " ^| findstr /c:"LISTENING"') do (
    taskkill /f /pid %%a >nul 2>&1
    if errorlevel 1 (
        echo   [警告] 无法结束 PID %%a，若启动失败请手动关闭该进程
    ) else (
        echo   - 已结束占用 8000 端口的进程 PID %%a
    )
)
timeout /t 1 /nobreak >nul 2>&1
echo.

echo 正在启动服务，浏览器将自动打开 http://127.0.0.1:8000
echo 关闭本窗口即停止服务
echo.
start "" http://127.0.0.1:8000
.venv\Scripts\python.exe app.py

pause
