@echo off
chcp 65001 >nul
title RAG 评估系统

echo ============================================================
echo   🚀 招股说明书 RAG 系统评估
echo ============================================================
echo.
echo [1] 运行 V2 增强版评估（推荐，零依赖）
echo [2] 安装 RAGAS 库（隔离安装）
echo [3] 运行 RAGAS 官方库评估
echo [4] 退出
echo.
set /p choice=请选择 (1-4): 

if "%choice%"=="1" goto v2
if "%choice%"=="2" goto install
if "%choice%"=="3" goto ragas