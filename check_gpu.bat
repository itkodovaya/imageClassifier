@echo off
chcp 65001 >nul
echo ==========================================
echo   Проверка GPU для обучения
echo ==========================================
echo.

echo [1] Проверка NVIDIA драйвера и GPU...
nvidia-smi
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ОШИБКА] nvidia-smi не найден или не работает.
    echo.
    echo Возможные причины:
    echo   - Драйвер NVIDIA не установлен
    echo   - CUDA не установлена или CUDA_PATH не задан
    echo   - Видеокарта не поддерживает CUDA
    echo.
    echo Действия:
    echo   1. Скачайте драйвер с https://www.nvidia.com/drivers
    echo   2. Установите CUDA Toolkit с https://developer.nvidia.com/cuda-downloads
    echo   3. Перезапустите компьютер после установки
    goto :eof
)

echo.
echo [INFO] CUDA 13 - driver 580+, CUDA 12 - driver 525+
echo.
echo [2] CUDA_PATH...
if defined CUDA_PATH (
    echo CUDA_PATH = %CUDA_PATH%
    if exist "%CUDA_PATH%\bin\nvcc.exe" (
        echo nvcc найден: %CUDA_PATH%\bin\nvcc.exe
    ) else (
        echo [ВНИМАНИЕ] nvcc не найден по пути CUDA_PATH
    )
) else (
    echo [ВНИМАНИЕ] CUDA_PATH не задана.
    echo Установите CUDA Toolkit - переменная создаётся автоматически.
)

echo.
echo ==========================================
echo   Проверка завершена
echo ==========================================
pause
