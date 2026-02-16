@echo off
REM Скрипт для автоматической загрузки изображений по категориям для обучения модели
REM Использует PowerShell для загрузки изображений из Unsplash

echo ========================================
echo Загрузка изображений для обучения модели
echo ========================================
echo.

REM Список категорий для загрузки
set CATEGORIES=dessert cake ice_cream chocolate cookie pizza burger sushi pasta coffee tea apple banana car truck bus motorcycle airplane boat dog cat bird horse elephant lion tiger shirt dress shoes phone laptop camera chair table sofa bed tree flower mountain ocean house building ball football basketball

REM Создаем папку для изображений
if not exist "training_images" mkdir training_images

REM Загружаем по 20 изображений для каждой категории
for %%c in (%CATEGORIES%) do (
    echo Загрузка категории: %%c
    if not exist "training_images\%%c" mkdir "training_images\%%c"
    
    REM Используем PowerShell для загрузки из Unsplash
    powershell -Command "for ($i=0; $i -lt 20; $i++) { $url = 'https://source.unsplash.com/800x600/?%%c&sig=' + $i; $output = 'training_images\%%c\image_' + $i + '.jpg'; try { Invoke-WebRequest -Uri $url -OutFile $output -TimeoutSec 10 -ErrorAction SilentlyContinue } catch {} }"
    
    echo Завершено: %%c
    echo.
)

echo ========================================
echo Загрузка завершена!
echo ========================================
pause

