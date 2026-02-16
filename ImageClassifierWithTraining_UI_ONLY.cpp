// Временный файл для запуска ТОЛЬКО UI без консольного меню
// Это копия ImageClassifierWithTraining.cpp с гарантией, что UI откроется первым

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commdlg.h>
#endif

#include "UniversalImageClassifier.h"
#include "ImageDownloader.h"
#include "CudaAccelerator.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <thread>
#include <future>
#include <vector>
#include <string>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <map>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Включаем все функции из основного файла
extern void drawMainScreen();
extern void drawTrainingScreen();
extern void onMouse(int event, int x, int y, int flags, void* userdata);

// Простой main, который ТОЛЬКО открывает UI
int main() {
#ifdef _WIN32
    // Скрываем консоль СРАЗУ
    FreeConsole();
    // Перенаправляем весь вывод
    FILE* null_out;
    FILE* null_err;
    freopen_s(&null_out, "nul", "w", stdout);
    freopen_s(&null_err, "nul", "w", stderr);
#endif
    
    // Создаем окно ПЕРВЫМ
    namedWindow("Image Classifier", WINDOW_NORMAL);
    resizeWindow("Image Classifier", 1200, 900);
    setMouseCallback("Image Classifier", onMouse);
    moveWindow("Image Classifier", 100, 100);
    
    // Показываем окно
    Mat initial_screen = Mat::zeros(900, 1200, CV_8UC3);
    putText(initial_screen, "Loading UI...", Point(500, 450), 
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
    imshow("Image Classifier", initial_screen);
    waitKey(10);
    
#ifdef _WIN32
    Sleep(200);
    HWND hwnd = FindWindow(NULL, L"Image Classifier");
    if (hwnd) {
        ShowWindow(hwnd, SW_SHOW);
        SetForegroundWindow(hwnd);
        BringWindowToTop(hwnd);
    }
#endif
    
    // Теперь инициализируем классификатор (тихо)
    // ... остальной код из основного файла
    
    // Простой цикл
    while (true) {
        drawMainScreen();
        int key = waitKey(1);
        if (key == 27) break;
        
        try {
            if (getWindowProperty("Image Classifier", WND_PROP_VISIBLE) < 1) {
                break;
            }
        } catch(...) { 
            break; 
        }
    }
    
    destroyAllWindows();
    return 0;
}

