#include "TextRenderer.h"

#ifdef _WIN32
// Используем только необходимые функции Windows API через extern
// чтобы избежать конфликтов с std::byte

extern "C" {
#ifdef _MSC_VER
    __declspec(dllimport) void* __stdcall CreateCompatibleDC(void*);
    __declspec(dllimport) void* __stdcall CreateDIBSection(void*, void*, unsigned int, void**, void*, unsigned int);
    __declspec(dllimport) void* __stdcall SelectObject(void*, void*);
    __declspec(dllimport) int __stdcall SetBkMode(void*, int);
    __declspec(dllimport) unsigned long __stdcall SetTextColor(void*, unsigned long);
    __declspec(dllimport) int __stdcall TextOutW(void*, int, int, const wchar_t*, int);
    __declspec(dllimport) int __stdcall DeleteObject(void*);
    __declspec(dllimport) int __stdcall DeleteDC(void*);
    __declspec(dllimport) void* __stdcall CreateFontW(int, int, int, int, int, int, int, int, unsigned int, int, int, int, int, const wchar_t*);
    #define CreateFont CreateFontW
    __declspec(dllimport) int __stdcall MultiByteToWideChar(unsigned int, unsigned long, const char*, int, wchar_t*, int);
    __declspec(dllimport) void* __stdcall GetStockObject(int);
    __declspec(dllimport) int __stdcall FillRect(void*, void*, void*);
    __declspec(dllimport) int __stdcall GetDIBits(void*, void*, unsigned int, unsigned int, void*, void*, unsigned int);
    __declspec(dllimport) int __stdcall GetTextExtentPoint32W(void*, const wchar_t*, int, void*);
    #else
    void* __attribute__((stdcall)) CreateCompatibleDC(void*);
    void* __attribute__((stdcall)) CreateDIBSection(void*, void*, unsigned int, void**, void*, unsigned int);
    void* __attribute__((stdcall)) SelectObject(void*, void*);
    int __attribute__((stdcall)) SetBkMode(void*, int);
    unsigned long __attribute__((stdcall)) SetTextColor(void*, unsigned long);
    int __attribute__((stdcall)) TextOutW(void*, int, int, const wchar_t*, int);
    int __attribute__((stdcall)) DeleteObject(void*);
    int __attribute__((stdcall)) DeleteDC(void*);
    void* __attribute__((stdcall)) CreateFontW(int, int, int, int, int, int, int, int, unsigned int, int, int, int, int, const wchar_t*);
    #define CreateFont CreateFontW
    int __attribute__((stdcall)) MultiByteToWideChar(unsigned int, unsigned long, const char*, int, wchar_t*, int);
    void* __attribute__((stdcall)) GetStockObject(int);
    int __attribute__((stdcall)) FillRect(void*, void*, void*);
    int __attribute__((stdcall)) GetDIBits(void*, void*, unsigned int, unsigned int, void*, void*, unsigned int);
    int __attribute__((stdcall)) GetTextExtentPoint32W(void*, const wchar_t*, int, void*);


    #endif
}

struct RECT {
    long left;
    long top;
    long right;
    long bottom;
};

#define BLACK_BRUSH 4

#define TRANSPARENT 1
#define DEFAULT_CHARSET 1
#define OUT_DEFAULT_PRECIS 0
#define CLIP_DEFAULT_PRECIS 0
#define ANTIALIASED_QUALITY 4
#define DEFAULT_PITCH 0
#define FF_DONTCARE 0
#define FW_NORMAL 400
#define FW_BOLD 700
#define CP_UTF8 65001
#define DIB_RGB_COLORS 0
#define BI_RGB 0
#define FALSE 0
#define TRUE 1
#define RGB(r, g, b) ((unsigned long)(((unsigned char)(r) | ((unsigned short)((unsigned char)(g)) << 8)) | (((unsigned long)(unsigned char)(b)) << 16)))

struct BITMAPINFOHEADER {
    unsigned long biSize;
    long biWidth;
    long biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned long biCompression;
    unsigned long biSizeImage;
    long biXPelsPerMeter;
    long biYPelsPerMeter;
    unsigned long biClrUsed;
    unsigned long biClrImportant;
};

struct BITMAPINFO {
    BITMAPINFOHEADER bmiHeader;
    unsigned long bmiColors[1];
};

typedef void* HDC;
typedef void* HBITMAP;
typedef void* HFONT;
typedef void* HBRUSH;

struct TEXT_SIZE {
    long cx;
    long cy;
};
#endif

using namespace cv;
using namespace std;

// Функция для получения размера русского текста
Size getTextSizeRussian(const string& text, double fontScale, int thickness) {
#ifdef _WIN32
    if (text.empty()) return Size(0, 0);
    
    // Конвертация UTF-8 в широкую строку
    int wsize = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, NULL, 0);
    if (wsize <= 0) return Size(0, 0);
    
    wstring wtext(wsize, 0);
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, &wtext[0], wsize);
    wtext.resize(wsize - 1);
    
    // Создание временного DC для измерения текста
    HDC hdc = CreateCompatibleDC(NULL);
    if (!hdc) return Size(0, 0);
    
    int fontSize = static_cast<int>(fontScale * 30);
    HFONT hFont = CreateFont(fontSize, 0, 0, 0, 
                            thickness > 1 ? FW_BOLD : FW_NORMAL,
                            FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                            ANTIALIASED_QUALITY, DEFAULT_PITCH | FF_DONTCARE, 
                            L"Arial");
    HFONT hOldFont = (HFONT)SelectObject(hdc, hFont);
    
    struct TEXT_SIZE textSizeStruct;
    GetTextExtentPoint32W(hdc, wtext.c_str(), static_cast<int>(wtext.length()), (void*)&textSizeStruct);
    
    SelectObject(hdc, hOldFont);
    DeleteObject(hFont);
    DeleteDC(hdc);
    
    return Size(textSizeStruct.cx, textSizeStruct.cy);
#else
    // Fallback для не-Windows систем
    int baseline = 0;
    Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
    return textSize;
#endif
}

// Функция для отображения русского текста с использованием Windows API
void putTextRussian(Mat& img, const string& text, Point pos, int fontFace, 
                   double fontScale, Scalar color, int thickness) {
#ifdef _WIN32
    if (text.empty()) return;
    
    // Конвертация UTF-8 в широкую строку
    int wsize = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, NULL, 0);
    if (wsize <= 0) {
        putText(img, text, pos, fontFace, fontScale, color, thickness);
        return;
    }
    
    wstring wtext(wsize, 0);
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, &wtext[0], wsize);
    wtext.resize(wsize - 1);
    
    // Оценка размера текста (увеличен для лучшей видимости)
    int fontSize = static_cast<int>(fontScale * 30); // Увеличен множитель
    int textWidth = static_cast<int>(wtext.length() * fontSize * 0.65) + 30; // Больше отступы
    int textHeight = fontSize + 25;
    
    // Создание DC и битмапа
    HDC hdc = CreateCompatibleDC(NULL);
    if (!hdc) {
        putText(img, text, pos, fontFace, fontScale, color, thickness);
        return;
    }
    
    BITMAPINFO bmi = {0};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = textWidth;
    bmi.bmiHeader.biHeight = -textHeight; // Отрицательное для top-down
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    
    void* pBits = NULL;
    HBITMAP hBitmap = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, &pBits, NULL, 0);
    if (!hBitmap) {
        DeleteDC(hdc);
        putText(img, text, pos, fontFace, fontScale, color, thickness);
        return;
    }
    
    SelectObject(hdc, hBitmap);
    SetBkMode(hdc, TRANSPARENT);
    
    // Заливаем фон черным (будет использоваться как маска)
    // Просто заполняем память нулями
    memset(pBits, 0, textHeight * textWidth * 4);
    
    // Настройка шрифта
    HFONT hFont = CreateFont(fontSize, 0, 0, 0, 
                            thickness > 1 ? FW_BOLD : FW_NORMAL,
                            FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                            ANTIALIASED_QUALITY, DEFAULT_PITCH | FF_DONTCARE, 
                            L"Arial");
    HFONT hOldFont = (HFONT)SelectObject(hdc, hFont);
    
    // Цвет текста (BGR -> RGB)
    SetTextColor(hdc, RGB(color[2], color[1], color[0]));
    
    // Отрисовка текста
    TextOutW(hdc, 10, 10, wtext.c_str(), static_cast<int>(wtext.length()));
    
    // Копирование из битмапа в Mat
    // Используем GetDIBits для правильного чтения данных в формате BGRA
    Mat textImg(textHeight, textWidth, CV_8UC4);
    
    BITMAPINFO bmi_read = {0};
    bmi_read.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi_read.bmiHeader.biWidth = textWidth;
    bmi_read.bmiHeader.biHeight = -textHeight; // Top-down DIB
    bmi_read.bmiHeader.biPlanes = 1;
    bmi_read.bmiHeader.biBitCount = 32;
    bmi_read.bmiHeader.biCompression = BI_RGB;
    
    // Получаем данные через GetDIBits (правильный способ чтения DIB)
    int lines = GetDIBits(hdc, hBitmap, 0, textHeight, textImg.data, &bmi_read, DIB_RGB_COLORS);
    
    if (lines == 0 || lines != textHeight) {
        // Fallback: прямое копирование с правильным учетом stride
        int stride = ((textWidth * 4 + 3) / 4) * 4; // Выравнивание по 4 байтам
        unsigned char* src = (unsigned char*)pBits;
        unsigned char* dst = textImg.data;
        
        for (int y = 0; y < textHeight; y++) {
            for (int x = 0; x < textWidth; x++) {
                int src_idx = y * stride + x * 4;
                int dst_idx = y * textWidth * 4 + x * 4;
                
                // DIB 32-bit формат: BGRA (Blue, Green, Red, Alpha/Reserved)
                // Копируем напрямую
                dst[dst_idx + 0] = src[src_idx + 0]; // B
                dst[dst_idx + 1] = src[src_idx + 1]; // G
                dst[dst_idx + 2] = src[src_idx + 2]; // R
                
                // Альфа: если пиксель не черный (есть текст), делаем непрозрачным
                unsigned char r = src[src_idx + 2];
                unsigned char g = src[src_idx + 1];
                unsigned char b = src[src_idx + 0];
                if (r > 20 || g > 20 || b > 20) {
                    dst[dst_idx + 3] = 255; // Непрозрачный
                } else {
                    dst[dst_idx + 3] = 0; // Прозрачный
                }
            }
        }
    } else {
        // GetDIBits успешно скопировал данные, но нужно обработать альфа-канал
        unsigned char* dst = textImg.data;
        for (int i = 0; i < textHeight * textWidth; i++) {
            int idx = i * 4;
            unsigned char r = dst[idx + 2];
            unsigned char g = dst[idx + 1];
            unsigned char b = dst[idx + 0];
            // Устанавливаем альфа на основе яркости
            if (r > 20 || g > 20 || b > 20) {
                dst[idx + 3] = 255;
            } else {
                dst[idx + 3] = 0;
            }
        }
    }
    
    // Конвертируем BGRA в BGR
    Mat textImgBGR;
    cvtColor(textImg, textImgBGR, COLOR_BGRA2BGR);
    
    // Создание маски из альфа-канала (более надежно)
    Mat mask;
    vector<Mat> channels;
    split(textImg, channels);
    if (channels.size() >= 4) {
        // Используем альфа-канал напрямую
        mask = channels[3];
    } else {
        // Fallback: создаем маску из яркости
        Mat gray;
        cvtColor(textImgBGR, gray, COLOR_BGR2GRAY);
        threshold(gray, mask, 10, 255, THRESH_BINARY);
    }
    
    // Вставка текста в изображение (только непрозрачные пиксели)
    Rect roi(pos.x, pos.y, min(textWidth, img.cols - pos.x), 
             min(textHeight, img.rows - pos.y));
    if (roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= img.cols && 
        roi.y + roi.height <= img.rows) {
        Mat roiImg = img(roi);
        Mat textRoi = textImgBGR(Rect(0, 0, roi.width, roi.height));
        Mat maskRoi = mask(Rect(0, 0, roi.width, roi.height));
        
        // Копируем только пиксели текста
        textRoi.copyTo(roiImg, maskRoi);
    }
    
    // Очистка
    SelectObject(hdc, hOldFont);
    DeleteObject(hFont);
    DeleteObject(hBitmap);
    DeleteDC(hdc);
#else
    // Fallback для не-Windows систем
    putText(img, text, pos, fontFace, fontScale, color, thickness);
#endif
}

