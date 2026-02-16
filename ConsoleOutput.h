#ifndef CONSOLE_OUTPUT_H
#define CONSOLE_OUTPUT_H

#include <string>
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#endif

using namespace std;

#ifdef _WIN32
// Глобальные дескрипторы консоли
extern HANDLE g_hConsoleOutput;
extern HANDLE g_hConsoleError;
extern bool g_console_initialized;

// Инициализация консоли
void init_console_output();

// Конвертация UTF-8 в широкую строку
wstring utf8_to_wstring(const string& str);

// Безопасный вывод в консоль
void safe_cout(const string& text);
void safe_cerr(const string& text);

// Макросы для удобного использования
#define COUT_SAFE safe_cout
#define CERR_SAFE safe_cerr
#else
#define COUT_SAFE(x) cout << x
#define CERR_SAFE(x) cerr << x
inline void init_console_output() {}
#endif

#endif // CONSOLE_OUTPUT_H

