#include "ConsoleOutput.h"

#ifdef _WIN32
// Глобальные дескрипторы консоли
HANDLE g_hConsoleOutput = NULL;
HANDLE g_hConsoleError = NULL;
bool g_console_initialized = false;

void init_console_output() {
    if (!g_console_initialized) {
        g_hConsoleOutput = GetStdHandle(STD_OUTPUT_HANDLE);
        g_hConsoleError = GetStdHandle(STD_ERROR_HANDLE);
        g_console_initialized = true;
    }
}

// Конвертация UTF-8 строки в широкую строку
wstring utf8_to_wstring(const string& str) {
    if (str.empty()) return wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.length(), NULL, 0);
    if (size_needed <= 0) return wstring();
    wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.length(), &wstr[0], size_needed);
    return wstr;
}

void safe_cout(const string& text) {
    init_console_output();
    
    if (!text.empty() && g_hConsoleOutput && g_hConsoleOutput != INVALID_HANDLE_VALUE) {
        wstring wstr = utf8_to_wstring(text);
        if (!wstr.empty()) {
            DWORD written = 0;
            if (WriteConsoleW(g_hConsoleOutput, wstr.c_str(), (DWORD)wstr.length(), &written, NULL)) {
                return; // Успешно выведено
            }
        }
    }
    // Fallback на обычный вывод
    cout << text;
    cout.flush();
}

void safe_cerr(const string& text) {
    init_console_output();
    
    if (!text.empty() && g_hConsoleError && g_hConsoleError != INVALID_HANDLE_VALUE) {
        wstring wstr = utf8_to_wstring(text);
        if (!wstr.empty()) {
            DWORD written = 0;
            if (WriteConsoleW(g_hConsoleError, wstr.c_str(), (DWORD)wstr.length(), &written, NULL)) {
                return; // Успешно выведено
            }
        }
    }
    // Fallback на обычный вывод
    cerr << text;
    cerr.flush();
}
#endif

