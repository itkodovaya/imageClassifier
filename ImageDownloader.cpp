#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include "ImageDownloader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

namespace fs = std::filesystem;

ImageDownloader::ImageDownloader() {
    // Создание директорий для загрузок
    if (!fs::exists("downloaded_images")) {
        fs::create_directory("downloaded_images");
    }
    if (!fs::exists("training_images")) {
        fs::create_directory("training_images");
    }
}

string ImageDownloader::findDownloadTool() {
    // Поиск доступного инструмента для загрузки
    #ifdef _WIN32
    // Проверка curl
    int result = system("curl --version > nul 2>&1");
    if (result == 0) return "curl";
    
    // Проверка PowerShell (встроен в Windows)
    return "powershell";
    #else
    // Linux/Mac
    int result = system("which curl > /dev/null 2>&1");
    if (result == 0) return "curl";
    
    result = system("which wget > /dev/null 2>&1");
    if (result == 0) return "wget";
    
    return "";
    #endif
}

bool ImageDownloader::downloadWithCurl(const string& url, const string& output) {
    string command = "curl -L -o \"" + output + "\" \"" + url + "\"";
    int result = system(command.c_str());
    return result == 0 && fs::exists(output);
}

bool ImageDownloader::downloadWithWget(const string& url, const string& output) {
    string command = "wget -O \"" + output + "\" \"" + url + "\"";
    int result = system(command.c_str());
    return result == 0 && fs::exists(output);
}

bool ImageDownloader::downloadFromURL(const string& url, const string& output_path) {
    string tool = findDownloadTool();
    
    if (tool == "curl") {
        return downloadWithCurl(url, output_path);
    } else if (tool == "wget") {
        return downloadWithWget(url, output_path);
    } else if (tool == "powershell") {
        // Использование PowerShell для загрузки
        #ifdef _WIN32
        string command = "powershell -Command \"Invoke-WebRequest -Uri '" + url + 
                        "' -OutFile '" + output_path + "'\"";
        int result = system(command.c_str());
        return result == 0 && fs::exists(output_path);
        #endif
    }
    
    return false;
}

vector<string> ImageDownloader::generateImageURLs(const string& query, int count) {
    vector<string> urls;
    
    // Использование Unsplash Source API (публичный, не требует ключа)
    // Формат: https://source.unsplash.com/featured/?{query}
    // Или можно использовать другие публичные источники
    
    // Генерация URL для Unsplash с разными размерами для разнообразия
    vector<string> sizes = {"400x300", "600x400", "800x600", "1024x768"};
    for (int i = 0; i < count; ++i) {
        // Unsplash Source API - бесплатный публичный API
        string size = sizes[i % sizes.size()];
        string url = "https://source.unsplash.com/" + size + "/?" + query;
        if (i > 0) {
            url += "&sig=" + to_string(i); // Добавляем сигнатуру для разных изображений
        }
        urls.push_back(url);
    }
    
    // Дополнительно: используем Picsum Photos для разнообразия (если Unsplash не работает)
    // Picsum предоставляет случайные изображения
    for (int i = 0; i < min(count / 2, 20); ++i) {
        string url = "https://picsum.photos/800/600?random=" + to_string(i);
        urls.push_back(url);
    }
    
    // Альтернатива: можно использовать другие источники
    // - Picsum Photos: https://picsum.photos/800/600?random={i}
    // - Lorem Picsum: https://picsum.photos/800/600
    // - Placeholder.com: https://via.placeholder.com/800x600
    
    return urls;
}

vector<string> ImageDownloader::downloadImages(const string& query, 
                                              int count,
                                              const string& output_dir) {
    vector<string> downloaded_paths;
    
    // Создание директории
    if (!fs::exists(output_dir)) {
        fs::create_directory(output_dir);
    }
    
    // Генерация URL
    vector<string> urls = generateImageURLs(query, count);
    
    if (urls.empty()) {
        cout << "ВНИМАНИЕ: Автоматическая загрузка из интернета требует настройки API." << endl;
        cout << "Используйте готовые датасеты или укажите URL вручную." << endl;
        return downloaded_paths;
    }
    
    // Загрузка
    cout << "Загрузка " << urls.size() << " изображений..." << endl;
    for (size_t i = 0; i < urls.size(); ++i) {
        string filename = output_dir + "/image_" + to_string(i) + ".jpg";
        if (downloadFromURL(urls[i], filename)) {
            downloaded_paths.push_back(filename);
            if ((i + 1) % 10 == 0) {
                cout << "Загружено: " << (i + 1) << "/" << urls.size() << endl;
            }
        }
    }
    
    return downloaded_paths;
}

vector<string> ImageDownloader::downloadBatch(const vector<string>& urls,
                                             const string& output_dir) {
    vector<string> downloaded_paths;
    
    if (!fs::exists(output_dir)) {
        fs::create_directory(output_dir);
    }
    
    cout << "Загрузка " << urls.size() << " изображений..." << endl;
    for (size_t i = 0; i < urls.size(); ++i) {
        string filename = output_dir + "/img_" + to_string(i) + ".jpg";
        if (downloadFromURL(urls[i], filename)) {
            downloaded_paths.push_back(filename);
        }
    }
    
    return downloaded_paths;
}

