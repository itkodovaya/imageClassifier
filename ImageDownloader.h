#ifndef IMAGE_DOWNLOADER_H
#define IMAGE_DOWNLOADER_H

#include <string>
#include <vector>
#include <map>

using namespace std;

// Класс для загрузки изображений из интернета
class ImageDownloader {
public:
    ImageDownloader();
    
    // Загрузка изображений по запросу
    vector<string> downloadImages(const string& query, 
                                  int count = 100,
                                  const string& output_dir = "downloaded_images");
    
    // Загрузка из конкретного URL
    bool downloadFromURL(const string& url, const string& output_path);
    
    // Пакетная загрузка из списка URL
    vector<string> downloadBatch(const vector<string>& urls,
                                 const string& output_dir);
    
    // Генерация URL для популярных источников
    vector<string> generateImageURLs(const string& query, int count);
    
private:
    // Использование системных утилит для загрузки
    bool downloadWithCurl(const string& url, const string& output);
    bool downloadWithWget(const string& url, const string& output);
    
    string findDownloadTool();
};

#endif

