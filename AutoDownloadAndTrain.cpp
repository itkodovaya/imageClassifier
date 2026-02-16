// AutoDownloadAndTrain.cpp - Автоматическая загрузка изображений из интернета и обучение модели
// Компилируется и запускается отдельно

#include "UniversalImageClassifier.h"
#include "ImageDownloader.h"
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

using namespace std;
using namespace cv;

// Список категорий для загрузки и обучения
const vector<string> CATEGORIES = {
    // Еда
    "dessert", "cake", "ice_cream", "chocolate", "cookie", "pie", "pudding",
    "pizza", "burger", "sandwich", "sushi", "pasta", "salad", "soup",
    "coffee", "tea", "juice", "apple", "banana", "orange", "strawberry",
    "bread", "cheese", "milk", "egg",
    
    // Транспорт
    "car", "sports_car", "truck", "bus", "motorcycle", "bicycle", "scooter",
    "airplane", "helicopter", "boat", "ship", "train",
    
    // Животные
    "dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep", "goat",
    "rabbit", "elephant", "lion", "tiger", "bear", "wolf", "fox", "deer",
    "zebra", "giraffe", "monkey", "panda",
    
    // Одежда
    "shirt", "pants", "dress", "skirt", "jacket", "coat", "sweater",
    "hat", "cap", "shoes", "boots", "sneakers",
    
    // Электроника
    "phone", "smartphone", "tablet", "laptop", "computer", "monitor",
    "keyboard", "mouse", "camera", "television", "speaker", "headphones",
    
    // Мебель
    "chair", "table", "desk", "sofa", "couch", "bed", "wardrobe", "cabinet",
    
    // Природа
    "tree", "flower", "rose", "tulip", "sunflower", "mountain", "hill",
    "river", "lake", "ocean", "beach", "forest", "desert",
    
    // Здания
    "house", "building", "apartment", "office", "school", "hospital",
    "church", "temple", "castle", "tower", "bridge",
    
    // Спорт
    "football", "basketball", "soccer_ball", "tennis_ball", "baseball",
    "volleyball", "golf_ball", "skateboard", "bicycle"
};

void downloadCategoryImages(ImageDownloader& downloader, const string& category, int count) {
    cout << "  Загрузка: " << category << " (" << count << " изображений)..." << flush;
    
    string categoryDir = "training_images/" + category;
    if (!fs::exists(categoryDir)) {
        fs::create_directories(categoryDir);
    }
    
    vector<string> paths = downloader.downloadImages(category, count, categoryDir);
    
    cout << " ✓ Загружено: " << paths.size() << " изображений" << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "АВТОМАТИЧЕСКАЯ ЗАГРУЗКА И ОБУЧЕНИЕ" << endl;
    cout << "========================================" << endl;
    cout << endl;
    
    // Создаем папку для обучения
    if (!fs::exists("training_images")) {
        fs::create_directories("training_images");
    }
    
    // Инициализируем загрузчик
    cout << "Инициализация загрузчика изображений..." << endl;
    ImageDownloader downloader;
    
    // Загружаем изображения для каждой категории
    cout << endl;
    cout << "Начинаю загрузку " << CATEGORIES.size() << " категорий..." << endl;
    cout << "Это может занять некоторое время..." << endl;
    cout << endl;
    
    int imagesPerCategory = 30;  // По 30 изображений на категорию
    int totalCategories = CATEGORIES.size();
    int currentCategory = 0;
    
    for (const string& category : CATEGORIES) {
        currentCategory++;
        cout << "[" << currentCategory << "/" << totalCategories << "] ";
        
        try {
            downloadCategoryImages(downloader, category, imagesPerCategory);
        } catch (const exception& e) {
            cout << " ✗ Ошибка: " << e.what() << endl;
        }
        
        // Небольшая задержка между категориями
        this_thread::sleep_for(chrono::milliseconds(500));
    }
    
    cout << endl;
    cout << "========================================" << endl;
    cout << "ЗАГРУЗКА ЗАВЕРШЕНА!" << endl;
    cout << "========================================" << endl;
    cout << endl;
    
    // Подсчитываем загруженные изображения
    int totalImages = 0;
    map<string, int> category_counts;
    
    for (const auto& category_entry : fs::directory_iterator("training_images")) {
        if (category_entry.is_directory()) {
            string category_name = category_entry.path().filename().string();
            int count = 0;
            
            for (const auto& entry : fs::directory_iterator(category_entry.path())) {
                if (entry.is_regular_file()) {
                    string ext = entry.path().extension().string();
                    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                        count++;
                        totalImages++;
                    }
                }
            }
            
            if (count > 0) {
                category_counts[category_name] = count;
            }
        }
    }
    
    cout << "Статистика загрузки:" << endl;
    cout << "  - Категорий: " << category_counts.size() << endl;
    cout << "  - Всего изображений: " << totalImages << endl;
    cout << endl;
    
    if (totalImages == 0) {
        cout << "ОШИБКА: Не загружено ни одного изображения!" << endl;
        cout << "Проверьте интернет-соединение и попробуйте снова." << endl;
        return 1;
    }
    
    // Начинаем обучение модели
    cout << "========================================" << endl;
    cout << "НАЧИНАЮ ОБУЧЕНИЕ МОДЕЛИ" << endl;
    cout << "========================================" << endl;
    cout << endl;
    
    // Собираем все изображения и метки
    vector<string> image_paths;
    vector<int> labels;
    map<string, int> category_to_label;
    int next_label = 0;
    
    cout << "Сканирование загруженных изображений..." << endl;
    
    for (const auto& category_entry : fs::directory_iterator("training_images")) {
        if (category_entry.is_directory()) {
            string category_name = category_entry.path().filename().string();
            
            // Создаем метку для этой категории
            if (category_to_label.find(category_name) == category_to_label.end()) {
                category_to_label[category_name] = next_label++;
            }
            int category_label = category_to_label[category_name];
            
            // Собираем все изображения из этой категории
            for (const auto& entry : fs::directory_iterator(category_entry.path())) {
                if (entry.is_regular_file()) {
                    string ext = entry.path().extension().string();
                    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                        image_paths.push_back(entry.path().string());
                        labels.push_back(category_label);
                    }
                }
            }
        }
    }
    
    int num_classes = category_to_label.size();
    cout << "  - Найдено классов: " << num_classes << endl;
    cout << "  - Найдено изображений: " << image_paths.size() << endl;
    cout << endl;
    
    // Создаем и настраиваем классификатор
    cout << "Инициализация классификатора (" << num_classes << " классов)..." << endl;
    UniversalImageClassifier classifier(num_classes);
    
    // Добавляем названия классов
    for (const auto& [category, label] : category_to_label) {
        classifier.addClass(category);
        cout << "  Класс " << label << ": " << category << endl;
    }
    cout << endl;
    
    // Обучение модели
    cout << "Начало обучения модели..." << endl;
    cout << "Это может занять некоторое время..." << endl;
    cout << "Прогресс будет отображаться ниже:" << endl;
    cout << endl;
    
    int epochs = 50;  // Количество эпох обучения
    classifier.trainOnDataset(image_paths, labels, epochs);
    
    // Сохранение модели
    cout << endl;
    cout << "Сохранение обученной модели..." << endl;
    classifier.saveModel("universal_classifier_model.json");
    
    cout << endl;
    cout << "========================================" << endl;
    cout << "ОБУЧЕНИЕ ЗАВЕРШЕНО!" << endl;
    cout << "========================================" << endl;
    cout << "Модель сохранена в: universal_classifier_model.json" << endl;
    cout << endl;
    cout << "Теперь можно использовать модель для классификации!" << endl;
    cout << "Запустите ImageClassifierWithTraining.exe и загрузите изображение." << endl;
    cout << endl;
    
    return 0;
}

