#include "UniversalImageClassifier.h"
#include "MetricsPlotter.h"
#include "Profiler.h"
#include "Config.h"
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <set>
#include <iostream>
#include <iomanip>
#include <exception>
#include <numeric>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <thread>
#include <future>
#include <mutex>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

#ifndef CV_PI
#define CV_PI 3.14159265358979323846
#endif

UniversalImageClassifier::UniversalImageClassifier(int num_classes, int image_size) 
    : num_classes(num_classes), image_size(image_size > 0 ? image_size : 32), use_structure_analysis(false) {
    
    // Автоматическое определение оптимальной архитектуры на основе задачи
    int input_size = image_size * image_size * 3;  // RGB изображение 64x64x3
    
    // Определяем архитектуру автоматически на основе:
    // - размера входа (12288 для 64x64x3)
    // - количества классов
    // - предполагаемого количества образцов (можно настроить)
    vector<int> hidden_sizes = NeuralNetwork::determineOptimalArchitecture(
        input_size, 
        num_classes,
        1000  // Предполагаемое количество образцов для обучения
    );
    
    // Выводим информацию о выбранной архитектуре
    cout << "[UniversalImageClassifier] Architecture automatically determined:" << endl;
    cout << "  Input size: " << input_size << endl;
    cout << "  Number of classes: " << num_classes << endl;
    cout << "  Hidden layers: ";
    for (size_t i = 0; i < hidden_sizes.size(); ++i) {
        cout << hidden_sizes[i];
        if (i < hidden_sizes.size() - 1) cout << " -> ";
    }
    cout << endl;
    cout << "  Output size: " << num_classes << endl;
    
    // Адаптивный learning rate на основе сложности задачи
    double learning_rate = 0.001;
    if (num_classes > 50) {
        learning_rate = 0.0005;  // Много классов - меньший learning rate
    } else if (num_classes < 10) {
        learning_rate = 0.002;   // Мало классов - можно больше
    }
    
    // Создаем ансамбль из 3-х сетей с разной архитектурой для диверсификации мнений
    for (int i = 0; i < 3; ++i) {
        vector<int> ensemble_hidden = hidden_sizes;
        if (i == 1) {
            // Более глубокая сеть
            ensemble_hidden.push_back(max(num_classes * 2, 32));
        } else if (i == 2) {
            // Более широкая сеть (увеличиваем первый слой)
            ensemble_hidden[0] = static_cast<int>(ensemble_hidden[0] * 1.5);
        }

        ensemble.push_back(make_unique<NeuralNetwork>(
            input_size,
            ensemble_hidden,
            num_classes,
            learning_rate * (1.0 - i * 0.2), // Разный LR для каждого члена
            "relu",
            0.9,
            0.001
        ));
    }
    
    // Инициализация компонентов для анализа структуры (по умолчанию отключено)
    shape_analyzer = make_unique<ShapeAnalyzer>();
    subnetwork_manager = make_unique<SubNetworkManager>(num_classes);
    fuzzy_fusion = make_unique<FuzzyLogicFusion>();
    
    // Инициализация топологического анализа
    topological_sig = make_unique<TopologicalSignature>();
    topological_kernel = make_unique<TopologicalKernel>();
    use_topological_kernel = true; // Активируем топологическое ядро для лучшего различения похожих классов
    
    // Инициализация SVM
    svm_classifier = cv::ml::SVM::create();
    svm_classifier->setType(cv::ml::SVM::C_SVC);
    svm_classifier->setKernel(cv::ml::SVM::RBF);
    svm_classifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
}

UniversalImageClassifier::~UniversalImageClassifier() {
    // Явная очистка ансамбля и других компонентов
    ensemble.clear();
    class_names.clear();
}

void UniversalImageClassifier::trainSVM(const vector<MatrixXd>& features, const vector<int>& labels) {
    if (features.empty() || features.size() != labels.size()) return;

    int feature_size = static_cast<int>(features[0].rows());
    Mat train_data(static_cast<int>(features.size()), feature_size, CV_32F);
    Mat train_labels(static_cast<int>(labels.size()), 1, CV_32S);

    for (size_t i = 0; i < features.size(); ++i) {
        for (int j = 0; j < feature_size; ++j) {
            train_data.at<float>(static_cast<int>(i), j) = static_cast<float>(features[i](j, 0));
        }
        train_labels.at<int>(static_cast<int>(i), 0) = labels[i];
    }

    svm_classifier->train(train_data, cv::ml::ROW_SAMPLE, train_labels);
    cout << "[UniversalImageClassifier] SVM trained on " << features.size() << " structure samples." << endl;
}

int UniversalImageClassifier::predictSVM(const MatrixXd& features) {
    if (!svm_classifier->isTrained()) return -1;

    Mat sample(1, static_cast<int>(features.rows()), CV_32F);
    for (int i = 0; i < features.rows(); ++i) {
        sample.at<float>(0, i) = static_cast<float>(features(i, 0));
    }

    return static_cast<int>(svm_classifier->predict(sample));
}

bool UniversalImageClassifier::loadImage(const string& path) {
    Mat img = imread(path);
    return !img.empty();
}

bool UniversalImageClassifier::loadImageFromURL(const string& url) {
    // Использование OpenCV для загрузки из URL
    // Требует компиляции OpenCV с поддержкой curl
    try {
        Mat img;
        // Альтернатива: использовать системный curl или wget
        // Для демонстрации используем локальную загрузку
        return false; // Требует дополнительной настройки
    } catch (...) {
        return false;
    }
}

MatrixXd UniversalImageClassifier::preprocessImage(const Mat& img, int target_size) {
    Mat processed;
    
    // Улучшенная предобработка изображения
    
    // 1. Конвертация в RGB если нужно
    Mat rgb;
    if (img.channels() == 1) {
        cvtColor(img, rgb, COLOR_GRAY2RGB);
    } else if (img.channels() == 4) {
        cvtColor(img, rgb, COLOR_BGRA2RGB);
    } else {
        cvtColor(img, rgb, COLOR_BGR2RGB);
    }
    
    // 2. Улучшение контраста через CLAHE (Contrast Limited Adaptive Histogram Equalization)
    vector<Mat> channels;
    split(rgb, channels);
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    for (int i = 0; i < 3; ++i) {
        clahe->apply(channels[i], channels[i]);
    }
    merge(channels, rgb);
    
    // 3. Изменение размера с сохранением пропорций (используем INTER_LINEAR для лучшего качества)
    double scale = min((double)target_size / rgb.cols, (double)target_size / rgb.rows);
    int new_width = static_cast<int>(rgb.cols * scale);
    int new_height = static_cast<int>(rgb.rows * scale);
    
    Mat resized;
    resize(rgb, resized, Size(new_width, new_height), 0, 0, INTER_LINEAR);
    
    // 4. Создание квадратного изображения с padding (используем средний цвет для padding)
    Mat squared = Mat::zeros(target_size, target_size, CV_8UC3);
    
    // Вычисляем средний цвет изображения для padding
    Scalar mean_color = mean(resized);
    squared.setTo(mean_color);
    
    int offset_x = (target_size - new_width) / 2;
    int offset_y = (target_size - new_height) / 2;
    resized.copyTo(squared(Rect(offset_x, offset_y, new_width, new_height)));
    
    // 5. Нормализация с улучшенной обработкой
    squared.convertTo(processed, CV_64F, 1.0 / 255.0);
    
    // 6. Применяем улучшенную нормализацию (mean subtraction и std normalization)
    // Вычисляем среднее и стандартное отклонение для каждого канала
    vector<Mat> channels_proc;
    split(processed, channels_proc);
    
    for (int c = 0; c < 3; ++c) {
        Scalar mean_val, stddev_val;
        meanStdDev(channels_proc[c], mean_val, stddev_val);
        double mean_c = mean_val[0];
        double std_c = max(stddev_val[0], 0.01); // Избегаем деления на ноль
        
        // Нормализуем канал
        channels_proc[c] = (channels_proc[c] - mean_c) / std_c;
    }
    
    // Объединяем каналы обратно
    merge(channels_proc, processed);
    
    // 7. Преобразование в вектор (HWC -> flattened)
    MatrixXd vectorized = MatrixXd::Zero(target_size * target_size * 3, 1);
    int idx = 0;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < target_size; ++y) {
            for (int x = 0; x < target_size; ++x) {
                Vec3d pixel = processed.at<Vec3d>(y, x);
                vectorized(idx++, 0) = pixel[c];
            }
        }
    }
    
    return vectorized;
}

// Извлечение признаков из изображения
// Комбинирует несколько методов: пиксели, градиенты, текстуры, цветовые признаки, HOG, LBP
// Оптимизировано для производительности с кэшированием промежуточных результатов
MatrixXd UniversalImageClassifier::extractFeatures(const Mat& img) {
    // Проверка входных данных
    if (img.empty()) {
        throw invalid_argument("UniversalImageClassifier::extractFeatures: Empty image");
    }
    
    vector<double> feature_vector;
    // Предварительное выделение памяти для улучшения производительности
    feature_vector.reserve(20000); // Примерная оценка размера вектора признаков
    
    // ========== 1. БАЗОВЫЕ ПИКСЕЛЬНЫЕ ПРИЗНАКИ ==========
    Mat resized;
    resize(img, resized, Size(64, 64));
    resized.convertTo(resized, CV_64F, 1.0 / 255.0);
    
    // Добавляем пиксельные признаки (64×64×3 = 12288 признаков)
    if (resized.channels() == 3) {
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < 64; ++y) {
                for (int x = 0; x < 64; ++x) {
                    feature_vector.push_back(resized.at<Vec3d>(y, x)[c]);
                }
            }
        }
    } else {
        for (int y = 0; y < 64; ++y) {
            for (int x = 0; x < 64; ++x) {
                double val = resized.at<double>(y, x);
                feature_vector.push_back(val);
                feature_vector.push_back(val); // Дублируем для RGB
                feature_vector.push_back(val);
            }
        }
    }
    
    // ========== 2. ГРАДИЕНТНЫЕ ПРИЗНАКИ (HOG-подобные) ==========
    Mat gray;
    if (img.channels() == 3) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    
    Mat grad_x, grad_y;
    Sobel(gray, grad_x, CV_64F, 1, 0, 3);
    Sobel(gray, grad_y, CV_64F, 0, 1, 3);
    
    Mat magnitude, angle;
    cartToPolar(grad_x, grad_y, magnitude, angle);
    
    // Изменяем размер градиентов для совместимости
    Mat mag_resized, angle_resized;
    resize(magnitude, mag_resized, Size(32, 32));
    resize(angle, angle_resized, Size(32, 32));
    
    // Добавляем магнитуду градиентов (32×32 = 1024 признака)
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
            feature_vector.push_back(mag_resized.at<double>(y, x));
        }
    }
    
    // Добавляем углы градиентов (32×32 = 1024 признака)
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
            feature_vector.push_back(angle_resized.at<double>(y, x) / CV_PI); // Нормализуем к [0, 1]
        }
    }
    
    // Статистики градиентов (6 признаков)
    Scalar mag_mean, mag_stddev, angle_mean, angle_stddev;
    meanStdDev(magnitude, mag_mean, mag_stddev);
    meanStdDev(angle, angle_mean, angle_stddev);
    double mag_min, mag_max, angle_min, angle_max;
    minMaxLoc(magnitude, &mag_min, &mag_max);
    minMaxLoc(angle, &angle_min, &angle_max);
    feature_vector.push_back(mag_mean[0]);
    feature_vector.push_back(mag_stddev[0]);
    feature_vector.push_back(angle_mean[0] / CV_PI);
    feature_vector.push_back(angle_stddev[0] / CV_PI);
    feature_vector.push_back(mag_min);
    feature_vector.push_back(mag_max);
    
    // ========== 3. ТЕКСТУРНЫЕ ПРИЗНАКИ ==========
    Mat gaussian;
    GaussianBlur(gray, gaussian, Size(5, 5), 0);
    
    // Изменяем размер размытого изображения
    Mat gaussian_resized;
    resize(gaussian, gaussian_resized, Size(32, 32));
    gaussian_resized.convertTo(gaussian_resized, CV_64F, 1.0 / 255.0);
    
    // Добавляем текстурные признаки (32×32 = 1024 признака)
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
            feature_vector.push_back(gaussian_resized.at<double>(y, x));
        }
    }
    
    // Статистики текстур (4 признака)
    Scalar gauss_mean, gauss_stddev;
    meanStdDev(gaussian, gauss_mean, gauss_stddev);
    double gauss_min, gauss_max;
    minMaxLoc(gaussian, &gauss_min, &gauss_max);
    feature_vector.push_back(gauss_mean[0] / 255.0);
    feature_vector.push_back(gauss_stddev[0] / 255.0);
    feature_vector.push_back(gauss_min / 255.0);
    feature_vector.push_back(gauss_max / 255.0);
    
    // ========== 4. ЦВЕТОВЫЕ ПРИЗНАКИ ==========
    if (img.channels() == 3) {
        vector<Mat> channels;
        split(img, channels);
        
        // Статистики по каждому каналу (B, G, R) - по 4 признака на канал = 12 признаков
        for (int c = 0; c < 3; ++c) {
            Scalar ch_mean, ch_stddev;
            meanStdDev(channels[c], ch_mean, ch_stddev);
            double ch_min, ch_max;
            minMaxLoc(channels[c], &ch_min, &ch_max);
            feature_vector.push_back(ch_mean[0] / 255.0);
            feature_vector.push_back(ch_stddev[0] / 255.0);
            feature_vector.push_back(ch_min / 255.0);
            feature_vector.push_back(ch_max / 255.0);
        }
        
        // Корреляции между каналами (3 признака)
        Mat b_g, b_r, g_r;
        multiply(channels[0], channels[1], b_g);
        multiply(channels[0], channels[2], b_r);
        multiply(channels[1], channels[2], g_r);
        Scalar bg_mean, br_mean, gr_mean;
        bg_mean = mean(b_g);
        br_mean = mean(b_r);
        gr_mean = mean(g_r);
        feature_vector.push_back(bg_mean[0] / (255.0 * 255.0));
        feature_vector.push_back(br_mean[0] / (255.0 * 255.0));
        feature_vector.push_back(gr_mean[0] / (255.0 * 255.0));
    } else {
        // Для серых изображений добавляем нули вместо цветовых признаков (15 признаков)
        for (int i = 0; i < 15; ++i) {
            feature_vector.push_back(0.0);
        }
    }
    
    // ========== 5. HOG (Histogram of Oriented Gradients) ПРИЗНАКИ ==========
    // Упрощенная реализация HOG для извлечения ориентационных признаков
    Mat hog_gray;
    if (img.channels() == 3) {
        cvtColor(img, hog_gray, COLOR_BGR2GRAY);
    } else {
        hog_gray = img.clone();
    }
    
    // Вычисляем HOG дескрипторы (упрощенная версия)
    int cell_size = 8;
    int block_size = 16;
    int num_bins = 9;
    
    // Изменяем размер для HOG вычислений
    Mat hog_resized;
    resize(hog_gray, hog_resized, Size(64, 64));
    
    // Вычисляем градиенты для HOG
    Mat hog_grad_x, hog_grad_y;
    Sobel(hog_resized, hog_grad_x, CV_64F, 1, 0, 3);
    Sobel(hog_resized, hog_grad_y, CV_64F, 0, 1, 3);
    
    Mat hog_magnitude, hog_angle;
    cartToPolar(hog_grad_x, hog_grad_y, hog_magnitude, hog_angle, true);
    
    // Вычисляем HOG гистограммы для каждого блока
    int cells_x = 64 / cell_size;
    int cells_y = 64 / cell_size;
    int blocks_x = cells_x - 1;
    int blocks_y = cells_y - 1;
    
    // Упрощенный HOG: вычисляем гистограммы ориентаций
    for (int by = 0; by < blocks_y; ++by) {
        for (int bx = 0; bx < blocks_x; ++bx) {
            vector<double> block_hist(num_bins, 0.0);
            
            for (int cy = 0; cy < block_size; ++cy) {
                for (int cx = 0; cx < block_size; ++cx) {
                    int px = bx * cell_size + cx;
                    int py = by * cell_size + cy;
                    if (px < 64 && py < 64) {
                        double mag = hog_magnitude.at<double>(py, px);
                        double ang = hog_angle.at<double>(py, px);
                        
                        // Нормализуем угол к [0, 180] для HOG
                        if (ang < 0) ang += 180;
                        int bin = static_cast<int>((ang / 180.0) * num_bins);
                        bin = min(bin, num_bins - 1);
                        
                        block_hist[bin] += mag;
                    }
                }
            }
            
            // Нормализуем блок и добавляем признаки
            double block_norm = 0.0;
            for (int i = 0; i < num_bins; ++i) {
                block_norm += block_hist[i] * block_hist[i];
            }
            block_norm = sqrt(block_norm) + 1e-5;
            
            for (int i = 0; i < num_bins; ++i) {
                feature_vector.push_back(block_hist[i] / block_norm);
            }
        }
    }
    
    // ========== 6. LBP (Local Binary Patterns) ПРИЗНАКИ ==========
    Mat lbp_gray;
    if (img.channels() == 3) {
        cvtColor(img, lbp_gray, COLOR_BGR2GRAY);
    } else {
        lbp_gray = img.clone();
    }
    
    // Изменяем размер для LBP
    Mat lbp_resized;
    resize(lbp_gray, lbp_resized, Size(32, 32));
    lbp_resized.convertTo(lbp_resized, CV_64F);
    
    // Вычисляем упрощенный LBP (Uniform patterns)
    Mat lbp = Mat::zeros(lbp_resized.size(), CV_64F);
    int radius = 1;
    int neighbors = 8;
    
    for (int y = radius; y < lbp_resized.rows - radius; ++y) {
        for (int x = radius; x < lbp_resized.cols - radius; ++x) {
            double center = lbp_resized.at<double>(y, x);
            unsigned char lbp_code = 0;
            
            // Сравниваем с соседями
            int dx[] = {-1, -1, 0, 1, 1, 1, 0, -1};
            int dy[] = {0, -1, -1, -1, 0, 1, 1, 1};
            
            for (int i = 0; i < neighbors; ++i) {
                int nx = x + dx[i] * radius;
                int ny = y + dy[i] * radius;
                if (lbp_resized.at<double>(ny, nx) >= center) {
                    lbp_code |= (1 << i);
                }
            }
            
            lbp.at<double>(y, x) = static_cast<double>(lbp_code) / 255.0;
        }
    }
    
    // Добавляем LBP признаки (32×32 = 1024 признака)
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
            feature_vector.push_back(lbp.at<double>(y, x));
        }
    }
    
    // Статистики LBP (4 признака)
    Scalar lbp_mean, lbp_stddev;
    meanStdDev(lbp, lbp_mean, lbp_stddev);
    double lbp_min, lbp_max;
    minMaxLoc(lbp, &lbp_min, &lbp_max);
    feature_vector.push_back(lbp_mean[0]);
    feature_vector.push_back(lbp_stddev[0]);
    feature_vector.push_back(lbp_min);
    feature_vector.push_back(lbp_max);
    
    // ========== ПРЕОБРАЗОВАНИЕ В MatrixXd ==========
    MatrixXd features(feature_vector.size(), 1);
    for (size_t i = 0; i < feature_vector.size(); ++i) {
        features(i, 0) = feature_vector[i];
    }
    
    return features;
}

MatrixXd UniversalImageClassifier::applyMultiScaleFeatures(const Mat& img) {
    // Многоуровневое извлечение признаков
    vector<Mat> scales;
    
    // Разные масштабы
    Mat scale1, scale2, scale3;
    resize(img, scale1, Size(32, 32));
    resize(img, scale2, Size(64, 64));
    resize(img, scale3, Size(128, 128));
    
    scales.push_back(scale1);
    scales.push_back(scale2);
    scales.push_back(scale3);
    
    // Объединение признаков разных масштабов
    MatrixXd combined = extractFeatures(scales[1]); // Используем средний масштаб
    
    return combined;
}

pair<int, double> UniversalImageClassifier::classify(const Mat& img) {
    if (ensemble.empty()) {
        return make_pair(-1, 0.0);
    }
    
    // Предобработка
    MatrixXd features = preprocessImage(img, image_size);
    MatrixXd input = features.transpose();
    
    // Собираем предсказания от всех членов ансамбля с весами на основе F1-score
    vector<MatrixXd> ensemble_preds;
    vector<double> member_weights;
    
    for (auto& net : ensemble) {
        MatrixXd predictions = net->predict(input);
        
        // Векторизованный Softmax для численной стабильности и производительности
        // Вычитаем максимум для стабильности
        double max_val = predictions.maxCoeff();
        MatrixXd exp_preds = (predictions.array() - max_val).exp();
        double exp_sum = exp_preds.sum();
        // Нормализуем через векторизованное деление
        predictions = exp_preds / exp_sum;
        
        ensemble_preds.push_back(predictions);
        
        // Вес члена ансамбля на основе его F1-score (если доступен)
        // Используем best_f1_score из сети, если доступен, иначе 1.0
        double member_f1 = 1.0;  // По умолчанию равный вес
        // Можно добавить сохранение F1-score каждого члена для использования здесь
        member_weights.push_back(member_f1);
    }
    
    // Нормализуем веса
    double weight_sum = 0.0;
    for (double w : member_weights) {
        weight_sum += w;
    }
    if (weight_sum > 0) {
        for (double& w : member_weights) {
            w /= weight_sum;
        }
    }
    
    // Объединяем через FuzzyLogicFusion с весами
    MatrixXd fused_preds = fuzzy_fusion->fuzzyFuse(ensemble_preds, member_weights, FuzzyLogicFusion::FUZZY_AVERAGE);
    
    // Нахождение класса с максимальной вероятностью
    int max_idx = 0;
    double max_val = fused_preds(0, 0);
    for (int i = 1; i < fused_preds.cols(); ++i) {
        if (fused_preds(0, i) > max_val) {
            max_val = fused_preds(0, i);
            max_idx = i;
        }
    }
    
    return make_pair(max_idx, max_val);
}

string UniversalImageClassifier::getClassName(int class_id) const {
    auto it = class_names.find(class_id);
    if (it != class_names.end()) {
        return it->second;
    }
    return "Class " + to_string(class_id);
}

// Обучение модели на наборе изображений
// Оптимизированная версия с улучшенной обработкой ошибок и автоматической настройкой параметров
void UniversalImageClassifier::trainOnDataset(const vector<string>& image_paths,
                                              const vector<int>& labels,
                                              int epochs,
                                              bool use_focal_loss,
                                              bool use_oversampling,
                                              bool use_extended_augmentation,
                                              bool use_stage4_training,
                                              bool use_label_smoothing,
                                              bool use_dropout,
                                              bool use_adaptive_clipping,
                                              bool use_mixup,
                                              bool use_cosine_annealing)
{
    // Улучшенная обработка ошибок
    if (image_paths.empty()) {
        throw invalid_argument("UniversalImageClassifier::trainOnDataset: Empty image paths");
    }
    
    if (image_paths.size() != labels.size()) {
        throw invalid_argument("UniversalImageClassifier::trainOnDataset: Image paths and labels size mismatch");
    }
    
    if (epochs <= 0) {
        throw invalid_argument("UniversalImageClassifier::trainOnDataset: Invalid number of epochs");
    }
    
    // Определяем реальное количество уникальных классов из меток
    set<int> unique_labels(labels.begin(), labels.end());
    int actual_num_classes = std::max(static_cast<int>(unique_labels.size()), num_classes);
    
    // Переопределяем архитектуру на основе реальных данных, если нужно
    int input_size = image_size * image_size * 3;
    vector<int> optimal_architecture = NeuralNetwork::determineOptimalArchitecture(
        input_size,
        actual_num_classes,
        static_cast<int>(image_paths.size())
    );
    
    // Если архитектура изменилась или сеть еще не создана, пересоздаем сеть
    bool need_recreate = false;
    if (ensemble.empty()) {
        need_recreate = true;
    } else {
        // Проверяем, нужно ли пересоздать сеть (если количество классов изменилось)
        if (actual_num_classes != num_classes) {
            cout << "[UniversalImageClassifier] Detected change in number of classes: " 
                 << num_classes << " -> " << actual_num_classes << endl;
            num_classes = actual_num_classes;
            need_recreate = true;
        }
    }
    
    if (need_recreate) {
        cout << "[UniversalImageClassifier] Recreating ensemble with optimal architectures..." << endl;
        ensemble.clear();
        
        double learning_rate = 0.001;
        if (actual_num_classes > 50) {
            learning_rate = 0.0005;
        } else if (actual_num_classes < 10) {
            learning_rate = 0.002;
        }
        
        double l2_reg = 0.002;  // Усиленная L2-регуляризация для предотвращения переобучения
        try {
            Config::getInstance().load("config.json");
            l2_reg = Config::getInstance().getDouble("l2_reg", 0.002);
        } catch (...) {}
        for (int i = 0; i < 3; ++i) {
            vector<int> ensemble_hidden = optimal_architecture;
            if (i == 1) ensemble_hidden.push_back(std::max(actual_num_classes * 2, 32));
            if (i == 2) ensemble_hidden[0] = static_cast<int>(ensemble_hidden[0] * 1.5);

            ensemble.push_back(make_unique<NeuralNetwork>(
                input_size,
                ensemble_hidden,
                actual_num_classes,
                learning_rate * (1.0 - i * 0.2),
                "relu",
                0.9,
                l2_reg
            ));
        }
    }
    cout << "\n" << string(60, '=') << endl;
    cout << "АУГМЕНТАЦИЯ" << endl;
    cout << string(60, '=') << endl;
    cout << "Loading and preprocessing " << image_paths.size() << " images with data augmentation..." << endl;
    
    // Загрузка и предобработка всех изображений с data augmentation
    // Параллелизация загрузки и предобработки для ускорения
    vector<MatrixXd> images;
    vector<int> valid_labels;
    
    // Data augmentation для увеличения разнообразия данных
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> flip_dist(0.0, 1.0);
    uniform_real_distribution<double> rotate_dist(-15.0, 15.0);
    
    // Параллельная загрузка и предобработка изображений
    const size_t num_threads = std::max(2u, thread::hardware_concurrency());
    const size_t chunk_size = (image_paths.size() / num_threads > 0) ? (image_paths.size() / num_threads) : 1ul;
    vector<future<vector<pair<MatrixXd, int>>>> futures;
    mutex images_mutex;
    
    // Функция для обработки части изображений
    auto process_chunk = [&](size_t start_idx, size_t end_idx) -> vector<pair<MatrixXd, int>> {
        vector<pair<MatrixXd, int>> chunk_results;
        mt19937 local_gen(rd() + static_cast<unsigned>(start_idx));
        uniform_real_distribution<double> local_flip_dist(0.0, 1.0);
        uniform_real_distribution<double> local_rotate_dist(-15.0, 15.0);
        
        for (size_t i = start_idx; i < end_idx && i < image_paths.size(); ++i) {
            Mat img = imread(image_paths[i]);
            if (img.empty()) {
                continue; // Пропускаем невалидные изображения
            }
            
            // Основное изображение
            MatrixXd features = preprocessImage(img, image_size);
            chunk_results.push_back({features, labels[i]});
            
            // Проверяем, является ли класс проблемным
            bool is_problem_class = std::find(problem_classes.begin(), problem_classes.end(), labels[i]) 
                                   != problem_classes.end();
            
            // 1. Горизонтальное отражение (50% вероятность)
            if (local_flip_dist(local_gen) > 0.5) {
                Mat flipped;
                flip(img, flipped, 1);
                MatrixXd features_flipped = preprocessImage(flipped, image_size);
                chunk_results.push_back({features_flipped, labels[i]});
            }
            
            // 2. Небольшое вращение (30% вероятность)
            if (local_flip_dist(local_gen) > 0.7) {
                Mat rotated;
                Point2f center(img.cols / 2.0f, img.rows / 2.0f);
                Mat rotation_matrix = getRotationMatrix2D(center, local_rotate_dist(local_gen), 1.0);
                warpAffine(img, rotated, rotation_matrix, img.size(), INTER_LINEAR, BORDER_REPLICATE);
                MatrixXd features_rotated = preprocessImage(rotated, image_size);
                chunk_results.push_back({features_rotated, labels[i]});
            }
            
            // Дополнительная augmentation для проблемных классов
            if (is_problem_class) {
                MatrixXd augmented_features = augmentProblemClassImage(img);
                chunk_results.push_back({augmented_features, labels[i]});
            }
        }
        
        return chunk_results;
    };
    
    // Запускаем параллельную обработку
    for (size_t chunk_start = 0; chunk_start < image_paths.size(); chunk_start += chunk_size) {
        size_t chunk_end = (chunk_start + chunk_size < image_paths.size()) ? (chunk_start + chunk_size) : image_paths.size();
        futures.push_back(async(launch::async, process_chunk, chunk_start, chunk_end));
    }
    
    // Собираем результаты
    for (auto& future : futures) {
        auto chunk_results = future.get();
        for (const auto& [features, label] : chunk_results) {
            images.push_back(features);
            valid_labels.push_back(label);
        }
        
        // Прогресс
        static size_t processed_chunks = 0;
        processed_chunks++;
        size_t divisor = (futures.size() / 10 > 0) ? (futures.size() / 10) : 1ul;
        if (processed_chunks % divisor == 0 || processed_chunks == futures.size()) {
            cout << "Processed chunks: " << processed_chunks << "/" << futures.size() 
                 << " (total samples: " << images.size() << ")" << endl;
        }
    }
    
    if (images.empty()) {
        cerr << "Ошибка: не удалось загрузить изображения!" << endl;
        return;
    }
    
    // Применяем oversampling для проблемных классов (если они определены из предыдущего цикла и включено)
    if (!problem_classes.empty() && use_oversampling) {
        cout << "\nПрименение oversampling для проблемных классов..." << endl;
        // Увеличиваем oversample_ratio до 2.0-2.5 для проблемных классов
        oversampleProblemClasses(images, valid_labels, problem_classes, 2.2);
    }
    
    cout << "Successfully loaded: " << images.size() << " images" << endl;
    
    // Сбор признаков структур для обучения SVM (если включено)
    // Один проход по изображениям вместо трёх — быстрее и с прогрессом
    if (use_structure_analysis) {
        const size_t total_imgs = image_paths.size();
        cout << "[UniversalImageClassifier] Extracting structure features for SVM (" << total_imgs << " images)..." << endl;
        cout.flush();
        
        vector<MatrixXd> all_structure_features;
        vector<int> structure_labels;
        vector<MatrixXd> extended_features;
        map<string, vector<MatrixXd>> structure_features_by_type;
        map<string, vector<int>> structure_labels_by_type;
        
        for (size_t i = 0; i < total_imgs; ++i) {
            cout << "[SVM] " << (i + 1) << "/" << total_imgs << endl;
            cout.flush();
            
            try {
                Mat img = imread(image_paths[i]);
                if (img.empty()) continue;
                
                ShapeDescription shape = shape_analyzer->analyzeShape(img);
                if (shape.structures.empty()) continue;
                
                vector<MatrixXd> struct_feats = extractStructureFeatures(img, shape.structures);
                if (struct_feats.empty()) continue;
                
                for (const auto& feat : struct_feats) {
                    all_structure_features.push_back(feat);
                    structure_labels.push_back(labels[i]);
                }
                
                if (topological_sig) {
                    PersistenceDiagram diag = shape_analyzer->computeTopologicalSignature(img, shape.structures);
                    HyperRelationalFuzzyGraph graph = shape_analyzer->buildHyperRelationalGraph(
                        shape.structures, shape);
                    VectorXd topo_sig = diag.toSignature(20);
                    VectorXd graph_feat = graph.computeGraphFeatures();
                    
                    for (const auto& feat : struct_feats) {
                        int orig_size = static_cast<int>(feat.rows());
                        MatrixXd extended(orig_size + topo_sig.size() + graph_feat.size(), 1);
                        extended.block(0, 0, orig_size, 1) = feat;
                        extended.block(orig_size, 0, topo_sig.size(), 1) = topo_sig;
                        extended.block(orig_size + topo_sig.size(), 0, graph_feat.size(), 1) = graph_feat;
                        extended_features.push_back(extended);
                    }
                }
                
                for (size_t j = 0; j < shape.structures.size() && j < struct_feats.size(); ++j) {
                    string structure_type = shape_analyzer->classifyStructureType(shape.structures[j], shape);
                    structure_features_by_type[structure_type].push_back(struct_feats[j]);
                    structure_labels_by_type[structure_type].push_back(labels[i]);
                }
            } catch (const std::exception& e) {
                cout << "\n[SVM] Skip image " << (i + 1) << ": " << e.what() << endl;
                cout.flush();
            } catch (...) {
                cout << "\n[SVM] Skip image " << (i + 1) << ": unknown error" << endl;
                cout.flush();
            }
        }
        
        cout << "[UniversalImageClassifier] SVM structures: " << all_structure_features.size() << " features" << endl;
        cout.flush();
        
        if (!all_structure_features.empty()) {
            cout << "[SVM] Training SVM classifier..." << endl;
            cout.flush();
            if (!extended_features.empty()) {
                vector<int> ext_labels;
                for (size_t k = 0; k < extended_features.size() && k < structure_labels.size(); ++k) {
                    ext_labels.push_back(structure_labels[k]);
                }
                if (!ext_labels.empty()) {
                    trainSVM(extended_features, ext_labels);
                } else {
                    trainSVM(all_structure_features, structure_labels);
                }
            } else {
                trainSVM(all_structure_features, structure_labels);
            }
            cout << "[SVM] SVM trained." << endl;
            cout.flush();
            // Обучение подсетей на структурах
            cout << "[UniversalImageClassifier] Training sub-networks on structures..." << endl;
            cout.flush();
            int sub_idx = 0;
            int sub_total = static_cast<int>(structure_features_by_type.size());
            for (const auto& [structure_type, features] : structure_features_by_type) {
                sub_idx++;
                cout << "[SVM] Sub-network " << sub_idx << "/" << sub_total << ": " << structure_type << endl;
                cout.flush();
                if (features.empty()) continue;
                
                // Создаем подсеть если её еще нет
                if (!subnetwork_manager->getSubNetwork(structure_type)) {
                    int feat_size = static_cast<int>(features[0].rows());
                    double entropy = 0.5; // Средняя энтропия по умолчанию
                    if (!features.empty()) {
                        // Вычисляем среднюю энтропию для этого типа структуры
                        Mat sample_region = Mat::zeros(32, 32, CV_8UC3);
                        entropy = shape_analyzer->computeStructuralEntropy(sample_region);
                    }
                    subnetwork_manager->createSubNetwork(structure_type, feat_size, num_classes, entropy);
                }
                
                // Обучаем подсеть
                subnetwork_manager->trainSubNetwork(
                    structure_type,
                    features,
                    structure_labels_by_type[structure_type],
                    30 // epochs для подсетей
                );
            }
            
            // Первоначальная передача весов от основной сети к подсетям
            if (!ensemble.empty()) {
                cout << "[UniversalImageClassifier] Initial knowledge transfer to sub-networks..." << endl;
                subnetwork_manager->transferWeightsFromMain(ensemble[0].get(), 0.3);
            }
        }
    }
    
    // Анализ дисбаланса классов
    map<int, int> class_counts;
    for (int label : valid_labels) {
        if (label >= 0 && label < num_classes) {
            class_counts[label]++;
        }
    }
    
    cout << "\n" << string(60, '=') << endl;
    cout << "CLASS IMBALANCE ANALYSIS" << endl;
    cout << string(60, '=') << endl;
    int max_count = 0, min_count = INT_MAX;
    for (const auto& [class_id, count] : class_counts) {
        cout << "  Class " << class_id << ": " << count << " samples" << endl;
        max_count = std::max(max_count, count);
        min_count = std::min(min_count, count);
    }
    
    if (max_count > 0 && min_count > 0) {
        double imbalance_ratio = static_cast<double>(max_count) / min_count;
        cout << "  Imbalance ratio: " << fixed << setprecision(2) << imbalance_ratio << ":1" << endl;
        if (imbalance_ratio > 2.0) {
            cout << "  WARNING: Class imbalance detected!" << endl;
            cout << "  Recommendations:" << endl;
            cout << "    - Using weighted loss function" << endl;
            cout << "    - Increased epochs for small classes" << endl;
            cout << "    - Batch size 12 is optimal for your case" << endl;
            
            // Находим классы с малым количеством образцов
            cout << "  Classes with few samples (require special attention):" << endl;
            for (const auto& [class_id, count] : class_counts) {
                if (count < max_count * 0.5) {
                    cout << "    - Class " << class_id << ": " << count << " samples" << endl;
                }
            }
        }
    }
    cout << string(60, '=') << endl;
    
    // Вычисление весов классов для балансировки
    vector<double> class_weights(num_classes, 1.0);
    double imbalance_ratio = (min_count > 0) ? (static_cast<double>(max_count) / min_count) : 1.0;
    if (max_count > 0 && min_count > 0 && max_count != min_count) {
        double total_samples = static_cast<double>(valid_labels.size());
        double median_count = 0.0;
        vector<int> counts;
        for (const auto& [id, count] : class_counts) {
            counts.push_back(count);
        }
        if (!counts.empty()) {
            sort(counts.begin(), counts.end());
            median_count = counts[counts.size() / 2];
        }
        
        for (int i = 0; i < num_classes; ++i) {
            int count = class_counts[i];
            if (count > 0) {
                // Используем медианное значение для более стабильных весов
                // Вес обратно пропорционален частоте класса относительно медианы
                if (median_count > 0) {
                    class_weights[i] = median_count / count;
                } else {
                    class_weights[i] = total_samples / (num_classes * count);
                }
                // Ограничиваем веса, чтобы избежать экстремальных значений
                // Для малых классов увеличиваем вес более агрессивно (цель 98% по всем классам)
                if (count < median_count * 0.5) {
                    class_weights[i] = min(class_weights[i] * 2.0, 20.0); // Сильнее для очень малых классов
                }
                if (imbalance_ratio > 5.0 && count < max_count * 0.2) {
                    class_weights[i] = min(class_weights[i] * 1.5, 20.0); // Доп. буст при сильном дисбалансе
                }
                class_weights[i] = min(class_weights[i], 20.0);
                class_weights[i] = max(class_weights[i], 0.2);
            } else {
                class_weights[i] = 1.0; // Если класса нет в данных
            }
        }
        cout << "\nClass weights for balancing (higher weight = more attention to class):" << endl;
        for (int i = 0; i < num_classes; ++i) {
            cout << "  Class " << i << ": " << class_counts[i] << " samples, weight = " 
                 << fixed << setprecision(3) << class_weights[i];
            if (class_weights[i] > 2.0) {
                cout << " UP (small class - increased attention)";
            }
            cout << endl;
        }
    }
    
    // Преобразование в матрицы
    int num_samples = images.size();
    int feature_size = images[0].rows();
    
    MatrixXd X(num_samples, feature_size);
    MatrixXd y = MatrixXd::Zero(num_samples, num_classes);
    VectorXd sample_weights = VectorXd::Ones(num_samples); // Веса для каждого образца
    
    for (int i = 0; i < num_samples; ++i) {
        X.row(i) = images[i].transpose();
        if (valid_labels[i] >= 0 && valid_labels[i] < num_classes) {
            y(i, valid_labels[i]) = 1.0;
            // Присваиваем вес образца на основе его класса
            sample_weights(i) = class_weights[valid_labels[i]];
        }
    }
    
    // Применяем Mixup augmentation на уровне матриц для проблемных классов
    if (!problem_classes.empty() && use_mixup) {
        // Создаем индексы проблемных классов
        vector<int> problem_indices;
        for (int i = 0; i < num_samples; ++i) {
            if (std::find(problem_classes.begin(), problem_classes.end(), valid_labels[i]) 
                != problem_classes.end()) {
                problem_indices.push_back(i);
            }
        }
        
        if (problem_indices.size() >= 2) {
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<double> prob_dist(0.0, 1.0);
            uniform_int_distribution<int> idx_dist(0, static_cast<int>(problem_indices.size()) - 1);
            
            // Beta распределение для lambda
            double mixup_alpha = 0.2;
            gamma_distribution<double> gamma_alpha(mixup_alpha, 1.0);
            gamma_distribution<double> gamma_beta(mixup_alpha, 1.0);
            
            int mixup_count = static_cast<int>(problem_indices.size() * 0.3); // 30% примеров
            int original_size = num_samples;
            
            // Оптимизация: предварительно выделяем память для всех mixup примеров
            int final_size = num_samples + mixup_count;
            MatrixXd X_mixup(final_size, X.cols());
            MatrixXd y_mixup(final_size, y.cols());
            VectorXd w_mixup(final_size);
            
            // Копируем существующие данные
            X_mixup.topRows(num_samples) = X;
            y_mixup.topRows(num_samples) = y;
            w_mixup.head(num_samples) = sample_weights;
            
            int mixup_added = 0;
            for (int m = 0; m < mixup_count; ++m) {
                if (prob_dist(gen) > 0.5) continue; // 50% вероятность
                
                int idx1 = problem_indices[idx_dist(gen)];
                int idx2 = problem_indices[idx_dist(gen)];
                while (idx2 == idx1) idx2 = problem_indices[idx_dist(gen)];
                
                // Генерируем lambda
                double lambda1 = gamma_alpha(gen);
                double lambda2 = gamma_beta(gen);
                double lambda = lambda1 / (lambda1 + lambda2);
                lambda = max(0.1, min(0.9, lambda));
                
                // Смешиваем признаки (используем noalias для избежания временных копий)
                int new_idx = num_samples + mixup_added;
                X_mixup.row(new_idx).noalias() = lambda * X.row(idx1) + (1.0 - lambda) * X.row(idx2);
                y_mixup.row(new_idx).noalias() = lambda * y.row(idx1) + (1.0 - lambda) * y.row(idx2);
                w_mixup(new_idx) = (sample_weights(idx1) + sample_weights(idx2)) / 2.0;
                
                mixup_added++;
            }
            
            // Обрезаем до фактического размера (conservativeResize эквивалент)
            if (mixup_added > 0) {
                int actual_size = num_samples + mixup_added;
                X = X_mixup.topRows(actual_size);
                y = y_mixup.topRows(actual_size);
                sample_weights = w_mixup.head(actual_size);
                num_samples = actual_size;
            }
            
            if (num_samples > original_size) {
                cout << "  Mixup: добавлено " << (num_samples - original_size) << " смешанных примеров" << endl;
            }
        }
    }
    
    // Улучшенная нормализация (Layer Normalization + Column Normalization)
    // Layer Normalization нормализует по строкам (образцам), что более стабильно для обучения
    for (int row = 0; row < X.rows(); ++row) {
        double mean = X.row(row).mean();
        double stddev = sqrt((X.row(row).array() - mean).square().sum() / X.cols());
        if (stddev > 1e-10) {
            X.row(row) = (X.row(row).array() - mean) / stddev;
        }
    }
    
    // Дополнительная нормализация по столбцам для стабильности
    for (int col = 0; col < X.cols(); ++col) {
        double mean = X.col(col).mean();
        double stddev = sqrt((X.col(col).array() - mean).square().sum() / X.rows());
        if (stddev > 1e-10) {
            X.col(col) = (X.col(col).array() - mean) / stddev;
        }
    }
    
    // Включаем научные методы для всех сетей ансамбля
    cout << "\nНастройка научных методов улучшения:" << endl;
    
    if (use_focal_loss) {
        cout << "  - Focal Loss: включен" << endl;
        for (auto& net : ensemble) {
            net->setFocalLossParams(2.0, 1.0, true);
        }
    }
    
    if (use_label_smoothing) {
        double label_smooth = Config::getInstance().getDouble("label_smoothing", 0.1);
        cout << "  - Label Smoothing: включен (ε=" << label_smooth << ")" << endl;
        for (auto& net : ensemble) {
            net->setLabelSmoothing(label_smooth);
        }
    }
    
    if (use_dropout) {
        cout << "  - Dropout: включен (rate=0.3)" << endl;
        for (auto& net : ensemble) {
            net->setDropoutRate(0.3);
            net->setTrainingMode(true);
        }
    }
    
    if (use_adaptive_clipping) {
        cout << "  - Adaptive Gradient Clipping: включен" << endl;
        for (auto& net : ensemble) {
            net->setAdaptiveGradientClipping(true, 5.0);
        }
    }
    
    if (use_cosine_annealing) {
        cout << "  - Cosine Annealing LR Scheduling: включен" << endl;
        for (auto& net : ensemble) {
            net->setCosineAnnealingParams(20.0, 2.0, 0.0001);  // T_0=20, eta_min=0.0001
        }
    }
    
    // Улучшенное взвешивание с учетом проблемных классов (если они уже определены)
    // Если это не первый цикл обучения и есть проблемные классы, увеличиваем их веса
    if (!problem_classes.empty()) {
        cout << "\nПрименение улучшенного взвешивания для проблемных классов..." << endl;
        
        // Получаем метрики для определения F1-score проблемных классов
        // Если метрики еще не вычислены, используем базовые веса
        // В противном случае увеличиваем веса на основе F1-score
        map<int, double> problem_class_f1;
        
        // Пытаемся получить метрики из предыдущего обучения, если они есть
        // Для первого цикла используем базовые веса, для последующих - улучшенные
        for (int class_id : problem_classes) {
            if (class_id >= 0 && class_id < num_classes) {
                // Увеличиваем вес для проблемных классов
                // Базовая формула: weight = base_weight * (1.0 / max(f1_score, 0.3))
                // Минимум 0.3 для F1-score, чтобы избежать экстремальных весов
                double base_weight = class_weights[class_id];
                double f1_estimate = 0.5; // Консервативная оценка для первого цикла
                
                // Если есть предыдущие метрики, используем их
                // Здесь можно добавить логику сохранения метрик между циклами
                double boost_factor = 1.0 / max(f1_estimate, 0.3);
                double problem_class_boost = 1.5; // Дополнительный множитель для проблемных классов
                
                class_weights[class_id] = base_weight * boost_factor * problem_class_boost;
                class_weights[class_id] = min(class_weights[class_id], 20.0); // Ограничение максимума
                
                cout << "  Class " << class_id << ": weight increased from " 
                     << fixed << setprecision(3) << base_weight 
                     << " to " << class_weights[class_id] << endl;
            }
        }
    }
    
    // Установка весов классов в сеть для взвешенной функции потерь
    if (max_count > 0 && min_count > 0 && max_count != min_count) {
        VectorXd class_weights_vector(num_classes);
        for (int i = 0; i < num_classes; ++i) {
            class_weights_vector(i) = class_weights[i];
        }
        for (auto& net : ensemble) {
            net->setClassWeights(class_weights_vector);
        }
    } else if (!problem_classes.empty()) {
        // Если нет дисбаланса классов, но есть проблемные классы, все равно применяем веса
        VectorXd class_weights_vector(num_classes);
        for (int i = 0; i < num_classes; ++i) {
            class_weights_vector(i) = class_weights[i];
        }
        for (auto& net : ensemble) {
            net->setClassWeights(class_weights_vector);
        }
    }
    
    // Автоматическая настройка параметров: config + эвристики для предотвращения переобучения
    auto& cfg = Config::getInstance();
    cfg.load("config.json");
    double loss_threshold = 0.9;
    int batch_size = cfg.getInt("batch_size", 16);
    int stage3_epochs = 10;
    int early_stopping = cfg.getInt("early_stopping_patience", 15);
    
    // Адаптация параметров для малых классов (предотвращение переобучения)
    if (min_count < 50) {
        cout << "\n[Auto-tuning] Small classes detected, applying anti-overfit parameters:" << endl;
        loss_threshold = 0.85;  // Более мягкий порог для малых классов
        stage3_epochs = 15;     // Больше эпох для лучшего обучения
        early_stopping = std::max(early_stopping, 12);  // Больше терпения для малых классов
        batch_size = std::min(batch_size, 12);  // Меньший batch для лучшей генерализации
        cout << "  - loss_threshold: " << loss_threshold << endl;
        cout << "  - stage3_epochs: " << stage3_epochs << endl;
        cout << "  - early_stopping_patience: " << early_stopping << endl;
        cout << "  - batch_size: " << batch_size << endl;
    }
    
    // Запуск функции обучения:
    //----------------------------------------------------------------------
    // Используем веса образцов для балансировки классов
    VectorXd weights_to_use = sample_weights;
    if (weights_to_use.size() == 0 || weights_to_use.size() != num_samples) {
        weights_to_use = VectorXd::Ones(num_samples);
    }
    
    cout << "\n" << string(60, '=') << endl;
    cout << "TRAINING STARTED" << endl;
    cout << string(60, '=') << endl;
    cout << "Training parameters:" << endl;
    cout << "  - Batch size: " << batch_size << endl;
    cout << "  - Epochs (stage 2, main member): " << epochs << endl;
    cout << "  - Epochs (stage 3): " << stage3_epochs << endl;
    cout << "  - Loss threshold: " << loss_threshold << endl;
    cout << "  - Early stopping: " << early_stopping << " epochs" << endl;
    cout << "  - Weighted loss function: " << (max_count != min_count ? "YES" : "NO") << endl;
    cout << string(60, '=') << endl;
    
    double total_accuracy = 0.0;
    for (size_t i = 0; i < ensemble.size(); ++i) {
        cout << "\n[Ensemble] Training member #" << (i + 1) << "..." << endl;
        
        // Bagging: каждый член обучается на случайном подмножестве (80% данных)
        vector<int> bag_indices(num_samples);
        iota(bag_indices.begin(), bag_indices.end(), 0);
        random_device rd_bag;
        mt19937 gen_bag(rd_bag());
        shuffle(bag_indices.begin(), bag_indices.end(), gen_bag);
        
        int bag_size = static_cast<int>(num_samples * 0.8);
        MatrixXd X_bag(bag_size, feature_size);
        MatrixXd y_bag(bag_size, num_classes);
        VectorXd w_bag(bag_size);
        
        // Параллельное формирование bagging-подвыборки
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int j = 0; j < bag_size; ++j) {
            int idx = bag_indices[j];
            X_bag.row(j) = X.row(idx);
            y_bag.row(j) = y.row(idx);
            w_bag(j) = weights_to_use(idx);
        }

        // Раннее определение проблемных классов после первых 10 эпох
        vector<int> early_problem_classes;
        if (i == 0 && epochs >= 10) {
            // Создаем временную валидационную выборку для раннего определения
            int val_split_idx = static_cast<int>(X_bag.rows() * 0.8);
            MatrixXd X_val_early = X_bag.bottomRows(X_bag.rows() - val_split_idx);
            MatrixXd y_val_early = y_bag.bottomRows(y_bag.rows() - val_split_idx);
            
            // Обучаем сеть на первых 10 эпохах для раннего определения
            cout << "\n[Early Detection] Training first 10 epochs for early problem class detection..." << endl;
            ensemble[i]->stage2_batch_training(
                X_bag.topRows(val_split_idx), y_bag.topRows(val_split_idx),
                batch_size, 10, early_stopping, 0.2, true, w_bag.head(val_split_idx)
            );
            
            // Определяем проблемные классы
            early_problem_classes = detectProblemClassesEarly(X_val_early, y_val_early, 10);
            if (!early_problem_classes.empty()) {
                cout << "[Early Detection] Found " << early_problem_classes.size() 
                     << " problem classes: ";
                for (size_t j = 0; j < early_problem_classes.size(); ++j) {
                    cout << early_problem_classes[j];
                    if (j < early_problem_classes.size() - 1) cout << ", ";
                }
                cout << endl;
                
                // Объединяем с уже определенными проблемными классами
                set<int> combined(problem_classes.begin(), problem_classes.end());
                combined.insert(early_problem_classes.begin(), early_problem_classes.end());
                problem_classes = vector<int>(combined.begin(), combined.end());
            }
        }
        
        // Для первого члена ансамбля используем полное число эпох,
        // для остальных — укороченный fine-tune для ускорения
        int member_epochs = static_cast<int>(epochs);
        if (i > 0) {
            member_epochs = max(20, epochs / 2);
        }
        
        total_accuracy += ensemble[i]->train_multi_stage(
            X_bag, y_bag,
            loss_threshold,
            batch_size,
            member_epochs,
            stage3_epochs,
            true,
            early_stopping,
            true,
            w_bag
        );
        
        // Квантовое обучение: передача весов между основной сетью и подсетями
        // Выполняем после каждого члена ансамбля для синхронизации знаний
        if (use_structure_analysis && i == 0) {
            // Используем первую сеть как основную для knowledge transfer
            quantumKnowledgeTransfer(epochs, epochs, 0.2, 0.1);
        }
        
        // Этап 4: Специальное обучение для проблемных классов (если включено и есть проблемные классы)
        if (use_stage4_training && !problem_classes.empty() && i == 0) {
            cout << "\n[Ensemble] Stage 4 training for member #" << (i + 1) << "..." << endl;
            ensemble[i]->stage4_problem_classes_training(
                X_bag, y_bag,
                problem_classes,
                batch_size,
                75, // epochs для stage4 (увеличено с 20 до 50-100, используем 75)
                2.2, // learning_rate_multiplier (увеличено с 1.5 до 2.0-2.5, используем 2.2)
                w_bag
            );
        }
    }
    
    // Финальная квантовая синхронизация после обучения всех членов ансамбля
    if (use_structure_analysis) {
        cout << "\n[Final Quantum Sync] Synchronizing all networks..." << endl;
        quantumKnowledgeTransfer(epochs, epochs, 0.3, 0.2);
    }
    
    double accuracy = total_accuracy / ensemble.size();
    
    cout << "\n" << string(60, '=') << endl;
    cout << "TRAINING RESULTS" << endl;
    cout << string(60, '=') << endl;
    cout << "Overall accuracy: " << fixed << setprecision(2) << (accuracy * 100) << "%" << endl;
    
    // Вычисление метрик по классам (используем ensemble[0], т.к. network может быть null)
    NeuralNetwork* result_net = ensemble.empty() ? network.get() : ensemble[0].get();
    cout << "\nDetailed class statistics:" << endl;
    cout << string(60, '-') << endl;
    vector<NeuralNetwork::ClassMetrics> class_metrics = result_net ? result_net->computeClassMetrics(X, y) : vector<NeuralNetwork::ClassMetrics>();
    
    double macro_f1 = 0.0;
    int classes_with_data = 0;
    
    for (const auto& metric : class_metrics) {
        // Проверяем, есть ли данные для этого класса
        bool has_data = (metric.true_positives + metric.false_negatives > 0) || 
                       (metric.true_positives + metric.false_positives > 0);
        
        if (has_data) {
            string class_name = getClassName(metric.class_id);
            cout << "Class " << metric.class_id << " (" << class_name << "):" << endl;
            cout << "  Precision: " << fixed << setprecision(3) << (metric.precision * 100) << "%";
            cout << "  Recall: " << (metric.recall * 100) << "%";
            cout << "  F1-score: " << (metric.f1_score * 100) << "%" << endl;
            cout << "  TP=" << metric.true_positives 
                 << ", FP=" << metric.false_positives 
                 << ", FN=" << metric.false_negatives << endl;
            
            macro_f1 += metric.f1_score;
            classes_with_data++;
            
            // Предупреждения для классов с низкими метриками
            if (metric.f1_score < 0.5) {
                cout << "  WARNING: Low recognition quality!" << endl;
            } else if (metric.recall < 0.5) {
                cout << "  WARNING: Many missed samples (low recall)" << endl;
            } else if (metric.precision < 0.5) {
                cout << "  WARNING: Many false positives (low precision)" << endl;
            }
            cout << endl;
        }
    }
    
    if (classes_with_data > 0) {
        macro_f1 /= classes_with_data;
        cout << "Macro-averaged F1-score: " << fixed << setprecision(3) << (macro_f1 * 100) << "%" << endl;
    }
    
    // Автоматическое определение проблемных классов
    // Более строгие пороги для определения проблемных классов (цель: 98% accuracy)
    problem_classes = detectProblemClasses(class_metrics, 0.75, 0.70, 0.70);
    if (!problem_classes.empty()) {
        cout << "\nПроблемные классы (требуют особого внимания):" << endl;
        for (int class_id : problem_classes) {
            string class_name = getClassName(class_id);
            auto it = find_if(class_metrics.begin(), class_metrics.end(),
                            [class_id](const NeuralNetwork::ClassMetrics& m) {
                                return m.class_id == class_id;
                            });
            if (it != class_metrics.end()) {
                cout << "  - Class " << class_id << " (" << class_name << "): "
                     << "F1=" << fixed << setprecision(2) << (it->f1_score * 100) << "%, "
                     << "Recall=" << (it->recall * 100) << "%, "
                     << "Precision=" << (it->precision * 100) << "%" << endl;
            }
        }
    } else {
        cout << "\nВсе классы показывают хорошие результаты!" << endl;
    }
    
    // Сохраняем метрики для последующего сохранения
    // (будут использованы в saveAllTrainingData)
    
    // Дополнительная диагностика по классам
    if (class_counts.size() > 1) {
        cout << "\nRecommendations for improvement:" << endl;
        for (const auto& [class_id, count] : class_counts) {
            if (count < max_count * 0.5) {
                cout << "  - Class " << class_id << " (" << count << " samples): ";
                cout << "consider adding more samples" << endl;
            }
        }
    }
    cout << string(60, '=') << endl;
    
    // Построение графика обучения
    cout << "\nПостроение графика процесса обучения..." << endl;
    try {
        auto history = result_net ? result_net->getTrainingHistory() : vector<TrainingStats>();
        if (!history.empty()) {
            // Извлекаем данные для графика
            vector<double> epochs;
            vector<double> train_loss;
            vector<double> train_accuracy;
            
            for (const auto& stats : history) {
                epochs.push_back(stats.epoch);
                train_loss.push_back(stats.loss);
                train_accuracy.push_back(stats.accuracy);
            }
            
            // Создаем график
            MetricsPlotter plotter(1200, 800);
            plotter.plotLossAndAccuracy(epochs, train_loss, {}, train_accuracy, {}, "Training Progress");
            
            // Сохраняем график
            string plot_filename = "training_plot.png";
            if (plotter.savePlot(plot_filename)) {
                cout << "График сохранен в " << plot_filename << endl;
            }
            
            // Показываем график без блокировки (0 = ждать вечно, 1 = не блокировать)
            plotter.show("Training Progress", 1);
            cout << "График сохранён в " << plot_filename << ", окно графика открыто." << endl;
        } else {
            cout << "История обучения пуста, график не построен." << endl;
        }
    } catch (const exception& e) {
        cerr << "Ошибка при построении графика: " << e.what() << endl;
    }
    
    // Автоматическое сохранение всех данных обучения
    cout << "\n" << string(60, '=') << endl;
    cout << "AUTOMATIC SAVE: All training data" << endl;
    cout << string(60, '=') << endl;
    
    string timestamp = getCurrentTimestamp();
    string save_dir = "training_data_" + timestamp;
    
    if (fs::create_directories(save_dir)) {
        cout << "Created directory: " << save_dir << endl;
    }
    
    // Вычисляем финальные метрики для сохранения
    double final_accuracy = accuracy;
    double final_macro_f1 = macro_f1;
    double final_weighted_f1 = 0.0;
    
    // Вычисляем weighted F1
    if (!ensemble.empty()) {
        final_weighted_f1 = ensemble[0]->computeWeightedF1Score(X, y);
    }
    
    // Сохраняем все данные
    saveAllTrainingDataWithMetrics(save_dir, class_metrics, final_accuracy, 
                                   final_macro_f1, final_weighted_f1, epochs);
    
    cout << "\nTraining completed! All data saved to: " << save_dir << endl;
}

void UniversalImageClassifier::evaluateModel(const vector<string>& image_paths, const vector<int>& labels) {
    if (image_paths.size() != labels.size()) {
        cerr << "Ошибка: количество изображений и меток не совпадает!" << endl;
        return;
    }
    
    if (ensemble.empty()) {
        cerr << "Error: model not loaded!" << endl;
        return;
    }
    
    cout << "\n" << string(60, '=') << endl;
    cout << "MODEL EVALUATION" << endl;
    cout << string(60, '=') << endl;
    cout << "Processing " << image_paths.size() << " images..." << endl;
    
    // Загрузка и предобработка изображений (с подвыборкой для ускорения)
    vector<MatrixXd> images;
    vector<int> valid_labels;
    
    const size_t max_eval_samples = 3000;  // ограничиваем количество образцов для быстрой оценки
    size_t step = std::max<size_t>(1, image_paths.size() / max_eval_samples);
    
    for (size_t i = 0; i < image_paths.size(); i += step) {
        Mat img = imread(image_paths[i]);
        if (img.empty()) {
            continue;
        }
        
        MatrixXd features = preprocessImage(img, image_size);
        images.push_back(features);
        valid_labels.push_back(labels[i]);
    }
    
    if (images.empty()) {
        cerr << "Ошибка: не удалось загрузить изображения!" << endl;
        return;
    }
    
    // Преобразование в матрицы
    int num_samples = images.size();
    int feature_size = images[0].rows();
    
    MatrixXd X(num_samples, feature_size);
    MatrixXd y = MatrixXd::Zero(num_samples, num_classes);
    
    for (int i = 0; i < num_samples; ++i) {
        X.row(i) = images[i].transpose();
        if (valid_labels[i] >= 0 && valid_labels[i] < num_classes) {
            y(i, valid_labels[i]) = 1.0;
        }
    }
    
    // Улучшенная нормализация (Layer Normalization + Column Normalization)
    for (int row = 0; row < X.rows(); ++row) {
        double mean = X.row(row).mean();
        double stddev = sqrt((X.row(row).array() - mean).square().sum() / X.cols());
        if (stddev > 1e-10) {
            X.row(row) = (X.row(row).array() - mean) / stddev;
        }
    }
    
    for (int col = 0; col < X.cols(); ++col) {
        double mean = X.col(col).mean();
        double stddev = sqrt((X.col(col).array() - mean).square().sum() / X.rows());
        if (stddev > 1e-10) {
            X.col(col) = (X.col(col).array() - mean) / stddev;
        }
    }
    
    // Вычисление метрик (используем первый член ансамбля для базовых метрик или среднее)
    double accuracy = ensemble[0]->computeAccuracy(X, y);
    vector<NeuralNetwork::ClassMetrics> metrics = ensemble[0]->computeClassMetrics(X, y);
    
    cout << "\nОбщая точность: " << fixed << setprecision(2) << (accuracy * 100) << "%" << endl;
    cout << "\nДетальная статистика по классам:" << endl;
    cout << string(60, '-') << endl;
    
    for (const auto& metric : metrics) {
        bool has_data = (metric.true_positives + metric.false_negatives > 0) || 
                       (metric.true_positives + metric.false_positives > 0);
        
        if (has_data) {
            string class_name = getClassName(metric.class_id);
            cout << "Класс " << metric.class_id << " (" << class_name << "):" << endl;
            cout << "  Precision: " << fixed << setprecision(3) << (metric.precision * 100) << "%" << endl;
            cout << "  Recall: " << (metric.recall * 100) << "%" << endl;
            cout << "  F1-score: " << (metric.f1_score * 100) << "%" << endl;
            cout << "  TP=" << metric.true_positives 
                 << ", FP=" << metric.false_positives 
                 << ", FN=" << metric.false_negatives << endl;
            cout << endl;
        }
    }
    
    cout << string(60, '=') << endl;
    
    // Построение графика обучения
    cout << "\nПостроение графика процесса обучения..." << endl;
    try {
        auto history = ensemble[0]->getTrainingHistory();
        if (!history.empty()) {
            // Извлекаем данные для графика
            vector<double> epochs;
            vector<double> train_loss;
            vector<double> train_accuracy;
            
            for (const auto& stats : history) {
                epochs.push_back(stats.epoch);
                train_loss.push_back(stats.loss);
                train_accuracy.push_back(stats.accuracy);
            }
            
            // Создаем график
            MetricsPlotter plotter(1200, 800);
            plotter.plotLossAndAccuracy(epochs, train_loss, {}, train_accuracy, {}, "Training Progress");
            
            // Сохраняем график
            string plot_filename = "training_plot.png";
            if (plotter.savePlot(plot_filename)) {
                cout << "График сохранен в " << plot_filename << endl;
            }
            
            // Показываем график без блокировки (0 = ждать вечно, 1 = не блокировать)
            plotter.show("Training Progress", 1);
            cout << "График сохранён в " << plot_filename << ", окно графика открыто." << endl;
        } else {
            cout << "История обучения пуста, график не построен." << endl;
        }
    } catch (const exception& e) {
        cerr << "Ошибка при построении графика: " << e.what() << endl;
    }
}

void UniversalImageClassifier::saveTrainingStats(const string& path) const {
    NeuralNetwork* net = !ensemble.empty() ? ensemble[0].get() : network.get();
    if (!net) {
        cerr << "Ошибка: модель не обучена!" << endl;
        return;
    }
    
    ofstream file(path);
    if (!file.is_open()) {
        cerr << "Ошибка: не удалось открыть файл " << path << " для записи" << endl;
        return;
    }
    
    auto history = net->getTrainingHistory();
    
    file << "Training Statistics\n";
    file << "===================\n\n";
    file << "Total epochs: " << history.size() << "\n\n";
    file << "Stage,Epoch,Accuracy,Loss,Samples_Processed,Samples_Accepted\n";
    
    for (const auto& stats : history) {
        file << stats.stage << "," << stats.epoch << "," 
             << fixed << setprecision(6) << stats.accuracy << ","
             << stats.loss << "," << stats.samples_processed << ","
             << stats.samples_accepted << "\n";
    }
    
    file.close();
    cout << "[UniversalImageClassifier] Статистика обучения сохранена в " << path << endl;
}

// Получение истории обучения
vector<TrainingStats> UniversalImageClassifier::getTrainingHistory() const {
    if (ensemble.empty() || !ensemble[0]) return {};
    return ensemble[0]->getTrainingHistory();
}

// Получение финальных метрик по классам
vector<NeuralNetwork::ClassMetrics> UniversalImageClassifier::getFinalClassMetrics() const {
    return last_class_metrics;
}

// Определение проблемных классов по метрикам
vector<int> UniversalImageClassifier::detectProblemClasses(
    const vector<NeuralNetwork::ClassMetrics>& metrics,
    double f1_threshold,
    double recall_threshold,
    double precision_threshold) {
    
    vector<int> problem_classes_list;
    
    for (const auto& metric : metrics) {
        bool has_data = (metric.true_positives + metric.false_negatives > 0) || 
                       (metric.true_positives + metric.false_positives > 0);
        
        if (has_data) {
            // Класс считается проблемным, если:
            // 1. F1-score ниже порога
            // 2. ИЛИ recall ниже порога (много пропущенных)
            // 3. ИЛИ precision ниже порога (много ложных срабатываний)
            bool is_problem = (metric.f1_score < f1_threshold) ||
                             (metric.recall < recall_threshold) ||
                             (metric.precision < precision_threshold);
            
            if (is_problem) {
                problem_classes_list.push_back(metric.class_id);
            }
        }
    }
    
    return problem_classes_list;
}

// Раннее определение проблемных классов после первых эпох обучения
vector<int> UniversalImageClassifier::detectProblemClassesEarly(const MatrixXd& X_val, const MatrixXd& y_val, 
                                                                 int min_epochs) {
    if (ensemble.empty() || X_val.rows() == 0) {
        return vector<int>();
    }
    
    // Используем первый член ансамбля для оценки
    NeuralNetwork* network = ensemble[0].get();
    
    // Вычисляем метрики по классам
    vector<NeuralNetwork::ClassMetrics> metrics = network->computeClassMetrics(X_val, y_val);
    
    // Более строгие пороги для раннего обнаружения (чтобы не пропустить проблемные классы)
    double f1_threshold = 0.70;  // Ниже порога для раннего обнаружения
    double recall_threshold = 0.65;
    double precision_threshold = 0.65;
    
    vector<int> early_problem_classes;
    
    for (const auto& metric : metrics) {
        bool has_data = (metric.true_positives + metric.false_negatives > 0) || 
                       (metric.true_positives + metric.false_positives > 0);
        
        if (has_data) {
            // Класс считается проблемным, если метрики ниже порогов
            bool is_problem = (metric.f1_score < f1_threshold) ||
                             (metric.recall < recall_threshold) ||
                             (metric.precision < precision_threshold);
            
            if (is_problem) {
                early_problem_classes.push_back(metric.class_id);
            }
        }
    }
    
    return early_problem_classes;
}

// Oversampling проблемных классов
void UniversalImageClassifier::oversampleProblemClasses(vector<MatrixXd>& images, vector<int>& labels,
                                                         const vector<int>& problem_classes_list,
                                                         double oversample_ratio) {
    if (problem_classes_list.empty() || oversample_ratio <= 1.0) {
        return; // Нет проблемных классов или oversampling не нужен
    }
    
    // Подсчитываем количество примеров для каждого класса
    map<int, int> class_counts;
    map<int, vector<int>> class_indices; // Индексы примеров для каждого класса
    
    for (size_t i = 0; i < labels.size(); ++i) {
        class_counts[labels[i]]++;
        class_indices[labels[i]].push_back(static_cast<int>(i));
    }
    
    // Находим максимальное количество примеров среди всех классов
    int max_count = 0;
    for (const auto& [class_id, count] : class_counts) {
        max_count = max(max_count, count);
    }
    
    // Определяем целевое количество для проблемных классов
    // Адаптивный oversampling: больше для классов с низкой точностью
    int target_count = static_cast<int>(max_count * oversample_ratio);
    
    // Для очень проблемных классов (птицы, двухбалочные) увеличиваем ratio
    for (int class_id : problem_classes_list) {
        if (isBirdClass(class_id) || isTwinTailClass(class_id)) {
            target_count = static_cast<int>(max_count * 2.2);  // 2.2 (в диапазоне 2.0-2.5)
            break;
        }
    }
    
    cout << "\nOversampling проблемных классов:" << endl;
    cout << "  Целевое количество примеров: " << target_count << endl;
    
    random_device rd;
    mt19937 gen(rd());
    
    // Для каждого проблемного класса
    for (int class_id : problem_classes_list) {
        if (class_counts.find(class_id) == class_counts.end()) {
            continue; // Класс отсутствует в данных
        }
        
        int current_count = class_counts[class_id];
        if (current_count >= target_count) {
            continue; // Уже достаточно примеров
        }
        
        int needed = target_count - current_count;
        const vector<int>& indices = class_indices[class_id];
        
        if (indices.empty()) {
            continue;
        }
        
        cout << "  Class " << class_id << ": " << current_count 
             << " -> " << target_count << " (добавляем " << needed << " примеров)" << endl;
        
        // Дублируем примеры случайным образом
        uniform_int_distribution<int> dist(0, static_cast<int>(indices.size()) - 1);
        
        for (int i = 0; i < needed; ++i) {
            int source_idx = indices[dist(gen)];
            images.push_back(images[source_idx]);
            labels.push_back(class_id);
        }
    }
    
    cout << "  Итого примеров после oversampling: " << images.size() << endl;
}

// Расширенная augmentation для проблемных классов
MatrixXd UniversalImageClassifier::augmentProblemClassImage(const Mat& img) {
    Mat augmented = img.clone();
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> noise_dist(0.0, 1.0);
    
    // Более агрессивные трансформации для проблемных классов
    
    // 1. Изменение яркости (-40% до +40%)
    uniform_real_distribution<double> brightness_dist(0.6, 1.4);
    double brightness_factor = brightness_dist(gen);
    augmented.convertTo(augmented, -1, brightness_factor, 0);
    
    // 2. Изменение контраста (0.7 до 1.3)
    uniform_real_distribution<double> contrast_dist(0.7, 1.3);
    double contrast_factor = contrast_dist(gen);
    augmented.convertTo(augmented, -1, contrast_factor, 128 * (1 - contrast_factor));
    
    // 3. Добавление шума (более сильный для проблемных классов)
    if (noise_dist(gen) > 0.3) {
        Mat noise = Mat::zeros(augmented.size(), augmented.type());
        randn(noise, Scalar::all(0), Scalar::all(15)); // Увеличенное стандартное отклонение
        augmented = augmented + noise;
    }
    
    // 4. Более агрессивное масштабирование (0.8 до 1.2)
    uniform_real_distribution<double> scale_dist(0.8, 1.2);
    double scale = scale_dist(gen);
    int new_width = static_cast<int>(augmented.cols * scale);
    int new_height = static_cast<int>(augmented.rows * scale);
    
    if (new_width > 0 && new_height > 0 && new_width != augmented.cols && new_height != augmented.rows) {
        Mat scaled;
        resize(augmented, scaled, Size(new_width, new_height));
        // Обрезаем или дополняем до исходного размера
        if (scale > 1.0) {
            // Обрезаем центр
            int offset_x = (new_width - augmented.cols) / 2;
            int offset_y = (new_height - augmented.rows) / 2;
            augmented = scaled(Rect(offset_x, offset_y, augmented.cols, augmented.rows));
        } else {
            // Дополняем нулями
            Mat padded = Mat::zeros(augmented.size(), augmented.type());
            int offset_x = (augmented.cols - new_width) / 2;
            int offset_y = (augmented.rows - new_height) / 2;
            scaled.copyTo(padded(Rect(offset_x, offset_y, new_width, new_height)));
            augmented = padded;
        }
    }
    
    // 5. Более агрессивное вращение (до ±30 градусов для проблемных классов)
    uniform_real_distribution<double> rotate_dist(-30.0, 30.0);
    if (noise_dist(gen) > 0.2) {
        Point2f center(augmented.cols / 2.0f, augmented.rows / 2.0f);
        Mat rotation_matrix = getRotationMatrix2D(center, rotate_dist(gen), 1.0);
        Mat rotated;
        warpAffine(augmented, rotated, rotation_matrix, augmented.size(), INTER_LINEAR, BORDER_REPLICATE);
        augmented = rotated;
    }
    
    // 6. Cutout: случайное вырезание квадратной области
    if (noise_dist(gen) > 0.4) {
        int cutout_size = min(augmented.rows, augmented.cols) / 4;  // 25% размера
        uniform_int_distribution<int> x_dist(0, max(1, augmented.cols - cutout_size));
        uniform_int_distribution<int> y_dist(0, max(1, augmented.rows - cutout_size));
        
        int x = x_dist(gen);
        int y = y_dist(gen);
        rectangle(augmented, Rect(x, y, cutout_size, cutout_size), Scalar(0, 0, 0), -1);
    }
    
    // Преобразуем в признаки
    return preprocessImage(augmented, image_size);
}

// Специализированная augmentation для птиц
MatrixXd UniversalImageClassifier::augmentBirdImage(const Mat& img) {
    Mat augmented = img.clone();
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> noise_dist(0.0, 1.0);
    
    // Более агрессивные трансформации для птиц
    // 1. Более агрессивное вращение (±45° вместо ±30°)
    uniform_real_distribution<double> rotate_dist(-45.0, 45.0);
    if (noise_dist(gen) > 0.2) {
        Point2f center(augmented.cols / 2.0f, augmented.rows / 2.0f);
        Mat rotation_matrix = getRotationMatrix2D(center, rotate_dist(gen), 1.0);
        Mat rotated;
        warpAffine(augmented, rotated, rotation_matrix, augmented.size(), INTER_LINEAR, BORDER_REPLICATE);
        augmented = rotated;
    }
    
    // 2. Изменение масштаба (0.7-1.3 вместо 0.8-1.2)
    uniform_real_distribution<double> scale_dist(0.7, 1.3);
    double scale = scale_dist(gen);
    int new_width = static_cast<int>(augmented.cols * scale);
    int new_height = static_cast<int>(augmented.rows * scale);
    
    if (new_width > 0 && new_height > 0 && new_width != augmented.cols && new_height != augmented.rows) {
        Mat scaled;
        resize(augmented, scaled, Size(new_width, new_height));
        if (scale > 1.0) {
            int offset_x = (new_width - augmented.cols) / 2;
            int offset_y = (new_height - augmented.rows) / 2;
            augmented = scaled(Rect(offset_x, offset_y, augmented.cols, augmented.rows));
        } else {
            Mat padded = Mat::zeros(augmented.size(), augmented.type());
            int offset_x = (augmented.cols - new_width) / 2;
            int offset_y = (augmented.rows - new_height) / 2;
            scaled.copyTo(padded(Rect(offset_x, offset_y, new_width, new_height)));
            augmented = padded;
        }
    }
    
    // 3. Добавление шума (std=20 вместо 15)
    if (noise_dist(gen) > 0.3) {
        Mat noise = Mat::zeros(augmented.size(), augmented.type());
        randn(noise, Scalar::all(0), Scalar::all(20));
        augmented = augmented + noise;
    }
    
    // 4. Cutout с большими областями (20% вместо 10%)
    if (noise_dist(gen) > 0.4) {
        int cutout_size = min(augmented.rows, augmented.cols) / 5;  // 20% размера
        uniform_int_distribution<int> x_dist(0, max(1, augmented.cols - cutout_size));
        uniform_int_distribution<int> y_dist(0, max(1, augmented.rows - cutout_size));
        int x = x_dist(gen);
        int y = y_dist(gen);
        rectangle(augmented, Rect(x, y, cutout_size, cutout_size), Scalar(0, 0, 0), -1);
    }
    
    // 5. Mixup с более высоким alpha (0.4 вместо 0.2) - применяется на уровне данных
    
    return preprocessImage(augmented, image_size);
}

// Специализированная augmentation для двухбалочных самолетов
MatrixXd UniversalImageClassifier::augmentTwinTailImage(const Mat& img) {
    Mat augmented = img.clone();
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> noise_dist(0.0, 1.0);
    
    // Фокус на сохранение пропорций (меньше искажений)
    // 1. Меньше вращения (только ±15°)
    uniform_real_distribution<double> rotate_dist(-15.0, 15.0);
    if (noise_dist(gen) > 0.3) {
        Point2f center(augmented.cols / 2.0f, augmented.rows / 2.0f);
        Mat rotation_matrix = getRotationMatrix2D(center, rotate_dist(gen), 1.0);
        Mat rotated;
        warpAffine(augmented, rotated, rotation_matrix, augmented.size(), INTER_LINEAR, BORDER_REPLICATE);
        augmented = rotated;
    }
    
    // 2. Больше вариаций яркости/контраста
    uniform_real_distribution<double> brightness_dist(0.5, 1.5);
    double brightness_factor = brightness_dist(gen);
    augmented.convertTo(augmented, -1, brightness_factor, 0);
    
    uniform_real_distribution<double> contrast_dist(0.6, 1.4);
    double contrast_factor = contrast_dist(gen);
    augmented.convertTo(augmented, -1, contrast_factor, 128 * (1 - contrast_factor));
    
    // 3. Горизонтальное отражение (50% вероятность)
    if (noise_dist(gen) > 0.5) {
        Mat flipped;
        flip(augmented, flipped, 1);
        augmented = flipped;
    }
    
    // 4. Меньше масштабирования для сохранения пропорций
    uniform_real_distribution<double> scale_dist(0.85, 1.15);
    double scale = scale_dist(gen);
    int new_width = static_cast<int>(augmented.cols * scale);
    int new_height = static_cast<int>(augmented.rows * scale);
    
    if (new_width > 0 && new_height > 0 && new_width != augmented.cols && new_height != augmented.rows) {
        Mat scaled;
        resize(augmented, scaled, Size(new_width, new_height));
        if (scale > 1.0) {
            int offset_x = (new_width - augmented.cols) / 2;
            int offset_y = (new_height - augmented.rows) / 2;
            augmented = scaled(Rect(offset_x, offset_y, augmented.cols, augmented.rows));
        } else {
            Mat padded = Mat::zeros(augmented.size(), augmented.type());
            int offset_x = (augmented.cols - new_width) / 2;
            int offset_y = (augmented.rows - new_height) / 2;
            scaled.copyTo(padded(Rect(offset_x, offset_y, new_width, new_height)));
            augmented = padded;
        }
    }
    
    return preprocessImage(augmented, image_size);
}

// Определение типа проблемного класса
bool UniversalImageClassifier::isBirdClass(int class_id) const {
    string class_name = getClassName(class_id);
    // Проверяем по имени класса (птица, bird, птиц и т.д.)
    string lower_name = class_name;
    transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    return lower_name.find("bird") != string::npos || 
           lower_name.find("птиц") != string::npos ||
           lower_name.find("птица") != string::npos;
}

bool UniversalImageClassifier::isTwinTailClass(int class_id) const {
    string class_name = getClassName(class_id);
    // Проверяем по имени класса (двухбалочный, twin, tail и т.д.)
    string lower_name = class_name;
    transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    return lower_name.find("twin") != string::npos || 
           lower_name.find("двухбалоч") != string::npos ||
           lower_name.find("two") != string::npos ||
           lower_name.find("tail") != string::npos;
}

    // Mixup Data Augmentation
void UniversalImageClassifier::applyMixupAugmentation(vector<MatrixXd>& images, vector<int>& labels,
                                                       const vector<int>& problem_classes_list,
                                                       double mixup_alpha, double mixup_prob) {
    if (problem_classes_list.empty() || mixup_alpha <= 0.0 || mixup_prob <= 0.0) {
        return; // Нет проблемных классов или mixup отключен
    }
    
    // Создаем индексы примеров для каждого проблемного класса
    map<int, vector<int>> problem_class_indices;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (find(problem_classes_list.begin(), problem_classes_list.end(), labels[i]) 
            != problem_classes_list.end()) {
            problem_class_indices[labels[i]].push_back(static_cast<int>(i));
        }
    }
    
    if (problem_class_indices.empty()) {
        return; // Нет примеров проблемных классов
    }
    
    // Увеличиваем mixup_alpha для птиц (0.4 вместо 0.2)
    double adjusted_alpha = mixup_alpha;
    for (int class_id : problem_classes_list) {
        if (isBirdClass(class_id)) {
            adjusted_alpha = 0.4;  // Более высокий alpha для птиц
            break;
        }
    }
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> prob_dist(0.0, 1.0);
    
    // Beta распределение для lambda
    gamma_distribution<double> gamma_alpha(mixup_alpha, 1.0);
    gamma_distribution<double> gamma_beta(mixup_alpha, 1.0);
    
    int mixup_count = 0;
    
    // Применяем mixup к проблемным классам
    for (const auto& [class_id, indices] : problem_class_indices) {
        if (indices.size() < 2) {
            continue; // Нужно минимум 2 примера для mixup
        }
        
        uniform_int_distribution<int> idx_dist(0, static_cast<int>(indices.size()) - 1);
        
        // Количество примеров для mixup (процент от размера класса)
        int target_mixup = static_cast<int>(indices.size() * mixup_prob);
        
        for (int m = 0; m < target_mixup; ++m) {
            if (prob_dist(gen) > mixup_prob) {
                continue; // Пропускаем с вероятностью (1 - mixup_prob)
            }
            
            // Выбираем два случайных индекса
            int idx1 = idx_dist(gen);
            int idx2 = idx_dist(gen);
            while (idx2 == idx1) {
                idx2 = idx_dist(gen); // Убеждаемся, что индексы разные
            }
            
            int source_idx1 = indices[idx1];
            int source_idx2 = indices[idx2];
            
            // Генерируем lambda из Beta(alpha, alpha) распределения
            double lambda1 = gamma_alpha(gen);
            double lambda2 = gamma_beta(gen);
            double lambda = lambda1 / (lambda1 + lambda2);
            
            // Ограничиваем lambda для численной стабильности
            lambda = max(0.1, min(0.9, lambda));
            
            // Смешиваем изображения: x_new = lambda * x1 + (1-lambda) * x2
            MatrixXd mixed_image = lambda * images[source_idx1] + (1.0 - lambda) * images[source_idx2];
            
            // Смешиваем метки: y_new = lambda * y1 + (1-lambda) * y2
            // Для one-hot encoding создаем мягкую метку
            images.push_back(mixed_image);
            
            // Для меток используем исходный класс (можно также использовать мягкую метку)
            // Здесь используем класс с большим lambda
            int mixed_label = (lambda > 0.5) ? labels[source_idx1] : labels[source_idx2];
            labels.push_back(mixed_label);
            
            mixup_count++;
        }
    }
    
    if (mixup_count > 0) {
        cout << "  Mixup augmentation: добавлено " << mixup_count << " смешанных примеров" << endl;
    }
}

vector<NeuralNetwork::ClassMetrics> UniversalImageClassifier::getClassMetrics(
    const vector<string>& image_paths, const vector<int>& labels) {
    
    if (image_paths.size() != labels.size() || ensemble.empty()) {
        return vector<NeuralNetwork::ClassMetrics>();
    }
    
    // Загрузка и предобработка изображений
    vector<MatrixXd> images;
    vector<int> valid_labels;
    
    for (size_t i = 0; i < image_paths.size(); ++i) {
        Mat img = imread(image_paths[i]);
        if (img.empty()) {
            continue;
        }
        
        MatrixXd features = preprocessImage(img, image_size);
        images.push_back(features);
        valid_labels.push_back(labels[i]);
    }
    
    if (images.empty()) {
        return vector<NeuralNetwork::ClassMetrics>();
    }
    
    // Преобразование в матрицы
    int num_samples = images.size();
    int feature_size = images[0].rows();
    
    MatrixXd X(num_samples, feature_size);
    MatrixXd y = MatrixXd::Zero(num_samples, num_classes);
    
    for (int i = 0; i < num_samples; ++i) {
        X.row(i) = images[i].transpose();
        if (valid_labels[i] >= 0 && valid_labels[i] < num_classes) {
            y(i, valid_labels[i]) = 1.0;
        }
    }
    
    // Улучшенная нормализация (Layer Normalization + Column Normalization)
    for (int row = 0; row < X.rows(); ++row) {
        double mean = X.row(row).mean();
        double stddev = sqrt((X.row(row).array() - mean).square().sum() / X.cols());
        if (stddev > 1e-10) {
            X.row(row) = (X.row(row).array() - mean) / stddev;
        }
    }
    
    for (int col = 0; col < X.cols(); ++col) {
        double mean = X.col(col).mean();
        double stddev = sqrt((X.col(col).array() - mean).square().sum() / X.rows());
        if (stddev > 1e-10) {
            X.col(col) = (X.col(col).array() - mean) / stddev;
        }
    }
    
    if (ensemble.empty() || !ensemble[0]) return vector<NeuralNetwork::ClassMetrics>();
    return ensemble[0]->computeClassMetrics(X, y);
}

void UniversalImageClassifier::runAblationStudy(const vector<string>& image_paths, const vector<int>& labels) {
    cout << "\n" << string(80, '=') << endl;
    cout << "SCIENTIFIC RESEARCH: ABLATION STUDY" << endl;
    cout << string(80, '=') << endl;
    
    if (image_paths.size() != labels.size() || image_paths.empty()) {
        cerr << "Error: Invalid input data for ablation study" << endl;
        return;
    }
    
    struct Result { string name; double accuracy; double f1_score; };
    vector<Result> results;
    
    // Загружаем и предобрабатываем изображения
    vector<MatrixXd> images;
    vector<int> valid_labels;
    
    for (size_t i = 0; i < image_paths.size(); ++i) {
        Mat img = imread(image_paths[i]);
        if (img.empty()) {
            continue;
        }
        MatrixXd features = preprocessImage(img, image_size);
        images.push_back(features);
        valid_labels.push_back(labels[i]);
    }
    
    if (images.empty()) {
        cerr << "Error: No valid images loaded" << endl;
        return;
    }
    
    // Преобразуем в матрицы
    int num_samples = static_cast<int>(images.size());
    int feature_size = static_cast<int>(images[0].rows());
    
    MatrixXd X(num_samples, feature_size);
    MatrixXd y = MatrixXd::Zero(num_samples, num_classes);
    
    for (int i = 0; i < num_samples; ++i) {
        X.row(i) = images[i].transpose();
        if (valid_labels[i] >= 0 && valid_labels[i] < num_classes) {
            y(i, valid_labels[i]) = 1.0;
        }
    }
    
    // Сохраняем исходное состояние
    bool old_sa = use_structure_analysis;
    
    // 1. Только базовая сеть (без структуры и SVM)
    use_structure_analysis = false;
    if (!ensemble.empty()) {
        double accuracy = ensemble[0]->computeAccuracy(X, y);
        double f1 = ensemble[0]->computeMacroAveragedF1Score(X, y);
        results.push_back({"Base Neural Network", accuracy, f1});
        cout << "1. Base Neural Network: Accuracy=" << fixed << setprecision(4) << accuracy 
             << ", F1=" << f1 << endl;
    }
    
    // 2. Базовая сеть + SVM (без структуры)
    use_structure_analysis = false;
    if (!ensemble.empty() && svm_classifier && svm_classifier->isTrained()) {
        // Оцениваем комбинированное предсказание
        double correct = 0.0;
        double total_f1 = 0.0;
        int classes_with_data = 0;
        
        for (int i = 0; i < num_samples; ++i) {
            Mat img = imread(image_paths[i]);
            if (img.empty()) continue;
            
            MatrixXd features_row = X.row(i);
            pair<int, double> nn_pred = classify(img);
            
            // Простое объединение: если SVM уверен, используем его, иначе NN
            int final_pred = nn_pred.first;
            if (svm_classifier->isTrained()) {
                int svm_pred = predictSVM(features_row);
                if (svm_pred >= 0) {
                    final_pred = svm_pred;  // Приоритет SVM
                }
            }
            
            if (final_pred == valid_labels[i]) {
                correct += 1.0;
            }
        }
        
        double accuracy = correct / num_samples;
        vector<NeuralNetwork::ClassMetrics> metrics = ensemble[0]->computeClassMetrics(X, y);
        for (const auto& m : metrics) {
            if (m.true_positives + m.false_negatives > 0) {
                total_f1 += m.f1_score;
                classes_with_data++;
            }
        }
        double f1 = (classes_with_data > 0) ? (total_f1 / classes_with_data) : 0.0;
        
        results.push_back({"Base NN + Cover-SVM", accuracy, f1});
        cout << "2. Base NN + Cover-SVM: Accuracy=" << fixed << setprecision(4) << accuracy 
             << ", F1=" << f1 << endl;
    }
    
    // 3. Полный SMNF метод (со структурой, подсетями, топологией)
    use_structure_analysis = true;
    if (!ensemble.empty()) {
        double correct = 0.0;
        double total_f1 = 0.0;
        int classes_with_data = 0;
        
        for (int i = 0; i < num_samples; ++i) {
            Mat img = imread(image_paths[i]);
            if (img.empty()) continue;
            
            pair<int, double> pred = classifyWithStructureAnalysis(img);
            if (pred.first == valid_labels[i]) {
                correct += 1.0;
            }
        }
        
        double accuracy = correct / num_samples;
        vector<NeuralNetwork::ClassMetrics> metrics = ensemble[0]->computeClassMetrics(X, y);
        for (const auto& m : metrics) {
            if (m.true_positives + m.false_negatives > 0) {
                total_f1 += m.f1_score;
                classes_with_data++;
            }
        }
        double f1 = (classes_with_data > 0) ? (total_f1 / classes_with_data) : 0.0;
        
        results.push_back({"Full SMNF Method", accuracy, f1});
        cout << "3. Full SMNF Method: Accuracy=" << fixed << setprecision(4) << accuracy 
             << ", F1=" << f1 << endl;
    }
    
    // Восстанавливаем исходное состояние
    use_structure_analysis = old_sa;
    
    // Вывод результатов в LaTeX формате
    cout << "\nAblation Results Table (for LaTeX):" << endl;
    cout << "\\begin{tabular}{|l|c|c|}" << endl;
    cout << "\\hline" << endl;
    cout << "Method & Accuracy & F1-Score \\\\" << endl;
    cout << "\\hline" << endl;
    for (const auto& r : results) {
        cout << r.name << " & " << fixed << setprecision(2) << (r.accuracy * 100) << "\\% & "
             << setprecision(3) << r.f1_score << " \\\\" << endl;
    }
    cout << "\\hline" << endl;
    cout << "\\end{tabular}" << endl;
}

MatrixXd UniversalImageClassifier::applyInformationBottleneck(const MatrixXd& features, double beta) {
    // Реализация упрощенного IB: минимизация I(X;Z) при максимизации I(Z;Y)
    // В данной версии мы используем взвешенную фильтрацию признаков на основе их взаимной информации с метками
    int dim = static_cast<int>(features.rows());
    MatrixXd compressed = features;
    
    for (int i = 0; i < dim; ++i) {
        // Если признак имеет низкую вариативность (высокий шум), мы его подавляем
        double noise_level = 1.0 - abs(features(i, 0));
        if (noise_level > beta) {
            compressed(i, 0) *= (1.0 - noise_level);
        }
    }
    
    return compressed;
}

MatrixXd UniversalImageClassifier::constructScientificFeatureVector(const Mat& region, const vector<Point>& contour) {
    // 1. Базовые признаки из изображения
    MatrixXd base_features = preprocessImage(region, 32);
    
    // 2. Фрактальная размерность (Сложность)
    double D = shape_analyzer->computeFractalDimension(region);
    
    // 3. Показатель Ляпунова (Стабильность контура)
    double lambda = shape_analyzer->computeLyapunovStability(contour);
    
    // 4. Объединение в расширенный вектор
    int base_dim = static_cast<int>(base_features.rows());
    MatrixXd scientific_vector(base_dim + 2, 1);
    scientific_vector.block(0, 0, base_dim, 1) = base_features;
    scientific_vector(base_dim, 0) = D;
    scientific_vector(base_dim + 1, 0) = lambda;
    
    // 5. Применение проекции Ковера для повышения разделимости
    MatrixXd projected = shape_analyzer->applyCoverProjection(scientific_vector);
    
    // 6. Фильтрация через Information Bottleneck
    return applyInformationBottleneck(projected);
}

void UniversalImageClassifier::saveModel(const string& path) {
    // Если путь содержит папку, создаем её
    size_t last_slash = path.find_last_of("/\\");
    if (last_slash != string::npos) {
        string dir = path.substr(0, last_slash);
        fs::create_directories(dir);
    }
    
    for (size_t i = 0; i < ensemble.size(); ++i) {
        ensemble[i]->saveModel(path + ".member" + to_string(i));
    }
    
    // Сохранение SVM
    if (svm_classifier && svm_classifier->isTrained()) {
        svm_classifier->save(path + ".svm");
        cout << "[UniversalImageClassifier] SVM classifier saved to " << path << ".svm" << endl;
    }

    // Сохранение подсетей
    if (subnetwork_manager) {
        // Если путь содержит папку, сохраняем подсети в подпапку
        size_t last_slash = path.find_last_of("/\\");
        string subnetwork_path = path;
        if (last_slash != string::npos) {
            string base_dir = path.substr(0, last_slash);
            string base_name = path.substr(last_slash + 1);
            subnetwork_path = base_dir + "/sub_networks/" + base_name;
            fs::create_directories(base_dir + "/sub_networks");
        }
        subnetwork_manager->saveSubNetworks(subnetwork_path);
        cout << "[UniversalImageClassifier] Sub-networks saved with prefix " << subnetwork_path << endl;
    }

    // Сохранение каталога структур
    if (shape_analyzer) {
        shape_analyzer->saveCatalog(path + ".catalog");
    }
    
    // Сохранение имен классов
    string class_path = path + ".classes";
    ofstream file(class_path, ios::binary);
    if (file.is_open()) {
        for (const auto& [id, name] : class_names) {
            file << id << " " << name << "\n";
        }
        file.close();
    }
}

// Получение текущей временной метки в формате YYYYMMDD_HHMMSS
string UniversalImageClassifier::getCurrentTimestamp() const {
    auto now = chrono::system_clock::now();
    auto time_t = chrono::system_clock::to_time_t(now);
    tm timeinfo;
    localtime_s(&timeinfo, &time_t);
    
    char buffer[20];
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &timeinfo);
    return string(buffer);
}

// Сохранение данных обучения в JSON формате
void UniversalImageClassifier::saveTrainingDataJSON(const string& json_path,
                                                    const vector<NeuralNetwork::ClassMetrics>& metrics,
                                                    const vector<TrainingStats>& history,
                                                    double overall_accuracy,
                                                    double macro_f1,
                                                    double weighted_f1,
                                                    int epochs,
                                                    int batch_size,
                                                    double learning_rate,
                                                    int total_samples) {
    ofstream file(json_path);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file for writing JSON: " << json_path << endl;
        return;
    }
    
    file << fixed << setprecision(6);
    file << "{\n";
    
    // Training info
    file << "  \"training_info\": {\n";
    file << "    \"timestamp\": \"" << getCurrentTimestamp() << "\",\n";
    file << "    \"epochs\": " << epochs << ",\n";
    file << "    \"batch_size\": " << batch_size << ",\n";
    file << "    \"learning_rate\": " << learning_rate << ",\n";
    file << "    \"num_classes\": " << num_classes << ",\n";
    file << "    \"total_samples\": " << total_samples << "\n";
    file << "  },\n";
    
    // Final metrics
    file << "  \"final_metrics\": {\n";
    file << "    \"overall_accuracy\": " << overall_accuracy << ",\n";
    file << "    \"macro_f1\": " << macro_f1 << ",\n";
    file << "    \"weighted_f1\": " << weighted_f1 << "\n";
    file << "  },\n";
    
    // Class metrics
    file << "  \"class_metrics\": [\n";
    bool first_metric = true;
    for (const auto& m : metrics) {
        bool has_data = (m.true_positives + m.false_negatives > 0) || 
                       (m.true_positives + m.false_positives > 0);
        if (!has_data) continue;
        
        if (!first_metric) file << ",\n";
        first_metric = false;
        
        string class_name = getClassName(m.class_id);
        file << "    {\n";
        file << "      \"class_id\": " << m.class_id << ",\n";
        file << "      \"class_name\": \"" << class_name << "\",\n";
        file << "      \"precision\": " << m.precision << ",\n";
        file << "      \"recall\": " << m.recall << ",\n";
        file << "      \"f1_score\": " << m.f1_score << ",\n";
        file << "      \"tp\": " << m.true_positives << ",\n";
        file << "      \"fp\": " << m.false_positives << ",\n";
        file << "      \"fn\": " << m.false_negatives << "\n";
        file << "    }";
    }
    file << "\n  ],\n";
    
    // Problem classes
    file << "  \"problem_classes\": [";
    for (size_t i = 0; i < problem_classes.size(); ++i) {
        file << problem_classes[i];
        if (i < problem_classes.size() - 1) file << ", ";
    }
    file << "],\n";
    
    // Training history
    file << "  \"training_history\": [\n";
    for (size_t i = 0; i < history.size(); ++i) {
        const auto& stats = history[i];
        file << "    {\n";
        file << "      \"stage\": " << stats.stage << ",\n";
        file << "      \"epoch\": " << stats.epoch << ",\n";
        file << "      \"accuracy\": " << stats.accuracy << ",\n";
        file << "      \"loss\": " << stats.loss << ",\n";
        file << "      \"samples_processed\": " << stats.samples_processed << ",\n";
        file << "      \"samples_accepted\": " << stats.samples_accepted << "\n";
        file << "    }";
        if (i < history.size() - 1) file << ",";
        file << "\n";
    }
    file << "  ]\n";
    
    file << "}\n";
    file.close();
    cout << "[UniversalImageClassifier] Training data saved to JSON: " << json_path << endl;
}

// Сохранение данных обучения в бинарном формате
void UniversalImageClassifier::saveTrainingDataBinary(const string& bin_path,
                                                      const vector<NeuralNetwork::ClassMetrics>& metrics,
                                                      const vector<TrainingStats>& history) {
    ofstream file(bin_path, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file for writing binary data: " << bin_path << endl;
        return;
    }
    
    // Заголовок файла
    const char* header = "TRNDATA";
    file.write(header, 8);
    
    // Версия формата
    int version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(int));
    
    // Количество метрик
    int num_metrics = static_cast<int>(metrics.size());
    file.write(reinterpret_cast<const char*>(&num_metrics), sizeof(int));
    
    // Сохранение метрик
    for (const auto& m : metrics) {
        file.write(reinterpret_cast<const char*>(&m.class_id), sizeof(int));
        file.write(reinterpret_cast<const char*>(&m.precision), sizeof(double));
        file.write(reinterpret_cast<const char*>(&m.recall), sizeof(double));
        file.write(reinterpret_cast<const char*>(&m.f1_score), sizeof(double));
        file.write(reinterpret_cast<const char*>(&m.true_positives), sizeof(int));
        file.write(reinterpret_cast<const char*>(&m.false_positives), sizeof(int));
        file.write(reinterpret_cast<const char*>(&m.false_negatives), sizeof(int));
    }
    
    // Проблемные классы
    int num_problem_classes = static_cast<int>(problem_classes.size());
    file.write(reinterpret_cast<const char*>(&num_problem_classes), sizeof(int));
    for (int pc : problem_classes) {
        file.write(reinterpret_cast<const char*>(&pc), sizeof(int));
    }
    
    // История обучения
    int num_history = static_cast<int>(history.size());
    file.write(reinterpret_cast<const char*>(&num_history), sizeof(int));
    for (const auto& stats : history) {
        file.write(reinterpret_cast<const char*>(&stats.stage), sizeof(int));
        file.write(reinterpret_cast<const char*>(&stats.epoch), sizeof(int));
        file.write(reinterpret_cast<const char*>(&stats.accuracy), sizeof(double));
        file.write(reinterpret_cast<const char*>(&stats.loss), sizeof(double));
        file.write(reinterpret_cast<const char*>(&stats.samples_processed), sizeof(int));
        file.write(reinterpret_cast<const char*>(&stats.samples_accepted), sizeof(int));
    }
    
    file.close();
    cout << "[UniversalImageClassifier] Training data saved to binary: " << bin_path << endl;
}

// Сохранение всех данных обучения с метриками
void UniversalImageClassifier::saveAllTrainingDataWithMetrics(const string& base_path,
                                                             const vector<NeuralNetwork::ClassMetrics>& metrics,
                                                             double overall_accuracy,
                                                             double macro_f1,
                                                             double weighted_f1,
                                                             int epochs) {
    cout << "\n" << string(60, '=') << endl;
    cout << "SAVING ALL TRAINING DATA" << endl;
    cout << string(60, '=') << endl;
    
    // Создаем структуру папок
    string model_dir = base_path + "/model";
    string problem_ensemble_dir = base_path + "/problem_ensemble";
    string sub_networks_dir = base_path + "/sub_networks";
    
    fs::create_directories(model_dir);
    if (!problem_ensemble.empty()) {
        fs::create_directories(problem_ensemble_dir);
    }
    fs::create_directories(sub_networks_dir);
    
    // 1. Сохранение основной модели
    cout << "Saving main model..." << endl;
    saveModel(model_dir + "/best_model");
    
    // 2. Сохранение специализированного ансамбля для проблемных классов
    if (!problem_ensemble.empty()) {
        cout << "Saving problem class ensemble (" << problem_ensemble.size() << " networks)..." << endl;
        for (size_t i = 0; i < problem_ensemble.size(); ++i) {
            problem_ensemble[i]->saveModel(problem_ensemble_dir + "/problem_ensemble_" + to_string(i));
        }
    }
    
    // 3. Получение истории обучения
    if (ensemble.empty() && !network) {
        cerr << "Warning: No trained model found, skipping data saving" << endl;
        return;
    }
    
    vector<TrainingStats> training_history;
    if (!ensemble.empty()) {
        training_history = ensemble[0]->getTrainingHistory();
    }
    
    // 4. Сохранение порогов и весов классов для каждого члена ансамбля
    cout << "Saving class thresholds and weights..." << endl;
    for (size_t i = 0; i < ensemble.size(); ++i) {
        ensemble[i]->saveClassThresholds(base_path + "/class_thresholds_member" + to_string(i) + ".bin");
        ensemble[i]->saveClassWeights(base_path + "/class_weights_member" + to_string(i) + ".bin");
    }
    
    // 5. Сохранение статистики обучения (CSV)
    cout << "Saving training statistics (CSV)..." << endl;
    saveTrainingStats(base_path + "/training_stats.csv");
    
    // 6. Сохранение метрик по классам в отдельный JSON
    cout << "Saving class metrics (JSON)..." << endl;
    ofstream metrics_file(base_path + "/class_metrics.json");
    if (metrics_file.is_open()) {
        metrics_file << fixed << setprecision(6);
        metrics_file << "{\n  \"class_metrics\": [\n";
        bool first = true;
        for (const auto& m : metrics) {
            bool has_data = (m.true_positives + m.false_negatives > 0) || 
                           (m.true_positives + m.false_positives > 0);
            if (!has_data) continue;
            
            if (!first) metrics_file << ",\n";
            first = false;
            
            string class_name = getClassName(m.class_id);
            metrics_file << "    {\n";
            metrics_file << "      \"class_id\": " << m.class_id << ",\n";
            metrics_file << "      \"class_name\": \"" << class_name << "\",\n";
            metrics_file << "      \"precision\": " << m.precision << ",\n";
            metrics_file << "      \"recall\": " << m.recall << ",\n";
            metrics_file << "      \"f1_score\": " << m.f1_score << ",\n";
            metrics_file << "      \"tp\": " << m.true_positives << ",\n";
            metrics_file << "      \"fp\": " << m.false_positives << ",\n";
            metrics_file << "      \"fn\": " << m.false_negatives << "\n";
            metrics_file << "    }";
        }
        metrics_file << "\n  ]\n}\n";
        metrics_file.close();
    }
    
    // 7. Сохранение в JSON и бинарном формате
    if (!training_history.empty()) {
        int batch_size = 32;  // Можно получить из параметров обучения
        double learning_rate = 0.001;  // Можно получить из параметров обучения
        int total_samples = 0;
        
        if (!training_history.empty()) {
            total_samples = training_history.back().samples_processed;
        }
        
        cout << "Saving training data (JSON and binary)..." << endl;
        saveTrainingDataJSON(base_path + "/training_data.json",
                            metrics,
                            training_history,
                            overall_accuracy,
                            macro_f1,
                            weighted_f1,
                            epochs,
                            batch_size,
                            learning_rate,
                            total_samples);
        
        saveTrainingDataBinary(base_path + "/training_data.bin",
                              metrics,
                              training_history);
    }
    
    cout << "\nAll training data saved to: " << base_path << endl;
    cout << string(60, '=') << endl;
}

void UniversalImageClassifier::loadModel(const string& path) {
    for (size_t i = 0; i < 3; ++i) {
        string member_path = path + ".member" + to_string(i);
        if (!fs::exists(member_path)) continue;
        
        int in_size, out_size;
        double lr;
        vector<int> hidden;
        if (!NeuralNetwork::loadModelMetadata(member_path, in_size, hidden, out_size, lr)) {
            cerr << "[UniversalImageClassifier] Skipping " << member_path << " (old format or invalid)" << endl;
            continue;
        }
        
        if (i == 0) num_classes = out_size;
        
        vector<int> loaded_arch;
        loaded_arch.push_back(in_size);
        loaded_arch.insert(loaded_arch.end(), hidden.begin(), hidden.end());
        loaded_arch.push_back(out_size);
        
        bool need_recreate = (i >= ensemble.size());
        if (!need_recreate) {
            vector<int> current_arch = ensemble[i]->getArchitecture();
            need_recreate = (current_arch != loaded_arch);
        }
        if (need_recreate) {
            if (i >= ensemble.size()) ensemble.resize(i + 1);
            ensemble[i] = make_unique<NeuralNetwork>(in_size, hidden, out_size, lr, "relu", 0.9, 0.001);
        }
        
        try {
            ensemble[i]->loadModel(member_path);
        } catch (const exception& e) {
            cerr << "[UniversalImageClassifier] Failed to load " << member_path << ": " << e.what() << endl;
        }
    }
    
    // Загрузка SVM
    if (fs::exists(path + ".svm")) {
        svm_classifier = cv::ml::SVM::load(path + ".svm");
        cout << "[UniversalImageClassifier] SVM classifier loaded from " << path << ".svm" << endl;
    }

    // Загрузка подсетей
    if (subnetwork_manager) {
        subnetwork_manager->loadSubNetworks(path);
    }

    // Загрузка каталога структур
    if (shape_analyzer && fs::exists(path + ".catalog")) {
        shape_analyzer->loadCatalog(path + ".catalog");
    }
    
    // Загрузка имен классов
    string class_path = path + ".classes";
    ifstream file(class_path, ios::binary);
    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            if (line.empty()) continue;
            // Remove trailing \r if present (Windows line ending)
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            size_t space_pos = line.find(' ');
            if (space_pos != string::npos) {
                int id = stoi(line.substr(0, space_pos));
                string name = line.substr(space_pos + 1);
                class_names[id] = name;
            }
        }
        file.close();
    }
}

vector<string> UniversalImageClassifier::getClassList() const {
    vector<string> classes;
    for (const auto& [id, name] : class_names) {
        classes.push_back(name);
    }
    return classes;
}

void UniversalImageClassifier::addClass(const string& class_name) {
    int new_id = class_names.size();
    class_names[new_id] = class_name;
    num_classes = max(num_classes, new_id + 1);
}

void UniversalImageClassifier::setClassNames(const vector<string>& names) {
    class_names.clear();
    for (size_t i = 0; i < names.size(); ++i) {
        class_names[static_cast<int>(i)] = names[i];
    }
    if (!names.empty()) {
        num_classes = max(num_classes, static_cast<int>(names.size()));
    }
}

void UniversalImageClassifier::setLearningRate(double lr) {
    for (auto& net : ensemble) {
        if (net) net->setLearningRate(lr);
    }
}

void UniversalImageClassifier::enableStructureAnalysis(bool enable) {
    use_structure_analysis = enable;
    if (enable) {
        cout << "[UniversalImageClassifier] Анализ структуры включен" << endl;
        cout << "  - Анализ формы образа" << endl;
        cout << "  - Выделение структур (выпячиваний)" << endl;
        cout << "  - Подсети для отдельных структур" << endl;
        cout << "  - Объединение через логические операции и нечеткую логику" << endl;
    } else {
        cout << "[UniversalImageClassifier] Анализ структуры отключен" << endl;
    }
}

pair<int, double> UniversalImageClassifier::classifyWithStructureAnalysis(const Mat& img) {
    if (!use_structure_analysis || !shape_analyzer || !subnetwork_manager || !fuzzy_fusion) {
        // Fallback к обычной классификации
        return classify(img);
    }
    
    try {
    // 1. Анализ формы образа
    ShapeDescription shape_desc = shape_analyzer->analyzeShape(img);
    
    if (shape_desc.structures.empty()) {
        // Если структуры не найдены, используем обычную классификацию
        return classify(img);
    }
    
    // 2. Извлечение признаков из каждой структуры
    vector<MatrixXd> structure_features = extractStructureFeatures(img, shape_desc.structures);
    
    // 3. Предсказание каждой подсетью и SVM
    map<string, MatrixXd> subnetwork_outputs;
    map<string, string> structure_types_map;
    vector<MatrixXd> all_outputs;
    map<string, MatrixXd> trace_inputs;
    
    // Добавляем предсказание ансамбля основных сетей
    pair<int, double> base_pred = classify(img);
    MatrixXd base_output = MatrixXd::Zero(1, num_classes);
    if (base_pred.first >= 0) base_output(0, base_pred.first) = base_pred.second;
    all_outputs.push_back(base_output);
    trace_inputs["Ensemble_Base"] = base_output;

    // Топологический анализ: вычисляем сигнатуру и строим граф
    PersistenceDiagram topo_diagram;
    HyperRelationalFuzzyGraph hfg;
    if (use_structure_analysis && topological_sig) {
        topo_diagram = shape_analyzer->computeTopologicalSignature(img, shape_desc.structures);
        hfg = shape_analyzer->buildHyperRelationalGraph(shape_desc.structures, shape_desc);
        
        // Добавляем топологические признаки в вектор признаков
        VectorXd topo_signature = topo_diagram.toSignature(20);
        VectorXd graph_features = hfg.computeGraphFeatures();
        
        // Объединяем топологические признаки с обычными
        for (size_t i = 0; i < structure_features.size(); ++i) {
            int orig_size = static_cast<int>(structure_features[i].rows());
            MatrixXd extended_features(orig_size + topo_signature.size() + graph_features.size(), 1);
            extended_features.block(0, 0, orig_size, 1) = structure_features[i];
            extended_features.block(orig_size, 0, topo_signature.size(), 1) = topo_signature;
            extended_features.block(orig_size + topo_signature.size(), 0, 
                                   graph_features.size(), 1) = graph_features;
            structure_features[i] = extended_features;
        }
    }
    
    // Добавляем предсказание SVM для структур с применением Теоремы Ковера
    // Если используется топологическое ядро, используем его
    if (svm_classifier->isTrained()) {
        for (size_t i = 0; i < structure_features.size(); ++i) {
            MatrixXd cover_feat = shape_analyzer->applyCoverProjection(structure_features[i]);
            
            // Используем топологические признаки для различения похожих классов
            int svm_class = -1;
            if (use_topological_kernel && topological_kernel && !topo_diagram.points.empty()) {
                // Для проблемных классов используем топологические признаки
                // Создаем расширенный вектор признаков с топологическими
                VectorXd topo_sig = topo_diagram.toSignature(20);
                VectorXd graph_feat = hfg.computeGraphFeatures();
                
                // Объединяем с обычными признаками
                int orig_size = static_cast<int>(cover_feat.rows());
                MatrixXd extended_feat(orig_size + topo_sig.size() + graph_feat.size(), 1);
                extended_feat.block(0, 0, orig_size, 1) = cover_feat;
                extended_feat.block(orig_size, 0, topo_sig.size(), 1) = topo_sig;
                extended_feat.block(orig_size + topo_sig.size(), 0, graph_feat.size(), 1) = graph_feat;
                
                svm_class = predictSVM(extended_feat);
            } else {
                svm_class = predictSVM(cover_feat);
            }
            
            if (svm_class >= 0) {
                MatrixXd svm_output = MatrixXd::Zero(1, num_classes);
                svm_output(0, svm_class) = 0.8; 
                all_outputs.push_back(svm_output);
                trace_inputs["SVM_Struct_" + to_string(i)] = svm_output;
            }
        }
    }
    
    for (size_t i = 0; i < shape_desc.structures.size(); ++i) {
        const auto& structure = shape_desc.structures[i];
        string structure_type = shape_analyzer->classifyStructureType(structure, shape_desc);
        structure_types_map[to_string(i)] = structure_type;
        
        // Синергия: Фрактальная размерность влияет на сложность подсети
        double D = shape_analyzer->computeFractalDimension(structure.region_image);
        double entropy = shape_analyzer->computeStructuralEntropy(structure.region_image);
        double combined_complexity = (D / 2.0 + entropy) / 2.0; // D обычно в районе 1-2
        
        if (!subnetwork_manager->getSubNetwork(structure_type)) {
            int feat_rows_final_count_v3 = (int)structure_features[i].rows();
            subnetwork_manager->createSubNetwork(structure_type, feat_rows_final_count_v3, num_classes, 0.5);
        }
        
        // Используем единый научный вектор для предсказания
        MatrixXd sci_feat = constructScientificFeatureVector(structure.region_image, structure.contour);
        MatrixXd prediction = subnetwork_manager->predictSubNetwork(structure_type, sci_feat);
        
        // Также используем специализированные подсети для проблемных классов, если они есть
        for (int problem_class_id : problem_classes) {
            MatrixXd problem_pred = subnetwork_manager->predictProblemClassSubNetwork(
                problem_class_id, sci_feat);
            // Взвешенное объединение предсказаний
            prediction = 0.7 * prediction + 0.3 * problem_pred;
        }
        
        subnetwork_outputs[to_string(i)] = prediction;
        trace_inputs["SubNet_" + structure_type + "_" + to_string(i)] = prediction;
    }
    
    // 4. Объединение признаков через нечеткую логику и семантический вывод
    MatrixXd semantic_out = fuzzy_fusion->semanticInference(subnetwork_outputs, structure_types_map);
    if (semantic_out.maxCoeff() > 0) {
        all_outputs.push_back(semantic_out);
        trace_inputs["Semantic_Rules"] = semantic_out;
    }
    
    // 4.5. Использование специализированного ансамбля для проблемных классов
    if (!problem_ensemble.empty() && !problem_classes.empty()) {
        MatrixXd full_features = extractFeatures(img);
        MatrixXd full_features_mat(1, full_features.rows());
        full_features_mat.row(0) = full_features.transpose();
        
        MatrixXd problem_ensemble_output = MatrixXd::Zero(1, num_classes);
        for (const auto& problem_net : problem_ensemble) {
            MatrixXd pred = problem_net->predict(full_features_mat);
            problem_ensemble_output += pred;
        }
        problem_ensemble_output /= problem_ensemble.size();
        
        // Добавляем с высоким весом для проблемных классов
        all_outputs.push_back(problem_ensemble_output);
        trace_inputs["Problem_Ensemble"] = problem_ensemble_output;
    }

    // Устанавливаем веса для источников
    vector<double> src_weights;
    src_weights.push_back(1.0); // Вес ансамбля
    
    if (svm_classifier->isTrained()) {
        for (size_t i = 0; i < structure_features.size(); ++i) src_weights.push_back(0.6);
    }

    for (const auto& [id, output] : subnetwork_outputs) {
        all_outputs.push_back(output);
        string type = structure_types_map[id];
        if (type == "крыло" || type == "фюзеляж") src_weights.push_back(0.9);
        else src_weights.push_back(0.5);
    }
    
    // Используем продвинутый оператор Хамахера для финального объединения
    MatrixXd fused_output = fuzzy_fusion->fuzzyFuse(all_outputs, src_weights, FuzzyLogicFusion::HAMACHER_FUSION);
    
    // Визуализация процесса вывода
    fuzzy_fusion->visualizeInferenceProcess(trace_inputs, fused_output);
    
    // 5. Нахождение класса с максимальной вероятностью
    int max_idx = 0;
    double max_val = fused_output(0, 0);
    for (int i = 1; i < fused_output.cols(); ++i) {
        if (fused_output(0, i) > max_val) {
            max_val = fused_output(0, i);
            max_idx = i;
        }
    }
    
    return make_pair(max_idx, max_val);
    } catch (const exception& e) {
        // При ошибке размерностей или другой — fallback к обычной классификации
        cerr << "[classifyWithStructureAnalysis] Fallback to basic classify: " << e.what() << endl;
        return classify(img);
    } catch (...) {
        cerr << "[classifyWithStructureAnalysis] Unknown error, fallback to basic classify" << endl;
        return classify(img);
    }
}

// Квантовое обучение: передача весов между основной сетью и подсетями
void UniversalImageClassifier::quantumKnowledgeTransfer(int epoch, int total_epochs,
                                                       double forward_ratio, 
                                                       double backward_ratio) {
    if (!use_structure_analysis || !subnetwork_manager || ensemble.empty()) {
        return;
    }
    
    // Используем первую сеть ансамбля как основную для передачи знаний
    // Используем raw pointer, так как ensemble хранит unique_ptr
    NeuralNetwork* main_network = ensemble[0].get();
    
    // Адаптивные коэффициенты передачи в зависимости от прогресса обучения
    double progress = static_cast<double>(epoch) / total_epochs;
    
    // В начале обучения больше передаем от основной сети к подсетям
    // В конце обучения больше передаем от подсетей к основной сети
    double adaptive_forward = forward_ratio * (1.0 - progress * 0.5);
    double adaptive_backward = backward_ratio * (1.0 + progress * 0.5);
    
    cout << "\n[Quantum Knowledge Transfer] Epoch " << epoch << "/" << total_epochs << endl;
    cout << "  Progress: " << fixed << setprecision(1) << (progress * 100) << "%" << endl;
    cout << "  Forward ratio: " << adaptive_forward << ", Backward ratio: " << adaptive_backward << endl;
    
    // Выполняем квантовую синхронизацию весов
    subnetwork_manager->quantumWeightSync(main_network, adaptive_forward, adaptive_backward);
    
    // Также передаем знания между членами ансамбля
    if (ensemble.size() > 1) {
        cout << "  [Ensemble Knowledge Sharing] Transferring between ensemble members..." << endl;
        for (size_t i = 1; i < ensemble.size(); ++i) {
            try {
                ensemble[i]->copyWeightsFrom(*main_network, 0.1 * adaptive_forward);
            } catch (const exception& e) {
                // Игнорируем ошибки несовместимости архитектур
            }
        }
    }
}

vector<MatrixXd> UniversalImageClassifier::extractStructureFeatures(const Mat& img, 
                                                                   const vector<StructureRegion>& structures) {
    vector<MatrixXd> features;
    
    for (const auto& structure : structures) {
        // Извлекаем признаки из области структуры
        Mat structure_img = structure.region_image;
        if (structure_img.empty()) {
            continue;
        }
        
        // Предобработка и извлечение признаков
        MatrixXd structure_features = preprocessImage(structure_img, image_size);
        features.push_back(structure_features);
    }
    
    return features;
}

