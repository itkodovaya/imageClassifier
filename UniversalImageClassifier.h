#ifndef UNIVERSAL_IMAGE_CLASSIFIER_H
#define UNIVERSAL_IMAGE_CLASSIFIER_H

#include "NeuralNetwork.h"
#include "ShapeAnalyzer.h"
#include "SubNetworkManager.h"
#include "FuzzyLogicFusion.h"
#include "TopologicalSignature.h"
#include "HyperRelationalFuzzyGraph.h"
#include "TopologicalKernel.h"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>

// Поддержка DLL экспорта для компиляции с MVS (Microsoft Visual Studio)
#ifdef _MSC_VER
#ifdef BUILDING_DLL
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif
#else
#define DLLEXPORT
#endif

using namespace Eigen;
using namespace cv;
using namespace std;

// Универсальный классификатор изображений
class UniversalImageClassifier {
public:
    UniversalImageClassifier(int num_classes = 100, int target_size = 32);
    
    // Загрузка изображения из файла
    bool loadImage(const string& path);
    
    // Загрузка изображения из URL (через curl или прямой доступ)
    bool loadImageFromURL(const string& url);
    
    // Предобработка изображения
    MatrixXd preprocessImage(const Mat& img, int target_size = 64);
    
    // Классификация
    pair<int, double> classify(const Mat& img);
    
    // Классификация с возвратом top-K вероятностей (class_id, probability)
    vector<pair<int, double>> classifyTopK(const Mat& img, int k = 5);
    string getClassName(int class_id) const;
    
    // Обучение на наборе изображений
    void trainOnDataset(const vector<string>& image_paths, 
                       const vector<int>& labels,
                       int epochs = 200,
                       bool use_focal_loss = false,
                       bool use_oversampling = true,
                       bool use_extended_augmentation = true,
                       bool use_stage4_training = true,
                       bool use_label_smoothing = false,
                       bool use_dropout = false,
                       bool use_adaptive_clipping = false,
                       bool use_mixup = false,
                       bool use_cosine_annealing = false);
    
    // Сохранение/загрузка
    void saveModel(const string& path);
    void loadModel(const string& path);
    
    // Получение списка классов
    vector<string> getClassList() const;
    
    // Добавление нового класса
    void addClass(const string& class_name);
    
    // Установка имён классов по id (ordered_names[id] = name)
    void setClassNames(const vector<string>& names);
    
    // Установка learning rate для всех сетей ансамбля (применяется из config)
    void setLearningRate(double lr);

    // Включение/выключение анализа структуры и подсетей
    void enableStructureAnalysis(bool enable);
    bool isStructureAnalysisEnabled() const { return use_structure_analysis; }
    
    virtual ~UniversalImageClassifier(); // Добавляем деструктор для очистки
    
    // Классификация с анализом структуры (новая архитектура)
    pair<int, double> classifyWithStructureAnalysis(const Mat& img);
    
    // Оценка качества модели по классам (precision, recall, F1)
    void evaluateModel(const vector<string>& image_paths, const vector<int>& labels);
    
    // Сохранение статистики обучения в файл
    void saveTrainingStats(const string& path) const;
    
    // Получение данных обучения
    vector<int> getProblemClasses() const { return problem_classes; }
    vector<NeuralNetwork::ClassMetrics> getFinalClassMetrics() const;
    vector<TrainingStats> getTrainingHistory() const;
    
    // Сохранение всех данных обучения (автоматическое)
    void saveAllTrainingData(const string& base_path);
    void saveAllTrainingDataWithMetrics(const string& base_path,
                                       const vector<NeuralNetwork::ClassMetrics>& metrics,
                                       double overall_accuracy,
                                       double macro_f1,
                                       double weighted_f1,
                                       int epochs);
    void saveTrainingDataJSON(const string& json_path, 
                             const vector<NeuralNetwork::ClassMetrics>& metrics,
                             const vector<TrainingStats>& history,
                             double overall_accuracy,
                             double macro_f1,
                             double weighted_f1,
                             int epochs,
                             int batch_size,
                             double learning_rate,
                             int total_samples);
    void saveTrainingDataBinary(const string& bin_path,
                               const vector<NeuralNetwork::ClassMetrics>& metrics,
                               const vector<TrainingStats>& history);
    string getCurrentTimestamp() const; // YYYYMMDD_HHMMSS
    
    // Получение метрик по классам после обучения
    vector<NeuralNetwork::ClassMetrics> getClassMetrics(const vector<string>& image_paths, 
                                                        const vector<int>& labels);
    
    // Определение проблемных классов по метрикам
    vector<int> detectProblemClasses(const vector<NeuralNetwork::ClassMetrics>& metrics,
                                     double f1_threshold = 0.75,
                                     double recall_threshold = 0.70,
                                     double precision_threshold = 0.70);
    
    // Раннее определение проблемных классов после первых эпох обучения
    vector<int> detectProblemClassesEarly(const MatrixXd& X_val, const MatrixXd& y_val, 
                                         int min_epochs = 10);
    
    // Автоматическое определение классов с низкой точностью
    vector<int> detectLowAccuracyClasses(const vector<NeuralNetwork::ClassMetrics>& metrics,
                                        double accuracy_threshold = 0.85);
    
    // Запуск абляционного исследования для научной статьи
    void runAblationStudy(const vector<string>& image_paths, const vector<int>& labels);
    
    // Сжатие признаков через Информационное Бутылочное Горлышко (Information Bottleneck)
    MatrixXd applyInformationBottleneck(const MatrixXd& features, double beta = 0.5);
    
    // Создание единого вектора научных признаков (Fractal, Chaos, Cover)
    MatrixXd constructScientificFeatureVector(const Mat& region, const vector<Point>& contour);
    
    // Квантовое обучение: передача весов между основной сетью и подсетями
    void quantumKnowledgeTransfer(int epoch, int total_epochs, 
                                 double forward_ratio = 0.2, 
                                 double backward_ratio = 0.1);
    
private:
    unique_ptr<NeuralNetwork> network;
    map<int, string> class_names;
    int num_classes;
    int image_size;
    
    // Новая архитектура: анализ структуры и подсети
    bool use_structure_analysis;
    unique_ptr<ShapeAnalyzer> shape_analyzer;
    unique_ptr<SubNetworkManager> subnetwork_manager;
    unique_ptr<FuzzyLogicFusion> fuzzy_fusion;
    
    // Специализированный ансамбль для проблемных классов
    vector<unique_ptr<NeuralNetwork>> problem_ensemble;
    
    // Топологический анализ
    unique_ptr<TopologicalSignature> topological_sig;
    unique_ptr<TopologicalKernel> topological_kernel;
    bool use_topological_kernel;
    
    // Проблемные классы (определяются автоматически)
    vector<int> problem_classes;
    
    // Сохраненные метрики по классам после последнего обучения
    mutable vector<NeuralNetwork::ClassMetrics> last_class_metrics;
    
    // Уникальный алгоритм предобработки
    MatrixXd extractFeatures(const Mat& img);
    MatrixXd applyMultiScaleFeatures(const Mat& img);
    
    // Извлечение признаков из структур
    vector<MatrixXd> extractStructureFeatures(const Mat& img, const vector<StructureRegion>& structures);
    
    // Oversampling проблемных классов
    void oversampleProblemClasses(vector<MatrixXd>& images, vector<int>& labels,
                                   const vector<int>& problem_classes_list,
                                   double oversample_ratio = 1.5);
    
    // Расширенная augmentation для проблемных классов
    MatrixXd augmentProblemClassImage(const Mat& img);
    
    // Специализированная augmentation для птиц
    MatrixXd augmentBirdImage(const Mat& img);
    
    // Специализированная augmentation для двухбалочных самолетов
    MatrixXd augmentTwinTailImage(const Mat& img);
    
    // Определение типа проблемного класса
    bool isBirdClass(int class_id) const;
    bool isTwinTailClass(int class_id) const;
    
    // Mixup Data Augmentation
    void applyMixupAugmentation(vector<MatrixXd>& images, vector<int>& labels,
                                const vector<int>& problem_classes_list,
                                double mixup_alpha = 0.2, double mixup_prob = 0.5);
    
    // Поддержка SVM для классификации структур
    void trainSVM(const vector<MatrixXd>& features, const vector<int>& labels);
    int predictSVM(const MatrixXd& features);
    
private:
    vector<unique_ptr<NeuralNetwork>> ensemble; // Ансамбль нейронных сетей
    Ptr<cv::ml::SVM> svm_classifier; // SVM для сегментов/структур
};

#endif

