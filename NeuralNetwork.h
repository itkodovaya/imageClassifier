#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include "CudaAccelerator.h"

// Поддержка mixed precision: float для весов, double для критических вычислений
// Можно включить через определение USE_FLOAT_WEIGHTS
#ifdef USE_FLOAT_WEIGHTS
    using WeightType = float;
    using WeightMatrix = Eigen::MatrixXf;
    using WeightVector = Eigen::VectorXf;
#else
    using WeightType = double;
    using WeightMatrix = Eigen::MatrixXd;
    using WeightVector = Eigen::VectorXd;
#endif

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
using namespace std;

// Структура для статистики обучения
struct TrainingStats {
    int stage;
    int epoch;
    double accuracy;
    double loss;
    int samples_processed;
    int samples_accepted;
};

// Класс нейронной сети с многоэтапным обучением
class NeuralNetwork {
public:
    // Статический метод для автоматического определения архитектуры сети
    // на основе задачи (размер входа, количество классов, сложность)
    static vector<int> determineOptimalArchitecture(int input_size, int output_size, 
                                                   int num_samples = 1000);
    
    // Конструктор
    NeuralNetwork(int input_size, const vector<int>& hidden_sizes, int output_size,
                  double learning_rate = 0.01, const string& activation = "sigmoid",
                  double momentum = 0.9, double l2_reg = 0.0001);
    
    // Прямое распространение
    void forward(const MatrixXd& X, vector<MatrixXd>& activations, vector<MatrixXd>& z_values);
    
    // Обратное распространение
    void backward(const MatrixXd& X, const MatrixXd& y, 
                  const vector<MatrixXd>& activations, const vector<MatrixXd>& z_values,
                  vector<MatrixXd>& weight_gradients, vector<MatrixXd>& bias_gradients);
    
    // Обратное распространение с кросс-энтропийной потерей (для категорий)
    void backwardCrossEntropy(const MatrixXd& X, const MatrixXd& y, 
                              const vector<MatrixXd>& activations, const vector<MatrixXd>& z_values,
                              vector<MatrixXd>& weight_gradients, vector<MatrixXd>& bias_gradients,
                              const VectorXd& sample_weights = VectorXd());
    
    // Обновление весов
    void updateWeights(const vector<MatrixXd>& weight_gradients, 
                      const vector<MatrixXd>& bias_gradients);
    
    // Вычисление функции потерь
    double computeLoss(const MatrixXd& X, const MatrixXd& y);
    
    // Вычисление кросс-энтропийной потери для категорий (categorical cross-entropy)
    double computeCrossEntropyLoss(const MatrixXd& X, const MatrixXd& y);
    
    // Установка весов классов для взвешенной функции потерь
    void setClassWeights(const VectorXd& weights);
    
    // Метрики по классам (precision, recall, F1-score) - объявлено раньше для использования ниже
    struct ClassMetrics {
        int class_id;
        double precision;  // Точность (сколько из предсказанных правильные)
        double recall;     // Полнота (сколько из реальных найдены)
        double f1_score;   // F1-мера (гармоническое среднее precision и recall)
        int true_positives;
        int false_positives;
        int false_negatives;
    };
    
    // Адаптивная настройка весов классов на основе F1-score
    void updateClassWeightsFromF1(const vector<ClassMetrics>& metrics);
    
    // Focal Loss для фокуса на сложных примерах
    void setFocalLossParams(double gamma = 2.0, double alpha = 1.0, bool use_focal = false);
    double computeFocalLoss(const MatrixXd& X, const MatrixXd& y);
    
    // Label Smoothing
    void setLabelSmoothing(double smoothing = 0.1);
    MatrixXd applyLabelSmoothing(const MatrixXd& y_hard);
    
    // Dropout Regularization
    void setDropoutRate(double rate = 0.3);
    void setTrainingMode(bool training);
    MatrixXd applyDropout(const MatrixXd& x, double rate);
    
    // Adaptive Gradient Clipping
    void setAdaptiveGradientClipping(bool enable, double initial_threshold = 5.0);
    void applyAdaptiveGradientClipping(MatrixXd& grad, const MatrixXd& weights);
    
    // Вычисление точности
    double computeAccuracy(const MatrixXd& X, const MatrixXd& y);
    
    // Вычисление метрик по каждому классу
    vector<ClassMetrics> computeClassMetrics(const MatrixXd& X, const MatrixXd& y);
    
    // Вычисление weighted F1-score (взвешенного по количеству образцов)
    double computeWeightedF1Score(const MatrixXd& X, const MatrixXd& y);
    
    // Вычисление macro-averaged F1-score (более строгая метрика)
    double computeMacroAveragedF1Score(const MatrixXd& X, const MatrixXd& y);
    
    // Поиск оптимальных thresholds для каждого класса (ROC-анализ)
    VectorXd findOptimalThresholds(const MatrixXd& X_val, const MatrixXd& y_val);
    
    // Предсказание с использованием thresholds
    MatrixXd predictWithThresholds(const MatrixXd& X);
    
    // Этап 1: Контроль качества
    vector<int> stage1_quality_control(const MatrixXd& X, const MatrixXd& y, 
                                       double loss_threshold = 0.8,
                                       bool use_cross_entropy = false);
    
    // Этап 2: Батчевое обучение
    void stage2_batch_training(const MatrixXd& X, const MatrixXd& y,
                               int batch_size = 32, int epochs = 100,
                               int early_stopping_patience = 10,
                               double validation_split = 0.2,
                               bool use_cross_entropy = false,
                               const VectorXd& sample_weights = VectorXd());
    
    // Этап 3: Обучение с накоплением производных
    void stage3_accumulated_training(const MatrixXd& X, const MatrixXd& y,
                                     int batch_size = 32, int epochs = 50,
                                     bool use_cross_entropy = false,
                                     const VectorXd& sample_weights = VectorXd());
    
    // Этап 4: Специальное обучение для проблемных классов
    void stage4_problem_classes_training(const MatrixXd& X, const MatrixXd& y,
                                         const vector<int>& problem_classes,
                                         int batch_size = 32, int epochs = 20,
                                         double learning_rate_multiplier = 1.5,
                                         const VectorXd& sample_weights = VectorXd());
    
    // Полный цикл многоэтапного обучения
    double train_multi_stage(const MatrixXd& X, const MatrixXd& y,
                            double loss_threshold = 0.8,
                            int batch_size = 32,
                            int stage2_epochs = 100,
                            int stage3_epochs = 50,
                            bool use_stage3 = true,
                            int early_stopping_patience = 10,
                            bool use_cross_entropy = false,
                            const VectorXd& sample_weights = VectorXd());
    
    // Предсказание
    MatrixXd predict(const MatrixXd& X);
    
    // Методы для динамического морфинга топологии
    void addNeuronsToLayer(int layer_idx, int count);
    void addLayer(int neurons);
    
    // Сохранение/загрузка модели
    void saveModel(const string& filepath);
    void loadModel(const string& filepath);
    
    // Чтение метаданных (архитектура) из файла без загрузки весов
    static bool loadModelMetadata(const string& filepath, int& out_input_size, 
                                  vector<int>& out_hidden_sizes, int& out_output_size, double& out_lr);
    
    // Получение истории обучения (копия под мьютексом — потокобезопасно)
    vector<TrainingStats> getTrainingHistory() const;
    
    // Получение порогов и весов классов
    VectorXd getClassThresholds() const { return class_thresholds; }
    VectorXd getClassWeights() const { return class_weights; }
    
    // Сохранение порогов и весов классов
    void saveClassThresholds(const string& path) const;
    void saveClassWeights(const string& path) const;
    
    // Включение/выключение CUDA ускорения
    void enableCuda(bool enable) { use_cuda = enable && CudaAccelerator::isAvailable(); }
    bool isCudaEnabled() const { return use_cuda; }
    
    // Установка параметров Cosine Annealing
    void setCosineAnnealingParams(double T_0_param, double T_mult_param, double eta_min_param);
    
    // Установка learning rate (для применения из config)
    void setLearningRate(double lr);

    // Обновление Learning Rate по Cosine Annealing
    void updateLearningRateCosineAnnealing(int epoch, int total_epochs);
    
    // Knowledge Transfer: получение и установка весов для передачи знаний между сетями
    vector<MatrixXd> getWeights() const { return weights; }
    vector<MatrixXd> getBiases() const { 
        vector<MatrixXd> result;
        result.reserve(biases.size());
        for (const auto& bias : biases) {
            // Преобразуем VectorXd в MatrixXd (колонку размером Nx1)
            MatrixXd bias_matrix(bias.size(), 1);
            for (int j = 0; j < bias.size(); ++j) {
                bias_matrix(j, 0) = bias(j);
            }
            result.push_back(bias_matrix);
        }
        return result;
    }
    void setWeights(const vector<MatrixXd>& new_weights);
    void setBiases(const vector<MatrixXd>& new_biases);
    
    // Копирование весов из другой сети (с адаптацией размеров если нужно)
    void copyWeightsFrom(const NeuralNetwork& source, double transfer_ratio = 1.0);
    
    // Получение архитектуры сети
    vector<int> getArchitecture() const;
    
private:
    // Параметры сети
    int input_size;
    vector<int> hidden_sizes;
    int output_size;
    double learning_rate;
    double initial_learning_rate;
    string activation_type;
    double momentum;
    double l2_reg;
    
    // Веса и смещения
    vector<MatrixXd> weights;
    vector<MatrixXd> biases;
    
    // Momentum buffers
    vector<MatrixXd> velocity_w;
    vector<MatrixXd> velocity_b;
    
    // История обучения
    mutable mutex training_history_mutex;
    vector<TrainingStats> training_history;
    double best_accuracy;
    double best_f1_score;  // Лучший F1-score для отслеживания
    vector<MatrixXd> best_weights;
    vector<MatrixXd> best_biases;
    
    // Per-class thresholds для оптимизации precision/recall
    VectorXd class_thresholds;
    
    // Веса классов для балансировки (взвешенная функция потерь)
    VectorXd class_weights;
    
    // Параметры Focal Loss
    double focal_gamma;      // Focusing parameter (обычно 2.0)
    double focal_alpha;      // Balancing parameter (обычно 1.0)
    bool use_focal_loss;     // Флаг использования Focal Loss
    
    // Label Smoothing
    double label_smoothing;  // Параметр сглаживания меток (0.0 - без сглаживания, 0.1 - стандартное значение)
    
    // Dropout Regularization
    double dropout_rate;     // Вероятность отключения нейрона (0.0 - без dropout, 0.3-0.5 - стандартные значения)
    bool use_dropout;        // Флаг использования dropout (только во время обучения)
    bool training_mode;      // Режим обучения (true) или инференса (false)
    
    // Adaptive Gradient Clipping
    bool use_adaptive_clipping;  // Флаг использования адаптивного clipping
    double adaptive_clip_threshold;  // Адаптивный порог для clipping
    double gradient_norm_history;    // История норм градиентов для адаптации
    int gradient_norm_count;        // Счетчик для скользящего среднего
    
    // Cosine Annealing параметры
    bool use_cosine_annealing;   // Флаг использования cosine annealing
    double T_0;                  // Начальный период
    double T_mult;              // Множитель периода
    double eta_min;              // Минимальный learning rate
    int current_T;               // Текущий период
    int step_in_T;               // Шаг в текущем периоде
    
    // Вспомогательные функции
    MatrixXd activation(const MatrixXd& x);
    MatrixXd activation_derivative(const MatrixXd& x);
    void adjustLearningRate(double factor = 0.5);
    void initializeWeights();
    
    // Флаг использования CUDA
    bool use_cuda;
    
    // Гибридные функции для CPU+GPU вычислений
    MatrixXd hybridMatrixMultiply(const MatrixXd& A, const MatrixXd& B);
    MatrixXd hybridActivation(const MatrixXd& x);
    MatrixXd hybridActivationDerivative(const MatrixXd& x);
};

#endif // NEURAL_NETWORK_H

