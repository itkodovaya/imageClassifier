#include "NeuralNetwork.h"
#include "CudaAccelerator.h"
#include "Profiler.h"
#include "BlasWrapper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <limits>
#include <thread>
#include <future>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Функция softmax для нормализации вероятностей
static MatrixXd softmax(const MatrixXd& x) {
    // Вычитаем максимум для численной стабильности
    MatrixXd max_vals = x.rowwise().maxCoeff();
    MatrixXd exp_x = (x.array() - max_vals.replicate(1, x.cols()).array()).exp();
    MatrixXd sum_exp = exp_x.rowwise().sum();
    return exp_x.cwiseQuotient(sum_exp.replicate(1, x.cols()));
}

// Автоматическое определение оптимальной архитектуры сети на основе задачи (улучшенная версия)
vector<int> NeuralNetwork::determineOptimalArchitecture(int input_size, int output_size, int num_samples) {
    vector<int> hidden_sizes;
    
    // Нормализация входных параметров
    input_size = max(input_size, 1);
    output_size = max(output_size, 1);
    num_samples = max(num_samples, 10);
    
    // 1. Оценка сложности задачи (многомерный анализ)
    double input_complexity = log10(static_cast<double>(input_size) + 1.0);
    double output_complexity = log10(static_cast<double>(output_size) + 1.0);
    double data_complexity = log10(static_cast<double>(num_samples) + 1.0);
    double overall_complexity = (input_complexity + output_complexity + data_complexity) / 3.0;
    
    // 2. Оценка соотношения данных к параметрам (data-to-parameters ratio)
    // Важно для предотвращения переобучения
    double estimated_params_per_layer = input_size * 0.5;  // Примерная оценка
    double data_to_params_ratio = static_cast<double>(num_samples) / max(estimated_params_per_layer, 1.0);
    
    // 3. Определение типа задачи
    bool is_image_task = (input_size > 1000 && input_size % 3 == 0);  // RGB изображения
    bool is_high_dimensional = (input_size > 500);
    bool is_multi_class = (output_size > 10);
    bool is_binary = (output_size == 2);
    
    // 4. Определение количества слоев на основе улучшенных эвристик
    int num_layers;
    
    // Базовое количество слоев на основе общей сложности
    if (overall_complexity < 2.5) {
        num_layers = 2;  // Очень простая задача
    } else if (overall_complexity < 3.5) {
        num_layers = 2 + (is_multi_class ? 1 : 0);  // Простая задача
    } else if (overall_complexity < 4.5) {
        num_layers = 3 + (is_high_dimensional ? 1 : 0);  // Средняя сложность
    } else if (overall_complexity < 5.5) {
        num_layers = 4 + (is_image_task ? 1 : 0);  // Высокая сложность
    } else {
        num_layers = 5 + (is_image_task && is_multi_class ? 1 : 0);  // Очень сложная задача
    }
    
    // Корректировка на основе соотношения данных к параметрам
    if (data_to_params_ratio < 5.0) {
        // Мало данных - уменьшаем количество слоев для предотвращения переобучения
        num_layers = max(2, num_layers - 1);
    } else if (data_to_params_ratio > 50.0) {
        // Много данных - можно увеличить сложность
        num_layers = min(7, num_layers + 1);
    }
    
    // Ограничение: минимум 2, максимум 7 слоев
    num_layers = max(2, min(num_layers, 7));
    
    // 5. Определение размеров слоев с улучшенными эвристиками
    
    // Базовые размеры на основе типа задачи
    int first_layer_size, last_layer_size;
    
    if (is_image_task) {
        // Для изображений: первый слой должен быть достаточно большим для извлечения признаков
        // Используем правило: первый слой = min(input_size/4, 2048, но не меньше 256)
        first_layer_size = min(max(input_size / 4, 256), 2048);
        // Последний слой: достаточно большой для классификации, но не слишком
        last_layer_size = max(output_size * 4, min(128, output_size * 8));
    } else if (is_high_dimensional) {
        // Высокомерные данные: постепенное сжатие
        first_layer_size = min(input_size / 2, 1024);
        last_layer_size = max(output_size * 3, 64);
    } else if (input_size < 100) {
        // Низкомерные данные: компактная сеть
        first_layer_size = max(input_size * 2, output_size * 4);
        last_layer_size = max(output_size * 2, 32);
        num_layers = min(num_layers, 3);  // Ограничиваем глубину
    } else {
        // Средние размеры: адаптивная архитектура
        first_layer_size = min(input_size / 2, 512);
        last_layer_size = max(output_size * 2, 32);
    }
    
    // Корректировка на основе количества данных
    if (data_to_params_ratio < 3.0) {
        // Мало данных - уменьшаем размеры слоев
        first_layer_size = static_cast<int>(first_layer_size * 0.7);
        last_layer_size = static_cast<int>(last_layer_size * 0.8);
    } else if (data_to_params_ratio > 30.0) {
        // Много данных - можно увеличить
        first_layer_size = min(static_cast<int>(first_layer_size * 1.3), 2048);
        last_layer_size = static_cast<int>(last_layer_size * 1.2);
    }
    
    // 6. Генерация размеров слоев с улучшенной прогрессией
    if (num_layers == 1) {
        // Один слой: среднее между первым и последним
        hidden_sizes.push_back((first_layer_size + last_layer_size) / 2);
    } else {
        // Используем комбинированную прогрессию: геометрическую с коррекцией
        
        // Вычисляем оптимальный коэффициент для геометрической прогрессии
        double geometric_ratio = pow(static_cast<double>(last_layer_size) / first_layer_size, 
                                     1.0 / (num_layers - 1));
        
        // Для первых слоев используем более медленное уменьшение (больше емкости)
        // Для последних слоев - более быстрое (компрессия признаков)
        for (int i = 0; i < num_layers; ++i) {
            double progress = static_cast<double>(i) / (num_layers - 1);
            
            // Комбинированная прогрессия: геометрическая с коррекцией
            // Используем нелинейную функцию для более плавного перехода
            double non_linear_factor = 1.0 - pow(progress, 1.5);  // Более медленное уменьшение в начале
            int size = static_cast<int>(first_layer_size * pow(geometric_ratio, i) * 
                                       (1.0 + 0.2 * non_linear_factor));
            
            // Округляем до оптимальных значений для производительности
            if (size > 256) {
                size = ((size + 32) / 64) * 64;  // Кратно 64 для больших слоев
            } else if (size > 128) {
                size = ((size + 16) / 32) * 32;  // Кратно 32 для средних
            } else {
                size = ((size + 8) / 16) * 16;   // Кратно 16 для малых
            }
            
            // Обеспечиваем монотонное уменьшение
            if (i > 0 && size >= hidden_sizes.back()) {
                size = max(hidden_sizes.back() - 32, last_layer_size);
            }
            
            hidden_sizes.push_back(size);
        }
    }
    
    // 7. Финальная корректировка на основе специфики задачи
    
    // Корректировка на основе количества классов (более точная)
    if (!hidden_sizes.empty()) {
        if (output_size > 100) {
            // Много классов - увеличиваем емкость, особенно последних слоев
            for (size_t i = 0; i < hidden_sizes.size(); ++i) {
                double factor = 1.0 + 0.15 * (1.0 - static_cast<double>(i) / hidden_sizes.size());
                hidden_sizes[i] = static_cast<int>(hidden_sizes[i] * factor);
            }
        } else if (output_size < 5 && !is_binary) {
            // Мало классов - можно уменьшить, но не для бинарной классификации
            for (size_t i = 0; i < hidden_sizes.size(); ++i) {
                hidden_sizes[i] = static_cast<int>(hidden_sizes[i] * 0.85);
            }
        }
    }
    
    // 8. Применение ограничений и оптимизация
    
    // Ограничения: минимум 16, максимум 4096 нейронов на слой
    // Но для больших задач можно увеличить максимум
    int max_neurons = (overall_complexity > 5.0) ? 8192 : 4096;
    max_neurons = (is_image_task && num_samples > 10000) ? 8192 : max_neurons;
    
    for (size_t i = 0; i < hidden_sizes.size(); ++i) {
        hidden_sizes[i] = max(16, min(hidden_sizes[i], max_neurons));
    }
    
    // 9. Удаление дубликатов и обеспечение монотонности
    // Сортируем по убыванию (от большего к меньшему)
    sort(hidden_sizes.begin(), hidden_sizes.end(), greater<int>());
    
    // Удаляем дубликаты и слишком близкие значения
    vector<int> unique_sizes;
    for (size_t i = 0; i < hidden_sizes.size(); ++i) {
        if (unique_sizes.empty() || 
            abs(unique_sizes.back() - hidden_sizes[i]) > unique_sizes.back() * 0.1) {
            unique_sizes.push_back(hidden_sizes[i]);
        }
    }
    hidden_sizes = unique_sizes;
    
    // 10. Финальная проверка и коррекция
    if (hidden_sizes.empty()) {
        // Резервный вариант: простая архитектура
        int default_size = max(input_size / 4, output_size * 2);
        default_size = max(32, min(default_size, 512));
        hidden_sizes.push_back(default_size);
    }
    
    // Обеспечиваем, что первый слой не слишком мал, а последний не слишком велик
    if (hidden_sizes[0] < input_size / 8) {
        hidden_sizes[0] = max(hidden_sizes[0], input_size / 8);
    }
    if (hidden_sizes.back() > output_size * 10) {
        hidden_sizes.back() = min(hidden_sizes.back(), output_size * 10);
    }
    
    // Убеждаемся, что архитектура монотонно убывает
    for (size_t i = 1; i < hidden_sizes.size(); ++i) {
        if (hidden_sizes[i] >= hidden_sizes[i-1]) {
            hidden_sizes[i] = max(hidden_sizes[i-1] - 32, last_layer_size);
        }
    }
    
    return hidden_sizes;
}

// Конструктор
NeuralNetwork::NeuralNetwork(int input_size, const vector<int>& hidden_sizes, int output_size,
                             double learning_rate, const string& activation,
                             double momentum, double l2_reg)
    : input_size(input_size), hidden_sizes(hidden_sizes), output_size(output_size),
      learning_rate(learning_rate), initial_learning_rate(learning_rate),
      activation_type(activation), momentum(momentum), l2_reg(l2_reg),
      best_accuracy(0.0), best_f1_score(0.0), use_cuda(false),
      focal_gamma(2.0), focal_alpha(1.0), use_focal_loss(false),
      label_smoothing(0.0), dropout_rate(0.0), use_dropout(false), training_mode(true),
      use_adaptive_clipping(false), adaptive_clip_threshold(5.0),
      gradient_norm_history(0.0), gradient_norm_count(0),
      use_cosine_annealing(false), T_0(10.0), T_mult(2.0), eta_min(0.0),
      current_T(0), step_in_T(0) {
    // Инициализация весов классов (по умолчанию все равны 1.0)
    class_weights = VectorXd::Ones(output_size);
    
    // Инициализация thresholds (по умолчанию 0.5 для всех классов)
    class_thresholds = VectorXd::Constant(output_size, 0.5);
    
    // Инициализируем BLAS для оптимизации матричных операций
    static bool blas_initialized = false;
    if (!blas_initialized) {
        BlasWrapper::initialize();
        blas_initialized = true;
    }
    
    // Инициализируем CUDA если доступна
    CudaAccelerator::initialize();
    use_cuda = CudaAccelerator::isAvailable();
    if (use_cuda) {
        cout << "[NeuralNetwork] CUDA acceleration enabled" << endl;
    }
    initializeWeights();
}

// Инициализация весов
void NeuralNetwork::initializeWeights() {
    random_device rd;
    mt19937 gen(rd());
    
    vector<int> layer_sizes;
    layer_sizes.push_back(input_size);
    layer_sizes.insert(layer_sizes.end(), hidden_sizes.begin(), hidden_sizes.end());
    layer_sizes.push_back(output_size);
    
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        double limit;
        if (activation_type == "relu") {
            limit = sqrt(2.0 / layer_sizes[i]);
        } else {
            limit = sqrt(6.0 / (layer_sizes[i] + layer_sizes[i + 1]));
        }
        
        normal_distribution<double> dist(0.0, limit);
        MatrixXd w = MatrixXd::NullaryExpr(layer_sizes[i + 1], layer_sizes[i],
            [&]() { return dist(gen); });
        VectorXd b = VectorXd::Zero(layer_sizes[i + 1]);
        
        weights.push_back(w);
        biases.push_back(b);
        velocity_w.push_back(MatrixXd::Zero(w.rows(), w.cols()));
        velocity_b.push_back(VectorXd::Zero(b.size()));
    }
}

// Функция активации
MatrixXd NeuralNetwork::activation(const MatrixXd& x) {
    if (activation_type == "sigmoid") {
        return (1.0 / (1.0 + (-x.array()).exp())).matrix();
    } else if (activation_type == "relu") {
        return x.cwiseMax(0.0);
    } else if (activation_type == "tanh") {
        return x.array().tanh().matrix();
    }
    return x;
}

// Производная функции активации
MatrixXd NeuralNetwork::activation_derivative(const MatrixXd& x) {
    if (activation_type == "sigmoid") {
        MatrixXd s = activation(x);
        return (s.array() * (1.0 - s.array())).matrix();
    } else if (activation_type == "relu") {
        return (x.array() > 0.0).cast<double>().matrix();
    } else if (activation_type == "tanh") {
        MatrixXd t = x.array().tanh().matrix();
        return (1.0 - t.array().square()).matrix();
    }
    return MatrixXd::Ones(x.rows(), x.cols());
}

// Гибридное матричное умножение (CPU + GPU + BLAS)
MatrixXd NeuralNetwork::hybridMatrixMultiply(const MatrixXd& A, const MatrixXd& B) {
    PROFILE_SCOPE("NeuralNetwork::hybridMatrixMultiply");
    
    // GPU выгоден при больших матрицах: много строк ИЛИ много элементов (cols)
    if (use_cuda && (A.rows() > 20 || A.cols() > 500)) {
        // Используем CUDA для больших матриц (через cuBLAS, если доступен)
        return CudaAccelerator::matrixMultiplyGPU(A, B);
    }
    
    // Используем BLAS если доступен для оптимизации матричных операций
    if (BlasWrapper::isAvailable() && A.rows() >= BlasWrapper::BLAS_THRESHOLD) {
        return BlasWrapper::gemm(A, B, 1.0, 0.0);
    }
    
    // Для маленьких матриц или если BLAS недоступен - используем Eigen
    return A * B;
}

// Гибридная активация
MatrixXd NeuralNetwork::hybridActivation(const MatrixXd& x) {
    if (use_cuda && (x.rows() > 5 || x.rows() * x.cols() > 500)) {
        if (activation_type == "relu") {
            return CudaAccelerator::reluGPU(x);
        }
    }
    return activation(x);
}

// Гибридная производная активации
MatrixXd NeuralNetwork::hybridActivationDerivative(const MatrixXd& x) {
    if (use_cuda && (x.rows() > 5 || x.rows() * x.cols() > 500)) {
        if (activation_type == "relu") {
            return CudaAccelerator::reluDerivativeGPU(x);
        }
    }
    return activation_derivative(x);
}

// Прямое распространение с гибридным ускорением
// Оптимизированная версия с векторизованными операциями и предварительным выделением памяти
void NeuralNetwork::forward(const MatrixXd& X, vector<MatrixXd>& activations, vector<MatrixXd>& z_values)
{
    PROFILE_FUNCTION();
    
    // Проверка входных данных
    if (X.rows() == 0 || X.cols() != input_size) {
        throw invalid_argument("NeuralNetwork::forward: Invalid input dimensions");
    }
    
    // Предварительное выделение памяти для улучшения производительности
    activations.clear();
    z_values.clear();
    activations.reserve(weights.size() + 1);
    z_values.reserve(weights.size());
    activations.push_back(X);
    
    MatrixXd current_input = X;
    for (size_t i = 0; i < weights.size(); ++i) {
        // Используем гибридное умножение (CPU + GPU)
        MatrixXd z = hybridMatrixMultiply(current_input, weights[i].transpose());
        
        // Добавляем смещение к каждой строке (векторизованная операция)
        // Используем rowwise() для эффективного добавления bias ко всем строкам одновременно
        for (int row = 0; row < z.rows(); ++row) {
            z.row(row) += biases[i].transpose();
        }
        
        z_values.push_back(z);
        
        // Используем гибридную активацию
        MatrixXd a = hybridActivation(z);
        activations.push_back(a);
        current_input = a;
    }
}

// Обратное распространение
void NeuralNetwork::backward(const MatrixXd& X, const MatrixXd& y,
                            const vector<MatrixXd>& activations, const vector<MatrixXd>& z_values,
                            vector<MatrixXd>& weight_gradients, vector<MatrixXd>& bias_gradients) {
    PROFILE_FUNCTION();
    
    int m = X.rows();
    
    // Предварительное выделение памяти
    weight_gradients.clear();
    bias_gradients.clear();
    weight_gradients.reserve(weights.size());
    bias_gradients.reserve(biases.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        weight_gradients.push_back(MatrixXd::Zero(weights[i].rows(), weights[i].cols()));
        bias_gradients.push_back(MatrixXd::Zero(biases[i].rows(), 1));
    }
    
    // Ошибка выходного слоя
    MatrixXd delta = activations.back() - y;
    
    // Обратное распространение
    // Предварительно выделяем память для промежуточных вычислений
    for (int i = static_cast<int>(weights.size()) - 1; i >= 0; --i) {
        // Градиенты для текущего слоя (используем noalias для избежания временных копий)
        weight_gradients[i].noalias() = (delta.transpose() * activations[i]) / m;
        bias_gradients[i] = delta.colwise().mean().transpose();
        
        // Распространение ошибки на предыдущий слой (гибридное)
        if (i > 0) {
            MatrixXd deriv = hybridActivationDerivative(z_values[i - 1]);
            MatrixXd delta_weights = hybridMatrixMultiply(delta, weights[i]);
            delta = delta_weights.cwiseProduct(deriv);
        }
    }
}

// Обратное распространение с кросс-энтропийной потерей (для категорий)
void NeuralNetwork::backwardCrossEntropy(const MatrixXd& X, const MatrixXd& y,
                                        const vector<MatrixXd>& activations, const vector<MatrixXd>& z_values,
                                        vector<MatrixXd>& weight_gradients, vector<MatrixXd>& bias_gradients,
                                        const VectorXd& sample_weights) {
    int m = X.rows();
    
    weight_gradients.clear();
    bias_gradients.clear();
    for (size_t i = 0; i < weights.size(); ++i) {
        weight_gradients.push_back(MatrixXd::Zero(weights[i].rows(), weights[i].cols()));
        bias_gradients.push_back(MatrixXd::Zero(biases[i].rows(), 1));
    }
    
    // Для кросс-энтропии с softmax градиент выходного слоя = probs - y_true
    // где probs = softmax(logits)
    MatrixXd logits = activations.back();
    MatrixXd probs = softmax(logits);
    
    // Добавляем epsilon для численной стабильности
    const double epsilon = 1e-15;
    probs = probs.array().max(epsilon).min(1.0 - epsilon);
    
    // Применяем Label Smoothing к меткам
    MatrixXd y_smooth = applyLabelSmoothing(y);
    
    // Ошибка выходного слоя: градиент кросс-энтропии после softmax
    MatrixXd delta = (probs - y_smooth) / m;  // Деление на m для усреднения по батчу
    
    // Применяем Focal Loss модификацию, если включена
    if (use_focal_loss) {
        for (int i = 0; i < m; ++i) {
            // Находим истинный класс
            int true_class = -1;
            for (int j = 0; j < output_size; ++j) {
                if (y(i, j) > 0.5) {
                    true_class = j;
                    break;
                }
            }
            
            if (true_class >= 0 && true_class < output_size) {
                double p_t = probs(i, true_class);
                // Модифицирующий фактор Focal Loss: (1-p_t)^γ
                double modulating_factor = pow(1.0 - p_t, focal_gamma);
                
                // Применяем модификацию к градиентам
                // Для правильного класса: delta уменьшается на (1-p_t)^γ
                // Для неправильных классов: delta увеличивается
                for (int j = 0; j < output_size; ++j) {
                    if (j == true_class) {
                        // Для правильного класса: фокус на сложных примерах
                        delta(i, j) *= focal_alpha * modulating_factor;
                    } else {
                        // Для неправильных классов: стандартный градиент
                        delta(i, j) *= (1.0 - focal_alpha * modulating_factor);
                    }
                }
            }
        }
    }
    
    // Применяем веса образцов и классов, если они предоставлены
    if (sample_weights.size() == m) {
        for (int i = 0; i < m; ++i) {
            delta.row(i) *= sample_weights(i);
        }
    }
    
    // Применяем веса классов к градиентам
    if (class_weights.size() == output_size) {
        for (int i = 0; i < m; ++i) {
            // Находим истинный класс для этого образца
            int true_class = -1;
            for (int j = 0; j < output_size; ++j) {
                if (y(i, j) > 0.5) {
                    true_class = j;
                    break;
                }
            }
            if (true_class >= 0 && true_class < class_weights.size()) {
                delta.row(i) *= class_weights(true_class);
            }
        }
    }
    
    // Обратное распространение
    for (int i = static_cast<int>(weights.size()) - 1; i >= 0; --i) {
        // Градиенты для текущего слоя
        weight_gradients[i] = delta.transpose() * activations[i];
        bias_gradients[i] = delta.colwise().sum().transpose();
        
        // Распространение ошибки на предыдущий слой (гибридное)
        if (i > 0) {
            MatrixXd deriv = hybridActivationDerivative(z_values[i - 1]);
            MatrixXd delta_weights = hybridMatrixMultiply(delta, weights[i]);
            delta = delta_weights.cwiseProduct(deriv);
        }
    }
}

// Обновление весов (улучшенная версия)
void NeuralNetwork::updateWeights(const vector<MatrixXd>& weight_gradients,
                                  const vector<MatrixXd>& bias_gradients) {
    // Параметры для улучшенного обучения
    const double gradient_clip_threshold = 2.0;  // Stricter to prevent explosion
    const double weight_decay_factor = 0.0001;   // Дополнительный weight decay
    const double min_weight_value = -10.0;        // Минимальное значение веса
    const double max_weight_value = 10.0;        // Максимальное значение веса
    
    for (size_t i = 0; i < weights.size(); ++i) {
        // 1. Gradient clipping для предотвращения взрывающихся градиентов
        MatrixXd w_grad = weight_gradients[i];
        MatrixXd b_grad = bias_gradients[i];
        
        // Применяем адаптивный gradient clipping, если включен
        if (use_adaptive_clipping) {
            applyAdaptiveGradientClipping(w_grad, weights[i]);
            // Для bias используем тот же порог, но без учета весов
            double b_grad_norm = b_grad.norm();
            if (b_grad_norm > adaptive_clip_threshold) {
                b_grad = (b_grad / b_grad_norm) * adaptive_clip_threshold;
            }
        } else {
            // Стандартный gradient clipping
            double w_grad_norm = w_grad.norm();
            double b_grad_norm = b_grad.norm();
            
            // Обрезаем градиенты, если они слишком большие
            if (w_grad_norm > gradient_clip_threshold) {
                w_grad = (w_grad / w_grad_norm) * gradient_clip_threshold;
            }
            if (b_grad_norm > gradient_clip_threshold) {
                b_grad = (b_grad / b_grad_norm) * gradient_clip_threshold;
            }
        }
        
        // 2. Добавление L2 регуляризации с оптимизацией
        w_grad.noalias() += l2_reg * weights[i];
        
        // 3. Дополнительный weight decay для лучшей регуляризации
        // Используем знак весов для L1-подобной регуляризации
        // sign(x) = 1 если x > 0, -1 если x < 0, 0 если x == 0
        MatrixXd weight_sign = weights[i].unaryExpr([](double x) {
            return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
        });
        w_grad.noalias() += weight_decay_factor * weight_sign;
        
        // 4. Обновление velocity с momentum (оптимизированное)
        velocity_w[i] = momentum * velocity_w[i] - learning_rate * w_grad;
        velocity_b[i] = momentum * velocity_b[i] - learning_rate * b_grad;
        
        // 5. Проверка на NaN и Inf в velocity
        bool has_nan_w = !velocity_w[i].array().isFinite().all();
        bool has_nan_b = !velocity_b[i].array().isFinite().all();
        
        if (has_nan_w || has_nan_b) {
            // Если обнаружены NaN/Inf, сбрасываем velocity для этого слоя
            if (has_nan_w) {
                velocity_w[i].setZero();
            }
            if (has_nan_b) {
                velocity_b[i].setZero();
            }
            continue; // Пропускаем обновление весов для этого слоя
        }
        
        // 6. Обновление весов с оптимизацией (noalias для избежания временных копий)
        weights[i].noalias() += velocity_w[i];
        biases[i].noalias() += velocity_b[i];
        
        // 7. Ограничение весов для предотвращения переобучения
        weights[i] = weights[i].cwiseMax(min_weight_value).cwiseMin(max_weight_value);
        biases[i] = biases[i].cwiseMax(min_weight_value).cwiseMin(max_weight_value);
        
        // 8. Финальная проверка на NaN/Inf в весах
        bool has_nan_weights = !weights[i].array().isFinite().all();
        bool has_nan_biases = !biases[i].array().isFinite().all();
        
        if (has_nan_weights || has_nan_biases) {
            // Если веса стали некорректными, инициализируем их заново
            random_device rd;
            mt19937 gen(rd());
            double limit = sqrt(2.0 / weights[i].cols());
            normal_distribution<double> dist(0.0, limit);
            weights[i] = MatrixXd::NullaryExpr(weights[i].rows(), weights[i].cols(),
                [&]() { return dist(gen); });
            biases[i].setZero();
            velocity_w[i].setZero();
            velocity_b[i].setZero();
        }
    }
}

// Вычисление функции потерь
double NeuralNetwork::computeLoss(const MatrixXd& X, const MatrixXd& y) {
    vector<MatrixXd> activations, z_values;
    forward(X, activations, z_values);
    MatrixXd predictions = activations.back();
    return (predictions - y).array().square().mean();
}

// Вычисление кросс-энтропийной потери для категорий
double NeuralNetwork::computeCrossEntropyLoss(const MatrixXd& X, const MatrixXd& y) {
    vector<MatrixXd> activations, z_values;
    forward(X, activations, z_values);
    MatrixXd logits = activations.back();
    
    // Применяем softmax для получения вероятностей
    MatrixXd probs = softmax(logits);
    
    // Добавляем epsilon для численной стабильности (избегаем log(0))
    const double epsilon = 1e-15;
    probs = probs.array().max(epsilon).min(1.0 - epsilon);
    
    // Применяем Label Smoothing к меткам
    MatrixXd y_smooth = applyLabelSmoothing(y);
    
    // Вычисляем потерю: либо Focal Loss, либо стандартная кросс-энтропия
    MatrixXd loss_per_sample = MatrixXd::Zero(X.rows(), 1);
    
    if (use_focal_loss) {
        // Используем Focal Loss
        for (int i = 0; i < X.rows(); ++i) {
            int true_class = -1;
            for (int j = 0; j < output_size; ++j) {
                if (y(i, j) > 0.5) {
                    true_class = j;
                    break;
                }
            }
            
            if (true_class >= 0 && true_class < output_size) {
                double p_t = probs(i, true_class);
                double modulating_factor = pow(1.0 - p_t, focal_gamma);
                double ce_loss = -log(p_t);
                double focal_loss = focal_alpha * modulating_factor * ce_loss;
                
                // Применяем веса классов
                if (class_weights.size() == output_size) {
                    focal_loss *= class_weights(true_class);
                }
                
                loss_per_sample(i, 0) = focal_loss;
            }
        }
    } else {
        // Стандартная кросс-энтропийная потеря с Label Smoothing
        MatrixXd log_probs = probs.array().log();
        loss_per_sample = -(y_smooth.array() * log_probs.array()).rowwise().sum();
        
        // Применяем веса классов, если они установлены
        if (class_weights.size() == output_size) {
            for (int i = 0; i < X.rows(); ++i) {
                int true_class = -1;
                for (int j = 0; j < output_size; ++j) {
                    if (y(i, j) > 0.5) {
                        true_class = j;
                        break;
                    }
                }
                if (true_class >= 0 && true_class < class_weights.size()) {
                    loss_per_sample(i, 0) *= class_weights(true_class);
                }
            }
        }
    }
    
    return loss_per_sample.mean();
}

void NeuralNetwork::setClassWeights(const VectorXd& weights) {
    if (weights.size() == output_size) {
        class_weights = weights;
        cout << "[NeuralNetwork] Class weights set for weighted loss function" << endl;
    } else {
            cerr << "[NeuralNetwork] Warning: weights size (" << weights.size()
             << ") не соответствует количеству классов (" << output_size << ")" << endl;
    }
}

// Адаптивная настройка весов классов на основе F1-score
void NeuralNetwork::updateClassWeightsFromF1(const vector<ClassMetrics>& metrics) {
    if (metrics.size() != static_cast<size_t>(output_size)) {
        return;  // Несоответствие размеров
    }
    
    // Сохраняем базовые веса (если еще не установлены, используем единичные)
    static VectorXd base_weights = class_weights;
    if (base_weights.size() != output_size) {
        base_weights = VectorXd::Ones(output_size);
    }
    
    // Обновляем веса на основе F1-score
    for (int i = 0; i < output_size; ++i) {
        double f1 = metrics[i].f1_score;
        
        // Формула: weight = base_weight * (1.0 / max(f1_score, 0.1))
        // Более агрессивные веса для классов с низким F1-score
        double f1_boost = 1.0 / max(f1, 0.1);
        
        // Дополнительный множитель для проблемных классов (F1 < 0.7)
        double problem_class_multiplier = (f1 < 0.7) ? 1.5 : 1.0;
        
        // Ограничиваем максимальный вес (чтобы избежать экстремальных значений)
        double new_weight = base_weights(i) * f1_boost * problem_class_multiplier;
        class_weights(i) = min(new_weight, 20.0);  // Максимум 20.0
    }
}

// Установка параметров Focal Loss
void NeuralNetwork::setFocalLossParams(double gamma, double alpha, bool use_focal) {
    focal_gamma = max(0.0, gamma);  // gamma >= 0
    focal_alpha = max(0.0, min(1.0, alpha));  // alpha в [0, 1]
    use_focal_loss = use_focal;
    if (use_focal_loss) {
        cout << "[NeuralNetwork] Focal Loss enabled: gamma=" << focal_gamma 
             << ", alpha=" << focal_alpha << endl;
    }
}

// Установка Label Smoothing
void NeuralNetwork::setLabelSmoothing(double smoothing) {
    label_smoothing = max(0.0, min(1.0, smoothing)); // Ограничиваем в [0, 1]
    if (label_smoothing > 0.0) {
        cout << "[NeuralNetwork] Label Smoothing enabled: ε=" << label_smoothing << endl;
    }
}

// Установка Dropout Rate
void NeuralNetwork::setDropoutRate(double rate) {
    dropout_rate = max(0.0, min(1.0, rate)); // Ограничиваем в [0, 1]
    use_dropout = (dropout_rate > 0.0);
    if (use_dropout) {
        cout << "[NeuralNetwork] Dropout enabled: rate=" << dropout_rate << endl;
    }
}

// Установка режима обучения
void NeuralNetwork::setTrainingMode(bool training) {
    training_mode = training;
}

// Применение Dropout (оптимизированная векторизованная версия)
// Использует векторизованные Eigen операции вместо двойного цикла для значительного ускорения
MatrixXd NeuralNetwork::applyDropout(const MatrixXd& x, double rate) {
    // Проверка входных данных
    if (x.rows() == 0 || x.cols() == 0) {
        return x;
    }
    
    if (rate < 0.0 || rate >= 1.0) {
        throw invalid_argument("NeuralNetwork::applyDropout: rate must be in [0, 1)");
    }
    
    if (!training_mode || rate <= 0.0) {
        // Во время инференса масштабируем веса
        if (!training_mode && rate > 0.0) {
            return x * (1.0 - rate);
        }
        return x;
    }
    
    // Во время обучения случайно отключаем нейроны
    // Векторизованная реализация с использованием Eigen операций
    static thread_local random_device rd;
    static thread_local mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.0, 1.0);
    
    // Генерируем маску случайных значений для всего матричного блока
    MatrixXd mask = MatrixXd::NullaryExpr(x.rows(), x.cols(), 
        [&]() { return dist(gen); });
    
    // Векторизованное применение dropout: маска > rate означает сохранить нейрон
    // Используем cwiseProduct для поэлементного умножения
    MatrixXd scale_factor = (mask.array() > rate).cast<double>() / (1.0 - rate);
    MatrixXd dropped = x.cwiseProduct(scale_factor);
    
    return dropped;
}

// Установка Adaptive Gradient Clipping
void NeuralNetwork::setAdaptiveGradientClipping(bool enable, double initial_threshold) {
    use_adaptive_clipping = enable;
    adaptive_clip_threshold = initial_threshold;
    gradient_norm_history = 0.0;
    gradient_norm_count = 0;
    if (enable) {
        cout << "[NeuralNetwork] Adaptive Gradient Clipping enabled: initial_threshold=" 
             << initial_threshold << endl;
    }
}

// Применение Adaptive Gradient Clipping (оптимизированная версия с кэшированием норм)
void NeuralNetwork::applyAdaptiveGradientClipping(MatrixXd& grad, const MatrixXd& weights) {
    if (!use_adaptive_clipping) {
        return; // Адаптивный clipping отключен
    }
    
    // Вычисляем норму градиента (кэшируем для повторного использования)
    double grad_norm = grad.norm();
    
    // Кэшируем норму весов (вычисляем только при первом вызове или если веса изменились)
    static double cached_weight_norm = -1.0;
    static bool weight_norm_valid = false;
    
    // Проверяем, изменились ли веса (упрощенная проверка по размеру)
    static int cached_weight_rows = -1;
    static int cached_weight_cols = -1;
    
    if (weights.rows() != cached_weight_rows || weights.cols() != cached_weight_cols || !weight_norm_valid) {
        cached_weight_norm = weights.norm();
        cached_weight_rows = weights.rows();
        cached_weight_cols = weights.cols();
        weight_norm_valid = true;
    }
    double weight_norm = cached_weight_norm;
    
    // Обновляем историю норм градиентов (скользящее среднее)
    if (gradient_norm_count == 0) {
        gradient_norm_history = grad_norm;
    } else {
        // Экспоненциальное скользящее среднее
        double alpha = 0.1; // Коэффициент сглаживания
        gradient_norm_history = alpha * grad_norm + (1.0 - alpha) * gradient_norm_history;
    }
    gradient_norm_count++;
    
    // Адаптивный порог на основе нормы весов и истории градиентов
    // Формула: threshold = max(initial_threshold, weight_norm * factor)
    double weight_factor = 0.01; // Фактор для нормы весов
    double adaptive_threshold = max(adaptive_clip_threshold, weight_norm * weight_factor);
    
    // Также учитываем историю градиентов
    if (gradient_norm_history > 0.0) {
        adaptive_threshold = max(adaptive_threshold, gradient_norm_history * 1.5);
    }
    
    // Применяем clipping только если необходимо
    if (grad_norm > adaptive_threshold) {
        grad = (grad / grad_norm) * adaptive_threshold;
    }
}

// Установка параметров Cosine Annealing
void NeuralNetwork::setCosineAnnealingParams(double T_0_param, double T_mult_param, double eta_min_param) {
    T_0 = std::max(1.0, T_0_param);
    T_mult = std::max(1.0, T_mult_param);
    eta_min = std::max(0.0, eta_min_param);
    use_cosine_annealing = true;
    current_T = 0;
    step_in_T = 0;
    cout << "[NeuralNetwork] Cosine Annealing enabled: T_0=" << T_0 
         << ", T_mult=" << T_mult << ", eta_min=" << eta_min << endl;
}

vector<TrainingStats> NeuralNetwork::getTrainingHistory() const {
    lock_guard<mutex> lock(training_history_mutex);
    return training_history;
}

// Обновление Learning Rate по Cosine Annealing (оптимизированная версия с кэшированием)
void NeuralNetwork::setLearningRate(double lr) {
    learning_rate = lr;
    initial_learning_rate = lr;
}

void NeuralNetwork::updateLearningRateCosineAnnealing(int epoch, int total_epochs) {
    if (!use_cosine_annealing) {
        return;
    }
    
    // Вычисляем текущий период T_i
    int T_i = static_cast<int>(T_0 * pow(T_mult, current_T));
    
    // Если достигли конца текущего периода, перезапускаем
    if (step_in_T >= T_i) {
        current_T++;
        step_in_T = 0;
        T_i = static_cast<int>(T_0 * pow(T_mult, current_T));
        cout << "[NeuralNetwork] Cosine Annealing restart: T=" << T_i << ", lr=" << initial_learning_rate << endl;
    }
    
    // Формула Cosine Annealing: lr(t) = eta_min + (eta_max - eta_min) * (1 + cos(π * t / T)) / 2
    // Оптимизация: предвычисляем константы и используем более эффективные вычисления
    double eta_max = initial_learning_rate;
    double t = static_cast<double>(step_in_T);
    double T = static_cast<double>(T_i);
    
    // Кэшируем часто используемые значения
    static double cached_cos_arg = -1.0;
    static double cached_cos_val = 1.0;
    double cos_arg = M_PI * t / T;
    
    // Вычисляем cos только если аргумент изменился (для батчей с одинаковым step_in_T)
    if (abs(cos_arg - cached_cos_arg) > 1e-10) {
        cached_cos_val = cos(cos_arg);
        cached_cos_arg = cos_arg;
    }
    
    double new_lr = eta_min + (eta_max - eta_min) * (1.0 + cached_cos_val) / 2.0;
    
    learning_rate = (eta_min > new_lr) ? eta_min : new_lr;
    step_in_T++;
}

// Применение Label Smoothing к меткам (оптимизированная векторизованная версия)
MatrixXd NeuralNetwork::applyLabelSmoothing(const MatrixXd& y_hard) {
    if (label_smoothing <= 0.0) {
        return y_hard; // Без сглаживания
    }
    
    int m = y_hard.rows();
    int K = y_hard.cols();
    
    // Формула: y_smooth = (1-ε) * y_hard + ε/K
    // Векторизованная реализация для лучшей производительности
    double epsilon = label_smoothing;
    double uniform_prob = epsilon / K;
    
    // In-place операция: используем noalias() для избежания временных копий
    MatrixXd y_smooth = (1.0 - epsilon) * y_hard;
    y_smooth.array() += uniform_prob;
    
    return y_smooth;
}

// Вычисление Focal Loss
double NeuralNetwork::computeFocalLoss(const MatrixXd& X, const MatrixXd& y) {
    vector<MatrixXd> activations, z_values;
    forward(X, activations, z_values);
    MatrixXd logits = activations.back();
    
    // Применяем softmax для получения вероятностей
    MatrixXd probs = softmax(logits);
    
    // Добавляем epsilon для численной стабильности
    const double epsilon = 1e-15;
    probs = probs.array().max(epsilon).min(1.0 - epsilon);
    
    int m = X.rows();
    MatrixXd focal_loss_per_sample = MatrixXd::Zero(m, 1);
    
    for (int i = 0; i < m; ++i) {
        // Находим истинный класс
        int true_class = -1;
        for (int j = 0; j < output_size; ++j) {
            if (y(i, j) > 0.5) {
                true_class = j;
                break;
            }
        }
        
        if (true_class >= 0 && true_class < output_size) {
            // Вероятность правильного класса
            double p_t = probs(i, true_class);
            
            // Focal Loss: FL = -α(1-p_t)^γ * log(p_t)
            double modulating_factor = pow(1.0 - p_t, focal_gamma);
            double ce_loss = -log(p_t);
            double focal_loss = focal_alpha * modulating_factor * ce_loss;
            
            // Применяем веса классов, если установлены
            if (class_weights.size() == output_size) {
                focal_loss *= class_weights(true_class);
            }
            
            focal_loss_per_sample(i, 0) = focal_loss;
        }
    }
    
    return focal_loss_per_sample.mean();
}

// Вычисление точности (параллелизация по образцам)
double NeuralNetwork::computeAccuracy(const MatrixXd& X, const MatrixXd& y) {
    vector<MatrixXd> activations, z_values;
    forward(X, activations, z_values);
    MatrixXd predictions = activations.back();
    
    int correct = 0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:correct)
#endif
    for (int i = 0; i < X.rows(); ++i) {
        int pred_class, true_class;
        predictions.row(i).maxCoeff(&pred_class);
        y.row(i).maxCoeff(&true_class);
        if (pred_class == true_class) ++correct;
    }
    return static_cast<double>(correct) / X.rows();
}

// Вычисление метрик по каждому классу
vector<NeuralNetwork::ClassMetrics> NeuralNetwork::computeClassMetrics(const MatrixXd& X, const MatrixXd& y) {
    vector<ClassMetrics> metrics(output_size);
    
    vector<MatrixXd> activations, z_values;
    forward(X, activations, z_values);
    MatrixXd predictions = activations.back();
    
    // Инициализация метрик
    for (int i = 0; i < output_size; ++i) {
        metrics[i].class_id = i;
        metrics[i].precision = 0.0;
        metrics[i].recall = 0.0;
        metrics[i].f1_score = 0.0;
        metrics[i].true_positives = 0;
        metrics[i].false_positives = 0;
        metrics[i].false_negatives = 0;
    }
    
    // Подсчет TP, FP, FN для каждого класса (параллельно)
#ifdef _OPENMP
    #pragma omp parallel
    {
        vector<ClassMetrics> local_metrics(output_size);
        for (int c = 0; c < output_size; ++c) {
            local_metrics[c].class_id = c;
            local_metrics[c].true_positives = 0;
            local_metrics[c].false_positives = 0;
            local_metrics[c].false_negatives = 0;
        }
        #pragma omp for
        for (int i = 0; i < X.rows(); ++i) {
            int pred_class, true_class;
            predictions.row(i).maxCoeff(&pred_class);
            y.row(i).maxCoeff(&true_class);
            if (pred_class == true_class) {
                local_metrics[true_class].true_positives++;
            } else {
                local_metrics[true_class].false_negatives++;
                local_metrics[pred_class].false_positives++;
            }
        }
        #pragma omp critical
        for (int c = 0; c < output_size; ++c) {
            metrics[c].true_positives += local_metrics[c].true_positives;
            metrics[c].false_positives += local_metrics[c].false_positives;
            metrics[c].false_negatives += local_metrics[c].false_negatives;
        }
    }
#else
    for (int i = 0; i < X.rows(); ++i) {
        int pred_class, true_class;
        predictions.row(i).maxCoeff(&pred_class);
        y.row(i).maxCoeff(&true_class);
        if (pred_class == true_class) {
            metrics[true_class].true_positives++;
        } else {
            metrics[true_class].false_negatives++;
            metrics[pred_class].false_positives++;
        }
    }
#endif
    
    // Вычисление precision, recall, F1-score
    for (int i = 0; i < output_size; ++i) {
        int tp = metrics[i].true_positives;
        int fp = metrics[i].false_positives;
        int fn = metrics[i].false_negatives;
        
        // Precision = TP / (TP + FP)
        if (tp + fp > 0) {
            metrics[i].precision = static_cast<double>(tp) / (tp + fp);
        }
        
        // Recall = TP / (TP + FN)
        if (tp + fn > 0) {
            metrics[i].recall = static_cast<double>(tp) / (tp + fn);
        }
        
        // F1-score = 2 * (precision * recall) / (precision + recall)
        if (metrics[i].precision + metrics[i].recall > 0) {
            metrics[i].f1_score = 2.0 * metrics[i].precision * metrics[i].recall 
                                 / (metrics[i].precision + metrics[i].recall);
        }
    }
    
    return metrics;
}

// Вычисление weighted F1-score (взвешенного по количеству образцов)
double NeuralNetwork::computeWeightedF1Score(const MatrixXd& X, const MatrixXd& y) {
    vector<ClassMetrics> metrics = computeClassMetrics(X, y);
    
    // Подсчет количества образцов для каждого класса (параллельно)
    vector<int> class_counts(output_size, 0);
#ifdef _OPENMP
    #pragma omp parallel
    {
        vector<int> local_counts(output_size, 0);
        #pragma omp for
        for (int i = 0; i < X.rows(); ++i) {
            int true_class;
            y.row(i).maxCoeff(&true_class);
            if (true_class >= 0 && true_class < output_size) {
                local_counts[true_class]++;
            }
        }
        #pragma omp critical
        for (int c = 0; c < output_size; ++c) class_counts[c] += local_counts[c];
    }
#else
    for (int i = 0; i < X.rows(); ++i) {
        int true_class;
        y.row(i).maxCoeff(&true_class);
        if (true_class >= 0 && true_class < output_size) {
            class_counts[true_class]++;
        }
    }
#endif
    
    // Вычисление weighted F1-score
    double weighted_f1_sum = 0.0;
    int total_samples = 0;
    
    for (int i = 0; i < output_size; ++i) {
        int class_samples = class_counts[i];
        if (class_samples > 0) {
            weighted_f1_sum += metrics[i].f1_score * class_samples;
            total_samples += class_samples;
        }
    }
    
    if (total_samples == 0) {
        return 0.0;
    }
    
    return weighted_f1_sum / total_samples;
}

// Вычисление macro-averaged F1-score (более строгая метрика)
double NeuralNetwork::computeMacroAveragedF1Score(const MatrixXd& X, const MatrixXd& y) {
    vector<ClassMetrics> metrics = computeClassMetrics(X, y);
    
    double macro_f1_sum = 0.0;
    int classes_with_data = 0;
    
    for (const auto& metric : metrics) {
        bool has_data = (metric.true_positives + metric.false_negatives > 0) || 
                       (metric.true_positives + metric.false_positives > 0);
        if (has_data) {
            macro_f1_sum += metric.f1_score;
            classes_with_data++;
        }
    }
    
    if (classes_with_data == 0) {
        return 0.0;
    }
    
    return macro_f1_sum / classes_with_data;
}

// Поиск оптимальных thresholds для каждого класса (ROC-анализ)
VectorXd NeuralNetwork::findOptimalThresholds(const MatrixXd& X_val, const MatrixXd& y_val) {
    VectorXd optimal_thresholds = VectorXd::Constant(output_size, 0.5);
    
    // Получаем предсказания для всех классов
    MatrixXd predictions = predict(X_val);
    
    // Для каждого класса находим оптимальный threshold
    for (int class_idx = 0; class_idx < output_size; ++class_idx) {
        double best_f1 = 0.0;
        double best_threshold = 0.5;
        
        // Перебираем различные пороги от 0.1 до 0.9
        for (double threshold = 0.1; threshold <= 0.9; threshold += 0.05) {
            int tp = 0, fp = 0, fn = 0;
            
            for (int i = 0; i < X_val.rows(); ++i) {
                bool predicted = (predictions(i, class_idx) >= threshold);
                bool actual = (y_val(i, class_idx) >= 0.5);
                
                if (predicted && actual) tp++;
                else if (predicted && !actual) fp++;
                else if (!predicted && actual) fn++;
            }
            
            // Вычисляем F1-score для этого threshold
            double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
            double recall = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
            double f1 = (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;
            
            if (f1 > best_f1) {
                best_f1 = f1;
                best_threshold = threshold;
            }
        }
        
        optimal_thresholds(class_idx) = best_threshold;
    }
    
    return optimal_thresholds;
}

// Этап 1: Контроль качества
vector<int> NeuralNetwork::stage1_quality_control(const MatrixXd& X, const MatrixXd& y,
                                                  double loss_threshold, bool use_cross_entropy)
 {
    vector<int> accepted_indices;
    
    cout << "\n" << string(60, '=') << endl;
    cout << "STAGE 1: Quality control training (loss threshold: " << loss_threshold << ")" << endl;
    cout << string(60, '=') << endl;
    
    // Сохраняем исходные веса
    vector<MatrixXd> original_weights = weights;
    vector<MatrixXd> original_biases = biases;
    
    for (int i = 0; i < X.rows(); ++i) {
        // Восстанавливаем исходные веса для каждого образца
        weights = original_weights;
        biases = original_biases;
        
        // Подготовка одного образца
        MatrixXd sample_X = X.row(i);
        MatrixXd sample_y = y.row(i);
        
        // Прямое распространение
        vector<MatrixXd> activations, z_values;
        forward(sample_X, activations, z_values);
        
        // Обратное распространение
        vector<MatrixXd> weight_gradients, bias_gradients;
        backward(sample_X, sample_y, activations, z_values, weight_gradients, bias_gradients);
        
        // Обновление весов
        updateWeights(weight_gradients, bias_gradients);
        
        // Контроль качества: используем тот же образ
        double loss = use_cross_entropy ? computeCrossEntropyLoss(sample_X, sample_y) 
                                        : computeLoss(sample_X, sample_y);
        
        string status = (loss < loss_threshold) ? "[OK] ACCEPTED" : "[X] REJECTED";
        
        if (loss < loss_threshold) {
            accepted_indices.push_back(i);
        }
        
        if ((i + 1) % 100 == 0 || i == X.rows() - 1) {
            cout << "Sample " << (i + 1) << "/" << X.rows() 
                 << ": loss=" << fixed << setprecision(4) << loss << " " << status << endl;
        }
        // Push progress for UI (every 50 samples or last)
        if ((i + 1) % 50 == 0 || i == X.rows() - 1) {
            TrainingStats stats;
            stats.stage = 1;
            stats.epoch = X.rows();  // total for progress = samples_processed/epoch
            stats.accuracy = 0;
            stats.loss = loss;
            stats.samples_processed = i + 1;
            stats.samples_accepted = static_cast<int>(accepted_indices.size());
            {
                lock_guard<mutex> lock(training_history_mutex);
                training_history.push_back(stats);
            }
        }
    }
    
    cout << "\nAccepted samples: " << accepted_indices.size() << "/" << X.rows()
         << " (" << fixed << setprecision(1) << (100.0 * accepted_indices.size() / X.rows()) << "%)" << endl;
    
    return accepted_indices;
}

// Этап 2: Батчевое обучение
void NeuralNetwork::stage2_batch_training(const MatrixXd& X, const MatrixXd& y,
                                         int batch_size, int epochs,
                                         int early_stopping_patience,
                                         double validation_split, bool use_cross_entropy,
                                         const VectorXd& sample_weights) 
{
    PROFILE_FUNCTION();
    
    cout << "\n" << string(60, '=') << endl;
    cout << "STAGE 2: Batch training" << endl;
    cout << string(60, '=') << endl;
    
    // Перемешивание данных
    vector<int> indices(X.rows());
    iota(indices.begin(), indices.end(), 0);
    random_device rd;
    mt19937 gen(rd());
    shuffle(indices.begin(), indices.end(), gen);
    
    MatrixXd X_shuffled(X.rows(), X.cols());
    MatrixXd y_shuffled(y.rows(), y.cols());
    
    // Параллелизация копирования данных (безопасная операция)
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < X.rows(); ++i) {
        X_shuffled.row(i) = X.row(indices[i]);
        y_shuffled.row(i) = y.row(indices[i]);
    }
    
    // Разделение на обучающую и валидационную выборки
    int split_idx = static_cast<int>(X.rows() * (1.0 - validation_split));
    MatrixXd X_train = X_shuffled.topRows(split_idx);
    MatrixXd y_train = y_shuffled.topRows(split_idx);
    MatrixXd X_val = X_shuffled.bottomRows(X.rows() - split_idx);
    MatrixXd y_val = y_shuffled.bottomRows(y.rows() - split_idx);
    
    double best_val_loss = numeric_limits<double>::max();
    double best_val_accuracy = 0.0;  // Для логов
    double best_val_f1 = 0.0;        // Для логов
    int patience_counter_loss = 0;
    int no_improvement_epochs = 0;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Перемешивание обучающих данных в каждой эпохе
        vector<int> train_indices(X_train.rows());
        iota(train_indices.begin(), train_indices.end(), 0);
        random_device rd_epoch;
        mt19937 gen_epoch(rd_epoch());
        shuffle(train_indices.begin(), train_indices.end(), gen_epoch);
        
        MatrixXd X_train_epoch(X_train.rows(), X_train.cols());
        MatrixXd y_train_epoch(y_train.rows(), y_train.cols());
        
        // Параллелизация копирования данных
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < X_train.rows(); ++i) {
            X_train_epoch.row(i) = X_train.row(train_indices[i]);
            y_train_epoch.row(i) = y_train.row(train_indices[i]);
        }
        
        // Обучение по батчам
        double epoch_loss = 0.0;
        int num_batches = 0;
        for (int batch_start = 0; batch_start < X_train.rows(); batch_start += batch_size)
        {
            int batch_end = min(batch_start + batch_size, static_cast<int>(X_train.rows()));
            MatrixXd X_batch = X_train_epoch.middleRows(batch_start, batch_end - batch_start);
            MatrixXd y_batch = y_train_epoch.middleRows(batch_start, batch_end - batch_start);
            
            // Прямое распространение
            vector<MatrixXd> activations, z_values;
            forward(X_batch, activations, z_values);
            
            // Обратное распространение
            vector<MatrixXd> weight_gradients, bias_gradients;
            VectorXd batch_weights;
            if (sample_weights.size() == X.rows()) {
                batch_weights = VectorXd(batch_end - batch_start);
                for (int i = 0; i < batch_end - batch_start; ++i) {
                    int orig_idx = train_indices[batch_start + i];
                    batch_weights(i) = sample_weights(train_indices[orig_idx]);
                }
            }
            num_batches++;
            
            if (use_cross_entropy) {
                backwardCrossEntropy(X_batch, y_batch, activations, z_values, weight_gradients, bias_gradients, batch_weights);
            } else {
                backward(X_batch, y_batch, activations, z_values, weight_gradients, bias_gradients);
            }
            
            // Обновление весов
            updateWeights(weight_gradients, bias_gradients);
            
            // Вычисление потерь
            epoch_loss += (use_cross_entropy ? computeCrossEntropyLoss(X_batch, y_batch) 
                                            : computeLoss(X_batch, y_batch)) * (batch_end - batch_start);
        }
        if (num_batches > 0) {
            epoch_loss /= num_batches;
        }
        
        // Оценка на валидационной выборке
        double val_accuracy = computeAccuracy(X_val, y_val);
        // Используем macro-averaged F1 для более строгой оценки
        double val_f1 = computeMacroAveragedF1Score(X_val, y_val);
        double val_weighted_f1 = computeWeightedF1Score(X_val, y_val);  // Для информации
        double val_loss = use_cross_entropy ? computeCrossEntropyLoss(X_val, y_val) 
                                            : computeLoss(X_val, y_val);
        double train_accuracy = computeAccuracy(X_train, y_train);
        
        // Per-class early stopping: проверяем, все ли классы достигли F1 > 0.90
        vector<ClassMetrics> val_metrics = computeClassMetrics(X_val, y_val);
        bool all_classes_good = true;
        for (const auto& metric : val_metrics) {
            bool has_data = (metric.true_positives + metric.false_negatives > 0) || 
                           (metric.true_positives + metric.false_positives > 0);
            if (has_data && metric.f1_score < 0.90) {
                all_classes_good = false;
                break;
            }
        }
        
        // Сохранение лучшей модели по validation loss (критерий улучшения)
        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            best_val_f1 = val_f1;           // Для логов
            best_val_accuracy = val_accuracy;
            best_weights = weights;
            best_biases = biases;
            patience_counter_loss = 0;
        } else {
            patience_counter_loss++;
            no_improvement_epochs++;
        }
        
        // Per-class early stopping: если все классы достигли F1 > 0.90, можно остановиться раньше
        if (all_classes_good && epoch >= 20) {  // Минимум 20 эпох
            cout << "\n[Early Stopping] All classes achieved F1 > 0.90, stopping early at epoch " 
                 << (epoch + 1) << endl;
            break;
        }
        
        // Адаптивная настройка весов классов на основе F1-score валидации
        updateClassWeightsFromF1(val_metrics);
        
        // Оптимизация thresholds каждые 5 эпох или в конце (увеличена частота для лучшей адаптации)
        if ((epoch + 1) % 5 == 0 || epoch == epochs - 1) {
            class_thresholds = findOptimalThresholds(X_val, y_val);
        }
        
        // Сохранение статистики (с ограничением размера истории)
        TrainingStats stats;
        stats.stage = 2;
        stats.epoch = epoch + 1;
        stats.accuracy = val_accuracy;
        stats.loss = val_loss;
        stats.samples_processed = X_train.rows();
        stats.samples_accepted = X_train.rows();
        {
            lock_guard<mutex> lock(training_history_mutex);
            training_history.push_back(stats);
            const size_t max_history_size = 1000;
            if (training_history.size() > max_history_size) {
                training_history.erase(training_history.begin(), 
                                     training_history.begin() + (training_history.size() - max_history_size));
            }
        }
        
        // Обновление Learning Rate по Cosine Annealing
        if (use_cosine_annealing) {
            updateLearningRateCosineAnnealing(epoch, epochs);
        }
        
        // Адаптивная настройка learning rate на основе улучшения F1-score
        static double prev_val_f1 = 0.0;
        if (epoch > 0) {
            double f1_improvement = val_f1 - prev_val_f1;
            if (f1_improvement < 0.001 && learning_rate > initial_learning_rate * 0.01) {
                // Уменьшаем learning rate если F1-score не улучшается
                learning_rate *= 0.95;
            } else if (f1_improvement > 0.01 && learning_rate < initial_learning_rate) {
                // Небольшое увеличение если F1-score быстро улучшается
                learning_rate = min(initial_learning_rate, learning_rate * 1.02);
            }
        }
        prev_val_f1 = val_f1;
        
        // Динамическая корректировка dropout rate на основе recall
        if (use_dropout && !val_metrics.empty()) {
            double avg_recall = 0.0;
            int classes_with_data = 0;
            for (const auto& metric : val_metrics) {
                if (metric.true_positives + metric.false_negatives > 0) {
                    avg_recall += metric.recall;
                    classes_with_data++;
                }
            }
            if (classes_with_data > 0) {
                avg_recall /= classes_with_data;
                // Уменьшаем dropout если recall низкий (модель недопредсказывает)
                if (avg_recall < 0.6 && dropout_rate > 0.1) {
                    dropout_rate = max(0.1, dropout_rate * 0.95);
                } else if (avg_recall > 0.9 && dropout_rate < 0.5) {
                    dropout_rate = min(0.5, dropout_rate * 1.05);
                }
            }
        }
        
        // Вывод информации
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            // Улучшенный вывод прогресса в консоль
            double progress = (epoch + 1) / static_cast<double>(epochs) * 100.0;
            int bar_width = 50;
            int pos = static_cast<int>(bar_width * (epoch + 1) / epochs);
            
            cout << "\rEpoch " << (epoch + 1) << "/" << epochs << " [" << fixed << setprecision(1) << progress << "%] [";
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) cout << "=";
                else if (i == pos) cout << ">";
                else cout << " ";
            }
            cout << "] ";
            cout << "Train Acc=" << setprecision(4) << train_accuracy << ", "
                 << "Val Acc=" << val_accuracy << ", "
                 << "Val F1=" << val_f1 << ", "
                 << "Loss=" << epoch_loss / num_batches;
            cout.flush();
            
            // Полный вывод каждые 10 эпох или в конце
            if ((epoch + 1) % 10 == 0 || epoch == epochs - 1) {
                cout << "\n  Details: Train Loss=" << epoch_loss 
                     << ", Val Loss=" << val_loss 
                     << ", Train Acc=" << train_accuracy 
                     << ", Val Acc=" << val_accuracy 
                     << ", Val F1=" << val_f1
                     << ", Best Val Loss=" << best_val_loss << endl;
            }
        }
        
        // Catastrophic collapse detection: stop if loss spikes > 3x best
        if (val_loss > best_val_loss * 3.0 && epoch > 5) {
            cout << "\n[Early Stopping] Loss spike detected (val_loss=" << val_loss
                 << " > 3x best=" << best_val_loss << "), stopping to preserve model" << endl;
            break;
        }

        // Ранняя остановка на основе validation loss
        if (patience_counter_loss >= early_stopping_patience) {
            cout << "\nEarly stopping at epoch " << (epoch + 1) << ": "
                 << "validation loss did not improve for " << early_stopping_patience << " epochs" << endl;
            break;
        }
    }
    
    // Восстановление лучших весов (выбранных по validation loss)
    if (!best_weights.empty()) {
        weights = best_weights;
        biases = best_biases;
        cout << "\nRestored best weights with val_loss = " << fixed << setprecision(4)
             << best_val_loss << " (val F1: " << best_val_f1 << ", accuracy: " << best_val_accuracy << ")" << endl;
    }
    
    // Вывод отчета профилирования
    Profiler::getInstance().printReport();
    
    // Адаптивное уменьшение learning rate на основе статистики обучения
    // Оптимизация гиперпараметров: автоматическая настройка на основе прогресса
    if (no_improvement_epochs > 5 && learning_rate > initial_learning_rate * 0.01) {
        adjustLearningRate(0.8);
    }
    
    // Адаптивная настройка momentum на основе истории обучения
    if (training_history.size() > 10) {
        // Вычисляем среднюю скорость улучшения за последние 10 эпох
        double recent_improvement = 0.0;
        int count = 0;
        for (size_t i = training_history.size() - 10; i < training_history.size() - 1; ++i) {
            if (i + 1 < training_history.size()) {
                recent_improvement += training_history[i + 1].accuracy - training_history[i].accuracy;
                count++;
            }
        }
        if (count > 0) {
            recent_improvement /= count;
            
            // Если улучшение медленное, увеличиваем momentum для более стабильного обучения
            if (recent_improvement < 0.001 && momentum < 0.99) {
                momentum = min(0.99, momentum + 0.01);
            }
            // Если улучшение хорошее, можно немного уменьшить momentum для более быстрой адаптации
            else if (recent_improvement > 0.01 && momentum > 0.85) {
                momentum = max(0.85, momentum - 0.005);
            }
        }
    }
}

// Этап 3: Обучение с накоплением производных
void NeuralNetwork::stage3_accumulated_training(const MatrixXd& X, const MatrixXd& y,
                                                int batch_size, int epochs, bool use_cross_entropy,
                                                const VectorXd& sample_weights) {
    PROFILE_FUNCTION();
    
    cout << "\n" << string(60, '=') << endl;
    cout << "STAGE 3: Training with gradient accumulation" << endl;
    cout << string(60, '=') << endl;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Перемешивание данных
        vector<int> indices(X.rows());
        iota(indices.begin(), indices.end(), 0);
        random_device rd_stage3;
        mt19937 gen_stage3(rd_stage3());
        shuffle(indices.begin(), indices.end(), gen_stage3);
        
        MatrixXd X_shuffled(X.rows(), X.cols());
        MatrixXd y_shuffled(y.rows(), y.cols());
        
        // Параллелизация копирования данных
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < X.rows(); ++i) {
            X_shuffled.row(i) = X.row(indices[i]);
            y_shuffled.row(i) = y.row(indices[i]);
        }
        
        // Накопление градиентов по батчам
        vector<MatrixXd> accumulated_weight_grads(weights.size());
        vector<MatrixXd> accumulated_bias_grads(biases.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            accumulated_weight_grads[i] = MatrixXd::Zero(weights[i].rows(), weights[i].cols());
            accumulated_bias_grads[i] = MatrixXd::Zero(biases[i].rows(), 1);
        }
        int batch_count = 0;
        
        for (int batch_start = 0; batch_start < X.rows(); batch_start += batch_size) {
            int batch_end = min(batch_start + batch_size, static_cast<int>(X.rows()));
            MatrixXd X_batch = X_shuffled.middleRows(batch_start, batch_end - batch_start);
            MatrixXd y_batch = y_shuffled.middleRows(batch_start, batch_end - batch_start);
            
            // Прямое распространение
            vector<MatrixXd> activations, z_values;
            forward(X_batch, activations, z_values);
            
            // Обратное распространение
            vector<MatrixXd> weight_gradients, bias_gradients;
            VectorXd batch_weights;
            if (sample_weights.size() == X.rows()) {
                batch_weights = VectorXd(batch_end - batch_start);
                for (int i = 0; i < batch_end - batch_start; ++i) {
                    int orig_idx = indices[batch_start + i];
                    batch_weights(i) = sample_weights(orig_idx);
                }
            }
            
            if (use_cross_entropy) {
                backwardCrossEntropy(X_batch, y_batch, activations, z_values, weight_gradients, bias_gradients, batch_weights);
            } else {
                backward(X_batch, y_batch, activations, z_values, weight_gradients, bias_gradients);
            }
            
            // Накопление градиентов
            for (size_t i = 0; i < weights.size(); ++i) {
                accumulated_weight_grads[i] += weight_gradients[i];
                accumulated_bias_grads[i] += bias_gradients[i];
            }
            batch_count++;
        }
        
        // Обновление весов после обработки всех батчей (с momentum)
        for (size_t i = 0; i < weights.size(); ++i) {
            MatrixXd avg_weight_grad = accumulated_weight_grads[i] / batch_count;
            MatrixXd avg_bias_grad = accumulated_bias_grads[i] / batch_count;
            
            // Добавление L2 регуляризации
            avg_weight_grad += l2_reg * weights[i];
            
            // Обновление velocity с momentum
            velocity_w[i] = momentum * velocity_w[i] - learning_rate * avg_weight_grad;
            velocity_b[i] = momentum * velocity_b[i] - learning_rate * avg_bias_grad;
            
            // Обновление весов
            weights[i] += velocity_w[i];
            biases[i] += velocity_b[i];
        }
        
        // Обновление Learning Rate по Cosine Annealing
        if (use_cosine_annealing) {
            updateLearningRateCosineAnnealing(epoch, epochs);
        }
        
        // Оценка
        double accuracy = computeAccuracy(X, y);
        double loss = use_cross_entropy ? computeCrossEntropyLoss(X, y) : computeLoss(X, y);
        
        TrainingStats stats;
        stats.stage = 3;
        stats.epoch = epoch + 1;
        stats.accuracy = accuracy;
        stats.loss = loss;
        stats.samples_processed = X.rows();
        stats.samples_accepted = X.rows();
        {
            lock_guard<mutex> lock(training_history_mutex);
            training_history.push_back(stats);
            const size_t max_history_size = 1000;
            if (training_history.size() > max_history_size) {
                training_history.erase(training_history.begin(), 
                                     training_history.begin() + (training_history.size() - max_history_size));
            }
        }
        
        // Улучшенный вывод прогресса для этапа 3
        double progress = (epoch + 1) / static_cast<double>(epochs) * 100.0;
        int bar_width = 50;
        int pos = static_cast<int>(bar_width * (epoch + 1) / epochs);
        
        cout << "\rEpoch " << (epoch + 1) << "/" << epochs << " [" << fixed << setprecision(1) << progress << "%] [";
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) cout << "=";
            else if (i == pos) cout << ">";
            else cout << " ";
        }
        cout << "] Acc=" << setprecision(4) << accuracy << ", Loss=" << loss;
        cout.flush();
        
        if ((epoch + 1) % 10 == 0 || epoch == epochs - 1) {
            cout << "\n  Stage 3: Acc=" << accuracy << ", Loss=" << loss << endl;
        }
    }
    
    // Вывод отчета профилирования для stage 3
    Profiler::getInstance().printReport();
}

// Полный цикл многоэтапного обучения
double NeuralNetwork::train_multi_stage(const MatrixXd& X, const MatrixXd& y,
                                        double loss_threshold, int batch_size,
                                        int stage2_epochs, int stage3_epochs,
                                        bool use_stage3, int early_stopping_patience, bool use_cross_entropy,
                                        const VectorXd& sample_weights) {
 //   cout << "\n" << string(80, '=') << endl;
  //  cout << "\nМногоэтапное обучение... " << endl;
  //  cout << string(80, '=') << endl;
    
    // Этап 1: Контроль качества
    vector<int> accepted_indices = stage1_quality_control(X, y, loss_threshold, use_cross_entropy);
    
    if (accepted_indices.empty()) {
        cout << "WARNING: No samples passed quality control!" << endl;
        cout << "Try reducing loss_threshold or check your data" << endl;
        return 0.0;
    }
    
    // Отбор принятых образцов
    MatrixXd X_selected(accepted_indices.size(), X.cols());
    MatrixXd y_selected(accepted_indices.size(), y.cols());
    VectorXd weights_selected(accepted_indices.size());
    for (size_t i = 0; i < accepted_indices.size(); ++i) {
        X_selected.row(i) = X.row(accepted_indices[i]);
        y_selected.row(i) = y.row(accepted_indices[i]);
        if (sample_weights.size() == X.rows()) {
            weights_selected(i) = sample_weights(accepted_indices[i]);
        } else {
            weights_selected(i) = 1.0;
        }
    }
    
    cout << "\nProceeding to stage 2 with " << accepted_indices.size() << " selected samples" << endl;
    
    // Этап 2: Батчевое обучение
    stage2_batch_training(X_selected, y_selected, batch_size, stage2_epochs, 
                         early_stopping_patience, 0.2, use_cross_entropy, weights_selected);
    
    // Этап 3: Обучение с накоплением (опционально)
    if (use_stage3) {
        stage3_accumulated_training(X_selected, y_selected, batch_size, stage3_epochs, use_cross_entropy, weights_selected);
    }
    
    // Финальная оценка
    double final_accuracy = computeAccuracy(X_selected, y_selected);
    double final_loss = use_cross_entropy ? computeCrossEntropyLoss(X_selected, y_selected) 
                                         : computeLoss(X_selected, y_selected);
    
    cout << "\n" << string(80, '=') << endl;
    cout << "TRAINING STAGES COMPLETED" << endl;
    cout << string(80, '=') << endl;
    cout << "Final accuracy: " << fixed << setprecision(4) << final_accuracy 
         << " (" << setprecision(2) << (final_accuracy * 100) << "%)" << endl;
    cout << "Final loss: " << setprecision(4) << final_loss << endl;
    cout << "Samples used: " << accepted_indices.size() << "/" << X.rows() << endl;
    
    // Вывод отчета профилирования
    Profiler::getInstance().printReport();
    
    return final_accuracy;
}

// Этап 4: Специальное обучение для проблемных классов
void NeuralNetwork::stage4_problem_classes_training(const MatrixXd& X, const MatrixXd& y,
                                                     const vector<int>& problem_classes,
                                                     int batch_size, int epochs,
                                                     double learning_rate_multiplier,
                                                     const VectorXd& sample_weights) {
    PROFILE_FUNCTION();
    
    if (problem_classes.empty()) {
        return; // Нет проблемных классов
    }
    
    cout << "\n" << string(60, '=') << endl;
    cout << "STAGE 4: Problem Classes Training (Enhanced)" << endl;
    cout << string(60, '=') << endl;
    
    // Сохраняем исходные параметры
    double original_lr = learning_rate;
    bool original_focal = use_focal_loss;
    
    // Увеличиваем learning rate для проблемных классов
    learning_rate *= learning_rate_multiplier;
    
    // Включаем Focal Loss для фокуса на сложных примерах
    if (!use_focal_loss) {
        setFocalLossParams(2.5, 1.0, true);  // Более высокий gamma для проблемных классов
    }
    cout << "Problem classes: ";
    for (size_t i = 0; i < problem_classes.size(); ++i) {
        cout << problem_classes[i];
        if (i < problem_classes.size() - 1) cout << ", ";
    }
    cout << endl;
    
    // Фильтруем данные: оставляем только проблемные классы
    vector<int> problem_indices;
    for (int i = 0; i < X.rows(); ++i) {
        // Проверяем, принадлежит ли образец к проблемному классу
        for (int j = 0; j < y.cols(); ++j) {
            if (y(i, j) > 0.5) {
                // Найден класс образца
                if (find(problem_classes.begin(), problem_classes.end(), j) != problem_classes.end()) {
                    problem_indices.push_back(i);
                    break;
                }
            }
        }
    }
    
    if (problem_indices.empty()) {
        cout << "WARNING: No samples found for problem classes!" << endl;
        return;
    }
    
    // Создаем подмножество данных с проблемными классами
    MatrixXd X_problem(problem_indices.size(), X.cols());
    MatrixXd y_problem(problem_indices.size(), y.cols());
    VectorXd weights_problem(problem_indices.size());
    
    for (size_t i = 0; i < problem_indices.size(); ++i) {
        X_problem.row(i) = X.row(problem_indices[i]);
        y_problem.row(i) = y.row(problem_indices[i]);
        if (sample_weights.size() == X.rows()) {
            weights_problem(i) = sample_weights(problem_indices[i]);
        } else {
            weights_problem(i) = 1.0;
        }
    }
    
    cout << "Selected " << problem_indices.size() << " samples from problem classes" << endl;
    
    // Сохраняем исходный learning rate
    double original_learning_rate = learning_rate;
    
    // Увеличиваем learning rate для проблемных классов
    learning_rate *= learning_rate_multiplier;
    cout << "Learning rate increased: " << original_learning_rate 
         << " -> " << learning_rate << " (x" << learning_rate_multiplier << ")" << endl;
    
    // Включаем Focal Loss для проблемных классов (используем уже объявленную переменную)
    if (!use_focal_loss) {
        setFocalLossParams(3.2, 1.0, true);  // gamma = 3.2 (в диапазоне 3.0-3.5)
        cout << "Focal Loss enabled for problem classes (gamma=3.2)" << endl;
    }
    
    // Включаем cosine annealing для Stage 4
    setCosineAnnealingParams(epochs, 1.0, learning_rate * 0.1);
    cout << "Cosine annealing enabled for Stage 4 training" << endl;
    
    // Обучение на проблемных классах
    random_device rd;
    mt19937 gen(rd());
    vector<int> indices(problem_indices.size());
    iota(indices.begin(), indices.end(), 0);
    
    // Итеративное обучение: несколько циклов с уменьшающимся learning rate
    int num_cycles = 2;  // 2 цикла обучения
    double cycle_learning_rate = learning_rate;
    
    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        if (cycle > 0) {
            // Уменьшаем learning rate для следующего цикла
            cycle_learning_rate *= 0.7;
            learning_rate = cycle_learning_rate;
            cout << "\n[Stage 4] Starting cycle " << (cycle + 1) << " with learning rate: " << learning_rate << endl;
        }
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
        shuffle(indices.begin(), indices.end(), gen);
        
        double epoch_loss = 0.0;
        int num_batches = 0;
        
        for (int batch_start = 0; batch_start < static_cast<int>(indices.size()); batch_start += batch_size) {
            int batch_end = min(batch_start + batch_size, static_cast<int>(indices.size()));
            
            MatrixXd X_batch(batch_end - batch_start, X_problem.cols());
            MatrixXd y_batch(batch_end - batch_start, y_problem.cols());
            
            for (int i = batch_start; i < batch_end; ++i) {
                X_batch.row(i - batch_start) = X_problem.row(indices[i]);
                y_batch.row(i - batch_start) = y_problem.row(indices[i]);
            }
            
            // Прямое распространение
            vector<MatrixXd> activations, z_values;
            forward(X_batch, activations, z_values);
            
            // Обратное распространение с Focal Loss
            vector<MatrixXd> weight_gradients, bias_gradients;
            VectorXd batch_weights;
            if (weights_problem.size() == X_problem.rows()) {
                batch_weights = VectorXd(batch_end - batch_start);
                for (int i = 0; i < batch_end - batch_start; ++i) {
                    batch_weights(i) = weights_problem(indices[batch_start + i]);
                }
            }
            
            backwardCrossEntropy(X_batch, y_batch, activations, z_values, weight_gradients, bias_gradients, batch_weights);
            
            // Обновление весов
            updateWeights(weight_gradients, bias_gradients);
            
            // Вычисление потерь
            epoch_loss += computeCrossEntropyLoss(X_batch, y_batch) * (batch_end - batch_start);
            num_batches++;
        }
        
        if (num_batches > 0) {
            epoch_loss /= num_batches;
        }
        
        // Оценка
        double accuracy = computeAccuracy(X_problem, y_problem);
        double loss = computeCrossEntropyLoss(X_problem, y_problem);
        
        // Сохранение статистики (с ограничением размера истории)
        TrainingStats stats;
        stats.stage = 4;
        stats.epoch = epoch + 1;
        stats.accuracy = accuracy;
        stats.loss = loss;
        stats.samples_processed = X_problem.rows();
        stats.samples_accepted = X_problem.rows();
        {
            lock_guard<mutex> lock(training_history_mutex);
            training_history.push_back(stats);
            const size_t max_history_size = 1000;
            if (training_history.size() > max_history_size) {
                training_history.erase(training_history.begin(), 
                                     training_history.begin() + (training_history.size() - max_history_size));
            }
        }
        
        // Обновление learning rate по cosine annealing
        updateLearningRateCosineAnnealing(epoch, epochs);
        
        // Вывод прогресса
        if ((epoch + 1) % 5 == 0 || epoch == 0 || epoch == epochs - 1) {
            double progress = (epoch + 1) / static_cast<double>(epochs) * 100.0;
            cout << "\rStage 4 Epoch " << (epoch + 1) << "/" << epochs 
                 << " [" << fixed << setprecision(1) << progress << "%] "
                 << "Acc=" << setprecision(4) << accuracy 
                 << ", Loss=" << loss
                 << ", lr=" << setprecision(6) << learning_rate;
            cout.flush();
            
            if ((epoch + 1) % 10 == 0 || epoch == epochs - 1) {
                cout << "\n  Problem classes: Acc=" << accuracy << ", Loss=" << loss << endl;
            }
        }
    }
    
    // Оценка после цикла
    double cycle_accuracy = computeAccuracy(X_problem, y_problem);
    double cycle_f1 = computeWeightedF1Score(X_problem, y_problem);
    cout << "\n[Stage 4] Cycle " << (cycle + 1) << " completed: accuracy=" 
         << fixed << setprecision(4) << cycle_accuracy 
         << ", F1=" << cycle_f1 << endl;
    }
    
    // Восстанавливаем исходный learning rate
    learning_rate = original_learning_rate;
    
    // Восстанавливаем исходное состояние Focal Loss
    if (!original_focal) {
        setFocalLossParams(2.0, 1.0, false);
    }
    
    cout << "\nStage 4 completed!" << endl;
    
    // Вывод отчета профилирования для stage 4
    Profiler::getInstance().printReport();
}

// Предсказание
MatrixXd NeuralNetwork::predict(const MatrixXd& X) {
    vector<MatrixXd> activations, z_values;
    forward(X, activations, z_values);
    return activations.back();
}

// Адаптивное уменьшение learning rate
void NeuralNetwork::adjustLearningRate(double factor) {
    learning_rate *= factor;
    cout << "Learning rate reduced to: " << fixed << setprecision(6) << learning_rate << endl;
}

// Сохранение модели (упрощенная версия)
// Методы для динамического морфинга топологии
void NeuralNetwork::addNeuronsToLayer(int layer_idx, int count) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(weights.size())) return;
    
    int old_out = static_cast<int>(weights[layer_idx].rows());
    int old_in = static_cast<int>(weights[layer_idx].cols());
    int new_out = old_out + count;
    
    // 1. Расширяем веса текущего слоя (выход)
    MatrixXd new_w = MatrixXd::Zero(new_out, old_in);
    new_w.topRows(old_out) = weights[layer_idx];
    
    // Инициализация новых весов (He initialization)
    random_device rd;
    mt19937 gen(rd());
    double limit = sqrt(2.0 / old_in);
    normal_distribution<double> dist(0.0, limit);
    for(int i = old_out; i < new_out; ++i)
        for(int j = 0; j < old_in; ++j)
            new_w(i, j) = dist(gen);
            
    weights[layer_idx] = new_w;
    
    // 2. Расширяем смещения
    VectorXd new_b = VectorXd::Zero(new_out);
    new_b.head(old_out) = biases[layer_idx];
    biases[layer_idx] = new_b;
    
    // 3. Расширяем velocity
    velocity_w[layer_idx] = MatrixXd::Zero(new_out, old_in);
    velocity_b[layer_idx] = VectorXd::Zero(new_out);
    
    // 4. Расширяем веса следующего слоя (вход), если он есть
    if (layer_idx + 1 < static_cast<int>(weights.size())) {
        int next_out = static_cast<int>(weights[layer_idx + 1].rows());
        MatrixXd next_w = MatrixXd::Zero(next_out, new_out);
        next_w.leftCols(old_out) = weights[layer_idx + 1];
        
        // Инициализация новых входов следующего слоя
        double next_limit = sqrt(2.0 / new_out);
        normal_distribution<double> next_dist(0.0, next_limit);
        for(int i = 0; i < next_out; ++i)
            for(int j = old_out; j < new_out; ++j)
                next_w(i, j) = next_dist(gen);
                
        weights[layer_idx + 1] = next_w;
        velocity_w[layer_idx + 1] = MatrixXd::Zero(next_out, new_out);
    }
    
    if (layer_idx < static_cast<int>(hidden_sizes.size())) {
        hidden_sizes[layer_idx] += count;
    }
    
    cout << "[NeuralNetwork] Morphed: Added " << count << " neurons to layer " << layer_idx << endl;
}

void NeuralNetwork::addLayer(int neurons) {
    int last_hidden_idx = static_cast<int>(weights.size()) - 2;
    int input_for_new = (last_hidden_idx >= 0) ? static_cast<int>(weights[last_hidden_idx].rows()) : input_size;
    int output_for_new = neurons;
    
    // 1. Создаем новый слой
    random_device rd;
    mt19937 gen(rd());
    double limit = sqrt(2.0 / input_for_new);
    normal_distribution<double> dist(0.0, limit);
    
    MatrixXd new_w = MatrixXd::NullaryExpr(output_for_new, input_for_new, [&]() { return dist(gen); });
    VectorXd new_b = VectorXd::Zero(output_for_new);
    
    // 2. Вставляем перед выходным слоем
    weights.insert(weights.end() - 1, new_w);
    biases.insert(biases.end() - 1, new_b);
    velocity_w.insert(velocity_w.end() - 1, MatrixXd::Zero(output_for_new, input_for_new));
    velocity_b.insert(velocity_b.end() - 1, VectorXd::Zero(output_for_new));
    
    // 3. Переподключаем выходной слой
    int final_layer_idx = static_cast<int>(weights.size()) - 1;
    double final_limit = sqrt(2.0 / output_for_new);
    normal_distribution<double> final_dist(0.0, final_limit);
    
    weights[final_layer_idx] = MatrixXd::NullaryExpr(output_size, output_for_new, [&]() { return final_dist(gen); });
    velocity_w[final_layer_idx] = MatrixXd::Zero(output_size, output_for_new);
    
    hidden_sizes.push_back(neurons);
    cout << "[NeuralNetwork] Morphed: Added new hidden layer with " << neurons << " neurons." << endl;
}

void NeuralNetwork::saveModel(const string& filepath) {
    ofstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error opening file for writing: " << filepath << endl;
        return;
    }
    file << fixed << setprecision(17);
    file << "NeuralNetwork Model" << endl;
    file << input_size << " " << output_size << " " << learning_rate << endl;
    file << hidden_sizes.size();
    for (int h : hidden_sizes) file << " " << h;
    file << endl;
    for (size_t i = 0; i < weights.size(); ++i) {
        file << weights[i].rows() << " " << weights[i].cols() << endl;
        for (int r = 0; r < weights[i].rows(); ++r) {
            for (int c = 0; c < weights[i].cols(); ++c) {
                if (c > 0) file << " ";
                file << weights[i](r, c);
            }
            file << endl;
        }
    }
    for (size_t i = 0; i < biases.size(); ++i) {
        file << biases[i].size() << endl;
        for (int j = 0; j < biases[i].size(); ++j) {
            if (j > 0) file << " ";
            file << biases[i](j);
        }
        file << endl;
    }
    file.close();
    cout << "Model saved to " << filepath << endl;
}

bool NeuralNetwork::loadModelMetadata(const string& filepath, int& out_input_size,
                                       vector<int>& out_hidden_sizes, int& out_output_size, double& out_lr) {
    ifstream file(filepath);
    if (!file.is_open()) return false;
    string line;
    if (!getline(file, line) || line != "NeuralNetwork Model") {
        file.close();
        return false;
    }
    if (!getline(file, line)) {
        file.close();
        return false;
    }
    istringstream iss(line);
    if (!(iss >> out_input_size >> out_output_size >> out_lr)) {
        file.close();
        return false;
    }
    if (!getline(file, line)) {
        file.close();
        return false;
    }
    iss.str(line);
    iss.clear();
    size_t num_hidden;
    if (!(iss >> num_hidden)) {
        file.close();
        return false;
    }
    out_hidden_sizes.resize(num_hidden);
    for (size_t i = 0; i < num_hidden; ++i) {
        if (!(iss >> out_hidden_sizes[i])) {
            file.close();
            return false;
        }
    }
    file.close();
    return true;
}

void NeuralNetwork::loadModel(const string& filepath) {
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error opening file for reading: " << filepath << endl;
        throw invalid_argument("NeuralNetwork::loadModel: Cannot open file " + filepath);
    }
    string line;
    if (!getline(file, line) || line != "NeuralNetwork Model") {
        file.close();
        throw invalid_argument("NeuralNetwork::loadModel: Invalid file format");
    }
    int loaded_input, loaded_output;
    double loaded_lr;
    if (!getline(file, line)) {
        file.close();
        throw invalid_argument("NeuralNetwork::loadModel: Missing metadata");
    }
    istringstream iss(line);
    if (!(iss >> loaded_input >> loaded_output >> loaded_lr)) {
        file.close();
        throw invalid_argument("NeuralNetwork::loadModel: Invalid metadata");
    }
    if (!getline(file, line)) {
        file.close();
        throw invalid_argument("NeuralNetwork::loadModel: Missing architecture");
    }
    iss.str(line);
    iss.clear();
    size_t num_hidden;
    if (!(iss >> num_hidden)) {
        file.close();
        throw invalid_argument("NeuralNetwork::loadModel: Invalid architecture");
    }
    vector<int> loaded_hidden(num_hidden);
    for (size_t i = 0; i < num_hidden; ++i) {
        if (!(iss >> loaded_hidden[i])) {
            file.close();
            throw invalid_argument("NeuralNetwork::loadModel: Invalid hidden sizes");
        }
    }
    if (loaded_input != input_size || loaded_output != output_size || loaded_hidden != hidden_sizes) {
        file.close();
        throw invalid_argument("NeuralNetwork::loadModel: Architecture mismatch (expected " +
            to_string(input_size) + "," + to_string(output_size) + " got " +
            to_string(loaded_input) + "," + to_string(loaded_output) + ")");
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        if (!getline(file, line)) {
            file.close();
            throw invalid_argument("NeuralNetwork::loadModel: Missing weight matrix " + to_string(i));
        }
        istringstream dims(line);
        int rows, cols;
        if (!(dims >> rows >> cols) || rows != (int)weights[i].rows() || cols != (int)weights[i].cols()) {
            file.close();
            throw invalid_argument("NeuralNetwork::loadModel: Weight dimension mismatch at layer " + to_string(i));
        }
        for (int r = 0; r < rows; ++r) {
            if (!getline(file, line)) {
                file.close();
                throw invalid_argument("NeuralNetwork::loadModel: Missing weight data");
            }
            istringstream row_ss(line);
            for (int c = 0; c < cols; ++c) {
                double val;
                if (!(row_ss >> val)) {
                    file.close();
                    throw invalid_argument("NeuralNetwork::loadModel: Invalid weight value");
                }
                weights[i](r, c) = val;
            }
        }
    }
    for (size_t i = 0; i < biases.size(); ++i) {
        if (!getline(file, line)) {
            file.close();
            throw invalid_argument("NeuralNetwork::loadModel: Missing bias size");
        }
        istringstream dims(line);
        int size;
        if (!(dims >> size) || size != (int)biases[i].size()) {
            file.close();
            throw invalid_argument("NeuralNetwork::loadModel: Bias size mismatch at layer " + to_string(i));
        }
        if (!getline(file, line)) {
            file.close();
            throw invalid_argument("NeuralNetwork::loadModel: Missing bias data");
        }
        istringstream data_ss(line);
        for (int j = 0; j < size; ++j) {
            double val;
            if (!(data_ss >> val)) {
                file.close();
                throw invalid_argument("NeuralNetwork::loadModel: Invalid bias value");
            }
            biases[i](j) = val;
        }
    }
    file.close();
    cout << "Model loaded from " << filepath << endl;
}

// Knowledge Transfer: установка весов
void NeuralNetwork::setWeights(const vector<MatrixXd>& new_weights) {
    if (new_weights.size() != weights.size()) {
        throw invalid_argument("NeuralNetwork::setWeights: Size mismatch");
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        if (new_weights[i].rows() != weights[i].rows() || new_weights[i].cols() != weights[i].cols()) {
            throw invalid_argument("NeuralNetwork::setWeights: Dimension mismatch at layer " + to_string(i));
        }
        weights[i] = new_weights[i];
    }
}

// Knowledge Transfer: установка смещений
void NeuralNetwork::setBiases(const vector<MatrixXd>& new_biases) {
    if (new_biases.size() != biases.size()) {
        throw invalid_argument("NeuralNetwork::setBiases: Size mismatch");
    }
    for (size_t i = 0; i < biases.size(); ++i) {
        if (new_biases[i].rows() != biases[i].rows() || new_biases[i].cols() != 1) {
            throw invalid_argument("NeuralNetwork::setBiases: Dimension mismatch at layer " + to_string(i));
        }
        biases[i] = new_biases[i].col(0);
    }
}

// Knowledge Transfer: копирование весов из другой сети с адаптацией размеров
void NeuralNetwork::copyWeightsFrom(const NeuralNetwork& source, double transfer_ratio) {
    const vector<MatrixXd>& src_weights = source.getWeights();
    const vector<MatrixXd>& src_biases = source.getBiases();
    
    // Копируем веса с адаптацией размеров
    for (size_t i = 0; i < min(weights.size(), src_weights.size()); ++i) {
        int min_rows = min(weights[i].rows(), src_weights[i].rows());
        int min_cols = min(weights[i].cols(), src_weights[i].cols());
        
        // Копируем совпадающую часть
        weights[i].block(0, 0, min_rows, min_cols) = 
            transfer_ratio * src_weights[i].block(0, 0, min_rows, min_cols) +
            (1.0 - transfer_ratio) * weights[i].block(0, 0, min_rows, min_cols);
    }
    
    // Копируем смещения (biases это VectorXd, но getBiases возвращает MatrixXd)
    for (size_t i = 0; i < min(biases.size(), src_biases.size()); ++i) {
        int min_size = min(static_cast<int>(biases[i].rows()), static_cast<int>(src_biases[i].rows()));
        // src_biases[i] это MatrixXd, преобразуем в VectorXd
        VectorXd src_bias_vec(src_biases[i].rows());
        if (src_biases[i].cols() == 1) {
            // Если это колонка, копируем данные
            for (int j = 0; j < src_biases[i].rows(); ++j) {
                src_bias_vec(j) = src_biases[i](j, 0);
            }
        } else if (src_biases[i].rows() == src_biases[i].cols()) {
            // Если это квадратная матрица (диагональ), берем диагональ
            src_bias_vec = src_biases[i].diagonal();
        } else {
            // Иначе берем первую колонку
            for (int j = 0; j < src_biases[i].rows(); ++j) {
                src_bias_vec(j) = src_biases[i](j, 0);
            }
        }
        // biases[i] это MatrixXd (колонка), обновляем через col(0)
        biases[i].col(0).head(min_size) = 
            transfer_ratio * src_bias_vec.head(min_size) +
            (1.0 - transfer_ratio) * biases[i].col(0).head(min_size);
    }
}

// Получение архитектуры сети
vector<int> NeuralNetwork::getArchitecture() const {
    vector<int> arch;
    arch.push_back(input_size);
    arch.insert(arch.end(), hidden_sizes.begin(), hidden_sizes.end());
    arch.push_back(output_size);
    return arch;
}

// Сохранение порогов для классов
void NeuralNetwork::saveClassThresholds(const string& path) const {
    ofstream file(path, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file for writing class thresholds: " << path << endl;
        return;
    }
    
    // Сохраняем размер и данные
    int size = static_cast<int>(class_thresholds.size());
    file.write(reinterpret_cast<const char*>(&size), sizeof(int));
    file.write(reinterpret_cast<const char*>(class_thresholds.data()), size * sizeof(double));
    
    file.close();
    cout << "[NeuralNetwork] Class thresholds saved to " << path << endl;
}

// Сохранение весов классов
void NeuralNetwork::saveClassWeights(const string& path) const {
    ofstream file(path, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file for writing class weights: " << path << endl;
        return;
    }
    
    // Сохраняем размер и данные
    int size = static_cast<int>(class_weights.size());
    file.write(reinterpret_cast<const char*>(&size), sizeof(int));
    file.write(reinterpret_cast<const char*>(class_weights.data()), size * sizeof(double));
    
    file.close();
    cout << "[NeuralNetwork] Class weights saved to " << path << endl;
}

