#include "FuzzyLogicFusion.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <random>
#include <iomanip>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

FuzzyLogicFusion::FuzzyLogicFusion() {
}

MatrixXd FuzzyLogicFusion::fuseFeatures(const vector<MatrixXd>& feature_vectors, FusionMethod method) {
    if (feature_vectors.empty()) {
        return MatrixXd::Zero(1, 1);
    }
    
    switch (method) {
        case LOGICAL_AND:
            return logicalAnd(feature_vectors);
        case LOGICAL_OR:
            return logicalOr(feature_vectors);
        case FUZZY_AND:
            return fuzzyAnd(feature_vectors);
        case FUZZY_OR:
            return fuzzyOr(feature_vectors);
        case FUZZY_AVERAGE:
            return fuzzyAverage(feature_vectors);
        case WEIGHTED_FUSION:
            return weightedFusion(feature_vectors, vector<double>());
        case YAGER_FUSION:
            return yagerFusion(feature_vectors);
        case HAMACHER_FUSION:
            return hamacherFusion(feature_vectors);
        default:
            return fuzzyAverage(feature_vectors);
    }
}

MatrixXd FuzzyLogicFusion::fuzzyFuse(const vector<MatrixXd>& feature_vectors,
                                     const vector<double>& weights,
                                     FusionMethod method) {
    return fuseFeatures(feature_vectors, method);
}

MatrixXd FuzzyLogicFusion::logicalAnd(const vector<MatrixXd>& feature_vectors) {
    if (feature_vectors.empty()) {
        return MatrixXd::Zero(1, 1);
    }
    
    // Нормализуем все признаки
    vector<MatrixXd> normalized;
    for (const auto& features : feature_vectors) {
        normalized.push_back(normalizeFeatures(features));
    }
    
    // Логическое И = минимум по каждому признаку
    MatrixXd result = normalized[0];
    
    for (size_t i = 1; i < normalized.size(); ++i) {
        for (int row = 0; row < result.rows(); ++row) {
            for (int col = 0; col < result.cols(); ++col) {
                result(row, col) = min(result(row, col), normalized[i](row, col));
            }
        }
    }
    
    return result;
}

MatrixXd FuzzyLogicFusion::logicalOr(const vector<MatrixXd>& feature_vectors) {
    if (feature_vectors.empty()) {
        return MatrixXd::Zero(1, 1);
    }
    
    // Нормализуем все признаки
    vector<MatrixXd> normalized;
    for (const auto& features : feature_vectors) {
        normalized.push_back(normalizeFeatures(features));
    }
    
    // Логическое ИЛИ = максимум по каждому признаку
    MatrixXd result = normalized[0];
    
    for (size_t i = 1; i < normalized.size(); ++i) {
        for (int row = 0; row < result.rows(); ++row) {
            for (int col = 0; col < result.cols(); ++col) {
                result(row, col) = max(result(row, col), normalized[i](row, col));
            }
        }
    }
    
    return result;
}

MatrixXd FuzzyLogicFusion::fuzzyAnd(const vector<MatrixXd>& feature_vectors, bool use_product) {
    if (feature_vectors.empty()) {
        return MatrixXd::Zero(1, 1);
    }
    
    // Нормализуем все признаки
    vector<MatrixXd> normalized;
    for (const auto& features : feature_vectors) {
        normalized.push_back(normalizeFeatures(features));
    }
    
    MatrixXd result = normalized[0];
    
    if (use_product) {
        // Произведение (t-norm)
        for (size_t i = 1; i < normalized.size(); ++i) {
            for (int row = 0; row < result.rows(); ++row) {
                for (int col = 0; col < result.cols(); ++col) {
                    result(row, col) *= normalized[i](row, col);
                }
            }
        }
    } else {
        // Минимум (стандартная t-norm)
        for (size_t i = 1; i < normalized.size(); ++i) {
            for (int row = 0; row < result.rows(); ++row) {
                for (int col = 0; col < result.cols(); ++col) {
                    result(row, col) = min(result(row, col), normalized[i](row, col));
                }
            }
        }
    }
    
    return result;
}

MatrixXd FuzzyLogicFusion::fuzzyOr(const vector<MatrixXd>& feature_vectors, bool use_probabilistic) {
    if (feature_vectors.empty()) {
        return MatrixXd::Zero(1, 1);
    }
    
    // Нормализуем все признаки
    vector<MatrixXd> normalized;
    for (const auto& features : feature_vectors) {
        normalized.push_back(normalizeFeatures(features));
    }
    
    MatrixXd result = normalized[0];
    
    if (use_probabilistic) {
        // Вероятностная сумма (s-norm): a + b - a*b
        for (size_t i = 1; i < normalized.size(); ++i) {
            for (int row = 0; row < result.rows(); ++row) {
                for (int col = 0; col < result.cols(); ++col) {
                    double a = result(row, col);
                    double b = normalized[i](row, col);
                    result(row, col) = a + b - a * b;
                }
            }
        }
    } else {
        // Максимум (стандартная s-norm)
        for (size_t i = 1; i < normalized.size(); ++i) {
            for (int row = 0; row < result.rows(); ++row) {
                for (int col = 0; col < result.cols(); ++col) {
                    result(row, col) = max(result(row, col), normalized[i](row, col));
                }
            }
        }
    }
    
    return result;
}

MatrixXd FuzzyLogicFusion::fuzzyAverage(const vector<MatrixXd>& feature_vectors, const vector<double>& weights) {
    if (feature_vectors.empty()) {
        return MatrixXd::Zero(1, 1);
    }
    
    // Нормализуем все признаки
    vector<MatrixXd> normalized;
    for (const auto& features : feature_vectors) {
        normalized.push_back(normalizeFeatures(features));
    }
    
    MatrixXd result = MatrixXd::Zero(normalized[0].rows(), normalized[0].cols());
    double total_weight = 0.0;
    
    if (weights.empty() || weights.size() != feature_vectors.size()) {
        // Равномерные веса
        double weight = 1.0 / feature_vectors.size();
        for (const auto& features : normalized) {
            result += features * weight;
        }
    } else {
        // Взвешенное среднее
        total_weight = accumulate(weights.begin(), weights.end(), 0.0);
        if (total_weight > 1e-6) {
            for (size_t i = 0; i < normalized.size(); ++i) {
                result += normalized[i] * (weights[i] / total_weight);
            }
        }
    }
    
    return result;
}

MatrixXd FuzzyLogicFusion::weightedFusion(const vector<MatrixXd>& feature_vectors, const vector<double>& weights) {
    return fuzzyAverage(feature_vectors, weights);
}

MatrixXd FuzzyLogicFusion::yagerFusion(const vector<MatrixXd>& feature_vectors, double p) {
    if (feature_vectors.empty()) return MatrixXd::Zero(1, 1);
    
    MatrixXd result = normalizeFeatures(feature_vectors[0]);
    for (size_t i = 1; i < feature_vectors.size(); ++i) {
        MatrixXd norm_f = normalizeFeatures(feature_vectors[i]);
        for (int r = 0; i < result.rows(); ++r) {
            for (int c = 0; c < result.cols(); ++c) {
                // Yager t-norm: 1 - min(1, ((1-a)^p + (1-b)^p)^(1/p))
                double a = result(r, c);
                double b = norm_f(r, c);
                double val = pow(pow(1.0 - a, p) + pow(1.0 - b, p), 1.0 / p);
                result(r, c) = 1.0 - min(1.0, val);
            }
        }
    }
    return result;
}

MatrixXd FuzzyLogicFusion::hamacherFusion(const vector<MatrixXd>& feature_vectors, double gamma) {
    if (feature_vectors.empty()) return MatrixXd::Zero(1, 1);
    
    MatrixXd result = normalizeFeatures(feature_vectors[0]);
    for (size_t i = 1; i < feature_vectors.size(); ++i) {
        MatrixXd norm_f = normalizeFeatures(feature_vectors[i]);
        for (int r = 0; i < result.rows(); ++r) {
            for (int c = 0; c < result.cols(); ++c) {
                // Hamacher t-norm: (ab) / (gamma + (1-gamma)(a+b-ab))
                double a = result(r, c);
                double b = norm_f(r, c);
                double denom = gamma + (1.0 - gamma) * (a + b - a * b);
                result(r, c) = (denom > 1e-6) ? (a * b / denom) : 0.0;
            }
        }
    }
    return result;
}

MatrixXd FuzzyLogicFusion::semanticInference(const map<string, MatrixXd>& part_predictions, 
                                            const map<string, string>& part_types) {
    // Реализация семантического вывода на основе правил
    // Пример правила: Если есть "крыло" с высокой уверенностью и "фюзеляж", это скорее всего самолет
    
    int num_classes = 0;
    if (!part_predictions.empty()) num_classes = static_cast<int>(part_predictions.begin()->second.cols());
    else return MatrixXd::Zero(1, 1);
    
    MatrixXd semantic_result = MatrixXd::Zero(1, num_classes);
    
    double wing_confidence = 0;
    double fuselage_confidence = 0;
    
    for (const auto& [type, pred] : part_predictions) {
        double max_conf = pred.maxCoeff();
        int best_class_idx = 0;
        for (int i = 1; i < pred.cols(); ++i) {
            if (pred(0, i) > pred(0, best_class_idx)) best_class_idx = i;
        }
        
        // Применяем веса, оптимизированные Генетическим Алгоритмом
        double weight = 1.0;
        if (source_weights.find(type) != source_weights.end()) {
            weight = source_weights.at(type);
        }
        
        double weighted_conf = max_conf * weight;
        
        if (type == "крыло") wing_confidence = max(wing_confidence, weighted_conf);
        if (type == "фюзеляж") fuselage_confidence = max(fuselage_confidence, weighted_conf);
    }
    
    // Нечеткое правило: IF wing AND fuselage THEN high confidence for aircraft classes
    double aircraft_rule = min(wing_confidence, fuselage_confidence);
    
    // Специализированные правила для проблемных классов
    // Правило для птиц: IF wing_shape == curved AND body_ratio > 0.3 THEN bird
    double curved_wing_confidence = 0.0;
    double body_ratio = 0.0;
    
    // Извлекаем признаки из топологических данных (если доступны)
    // Предполагаем, что топологические признаки передаются через part_types
    for (const auto& [type, pred] : part_predictions) {
        if (type.find("wing") != string::npos || type.find("крыло") != string::npos) {
            // Проверяем форму крыла (curved vs straight) через топологические признаки
            // Если есть топологические признаки H1 (отверстия), это может указывать на изогнутое крыло
            double h1_persistence = pred.maxCoeff();  // Упрощенная проверка
            if (h1_persistence > 0.6) {
                curved_wing_confidence = h1_persistence;
            }
        }
        if (type.find("body") != string::npos || type.find("фюзеляж") != string::npos) {
            // Вычисляем body_ratio (отношение длины к ширине)
            body_ratio = pred.maxCoeff();  // Упрощенная проверка
        }
    }
    
    // Правило для птиц
    double bird_rule = min(curved_wing_confidence, (body_ratio > 0.3 ? 1.0 : 0.0));
    if (bird_rule > 0.5) {
        // Ищем класс птиц (обычно это один из последних классов или определяется по имени)
        // Для упрощения применяем к классам с высоким индексом
        for (int i = max(0, num_classes - 3); i < num_classes; ++i) {
            semantic_result(0, i) = max(semantic_result(0, i), bird_rule);
        }
    }
    
    // Правило для двухбалочных: IF two_fuselages AND symmetry THEN twin_tail
    double two_fuselages_confidence = 0.0;
    double symmetry_confidence = 0.0;
    
    // Проверяем наличие двух фюзеляжей через топологические признаки H0 (компоненты связности)
    int fuselage_count = 0;
    for (const auto& [type, pred] : part_predictions) {
        if (type.find("fuselage") != string::npos || type.find("фюзеляж") != string::npos) {
            fuselage_count++;
            two_fuselages_confidence = max(two_fuselages_confidence, pred.maxCoeff());
        }
        if (type.find("symmetry") != string::npos || type.find("симметрия") != string::npos) {
            symmetry_confidence = pred.maxCoeff();
        }
    }
    
    if (fuselage_count >= 2) {
        two_fuselages_confidence = 1.0;
    }
    
    // Правило для двухбалочных
    double twin_tail_rule = min(two_fuselages_confidence, symmetry_confidence);
    if (twin_tail_rule > 0.5) {
        // Ищем класс двухбалочных (обычно определяется по имени или индексу)
        // Для упрощения применяем к классам с высоким индексом
        for (int i = max(0, num_classes - 2); i < num_classes; ++i) {
            semantic_result(0, i) = max(semantic_result(0, i), twin_tail_rule);
        }
    }
    
    if (aircraft_rule > 0.5) {
        // Предположим, что первые несколько классов - это самолеты/БПЛА
        for (int i = 0; i < min(num_classes, 4); ++i) {
            semantic_result(0, i) = max(semantic_result(0, i), aircraft_rule);
        }
    }
    
    return semantic_result;
}

void FuzzyLogicFusion::optimizeRulesGenetic(const vector<map<string, MatrixXd>>& training_preds,
                                          const vector<map<string, string>>& training_types,
                                          const vector<int>& true_labels) {
    // Полноценная реализация Генетического Алгоритма для оптимизации весов правил
    cout << "[FuzzyLogicFusion] Starting Genetic Optimization of fuzzy rules..." << endl;
    
    int pop_size = 30;
    int generations = 20;
    double mutation_rate = 0.1;
    
    vector<string> types = {"крыло", "фюзеляж", "хвост", "дополнительный_элемент"};
    
    // 1. Инициализация популяции
    vector<map<string, double>> population(pop_size);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < pop_size; ++i) {
        for (const auto& t : types) population[i][t] = dist(gen);
    }
    
    for (int g = 0; g < generations; ++g) {
        vector<double> fitness(pop_size, 0.0);
        
        // 2. Оценка фитнеса
        for (int i = 0; i < pop_size; ++i) {
            int correct = 0;
            source_weights = population[i]; // Временные веса
            
            for (size_t j = 0; j < training_preds.size(); ++j) {
                MatrixXd res = semanticInference(training_preds[j], training_types[j]);
                int pred_class = 0;
                for (int k = 1; k < res.cols(); ++k) {
                    if (res(0, k) > res(0, pred_class)) pred_class = k;
                }
                if (pred_class == true_labels[j]) correct++;
            }
            fitness[i] = static_cast<double>(correct) / training_preds.size();
        }
        
        // 3. Селекция (Tournament Selection)
        vector<map<string, double>> next_gen;
        for (int i = 0; i < pop_size; ++i) {
            int a = uniform_int_distribution<int>(0, pop_size - 1)(gen);
            int b = uniform_int_distribution<int>(0, pop_size - 1)(gen);
            next_gen.push_back(fitness[a] > fitness[b] ? population[a] : population[b]);
        }
        
        // 4. Кроссовер (One-Point Crossover)
        for (int i = 0; i < pop_size; i += 2) {
            if (dist(gen) < 0.7) {
                int point = uniform_int_distribution<int>(1, static_cast<int>(types.size()) - 1)(gen);
                for (int j = point; j < static_cast<int>(types.size()); ++j) {
                    swap(next_gen[i][types[j]], next_gen[i+1][types[j]]);
                }
            }
        }
        
        // 5. Мутация (Gaussian Mutation)
        normal_distribution<double> mut_dist_gauss(0.0, 0.1);
        for (int i = 0; i < pop_size; ++i) {
            if (dist(gen) < mutation_rate) {
                for (const auto& t : types) {
                    next_gen[i][t] = max(0.0, min(1.0, next_gen[i][t] + mut_dist_gauss(gen)));
                }
            }
        }
        
        population = next_gen;
        cout << "  Generation " << g << ": Best Fitness = " << *max_element(fitness.begin(), fitness.end()) << endl;
    }
    
    // Выбор лучшего индивида
    int best_idx = 0;
    double best_fit = -1.0;
    for (int i = 0; i < pop_size; ++i) {
        int correct = 0;
        source_weights = population[i];
        for (size_t j = 0; j < training_preds.size(); ++j) {
            MatrixXd res = semanticInference(training_preds[j], training_types[j]);
            int pred_class = 0;
            for (int k = 1; k < res.cols(); ++k) {
                if (res(0, k) > res(0, pred_class)) pred_class = k;
            }
            if (pred_class == true_labels[j]) correct++;
        }
        double fit = static_cast<double>(correct) / training_preds.size();
        if (fit > best_fit) {
            best_fit = fit;
            best_idx = i;
        }
    }
    
    source_weights = population[best_idx];
    cout << "[FuzzyLogicFusion] Genetic Optimization completed. Best Fitness: " << best_fit << endl;
}

MatrixXd FuzzyLogicFusion::quantumInference(const vector<MatrixXd>& predictions) {
    if (predictions.empty()) return MatrixXd::Zero(1, 1);
    
    try {
        int num_classes = static_cast<int>(predictions[0].cols());
        VectorXcd state = VectorXcd::Zero(num_classes);
        
        for (const auto& pred : predictions) {
            if (pred.cols() != num_classes) continue; // Защита от несовпадающих размеров
            
            for (int i = 0; i < num_classes; ++i) {
                double prob = pred(0, i);
                double phase = prob * M_PI; 
                state(i) += complex<double>(prob * cos(phase), prob * sin(phase));
            }
        }
        
        MatrixXd collapsed = MatrixXd::Zero(1, num_classes);
        double total_norm = 0;
        for (int i = 0; i < num_classes; ++i) {
            collapsed(0, i) = norm(state(i));
            total_norm += collapsed(0, i);
        }
        
        if (total_norm > 1e-6) collapsed /= total_norm;
        return collapsed;
    } catch (const std::exception& e) {
        cerr << "[FuzzyLogicFusion] Quantum Inference Error: " << e.what() << endl;
        return predictions[0]; // Fallback к первому предсказанию
    }
}

void FuzzyLogicFusion::visualizeInferenceProcess(const map<string, MatrixXd>& inputs, const MatrixXd& result) {
    cout << "\n" << string(40, '-') << endl;
    cout << "FUZZY INFERENCE TRACE" << endl;
    cout << string(40, '-') << endl;
    
    for (const auto& [name, val] : inputs) {
        double max_conf = val.maxCoeff();
        int class_id = 0;
        for (int i = 1; i < val.cols(); ++i) {
            if (val(0, i) > val(0, class_id)) class_id = i;
        }
        cout << "Source [" << name << "]: Best Class=" << class_id << " (Conf: " << fixed << setprecision(3) << max_conf << ")" << endl;
    }
    
    double final_conf = result.maxCoeff();
    int final_id = 0;
    for (int i = 1; i < result.cols(); ++i) {
        if (result(0, i) > result(0, final_id)) final_id = i;
    }
    cout << string(40, '-') << endl;
    cout << "FINAL FUSED DECISION: Class " << final_id << " (Overall Conf: " << final_conf << ")" << endl;
    cout << string(40, '-') << "\n" << endl;
}

void FuzzyLogicFusion::setSourceWeights(const map<string, double>& weights) {
    source_weights = weights;
}

MatrixXd FuzzyLogicFusion::normalizeFeatures(const MatrixXd& features) {
    MatrixXd normalized = features;
    
    // Нормализация к диапазону [0, 1]
    double min_val = normalized.minCoeff();
    double max_val = normalized.maxCoeff();
    double range = max_val - min_val;
    
    if (range > 1e-6) {
        normalized = (normalized.array() - min_val) / range;
    } else {
        normalized.setZero();
    }
    
    return normalized;
}

double FuzzyLogicFusion::fuzzyMembership(double value, double center, double width) {
    // Треугольная функция принадлежности
    double distance = abs(value - center);
    if (distance > width) {
        return 0.0;
    }
    return 1.0 - (distance / width);
}

