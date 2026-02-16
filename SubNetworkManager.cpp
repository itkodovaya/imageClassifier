#include "SubNetworkManager.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

SubNetworkManager::SubNetworkManager(int num_classes) : num_classes(num_classes) {
}

void SubNetworkManager::createSubNetwork(const string& structure_type, int input_size, int output_size, double entropy) {
    if (sub_networks.find(structure_type) != sub_networks.end()) {
        morphTopology(structure_type, entropy);
        return;
    }
    
    // Определяем архитектуру подсети с учетом энтропии
    // Чем выше энтропия (сложность), тем мощнее архитектура
    int complexity_factor = static_cast<int>(entropy * 1000);
    vector<int> hidden_sizes = NeuralNetwork::determineOptimalArchitecture(input_size, output_size, complexity_factor);
    
    auto sub_net = make_shared<NeuralNetwork>(
        input_size,
        hidden_sizes,
        output_size,
        0.0005,
        "relu",
        0.9,
        0.0001
    );
    
    sub_networks[structure_type] = sub_net;
    cout << "[SubNetworkManager] Created morphic sub-network for: " << structure_type << " (Entropy: " << entropy << ")" << endl;
}

void SubNetworkManager::morphTopology(const string& structure_type, double entropy) {
    if (sub_networks.find(structure_type) == sub_networks.end()) return;
    
    auto& sub_net = sub_networks[structure_type];
    
    // Если энтропия (сложность) высока, добавляем нейроны или слои
    if (entropy > 0.85) {
        cout << "[SubNetworkManager] High entropy (" << entropy << ") detected for " << structure_type << ". Adding new layer." << endl;
        sub_net->addLayer(32); // Добавляем новый слой
    } else if (entropy > 0.6) {
        cout << "[SubNetworkManager] Medium-high entropy (" << entropy << ") detected for " << structure_type << ". Adding neurons." << endl;
        sub_net->addNeuronsToLayer(0, 16); // Добавляем нейроны в первый слой
    }
}

void SubNetworkManager::trainSubNetwork(const string& structure_type,
                                       const vector<MatrixXd>& structure_features,
                                       const vector<int>& labels,
                                       int epochs) {
    if (sub_networks.find(structure_type) == sub_networks.end()) {
        cerr << "[SubNetworkManager] Подсеть для " << structure_type << " не найдена!" << endl;
        return;
    }
    
    auto& sub_net = sub_networks[structure_type];
    
    // Подготовка данных
    int num_samples = structure_features.size();
    if (num_samples == 0) {
        return;
    }
    
    int feature_size = structure_features[0].rows();
    MatrixXd X(num_samples, feature_size);
    MatrixXd y = MatrixXd::Zero(num_samples, num_classes);
    
    for (int i = 0; i < num_samples; ++i) {
        X.row(i) = structure_features[i].transpose();
        if (labels[i] >= 0 && labels[i] < num_classes) {
            y(i, labels[i]) = 1.0;
        }
    }
    
    // Обучение подсети
    cout << "[SubNetworkManager] Обучение подсети " << structure_type 
         << " на " << num_samples << " образцах..." << endl;
    
    double accuracy = sub_net->train_multi_stage(
        X, y,
        0.8,
        8,    // Меньший batch size для подсетей
        epochs,
        5,
        true,
        5,
        true
    );
    
    cout << "[SubNetworkManager] Подсеть " << structure_type 
         << " обучена с точностью: " << (accuracy * 100) << "%" << endl;
}

MatrixXd SubNetworkManager::predictSubNetwork(const string& structure_type, const MatrixXd& features) {
    if (sub_networks.find(structure_type) == sub_networks.end()) {
        return MatrixXd::Zero(1, num_classes);
    }
    
    return sub_networks[structure_type]->predict(features.transpose());
}

shared_ptr<NeuralNetwork> SubNetworkManager::getSubNetwork(const string& structure_type) {
    if (sub_networks.find(structure_type) != sub_networks.end()) {
        return sub_networks[structure_type];
    }
    return nullptr;
}

MatrixXd SubNetworkManager::combineSubNetworkFeatures(const map<string, MatrixXd>& subnetwork_outputs) {
    if (subnetwork_outputs.empty()) {
        return MatrixXd::Zero(1, num_classes);
    }
    
    // Простое усреднение выходов подсетей
    MatrixXd combined = MatrixXd::Zero(1, num_classes);
    int count = 0;
    
    for (const auto& [structure_type, output] : subnetwork_outputs) {
        combined += output;
        count++;
    }
    
    if (count > 0) {
        combined /= count;
    }
    
    return combined;
}

void SubNetworkManager::saveSubNetworks(const string& base_path) {
    for (const auto& [structure_type, sub_net] : sub_networks) {
        string path = base_path + "_" + structure_type + ".model";
        sub_net->saveModel(path);
    }
}

void SubNetworkManager::loadSubNetworks(const string& base_path) {
    namespace fs = std::filesystem;
    string dir = fs::path(base_path).parent_path().string();
    string prefix = fs::path(base_path).filename().string() + "_";

    if (dir.empty()) dir = ".";

    for (const auto& entry : fs::directory_iterator(dir)) {
        string filename = entry.path().filename().string();
        if (filename.find(prefix) == 0 && filename.find(".model") != string::npos) {
            string structure_type = filename.substr(prefix.length(), filename.length() - prefix.length() - 6);
            
            int in_size, out_size;
            double lr;
            vector<int> hidden;
            if (!NeuralNetwork::loadModelMetadata(entry.path().string(), in_size, hidden, out_size, lr)) {
                cerr << "[SubNetworkManager] Skipping " << filename << " (old format)" << endl;
                continue;
            }
            
            auto sub_net = make_shared<NeuralNetwork>(in_size, hidden, out_size, lr, "relu", 0.9, 0.001);
            try {
                sub_net->loadModel(entry.path().string());
                sub_networks[structure_type] = sub_net;
                cout << "[SubNetworkManager] Loaded sub-network for: " << structure_type << endl;
            } catch (...) {
                cerr << "[SubNetworkManager] Failed to load " << filename << endl;
            }
        }
    }
}

bool SubNetworkManager::detectNovelty(const MatrixXd& features, double vigilance) {
    // Реализация Fuzzy ART (Adaptive Resonance Theory) для детекции новизны
    if (sub_networks.empty()) return true;
    
    // 1. Нормализация входа (Complement Coding)
    int dim = static_cast<int>(features.rows());
    VectorXd I(dim * 2);
    for (int i = 0; i < dim; ++i) {
        double val = max(0.0, min(1.0, features(i, 0)));
        I(i) = val;
        I(i + dim) = 1.0 - val;
    }
    
    double max_match = 0.0;
    string best_type = "";
    
    for (const auto& [type, net] : sub_networks) {
        // В Fuzzy ART мы сравниваем вход с весовым вектором категории (прототипом)
        // Здесь мы используем среднее значение весов первого слоя как прототип
        MatrixXd proto = net->predict(features.transpose()); // Используем предсказание как меру отклика
        
        // Вычисляем функцию выбора (Choice Function)
        double norm_I = I.lpNorm<1>();
        double match = proto.maxCoeff(); // Упрощенно: уверенность сети в знакомом образе
        
        // Проверка резонанса (Vigilance Test)
        if (match > max_match) {
            max_match = match;
            best_type = type;
        }
    }
    
    // Если максимальный отклик ниже порога бдительности, объект считается новым
    bool is_novel = (max_match < vigilance);
    
    if (is_novel) {
        cout << "[SubNetworkManager] Novelty Detected! (Max Match: " << max_match << " < " << vigilance << ")" << endl;
    } else {
        cout << "[SubNetworkManager] Familiar structure: " << best_type << " (Match: " << max_match << ")" << endl;
    }
    
    return is_novel;
}

void SubNetworkManager::holographicStore(const string& type, const MatrixXd& full_features) {
    // Переводим признаки в фазовый вектор (голограмму)
    int dim = static_cast<int>(full_features.rows());
    VectorXcd phase_vec(dim);
    for (int i = 0; i < dim; ++i) {
        double angle = full_features(i, 0) * M_PI;
        phase_vec(i) = complex<double>(cos(angle), sin(angle));
    }
    
    // Ассоциативная матрица: M = V * V.adjoint()
    holographic_memory[type] = phase_vec * phase_vec.adjoint();
}

MatrixXd SubNetworkManager::holographicRetrieve(const MatrixXd& fragment_features) {
    if (holographic_memory.empty()) return fragment_features;
    
    int dim = static_cast<int>(fragment_features.rows());
    VectorXcd fragment_phase(dim);
    for (int i = 0; i < dim; ++i) {
        double angle = fragment_features(i, 0) * M_PI;
        fragment_phase(i) = complex<double>(cos(angle), sin(angle));
    }
    
    // Восстановление через суперпозицию всех ассоциаций
    VectorXcd reconstructed = VectorXcd::Zero(dim);
    for (const auto& [type, M] : holographic_memory) {
        reconstructed += M * fragment_phase;
    }
    
    // Перевод обратно в вещественные признаки
    MatrixXd result(dim, 1);
    for (int i = 0; i < dim; ++i) {
        result(i, 0) = abs(reconstructed(i)) / holographic_memory.size();
    }
    
    return result;
}

// Knowledge Transfer: передача весов от основной сети к подсетям
void SubNetworkManager::transferWeightsFromMain(NeuralNetwork* main_network, 
                                                double transfer_ratio) {
    if (!main_network) return;
    
    vector<MatrixXd> main_weights = main_network->getWeights();
    vector<MatrixXd> main_biases = main_network->getBiases();
    
    cout << "[SubNetworkManager] Transferring weights from main network to sub-networks (ratio: " 
         << transfer_ratio << ")..." << endl;
    
    for (auto& [structure_type, sub_net] : sub_networks) {
        if (!sub_net) continue;
        
        try {
            sub_net->copyWeightsFrom(*main_network, transfer_ratio);
            cout << "  - Transferred to " << structure_type << " sub-network" << endl;
        } catch (const exception& e) {
            cout << "  - Warning: Could not transfer to " << structure_type 
                 << " (architecture mismatch): " << e.what() << endl;
        }
    }
}

// Knowledge Transfer: обратная передача весов от подсетей к основной сети
void SubNetworkManager::transferWeightsToMain(NeuralNetwork* main_network,
                                              double transfer_ratio) {
    if (!main_network || sub_networks.empty()) return;
    
    cout << "[SubNetworkManager] Transferring weights from sub-networks to main network (ratio: " 
         << transfer_ratio << ")..." << endl;
    
    // Собираем веса от всех подсетей (взвешенное усреднение)
    vector<MatrixXd> aggregated_weights = main_network->getWeights();
    vector<MatrixXd> aggregated_biases = main_network->getBiases();
    
    int transfer_count = 0;
    for (const auto& [structure_type, sub_net] : sub_networks) {
        if (!sub_net) continue;
        
        try {
            vector<MatrixXd> sub_weights = sub_net->getWeights();
            vector<MatrixXd> sub_biases = sub_net->getBiases();
            
            // Агрегируем веса (взвешенное усреднение)
            for (size_t i = 0; i < min(aggregated_weights.size(), sub_weights.size()); ++i) {
                int min_rows = min(aggregated_weights[i].rows(), sub_weights[i].rows());
                int min_cols = min(aggregated_weights[i].cols(), sub_weights[i].cols());
                
                aggregated_weights[i].block(0, 0, min_rows, min_cols) = 
                    (1.0 - transfer_ratio) * aggregated_weights[i].block(0, 0, min_rows, min_cols) +
                    transfer_ratio * sub_weights[i].block(0, 0, min_rows, min_cols);
            }
            
            for (size_t i = 0; i < min(aggregated_biases.size(), sub_biases.size()); ++i) {
                int min_size = min(static_cast<int>(aggregated_biases[i].rows()), 
                                  static_cast<int>(sub_biases[i].rows()));
                // biases возвращаются как MatrixXd (колонка), используем col(0) для доступа
                aggregated_biases[i].block(0, 0, min_size, 1) = 
                    (1.0 - transfer_ratio) * aggregated_biases[i].block(0, 0, min_size, 1) +
                    transfer_ratio * sub_biases[i].block(0, 0, min_size, 1);
            }
            
            transfer_count++;
        } catch (const exception& e) {
            cout << "  - Warning: Could not transfer from " << structure_type 
                 << ": " << e.what() << endl;
        }
    }
    
    if (transfer_count > 0) {
        main_network->setWeights(aggregated_weights);
        main_network->setBiases(aggregated_biases);
        cout << "  - Aggregated weights from " << transfer_count << " sub-networks" << endl;
    }
}

// Квантовое обучение: синхронизация весов между подсетями и основной сетью
void SubNetworkManager::quantumWeightSync(NeuralNetwork* main_network,
                                         double forward_ratio, double backward_ratio) {
    if (!main_network) return;
    
    cout << "[SubNetworkManager] Quantum Weight Synchronization..." << endl;
    cout << "  Forward (main->sub): " << forward_ratio << ", Backward (sub->main): " << backward_ratio << endl;
    
    // 1. Прямая передача: основная сеть -> подсети
    transferWeightsFromMain(main_network, forward_ratio);
    
    // 2. Обратная передача: подсети -> основная сеть
    transferWeightsToMain(main_network, backward_ratio);
    
    cout << "[SubNetworkManager] Quantum sync completed" << endl;
}

// Создание специализированной подсети для проблемного класса
void SubNetworkManager::createProblemClassSubNetwork(int class_id, int input_size, int output_size,
                                                     const vector<int>& hidden_sizes) {
    string network_key = "problem_class_" + to_string(class_id);
    
    // Если подсеть уже существует, не создаем заново
    if (sub_networks.find(network_key) != sub_networks.end()) {
        cout << "[SubNetworkManager] Sub-network for problem class " << class_id << " already exists" << endl;
        return;
    }
    
    // Определяем архитектуру: более глубокая сеть (4-5 слоев) для проблемных классов
    vector<int> architecture;
    if (hidden_sizes.empty()) {
        // Автоматическая архитектура: 4-5 слоев с уменьшающимся размером
        int base_size = max(input_size / 4, 256);
        architecture = {base_size, base_size * 3 / 4, base_size / 2, base_size / 4};
        if (base_size >= 128) {
            architecture.push_back(base_size / 8);  // 5-й слой если достаточно большой
        }
    } else {
        architecture = hidden_sizes;
    }
    
    // Создаем сеть с более высоким learning rate для проблемных классов
    double learning_rate = 0.002;  // Выше чем обычно
    auto sub_net = make_shared<NeuralNetwork>(
        input_size,
        architecture,
        output_size,
        learning_rate,
        "relu",
        0.9,
        0.0001
    );
    
    sub_networks[network_key] = sub_net;
    cout << "[SubNetworkManager] Created specialized sub-network for problem class " << class_id 
         << " with " << architecture.size() << " hidden layers" << endl;
}

// Предсказание специализированной подсети для проблемного класса
MatrixXd SubNetworkManager::predictProblemClassSubNetwork(int class_id, const MatrixXd& features) {
    string network_key = "problem_class_" + to_string(class_id);
    
    if (sub_networks.find(network_key) == sub_networks.end()) {
        // Если подсеть не существует, возвращаем нулевой вектор
        return MatrixXd::Zero(1, num_classes);
    }
    
    // Выполняем предсказание
    return sub_networks[network_key]->predict(features.transpose());
}

// Обучение специализированной подсети для проблемного класса
void SubNetworkManager::trainProblemClassSubNetwork(int class_id,
                                                    const vector<MatrixXd>& features,
                                                    const vector<int>& labels,
                                                    int epochs) {
    string network_key = "problem_class_" + to_string(class_id);
    
    if (sub_networks.find(network_key) == sub_networks.end()) {
        cerr << "[SubNetworkManager] Error: Sub-network for problem class " << class_id 
             << " does not exist. Create it first using createProblemClassSubNetwork." << endl;
        return;
    }
    
    if (features.empty() || features.size() != labels.size()) {
        cerr << "[SubNetworkManager] Error: Invalid input data for training problem class sub-network." << endl;
        return;
    }
    
    // Преобразуем данные в матрицы
    int num_samples = static_cast<int>(features.size());
    int feature_size = static_cast<int>(features[0].rows());
    
    MatrixXd X(num_samples, feature_size);
    MatrixXd y = MatrixXd::Zero(num_samples, num_classes);
    
    for (int i = 0; i < num_samples; ++i) {
        X.row(i) = features[i].transpose();
        if (labels[i] >= 0 && labels[i] < num_classes) {
            y(i, labels[i]) = 1.0;
        }
    }
    
    // Обучение подсети
    shared_ptr<NeuralNetwork> sub_net = sub_networks[network_key];
    
    cout << "[SubNetworkManager] Training problem class sub-network " << class_id 
         << " on " << num_samples << " samples for " << epochs << " epochs..." << endl;
    
    // Обучение с использованием train_multi_stage для лучшего качества
    // Используем более высокий learning rate через adjustLearningRate (если доступен)
    // или просто обучаем с текущими параметрами
    sub_net->train_multi_stage(
        X, y,
        0.5,  // loss_threshold
        32,   // batch_size
        epochs,
        0,    // stage3_epochs
        true, // use_cross_entropy
        false, // early_stopping (отключаем для проблемных классов)
        true,  // use_focal_loss
        VectorXd() // sample_weights
    );
    
    cout << "[SubNetworkManager] Problem class sub-network " << class_id << " training completed." << endl;
}

