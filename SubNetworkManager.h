#ifndef SUBNETWORK_MANAGER_H
#define SUBNETWORK_MANAGER_H

#include "NeuralNetwork.h"
#include "ShapeAnalyzer.h"
#include <vector>
#include <string>
#include <map>
#include <memory>

using namespace std;
using namespace Eigen;

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

// Класс для управления подсетями для отдельных структур
class SubNetworkManager {
public:
    SubNetworkManager(int num_classes);
    
    // Создание подсети для структуры
    void createSubNetwork(const string& structure_type, int input_size, int output_size, double entropy = 0.5);
    
    // Создание специализированной подсети для проблемного класса
    void createProblemClassSubNetwork(int class_id, int input_size, int output_size, 
                                      const vector<int>& hidden_sizes = vector<int>());
    
    // Динамическое изменение топологии подсети на основе энтропии (сложности)
    void morphTopology(const string& structure_type, double entropy);
    
    // Обучение подсети на структурах определенного типа
    void trainSubNetwork(const string& structure_type, 
                        const vector<MatrixXd>& structure_features,
                        const vector<int>& labels,
                        int epochs = 50);
    
    // Предсказание подсети
    MatrixXd predictSubNetwork(const string& structure_type, const MatrixXd& features);
    
    // Предсказание специализированной подсети для проблемного класса
    MatrixXd predictProblemClassSubNetwork(int class_id, const MatrixXd& features);
    
    // Обучение специализированной подсети для проблемного класса
    void trainProblemClassSubNetwork(int class_id,
                                    const vector<MatrixXd>& features,
                                    const vector<int>& labels,
                                    int epochs = 75);
    
    // Получение подсети
    shared_ptr<NeuralNetwork> getSubNetwork(const string& structure_type);
    
    // Объединение признаков от всех подсетей
    MatrixXd combineSubNetworkFeatures(const map<string, MatrixXd>& subnetwork_outputs);
    
    // Сохранение/загрузка всех подсетей
    void saveSubNetworks(const string& base_path);
    void loadSubNetworks(const string& base_path);
    
    // Детекция новизны (Adaptive Resonance Theory - ART)
    bool detectNovelty(const MatrixXd& features, double vigilance = 0.7);
    
    // Голографическая ассоциативная память (HAM) для восстановления структур
    MatrixXd holographicRetrieve(const MatrixXd& fragment_features);
    void holographicStore(const string& type, const MatrixXd& full_features);
    
    // Knowledge Transfer: передача весов от основной сети к подсетям
    void transferWeightsFromMain(NeuralNetwork* main_network, 
                                 double transfer_ratio = 0.3);
    
    // Knowledge Transfer: обратная передача весов от подсетей к основной сети
    void transferWeightsToMain(NeuralNetwork* main_network,
                               double transfer_ratio = 0.1);
    
    // Квантовое обучение: синхронизация весов между подсетями и основной сетью
    void quantumWeightSync(NeuralNetwork* main_network,
                          double forward_ratio = 0.2, double backward_ratio = 0.1);
    
private:
    map<string, shared_ptr<NeuralNetwork>> sub_networks;
    map<string, MatrixXcd> holographic_memory; // Комплексная память
    int num_classes;
};

#endif // SUBNETWORK_MANAGER_H

