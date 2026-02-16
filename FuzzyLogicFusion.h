#ifndef FUZZY_LOGIC_FUSION_H
#define FUZZY_LOGIC_FUSION_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <map>

using namespace Eigen;
using namespace std;

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

// Класс для объединения признаков через логические операции и нечеткую логику
class FuzzyLogicFusion {
public:
    enum FusionMethod {
        LOGICAL_AND,      // Логическое И
        LOGICAL_OR,       // Логическое ИЛИ
        FUZZY_AND,        // Нечеткое И (минимум)
        FUZZY_OR,         // Нечеткое ИЛИ (максимум)
        FUZZY_AVERAGE,    // Нечеткое среднее
        WEIGHTED_FUSION,  // Взвешенное объединение
        YAGER_FUSION,     // Оператор Ягера
        HAMACHER_FUSION   // Оператор Хамахера
    };
    
    FuzzyLogicFusion();
    
    // Объединение признаков от нескольких источников через логические операции
    MatrixXd fuseFeatures(const vector<MatrixXd>& feature_vectors, FusionMethod method);
    
    // Нечеткое объединение с весами
    MatrixXd fuzzyFuse(const vector<MatrixXd>& feature_vectors, 
                      const vector<double>& weights,
                      FusionMethod method = FUZZY_AVERAGE);
    
    // Логическое И для признаков (минимум по каждому признаку)
    MatrixXd logicalAnd(const vector<MatrixXd>& feature_vectors);
    
    // Логическое ИЛИ для признаков (максимум по каждому признаку)
    MatrixXd logicalOr(const vector<MatrixXd>& feature_vectors);
    
    // Нечеткое И (t-norm: минимум или произведение)
    MatrixXd fuzzyAnd(const vector<MatrixXd>& feature_vectors, bool use_product = false);
    
    // Нечеткое ИЛИ (s-norm: максимум или вероятностная сумма)
    MatrixXd fuzzyOr(const vector<MatrixXd>& feature_vectors, bool use_probabilistic = false);
    
    // Нечеткое среднее
    MatrixXd fuzzyAverage(const vector<MatrixXd>& feature_vectors, const vector<double>& weights = {});
    
    // Взвешенное объединение
    MatrixXd weightedFusion(const vector<MatrixXd>& feature_vectors, const vector<double>& weights);
    
    // Оператор Ягера (параметрический t-norm)
    MatrixXd yagerFusion(const vector<MatrixXd>& feature_vectors, double p = 2.0);
    
    // Оператор Хамахера (рациональный t-norm)
    MatrixXd hamacherFusion(const vector<MatrixXd>& feature_vectors, double gamma = 0.5);
    
    // Установка весов для источников признаков
    void setSourceWeights(const map<string, double>& weights);
    
    // Семантический нечеткий вывод: If (part A is X) AND (part B is Y) THEN Class is Z
    MatrixXd semanticInference(const map<string, MatrixXd>& part_predictions, 
                               const map<string, string>& part_types);
    
    // Генетическая оптимизация весов правил
    void optimizeRulesGenetic(const vector<map<string, MatrixXd>>& training_preds,
                             const vector<map<string, string>>& training_types,
                             const vector<int>& true_labels);
    
    // Квантово-вдохновленный нечеткий вывод (суперпозиция гипотез)
    MatrixXd quantumInference(const vector<MatrixXd>& predictions);
    
    // Визуализация процесса вывода (трассировка правил)
    void visualizeInferenceProcess(const map<string, MatrixXd>& inputs, const MatrixXd& result);
    
private:
    map<string, double> source_weights;
    
    // Нормализация признаков к диапазону [0, 1]
    MatrixXd normalizeFeatures(const MatrixXd& features);
    
    // Применение нечеткой функции принадлежности
    double fuzzyMembership(double value, double center, double width);
};

#endif // FUZZY_LOGIC_FUSION_H

