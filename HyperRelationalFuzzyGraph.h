#ifndef HYPER_RELATIONAL_FUZZY_GRAPH_H
#define HYPER_RELATIONAL_FUZZY_GRAPH_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>
#include <set>

// Forward declarations
struct StructureRegion;
struct ShapeDescription;

using namespace cv;
using namespace Eigen;
using namespace std;

// Топологический ориентир (узел графа)
struct TopologicalLandmark {
    Point2f position;           // Позиция ориентира
    string type;                // Тип (например, "wing_center", "tail_tip", "fuselage_center")
    double confidence;          // Уверенность в обнаружении
    int structure_id;           // ID связанной структуры
    VectorXd features;          // Признаки ориентира
};

// Нечеткое пространственное отношение (ребро графа)
struct FuzzySpatialRelation {
    int from_node;              // Индекс узла-источника
    int to_node;                 // Индекс узла-назначения
    string relation_type;        // Тип отношения ("left", "above", "connected", "near", "far")
    double membership;           // Степень принадлежности [0, 1]
    double distance;             // Расстояние между узлами
    double angle;                // Угол между узлами
};

// Гиперреляционный нечеткий граф
class HyperRelationalFuzzyGraph {
public:
    HyperRelationalFuzzyGraph();
    
    // Добавление топологического ориентира
    int addLandmark(const TopologicalLandmark& landmark);
    
    // Добавление нечеткого пространственного отношения
    void addRelation(const FuzzySpatialRelation& relation);
    
    // Построение графа из структур
    void buildFromStructures(const std::vector<StructureRegion>& structures,
                            const ShapeDescription& shape_desc);
    
    // Вычисление признаков графа (векторное представление)
    VectorXd computeGraphFeatures() const;
    
    // Сравнение двух графов (нечеткое сходство)
    double compareGraphs(const HyperRelationalFuzzyGraph& other) const;
    
    // Получение узлов и ребер
    const vector<TopologicalLandmark>& getLandmarks() const { return landmarks; }
    const vector<FuzzySpatialRelation>& getRelations() const { return relations; }
    
    // Вычисление нечетких пространственных отношений
    void computeSpatialRelations();
    
private:
    vector<TopologicalLandmark> landmarks;
    vector<FuzzySpatialRelation> relations;
    
    // Вычисление степени принадлежности для отношения "слева"
    double computeLeftMembership(const TopologicalLandmark& from, 
                                const TopologicalLandmark& to) const;
    
    // Вычисление степени принадлежности для отношения "сверху"
    double computeAboveMembership(const TopologicalLandmark& from,
                                 const TopologicalLandmark& to) const;
    
    // Вычисление степени принадлежности для отношения "соединены"
    double computeConnectedMembership(const TopologicalLandmark& from,
                                      const TopologicalLandmark& to,
                                      const std::vector<StructureRegion>& structures) const;
    
    // Вычисление степени принадлежности для отношения "близко"
    double computeNearMembership(double distance, double max_distance) const;
    
    // Вычисление степени принадлежности для отношения "далеко"
    double computeFarMembership(double distance, double max_distance) const;
};

#endif // HYPER_RELATIONAL_FUZZY_GRAPH_H

