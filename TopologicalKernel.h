#ifndef TOPOLOGICAL_KERNEL_H
#define TOPOLOGICAL_KERNEL_H

#include "TopologicalSignature.h"
#include "HyperRelationalFuzzyGraph.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

using namespace cv;
using namespace Eigen;
using namespace std;

// Топологическое ядро для SVM
class TopologicalKernel {
public:
    TopologicalKernel();
    
    // Вычисление ядра между двумя диаграммами персистентности
    double computeKernel(const PersistenceDiagram& diag1, 
                        const PersistenceDiagram& diag2,
                        double sigma = 1.0);
    
    // Вычисление ядра через расстояние Вассерштейна
    double wassersteinKernel(const PersistenceDiagram& diag1,
                           const PersistenceDiagram& diag2,
                           double sigma = 1.0);
    
    // Вычисление ядра через изображения персистентности
    double persistenceImageKernel(const PersistenceDiagram& diag1,
                                 const PersistenceDiagram& diag2,
                                 int resolution = 50);
    
    // Вычисление ядра между двумя графами
    double graphKernel(const HyperRelationalFuzzyGraph& graph1,
                      const HyperRelationalFuzzyGraph& graph2);
    
    // Комбинированное топологическое ядро (диаграмма + граф)
    double combinedKernel(const PersistenceDiagram& diag1,
                         const HyperRelationalFuzzyGraph& graph1,
                         const PersistenceDiagram& diag2,
                         const HyperRelationalFuzzyGraph& graph2,
                         double alpha = 0.5);
    
private:
    // Вычисление RBF ядра на основе расстояния
    double rbfKernel(double distance, double sigma);
};

#endif // TOPOLOGICAL_KERNEL_H

