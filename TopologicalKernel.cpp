#include "TopologicalKernel.h"
#include <cmath>

TopologicalKernel::TopologicalKernel() {
}

double TopologicalKernel::rbfKernel(double distance, double sigma) {
    return exp(-distance * distance / (2.0 * sigma * sigma));
}

double TopologicalKernel::computeKernel(const PersistenceDiagram& diag1,
                                        const PersistenceDiagram& diag2,
                                        double sigma) {
    // Используем расстояние Вассерштейна для вычисления ядра
    double wasserstein_dist = diag1.wassersteinDistance(diag2, 2.0);
    return rbfKernel(wasserstein_dist, sigma);
}

double TopologicalKernel::wassersteinKernel(const PersistenceDiagram& diag1,
                                           const PersistenceDiagram& diag2,
                                           double sigma) {
    double wasserstein_dist = diag1.wassersteinDistance(diag2, 2.0);
    return rbfKernel(wasserstein_dist, sigma);
}

double TopologicalKernel::persistenceImageKernel(const PersistenceDiagram& diag1,
                                                 const PersistenceDiagram& diag2,
                                                 int resolution) {
    MatrixXd img1 = diag1.toPersistenceImage(resolution);
    MatrixXd img2 = diag2.toPersistenceImage(resolution);
    
    // Вычисляем скалярное произведение (линейное ядро на изображениях)
    double kernel_value = 0.0;
    for (int i = 0; i < img1.rows(); ++i) {
        for (int j = 0; j < img1.cols(); ++j) {
            kernel_value += img1(i, j) * img2(i, j);
        }
    }
    
    // Нормализуем
    double norm1 = sqrt(img1.array().square().sum());
    double norm2 = sqrt(img2.array().square().sum());
    if (norm1 > 1e-6 && norm2 > 1e-6) {
        kernel_value /= (norm1 * norm2);
    }
    
    return kernel_value;
}

double TopologicalKernel::graphKernel(const HyperRelationalFuzzyGraph& graph1,
                                     const HyperRelationalFuzzyGraph& graph2) {
    // Вычисляем сходство графов
    double similarity = graph1.compareGraphs(graph2);
    
    // Преобразуем сходство в ядро (RBF)
    double distance = 1.0 - similarity;
    return rbfKernel(distance, 1.0);
}

double TopologicalKernel::combinedKernel(const PersistenceDiagram& diag1,
                                        const HyperRelationalFuzzyGraph& graph1,
                                        const PersistenceDiagram& diag2,
                                        const HyperRelationalFuzzyGraph& graph2,
                                        double alpha) {
    // Комбинируем ядро диаграммы и ядро графа
    double diag_kernel = wassersteinKernel(diag1, diag2);
    double graph_kernel_val = graphKernel(graph1, graph2);
    
    return alpha * diag_kernel + (1.0 - alpha) * graph_kernel_val;
}

