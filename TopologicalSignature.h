#ifndef TOPOLOGICAL_SIGNATURE_H
#define TOPOLOGICAL_SIGNATURE_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace Eigen;
using namespace std;

// Точка в диаграмме персистентности (birth, death)
struct PersistencePoint {
    double birth;      // Рождение топологического признака
    double death;      // Смерть топологического признака
    int dimension;     // Размерность (0 - компоненты, 1 - отверстия, 2 - полости)
    double persistence() const { return death - birth; }
};

// Диаграмма персистентности (barcode)
struct PersistenceDiagram {
    vector<PersistencePoint> points;
    int dimension;
    
    // Вычисление расстояния Вассерштейна до другой диаграммы
    double wassersteinDistance(const PersistenceDiagram& other, double p = 2.0) const;
    
    // Преобразование в изображение персистентности (persistence image)
    MatrixXd toPersistenceImage(int resolution = 50, double bandwidth = 1.0) const;
    
    // Получение топологической сигнатуры (вектор признаков)
    VectorXd toSignature(int num_bins = 20) const;
};

// Класс для вычисления топологической сигнатуры
class TopologicalSignature {
public:
    TopologicalSignature();
    
    // Вычисление диаграммы персистентности через альфа-комплекс
    PersistenceDiagram computeAlphaComplexPersistence(const vector<Point2f>& points, 
                                                     double max_alpha = 100.0);
    
    // Вычисление диаграммы персистентности через рипс-комплекс
    PersistenceDiagram computeRipsComplexPersistence(const vector<Point2f>& points,
                                                    double max_radius = 100.0);
    
    // Вычисление топологической сигнатуры из изображения
    PersistenceDiagram computeImagePersistence(const Mat& img, 
                                              int num_points = 100,
                                              bool use_alpha = true);
    
    // Вычисление топологической сигнатуры из контура
    PersistenceDiagram computeContourPersistence(const vector<Point>& contour,
                                                double max_scale = 100.0);
    
    // Фильтрация по интенсивности для создания фильтрации
    vector<Mat> createIntensityFiltration(const Mat& img, int num_levels = 50);
    
private:
    // Вычисление расстояния между двумя точками
    double distance(const Point2f& p1, const Point2f& p2);
    
    // Построение симплициального комплекса
    vector<vector<int>> buildSimplicialComplex(const vector<Point2f>& points,
                                              const vector<double>& scales);
    
    // Вычисление персистентности для комплекса
    PersistenceDiagram computePersistence(const vector<Point2f>& points,
                                          const vector<vector<int>>& complex,
                                          const vector<double>& birth_times);
};

#endif // TOPOLOGICAL_SIGNATURE_H

