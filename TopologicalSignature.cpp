#include "TopologicalSignature.h"
#include <limits>
#include <queue>
#include <set>
#include <opencv2/imgproc.hpp>
#include <random>
#include <algorithm>

TopologicalSignature::TopologicalSignature() {
}

double TopologicalSignature::distance(const Point2f& p1, const Point2f& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// Вычисление диаграммы персистентности через альфа-комплекс
PersistenceDiagram TopologicalSignature::computeAlphaComplexPersistence(
    const vector<Point2f>& points, double max_alpha) {
    
    PersistenceDiagram diagram;
    diagram.dimension = 1;
    
    if (points.size() < 3) {
        return diagram;
    }
    
    // Ограничение: O(n^3) — при >40 точках слишком медленно
    vector<Point2f> work_points = points;
    const size_t max_pts = 40;
    if (work_points.size() > max_pts) {
        work_points.resize(max_pts);
    }
    
    int n = static_cast<int>(work_points.size());
    vector<double> distances;
    
    // Вычисляем все попарные расстояния
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double d = distance(work_points[i], work_points[j]);
            distances.push_back(d);
        }
    }
    
    sort(distances.begin(), distances.end());
    
    // Создаем фильтрацию: добавляем ребра по возрастанию расстояния
    vector<double> thresholds;
    for (size_t i = 0; i < distances.size() && distances[i] <= max_alpha; ++i) {
        if (thresholds.empty() || distances[i] > thresholds.back() + 0.01) {
            thresholds.push_back(distances[i]);
        }
    }
    
    // Вычисляем персистентность компонент (H0)
    vector<int> parent(n);
    for (int i = 0; i < n; ++i) parent[i] = i;
    
    auto find = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    
    // Каждая точка рождается при alpha = 0
    for (int i = 0; i < n; ++i) {
        PersistencePoint pt;
        pt.birth = 0.0;
        pt.dimension = 0;
        pt.death = numeric_limits<double>::infinity();
        diagram.points.push_back(pt);
    }
    
    // Объединяем компоненты при добавлении ребер
    for (double threshold : thresholds) {
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double d = distance(work_points[i], work_points[j]);
                if (d <= threshold + 1e-6) {
                    int root_i = find(i);
                    int root_j = find(j);
                    
                    if (root_i != root_j) {
                        // Объединяем компоненты
                        parent[root_j] = root_i;
                        
                        // Закрываем одну компоненту (смерть)
                        for (auto& pt : diagram.points) {
                            if (pt.dimension == 0 && pt.death == numeric_limits<double>::infinity()) {
                                pt.death = threshold;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Вычисляем персистентность отверстий (H1) - упрощенная версия
    // Для треугольника вычисляем, когда он закрывается
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            for (int k = j + 1; k < n; ++k) {
                double d1 = distance(work_points[i], work_points[j]);
                double d2 = distance(work_points[j], work_points[k]);
                double d3 = distance(work_points[k], work_points[i]);
                double max_edge = max({d1, d2, d3});
                
                if (max_edge <= max_alpha) {
                    PersistencePoint pt;
                    pt.birth = max_edge;
                    pt.dimension = 1;
                    pt.death = max_edge * 1.5; // Упрощенная оценка
                    diagram.points.push_back(pt);
                }
            }
        }
    }
    
    return diagram;
}

// Вычисление диаграммы персистентности через рипс-комплекс
PersistenceDiagram TopologicalSignature::computeRipsComplexPersistence(
    const vector<Point2f>& points, double max_radius) {
    
    // Рипс-комплекс похож на альфа-комплекс, но использует радиус вместо альфа
    return computeAlphaComplexPersistence(points, max_radius);
}

// Вычисление топологической сигнатуры из изображения
PersistenceDiagram TopologicalSignature::computeImagePersistence(
    const Mat& img, int num_points, bool use_alpha) {
    
    // Извлекаем точки из изображения (например, центры сегментов или ключевые точки)
    vector<Point2f> points;
    
    Mat gray;
    if (img.channels() == 3) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    
    // Убеждаемся, что gray — CV_8UC1 (findContours требует этого)
    if (gray.type() != CV_8UC1) {
        Mat gray8u;
        gray.convertTo(gray8u, CV_8UC1);
        gray = gray8u;
    }
    vector<vector<Point>> contours;
    findContours(gray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    for (const auto& contour : contours) {
        if (contour.size() > 0) {
            Moments m = moments(contour);
            if (m.m00 > 0) {
                Point2f center(static_cast<float>(m.m10 / m.m00), 
                              static_cast<float>(m.m01 / m.m00));
                points.push_back(center);
            }
        }
    }
    
    // Если точек мало, добавляем случайные точки из изображения
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> x_dist(0, img.cols - 1);
    uniform_int_distribution<int> y_dist(0, img.rows - 1);
    
    while (static_cast<int>(points.size()) < num_points) {
        int x = x_dist(gen);
        int y = y_dist(gen);
        if (gray.at<uchar>(y, x) > 128) { // Только яркие точки
            points.push_back(Point2f(static_cast<float>(x), static_cast<float>(y)));
        }
    }
    
    if (points.size() > num_points) {
        points.resize(num_points);
    }
    
    if (use_alpha) {
        return computeAlphaComplexPersistence(points, 100.0);
    } else {
        return computeRipsComplexPersistence(points, 100.0);
    }
}

// Вычисление топологической сигнатуры из контура
PersistenceDiagram TopologicalSignature::computeContourPersistence(
    const vector<Point>& contour, double max_scale) {
    
    // Преобразуем контур в точки Point2f
    vector<Point2f> points;
    for (const auto& pt : contour) {
        points.push_back(Point2f(static_cast<float>(pt.x), static_cast<float>(pt.y)));
    }
    
    return computeAlphaComplexPersistence(points, max_scale);
}

// Преобразование диаграммы в изображение персистентности
MatrixXd PersistenceDiagram::toPersistenceImage(int resolution, double bandwidth) const {
    MatrixXd image = MatrixXd::Zero(resolution, resolution);
    
    if (points.empty()) return image;
    
    // Находим диапазон значений
    double min_birth = points[0].birth;
    double max_death = points[0].death;
    for (const auto& pt : points) {
        min_birth = min(min_birth, pt.birth);
        max_death = max(max_death, pt.death);
    }
    
    double range_birth = max_death - min_birth;
    if (range_birth < 1e-6) range_birth = 1.0;
    
    // Создаем изображение персистентности
    for (const auto& pt : points) {
        double pers = pt.persistence();
        if (pers <= 0) continue;
        
        int x = static_cast<int>((pt.birth - min_birth) / range_birth * (resolution - 1));
        int y = static_cast<int>((pt.death - min_birth) / range_birth * (resolution - 1));
        
        x = max(0, min(resolution - 1, x));
        y = max(0, min(resolution - 1, y));
        
        // Добавляем вес с гауссовым ядром
        for (int i = 0; i < resolution; ++i) {
            for (int j = 0; j < resolution; ++j) {
                double dx = (i - x) / static_cast<double>(resolution) * range_birth;
                double dy = (j - y) / static_cast<double>(resolution) * range_birth;
                double dist_sq = dx * dx + dy * dy;
                image(i, j) += pers * exp(-dist_sq / (2.0 * bandwidth * bandwidth));
            }
        }
    }
    
    return image;
}

// Получение топологической сигнатуры (вектор признаков)
VectorXd PersistenceDiagram::toSignature(int num_bins) const {
    VectorXd signature = VectorXd::Zero(num_bins * 3); // Для H0, H1, H2
    
    if (points.empty()) return signature;
    
    // Находим диапазон персистентности
    double max_pers = 0.0;
    for (const auto& pt : points) {
        double pers = pt.persistence();
        if (pers > 0 && !isinf(pers)) {
            max_pers = max(max_pers, pers);
        }
    }
    
    if (max_pers < 1e-6) return signature;
    
    // Распределяем точки по бинам
    for (const auto& pt : points) {
        double pers = pt.persistence();
        if (pers <= 0 || isinf(pers)) continue;
        
        int dim = pt.dimension;
        if (dim < 0 || dim > 2) continue;
        
        int bin = static_cast<int>((pers / max_pers) * (num_bins - 1));
        bin = max(0, min(num_bins - 1, bin));
        
        signature(dim * num_bins + bin) += 1.0;
    }
    
    // Нормализуем
    double sum = signature.sum();
    if (sum > 1e-6) {
        signature /= sum;
    }
    
    return signature;
}

// Вычисление расстояния Вассерштейна
double PersistenceDiagram::wassersteinDistance(const PersistenceDiagram& other, double p) const {
    // Упрощенная реализация расстояния Вассерштейна
    // Используем оптимальное сопоставление точек
    
    if (points.empty() && other.points.empty()) return 0.0;
    if (points.empty()) {
        double sum = 0.0;
        for (const auto& pt : other.points) {
            double pers = pt.persistence();
            if (pers > 0 && !isinf(pers)) {
                sum += pow(pers, p);
            }
        }
        return pow(sum, 1.0 / p);
    }
    if (other.points.empty()) {
        double sum = 0.0;
        for (const auto& pt : points) {
            double pers = pt.persistence();
            if (pers > 0 && !isinf(pers)) {
                sum += pow(pers, p);
            }
        }
        return pow(sum, 1.0 / p);
    }
    
    // Простое сопоставление: находим ближайшие точки
    double total_distance = 0.0;
    vector<bool> used(other.points.size(), false);
    
    for (const auto& pt1 : points) {
        double min_dist = numeric_limits<double>::max();
        int best_match = -1;
        
        for (size_t j = 0; j < other.points.size(); ++j) {
            if (used[j] || pt1.dimension != other.points[j].dimension) continue;
            
            const auto& pt2 = other.points[j];
            double dist = sqrt(pow(pt1.birth - pt2.birth, 2) + pow(pt1.death - pt2.death, 2));
            if (dist < min_dist) {
                min_dist = dist;
                best_match = static_cast<int>(j);
            }
        }
        
        if (best_match >= 0) {
            used[best_match] = true;
            total_distance += pow(min_dist, p);
        } else {
            // Нет совпадения - добавляем персистентность
            double pers = pt1.persistence();
            if (pers > 0 && !isinf(pers)) {
                total_distance += pow(pers, p);
            }
        }
    }
    
    // Добавляем несовпавшие точки из other
    for (size_t j = 0; j < other.points.size(); ++j) {
        if (!used[j]) {
            double pers = other.points[j].persistence();
            if (pers > 0 && !isinf(pers)) {
                total_distance += pow(pers, p);
            }
        }
    }
    
    return pow(total_distance, 1.0 / p);
}

// Создание фильтрации по интенсивности
vector<Mat> TopologicalSignature::createIntensityFiltration(const Mat& img, int num_levels) {
    vector<Mat> filtration;
    
    Mat gray;
    if (img.channels() == 3) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    
    double min_val, max_val;
    minMaxLoc(gray, &min_val, &max_val);
    
    for (int i = 0; i <= num_levels; ++i) {
        double threshold_val = min_val + (max_val - min_val) * i / num_levels;
        Mat binary;
        threshold(gray, binary, threshold_val, 255, THRESH_BINARY);
        filtration.push_back(binary);
    }
    
    return filtration;
}

