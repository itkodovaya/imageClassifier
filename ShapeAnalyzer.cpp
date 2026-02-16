#include "ShapeAnalyzer.h"
#include "TopologicalSignature.h"
#include "HyperRelationalFuzzyGraph.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

ShapeAnalyzer::ShapeAnalyzer() 
    : min_structure_area(100.0), prominence_threshold(0.3), max_structures(10) {
}

ShapeDescription ShapeAnalyzer::analyzeShape(const Mat& img) {
    ShapeDescription shape_desc;
    
    // 1. Сегментация для выделения основных областей
    Mat segmented;
    if (img.cols * img.rows < 1000000) {
        segmented = segmentByMeanShift(img);
    } else {
        segmented = segmentByColor(img);
    }
    
    // 2. Предварительный анализ формы на основе сегментированного изображения
    shape_desc = preprocessShape(segmented);
    
    // 3. Выделение структур (выпячиваний)
    shape_desc.structures = extractStructures(img, shape_desc.shape_mask);
    
    return shape_desc;
}

Mat ShapeAnalyzer::segmentByMeanShift(const Mat& img, int sp, int sr) {
    if (img.empty()) return Mat();
    
    Mat res;
    pyrMeanShiftFiltering(img, res, sp, sr);
    return res;
}

Mat ShapeAnalyzer::segmentByColor(const Mat& img, int k) {
    if (img.empty()) return Mat();

    // Преобразование в формат для kmeans
    Mat data;
    img.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    // Выполнение k-means кластеризации
    Mat labels, centers;
    kmeans(data, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 
           3, KMEANS_PP_CENTERS, centers);

    // Создание сегментированного изображения
    centers = centers.reshape(3, centers.rows);
    data = data.reshape(3, data.rows);
    
    Mat segmented = Mat(img.size(), img.type());
    for (int i = 0; i < data.rows; i++) {
        int center_id = labels.at<int>(i);
        segmented.at<Vec3b>(i / img.cols, i % img.cols) = centers.at<Vec3f>(center_id);
    }

    return segmented;
}

MatrixXd ShapeAnalyzer::applyCoverProjection(const MatrixXd& features, int target_dim) {
    int input_dim = static_cast<int>(features.rows());
    
    mt19937 gen(42);
    normal_distribution<double> dist(0.0, 1.0);
    
    MatrixXd W = MatrixXd::NullaryExpr(target_dim, input_dim, [&]() { return dist(gen); });
    VectorXd b = VectorXd::NullaryExpr(target_dim, [&]() { 
        uniform_real_distribution<double> u_dist(0.0, 2.0 * M_PI);
        return u_dist(gen); 
    });
    
    MatrixXd projection = (W * features).colwise() + b;
    return (sqrt(2.0 / target_dim) * projection.array().cos()).matrix();
}

double ShapeAnalyzer::computeStructuralEntropy(const Mat& region) {
    if (region.empty()) return 0.0;
    
    Mat gray;
    if (region.channels() == 3) cvtColor(region, gray, COLOR_BGR2GRAY);
    else gray = region.clone();
    
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    Mat hist;
    calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, &histRange);
    
    hist /= region.total();
    double entropy = 0;
    for (int i = 0; i < histSize; i++) {
        float p = hist.at<float>(i);
        if (p > 0) entropy -= p * log2(p);
    }
    
    return entropy / 8.0;
}

void ShapeAnalyzer::clusterStructuresSOM(int grid_size) {
    if (structure_catalog.empty()) return;

    cout << "[ShapeAnalyzer] Starting SOM clustering for structural catalog..." << endl;

    vector<VectorXd> all_features;
    for (auto& [class_name, structures] : structure_catalog) {
        for (auto& s : structures) {
            VectorXd f(4);
            f << s.area, s.prominence, (double)s.bounding_box.width, (double)s.bounding_box.height;
            all_features.push_back(f);
        }
    }

    if (all_features.empty()) return;

    int num_nodes = grid_size * grid_size;
    vector<VectorXd> nodes(num_nodes);
    for(int i=0; i<num_nodes; ++i) nodes[i] = VectorXd::Random(4);

    double learning_rate = 0.1;
    for (int iter = 0; iter < 100; ++iter) {
        for (const auto& f : all_features) {
            int bmu_idx = 0;
            double min_dist = (nodes[0] - f).norm();
            for (int i = 1; i < num_nodes; ++i) {
                double d = (nodes[i] - f).norm();
                if (d < min_dist) {
                    min_dist = d;
                    bmu_idx = i;
                }
            }
            nodes[bmu_idx] += learning_rate * (f - nodes[bmu_idx]);
        }
        learning_rate *= 0.95;
    }

    for (auto& [class_name, structures] : structure_catalog) {
        for (auto& s : structures) {
            VectorXd f(4);
            f << s.area, s.prominence, (double)s.bounding_box.width, (double)s.bounding_box.height;
            int bmu_idx = 0;
            double min_dist = (nodes[0] - f).norm();
            for (int i = 1; i < num_nodes; ++i) {
                double d = (nodes[i] - f).norm();
                if (d < min_dist) {
                    min_dist = d;
                    bmu_idx = i;
                }
            }
            s.structure_type += "_cluster_" + to_string(bmu_idx);
        }
    }
    cout << "[ShapeAnalyzer] SOM clustering completed. Sub-types assigned." << endl;
}

Mat ShapeAnalyzer::generateAttentionMap(const Mat& img, const vector<StructureRegion>& structures) {
    Mat attention = Mat::zeros(img.size(), CV_8UC1);
    for (const auto& s : structures) {
        int radius = static_cast<int>(sqrt(s.area) * 1.5);
        circle(attention, s.center, radius, Scalar(static_cast<int>(s.prominence * 255)), -1);
    }
    GaussianBlur(attention, attention, Size(51, 51), 0);
    
    Mat heatmap;
    applyColorMap(attention, heatmap, COLORMAP_JET);
    
    Mat result;
    addWeighted(img, 0.6, heatmap, 0.4, 0, result);
    return result;
}

double ShapeAnalyzer::computeFractalDimension(const Mat& region) {
    if (region.empty()) return 0.0;
    
    Mat gray, binary;
    if (region.channels() == 3) cvtColor(region, gray, COLOR_BGR2GRAY);
    else gray = region.clone();
    
    threshold(gray, binary, 0, 255, THRESH_BINARY + THRESH_OTSU);
    
    // Убеждаемся, что binary имеет тип CV_8UC1
    if (binary.type() != CV_8UC1) {
        binary.convertTo(binary, CV_8UC1);
    }
    
    int width = binary.cols;
    int height = binary.rows;
    int max_size = min(width, height);
    
    vector<double> sizes;
    vector<double> counts;
    
    for (int size = 2; size <= max_size / 2; size *= 2) {
        int count = 0;
        for (int y = 0; y < height; y += size) {
            for (int x = 0; x < width; x += size) {
                Rect box(x, y, min(size, width - x), min(size, height - y));
                // Проверяем валидность box перед использованием
                if (box.x >= 0 && box.y >= 0 && box.width > 0 && box.height > 0 &&
                    box.x + box.width <= width && box.y + box.height <= height) {
                    if (countNonZero(binary(box)) > 0) {
                        count++;
                    }
                }
            }
        }
        sizes.push_back(log(1.0 / size));
        counts.push_back(log(count));
    }
    
    if (sizes.size() < 2) return 1.0;
    
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        sum_x += sizes[i];
        sum_y += counts[i];
        sum_xy += sizes[i] * counts[i];
        sum_xx += sizes[i] * sizes[i];
    }
    
    double slope = (sizes.size() * sum_xy - sum_x * sum_y) / (sizes.size() * sum_xx - sum_x * sum_x);
    return abs(slope);
}

double ShapeAnalyzer::computeLyapunovStability(const vector<Point>& contour) {
    if (contour.size() < 10) return 0.0;
    
    try {
        Moments m = moments(contour);
        if (m.m00 < 1e-6) return 0.0;
        
        Point2f center(m.m10 / m.m00, m.m01 / m.m00);
        
        vector<double> radii;
        for (const auto& pt : contour) {
            radii.push_back(norm(Point2f(pt) - center));
        }
        
        double sum_lambda = 0;
        int count = 0;
        for (size_t i = 0; i < radii.size() - 5; ++i) {
            double d0 = abs(radii[i+1] - radii[i]);
            double d1 = abs(radii[i+5] - radii[i+4]);
            if (d0 > 1e-6) {
                sum_lambda += log(max(1e-9, d1 / d0));
                count++;
            }
        }
        
        return count > 0 ? (sum_lambda / count) : 0.0;
    } catch (...) {
        return 0.0;
    }
}

ShapeDescription ShapeAnalyzer::preprocessShape(const Mat& img) {
    ShapeDescription shape_desc;
    
    Mat gray;
    if (img.channels() == 3) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    
    Mat binary;
    threshold(gray, binary, 0, 255, THRESH_BINARY + THRESH_OTSU);
    
    // Убеждаемся, что binary имеет тип CV_8UC1
    if (binary.type() != CV_8UC1) {
        binary.convertTo(binary, CV_8UC1);
    }
    
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);
    morphologyEx(binary, binary, MORPH_OPEN, kernel);
    
    shape_desc.shape_mask = binary;
    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return shape_desc;
    }
    
    auto largest_contour = max_element(contours.begin(), contours.end(),
        [](const vector<Point>& a, const vector<Point>& b) {
            return contourArea(a) < contourArea(b);
        });
    
    Moments moments = cv::moments(*largest_contour);
    if (moments.m00 != 0) {
        shape_desc.centroid = Point2f(moments.m10 / moments.m00, moments.m01 / moments.m00);
    }
    
    Rect bounding_rect = boundingRect(*largest_contour);
    shape_desc.aspect_ratio = static_cast<double>(bounding_rect.width) / max(bounding_rect.height, 1);
    
    double area = contourArea(*largest_contour);
    shape_desc.compactness = computeCompactness(*largest_contour, area);
    shape_desc.elongation = computeElongation(binary);
    
    RotatedRect rotated_rect = minAreaRect(*largest_contour);
    shape_desc.orientation = rotated_rect.angle;
    
    return shape_desc;
}

vector<StructureRegion> ShapeAnalyzer::extractStructures(const Mat& img, const Mat& shape_mask) {
    vector<StructureRegion> structures;
    
    Mat gray;
    if (img.channels() == 3) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    
    Mat masked;
    gray.copyTo(masked, shape_mask);
    
    Mat grad_x, grad_y;
    Sobel(masked, grad_x, CV_64F, 1, 0, 3);
    Sobel(masked, grad_y, CV_64F, 0, 1, 3);
    
    Mat magnitude, angle;
    cartToPolar(grad_x, grad_y, magnitude, angle);
    
    // Конвертируем magnitude в CV_8UC1 перед threshold и findContours
    Mat magnitude_8u;
    // Нормализуем magnitude к диапазону [0, 255] и конвертируем в CV_8UC1
    double min_val, max_val;
    minMaxLoc(magnitude, &min_val, &max_val);
    if (max_val > min_val) {
        magnitude.convertTo(magnitude_8u, CV_8UC1, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
    } else {
        magnitude.convertTo(magnitude_8u, CV_8UC1, 0, 0);
    }
    
    Mat binary_structures;
    threshold(magnitude_8u, binary_structures, 50, 255, THRESH_BINARY);
    
    // Убеждаемся, что binary_structures имеет тип CV_8UC1
    if (binary_structures.type() != CV_8UC1) {
        binary_structures.convertTo(binary_structures, CV_8UC1);
    }
    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    cv::findContours(binary_structures, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area < min_structure_area) {
            continue;
        }
        
        StructureRegion structure;
        structure.contour = contour;
        structure.bounding_box = boundingRect(contour);
        structure.area = area;
        
        Moments m = moments(contour);
        if (m.m00 != 0) {
            structure.center = Point2f(m.m10 / m.m00, m.m01 / m.m00);
        }
        
        structure.prominence = computeProminence(img, contour);
        
        Rect roi = structure.bounding_box;
        roi.x = max(0, roi.x - 5);
        roi.y = max(0, roi.y - 5);
        roi.width = min(img.cols - roi.x, roi.width + 10);
        roi.height = min(img.rows - roi.y, roi.height + 10);
        
        // Проверяем, что ROI валиден перед использованием
        if (roi.x >= 0 && roi.y >= 0 && roi.width > 0 && roi.height > 0 &&
            roi.x + roi.width <= img.cols && roi.y + roi.height <= img.rows) {
            img(roi).copyTo(structure.region_image);
        } else {
            // Если ROI невалиден, используем исходный bounding_box
            Rect safe_roi = structure.bounding_box;
            safe_roi.x = max(0, safe_roi.x);
            safe_roi.y = max(0, safe_roi.y);
            safe_roi.width = min(img.cols - safe_roi.x, safe_roi.width);
            safe_roi.height = min(img.rows - safe_roi.y, safe_roi.height);
            if (safe_roi.width > 0 && safe_roi.height > 0 &&
                safe_roi.x + safe_roi.width <= img.cols && safe_roi.y + safe_roi.height <= img.rows) {
                img(safe_roi).copyTo(structure.region_image);
            } else {
                // Если даже safe_roi невалиден, создаем пустое изображение
                structure.region_image = Mat();
            }
        }
        
        structures.push_back(structure);
    }
    
    structures = filterStructures(structures, prominence_threshold);
    
    if (structures.size() > max_structures) {
        sort(structures.begin(), structures.end(),
            [](const StructureRegion& a, const StructureRegion& b) {
                return a.prominence > b.prominence;
            });
        structures.resize(max_structures);
    }
    
    return structures;
}

// Топологический анализ: вычисление сигнатуры
struct PersistenceDiagram ShapeAnalyzer::computeTopologicalSignature(
    const Mat& img, const vector<StructureRegion>& structures) {
    
    if (!topological_sig) {
        topological_sig = std::make_unique<TopologicalSignature>();
    }
    
    // Вычисляем персистентность для всего изображения
    struct PersistenceDiagram diagram = topological_sig->computeImagePersistence(img, 100, true);
    
    // Добавляем персистентность для каждой структуры (ограничиваем до 8 для скорости)
    const size_t max_structures = 8;
    for (size_t s = 0; s < std::min(structures.size(), max_structures); ++s) {
        const auto& structure = structures[s];
        if (!structure.contour.empty()) {
            struct PersistenceDiagram struct_diag = topological_sig->computeContourPersistence(
                structure.contour, 100.0);
            // Объединяем диаграммы
            diagram.points.insert(diagram.points.end(), 
                                 struct_diag.points.begin(), 
                                 struct_diag.points.end());
        }
    }
    
    return diagram;
}

// Построение гиперреляционного нечеткого графа
class HyperRelationalFuzzyGraph ShapeAnalyzer::buildHyperRelationalGraph(
    const vector<StructureRegion>& structures,
    const ShapeDescription& shape_desc) {
    
    HyperRelationalFuzzyGraph graph;
    graph.buildFromStructures(structures, shape_desc);
    return graph;
}

double ShapeAnalyzer::computeProminence(const Mat& img, const vector<Point>& contour) {
    Mat gray;
    if (img.channels() == 3) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    
    Mat grad_x, grad_y, magnitude, angle;
    Sobel(gray, grad_x, CV_64F, 1, 0, 3);
    Sobel(gray, grad_y, CV_64F, 0, 1, 3);
    cartToPolar(grad_x, grad_y, magnitude, angle);
    
    double boundary_gradient = 0.0;
    int boundary_pixels = 0;
    
    Mat structure_mask = Mat::zeros(img.size(), CV_8UC1);
    fillPoly(structure_mask, vector<vector<Point>>{contour}, Scalar(255));
    
    Mat dilated, boundary_mask;
    dilate(structure_mask, dilated, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
    subtract(dilated, structure_mask, boundary_mask);
    
    for (int y = 0; y < magnitude.rows; ++y) {
        for (int x = 0; x < magnitude.cols; ++x) {
            if (boundary_mask.at<uchar>(y, x) > 0) {
                boundary_gradient += magnitude.at<double>(y, x);
                boundary_pixels++;
            }
        }
    }
    
    if (boundary_pixels == 0) {
        return 0.0;
    }
    
    double avg_boundary_gradient = boundary_gradient / boundary_pixels;
    
    Rect bbox = boundingRect(contour);
    Rect expanded_roi(max(0, bbox.x - bbox.width), max(0, bbox.y - bbox.height),
                      min(img.cols, bbox.width * 3), min(img.rows, bbox.height * 3));
    
    // Проверяем и корректируем expanded_roi перед использованием
    expanded_roi.x = max(0, expanded_roi.x);
    expanded_roi.y = max(0, expanded_roi.y);
    expanded_roi.width = min(magnitude.cols - expanded_roi.x, expanded_roi.width);
    expanded_roi.height = min(magnitude.rows - expanded_roi.y, expanded_roi.height);
    
    // Проверяем валидность ROI
    if (expanded_roi.width > 0 && expanded_roi.height > 0 &&
        expanded_roi.x + expanded_roi.width <= magnitude.cols &&
        expanded_roi.y + expanded_roi.height <= magnitude.rows) {
        Mat surrounding_region = magnitude(expanded_roi);
        Mat structure_region_mask = structure_mask(expanded_roi);
        
        double surrounding_gradient = 0.0;
        int surrounding_pixels = 0;
        
        for (int y = 0; y < surrounding_region.rows; ++y) {
            for (int x = 0; x < surrounding_region.cols; ++x) {
                if (structure_region_mask.at<uchar>(y, x) == 0) {
                    surrounding_gradient += surrounding_region.at<double>(y, x);
                    surrounding_pixels++;
                }
            }
        }
        
        if (surrounding_pixels == 0) {
            return avg_boundary_gradient / 255.0;
        }
        
        double avg_surrounding_gradient = surrounding_gradient / surrounding_pixels;
        
        if (avg_surrounding_gradient < 1e-6) {
            return 1.0;
        }
        
        double prominence = avg_boundary_gradient / max(avg_surrounding_gradient, 1.0);
        return min(prominence / 10.0, 1.0);
    } else {
        // Если ROI невалиден, возвращаем базовую оценку
        return avg_boundary_gradient / 255.0;
    }
}

vector<vector<Point>> ShapeAnalyzer::findContours(const Mat& binary) {
    // Убеждаемся, что binary имеет тип CV_8UC1 перед вызовом findContours
    Mat binary_8u = binary;
    if (binary.type() != CV_8UC1) {
        binary.convertTo(binary_8u, CV_8UC1);
    }
    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    cv::findContours(binary_8u, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    return contours;
}

vector<StructureRegion> ShapeAnalyzer::filterStructures(const vector<StructureRegion>& structures, 
                                                          double min_prominence) {
    vector<StructureRegion> filtered;
    for (const auto& structure : structures) {
        if (structure.prominence >= min_prominence && structure.area >= min_structure_area) {
            filtered.push_back(structure);
        }
    }
    return filtered;
}

double ShapeAnalyzer::computeCompactness(const vector<Point>& contour, double area) {
    double perimeter = arcLength(contour, true);
    if (perimeter < 1e-6) {
        return 0.0;
    }
    return (4.0 * M_PI * area) / (perimeter * perimeter);
}

double ShapeAnalyzer::computeElongation(const Mat& shape_mask) {
    // Убеждаемся, что shape_mask имеет тип CV_8UC1 перед вызовом findContours
    Mat mask_8u = shape_mask;
    if (shape_mask.type() != CV_8UC1) {
        shape_mask.convertTo(mask_8u, CV_8UC1);
    }
    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    cv::findContours(mask_8u, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return 0.0;
    }
    
    RotatedRect rect = minAreaRect(contours[0]);
    double width = rect.size.width;
    double height = rect.size.height;
    
    if (height < 1e-6) {
        return 0.0;
    }
    
    return max(width, height) / min(width, height);
}

vector<Rect> ShapeAnalyzer::divideIntoSectors(const Mat& img, const ShapeDescription& shape) {
    vector<Rect> sectors;
    
    Point2f center = shape.centroid;
    double angle = shape.orientation;
    
    int num_sectors = 8;
    
    for (int i = 0; i < num_sectors; ++i) {
        double sector_angle = (angle + i * 360.0 / num_sectors) * M_PI / 180.0;
        
        int sector_size = min(img.cols, img.rows) / 3;
        Point2f sector_center(
            center.x + cos(sector_angle) * sector_size / 2,
            center.y + sin(sector_angle) * sector_size / 2
        );
        
        Rect sector(
            max(0, static_cast<int>(sector_center.x - sector_size / 2)),
            max(0, static_cast<int>(sector_center.y - sector_size / 2)),
            min(sector_size, img.cols - max(0, static_cast<int>(sector_center.x - sector_size / 2))),
            min(sector_size, img.rows - max(0, static_cast<int>(sector_center.y - sector_size / 2)))
        );
        
        sectors.push_back(sector);
    }
    
    return sectors;
}

string ShapeAnalyzer::classifyStructureType(const StructureRegion& structure, 
                                             const ShapeDescription& shape) {
    Point2f relative_pos(
        (structure.center.x - shape.centroid.x) / shape.aspect_ratio,
        (structure.center.y - shape.centroid.y) / shape.aspect_ratio
    );
    
    double distance_from_center = sqrt(relative_pos.x * relative_pos.x + relative_pos.y * relative_pos.y);
    double angle = atan2(relative_pos.y, relative_pos.x) * 180.0 / M_PI;
    
    if (distance_from_center < 0.3) {
        return "фюзеляж";
    } else if (abs(angle) < 45 || abs(angle) > 135) {
        return "крыло";
    } else if (abs(angle) > 135 || abs(angle) < -135) {
        return "хвост";
    } else {
        return "дополнительный_элемент";
    }
}

void ShapeAnalyzer::addStructureToCatalog(const StructureRegion& structure, const string& class_name) {
    structure_catalog[class_name].push_back(structure);
}

vector<StructureRegion> ShapeAnalyzer::getStructureCatalog(const string& class_name) {
    if (structure_catalog.find(class_name) != structure_catalog.end()) {
        return structure_catalog[class_name];
    }
    return vector<StructureRegion>();
}

void ShapeAnalyzer::saveCatalog(const string& path) {
    ofstream file(path);
    if (!file.is_open()) {
        cerr << "[ShapeAnalyzer] Error: Could not open file for saving catalog: " << path << endl;
        return;
    }

    for (const auto& [class_name, structures] : structure_catalog) {
        file << "CLASS " << class_name << " " << structures.size() << endl;
        for (const auto& struct_region : structures) {
            file << "STRUCT " << struct_region.structure_type << " " 
                 << struct_region.area << " " << struct_region.prominence << " "
                 << struct_region.center.x << " " << struct_region.center.y << " "
                 << struct_region.bounding_box.x << " " << struct_region.bounding_box.y << " "
                 << struct_region.bounding_box.width << " " << struct_region.bounding_box.height << endl;
            
            file << "CONTOUR " << struct_region.contour.size();
            for (const auto& pt : struct_region.contour) {
                file << " " << pt.x << " " << pt.y;
            }
            file << endl;
        }
    }
    file.close();
    cout << "[ShapeAnalyzer] Structure catalog saved to " << path << endl;
}

void ShapeAnalyzer::loadCatalog(const string& path) {
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "[ShapeAnalyzer] Error: Could not open file for loading catalog: " << path << endl;
        return;
    }

    structure_catalog.clear();
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string type;
        ss >> type;
        
        if (type == "CLASS") {
            string class_name;
            int count;
            ss >> class_name >> count;
            
            for (int i = 0; i < count; ++i) {
                StructureRegion struct_region;
                string struct_line;
                
                if (!getline(file, struct_line)) break;
                stringstream ss_struct(struct_line);
                string s_type;
                ss_struct >> s_type >> struct_region.structure_type 
                          >> struct_region.area >> struct_region.prominence
                          >> struct_region.center.x >> struct_region.center.y
                          >> struct_region.bounding_box.x >> struct_region.bounding_box.y
                          >> struct_region.bounding_box.width >> struct_region.bounding_box.height;
                
                if (!getline(file, struct_line)) break;
                stringstream ss_contour(struct_line);
                string c_type;
                int pt_count;
                ss_contour >> c_type >> pt_count;
                for (int j = 0; j < pt_count; ++j) {
                    int x, y;
                    ss_contour >> x >> y;
                    struct_region.contour.push_back(Point(x, y));
                }
                
                structure_catalog[class_name].push_back(struct_region);
            }
        }
    }
    file.close();
    cout << "[ShapeAnalyzer] Structure catalog loaded from " << path << " (" << structure_catalog.size() << " classes)" << endl;
}
