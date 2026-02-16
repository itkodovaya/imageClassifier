#ifndef SHAPE_ANALYZER_H
#define SHAPE_ANALYZER_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <map>
#include "TopologicalSignature.h"
#include "HyperRelationalFuzzyGraph.h"

using namespace cv;
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

// Структура для описания выделенной структуры (выпячивания)
struct StructureRegion {
    Rect bounding_box;           // Ограничивающий прямоугольник
    Point2f center;              // Центр структуры
    double area;                 // Площадь структуры
    double prominence;           // Выпячивание (насколько структура выделяется)
    vector<Point> contour;        // Контур структуры
    Mat region_image;            // Изображение области структуры
    string structure_type;       // Тип структуры (крыло, фюзеляж, хвост и т.д.)
    double confidence;           // Уверенность в выделении
};

// Структура для описания формы образа
struct ShapeDescription {
    Point2f centroid;            // Центроид формы
    double aspect_ratio;         // Соотношение сторон
    double compactness;          // Компактность (отношение площади к периметру)
    double elongation;           // Вытянутость
    double orientation;          // Ориентация (угол поворота)
    vector<StructureRegion> structures;  // Выделенные структуры
    Mat shape_mask;              // Маска формы
};

// Класс для анализа формы образа и выделения структур
class ShapeAnalyzer {
public:
    ShapeAnalyzer();
    
    // Основной метод: анализ формы и выделение структур
    ShapeDescription analyzeShape(const Mat& img);
    
    // Сегментация изображения на основе цвета и схожести пикселей
    Mat segmentByColor(const Mat& img, int k = 5);
    
    // Сегментация методом MeanShift для более гладких областей
    Mat segmentByMeanShift(const Mat& img, int sp = 20, int sr = 40);
    
    // Проекция в пространство высокой размерности (Теорема Ковера)
    MatrixXd applyCoverProjection(const MatrixXd& features, int target_dim = 2048);
    
    // Вычисление структурной энтропии (сложности части)
    double computeStructuralEntropy(const Mat& region);
    
    // Самоорганизующаяся карта (SOM) для кластеризации структур
    void clusterStructuresSOM(int grid_size = 5);
    
    // Визуализация "внимания" (структурная карта)
    Mat generateAttentionMap(const Mat& img, const vector<StructureRegion>& structures);
    
    // Вычисление фрактальной размерности (метод Box-Counting)
    double computeFractalDimension(const Mat& region);
    
    // Анализ стабильности по Ляпунову для контуров (Хаос-анализ)
    double computeLyapunovStability(const vector<Point>& contour);
    
    // Выделение структур (выпячиваний) из формы
    vector<StructureRegion> extractStructures(const Mat& img, const Mat& shape_mask);
    
    // Предварительный анализ формы образа
    ShapeDescription preprocessShape(const Mat& img);
    
    // Разделение изображения на сектора для свертки (не механически, а по форме)
    vector<Rect> divideIntoSectors(const Mat& img, const ShapeDescription& shape);
    
    // Определение типа структуры (крыло, фюзеляж, хвост и т.д.)
    string classifyStructureType(const StructureRegion& structure, const ShapeDescription& shape);
    
    // Создание каталога структур
    void addStructureToCatalog(const StructureRegion& structure, const string& class_name);
    
    // Получение каталога структур для класса
    vector<StructureRegion> getStructureCatalog(const string& class_name);
    
    // Сохранение/загрузка каталога
    void saveCatalog(const string& path);
    void loadCatalog(const string& path);
    
    // Топологический анализ
    PersistenceDiagram computeTopologicalSignature(const Mat& img, const vector<StructureRegion>& structures);
    HyperRelationalFuzzyGraph buildHyperRelationalGraph(const vector<StructureRegion>& structures,
                                                       const ShapeDescription& shape_desc);
    
private:
    std::unique_ptr<TopologicalSignature> topological_sig;
    // Вычисление выпячивания структуры (насколько она выделяется)
    double computeProminence(const Mat& img, const vector<Point>& contour);
    
    // Поиск контуров и выделение областей интереса
    vector<vector<Point>> findContours(const Mat& binary);
    
    // Фильтрация структур по значимости
    vector<StructureRegion> filterStructures(const vector<StructureRegion>& structures, 
                                             double min_prominence = 0.3);
    
    // Вычисление компактности формы
    double computeCompactness(const vector<Point>& contour, double area);
    
    // Вычисление вытянутости
    double computeElongation(const Mat& shape_mask);
    
    // Каталог структур по классам
    map<string, vector<StructureRegion>> structure_catalog;
    
    // Параметры анализа
    double min_structure_area;      // Минимальная площадь структуры
    double prominence_threshold;    // Порог выпячивания
    int max_structures;              // Максимальное количество структур
};

#endif // SHAPE_ANALYZER_H

