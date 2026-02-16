#ifndef METRICS_PLOTTER_H
#define METRICS_PLOTTER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

// Класс для построения графиков метрик обучения
class MetricsPlotter {
public:
    MetricsPlotter(int width = 1200, int height = 800);
    
    // Построение графика из истории метрик
    void plotMetrics(const map<string, vector<double>>& metrics_history, 
                     const string& window_name = "Training Metrics");
    
    // Построение графика из файла CSV
    bool plotMetricsFromFile(const string& csv_path, 
                             const string& window_name = "Training Metrics");
    
    // Построение графика loss и accuracy
    void plotLossAndAccuracy(const vector<double>& epochs,
                            const vector<double>& train_loss,
                            const vector<double>& val_loss = {},
                            const vector<double>& train_accuracy = {},
                            const vector<double>& val_accuracy = {},
                            const string& window_name = "Training Metrics");
    
    // Построение ROC-кривой и Precision-Recall кривой
    void plotROCAndPR(const vector<double>& fpr, const vector<double>& tpr,
                      const vector<double>& precision, const vector<double>& recall,
                      const string& window_name = "Scientific Validation");
    
    // Построение фазового портрета для Хаос-анализа
    void plotPhasePortrait(const vector<double>& data, const string& title = "Phase Portrait");
    
    // Сохранение графика в файл
    bool savePlot(const string& filename);
    
    // Показать график в окне
    void show(const string& window_name = "Training Metrics", int wait_key = 0);
    
    // Очистить график
    void clear();
    
    // Получить изображение графика
    Mat getPlot() const { return plot_image; }

private:
    Mat plot_image;
    int plot_width;
    int plot_height;
    int margin_left;
    int margin_right;
    int margin_top;
    int margin_bottom;
    int plot_area_width;
    int plot_area_height;
    
    // Цвета для графиков
    Scalar color_train_loss;
    Scalar color_val_loss;
    Scalar color_train_acc;
    Scalar color_val_acc;
    Scalar color_perplexity;
    Scalar color_background;
    Scalar color_grid;
    Scalar color_text;
    Scalar color_axis;
    
    // Вспомогательные функции
    void drawGrid();
    void drawAxes(const string& x_label, const string& y_label);
    void drawLine(const vector<double>& x_data, const vector<double>& y_data,
                  Scalar color, int thickness = 2, const string& label = "");
    void drawLegend(const vector<pair<string, Scalar>>& legend_items);
    pair<double, double> getMinMax(const vector<double>& data);
    void normalizeData(const vector<double>& data, vector<Point2f>& points,
                      double min_val, double max_val, int start_x, int end_x);
    Point2f dataToPixel(double x, double x_min, double x_max,
                       double y, double y_min, double y_max);
};

#endif // METRICS_PLOTTER_H

