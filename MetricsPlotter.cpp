#include "MetricsPlotter.h"
#include <fstream>
#include <sstream>
#include <iomanip>

MetricsPlotter::MetricsPlotter(int width, int height) 
    : plot_width(width), plot_height(height),
      margin_left(80), margin_right(40), margin_top(60), margin_bottom(80),
      color_train_loss(Scalar(0, 150, 255)),      // Оранжевый
      color_val_loss(Scalar(0, 100, 200)),         // Синий
      color_train_acc(Scalar(0, 200, 0)),         // Зеленый
      color_val_acc(Scalar(0, 150, 150)),         // Бирюзовый
      color_perplexity(Scalar(200, 0, 200)),      // Фиолетовый
      color_background(Scalar(30, 30, 30)),       // Темно-серый фон
      color_grid(Scalar(60, 60, 60)),             // Серые линии сетки
      color_text(Scalar(255, 255, 255)),          // Белый текст
      color_axis(Scalar(200, 200, 200))           // Светло-серые оси
{
    plot_area_width = plot_width - margin_left - margin_right;
    plot_area_height = plot_height - margin_top - margin_bottom;
    clear();
}

void MetricsPlotter::clear() {
    plot_image = Mat(plot_height, plot_width, CV_8UC3, color_background);
}

void MetricsPlotter::drawGrid() {
    // Вертикальные линии (для значений X)
    int num_vertical_lines = 10;
    for (int i = 0; i <= num_vertical_lines; ++i) {
        int x = margin_left + (i * plot_area_width / num_vertical_lines);
        line(plot_image, Point(x, margin_top), 
             Point(x, plot_height - margin_bottom), color_grid, 1);
    }
    
    // Горизонтальные линии (для значений Y)
    int num_horizontal_lines = 8;
    for (int i = 0; i <= num_horizontal_lines; ++i) {
        int y = margin_top + (i * plot_area_height / num_horizontal_lines);
        line(plot_image, Point(margin_left, y), 
             Point(plot_width - margin_right, y), color_grid, 1);
    }
}

void MetricsPlotter::drawAxes(const string& x_label, const string& y_label) {
    // Оси
    line(plot_image, Point(margin_left, plot_height - margin_bottom),
         Point(plot_width - margin_right, plot_height - margin_bottom), 
         color_axis, 2);
    line(plot_image, Point(margin_left, margin_top),
         Point(margin_left, plot_height - margin_bottom), 
         color_axis, 2);
    
    // Подписи осей
    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.6;
    int thickness = 2;
    
    // X-axis label
    Size text_size = getTextSize(x_label, font_face, font_scale, thickness, nullptr);
    putText(plot_image, x_label,
            Point((plot_width - text_size.width) / 2, plot_height - 20),
            font_face, font_scale, color_text, thickness);
    
    // Y-axis label (повернутый)
    text_size = getTextSize(y_label, font_face, font_scale, thickness, nullptr);
    Mat rotated;
    plot_image.copyTo(rotated);
    putText(rotated, y_label, Point(20, (plot_height + text_size.width) / 2),
            font_face, font_scale, color_text, thickness);
    // Поворот текста для Y-оси (упрощенная версия)
    putText(plot_image, y_label, Point(10, plot_height / 2),
            font_face, font_scale, color_text, thickness);
}

Point2f MetricsPlotter::dataToPixel(double x, double x_min, double x_max,
                                   double y, double y_min, double y_max) {
    double x_range = (x_max - x_min);
    double y_range = (y_max - y_min);
    
    if (x_range < 1e-10) x_range = 1.0;
    if (y_range < 1e-10) y_range = 1.0;
    
    double x_pixel = margin_left + ((x - x_min) / x_range) * plot_area_width;
    double y_pixel = plot_height - margin_bottom - 
                     ((y - y_min) / y_range) * plot_area_height;
    return Point2f(static_cast<float>(x_pixel), static_cast<float>(y_pixel));
}

pair<double, double> MetricsPlotter::getMinMax(const vector<double>& data) {
    if (data.empty()) return make_pair(0.0, 1.0);
    
    double min_val = *min_element(data.begin(), data.end());
    double max_val = *max_element(data.begin(), data.end());
    
    // Добавляем небольшой отступ
    double range = max_val - min_val;
    if (range < 1e-10) {
        min_val -= 0.1;
        max_val += 0.1;
    } else {
        min_val -= range * 0.05;
        max_val += range * 0.05;
    }
    
    return make_pair(min_val, max_val);
}

void MetricsPlotter::drawLine(const vector<double>& x_data, 
                              const vector<double>& y_data,
                              Scalar color, int thickness, const string& label) {
    if (x_data.empty() || y_data.empty() || x_data.size() != y_data.size()) {
        return;
    }
    
    auto x_range = getMinMax(x_data);
    auto y_range = getMinMax(y_data);
    
    vector<Point2f> points;
    for (size_t i = 0; i < x_data.size(); ++i) {
        Point2f pt = dataToPixel(x_data[i], x_range.first, x_range.second,
                                 y_data[i], y_range.first, y_range.second);
        points.push_back(pt);
    }
    
    // Рисуем линии
    for (size_t i = 0; i < points.size() - 1; ++i) {
        line(plot_image, Point(points[i].x, points[i].y),
             Point(points[i+1].x, points[i+1].y), color, thickness);
    }
    
    // Рисуем точки
    for (const auto& pt : points) {
        circle(plot_image, Point(pt.x, pt.y), 3, color, -1);
    }
}

void MetricsPlotter::drawLegend(const vector<pair<string, Scalar>>& legend_items) {
    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.5;
    int thickness = 1;
    int start_x = plot_width - margin_right - 150;
    int start_y = margin_top + 20;
    int line_height = 25;
    
    for (size_t i = 0; i < legend_items.size(); ++i) {
        int y = start_y + i * line_height;
        
        // Цветная линия
        line(plot_image, Point(start_x, y), Point(start_x + 20, y),
             legend_items[i].second, 2);
        
        // Текст
        putText(plot_image, legend_items[i].first,
                Point(start_x + 25, y + 5),
                font_face, font_scale, color_text, thickness);
    }
}

void MetricsPlotter::plotLossAndAccuracy(const vector<double>& epochs,
                                        const vector<double>& train_loss,
                                        const vector<double>& val_loss,
                                        const vector<double>& train_accuracy,
                                        const vector<double>& val_accuracy,
                                        const string& window_name) {
    clear();
    drawGrid();
    
    // Определяем диапазоны для двух Y-осей (loss и accuracy)
    auto loss_range = getMinMax(train_loss);
    if (!val_loss.empty()) {
        auto val_loss_range = getMinMax(val_loss);
        loss_range.first = min(loss_range.first, val_loss_range.first);
        loss_range.second = max(loss_range.second, val_loss_range.second);
    }
    
    auto epoch_range = getMinMax(epochs);
    
    // Рисуем loss
    if (!train_loss.empty() && train_loss.size() == epochs.size()) {
        drawLine(epochs, train_loss, color_train_loss, 2, "Train Loss");
    }
    if (!val_loss.empty() && val_loss.size() == epochs.size()) {
        drawLine(epochs, val_loss, color_val_loss, 2, "Val Loss");
    }
    
    // Рисуем accuracy (если есть) - используем правую Y-ось
    if (!train_accuracy.empty() && train_accuracy.size() == epochs.size()) {
        // Нормализуем accuracy к диапазону loss для визуализации
        auto acc_range = getMinMax(train_accuracy);
        vector<double> normalized_acc;
        for (double acc : train_accuracy) {
            double normalized = loss_range.first + 
                (acc - acc_range.first) / (acc_range.second - acc_range.first) *
                (loss_range.second - loss_range.first);
            normalized_acc.push_back(normalized);
        }
        drawLine(epochs, normalized_acc, color_train_acc, 2, "Train Acc");
    }
    if (!val_accuracy.empty() && val_accuracy.size() == epochs.size()) {
        auto acc_range = getMinMax(val_accuracy);
        vector<double> normalized_acc;
        for (double acc : val_accuracy) {
            double normalized = loss_range.first + 
                (acc - acc_range.first) / (acc_range.second - acc_range.first) *
                (loss_range.second - loss_range.first);
            normalized_acc.push_back(normalized);
        }
        drawLine(epochs, normalized_acc, color_val_acc, 2, "Val Acc");
    }
    
    // Подписи осей
    drawAxes("Epoch", "Loss / Accuracy");
    
    // Легенда
    vector<pair<string, Scalar>> legend;
    if (!train_loss.empty()) legend.push_back({"Train Loss", color_train_loss});
    if (!val_loss.empty()) legend.push_back({"Val Loss", color_val_loss});
    if (!train_accuracy.empty()) legend.push_back({"Train Acc", color_train_acc});
    if (!val_accuracy.empty()) legend.push_back({"Val Acc", color_val_acc});
    drawLegend(legend);
    
    // Подписи значений на осях
    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.4;
    int thickness = 1;
    
    // X-axis labels
    for (int i = 0; i <= 10; ++i) {
        double value = epoch_range.first + 
                      (epoch_range.second - epoch_range.first) * i / 10.0;
        int x = margin_left + (i * plot_area_width / 10);
        string label = to_string(static_cast<int>(value));
        Size text_size = getTextSize(label, font_face, font_scale, thickness, nullptr);
        putText(plot_image, label,
                Point(x - text_size.width / 2, plot_height - margin_bottom + 20),
                font_face, font_scale, color_text, thickness);
    }
    
    // Y-axis labels (loss)
    for (int i = 0; i <= 8; ++i) {
        double value = loss_range.first + 
                      (loss_range.second - loss_range.first) * (8 - i) / 8.0;
        int y = margin_top + (i * plot_area_height / 8);
        string label = to_string(value).substr(0, 5);
        putText(plot_image, label,
                Point(margin_left - 50, y + 5),
                font_face, font_scale, color_text, thickness);
    }
}

void MetricsPlotter::plotMetrics(const map<string, vector<double>>& metrics_history,
                                 const string& window_name) {
    if (metrics_history.empty()) {
        clear();
        putText(plot_image, "No metrics data available",
                Point(plot_width / 2 - 150, plot_height / 2),
                FONT_HERSHEY_SIMPLEX, 0.8, color_text, 2);
        return;
    }
    
    vector<double> epochs;
    vector<double> train_loss, val_loss;
    vector<double> train_acc, val_acc;
    vector<double> perplexity;
    
    // Извлекаем данные
    if (metrics_history.find("epoch") != metrics_history.end()) {
        epochs = metrics_history.at("epoch");
    } else if (!metrics_history.empty()) {
        // Если нет epochs, создаем последовательность
        size_t max_size = 0;
        for (const auto& pair : metrics_history) {
            max_size = max(max_size, pair.second.size());
        }
        for (size_t i = 0; i < max_size; ++i) {
            epochs.push_back(i + 1);
        }
    }
    
    if (metrics_history.find("train_loss") != metrics_history.end()) {
        train_loss = metrics_history.at("train_loss");
    }
    if (metrics_history.find("val_loss") != metrics_history.end()) {
        val_loss = metrics_history.at("val_loss");
    }
    if (metrics_history.find("train_accuracy") != metrics_history.end()) {
        train_acc = metrics_history.at("train_accuracy");
    }
    if (metrics_history.find("val_accuracy") != metrics_history.end()) {
        val_acc = metrics_history.at("val_accuracy");
    }
    if (metrics_history.find("perplexity") != metrics_history.end()) {
        perplexity = metrics_history.at("perplexity");
    }
    
    // Нормализуем размеры векторов
    size_t max_size = epochs.size();
    if (!train_loss.empty()) max_size = max(max_size, train_loss.size());
    if (!val_loss.empty()) max_size = max(max_size, val_loss.size());
    
    if (epochs.size() < max_size) {
        for (size_t i = epochs.size(); i < max_size; ++i) {
            epochs.push_back(i + 1);
        }
    }
    
    plotLossAndAccuracy(epochs, train_loss, val_loss, train_acc, val_acc, window_name);
}

bool MetricsPlotter::plotMetricsFromFile(const string& csv_path,
                                         const string& window_name) {
    ifstream file(csv_path);
    if (!file.is_open()) {
        return false;
    }
    
    map<string, vector<double>> metrics;
    string line;
    
    // Пропускаем заголовок
    if (getline(file, line)) {
        // Парсим заголовок для определения колонок
    }
    
    // Читаем данные
    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        vector<string> tokens;
        
        while (getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        
        if (tokens.size() >= 2) {
            // Предполагаем формат: epoch,train_loss,val_loss,perplexity
            if (metrics.find("epoch") == metrics.end()) {
                metrics["epoch"] = vector<double>();
                metrics["train_loss"] = vector<double>();
                metrics["val_loss"] = vector<double>();
                metrics["perplexity"] = vector<double>();
            }
            
            if (tokens.size() > 0) metrics["epoch"].push_back(stod(tokens[0]));
            if (tokens.size() > 1) metrics["train_loss"].push_back(stod(tokens[1]));
            if (tokens.size() > 2) metrics["val_loss"].push_back(stod(tokens[2]));
            if (tokens.size() > 3) metrics["perplexity"].push_back(stod(tokens[3]));
        }
    }
    
    file.close();
    
    if (!metrics.empty()) {
        plotMetrics(metrics, window_name);
        return true;
    }
    
    return false;
}

void MetricsPlotter::plotROCAndPR(const vector<double>& fpr, const vector<double>& tpr,
                                  const vector<double>& precision, const vector<double>& recall,
                                  const string& window_name) {
    clear();
    drawGrid();
    
    // Рисуем ROC-кривую (FPR vs TPR)
    if (!fpr.empty() && !tpr.empty()) {
        drawLine(fpr, tpr, color_train_acc, 3, "ROC Curve");
        // Диагональ случайного классификатора
        line(plot_image, dataToPixel(0,0,1,0,0,1), dataToPixel(1,0,1,1,0,1), color_grid, 1, LINE_AA);
    }
    
    // Рисуем PR-кривую (Recall vs Precision)
    if (!recall.empty() && !precision.empty()) {
        drawLine(recall, precision, color_train_loss, 3, "PR Curve");
    }
    
    drawAxes("False Positive Rate / Recall", "True Positive Rate / Precision");
    
    vector<pair<string, Scalar>> legend = {
        {"ROC (FPR vs TPR)", color_train_acc},
        {"PR (Recall vs Precision)", color_train_loss}
    };
    drawLegend(legend);
}

void MetricsPlotter::plotPhasePortrait(const vector<double>& data, const string& title) {
    clear();
    drawGrid();
    
    if (data.size() < 2) return;
    
    // Фазовый портрет: x(t) vs x(t+1)
    vector<double> x, y;
    for (size_t i = 0; i < data.size() - 1; ++i) {
        x.push_back(data[i]);
        y.push_back(data[i+1]);
    }
    
    // Находим границы для нормализации
    auto minmax_x = getMinMax(x);
    auto minmax_y = getMinMax(y);
    double min_val = min(minmax_x.first, minmax_y.first);
    double max_val = max(minmax_x.second, minmax_y.second);
    
    // Рисуем траекторию
    for (size_t i = 0; i < x.size() - 1; ++i) {
        Point2f p1 = dataToPixel(x[i], min_val, max_val, y[i], min_val, max_val);
        Point2f p2 = dataToPixel(x[i+1], min_val, max_val, y[i+1], min_val, max_val);
        line(plot_image, p1, p2, color_train_acc, 1, LINE_AA);
    }
    
    drawAxes("x(t)", "x(t+1)");
    putText(plot_image, title, Point(margin_left, margin_top - 10), 
            FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2);
}

bool MetricsPlotter::savePlot(const string& filename) {
    if (plot_image.empty()) {
        return false;
    }
    return imwrite(filename, plot_image);
}

void MetricsPlotter::show(const string& window_name, int wait_key) {
    if (plot_image.empty()) {
        return;
    }
    imshow(window_name, plot_image);
    if (wait_key >= 0) {
        waitKey(wait_key);
    }
}

