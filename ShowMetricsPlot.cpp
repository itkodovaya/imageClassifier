// ShowMetricsPlot.cpp - Демонстрация графика метрик обучения

#include "MetricsPlotter.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    cout << "Построение графика метрик обучения..." << endl;
    cout << "Нажмите любую клавишу для закрытия окна." << endl;
    
    MetricsPlotter plotter(1200, 800);
    
    // Создаем демонстрационные данные
    vector<double> epochs;
    vector<double> train_loss;
    vector<double> val_loss;
    vector<double> train_acc;
    vector<double> val_acc;
    
    // Генерируем реалистичные данные обучения
    for (int i = 1; i <= 20; ++i) {
        epochs.push_back(i);
        
        // Loss уменьшается со временем
        double base_loss = 2.5;
        double decay = exp(-i * 0.15);
        train_loss.push_back(base_loss * decay + 0.1 + (rand() % 100) / 1000.0);
        val_loss.push_back(train_loss.back() + 0.1 + (rand() % 100) / 1000.0);
        
        // Accuracy увеличивается со временем
        double base_acc = 0.2;
        double growth = 1.0 - exp(-i * 0.12);
        train_acc.push_back(base_acc + growth * 0.7 + (rand() % 50) / 1000.0);
        val_acc.push_back(train_acc.back() - 0.05 + (rand() % 50) / 1000.0);
        
        // Ограничиваем значения
        if (train_acc.back() > 0.95) train_acc.back() = 0.95;
        if (val_acc.back() > 0.92) val_acc.back() = 0.92;
        if (train_loss.back() < 0.1) train_loss.back() = 0.1;
        if (val_loss.back() < 0.15) val_loss.back() = 0.15;
    }
    
    // Строим график
    plotter.plotLossAndAccuracy(epochs, train_loss, val_loss, train_acc, val_acc, "Training Metrics");
    
    // Сохраняем график
    if (plotter.savePlot("training_plot.png")) {
        cout << "График сохранен: training_plot.png" << endl;
    }
    
    // Показываем график
    namedWindow("Training Metrics", WINDOW_NORMAL);
    resizeWindow("Training Metrics", 1200, 800);
    
    plotter.show("Training Metrics", 0);
    
    cout << "\nГрафик отображается. Нажмите любую клавишу в окне для закрытия..." << endl;
    
    // Ждем нажатия клавиши
    waitKey(0);
    
    destroyAllWindows();
    cout << "График закрыт." << endl;
    
    return 0;
}


