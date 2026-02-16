#ifndef PROFILER_H
#define PROFILER_H

#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <algorithm>

// Класс для профилирования производительности
// Использует RAII паттерн для автоматического измерения времени
class Profiler {
public:
    // Структура для хранения статистики по метрике
    struct MetricStats {
        std::string name;
        double total_time = 0.0;      // Общее время в секундах
        double min_time = 1e10;       // Минимальное время
        double max_time = 0.0;        // Максимальное время
        size_t call_count = 0;        // Количество вызовов
        double avg_time = 0.0;        // Среднее время
        
        void update(double elapsed) {
            total_time += elapsed;
            min_time = std::min(min_time, elapsed);
            max_time = std::max(max_time, elapsed);
            call_count++;
            avg_time = total_time / call_count;
        }
    };
    
    // RAII класс для автоматического измерения времени
    class ScopedTimer {
    public:
        ScopedTimer(const std::string& name, bool enabled = true)
            : name_(name), enabled_(enabled), start_time_(std::chrono::high_resolution_clock::now()) {
        }
        
        ~ScopedTimer() {
            if (enabled_) {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time_).count();
                double elapsed_seconds = duration / 1000000.0;
                Profiler::getInstance().recordMetric(name_, elapsed_seconds);
            }
        }
        
    private:
        std::string name_;
        bool enabled_;
        std::chrono::high_resolution_clock::time_point start_time_;
    };
    
    // Получить singleton экземпляр
    static Profiler& getInstance() {
        static Profiler instance;
        return instance;
    }
    
    // Записать метрику
    void recordMetric(const std::string& name, double elapsed_seconds) {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_[name].update(elapsed_seconds);
    }
    
    // Начать измерение (возвращает RAII таймер)
    static ScopedTimer start(const std::string& name, bool enabled = true) {
        return ScopedTimer(name, enabled && getInstance().isEnabled());
    }
    
    // Получить статистику по метрике
    MetricStats getStats(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = metrics_.find(name);
        if (it != metrics_.end()) {
            return it->second;
        }
        return MetricStats{name};
    }
    
    // Получить все метрики
    std::map<std::string, MetricStats> getAllStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return metrics_;
    }
    
    // Вывести отчет в консоль
    void printReport(std::ostream& os = std::cout) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (metrics_.empty()) {
            os << "[Profiler] No metrics recorded." << std::endl;
            return;
        }
        
        os << "\n" << std::string(80, '=') << std::endl;
        os << "PERFORMANCE PROFILING REPORT" << std::endl;
        os << std::string(80, '=') << std::endl;
        os << std::left << std::setw(40) << "Metric Name"
            << std::right << std::setw(10) << "Calls"
            << std::setw(12) << "Total (s)"
            << std::setw(12) << "Avg (ms)"
            << std::setw(12) << "Min (ms)"
            << std::setw(12) << "Max (ms)" << std::endl;
        os << std::string(80, '-') << std::endl;
        
        // Сортируем по общему времени (убывание)
        std::vector<std::pair<std::string, MetricStats>> sorted_metrics;
        for (const auto& [name, stats] : metrics_) {
            sorted_metrics.push_back({name, stats});
        }
        std::sort(sorted_metrics.begin(), sorted_metrics.end(),
            [](const auto& a, const auto& b) {
                return a.second.total_time > b.second.total_time;
            });
        
        for (const auto& [name, stats] : sorted_metrics) {
            os << std::left << std::setw(40) << name
                << std::right << std::setw(10) << stats.call_count
                << std::setw(12) << std::fixed << std::setprecision(4) << stats.total_time
                << std::setw(12) << std::setprecision(3) << (stats.avg_time * 1000.0)
                << std::setw(12) << std::setprecision(3) << (stats.min_time * 1000.0)
                << std::setw(12) << std::setprecision(3) << (stats.max_time * 1000.0) << std::endl;
        }
        
        os << std::string(80, '=') << std::endl;
        
        // Вычисляем общее время
        double total_time = 0.0;
        for (const auto& [name, stats] : metrics_) {
            total_time += stats.total_time;
        }
        os << "Total profiling time: " << std::fixed << std::setprecision(4) 
           << total_time << " seconds" << std::endl;
        os << std::string(80, '=') << std::endl << std::endl;
    }
    
    // Сохранить отчет в файл
    bool saveReport(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        printReport(file);
        file.close();
        return true;
    }
    
    // Очистить все метрики
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        metrics_.clear();
    }
    
    // Включить/выключить профилирование
    void setEnabled(bool enabled) {
        enabled_ = enabled;
    }
    
    bool isEnabled() const {
        return enabled_;
    }
    
    // Получить процентное распределение времени
    std::map<std::string, double> getTimeDistribution() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::map<std::string, double> distribution;
        
        double total_time = 0.0;
        for (const auto& [name, stats] : metrics_) {
            total_time += stats.total_time;
        }
        
        if (total_time > 0.0) {
            for (const auto& [name, stats] : metrics_) {
                distribution[name] = (stats.total_time / total_time) * 100.0;
            }
        }
        
        return distribution;
    }
    
private:
    Profiler() : enabled_(true) {}
    ~Profiler() = default;
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;
    
    mutable std::mutex mutex_;
    std::map<std::string, MetricStats> metrics_;
    bool enabled_;
};

// Макрос для удобного использования профилирования
#define PROFILE_SCOPE(name) \
    auto _profiler_timer = Profiler::start(name)

#define PROFILE_FUNCTION() \
    PROFILE_SCOPE(__FUNCTION__)

#endif // PROFILER_H

