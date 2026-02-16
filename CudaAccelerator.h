#ifndef CUDA_ACCELERATOR_H
#define CUDA_ACCELERATOR_H

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <string>

using namespace Eigen;
using namespace std;

// Класс для ускорения вычислений через CUDA
class CudaAccelerator {
public:
    // Инициализация CUDA
    static bool initialize();
    
    // Проверка доступности CUDA
    static bool isAvailable();
    
    // Получить количество доступных GPU
    static int getDeviceCount();
    
    // Установить устройство
    static bool setDevice(int device_id);
    
    // Матричное умножение на GPU
    static MatrixXd matrixMultiplyGPU(const MatrixXd& A, const MatrixXd& B);
    
    // Матричное сложение на GPU
    static MatrixXd matrixAddGPU(const MatrixXd& A, const MatrixXd& B);
    
    // Поэлементное умножение на GPU
    static MatrixXd elementWiseMultiplyGPU(const MatrixXd& A, const MatrixXd& B);
    
    // Активация ReLU на GPU
    static MatrixXd reluGPU(const MatrixXd& X);
    
    // Производная ReLU на GPU
    static MatrixXd reluDerivativeGPU(const MatrixXd& X);
    
    // Softmax на GPU
    static MatrixXd softmaxGPU(const MatrixXd& X);
    
    // Транспонирование на GPU
    static MatrixXd transposeGPU(const MatrixXd& X);
    
    // Сумма по столбцам на GPU
    static MatrixXd colwiseSumGPU(const MatrixXd& X);
    
    // Среднее по столбцам на GPU
    static MatrixXd colwiseMeanGPU(const MatrixXd& X);
    
    // Гибридное вычисление: часть на GPU, часть на CPU
    static MatrixXd hybridCompute(const MatrixXd& A, const MatrixXd& B, double gpu_ratio = 0.7);
    
    // Освобождение ресурсов
    static void cleanup();
    
private:
    static bool cuda_available;
    static bool initialized;
    static int current_device;
    // Храним handle в виде void*, чтобы не тянуть CUDA-типы в заголовок
    static void* cublas_handle;
};

#endif // CUDA_ACCELERATOR_H

