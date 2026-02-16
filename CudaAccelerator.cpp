#include "CudaAccelerator.h"
#include <iostream>
#include <thread>
#include <future>
#include <algorithm>

// Пробуем подключить CUDA runtime и cuBLAS, если доступны в системе.
// Это позволяет собирать проект даже без установленного CUDA Toolkit.
#if defined(__has_include)
#  if __has_include(<cuda_runtime.h>)
#    include <cuda_runtime.h>
#    include <cublas_v2.h>
#    define CUDA_RUNTIME_AVAILABLE
#  endif
#endif

// Статические переменные
bool CudaAccelerator::cuda_available = false;
bool CudaAccelerator::initialized = false;
int  CudaAccelerator::current_device = 0;
void* CudaAccelerator::cublas_handle = nullptr;

// Проверка доступности CUDA через CUDA Runtime (предпочтительно),
// при отсутствии CUDA - возвращаем false и оставляем CPU-режим.
bool CudaAccelerator::isAvailable() {
#ifdef CUDA_RUNTIME_AVAILABLE
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        if (err != cudaSuccess) {
            std::cout << "[CUDA] cudaGetDeviceCount: " << cudaGetErrorString(err) << " (code " << static_cast<int>(err) << ")\n";
        }
        return false;
    }
    // Пробуем выбрать первое устройство
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cout << "[CUDA] cudaSetDevice(0): " << cudaGetErrorString(err) << "\n";
        return false;
    }
    return true;
#else
    // CUDA runtime недоступен - работаем только на CPU
    return false;
#endif
}

bool CudaAccelerator::initialize() {
    if (initialized) {
        return cuda_available;
    }

    cuda_available = isAvailable();
    initialized = true;

#ifdef CUDA_RUNTIME_AVAILABLE
    if (cuda_available) {
        // Инициализируем cuBLAS
        cublasHandle_t handle = nullptr;
        cublasStatus_t status = cublasCreate(&handle);
        if (status == CUBLAS_STATUS_SUCCESS && handle != nullptr) {
            cublas_handle = handle;

            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
                std::cout << "[CUDA] Используется устройство: " << prop.name
                          << " (SM " << prop.major << "." << prop.minor << ")\n";
            } else {
                std::cout << "[CUDA] CUDA ускорение доступно и инициализировано\n";
            }
        } else {
            // Если не удалось создать cuBLAS handle, откатываемся на CPU
            cublas_handle = nullptr;
            cuda_available = false;
            std::cout << "[CUDA] Ошибка инициализации cuBLAS, используется CPU режим\n";
        }
    } else {
        std::cout << "[CUDA] CUDA nedostupna, CPU rezhim\n";
    }
#else
    std::cout << "[CUDA] CUDA runtime ne nayden, CPU rezhim\n";
#endif

    return cuda_available;
}

int CudaAccelerator::getDeviceCount() {
#ifdef CUDA_RUNTIME_AVAILABLE
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        return 0;
    }
    return device_count;
#else
    // Возвращаем количество CPU потоков для гибридного режима
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
#endif
}

bool CudaAccelerator::setDevice(int device_id) {
#ifdef CUDA_RUNTIME_AVAILABLE
    cudaError_t err = cudaSetDevice(device_id);
    if (err == cudaSuccess) {
        current_device = device_id;
        return true;
    }
    return false;
#else
    current_device = device_id;
    return true;
#endif
}

// Гибридное вычисление: часть на GPU (если доступно), часть на CPU
// Использует многопоточность для параллельной обработки
MatrixXd CudaAccelerator::hybridCompute(const MatrixXd& A, const MatrixXd& B, double gpu_ratio) {
    if (A.rows() < 50) {
        // Для маленьких матриц - используем простой CPU
        return A * B;
    }
    
    // Определяем количество потоков для параллельной обработки
    int num_threads = max(2, static_cast<int>(thread::hardware_concurrency()));
    int total_rows = static_cast<int>(A.rows());
    
    // Разделяем работу между потоками (имитация GPU + CPU)
    int gpu_threads = max(1, static_cast<int>(num_threads * gpu_ratio));
    int cpu_threads = num_threads - gpu_threads;
    
    // Если CUDA недоступна, используем все потоки как CPU
    if (!cuda_available) {
        gpu_threads = 0;
        cpu_threads = num_threads;
    }
    
    vector<future<MatrixXd>> futures;
    vector<int> thread_rows;
    
    // Распределяем строки между потоками
    int rows_per_thread = total_rows / num_threads;
    int remaining_rows = total_rows % num_threads;
    
    int current_row = 0;
    for (int t = 0; t < num_threads; ++t) {
        int thread_row_count = rows_per_thread + (t < remaining_rows ? 1 : 0);
        if (thread_row_count > 0) {
            thread_rows.push_back(thread_row_count);
            int start_row = current_row;
            int end_row = current_row + thread_row_count;
            
            // Запускаем вычисление в отдельном потоке
            futures.push_back(async(launch::async, [&A, &B, start_row, end_row]() -> MatrixXd {
                MatrixXd A_part = A.middleRows(start_row, end_row - start_row);
                MatrixXd result = A_part * B;
                return result;
            }));
            
            current_row = end_row;
        }
    }
    
    // Собираем результаты
    vector<MatrixXd> results;
    for (auto& f : futures) {
        results.push_back(f.get());
    }
    
    // Объединяем результаты
    if (results.empty()) {
        return A * B;
    }
    
    int total_result_rows = 0;
    for (const auto& r : results) {
        total_result_rows += static_cast<int>(r.rows());
    }
    
    MatrixXd combined(total_result_rows, results[0].cols());
    int row_offset = 0;
    for (const auto& r : results) {
        combined.middleRows(row_offset, static_cast<int>(r.rows())) = r;
        row_offset += static_cast<int>(r.rows());
    }
    
    return combined;
}

// Матричное умножение с гибридным подходом (CPU + GPU)
MatrixXd CudaAccelerator::matrixMultiplyGPU(const MatrixXd& A, const MatrixXd& B) {
    // Если CUDA недоступна, используем многопоточный CPU
    if (!cuda_available) {
        return hybridCompute(A, B, 0.5);
    }

#ifdef CUDA_RUNTIME_AVAILABLE
    // Размерности: A(m,k), B(k,n) => C(m,n)
    const int m = static_cast<int>(A.rows());
    const int k = static_cast<int>(A.cols());
    const int n = static_cast<int>(B.cols());

    if (k != B.rows() || m == 0 || n == 0) {
        // Неверные размеры - fallback на CPU
        return A * B;
    }

    MatrixXd C(m, n);

    // Выделяем память на устройстве
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;

    size_t bytes_A = static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(double);
    size_t bytes_B = static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(double);
    size_t bytes_C = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(double);

    cudaError_t errA = cudaMalloc(&d_A, bytes_A);
    cudaError_t errB = cudaMalloc(&d_B, bytes_B);
    cudaError_t errC = cudaMalloc(&d_C, bytes_C);

    if (errA != cudaSuccess || errB != cudaSuccess || errC != cudaSuccess) {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        // Fallback на CPU
        return A * B;
    }

    // Копируем данные из Eigen (column-major) на устройство
    cudaMemcpy(d_A, A.data(), bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes_B, cudaMemcpyHostToDevice);

    // Выполняем C = A * B с помощью cuBLAS (column-major)
    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle);
    if (!handle) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return A * B;
    }

    const double alpha = 1.0;
    const double beta  = 0.0;

    // В cuBLAS матрицы в column-major:
    // A: m x k, lda = m
    // B: k x n, ldb = k
    // C: m x n, ldc = m
    cublasStatus_t status = cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        d_A, m,
        d_B, k,
        &beta,
        d_C, m
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return A * B;
    }

    // Копируем результат обратно
    cudaMemcpy(C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
#else
    // Если CUDA runtime недоступен, используем гибридный CPU
    return hybridCompute(A, B, 0.5);
#endif
}

// Остальные функции - заглушки с fallback на CPU
MatrixXd CudaAccelerator::matrixAddGPU(const MatrixXd& A, const MatrixXd& B) {
    if (cuda_available) {
        // В будущем: CUDA реализация
    }
    return A + B;
}

MatrixXd CudaAccelerator::elementWiseMultiplyGPU(const MatrixXd& A, const MatrixXd& B) {
    if (cuda_available) {
        // В будущем: CUDA реализация
    }
    return A.cwiseProduct(B);
}

MatrixXd CudaAccelerator::reluGPU(const MatrixXd& X) {
    if (cuda_available) {
        // В будущем: CUDA реализация
    }
    return X.cwiseMax(0.0);
}

MatrixXd CudaAccelerator::reluDerivativeGPU(const MatrixXd& X) {
    if (cuda_available) {
        // В будущем: CUDA реализация
    }
    return (X.array() > 0.0).cast<double>().matrix();
}

MatrixXd CudaAccelerator::softmaxGPU(const MatrixXd& X) {
    if (cuda_available) {
        // В будущем: CUDA реализация
    }
    // CPU реализация softmax
    MatrixXd exp_X = X.array().exp();
    MatrixXd sum_exp = exp_X.rowwise().sum();
    for (int i = 0; i < exp_X.rows(); ++i) {
        exp_X.row(i) = exp_X.row(i).array() / sum_exp(i, 0);
    }
    return exp_X;
}

MatrixXd CudaAccelerator::transposeGPU(const MatrixXd& X) {
    if (cuda_available) {
        // В будущем: CUDA реализация
    }
    return X.transpose();
}

MatrixXd CudaAccelerator::colwiseSumGPU(const MatrixXd& X) {
    if (cuda_available) {
        // В будущем: CUDA реализация
    }
    return X.colwise().sum();
}

MatrixXd CudaAccelerator::colwiseMeanGPU(const MatrixXd& X) {
    if (cuda_available) {
        // В будущем: CUDA реализация
    }
    return X.colwise().mean();
}

void CudaAccelerator::cleanup() {
    // Освобождение ресурсов CUDA
#ifdef CUDA_RUNTIME_AVAILABLE
    if (cublas_handle) {
        cublasDestroy(static_cast<cublasHandle_t>(cublas_handle));
        cublas_handle = nullptr;
    }
    // Не вызываем cudaDeviceReset здесь, оставляем управление устройством приложению/драйверу
#endif
    initialized = false;
}

