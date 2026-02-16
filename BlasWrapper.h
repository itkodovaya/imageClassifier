#ifndef BLAS_WRAPPER_H
#define BLAS_WRAPPER_H

#include <Eigen/Dense>
#include <string>

using namespace Eigen;
using namespace std;

// Обертка для BLAS библиотек (OpenBLAS, Intel MKL)
// Автоматически определяет доступную библиотеку и использует её для оптимизации матричных операций
class BlasWrapper {
public:
    // Типы доступных BLAS библиотек
    enum BlasType {
        NONE,      // BLAS не доступен, используется Eigen
        OPENBLAS,  // OpenBLAS
        MKL        // Intel MKL
    };
    
    // Инициализация и определение доступной BLAS библиотеки
    static bool initialize();
    
    // Получить тип доступной BLAS библиотеки
    static BlasType getBlasType() { return blas_type_; }
    
    // Проверить доступность BLAS
    static bool isAvailable() { return blas_type_ != NONE; }
    
    // Оптимизированное матричное умножение: C = alpha * A * B + beta * C
    // Если BLAS доступен, использует его, иначе fallback на Eigen
    static MatrixXd gemm(const MatrixXd& A, const MatrixXd& B, 
                        double alpha = 1.0, double beta = 0.0);
    
    // Оптимизированное матричное умножение с транспонированием
    // transA: 'N' = нет транспонирования, 'T' = транспонировать A
    // transB: 'N' = нет транспонирования, 'T' = транспонировать B
    static MatrixXd gemmTrans(const MatrixXd& A, const MatrixXd& B,
                              char transA = 'N', char transB = 'N',
                              double alpha = 1.0);
    
    // Порог размера матрицы для использования BLAS (меньше - используем Eigen)
    static constexpr int BLAS_THRESHOLD = 50;
    
private:
    static BlasType blas_type_;
    static bool initialized_;
    
    // Определение доступной BLAS библиотеки
    static BlasType detectBlas();
    
    // OpenBLAS реализация
    static MatrixXd gemmOpenBLAS(const MatrixXd& A, const MatrixXd& B, 
                                double alpha, double beta);
    static MatrixXd gemmTransOpenBLAS(const MatrixXd& A, const MatrixXd& B,
                                      char transA, char transB, double alpha);
    
    // Intel MKL реализация
    static MatrixXd gemmMKL(const MatrixXd& A, const MatrixXd& B,
                           double alpha, double beta);
    static MatrixXd gemmTransMKL(const MatrixXd& A, const MatrixXd& B,
                                 char transA, char transB, double alpha);
};

#endif // BLAS_WRAPPER_H

