#include "BlasWrapper.h"
#include "Profiler.h"
#include <iostream>

// Определение макросов для проверки доступности BLAS библиотек
#ifdef USE_OPENBLAS
    #include <cblas.h>
    #define OPENBLAS_AVAILABLE
#endif

#ifdef USE_MKL
    #include <mkl.h>
    #define MKL_AVAILABLE
#endif

BlasWrapper::BlasType BlasWrapper::blas_type_ = NONE;
bool BlasWrapper::initialized_ = false;

bool BlasWrapper::initialize() {
    if (initialized_) {
        return blas_type_ != NONE;
    }
    
    blas_type_ = detectBlas();
    initialized_ = true;
    
    if (blas_type_ != NONE) {
        string blas_name = (blas_type_ == OPENBLAS) ? "OpenBLAS" : "Intel MKL";
        cout << "[BlasWrapper] " << blas_name << " detected and enabled" << endl;
    } else {
        cout << "[BlasWrapper] No BLAS library detected, using Eigen fallback" << endl;
    }
    
    return blas_type_ != NONE;
}

BlasWrapper::BlasType BlasWrapper::detectBlas() {
    // Проверяем доступность через условную компиляцию
    #ifdef OPENBLAS_AVAILABLE
        return OPENBLAS;
    #elif defined(MKL_AVAILABLE)
        return MKL;
    #else
        // Пытаемся определить во время выполнения через тестовое вычисление
        // Для простоты используем условную компиляцию
        return NONE;
    #endif
}

MatrixXd BlasWrapper::gemm(const MatrixXd& A, const MatrixXd& B, double alpha, double beta) {
    PROFILE_SCOPE("BlasWrapper::gemm");
    
    // Для маленьких матриц используем Eigen (меньше overhead)
    if (A.rows() < BLAS_THRESHOLD || B.cols() < BLAS_THRESHOLD) {
        if (beta == 0.0) {
            return alpha * (A * B);
        } else {
            MatrixXd C = MatrixXd::Zero(A.rows(), B.cols());
            return alpha * (A * B) + beta * C;
        }
    }
    
    // Используем BLAS если доступен
    if (blas_type_ == OPENBLAS) {
        return gemmOpenBLAS(A, B, alpha, beta);
    } else if (blas_type_ == MKL) {
        return gemmMKL(A, B, alpha, beta);
    }
    
    // Fallback на Eigen
    if (beta == 0.0) {
        return alpha * (A * B);
    } else {
        MatrixXd C = MatrixXd::Zero(A.rows(), B.cols());
        return alpha * (A * B) + beta * C;
    }
}

MatrixXd BlasWrapper::gemmTrans(const MatrixXd& A, const MatrixXd& B,
                                char transA, char transB, double alpha) {
    PROFILE_SCOPE("BlasWrapper::gemmTrans");
    
    // Для маленьких матриц используем Eigen
    if (A.rows() < BLAS_THRESHOLD || B.cols() < BLAS_THRESHOLD) {
        MatrixXd A_work = (transA == 'T') ? A.transpose() : A;
        MatrixXd B_work = (transB == 'T') ? B.transpose() : B;
        return alpha * (A_work * B_work);
    }
    
    // Используем BLAS если доступен
    if (blas_type_ == OPENBLAS) {
        return gemmTransOpenBLAS(A, B, transA, transB, alpha);
    } else if (blas_type_ == MKL) {
        return gemmTransMKL(A, B, transA, transB, alpha);
    }
    
    // Fallback на Eigen
    MatrixXd A_work = (transA == 'T') ? A.transpose() : A;
    MatrixXd B_work = (transB == 'T') ? B.transpose() : B;
    return alpha * (A_work * B_work);
}

#ifdef OPENBLAS_AVAILABLE
MatrixXd BlasWrapper::gemmOpenBLAS(const MatrixXd& A, const MatrixXd& B, 
                                   double alpha, double beta) {
    int m = static_cast<int>(A.rows());
    int n = static_cast<int>(B.cols());
    int k = static_cast<int>(A.cols());
    
    MatrixXd C = MatrixXd::Zero(m, n);
    
    // CBLAS: cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha,
                const_cast<double*>(A.data()), k,
                const_cast<double*>(B.data()), n,
                beta, C.data(), n);
    
    return C;
}

MatrixXd BlasWrapper::gemmTransOpenBLAS(const MatrixXd& A, const MatrixXd& B,
                                       char transA, char transB, double alpha) {
    int m = static_cast<int>(A.rows());
    int n = static_cast<int>(B.cols());
    int k = static_cast<int>(A.cols());
    
    CBLAS_TRANSPOSE transA_cblas = (transA == 'T') ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transB_cblas = (transB == 'T') ? CblasTrans : CblasNoTrans;
    
    // Корректируем размеры для транспонирования
    if (transA == 'T') {
        m = static_cast<int>(A.cols());
        k = static_cast<int>(A.rows());
    }
    if (transB == 'T') {
        n = static_cast<int>(B.rows());
        k = static_cast<int>(B.cols());
    }
    
    MatrixXd C = MatrixXd::Zero(m, n);
    
    int lda = (transA == 'T') ? static_cast<int>(A.rows()) : static_cast<int>(A.cols());
    int ldb = (transB == 'T') ? static_cast<int>(B.rows()) : static_cast<int>(B.cols());
    
    cblas_dgemm(CblasRowMajor, transA_cblas, transB_cblas,
                m, n, k, alpha,
                const_cast<double*>(A.data()), lda,
                const_cast<double*>(B.data()), ldb,
                0.0, C.data(), n);
    
    return C;
}
#else
MatrixXd BlasWrapper::gemmOpenBLAS(const MatrixXd& A, const MatrixXd& B, 
                                   double alpha, double beta) {
    // Fallback на Eigen
    if (beta == 0.0) {
        return alpha * (A * B);
    } else {
        MatrixXd C = MatrixXd::Zero(A.rows(), B.cols());
        return alpha * (A * B) + beta * C;
    }
}

MatrixXd BlasWrapper::gemmTransOpenBLAS(const MatrixXd& A, const MatrixXd& B,
                                       char transA, char transB, double alpha) {
    MatrixXd A_work = (transA == 'T') ? A.transpose() : A;
    MatrixXd B_work = (transB == 'T') ? B.transpose() : B;
    return alpha * (A_work * B_work);
}
#endif

#ifdef MKL_AVAILABLE
MatrixXd BlasWrapper::gemmMKL(const MatrixXd& A, const MatrixXd& B,
                             double alpha, double beta) {
    int m = static_cast<int>(A.rows());
    int n = static_cast<int>(B.cols());
    int k = static_cast<int>(A.cols());
    
    MatrixXd C = MatrixXd::Zero(m, n);
    
    // MKL: cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha,
                const_cast<double*>(A.data()), k,
                const_cast<double*>(B.data()), n,
                beta, C.data(), n);
    
    return C;
}

MatrixXd BlasWrapper::gemmTransMKL(const MatrixXd& A, const MatrixXd& B,
                                  char transA, char transB, double alpha) {
    int m = static_cast<int>(A.rows());
    int n = static_cast<int>(B.cols());
    int k = static_cast<int>(A.cols());
    
    CBLAS_TRANSPOSE transA_cblas = (transA == 'T') ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transB_cblas = (transB == 'T') ? CblasTrans : CblasNoTrans;
    
    if (transA == 'T') {
        m = static_cast<int>(A.cols());
        k = static_cast<int>(A.rows());
    }
    if (transB == 'T') {
        n = static_cast<int>(B.rows());
        k = static_cast<int>(B.cols());
    }
    
    MatrixXd C = MatrixXd::Zero(m, n);
    
    int lda = (transA == 'T') ? static_cast<int>(A.rows()) : static_cast<int>(A.cols());
    int ldb = (transB == 'T') ? static_cast<int>(B.rows()) : static_cast<int>(B.cols());
    
    cblas_dgemm(CblasRowMajor, transA_cblas, transB_cblas,
                m, n, k, alpha,
                const_cast<double*>(A.data()), lda,
                const_cast<double*>(B.data()), ldb,
                0.0, C.data(), n);
    
    return C;
}
#else
MatrixXd BlasWrapper::gemmMKL(const MatrixXd& A, const MatrixXd& B,
                             double alpha, double beta) {
    // Fallback на Eigen
    if (beta == 0.0) {
        return alpha * (A * B);
    } else {
        MatrixXd C = MatrixXd::Zero(A.rows(), B.cols());
        return alpha * (A * B) + beta * C;
    }
}

MatrixXd BlasWrapper::gemmTransMKL(const MatrixXd& A, const MatrixXd& B,
                                  char transA, char transB, double alpha) {
    MatrixXd A_work = (transA == 'T') ? A.transpose() : A;
    MatrixXd B_work = (transB == 'T') ? B.transpose() : B;
    return alpha * (A_work * B_work);
}
#endif

