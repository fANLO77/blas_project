#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <complex>
#include <cblas.h> 
using namespace std;
using namespace std::chrono;
template <typename T>
void my_gemmtr_lower(int N, int K, T alpha, const T* A, const T* B, T beta, T* C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= i; ++j) { 
            T sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j]; 
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}
template <typename T>
bool interface_test(int N, int K, T alpha, const T* A, const T* B, T beta, T* C) {
    try {
        my_gemmtr_lower(N, K, alpha, A, B, beta, C);
        return true;
    } catch (...) {
        cerr << "Функция упала при выполнении" << endl;
        return false;
    }
}
void run_performance_tests() {
    const int N = 2048; 
    const int K = 2048;
    vector<float> A(N * K, 1.0f);
    vector<float> B(K * N, 1.0f);
    vector<float> C_my(N * N, 0.0f);
    vector<float> C_ob(N * N, 0.0f);
    float alpha = 1.0f, beta = 0.0f;
    vector<int> threads = {1, 2, 4, 8, 16};
    cout << "Запуск тестов производительности" << endl;
    for (int t : threads) {
        openblas_set_num_threads(t);
        double total_rel_perf = 1.0; 
        cout << "\nПотоков OpenBLAS: " << t << endl;
        for (int run = 0; run < 10; ++run) {
            auto start_ob = high_resolution_clock::now();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, K, alpha, A.data(), K, B.data(), N, beta, C_ob.data(), N);
            auto end_ob = high_resolution_clock::now();
            duration<double> time_ob = end_ob - start_ob;
            auto start_my = high_resolution_clock::now();
            interface_test(N, K, alpha, A.data(), B.data(), beta, C_my.data());
            auto end_my = high_resolution_clock::now();
            duration<double> time_my = end_my - start_my;
            double rel_perf = (time_ob.count() / time_my.count()) * 100.0;
            total_rel_perf *= rel_perf;
            cout << "  Запуск " << run + 1 << ": My =" << time_my.count() << "s, OB =" << time_ob.count() << "s, Perf =" << rel_perf << "%" << endl;
        }
        double geom_mean = pow(total_rel_perf, 1.0 / 10.0);
        cout << "Среднее геометрическое производительности: " << geom_mean << "% от OpenBLAS" << endl;
    }
}
int main() {
    run_performance_tests();
    return 0;
}