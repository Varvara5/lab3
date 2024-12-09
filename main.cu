#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// Утилита для проверки ошибок CUDA
#define cudaCheckError(call) {                                          \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n"; \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

//xuy

// Ядро для деления строки матрицы на ведущий элемент
__global__ void normalize_row(float* matrix, int n, int row, float pivot) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col < n) {
        matrix[row * n + col] /= pivot;
    }
}

// Ядро для вычитания текущей строки из всех строк ниже
__global__ void eliminate_rows(float* matrix, int n, int row) {
    int target_row = blockIdx.x;
    int col = threadIdx.x;
    if (target_row > row && target_row < n && col < n) {
        float multiplier = matrix[target_row * n + row];
        matrix[target_row * n + col] -= multiplier * matrix[row * n + col];
    }
}

float determinant(float* h_matrix, int n) {
    // Выделение памяти на GPU
    float* d_matrix;
    size_t matrix_size = n * n * sizeof(float);
    cudaCheckError(cudaMalloc(&d_matrix, matrix_size));
    cudaCheckError(cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice));

    float det = 1.0f;

    for (int i = 0; i < n; i++) {
        // Считывание ведущего элемента
        float pivot;
        cudaCheckError(cudaMemcpy(&pivot, &d_matrix[i * n + i], sizeof(float), cudaMemcpyDeviceToHost));

        // Если ведущий элемент равен нулю, определитель равен нулю
        if (pivot == 0.0f) {
            det = 0.0f;
            break;
        }

        det *= pivot;  // Умножение на ведущий элемент

        // Нормализация строки
        int threads_per_block = 32;
        int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
        normalize_row<<<blocks_per_grid, threads_per_block>>>(d_matrix, n, i, pivot);
        cudaCheckError(cudaDeviceSynchronize());

        // Вычитание текущей строки из строк ниже
        eliminate_rows<<<n, n>>>(d_matrix, n, i);
        cudaCheckError(cudaDeviceSynchronize());
    }

    // Освобождение памяти
    cudaFree(d_matrix);

    return det;
}

int main() {
    // Ввод размера матрицы
    int n;
    std::cout << "Enter the size of the matrix (n x n): ";
    std::cin >> n;

    // Создание матрицы
    std::vector<float> h_matrix(n * n);
    std::cout << "Enter the elements of the matrix row by row:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cin >> h_matrix[i * n + j];
        }
    }

    // Вычисление определителя
    float det = determinant(h_matrix.data(), n);

    // Результат
    std::cout << "Determinant: " << det << "\n";

    return 0;
}

