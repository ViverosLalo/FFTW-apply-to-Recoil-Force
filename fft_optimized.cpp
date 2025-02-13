#include <iostream> // Para entrada y salida estándar
#include <fstream> // Para manejar archivos
#include <cmath> // Para funciones matemáticas
#include <vector> // Para utilizar el contenedor vector
#include <chrono> // Para medir tiempo
#include <complex> // Para manejar números complejos
#include <algorithm> // Para la función std::rotate
#include <fftw3.h> // Biblioteca FFTW
#include <sstream> // Para usar std::ostringstream

// Función Gaussiana
double gaussian(double x, double sigma_sq_2) {
    return exp(-x * x / sigma_sq_2);
}

int main() {
    int N;
    double b, sigma;
    const double threshold = 1e-16; // Umbral para reemplazar valores pequeños por cero

    std::cout << "Introduce el número de datos N: ";
    std::cin >> N;
    std::cout << "Introduce el fin del intervalo b: ";
    std::cin >> b;
    std::cout << "Introduce el parámetro sigma de la gaussiana: ";
    std::cin >> sigma;

    // Validación de entrada
    if (N <= 0 || b <= 0 || sigma <= 0) {
        std::cerr << "Error: Valores inválidos. Deben ser positivos." << std::endl;
        return 1;
    }

    double delta = b / (N - 1);
    double sigma_sq_2 = 2.0 * sigma * sigma;

    // Usar memoria alineada con FFTW
    double *in_real = fftw_alloc_real(N);
    fftw_complex *out = fftw_alloc_complex(N / 2 + 1);

    // Inicializar entrada
    for (int i = 0; i < N; ++i) {
        double x = i * delta;
        in_real[i] = gaussian(x, sigma_sq_2);
    }

    // Crear plan óptimo
    fftw_plan p = fftw_plan_dft_r2c_1d(N, in_real, out, FFTW_MEASURE);

    // Ejecutar FFT
    auto start = std::chrono::high_resolution_clock::now();
    fftw_execute(p);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // Calcular el dominio de frecuencias
    double freq_step = 1.0 / (N * delta);
    std::vector<double> frequencies(N);
    for (int i = 0; i < N; ++i) {
        frequencies[i] = (i - N / 2) * freq_step;
    }

    // Escalar los resultados
    for (int i = 0; i < N / 2 + 1; ++i) {
        out[i][0] /= N; // Parte real
        out[i][1] /= N; // Parte imaginaria
    }

    // Guardar resultados en un archivo
    std::ofstream file("fft_results.txt");
    std::ostringstream buffer;
    for (int i = 0; i < N ; ++i) {
        double real_part = (std::abs(out[i][0]) < threshold) ? 0 : out[i][0];
        double imag_part = (std::abs(out[i][1]) < threshold) ? 0 : out[i][1];
        buffer << frequencies[i] << " " << real_part << " " << imag_part << "\n";
    }
    file << buffer.str();
    file.close();

    // Limpiar recursos
    fftw_destroy_plan(p);
    fftw_free(in_real);
    fftw_free(out);
    fftw_cleanup();

    std::cout << "FFT completada y resultados guardados en fft_results.txt" << std::endl;
    std::cout << "Tiempo de cómputo: " << duration.count() << " microsegundos" << std::endl;

    return 0;
}