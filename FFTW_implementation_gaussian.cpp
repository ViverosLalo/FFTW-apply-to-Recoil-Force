
//---------- Intro de qué va el código ----------

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <complex>
#include <chrono>
#include <fftw3.h>
#include <algorithm>
#include <boost/math/special_functions/bessel.hpp>
#include <complex_bessel_bits/sph_besselFunctions.h> 

using namespace std;

//---------- CONSTANTES ----------

double Pi = boost::math::constants::pi<double>();

//---------- Función de prueba: Gaussiana no normalizada ----------
complex<double> gaussian(double x, double sigma) {
    return complex<double>(exp(-x * x / (2 * sigma * sigma)), 0.0);
}

int main() {
    int N;
    double T, sigma;
    const double threshold = 1e-10; // Umbral para reemplazar valores pequeños por cero

    cout << "Introduce el número de datos N: ";
    cin >> N;
    cout << "Introduce el fin del intervalo T: ";
    cin >> T;
    cout << "Introduce el parámetro sigma de la gaussiana: ";
    cin >> sigma;

    //---------- Particiones de los dominios espectral ----------
    vector<complex<double>> fvalues_in_freq_domain(N);
    double delta_omega= Pi / T;
    for(int r = 0; r < N; r++) {
        double omega = (-(N/2)+1+ r)*delta_omega;
        fvalues_in_freq_domain[r] = gaussian(omega, sigma); 
    }

    //---------- Realizar el corrimiento del arreglo para la FFT ----------
    int shift = (N/2)-1;
    rotate(fvalues_in_freq_domain.begin(), fvalues_in_freq_domain.begin() + shift, fvalues_in_freq_domain.end());

    // Arreglo de salida en el dominio temporal
    vector<complex<double>> fvalues_in_time_domain(N);

    // Plan de FFTW para transformada inversa compleja a "compleja"
    fftw_plan p = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(fvalues_in_freq_domain.data()), reinterpret_cast<fftw_complex*>(fvalues_in_time_domain.data()), FFTW_FORWARD, FFTW_ESTIMATE);

    auto start = chrono::high_resolution_clock::now();
    fftw_execute(p);
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    //Revertir el corrimiento (shift) del arreglo
    rotate(fvalues_in_time_domain.begin(), fvalues_in_time_domain.begin() + (N- shift), fvalues_in_time_domain.end());

    // Escalar los resultados
    for (int s = 0; s < N; s++) {
        fvalues_in_time_domain[s] *= delta_omega/(2*Pi); // Factor de escala dado por la partición en el dominio espectral y convención de transformada
        if (abs(fvalues_in_time_domain[s].real()) < threshold) fvalues_in_time_domain[s].real(0.0);
        if (abs(fvalues_in_time_domain[s].imag()) < threshold) fvalues_in_time_domain[s].imag(0.0);
    }

    // Calcular los valores del dominio temporal
    double delta_t = 2*T / N;
    vector<double> time_domain_values(N);
    for (int s = 0; s < N; s++) {
        time_domain_values[s] = (-(N/2) + 1 + s)*delta_t;
    }

    // Guardar los resultados en un archivo .txt
    ofstream file("time_domain_results.txt");
    for (int i = 0; i < N; i++) {
        file << time_domain_values[i] << " " << fvalues_in_time_domain[i].real() << " " << fvalues_in_time_domain[i].imag() << "\n";
    }
    file.close();

    fftw_destroy_plan(p);

    cout << "Transformada inversa de Fourier completada y resultados guardados en time_domain_results.txt" << endl;
    cout << "Tiempo de cómputo: " << duration.count() << " microsegundos" << endl;

    return 0;
}