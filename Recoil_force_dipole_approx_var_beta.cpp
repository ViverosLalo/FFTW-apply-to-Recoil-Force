
//---------------------------------------------------------------------------------------------------------
//
//          Este código calcula la componentes de la fuerza de retroceso 
//
//
//
//          Para compilar:
//              g++ -o Aproximacion_dipolar Aproximacion_dipolar.cpp -lfftw3 -lm -lcomplex_bessel -fopenmp
//          Para ejecutar:
//              ./Aproximacion_dipolar
//
//---------------------------------------------------------------------------------------------------------

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
#include <omp.h>

//using namespace boost::math;
using namespace sp_bessel;
using namespace std;

// ---------- CONSTANTES ----------

double Pi           = boost::math::constants::pi<double>();
double hbar         = 4.135667731/(2*Pi) ;//constante de Plank reducida en eV*fs
double cluz         = 299.792458;   //rapidez de la luz en nm/fs
double hbarcluz     = hbar*cluz;    //constante de Plank reducida por la rapidez de la luz en eV*nm
double hbaromega_p  = 13.14;    //frecuencia de plasma por la constante de Planck en eV
double hbargamma    = 0.197;    //dammping por la constante de Planck en eV
double alphafina    = 1/137.035999139;    //constante de estructura fina

//---------- FUNCIONES AUXILIARES Y MATERIALES ----------

//  Factor de Lorentz
double gammaLorentz(double beta){
    return 1. / sqrt(1. - pow(beta,2.));
}
//  Función dieléctrica del modelo de Drude
complex<double> epsilon_Drude(double hbaromega){
    return 1. - pow(hbaromega_p,2.) / (hbaromega * (hbaromega + 1i*hbargamma));
}
//  Índice de refracción del modelo de Drude
complex<double> N_Drude(double hbaromega){
    return complex<double>(sqrt(epsilon_Drude(hbaromega)));
}
//  Parámetro de tamaño
double x_tam(double hbaromega, double a){
    return hbaromega*a/hbarcluz;
}
//  Unidad imaginaria de un índice
complex<double> i_m(int m){
    if (m == 0 || m % 2 == 0) {
        return 1i; // 1i para m = 0 o m par
    } else {
        return 1; // 1 para m impar
    }
}


//---------- DEFINICIÓN DE LAS FUNCIONES DE RICCATI-BESSEL ----------

//  Función psi_l
complex<double> RB_psi(int l, complex<double> z){
    return z*sph_besselJ(l,z);
}
//  Función xi_l
complex<double> RB_xi(int l, complex<double> z){
    return z*1i*sph_hankelH1(l,z);
}
//  Derivada de la función psi_l
complex<double> D_RB_psi(int l, complex<double> z){
    return z*(double(l)*sph_besselJ(l-1,z)/double(2*l+1)-double(l+1)*sph_besselJ(l+1,z)/double(2*l+1))+sph_besselJ(l,z);
}
//  Derivada de la función xi_l
complex<double> D_RB_xi(int l, complex<double> z){
    return z*(double(l)*1i*sph_hankelH1(l-1,z)/double(2*l+1)-double(l+1)*1i*sph_hankelH1(l+1,z)/double(2*l+1))+1i*sph_hankelH1(l,z);
}


//---------- DEFINICIÓN DE LOS COEFICIENTES DE MIE ----------

//  Coeficiente de multipolos eléctricos
complex<double> tlE(int l, double hbaromega, double a){
    return -1.*(RB_psi(l,x_tam(hbaromega,a))*D_RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))-N_Drude(hbaromega)*RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))*D_RB_psi(l,x_tam(hbaromega,a)))/
    (RB_xi(l,x_tam(hbaromega,a))*D_RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))-N_Drude(hbaromega)*RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))*D_RB_xi(l,x_tam(hbaromega,a)));
}
//  Coeficiente de multipolos magnéticos
complex<double> tlM(int l, double hbaromega, double a){
    return -1.*(N_Drude(hbaromega)*RB_psi(l,x_tam(hbaromega,a))*D_RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))-RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))*D_RB_psi(l,x_tam(hbaromega,a)))/
    (N_Drude(hbaromega)*RB_xi(l,x_tam(hbaromega,a))*D_RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))-RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))*D_RB_xi(l,x_tam(hbaromega,a)));
}


//---------- DEFINICIÓN DE LAS FUNCIONES ESPECTRALES AUXILIARES SIMPLIFICADAS----------

//  Función espectral auxiliar simplificada eléctrica
complex<double> FEASelectrica(int l, int m, double a, double b, double beta, double hbaromega, double r){
    if(hbaromega>=0){
        return i_m(m)*tlE(l,hbaromega,a)*boost::math::cyl_bessel_k(m,abs(hbaromega)*b/(hbarcluz*beta*gammaLorentz(beta)))*exp(1i*hbaromega*r/hbarcluz);
    } 
    else{
        return conj(i_m(m)*tlE(l,-1.*hbaromega,a)*boost::math::cyl_bessel_k(m,abs(-1.*hbaromega)*b/(hbarcluz*beta*gammaLorentz(beta)))*exp(-1.*1i*hbaromega*r/hbarcluz));
    }
}
//  Función espectral auxiliar simplificada magnética
complex<double> FEASmagnetica(int l, int m, double a, double b, double beta, double hbaromega, double r){
    if(hbaromega>=0){
        return i_m(m)*tlM(l,hbaromega,a)*boost::math::cyl_bessel_k(m,abs(hbaromega)*b/(hbarcluz*beta*gammaLorentz(beta)))*exp(1i*hbaromega*r/hbarcluz);
    } 
    else{
        return conj(i_m(m)*tlM(l,-1.*hbaromega,a)*boost::math::cyl_bessel_k(m,abs(-1.*hbaromega)*b/(hbarcluz*beta*gammaLorentz(beta)))*exp(-1.*1i*hbaromega*r/hbarcluz));
    }
}

// Función para calcular la transformada de Fourier
//void calcular_transformada(std::vector<std::complex<double>>& datos, std::vector<std::complex<double>>& transformada, int N) {
//    fftw_plan p = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(datos.data()), reinterpret_cast<fftw_complex*>(transformada.data()), FFTW_FORWARD, FFTW_ESTIMATE);
//    fftw_execute(p);
//    fftw_destroy_plan
//}


int main() {

    int l, m;                   //  Orden de la función espectral auxiliar simplificada
    double a, b, beta;          //  Parámetros variables del problema
    const double r = cluz/2;    //  Radio de integración de la superficie: se escoge c/2 para que el pulso generado se centre en t=0.5 fs

    int n;                          //  Potencia de 2 para generar el número de datos
    const double T = 50;            //  El dominio temporal será [-T,T]. Para los cuálculos realizados, T=50 fs abarca todas las ventanas de los pulsos generados
    double beta_max, beta_min;      //  Límites para calcular la variación de beta
    const double threshold = 1e-12; //  Umbral para reemplazar valores pequeños por cero
    const double num_betas = 20;    //  Se considerará una partición de 20 valores de beta
    double t_min, t_max;               //  Intervalo de tiempo para acotar el número de datos en Mathematica y pueda procesarlos

    // Obtener los parámetros numéricos para la FFTW
    cout << "Introduce la potencia de 2 para el número de datos: ";
    cin >> n;
    //cout << "Introduce el fin del intervalo T: ";
    //cin >> T;

    // Obtener los parámetros físicos del sistema
    cout << "Introduce el valor del radio a: ";
    cin >> a;
    cout << "Introduce el valor del parámetro de impacto b: ";
    cin >> b;
    cout << "Introduce el valor mínimo de beta: ";
    cin >> beta_min;
    cout << "Introduce el valor máximo de beta: ";
    cin >> beta_max;
    cout << "Introduce el valor mínimo del intervalo de tiempo: ";
    cin >> t_min;
    cout << "Introduce el valor máximo del intervalo de tiempo: ";
    cin >> t_max;

    auto start = chrono::high_resolution_clock::now();  //  Inicia el cronómetro de ejecución de cálculos

    // Crear la partición de betas
    vector<double> betas(num_betas);
    double delta_beta = (beta_max - beta_min) / num_betas;
    for (int i = 0; i <= num_betas; ++i){
        betas[i] = beta_min + i*delta_beta;
    }

    int N = static_cast<int>(pow(2.,n));    //  Se genera el número de datos como una cantidad entera 

    //---------- Valores de las FEAS en la partición del dominio espectral ----------

    // Matrices para guardar los resultados de las componentes
    vector<vector<double>> Fuerza_de_retroceso_X_matriz(num_betas, vector<double>(N));
    vector<vector<double>> Fuerza_de_retroceso_Z_matriz(num_betas, vector<double>(N));

    double delta_omega= Pi / T;
    // CUIDADO: para los for internos usar (de preferencia) la variable j. La variable k está prohibida de usar porque es la del ciclo "for" externo 
    //#pragma omp parallel for
    for (int k = 0; k <= num_betas; k++){
        beta = betas[k];
        vector<complex<double>> OmegaE10_values_in_freq_domain(N);
        vector<complex<double>> OmegaE11_values_in_freq_domain(N);
        vector<complex<double>> OmegaM11_values_in_freq_domain(N);
        #pragma omp parallel for
        for(int j = 0; j < N; j++) {
            double omega = (-(N/2)+1+ j)*delta_omega + numeric_limits<double>::epsilon();
            OmegaE10_values_in_freq_domain[j] = FEASelectrica(1, 0, a, b, beta, hbar*omega, r);
            OmegaE11_values_in_freq_domain[j] = FEASelectrica(1, 1, a, b, beta, hbar*omega, r);
            OmegaM11_values_in_freq_domain[j] = FEASmagnetica(1, 1, a, b, beta, hbar*omega, r);
        }

        //---------- Realizar el corrimiento de los arreglos para la FFT ----------
        int shift = (N/2)-1;
        rotate(OmegaE10_values_in_freq_domain.begin(), OmegaE10_values_in_freq_domain.begin() + shift, OmegaE10_values_in_freq_domain.end());
        rotate(OmegaE11_values_in_freq_domain.begin(), OmegaE11_values_in_freq_domain.begin() + shift, OmegaE11_values_in_freq_domain.end());
        rotate(OmegaM11_values_in_freq_domain.begin(), OmegaM11_values_in_freq_domain.begin() + shift, OmegaM11_values_in_freq_domain.end());

        // Declarar los arreglos de salida en el dominio temporal
        vector<complex<double>> OmegaE10_values_in_time_domain(N);
        vector<complex<double>> OmegaE11_values_in_time_domain(N);
        vector<complex<double>> OmegaM11_values_in_time_domain(N);

        // Declarar plan de FFTW para transformada inversa compleja a "compleja"
        fftw_plan p1 = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(OmegaE10_values_in_freq_domain.data()), reinterpret_cast<fftw_complex*>(OmegaE10_values_in_time_domain.data()), FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan p2 = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(OmegaE11_values_in_freq_domain.data()), reinterpret_cast<fftw_complex*>(OmegaE11_values_in_time_domain.data()), FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan p3 = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(OmegaM11_values_in_freq_domain.data()), reinterpret_cast<fftw_complex*>(OmegaM11_values_in_time_domain.data()), FFTW_FORWARD, FFTW_ESTIMATE);

        // Ejecutar plan de FFTW
        fftw_execute(p1);
        fftw_execute(p2);
        fftw_execute(p3);

        //Revertir el corrimiento (shift) del arreglo
        rotate(OmegaE10_values_in_time_domain.begin(), OmegaE10_values_in_time_domain.begin() + (N - shift), OmegaE10_values_in_time_domain.end());
        rotate(OmegaE11_values_in_time_domain.begin(), OmegaE11_values_in_time_domain.begin() + (N - shift), OmegaE11_values_in_time_domain.end());
        rotate(OmegaM11_values_in_time_domain.begin(), OmegaM11_values_in_time_domain.begin() + (N - shift), OmegaM11_values_in_time_domain.end());

        // Escalar los resultados
        #pragma omp parallel for
        for (int s = 0; s < N; s++){
            OmegaE10_values_in_time_domain[s] *= delta_omega/(2*Pi); // Factor de escala dado por la partición en el dominio espectral y convención de transformada
            if (abs(OmegaE10_values_in_time_domain[s].real()) < threshold) OmegaE10_values_in_time_domain[s].real(0.0);
            OmegaE11_values_in_time_domain[s] *= delta_omega/(2*Pi);
            if (abs(OmegaE11_values_in_time_domain[s].real()) < threshold) OmegaE11_values_in_time_domain[s].real(0.0);
            OmegaM11_values_in_time_domain[s] *= delta_omega/(2*Pi);
            if (abs(OmegaM11_values_in_time_domain[s].real()) < threshold) OmegaM11_values_in_time_domain[s].real(0.0);
        }

        // Calcular las componentes de la fuerza de retroceso en aproximación dipolar
        const double factor_rel = 6/(beta*beta*beta*gammaLorentz(beta)*gammaLorentz(beta));
        #pragma omp parallel for
        for (int j = 0; j < N; j++){
            Fuerza_de_retroceso_X_matriz[k][j]  = factor_rel*(OmegaM11_values_in_time_domain[j].real())*(OmegaE10_values_in_time_domain[j].real())/gammaLorentz(beta);   //Componente x (transversal)
            Fuerza_de_retroceso_Z_matriz[k][j]  = -factor_rel*(OmegaM11_values_in_time_domain[j].real())*(OmegaE11_values_in_time_domain[j].real());                     //Componente z (longitudinal)
        }

        fftw_destroy_plan(p1);
        fftw_destroy_plan(p2);
        fftw_destroy_plan(p3);
    }

    // Calcular los valores del dominio temporal
    double delta_t = 2*T / N;
    vector<double> time_domain_values(N);
    #pragma omp parallel for
    for (int s = 0; s < N; s++) {
        time_domain_values[s] = (-(N/2) + 1 + s)*delta_t;
    }

    //  Valores de las iteraciones correspondientes a los valores de t_min y t_max en la partición temporal
    int s_min = static_cast<int>(-1. + N * (1. + t_min/T)/2);
    int s_max = static_cast<int>(-1. + N * (1. + t_max/T)/2);

    //  Guardar los resultados en un archivo .txt
    ofstream file1("recoil_force_x_dipolar_aprox_results.txt");
    for (int i = s_min; i < s_max; i++) {
        for (int k = 0; k <= num_betas; k++) {
            file1 << time_domain_values[i] << " " << betas[k] << " " << Fuerza_de_retroceso_X_matriz[k][i] << "\n";
        }
        file1 << ""; // Separar bloques de datos por sigma
    }
    file1.close();

    ofstream file2("recoil_force_z_dipolar_aprox_results.txt");
    for (int i = s_min; i < s_max; i++) {
        for (int k = 0; k <= num_betas; k++) {
            file2 << time_domain_values[i] << " " << betas[k] << " " << Fuerza_de_retroceso_Z_matriz[k][i] << "\n";
        }
        file2 << ""; // Separar bloques de datos por sigma
    }
    file2.close();

    auto stop = chrono::high_resolution_clock::now();   //  Para el cronómetro de ejecución de cálculos
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start)/pow(10.,6.);

    cout << "Transformada inversa de Fourier completada y resultados guardados en recoil_force_x_dipolar_aprox_results.txt y recoil_force_z_dipolar_aprox_results.txt" << endl;
    cout << "Tiempo de cómputo: " << duration.count() << " segundos" << endl;

    return 0;
}