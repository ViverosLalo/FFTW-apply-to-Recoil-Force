
//---------------------------------------------------------------------------------------------------------
//
//          Este código calcula la componentes de la fuerza de retroceso considerando sólo la interacción entre el dipolo eléctrico y el dipolo magnético.
//
//
//          Para compilar:
//              g++ -o Recoil_force_NTSPA Recoil_force_NTSPA.cpp IN31.cpp -lfftw3 -lm -lcomplex_bessel -fopenmp
//          Para ejecutar:
//              ./Recoil_force_NTSPA
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

using namespace boost::math;
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


//---------- DEFINICIÓN DE LOS COEFICIENTES DE MIE EN LA APROXIMACIÓN NO TRIVIAL DE PARTÍCULA PEQUEÑA----------

//  Coeficiente de multipolos eléctricos
complex<double> tlEapp(int l, double hbaromega, double a){
    if (l==1){
        return (2.0/3.0)*((epsilon_Drude(hbaromega) - 1.0)/(epsilon_Drude(hbaromega) + 2.0))*pow(x_tam(hbaromega,a),3.0) + 
               (2.0/5.0)*((epsilon_Drude(hbaromega) - 2.0)*(epsilon_Drude(hbaromega) - 1.0)/pow((epsilon_Drude(hbaromega) + 2.0),2.0))*pow(x_tam(hbaromega,a),5.0);
    } else if (l==2){
        return (1.0/15.0)*((epsilon_Drude(hbaromega) - 1.0)/(2.0*epsilon_Drude(hbaromega) + 3.0))*pow(x_tam(hbaromega,a),5.0);
    } else {
        return 0.0;
    }
}

//  Coeficiente de multipolos magnéticos
complex<double> tlMapp(int l, double hbaromega, double a){
    if (l==1){
        return (epsilon_Drude(hbaromega) - 1.0)*pow(x_tam(hbaromega,a),5.0)/45.0;
    } else {
        return 0.0;
    }
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


int main() {

    int l, m;                   //  Orden de la función espectral auxiliar simplificada
    double a, b, beta;          //  Parámetros variables del problema
    const double r = cluz/2;    //  Radio de integración de la superficie: se escoge c/2 para que el pulso generado se centre en t=0.5 fs

    int n;                          //  Potencia de 2 para generar el número de datos
    double T;                       //  El dominio temporal será [-T,T]
    const double threshold = 1e-12; //  Umbral para reemplazar valores pequeños por cero
    double t_min, t_max;               //  Intervalo de tiempo para acotar el número de datos en Mathematica y pueda procesarlos

    // Obtener los parámetros numéricos para la FFTW
    cout << "Introduce la potencia de 2 para el número de datos: ";
    cin >> n;
    cout << "Introduce el fin del intervalo T: ";
    cin >> T;

    // Obtener los parámetros físicos del sistema
    cout << "Introduce el valor del radio a: ";
    cin >> a;
    cout << "Introduce el valor del parámetro de impacto b: ";
    cin >> b;
    cout << "Introduce el valor de la rapidez relativa del electrón beta: ";
    cin >> beta;
    cout << "Introduce el valor mínimo del intervalo de tiempo [fs]: ";
    cin >> t_min;
    cout << "Introduce el valor máximo del intervalo de tiempo [fs]: ";
    cin >> t_max;

    auto start = chrono::high_resolution_clock::now();  //  Inicia el cronómetro de ejecución de cálculos

    int N = static_cast<int>(pow(2.,n));    //  Se genera el número de datos como una cantidad entera 

    //---------- Valores de las FEAS en la partición del dominio espectral ----------
    vector<complex<double>> OmegaE10_values_in_freq_domain(N);
    vector<complex<double>> OmegaE11_values_in_freq_domain(N);
    vector<complex<double>> OmegaM11_values_in_freq_domain(N);
    vector<complex<double>> OmegaE20_values_in_freq_domain(N);
    vector<complex<double>> OmegaE21_values_in_freq_domain(N);
    vector<complex<double>> OmegaE22_values_in_freq_domain(N);

    double delta_omega= Pi / T;
    #pragma omp parallel for
    for(int j = 0; j < N; j++) {
        double omega = (-(N/2)+1+ j)*delta_omega + numeric_limits<double>::epsilon(); 
        OmegaE10_values_in_freq_domain[j] = FEASelectrica(1, 0, a, b, beta, hbar*omega, r);
        OmegaE11_values_in_freq_domain[j] = FEASelectrica(1, 1, a, b, beta, hbar*omega, r);
        OmegaM11_values_in_freq_domain[j] = FEASmagnetica(1, 1, a, b, beta, hbar*omega, r);
        OmegaE20_values_in_freq_domain[j] = FEASelectrica(2, 0, a, b, beta, hbar*omega, r);
        OmegaE21_values_in_freq_domain[j] = FEASelectrica(2, 1, a, b, beta, hbar*omega, r);
        OmegaE22_values_in_freq_domain[j] = FEASelectrica(2, 2, a, b, beta, hbar*omega, r);
    }

    //---------- Realizar el corrimiento de los arreglos para la FFT ----------
    int shift = (N/2)-1;
    rotate(OmegaE10_values_in_freq_domain.begin(), OmegaE10_values_in_freq_domain.begin() + shift, OmegaE10_values_in_freq_domain.end());
    rotate(OmegaE11_values_in_freq_domain.begin(), OmegaE11_values_in_freq_domain.begin() + shift, OmegaE11_values_in_freq_domain.end());
    rotate(OmegaM11_values_in_freq_domain.begin(), OmegaM11_values_in_freq_domain.begin() + shift, OmegaM11_values_in_freq_domain.end());
    rotate(OmegaE20_values_in_freq_domain.begin(), OmegaE20_values_in_freq_domain.begin() + shift, OmegaE20_values_in_freq_domain.end());
    rotate(OmegaE21_values_in_freq_domain.begin(), OmegaE21_values_in_freq_domain.begin() + shift, OmegaE21_values_in_freq_domain.end());
    rotate(OmegaE22_values_in_freq_domain.begin(), OmegaE22_values_in_freq_domain.begin() + shift, OmegaE22_values_in_freq_domain.end());

    // Declarar los arreglos de salida en el dominio temporal
    vector<complex<double>> OmegaE10_values_in_time_domain(N);
    vector<complex<double>> OmegaE11_values_in_time_domain(N);
    vector<complex<double>> OmegaM11_values_in_time_domain(N);
    vector<complex<double>> OmegaE20_values_in_time_domain(N);
    vector<complex<double>> OmegaE21_values_in_time_domain(N);
    vector<complex<double>> OmegaE22_values_in_time_domain(N);

    // Declarar plan de FFTW para transformada inversa compleja a "compleja"
    fftw_plan pE1 = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(OmegaE10_values_in_freq_domain.data()), reinterpret_cast<fftw_complex*>(OmegaE10_values_in_time_domain.data()), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pE2 = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(OmegaE11_values_in_freq_domain.data()), reinterpret_cast<fftw_complex*>(OmegaE11_values_in_time_domain.data()), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pM1 = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(OmegaM11_values_in_freq_domain.data()), reinterpret_cast<fftw_complex*>(OmegaM11_values_in_time_domain.data()), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pE3 = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(OmegaE20_values_in_freq_domain.data()), reinterpret_cast<fftw_complex*>(OmegaE20_values_in_time_domain.data()), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pE4 = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(OmegaE21_values_in_freq_domain.data()), reinterpret_cast<fftw_complex*>(OmegaE21_values_in_time_domain.data()), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pE5 = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(OmegaE22_values_in_freq_domain.data()), reinterpret_cast<fftw_complex*>(OmegaE22_values_in_time_domain.data()), FFTW_FORWARD, FFTW_ESTIMATE);

    // Ejecutar plan de FFTW
    fftw_execute(pE1);
    fftw_execute(pE2);
    fftw_execute(pM1);
    fftw_execute(pE3);
    fftw_execute(pE4);
    fftw_execute(pE5);

    //Revertir el corrimiento (shift) del arreglo
    rotate(OmegaE10_values_in_time_domain.begin(), OmegaE10_values_in_time_domain.begin() + (N - shift), OmegaE10_values_in_time_domain.end());
    rotate(OmegaE11_values_in_time_domain.begin(), OmegaE11_values_in_time_domain.begin() + (N - shift), OmegaE11_values_in_time_domain.end());
    rotate(OmegaM11_values_in_time_domain.begin(), OmegaM11_values_in_time_domain.begin() + (N - shift), OmegaM11_values_in_time_domain.end());
    rotate(OmegaE20_values_in_time_domain.begin(), OmegaE20_values_in_time_domain.begin() + (N - shift), OmegaE20_values_in_time_domain.end());
    rotate(OmegaE21_values_in_time_domain.begin(), OmegaE21_values_in_time_domain.begin() + (N - shift), OmegaE21_values_in_time_domain.end());
    rotate(OmegaE22_values_in_time_domain.begin(), OmegaE22_values_in_time_domain.begin() + (N - shift), OmegaE22_values_in_time_domain.end());

    // Escalar los resultados
    #pragma omp parallel for
    for (int s = 0; s < N; s++) {
        OmegaE10_values_in_time_domain[s] *= delta_omega/(2*Pi); // Factor de escala dado por la partición en el dominio espectral y convención de transformada
        if (abs(OmegaE10_values_in_time_domain[s].real()) < threshold) OmegaE10_values_in_time_domain[s].real(0.0);
        OmegaE11_values_in_time_domain[s] *= delta_omega/(2*Pi);
        if (abs(OmegaE11_values_in_time_domain[s].real()) < threshold) OmegaE11_values_in_time_domain[s].real(0.0);
        OmegaM11_values_in_time_domain[s] *= delta_omega/(2*Pi);
        if (abs(OmegaM11_values_in_time_domain[s].real()) < threshold) OmegaM11_values_in_time_domain[s].real(0.0);
        OmegaE20_values_in_time_domain[s] *= delta_omega/(2*Pi);
        if (abs(OmegaE20_values_in_time_domain[s].real()) < threshold) OmegaE20_values_in_time_domain[s].real(0.0);
        OmegaE21_values_in_time_domain[s] *= delta_omega/(2*Pi); 
        if (abs(OmegaE21_values_in_time_domain[s].real()) < threshold) OmegaE21_values_in_time_domain[s].real(0.0);
        OmegaE22_values_in_time_domain[s] *= delta_omega/(2*Pi); 
        if (abs(OmegaE22_values_in_time_domain[s].real()) < threshold) OmegaE22_values_in_time_domain[s].real(0.0);
    }

    // Calcular las componentes de la fuerza de retroceso en aproximación dipolar
    vector<double> Fuerza_de_retroceso_DEDM_X(N);
    vector<double> Fuerza_de_retroceso_DEDM_Z(N);
    vector<double> Fuerza_de_retroceso_DEQE_X(N);
    vector<double> Fuerza_de_retroceso_DEQE_Z(N);
    vector<double> Fuerza_de_retroceso_X(N);
    vector<double> Fuerza_de_retroceso_Z(N);
    const double factor_rel = 6/(beta*beta*beta*gammaLorentz(beta)*gammaLorentz(beta));
    #pragma omp parallel for
    for (int j = 0; j<N ; j++){
        Fuerza_de_retroceso_DEDM_X[j] = factor_rel*OmegaE10_values_in_time_domain[j].real()*OmegaM11_values_in_time_domain[j].real()/gammaLorentz(beta);   //Componente x (transversal) dipolo eléctrico - dipolo magnético
        Fuerza_de_retroceso_DEQE_X[j] = factor_rel*(-2.0*(2.0 - beta*beta)*OmegaE21_values_in_time_domain[j].real()*OmegaE10_values_in_time_domain[j].real() + 
                                        2.0*OmegaE20_values_in_time_domain[j].real()*OmegaE11_values_in_time_domain[j].real() + 
                                        OmegaE22_values_in_time_domain[j].real()*OmegaE11_values_in_time_domain[j].real())/(beta*beta*gammaLorentz(beta));
        Fuerza_de_retroceso_DEDM_Z[j] = -factor_rel*(OmegaM11_values_in_time_domain[j].real())*(OmegaE11_values_in_time_domain[j].real());                     //Componente z (longitudinal) dipolo eléctrico - dipolo magnético
        Fuerza_de_retroceso_DEQE_Z[j] = -factor_rel*(8.0*(1.0 - beta*beta)*OmegaE20_values_in_time_domain[j].real()*OmegaE10_values_in_time_domain[j].real() + 
                                        (2.0 - beta*beta)*OmegaE21_values_in_time_domain[j].real()*OmegaE11_values_in_time_domain[j].real())/(beta*beta);
        Fuerza_de_retroceso_X[j] = Fuerza_de_retroceso_DEDM_X[j]+Fuerza_de_retroceso_DEQE_X[j];
        Fuerza_de_retroceso_Z[j] = Fuerza_de_retroceso_DEDM_Z[j]+Fuerza_de_retroceso_DEQE_Z[j];
    }

    // Calcular los valores del dominio temporal
    double delta_t = 2*T / N;
    vector<double> time_domain_values(N);
    #pragma omp parallel for
    for (int s = 0; s < N; s++) {
        time_domain_values[s] = (-(N/2) + 1 + s)*delta_t;
    }

    int s_min = static_cast<int>(-1. + N * (1. + t_min/T)/2);
    int s_max = static_cast<int>(-1. + N * (1. + t_max/T)/2);

     // Construir el nombre del archivo dinámicamente
    string nombre_archivo = "Resultados_Fuerza_de_retroceso_NTSPA_n" + to_string(n) + "_T" + to_string(static_cast<int>(T)) + "_a" + to_string(static_cast<int>(a)) + "_b" + to_string(static_cast<int>(b)) + "_beta" + to_string(static_cast<int>(100.001*beta)) + ".txt";

    // Guardar los resultados en un archivo .txt
    ofstream file(nombre_archivo);
    for (int i = s_min; i < s_max; i++) {
        file << time_domain_values[i] << " " << Fuerza_de_retroceso_DEDM_X[i] << " " << Fuerza_de_retroceso_DEDM_Z[i] << " " 
                                             << Fuerza_de_retroceso_DEQE_X[i] << " " << Fuerza_de_retroceso_DEQE_Z[i] << " " 
                                             << Fuerza_de_retroceso_X[i] << " " << Fuerza_de_retroceso_Z[i] << "\n";
    }
    file.close();

    auto stop = chrono::high_resolution_clock::now();   //  Para el cronómetro de ejecución de cálculos
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start)/pow(10.,6.);

    fftw_destroy_plan(pE1);
    fftw_destroy_plan(pE2);
    fftw_destroy_plan(pM1);
    fftw_destroy_plan(pE3);
    fftw_destroy_plan(pE4);
    fftw_destroy_plan(pE5);

    cout << "Resultados guardados en " << nombre_archivo << endl;
    cout << "Tiempo de cómputo: " << duration.count() << " segundos" << endl;

    return 0;
}