#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <complex>
#include <chrono>
#include <boost/math/special_functions/bessel.hpp>
#include <complex_bessel_bits/sph_besselFunctions.h> 

using namespace boost::math;
using namespace sp_bessel;
using namespace std;

//---------- CONSTANTES ----------

double Pi           = boost::math::constants::pi<double>();
double hbar         = 4.135667731/(2*Pi) ;//constante de Plank reducida en eV*fs
double cluz         = 299.792458;   //rapidez de la luz en nm/fs
double hbarcluz     = hbar*cluz;    //constante de Plank reducida por la rapidez de la luz en eV*nm
double hbaromega_p  = 13.14;    //frecuencia de plasma por la constante de Planck en eV
double hbargamma    = 0.197;    //dammping por la constante de Planck en eV
double alphafina    = 1/137.035999139;    //constante de estructura fina


//---------- FUNCIONES AUXILIARES Y MATERIALES ----------
//factor de Lorentz
double gammaLorentz(double beta){
    return 1. / sqrt(1. - pow(beta,2.));
}
//Función dieléctrica del modelo de Drude
complex<double> epsilon_Drude(double hbaromega){
    return 1. - pow(hbaromega_p,2.) / (hbaromega * (hbaromega + 1i*hbargamma));
}
//Índice de refracción del modelo de Drude
complex<double> N_Drude(double hbaromega){
    return complex<double>(sqrt(epsilon_Drude(hbaromega)));
}
//parámetro de tamaño
double x_tam(double hbaromega, double a){
    return hbaromega*a/hbarcluz;
}
//unidad imaginaria de un índice
complex<double> i_m(int m){
    if (m == 0 || m % 2 == 0) {
        return 1i; // 1i para m = 0 o m par
    } else {
        return 1; // 1 para m impar
    }
}


//---------- DEFINICIÓN DE LAS FUNCIONES DE RICCATI-BESSEL ----------
//Función psi_l
complex<double> RB_psi(int l, complex<double> z){
    return z*sph_besselJ(l,z);
}
//Función xi_l
complex<double> RB_xi(int l, complex<double> z){
    return z*1i*sph_hankelH1(l,z);
}
//Derivada de la función psi_l
complex<double> D_RB_psi(int l, complex<double> z){
    return z*(double(l)*sph_besselJ(l-1,z)/double(2*l+1)-double(l+1)*sph_besselJ(l+1,z)/double(2*l+1))+sph_besselJ(l,z);
}
//Derivada de la función xi_l
complex<double> D_RB_xi(int l, complex<double> z){
    return z*(double(l)*1i*sph_hankelH1(l-1,z)/double(2*l+1)-double(l+1)*1i*sph_hankelH1(l+1,z)/double(2*l+1))+1i*sph_hankelH1(l,z);
}


//---------- DEFINICIÓN DE LOS COEFICIENTES DE MIE ----------
//coeficiente de multipolos eléctricos
complex<double> tlE(int l, double hbaromega, double a){
    return -1.*(RB_psi(l,x_tam(hbaromega,a))*D_RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))-N_Drude(hbaromega)*RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))*D_RB_psi(l,x_tam(hbaromega,a)))/
    (RB_xi(l,x_tam(hbaromega,a))*D_RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))-N_Drude(hbaromega)*RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))*D_RB_xi(l,x_tam(hbaromega,a)));
}
//coeficiente de multipolos magnéticos
complex<double> tlM(int l, double hbaromega, double a){
    return -1.*(N_Drude(hbaromega)*RB_psi(l,x_tam(hbaromega,a))*D_RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))-RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))*D_RB_psi(l,x_tam(hbaromega,a)))/
    (N_Drude(hbaromega)*RB_xi(l,x_tam(hbaromega,a))*D_RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))-RB_psi(l,N_Drude(hbaromega)*x_tam(hbaromega,a))*D_RB_xi(l,x_tam(hbaromega,a)));
}


//---------- DEFINICIÓN DE LAS FUNCIONES ESPECTRALES AUXILIARES SIMPLIFICADAS----------
//función espectral auxiliar simplificada eléctrica
complex<double> FEASelectrica(int l, int m, double a, double b, double beta, double hbaromega, double r){
    if(hbaromega>=0){
        return i_m(m)*tlE(l,hbaromega,a)*boost::math::cyl_bessel_k(m,abs(hbaromega)*b/(hbarcluz*beta*gammaLorentz(beta)))*exp(1i*hbaromega*r/hbarcluz);
    } 
    else{
        return conj(i_m(m)*tlE(l,-1.*hbaromega,a)*boost::math::cyl_bessel_k(m,abs(-1.*hbaromega)*b/(hbarcluz*beta*gammaLorentz(beta)))*exp(-1.*1i*hbaromega*r/hbarcluz));
    }
}
//función espectral auxiliar simplificada magnética
complex<double> FEASmagnetica(int l, int m, double a, double b, double beta, double hbaromega, double r){
    if(hbaromega>=0){
        return i_m(m)*tlM(l,hbaromega,a)*boost::math::cyl_bessel_k(m,abs(hbaromega)*b/(hbarcluz*beta*gammaLorentz(beta)))*exp(1i*hbaromega*r/hbarcluz);
    } 
    else{
        return conj(i_m(m)*tlM(l,-1.*hbaromega,a)*boost::math::cyl_bessel_k(m,abs(-1.*hbaromega)*b/(hbarcluz*beta*gammaLorentz(beta)))*exp(-1.*1i*hbaromega*r/hbarcluz));
    }
}

int main() {
    int l, m;
    double a, b, beta, r;
    double hbaromega_min, hbaromega_max;
    const int N = 2048; // Número de puntos a graficar (puedes ajustar este valor aquí)

    // Obtener los parámetros de entrada
    cout << "Introduce el valor de l: ";
    cin >> l;
    cout << "Introduce el valor de m: ";
    cin >> m;
    cout << "Introduce el valor del radio a: ";
    cin >> a;
    cout << "Introduce el valor del parámetro de impacto b: ";
    cin >> b;
    cout << "Introduce el valor de la rapidez relativa del electrón beta: ";
    cin >> beta;
    cout << "Introduce el valor del radio r: ";
    cin >> r;
    cout << "Introduce el valor mínimo del intervalo de hbaromega: ";
    cin >> hbaromega_min;
    cout << "Introduce el valor máximo del intervalo de hbaromega: ";
    cin >> hbaromega_max;

    vector<double> hbaromega_values(N);
    double delta = (hbaromega_max - hbaromega_min) / (N - 1);
    for (int i = 0; i < N; ++i) {
        hbaromega_values[i] = hbaromega_min + i * delta;
    }

    vector<complex<double>> resultsE(N);
    vector<complex<double>> resultsM(N);

    // Iniciar el cronómetro
    auto start = chrono::high_resolution_clock::now();

    // Calcular los valores de la función eléctrica
    for (int i = 0; i < N; ++i) {
        double hbaromega = hbaromega_values[i];
        resultsE[i] = FEASelectrica(l, m, a, b, beta, hbaromega, r);
    }
    // Calcular los valores de la función magnética
    for (int i = 0; i < N; ++i) {
        double hbaromega = hbaromega_values[i];
        resultsM[i] = FEASmagnetica(l, m, a, b, beta, hbaromega, r);
    }

    // Detener el cronómetro
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    // Guardar los resultados en un archivo .txt
    ofstream file("FEAS_results.txt");
    for (int i = 0; i < N; ++i) {
        file << hbaromega_values[i] << " " << resultsE[i].real() << " " << resultsE[i].imag() << " " << resultsM[i].real() << " " << resultsM[i].imag() << "\n";
    }
    file.close();

    // Mostrar el tiempo de cómputo
    cout << "Cálculo completado y resultados guardados en FEAS_results.txt" << std::endl;
    cout << "Tiempo de cómputo: " << duration.count() << " microsegundos" << std::endl;

    return 0;
}